from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator
from urllib.parse import urlsplit
from uuid import uuid4

import aiohttp

from bot.config import DEFAULT_STYLE_HINT, FORMULA_STYLE_HINT, config
from bot.services.science import normalize_formulas

logger = logging.getLogger(__name__)

# Upper bound for a downloaded generated image.
MAX_DOWNLOAD_BYTES = 20 * 1024 * 1024


class AIStreamError(RuntimeError):
    """A streaming request failed at some point.

    The caller has not received a usable answer and should fall back to a
    regular single-shot request.
    """


async def _sse_lines(chunks: Any) -> AsyncIterator[str]:
    """Yield non-empty SSE lines from an async iterable of byte chunks."""
    buffer = ""
    async for raw in chunks:
        buffer += raw.decode("utf-8", "replace")
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            line = line.strip()
            if line:
                yield line
    tail = buffer.strip()
    if tail:
        yield tail


async def _transcode_to_wav(audio: bytes) -> bytes | None:
    """Convert audio to 16 kHz mono WAV with ffmpeg; None when unavailable."""
    try:
        process = await asyncio.create_subprocess_exec(
            "ffmpeg", "-v", "error", "-i", "pipe:0",
            "-ac", "1", "-ar", "16000", "-f", "wav", "pipe:1",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err = await asyncio.wait_for(process.communicate(audio), timeout=60)
    except (FileNotFoundError, asyncio.TimeoutError, OSError) as exc:
        logger.warning("ffmpeg transcoding is unavailable: %s", exc)
        return None
    if process.returncode != 0 or not out:
        logger.warning("ffmpeg failed: %s", err[:200])
        return None
    return out


def _media_log_info(value: str) -> dict[str, Any]:
    """Describe a media reference without logging its bytes or signed URL."""
    if value.startswith("data:") and "," in value:
        header, encoded = value.split(",", 1)
        info: dict[str, Any] = {"source": "inline", "mime": header[5:].split(";")[0]}
        if ";base64" in header:
            info["bytes"] = max(0, len(encoded) * 3 // 4 - (len(encoded) - len(encoded.rstrip("="))))
        return info
    return {"source": "remote"}


def _request_log_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Whitelist request details; text is complete, media is metadata only."""
    summary = {key: payload[key] for key in (
        "model", "temperature", "max_tokens", "stream", "asr_options", "parameters"
    ) if key in payload}
    messages = payload.get("messages") or (payload.get("input") or {}).get("messages") or []
    logged_messages = []
    for message in messages:
        content = message.get("content")
        entry: dict[str, Any] = {"role": message.get("role")}
        if isinstance(content, str):
            entry["text"] = content
        elif isinstance(content, list):
            parts = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                if isinstance(part.get("text"), str):
                    parts.append({"type": "text", "text": part["text"]})
                elif "image_url" in part:
                    ref = part["image_url"]
                    value = ref.get("url", "") if isinstance(ref, dict) else str(ref)
                    parts.append({"type": "image", **_media_log_info(value)})
                elif "input_audio" in part:
                    ref = part["input_audio"]
                    value = ref.get("data", "") if isinstance(ref, dict) else str(ref)
                    parts.append({"type": "audio", **_media_log_info(value)})
            entry["content"] = parts
        logged_messages.append(entry)
    summary["messages"] = logged_messages
    if payload.get("tools"):
        summary["tools"] = [tool.get("function", {}).get("name") for tool in payload["tools"]]
    return summary

# Model catalog for the /models menu. It lives in the ``AI_MODELS`` environment
# variable (see .env) so it can be updated without touching the code; the
# built-in fallback is ``bot.config.DEFAULT_AI_MODELS``.
#
# Entries the account cannot use are hidden at startup (``GET /models``), and
# models flagged/detected as vision-capable can analyze images.

# Substrings used to detect multimodal (image input) models.
_VISION_MARKERS = ("-vl", "vl-", "omni", "vision", "ocr")
# Qwen3.5 and later flagship / plus / flash models accept images (and video).
_QWEN_MULTIMODAL_RE = re.compile(r"^qwen3\.[5-9]-(?:max|plus|flash)")


def is_vision_model(model: str) -> bool:
    """Return True if ``model`` supports image input (vision).

    An explicit ``:vision`` / ``:novision`` flag in ``AI_MODELS`` always wins.
    Otherwise the id is matched against the Qwen-VL / Omni / OCR families and
    the Qwen3.5+ flagship, plus and flash models, which are multimodal.
    """
    lowered = (model or "").lower()
    for entry in config.ai_models:
        if str(entry.get("id", "")).lower() == lowered:
            flag = entry.get("vision")
            if flag is not None:
                return bool(flag)
            break
    if any(marker in lowered for marker in _VISION_MARKERS):
        return True
    return bool(_QWEN_MULTIMODAL_RE.match(lowered))


# Tool exposed to the chat model so it can answer with a generated image.
IMAGE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "generate_image",
        "description": (
            "Generate an image from a text description and send it to the user. "
            "Use this ONLY when the user explicitly asks to draw, create or "
            "generate a picture/image (for example 'draw a cat', 'нарисуй кота')."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": (
                        "A detailed description of the image to generate, "
                        "including subject, style, lighting and composition."
                    ),
                }
            },
            "required": ["prompt"],
        },
    },
}


@dataclass(slots=True)
class ToolCall:
    """A single tool call requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(slots=True)
class ChatResult:
    """Assistant reply: text content and/or requested tool calls.

    ``elapsed`` is the wall time of the API call in seconds (0.0 when unknown,
    e.g. for a result built by a test double); ``usage`` is the raw ``usage``
    object returned by the API.
    """

    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    elapsed: float = 0.0
    usage: dict[str, Any] | None = None

    @property
    def total_tokens(self) -> int | None:
        """Total tokens reported by the API, if any."""
        if not self.usage:
            return None
        value = self.usage.get("total_tokens")
        return int(value) if isinstance(value, (int, float)) else None


@dataclass(slots=True)
class ImageResult:
    """Generated image: its URL, the downloaded bytes and the elapsed time."""

    url: str | None = None
    data: bytes | None = None
    elapsed: float = 0.0


def _guess_mime(image_bytes: bytes) -> str:
    """Sniff the MIME type of an image from its magic bytes."""
    if image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if image_bytes[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if image_bytes[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if image_bytes[:2] == b"BM":
        return "image/bmp"
    if image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


def _image_data_url(image_bytes: bytes) -> str:
    """Encode raw image bytes into a base64 data URL."""
    mime = _guess_mime(image_bytes)
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _extract_usage(data: dict[str, Any]) -> dict[str, Any] | None:
    """Return the API ``usage`` block (token counters), when present."""
    usage = data.get("usage")
    return usage if isinstance(usage, dict) and usage else None


def _derive_native_base_url(compatible_base_url: str) -> str:
    """Convert an OpenAI-compatible base URL to the DashScope native API base.

    ``.../compatible-mode/v1`` -> ``.../api/v1`` (used by image generation).
    """
    base = compatible_base_url.rstrip("/")
    suffix = "/compatible-mode/v1"
    if base.endswith(suffix):
        return base[: -len(suffix)] + "/api/v1"
    return config.image_api_url or f"{base}/api/v1"


class AIClient:
    """Async client for Qwen (OpenAI-compatible chat + DashScope native APIs)."""

    def __init__(self, *, api_key: str | None = None, base_url: str | None = None) -> None:
        self.api_key = api_key or config.ai_api_key
        self.base_url = (base_url or config.ai_base_url).rstrip("/")
        self.native_base_url = (
            config.image_api_url.rstrip("/")
            or _derive_native_base_url(self.base_url)
        )
        self._session: aiohttp.ClientSession | None = None
        # Downloads use their own session: presigned storage URLs reject the
        # API's Authorization header (see ``download_image``).
        self._download_session: aiohttp.ClientSession | None = None
        # Model IDs reported as available by the API (``None`` = unknown).
        self._available_ids: set[str] | None = None

    # --- Session management ------------------------------------------------

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers=self._headers,
                timeout=aiohttp.ClientTimeout(total=config.request_timeout),
            )
        return self._session

    async def _get_download_session(self) -> aiohttp.ClientSession:
        """Session for fetching generated files — **without** default headers.

        Generated images are returned as presigned Alibaba Cloud OSS URLs: the
        signature lives in the query string, and OSS answers ``403`` when the
        request also carries an ``Authorization`` header ("only one auth
        mechanism allowed"). Session-level headers in aiohttp are merged into
        every request and cannot be dropped per call, so downloads need a
        separate, header-free session.
        """
        if self._download_session is None or self._download_session.closed:
            self._download_session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=config.image_timeout),
            )
        return self._download_session

    async def close(self) -> None:
        """Close the underlying HTTP sessions."""
        for attribute in ("_session", "_download_session"):
            session = getattr(self, attribute)
            if session is not None and not session.closed:
                await session.close()
            setattr(self, attribute, None)

    # --- Model catalog -----------------------------------------------------

    async def list_models(self) -> set[str] | None:
        """Return the set of model IDs available to this API key.

        Uses the OpenAI-compatible ``GET {base_url}/models`` endpoint. Returns
        ``None`` when the request fails so callers can fall back gracefully.
        """
        url = f"{self.base_url}/models"
        logger.info("AI API request: method=GET path=/models")
        try:
            session = await self._get_session()
            async with session.get(url) as resp:
                body = await resp.text()
                if resp.status != 200:
                    logger.warning(
                        "Model list unavailable (status %s): %s",
                        resp.status,
                        body[:200],
                    )
                    return None
                data = json.loads(body)
        except (asyncio.TimeoutError, aiohttp.ClientError, json.JSONDecodeError) as exc:
            logger.warning("Model list request failed: %s", exc)
            return None

        ids = {
            item["id"]
            for item in (data.get("data") or [])
            if isinstance(item, dict) and item.get("id")
        }
        return ids or None

    async def refresh_available_models(self) -> list[dict[str, Any]]:
        """Fetch the live model list and cache the curated, available subset."""
        self._available_ids = await self.list_models()
        return self.available_models()

    def available_models(self) -> list[dict[str, Any]]:
        """Catalog models, filtered to those the account can actually use."""
        catalog = config.ai_models
        if not self._available_ids:
            return list(catalog)
        filtered = [m for m in catalog if m["id"] in self._available_ids]
        return filtered or list(catalog)

    def is_model_available(self, model: str) -> bool | None:
        """Availability of ``model``: True/False, or ``None`` when unknown."""
        if self._available_ids is None:
            return None
        return model in self._available_ids

    # --- Chat completions --------------------------------------------------

    async def chat_completion(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str | None:
        """Send a chat completion request and return the assistant text."""
        result = await self.chat_completion_result(
            messages, model, temperature, max_tokens
        )
        return result.content

    async def chat_completion_result(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResult:
        """Chat completion returning the text plus timing and token usage."""
        payload = self._chat_payload(messages, model, temperature, max_tokens)
        status, data, elapsed = await self._post_chat(payload)
        if status == 200 and data is not None:
            return ChatResult(
                content=self._finalize(self._extract_content(data)),
                elapsed=elapsed,
                usage=_extract_usage(data),
            )
        return ChatResult(elapsed=elapsed)

    async def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResult:
        """Chat completion with function calling enabled.

        Falls back to a plain completion if the model/region rejects tools.
        """
        payload = self._chat_payload(messages, model, temperature, max_tokens)
        payload["tools"] = tools if tools is not None else [IMAGE_TOOL]

        status, data, elapsed = await self._post_chat(payload)
        if status == 200 and data is not None:
            return self._parse_result(data, elapsed)

        # Some models do not support the tools parameter -> retry without it.
        payload.pop("tools", None)
        status, data, elapsed = await self._post_chat(payload)
        if status == 200 and data is not None:
            return ChatResult(
                content=self._finalize(self._extract_content(data)),
                elapsed=elapsed,
                usage=_extract_usage(data),
            )
        return ChatResult(elapsed=elapsed)

    async def chat_completion_with_image(
        self,
        text: str,
        image_bytes: bytes,
        messages: list[dict[str, Any]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str | None:
        """Send a chat completion that includes a single image."""
        result = await self.chat_with_image_result(
            text, image_bytes, messages, model, temperature, max_tokens
        )
        return result.content

    async def chat_with_image_result(
        self,
        text: str,
        image_bytes: bytes,
        messages: list[dict[str, Any]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResult:
        """Vision completion returning the text plus timing and token usage."""
        payload = self._chat_payload(
            self.vision_messages(messages, text, image_bytes),
            model,
            temperature,
            max_tokens,
        )
        status, data, elapsed = await self._post_chat(payload)
        if status == 200 and data is not None:
            return ChatResult(
                content=self._finalize(self._extract_content(data)),
                elapsed=elapsed,
                usage=_extract_usage(data),
            )
        return ChatResult(elapsed=elapsed)

    @staticmethod
    def vision_messages(
        messages: list[dict[str, Any]], text: str, image_bytes: bytes
    ) -> list[dict[str, Any]]:
        """The message list with one image appended as the latest user turn."""
        full_messages = [dict(entry) for entry in messages]
        full_messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {
                        "type": "image_url",
                        "image_url": {"url": _image_data_url(image_bytes)},
                    },
                ],
            }
        )
        return full_messages

    # --- Streaming -----------------------------------------------------------

    async def stream_chat_completion(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        on_delta: Any = None,
    ) -> ChatResult:
        """Streaming chat completion over SSE.

        ``on_delta`` is awaited with the text accumulated so far after every
        chunk so the caller can show the answer progressively. Raises
        :class:`AIStreamError` on any failure (HTTP status, mid-stream error,
        stall, empty stream) — the caller should then fall back to a regular
        single-shot request.
        """
        payload = self._chat_payload(messages, model, temperature, max_tokens)
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}
        if tools:
            payload["tools"] = list(tools)

        request_id = uuid4().hex[:12]
        started = time.monotonic()
        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "AI API stream request id=%s path=/chat/completions payload=%s",
                request_id,
                json.dumps(_request_log_payload(payload), ensure_ascii=False),
            )

        parts: list[str] = []
        tool_acc: dict[int, dict[str, str]] = {}
        usage: dict[str, Any] | None = None
        saw_done = False
        malformed = 0

        async for line in self._stream_lines(payload):
            if not line.startswith("data:"):
                continue  # SSE comments, heartbeats, blank lines
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                saw_done = True
                break
            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                malformed += 1
                if malformed > 10:
                    raise AIStreamError("too many malformed SSE chunks")
                continue
            if not isinstance(event, dict):
                continue
            if event.get("error"):
                raise AIStreamError(f"API stream error: {str(event['error'])[:200]}")
            if isinstance(event.get("usage"), dict):
                usage = event["usage"]
            for choice in event.get("choices") or []:
                if not isinstance(choice, dict):
                    continue
                delta = choice.get("delta")
                if not isinstance(delta, dict):
                    delta = {}
                piece = self._normalize_content(delta.get("content"))
                if piece:
                    parts.append(piece)
                    if on_delta is not None:
                        try:
                            await on_delta("".join(parts))
                        except Exception as exc:  # noqa: BLE001 - UI must not kill the stream
                            logger.warning("Stream on_delta failed: %s", exc)
                for raw in delta.get("tool_calls") or []:
                    if not isinstance(raw, dict):
                        continue
                    slot = tool_acc.setdefault(
                        int(raw.get("index", 0) or 0),
                        {"id": "", "name": "", "arguments": ""},
                    )
                    slot["id"] = slot["id"] or str(raw.get("id") or "")
                    function = raw.get("function") or {}
                    if function.get("name"):
                        slot["name"] += str(function["name"])
                    if function.get("arguments"):
                        slot["arguments"] += str(function["arguments"])

        content = "".join(parts) or None
        if not saw_done and content is None and not tool_acc and usage is None:
            raise AIStreamError("the stream produced no data")

        tool_calls: list[ToolCall] = []
        for index, slot in sorted(tool_acc.items()):
            try:
                arguments = json.loads(slot["arguments"] or "{}")
            except json.JSONDecodeError:
                arguments = {}
            tool_calls.append(
                ToolCall(
                    id=slot["id"] or f"call_{index}",
                    name=slot["name"],
                    arguments=arguments if isinstance(arguments, dict) else {},
                )
            )

        logger.info(
            "AI API stream finished id=%s elapsed=%.2fs chunks=%d tools=%d",
            request_id,
            time.monotonic() - started,
            len(parts),
            len(tool_calls),
        )
        return ChatResult(
            content=self._finalize(content),
            tool_calls=tool_calls,
            elapsed=time.monotonic() - started,
            usage=usage,
        )

    async def _stream_lines(self, payload: dict[str, Any]) -> AsyncIterator[str]:
        """POST the streaming request and yield its raw SSE lines."""
        target = f"{self.base_url}/chat/completions"
        timeout = aiohttp.ClientTimeout(
            total=config.request_timeout,
            sock_read=max(10, config.stream_stall_timeout),
        )
        try:
            session = await self._get_session()
            async with session.post(target, json=payload, timeout=timeout) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.error(
                        "AI stream error (status %s): %s", resp.status, body[:500]
                    )
                    raise AIStreamError(f"HTTP {resp.status}")
                async for line in _sse_lines(resp.content):
                    yield line
        except AIStreamError:
            raise
        except (asyncio.TimeoutError, aiohttp.ClientError) as exc:
            raise AIStreamError(str(exc)) from exc

    # --- Speech recognition ------------------------------------------------

    async def transcribe_audio(
        self,
        audio_bytes: bytes,
        *,
        mime: str = "audio/ogg",
        model: str | None = None,
        language: str | None = None,
    ) -> str | None:
        """Transcribe audio using the OpenAI-compatible Qwen3-ASR-Flash model.

        Telegram voice messages are OGG/Opus, which this model accepts directly
        (base64 data URL, up to 7.5 MB of source audio / 5 minutes).
        """
        if len(audio_bytes) > config.voice_max_bytes:
            logger.warning("Audio exceeds the configured ASR input limit")
            return None
        asr_options: dict[str, Any] = {"enable_itn": True}
        if language:
            asr_options["language"] = language

        # A 400 with an empty body gives no hint about the cause, so a few
        # variants are tried. Qwen3-ASR-Flash is documented as
        # ``qwen3-asr-flash`` over a ``data:<mime>;base64,`` URI; the configured
        # model goes first, the documented one is the fallback, and when OGG is
        # refused the clip is transcoded to WAV (needs ffmpeg on the server).
        chosen = model or config.asr_model
        models = [chosen]
        if model is None and chosen != "qwen3-asr-flash":
            models.append("qwen3-asr-flash")

        variants: list[tuple[str, bytes]] = [(mime, audio_bytes)]
        wav: bytes | None = None
        wav_tried = False

        for asr_model in models:
            index = 0
            while index < len(variants):
                variant_mime, variant_bytes = variants[index]
                index += 1
                payload: dict[str, Any] = {
                    "model": asr_model,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_audio",
                                    "input_audio": {
                                        "data": f"data:{variant_mime};base64,"
                                        + base64.b64encode(variant_bytes).decode("ascii")
                                    },
                                }
                            ],
                        }
                    ],
                    "stream": False,
                    "asr_options": asr_options,
                }
                status, data, _elapsed = await self._post_chat(payload, retries=2)
                if status == 200 and data is not None:
                    return self._extract_content(data)
                logger.error(
                    "Speech recognition failed (model=%s, status %s, mime=%s)",
                    asr_model,
                    status,
                    variant_mime,
                )
                if status not in (400, 422):
                    return None  # network / auth / rate-limit: variants cannot help
                if not wav_tried:
                    wav_tried = True
                    wav = await _transcode_to_wav(audio_bytes)
                    if wav is not None and len(wav) <= config.voice_max_bytes * 4:
                        variants.append(("audio/wav", wav))
        return None

    # --- Image generation --------------------------------------------------

    async def generate_image_url(
        self,
        prompt: str,
        *,
        model: str | None = None,
        size: str | None = None,
    ) -> str | None:
        """Generate an image and return its public URL (valid for ~24 hours).

        Uses the DashScope native synchronous image-synthesis endpoint. The URL
        can be passed straight to Telegram (e.g. ``InlineQueryResultPhoto``) or
        downloaded with :meth:`generate_image`.
        """
        return (await self.generate_image_result(
            prompt, model=model, size=size, download=False
        )).url

    async def generate_image_result(
        self,
        prompt: str,
        *,
        model: str | None = None,
        size: str | None = None,
        download: bool = True,
    ) -> ImageResult:
        """Generate an image, optionally downloading it, with elapsed time."""
        started = time.monotonic()
        url = f"{self.native_base_url}/services/aigc/multimodal-generation/generation"
        parameters: dict[str, Any] = {
            "prompt_extend": True,
            "watermark": False,
            "n": 1,
        }
        if size:
            parameters["size"] = size

        payload: dict[str, Any] = {
            "model": model or config.image_model,
            "input": {
                "messages": [
                    {"role": "user", "content": [{"text": prompt}]}
                ]
            },
            "parameters": parameters,
        }

        status, data, elapsed = await self._post_chat(
            payload, url=url, timeout=config.image_timeout
        )
        if status != 200 or not data:
            logger.error("Image generation failed (status %s)", status)
            return ImageResult(elapsed=time.monotonic() - started)

        image_url = self._extract_image_url(data)
        if not image_url:
            logger.error("Image response has no URL: %s", str(data)[:300])
            return ImageResult(elapsed=time.monotonic() - started)

        image_bytes = await self.download_image(image_url) if download else None
        return ImageResult(
            url=image_url,
            data=image_bytes,
            elapsed=time.monotonic() - started,
        )

    async def generate_image(
        self,
        prompt: str,
        *,
        model: str | None = None,
        size: str | None = None,
    ) -> bytes | None:
        """Generate an image from a prompt and return the raw image bytes."""
        result = await self.generate_image_result(prompt, model=model, size=size)
        return result.data

    @staticmethod
    def _extract_image_url(data: dict[str, Any]) -> str | None:
        output = data.get("output") or {}

        choices = output.get("choices") or []
        if choices:
            message = choices[0].get("message") or {}
            for part in message.get("content") or []:
                if isinstance(part, dict) and part.get("image"):
                    return part["image"]

        results = output.get("results") or []
        if results and isinstance(results[0], dict):
            return results[0].get("url") or results[0].get("image_url")
        return None

    async def download_image(self, url: str) -> bytes | None:
        """Download a generated image from its (presigned) URL.

        Tries anonymously first — presigned OSS URLs answer ``403`` when an
        ``Authorization`` header is present — then falls back to an
        authenticated request only when the file is served by the API itself.
        The presigned query string is never logged.
        """
        if not url:
            return None

        safe_url = url.split("?", 1)[0]
        timeout = aiohttp.ClientTimeout(total=config.image_timeout)

        # Never forward the API key to an external image/storage host.
        source = urlsplit(url)
        trusted_origins = {
            (parsed.scheme, parsed.netloc)
            for parsed in (urlsplit(self.base_url), urlsplit(self.native_base_url))
        }
        header_variants = [None]
        if (source.scheme, source.netloc) in trusted_origins:
            header_variants.append(self._headers)
        for attempt, headers in enumerate(header_variants, start=1):
            try:
                session = await self._get_download_session()
                async with session.get(url, headers=headers, timeout=timeout) as resp:
                    declared = resp.content_length
                    if isinstance(declared, int) and declared > MAX_DOWNLOAD_BYTES:
                        logger.error("Image at %s is too large (%d bytes)", safe_url, declared)
                        return None
                    body = await resp.read()
                    if len(body) > MAX_DOWNLOAD_BYTES:
                        logger.error("Image at %s is too large", safe_url)
                        return None
                    if resp.status == 200 and body:
                        return body
                    logger.warning(
                        "Image download attempt %d failed (status %s) for %s: %s",
                        attempt,
                        resp.status,
                        safe_url,
                        body[:300],
                    )
            except (asyncio.TimeoutError, aiohttp.ClientError) as exc:
                logger.error("Image download failed for %s: %s", safe_url, exc)
                return None
            except Exception as exc:  # noqa: BLE001 - a bad URL must not kill a reply
                logger.warning(
                    "Image download attempt %d errored for %s: %s",
                    attempt,
                    safe_url,
                    exc,
                )

        logger.error("Image download failed for %s (all attempts)", safe_url)
        return None

    # --- Internals ---------------------------------------------------------

    @staticmethod
    def is_vision_model(model: str) -> bool:
        """Check whether the given model supports vision (image input)."""
        return is_vision_model(model)

    @staticmethod
    def _chat_payload(
        messages: list[dict[str, Any]],
        model: str | None,
        temperature: float | None,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        return {
            "model": model or config.default_model,
            "messages": AIClient._with_style_hint(messages),
            "temperature": (
                config.default_temperature if temperature is None else temperature
            ),
            "max_tokens": (
                config.default_max_tokens if max_tokens is None else max_tokens
            ),
        }

    @staticmethod
    def _with_style_hint(
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Append the Telegram formatting rules to the system prompt.

        Telegram cannot render LaTeX or sub/superscripts, so the model is asked
        to use Unicode notation directly. The caller's list is not modified.
        """
        hint = config.style_hint if config.enable_style_hint else ""
        if hint == DEFAULT_STYLE_HINT and config.render_formula_images:
            hint = FORMULA_STYLE_HINT
        if not hint:
            return messages

        result = [dict(entry) for entry in messages]
        for entry in result:
            if entry.get("role") == "system":
                content = entry.get("content")
                if isinstance(content, str) and hint not in content:
                    entry["content"] = f"{content}\n\n{hint}".strip()
                return result
        result.insert(0, {"role": "system", "content": hint})
        return result

    @staticmethod
    def _finalize(content: str | None) -> str | None:
        """Rewrite scientific notation in the answer as Unicode for Telegram."""
        if not content or config.render_formula_images or not config.normalize_formulas:
            return content
        return normalize_formulas(content)

    async def _post_chat(
        self,
        payload: dict[str, Any],
        *,
        url: str | None = None,
        timeout: int | None = None,
        retries: int = 3,
    ) -> tuple[int | None, dict[str, Any] | None, float]:
        """POST a JSON payload with retry on transient errors.

        Returns ``(status, json, elapsed)``; ``status`` is ``None`` on network
        failure and ``elapsed`` is the wall time in seconds, retries included.
        """
        target = url or f"{self.base_url}/chat/completions"
        request_timeout = aiohttp.ClientTimeout(total=timeout or config.request_timeout)
        started = time.monotonic()
        request_id = uuid4().hex[:12]
        endpoint = urlsplit(target).path
        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "AI API request id=%s path=%s timeout=%ss payload=%s",
                request_id, endpoint, request_timeout.total,
                json.dumps(_request_log_payload(payload), ensure_ascii=False),
            )

        def _elapsed() -> float:
            return time.monotonic() - started

        for attempt in range(1, retries + 1):
            try:
                logger.info("AI API send id=%s attempt=%s/%s", request_id, attempt, retries)
                session = await self._get_session()
                async with session.post(
                    target, json=payload, timeout=request_timeout
                ) as resp:
                    body = await resp.text()
                    logger.info(
                        "AI API response id=%s attempt=%s status=%s elapsed=%.2fs",
                        request_id, attempt, resp.status, _elapsed(),
                    )

                    if resp.status == 200:
                        try:
                            return 200, json.loads(body), _elapsed()
                        except json.JSONDecodeError:
                            logger.error("AI API returned invalid JSON: %s", body[:300])
                            return resp.status, None, _elapsed()

                    if resp.status in {429, 500, 502, 503, 504} and attempt < retries:
                        delay = min(2**attempt, 8)
                        logger.warning(
                            "AI API transient error %s (attempt %s/%s); "
                            "retrying in %ss: %s",
                            resp.status,
                            attempt,
                            retries,
                            delay,
                            body[:200],
                        )
                        await asyncio.sleep(delay)
                        continue

                    logger.error(
                        "AI API error (status %s %s): %s | request-id=%s content-type=%s",
                        resp.status,
                        resp.reason,
                        body[:500],
                        resp.headers.get("x-request-id") or resp.headers.get("x-dashscope-request-id"),
                        resp.headers.get("content-type"),
                    )
                    return resp.status, None, _elapsed()

            except asyncio.TimeoutError:
                logger.warning(
                    "AI API request timed out id=%s (attempt %s/%s)",
                    request_id, attempt, retries,
                )
                if attempt >= retries:
                    return None, None, _elapsed()
                await asyncio.sleep(min(2**attempt, 8))
            except aiohttp.ClientError as exc:
                logger.error("AI API request failed id=%s: %s", request_id, exc)
                if attempt >= retries:
                    return None, None, _elapsed()
                await asyncio.sleep(min(2**attempt, 8))

        return None, None, _elapsed()

    @classmethod
    def _extract_content(cls, data: dict[str, Any]) -> str | None:
        choices = data.get("choices") or []
        if not choices:
            return None
        return cls._normalize_content((choices[0].get("message") or {}).get("content"))

    @staticmethod
    def _normalize_content(content: Any) -> str | None:
        if content is None:
            return None
        if isinstance(content, list):
            parts = [
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("text")
            ]
            return "".join(parts) or None
        return str(content)

    @classmethod
    def _parse_result(cls, data: dict[str, Any], elapsed: float = 0.0) -> ChatResult:
        choices = data.get("choices") or []
        if not choices:
            return ChatResult()
        message = choices[0].get("message") or {}

        calls: list[ToolCall] = []
        for raw in message.get("tool_calls") or []:
            function = raw.get("function") or {}
            arguments_raw = function.get("arguments") or "{}"
            if isinstance(arguments_raw, str):
                try:
                    arguments = json.loads(arguments_raw)
                except json.JSONDecodeError:
                    arguments = {}
            elif isinstance(arguments_raw, dict):
                arguments = arguments_raw
            else:
                arguments = {}
            calls.append(
                ToolCall(
                    id=raw.get("id", ""),
                    name=function.get("name", ""),
                    arguments=arguments if isinstance(arguments, dict) else {},
                )
            )

        return ChatResult(
            content=cls._finalize(cls._normalize_content(message.get("content"))),
            tool_calls=calls,
            elapsed=elapsed,
            usage=_extract_usage(data),
        )


# Shared singleton used across handlers.
ai_client = AIClient()
