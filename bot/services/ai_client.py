from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import re
import shutil
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator
from urllib.parse import urlsplit
from uuid import uuid4

import aiohttp

from bot.config import (
    DEFAULT_RICH_STYLE_HINT,
    DEFAULT_STYLE_HINT,
    FORMULA_STYLE_HINT,
    config,
)
from bot.services.image_tools import canonical_operations
from bot.services.science import normalize_formulas

logger = logging.getLogger(__name__)

# Upper bound for a downloaded generated image.
MAX_DOWNLOAD_BYTES = 20 * 1024 * 1024

# Error bodies that mean "the account/model quota is exhausted" rather than a
# transient glitch. Repeating the same request is pointless — the client moves
# on to the next model in the fallback chain instead.
_QUOTA_ERROR_RE = re.compile(
    r"quota|rate[ _-]?limit|too many requests|insufficient|arrears|balance",
    re.IGNORECASE,
)


class AIStreamError(RuntimeError):
    """A streaming request failed at some point.

    The caller has not received a usable answer and should fall back to a
    regular single-shot request. ``status`` carries the HTTP code when the
    failure came from the API, so the caller can tell a model-specific problem
    from one no other model could survive (see ``_chain_fatal``).
    """

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


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
    except (TimeoutError, FileNotFoundError, OSError) as exc:
        logger.warning("ffmpeg transcoding is unavailable: %s", exc)
        return None
    if process.returncode != 0 or not out:
        logger.warning("ffmpeg failed: %s", err[:200])
        return None
    return out


async def _transcode_to_pcm16(audio: bytes, mime: str = "audio/ogg") -> bytes | None:
    """Convert any audio to 16 kHz mono raw PCM16 (s16le) with ffmpeg.

    Returns ``None`` when ffmpeg is unavailable or transcoding fails.
    The result has no header — just interleaved 16-bit signed samples
    suitable for the WebSocket Realtime ASR API.
    """
    try:
        process = await asyncio.create_subprocess_exec(
            "ffmpeg", "-v", "error",
            "-i", "pipe:0",
            "-ac", "1",
            "-ar", "16000",
            "-f", "s16le",
            "pipe:1",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err = await asyncio.wait_for(process.communicate(audio), timeout=60)
    except (TimeoutError, FileNotFoundError, OSError) as exc:
        logger.warning("ffmpeg PCM16 transcoding unavailable: %s", exc)
        return None
    if process.returncode != 0 or not out:
        logger.warning("ffmpeg PCM16 failed: %s", err[:200])
        return None
    return out


def ffmpeg_available() -> bool:
    """Return True when an ``ffmpeg`` executable is available on PATH.

    Voice recognition converts Telegram's OGG/Opus notes to raw PCM16 for the
    Realtime ASR WebSocket, so ffmpeg is a runtime dependency of the voice
    feature (see README → «Установка ffmpeg»).
    """
    return shutil.which("ffmpeg") is not None


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
    for entry in (*config.ai_models, *config.ai_fallback_models):
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

# Tool exposed to the chat model so it can answer with a Telegram sticker.
STICKER_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "generate_sticker",
        "description": (
            "Generate a Telegram sticker (a 512x512 die-cut sticker image) from "
            "a short description and send it to the user. Use this ONLY when "
            "the user explicitly asks for a STICKER (for example 'make a "
            "sticker of a cat', 'sticker with a thumbs up', 'стикер с котом'). "
            "For ordinary pictures/images use generate_image instead."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": (
                        "A short description of the sticker subject and mood, "
                        "e.g. 'a smiling cartoon cat giving a thumbs up'."
                    ),
                }
            },
            "required": ["prompt"],
        },
    },
}

# Appended to the user's prompt so the model produces a clean sticker-style
# image that survives the center-crop to a 512x512 square.
_STICKER_PROMPT_SUFFIX = (
    ", die-cut sticker style, single centered subject, bold clean outlines, "
    "vibrant flat colors, thick white border around the subject, isolated on "
    "a plain solid white background, no text, no watermark"
)

# Turns the text-to-image endpoint into a faithful photo restorer: the image
# is re-rendered by the network, but the instruction pins the content down.
# Used for the "enhance with neural network" path (guest mode's only option,
# the alternative to the local Pillow pipeline in bot.services.image_tools).
ENHANCE_AI_INSTRUCTION = (
    "Restore and improve the quality of this photograph. Keep the subject, "
    "composition, faces, lettering and every object exactly as they are — do "
    "not add, remove or move anything. Only raise the technical quality: "
    "sharper fine detail, less noise and grain, correct exposure, natural "
    "colours, wider dynamic range, higher effective resolution."
)

_ENHANCE_AI_HINTS = {
    "upscale": "higher effective resolution and enlarged fine detail",
    "denoise": "removing noise and grain",
    "sharpen": "crisper edges and micro-detail",
    "color": "richer natural colours",
    "contrast": "balanced contrast",
    "brightness": "corrected brightness and exposure",
    "restore": "full restoration of an old or faded photograph",
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
    # Model that actually produced the answer (may differ from the requested
    # one when the fallback chain was used).
    model: str | None = None

    @property
    def total_tokens(self) -> int | None:
        """Total tokens reported by the API, if any."""
        if not self.usage:
            return None
        value = self.usage.get("total_tokens")
        return int(value) if isinstance(value, (int, float)) else None

    @property
    def input_tokens(self) -> int | None:
        """Prompt (input) tokens reported by the API, if any.

        The OpenAI-compatible endpoint reports ``prompt_tokens``; the
        DashScope native format uses ``input_tokens``. Both are accepted.
        """
        return self._usage_int("prompt_tokens", "input_tokens")

    @property
    def output_tokens(self) -> int | None:
        """Completion (output) tokens reported by the API, if any.

        The OpenAI-compatible endpoint reports ``completion_tokens``; the
        DashScope native format uses ``output_tokens``. Both are accepted.
        """
        return self._usage_int("completion_tokens", "output_tokens")

    def _usage_int(self, *keys: str) -> int | None:
        """First numeric value among ``keys`` in the usage block, if any."""
        if not self.usage:
            return None
        for key in keys:
            value = self.usage.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return int(value)
        return None


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


# Telegram static-sticker constraints: WebP, one side exactly 512px (other
# ≤512), file ≤512 KB.
STICKER_SIZE = 512
STICKER_MAX_BYTES = 512 * 1024


def to_sticker_webp(image_bytes: bytes, size: int = STICKER_SIZE) -> bytes | None:
    """Convert image bytes into a Telegram-ready sticker (``size``×``size`` WebP).

    The image is center-cropped to a square (no distortion), resized, and
    re-encoded as WebP with decreasing quality until it fits the 512 KB cap.
    Returns None when Pillow is unavailable or the bytes are not a valid image.
    """
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover - Pillow is a declared dependency
        logger.warning("Pillow is not installed; cannot build a sticker")
        return None
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
        width, height = image.size
        side = min(width, height)
        left, top = (width - side) // 2, (height - side) // 2
        image = image.crop((left, top, left + side, top + side))
        image = image.resize((size, size), Image.LANCZOS)
        data = b""
        for quality in (90, 80, 70, 60, 50, 40):
            buffer = io.BytesIO()
            image.save(buffer, format="WEBP", quality=quality, method=6)
            data = buffer.getvalue()
            if len(data) <= STICKER_MAX_BYTES:
                return data
        return data or None
    except Exception as exc:  # noqa: BLE001 - a bad image must not kill a reply
        logger.error("Sticker conversion failed: %s", exc)
        return None


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


# HTTP statuses that are not model-specific: switching to the next model in the
# fallback chain cannot possibly help, so the chain stops instead of repeating a
# guaranteed failure N more times (each with its own retries). 401/403 mean the
# key or the account is wrong — every model would answer the same way, and an
# invalid key would otherwise burn the whole chain on *every* message.
# 413 means the request body is too large for the endpoint itself.
_CHAIN_FATAL_STATUSES = frozenset({401, 403, 413})


def _chain_fatal(status: int | None) -> bool:
    """True when trying the next model cannot help and the chain should stop."""
    return status in _CHAIN_FATAL_STATUSES


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
        except (TimeoutError, aiohttp.ClientError, json.JSONDecodeError) as exc:
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

    def model_chain(self, model: str | None, *, require_vision: bool = False) -> list[str]:
        """Ordered models to try for one request.

        Starts with the requested model, then every entry *after* it in the
        ``AI_MODELS`` list, and finally the emergency ``AI_FALLBACK_MODELS``.
        Unavailable catalog entries (failed ``GET /models``) are skipped, as
        are non-vision models when ``require_vision`` is set. The requested
        model is always first so the caller's choice is honoured.
        """
        primary = model or config.default_model
        catalog = [str(entry.get("id", "")) for entry in config.ai_models]
        following = (
            catalog[catalog.index(primary) + 1 :] if primary in catalog else catalog
        )
        candidates = [primary, *following]
        candidates.extend(
            str(entry.get("id", "")) for entry in config.ai_fallback_models
        )

        chain: list[str] = []
        for model_id in candidates:
            if not model_id or model_id in chain:
                continue
            if model_id != primary:
                if self._available_ids and model_id not in self._available_ids:
                    continue
                if require_vision and not is_vision_model(model_id):
                    continue
            chain.append(model_id)
        return chain or [primary]

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
        """Chat completion returning the text plus timing and token usage.

        When a model does not answer (API error, quota exhausted, empty
        reply), the next model in :meth:`model_chain` is tried automatically.
        """
        started = time.monotonic()
        for attempt_model in self.model_chain(model):
            payload = self._chat_payload(
                messages, attempt_model, temperature, max_tokens
            )
            status, data, _elapsed = await self._post_chat(payload, retries=2)
            if status == 200 and data is not None:
                content = self._finalize(self._extract_content(data))
                if content:
                    return ChatResult(
                        content=content,
                        elapsed=time.monotonic() - started,
                        usage=_extract_usage(data),
                        model=attempt_model,
                    )
            logger.warning(
                "Model %s did not answer (status %s); trying the next model",
                attempt_model,
                status,
            )
            if _chain_fatal(status):
                logger.error(
                    "HTTP %s is not model-specific; aborting the fallback chain",
                    status,
                )
                break
        return ChatResult(elapsed=time.monotonic() - started)

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

        Falls back to a plain completion if the model/region rejects tools,
        then moves on to the next model in the chain.
        """
        started = time.monotonic()
        for attempt_model in self.model_chain(model):
            payload = self._chat_payload(
                messages, attempt_model, temperature, max_tokens
            )
            payload["tools"] = tools if tools is not None else [IMAGE_TOOL]

            status, data, _elapsed = await self._post_chat(payload, retries=2)
            if status == 200 and data is not None:
                result = self._parse_result(data)
                if result.content or result.tool_calls:
                    result.elapsed = time.monotonic() - started
                    result.model = attempt_model
                    return result

            # Some models do not support the tools parameter -> retry the
            # same model without it, but only for a tool-shaped rejection.
            if status in {400, 404, 422}:
                payload.pop("tools", None)
                status, data, _elapsed = await self._post_chat(payload, retries=2)
                if status == 200 and data is not None:
                    content = self._finalize(self._extract_content(data))
                    if content:
                        return ChatResult(
                            content=content,
                            elapsed=time.monotonic() - started,
                            usage=_extract_usage(data),
                            model=attempt_model,
                        )

            logger.warning(
                "Model %s did not answer with tools (status %s); "
                "trying the next model",
                attempt_model,
                status,
            )
            if _chain_fatal(status):
                logger.error(
                    "HTTP %s is not model-specific; aborting the fallback chain",
                    status,
                )
                break
        return ChatResult(elapsed=time.monotonic() - started)

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
        """Vision completion returning the text plus timing and token usage.

        Only vision-capable models are tried, so a text-only fallback never
        receives an image request.
        """
        started = time.monotonic()
        for attempt_model in self.model_chain(model, require_vision=True):
            payload = self._chat_payload(
                self.vision_messages(messages, text, image_bytes),
                attempt_model,
                temperature,
                max_tokens,
            )
            status, data, _elapsed = await self._post_chat(payload, retries=2)
            if status == 200 and data is not None:
                content = self._finalize(self._extract_content(data))
                if content:
                    return ChatResult(
                        content=content,
                        elapsed=time.monotonic() - started,
                        usage=_extract_usage(data),
                        model=attempt_model,
                    )
            logger.warning(
                "Vision model %s did not answer (status %s); "
                "trying the next model",
                attempt_model,
                status,
            )
            if _chain_fatal(status):
                logger.error(
                    "HTTP %s is not model-specific; aborting the fallback chain",
                    status,
                )
                break
        return ChatResult(elapsed=time.monotonic() - started)

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
        chunk so the caller can show the answer progressively. A model that
        fails is skipped in favour of the next one from :meth:`model_chain`;
        when every model failed, :class:`AIStreamError` is raised and the
        caller should fall back to a regular single-shot request.
        """
        started = time.monotonic()
        last_error: AIStreamError | None = None
        for attempt_model in self.model_chain(model):
            payload = self._chat_payload(
                messages, attempt_model, temperature, max_tokens
            )
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}
            if tools:
                payload["tools"] = list(tools)
            try:
                result = await self._stream_attempt(payload, on_delta)
            except AIStreamError as exc:
                last_error = exc
                logger.warning(
                    "Model %s streaming failed (%s); trying the next model",
                    attempt_model,
                    exc,
                )
                if _chain_fatal(exc.status):
                    logger.error(
                        "HTTP %s is not model-specific; aborting the fallback chain",
                        exc.status,
                    )
                    break
                continue
            if not result.content and not result.tool_calls:
                last_error = AIStreamError("empty answer")
                logger.warning(
                    "Model %s returned an empty streamed answer; "
                    "trying the next model",
                    attempt_model,
                )
                continue
            result.elapsed = time.monotonic() - started
            result.model = attempt_model
            return result

        raise last_error or AIStreamError("no model produced an answer")

    async def _stream_attempt(
        self, payload: dict[str, Any], on_delta: Any = None
    ) -> ChatResult:
        """One streaming request; raises :class:`AIStreamError` on failure."""
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
                    raise AIStreamError(f"HTTP {resp.status}", status=resp.status)
                async for line in _sse_lines(resp.content):
                    yield line
        except AIStreamError:
            raise
        except (TimeoutError, aiohttp.ClientError) as exc:
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
        """Transcribe audio via WebSocket (Qwen-ASR-Realtime, Manual mode).

        Primary path: WebSocket Realtime API (PCM16, 16 kHz).  Telegram voice
        messages arrive as OGG/Opus — they are transcoded to PCM16 with ffmpeg
        before sending.  Falls back to the HTTP chat/completions format when
        WebSocket is unavailable (e.g. no ffmpeg, network restriction).
        """
        if len(audio_bytes) > config.voice_max_bytes:
            logger.warning("Audio exceeds the configured ASR input limit")
            return None

        chosen = model or config.asr_model

        # 1 — WebSocket Realtime API (primary)
        try:
            pcm = await _transcode_to_pcm16(audio_bytes, mime)
        except Exception:  # noqa: BLE001 - transcoding must not break the reply
            pcm = None
        if pcm:
            ws_ok, ws_text = await self._transcribe_ws(chosen, pcm)
            if ws_ok:
                # The API answered: an empty transcript means silence, not a
                # failure — do not burn a second (HTTP) attempt on it.
                return ws_text

        # 2 — HTTP fallback (compat chat/completions)
        asr_options: dict[str, Any] = {"enable_itn": True}
        if language:
            asr_options["language"] = language

        variants: list[tuple[str, bytes]] = [(mime, audio_bytes)]
        wav: bytes | None = None
        wav_tried = False

        index = 0
        while index < len(variants):
            variant_mime, variant_bytes = variants[index]
            index += 1
            b64 = base64.b64encode(variant_bytes).decode("ascii")
            data_url = f"data:{variant_mime};base64,{b64}"

            http_result = await self._transcribe_compat(
                chosen, data_url, asr_options,
            )
            if http_result is not None:
                return http_result

            logger.error(
                "Speech recognition HTTP failed (model=%s, mime=%s)",
                chosen,
                variant_mime,
            )
            if not wav_tried:
                wav_tried = True
                wav = await _transcode_to_wav(audio_bytes)
                if wav is not None and len(wav) <= config.voice_max_bytes * 4:
                    variants.append(("audio/wav", wav))
        return None

    # -- WebSocket Realtime ASR ------------------------------------------

    @staticmethod
    def _asr_ws_url(model: str) -> str:
        """Build the WebSocket URL for Qwen-ASR-Realtime.

        ``ASR_WS_URL`` (when set) wins; otherwise the URL is derived from
        ``AI_BASE_URL``: the first host label is the workspace id and the
        region is taken from the host (``cn-beijing`` or ``ap-southeast-1``).
        """
        if config.asr_ws_url:
            base = config.asr_ws_url.rstrip("/")
            separator = "&" if "?" in base else "?"
            return f"{base}{separator}model={model}"

        host = (config.ai_base_url or "").split("://")[-1].split("/")[0]
        workspace = host.split(".")[0] if host else "dashscope"
        region = "cn-beijing" if "cn-beijing" in host else "ap-southeast-1"
        return (
            f"wss://{workspace}.{region}.maas.aliyuncs.com"
            f"/api-ws/v1/realtime?model={model}"
        )

    async def _transcribe_ws(
        self,
        model: str,
        pcm16: bytes,
    ) -> tuple[bool, str | None]:
        """Transcribe raw PCM16 audio via WebSocket (Manual mode).

        Returns ``(answered, text)``.  ``answered`` is True when the API
        delivered a transcription event (``text`` may then be ``None`` for
        silence); False means the WebSocket path failed and the caller should
        try the HTTP fallback.

        The audio is streamed in ~1-second chunks: a 5-minute note is ~9.6 MB
        of PCM16, which would exceed a single WebSocket frame limit.
        """
        url = self._asr_ws_url(model)
        headers = {"Authorization": f"Bearer {self.api_key}"}
        timeout = aiohttp.ClientTimeout(total=60, sock_read=15)

        # 16 kHz mono PCM16: one second = 32 000 bytes.
        chunk_size = 32_000

        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(
                    url, headers=headers, timeout=timeout,
                ) as ws:
                    # 1 — Configure session (Manual mode: the client decides
                    # where a phrase ends, which suits a complete voice note).
                    await ws.send_json({
                        "type": "session.update",
                        "session": {
                            "turn_detection": None,
                            "input_audio_format": "pcm16",
                            "input_audio_sample_rate": 16000,
                        },
                    })
                    await self._ws_recv(
                        ws, expect={"session.created", "session.updated"}
                    )

                    # 2 — Stream the audio as base64 PCM16 chunks.
                    for offset in range(0, len(pcm16), chunk_size):
                        chunk = pcm16[offset : offset + chunk_size]
                        await ws.send_json({
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(chunk).decode("ascii"),
                        })

                    # 3 — Commit + finish (Manual mode).
                    await ws.send_json({"type": "input_audio_buffer.commit"})
                    await ws.send_json({"type": "session.finish"})

                    # 4 — Read responses until transcription or error.
                    async for msg in ws:
                        if msg.type != aiohttp.WSMsgType.TEXT:
                            continue
                        try:
                            event = json.loads(msg.data)
                        except json.JSONDecodeError:
                            continue
                        evt_type = event.get("type", "")

                        if evt_type == "error":
                            err = event.get("error", {})
                            logger.warning(
                                "ASR WebSocket error: %s",
                                str(err.get("message", ""))[:200],
                            )
                            return False, None

                        if evt_type == "conversation.item.input_audio_transcription.completed":
                            transcript = event.get("transcript")
                            if transcript is not None:
                                transcript = str(transcript).strip()
                            if not transcript:
                                for part in event.get("content") or []:
                                    if isinstance(part, dict):
                                        value = part.get("transcript", "")
                                        if value:
                                            transcript = str(value).strip()
                                            break
                            return True, transcript or None

                        if evt_type == "session.finished":
                            break

                    return True, None  # session ended without a transcript

        except (TimeoutError, aiohttp.ClientError) as exc:
            logger.warning("ASR WebSocket failed: %s", exc)
            return False, None
        except Exception as exc:  # noqa: BLE001 - must not break a voice reply
            logger.warning("ASR WebSocket unexpected error: %s", exc)
            return False, None

    @staticmethod
    async def _ws_recv(
        ws: Any,
        expect: set[str],
        timeout_sec: float = 10.0,
    ) -> dict[str, Any] | None:
        """Receive one JSON event, skipping unexpected ones.

        Returns the first event whose ``type`` is in *expect*, or ``None``
        on timeout.
        """
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            remaining = max(0.5, deadline - time.monotonic())
            try:
                msg = await asyncio.wait_for(ws.receive(), timeout=remaining)
            except TimeoutError:
                return None
            if msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    event = json.loads(msg.data)
                except json.JSONDecodeError:
                    continue
                if event.get("type") in expect:
                    return event
            elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                return None
        return None

    async def _transcribe_compat(
        self,
        model: str,
        data_url: str,
        asr_options: dict[str, Any],
    ) -> str | None:
        """HTTP fallback: OpenAI-compatible chat/completions endpoint."""
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {"data": data_url},
                        }
                    ],
                }
            ],
            "stream": False,
            "asr_options": asr_options,
        }
        status, data, _elapsed = await self._post_chat(payload, retries=1)
        if status == 200 and data is not None:
            return self._extract_content(data)
        return None

    # --- Image generation --------------------------------------------------

    async def generate_image_url(
        self,
        prompt: str,
        *,
        model: str | None = None,
        size: str | None = None,
        reference_images: list[bytes] | None = None,
    ) -> str | None:
        """Generate an image and return its public URL (valid for ~24 hours).

        Uses the DashScope native synchronous image-synthesis endpoint. The URL
        can be passed straight to Telegram (e.g. ``InlineQueryResultPhoto``) or
        downloaded with :meth:`generate_image`.
        """
        return (await self.generate_image_result(
            prompt, model=model, size=size, download=False,
            reference_images=reference_images,
        )).url

    async def generate_image_result(
        self,
        prompt: str,
        *,
        model: str | None = None,
        size: str | None = None,
        download: bool = True,
        reference_images: list[bytes] | None = None,
    ) -> ImageResult:
        """Generate an image, optionally downloading it, with elapsed time.

        ``reference_images``, when provided, are attached to the generation
        request as visual references the model can use for style, composition
        or content guidance.
        """
        started = time.monotonic()
        url = f"{self.native_base_url}/services/aigc/multimodal-generation/generation"
        parameters: dict[str, Any] = {
            "prompt_extend": True,
            "watermark": False,
            "n": 1,
        }
        if size:
            parameters["size"] = size

        content: list[dict[str, Any]] = [{"text": prompt}]
        if reference_images:
            for ref_bytes in reference_images:
                content.append({"image": _image_data_url(ref_bytes)})

        payload: dict[str, Any] = {
            "model": model or config.image_model,
            "input": {
                "messages": [
                    {"role": "user", "content": content}
                ]
            },
            "parameters": parameters,
        }

        status, data, elapsed = await self._post_chat(
            payload,
            url=url,
            timeout=config.image_timeout,
            fail_fast_on_quota=False,
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
        reference_images: list[bytes] | None = None,
    ) -> bytes | None:
        """Generate an image from a prompt and return the raw image bytes."""
        result = await self.generate_image_result(
            prompt, model=model, size=size, reference_images=reference_images,
        )
        return result.data

    async def generate_sticker(
        self,
        prompt: str,
        *,
        model: str | None = None,
        size: str | None = None,
    ) -> ImageResult:
        """Generate a sticker-style image and return it as a 512×512 WebP.

        The user's prompt gets a sticker-style suffix, the image is generated
        and downloaded, then center-cropped to a square and re-encoded as WebP
        so it can be sent as a real Telegram sticker (``answer_sticker``).
        ``result.data`` holds the WebP bytes, or None when either generation or
        the conversion failed; ``result.url`` still points at the source image.
        """
        full_prompt = prompt.strip() + _STICKER_PROMPT_SUFFIX
        result = await self.generate_image_result(
            full_prompt, model=model, size=size, download=True,
        )
        if result.data is None:
            return result
        webp = to_sticker_webp(result.data)
        return ImageResult(url=result.url, data=webp, elapsed=result.elapsed)

    async def generate_image_with_references(
        self,
        prompt: str,
        reference_images: list[bytes],
        *,
        model: str | None = None,
        size: str | None = None,
        download: bool = True,
    ) -> ImageResult:
        """Generate an image using one or more reference images as guidance.

        The reference images are sent alongside the text prompt so the model
        can use them for style transfer, composition or content inspiration.
        """
        return await self.generate_image_result(
            prompt,
            model=model,
            size=size,
            download=download,
            reference_images=reference_images or None,
        )

    async def enhance_image_via_ai(
        self,
        image_bytes: bytes,
        *,
        operations: list[str] | None = None,
        caption: str = "",
        model: str | None = None,
        size: str | None = None,
        download: bool = True,
    ) -> ImageResult:
        """Improve a photo with the image-synthesis model (neural enhance).

        Unlike the local Pillow pipeline, the photo is re-rendered by the
        network, so lost detail can come back — at the cost of possibly
        small changes in fine details. ``operations`` (free-form tokens,
        see ``bot.services.image_tools``) steer the instruction; ``caption``
        is the user's own wording, kept as extra context. With
        ``download=False`` only the URL is returned — that is how guest
        mode hands out results it cannot upload.
        """
        parts = [ENHANCE_AI_INSTRUCTION]
        hints = [
            _ENHANCE_AI_HINTS[op]
            for op in canonical_operations(operations)
            if op in _ENHANCE_AI_HINTS
        ]
        if hints:
            parts.append("Focus on " + ", ".join(hints) + ".")
        extra = (caption or "").strip()
        if extra:
            parts.append(f"User request: {extra}")
        return await self.generate_image_result(
            "\n".join(parts),
            model=model,
            size=size,
            download=download,
            reference_images=[image_bytes],
        )

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
            except (TimeoutError, aiohttp.ClientError) as exc:
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

        Which rules depend on the delivery mode: rich messages render tables
        and LaTeX natively, the legacy path needs Unicode notation. The
        caller's list is not modified.
        """
        hint = config.style_hint if config.enable_style_hint else ""
        if hint == DEFAULT_STYLE_HINT:
            # Neither PNG math nor plain Markdown fits a rich message's
            # limitations — the model may use tables and LaTeX natively.
            if config.rich_messages:
                hint = DEFAULT_RICH_STYLE_HINT
            elif config.render_formula_images:
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
        """Rewrite scientific notation in the answer as Unicode for Telegram.

        With rich messages the raw LaTeX survives for Telegram's native
        renderer, so the Unicode rewrite only matters on the legacy path.
        """
        if (
            not content
            or config.rich_messages
            or config.render_formula_images
            or not config.normalize_formulas
        ):
            return content
        return normalize_formulas(content)

    async def _post_chat(
        self,
        payload: dict[str, Any],
        *,
        url: str | None = None,
        timeout: int | None = None,
        retries: int = 3,
        fail_fast_on_quota: bool = True,
    ) -> tuple[int | None, dict[str, Any] | None, float]:
        """POST a JSON payload with retry on transient errors.

        Returns ``(status, json, elapsed)``; ``status`` is ``None`` on network
        failure and ``elapsed`` is the wall time in seconds, retries included.
        ``fail_fast_on_quota`` returns a quota/rate-limit answer immediately so
        the chat fallback chain can switch models; set it to ``False`` for
        requests without an alternative model (e.g. image generation).
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

                    if fail_fast_on_quota and (
                        resp.status == 429 or _QUOTA_ERROR_RE.search(body)
                    ):
                        # A quota / rate-limit answer will not improve by
                        # repeating the same request: report it so the caller
                        # can switch to the next model in the chain.
                        logger.warning(
                            "AI API quota or rate limit reached (status %s); "
                            "switching models: %s",
                            resp.status,
                            body[:200],
                        )
                        return resp.status, None, _elapsed()

                    if resp.status in {500, 502, 503, 504} and attempt < retries:
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

            except TimeoutError:
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
