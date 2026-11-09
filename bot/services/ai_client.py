from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import shutil
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

import aiohttp
from defusedxml import ElementTree

from bot.config import (
    DEFAULT_RICH_STYLE_HINT,
    DEFAULT_STYLE_HINT,
    FORMULA_STYLE_HINT,
    config,
)
from bot.services import thinking
from bot.services.ai_errors import AIRequestError, provider_error, request_error
from bot.services.http_body import read_bounded
from bot.services.image_tools import canonical_operations
from bot.services.science import normalize_formulas
from bot.services.stickers import (  # noqa: F401 - совместимость публичного API
    STICKER_MAX_BYTES,
    STICKER_SIZE,
    prepare_sticker_result,
    to_sticker_webp,
)
from bot.services.tool_markup import (
    TOOL_CALLS_HINT,
    TOOL_CALLS_RETRY_HINT,
    has_json_tool_calls,
    strip_tool_markup,
    tool_stream_preview,
)

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
    ``errors`` сохраняет безопасные причины всех неудачных попыток цепочки.
    """

    def __init__(
        self, message: str, *, status: int | None = None,
        code: str = "STREAM_ERROR", errors: list[AIRequestError] | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.errors = list(errors or [])


async def _sse_lines(chunks: Any) -> AsyncIterator[str]:
    """Yield non-empty SSE lines from an async iterable of byte chunks."""
    buffer = ""
    async for raw in chunks:
        buffer += raw.decode("utf-8", "replace")
        if len(buffer) > _MAX_SSE_BUFFER:
            raise AIStreamError("SSE line is too long")
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            line = line.strip()
            if line:
                yield line
    tail = buffer.strip()
    if tail:
        yield tail


# Bounds for provider responses: a misbehaving endpoint must not be able to
# materialise an arbitrarily large JSON/error body in the bot's memory.
_MAX_PROVIDER_BODY = 8 * 1024 * 1024
_MAX_PROVIDER_ERROR_BODY = 512 * 1024
_MAX_SSE_BUFFER = 1 * 1024 * 1024


async def _read_response_text(response: Any, limit: int) -> str:
    """Read at most ``limit`` bytes of a provider response and decode them."""
    data = await read_bounded(response.content, limit)
    if len(data) > limit:
        data = data[:limit]
    encoding = getattr(response, "charset", None) or "utf-8"
    return data.decode(encoding, "replace")


_TRANSCODE_TIMEOUT = 60
_MAX_TRANSCODE_BYTES = 300 * 16_000 * 2 + 65_536  # пять минут PCM16 + WAV-заголовок
_MAX_TRANSCODE_ERRORS = 65_536

# ffmpeg is a native decoder for untrusted media: do not hand it the bot's
# secrets (BOT_TOKEN, AI_API_KEY, DATABASE_URL, …) through the environment.
_FFMPEG_ENV_ALLOWLIST = frozenset({
    "PATH", "HOME", "USER", "LOGNAME", "SHELL", "TMPDIR", "TEMP", "TMP",
    "LANG", "LANGUAGE", "LC_ALL", "LC_CTYPE", "LC_MESSAGES", "TZ",
    "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT", "OS",
    "USERPROFILE", "HOMEDRIVE", "HOMEPATH", "APPDATA", "LOCALAPPDATA",
    "PROGRAMDATA", "PROGRAMFILES", "PROGRAMFILES(X86)", "NUMBER_OF_PROCESSORS",
    "PROCESSOR_ARCHITECTURE",
})


def _ffmpeg_environment() -> dict[str, str]:
    """Environment for the ffmpeg child: system basics, never bot secrets."""
    return {
        key: value
        for key, value in os.environ.items()
        if key.upper() in _FFMPEG_ENV_ALLOWLIST
    }


async def _transcode_audio(audio: bytes, output_format: str) -> bytes | None:
    """Декодировать аудио без доступа к сети/файлам, с лимитами и уборкой процесса."""
    if len(audio) > config.voice_max_bytes:
        return None
    process = None
    tasks: list[asyncio.Task] = []

    async def read_limited(stream: asyncio.StreamReader, limit: int) -> bytes:
        result = bytearray()
        while chunk := await stream.read(65_536):
            if len(result) + len(chunk) > limit:
                raise ValueError("ffmpeg output limit exceeded")
            result.extend(chunk)
        return bytes(result)

    try:
        command = shutil.which("ffmpeg") or "ffmpeg"
        process = await asyncio.create_subprocess_exec(
            command, "-nostdin", "-v", "error",
            "-protocol_whitelist", "pipe",
            "-format_whitelist", "ogg,mp3,wav,mov,matroska,webm,aac,flac",
            "-i", "pipe:0", "-map", "0:a:0", "-vn", "-sn", "-dn",
            "-t", "300", "-ac", "1", "-ar", "16000",
            "-c:a", "pcm_s16le", "-threads", "1",
            "-fs", str(_MAX_TRANSCODE_BYTES), "-f", output_format, "pipe:1",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=_ffmpeg_environment(),
        )

        async def feed() -> None:
            try:
                process.stdin.write(audio)
                await process.stdin.drain()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                process.stdin.close()

        tasks = [
            asyncio.create_task(feed()),
            asyncio.create_task(read_limited(process.stdout, _MAX_TRANSCODE_BYTES)),
            asyncio.create_task(read_limited(process.stderr, _MAX_TRANSCODE_ERRORS)),
            asyncio.create_task(process.wait()),
        ]
        _, out, _err, returncode = await asyncio.wait_for(
            asyncio.gather(*tasks), timeout=_TRANSCODE_TIMEOUT
        )
        return out if returncode == 0 and out else None
    except (TimeoutError, OSError, ValueError) as exc:
        logger.warning("ffmpeg conversion failed (%s)", type(exc).__name__)
        return None
    finally:
        # gather не отменяет остальных участников при ошибке одного reader.
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if process is not None:
            if process.returncode is None:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
            # Дренируем оставшиеся pipe-буферы уже убитого процесса, чтобы
            # wait() не завис на заполненном stdout/stderr.
            await process.communicate()


async def _transcode_to_wav(audio: bytes) -> bytes | None:
    """Convert audio to bounded 16 kHz mono WAV; None when unavailable."""
    return await _transcode_audio(audio, "wav")


async def _transcode_to_pcm16(audio: bytes, mime: str = "audio/ogg") -> bytes | None:
    """Convert any audio to 16 kHz mono raw PCM16 (s16le) with ffmpeg.

    Returns ``None`` when ffmpeg is unavailable or transcoding fails.
    The result has no header — just interleaved 16-bit signed samples
    suitable for the WebSocket Realtime ASR API.
    """
    return await _transcode_audio(audio, "s16le")


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
    """Metadata-only view of a request payload: message **sizes**, not texts.

    Conversation content is personal data, so the default request log carries
    roles, character counts, media metadata, tool names and parameters — never
    the prompts, the history or tool results. Full texts require an explicit
    opt-in (``LOG_PROMPTS=true``) and DEBUG level, see
    :func:`_full_request_log_payload`.
    """
    summary = {key: payload[key] for key in (
        "model", "temperature", "max_tokens", "stream", "asr_options", "parameters"
    ) if key in payload}
    messages = payload.get("messages") or (payload.get("input") or {}).get("messages") or []
    logged_messages = []
    for message in messages:
        content = message.get("content")
        entry: dict[str, Any] = {"role": message.get("role")}
        if isinstance(content, str):
            entry["chars"] = len(content)
        elif isinstance(content, list):
            parts = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                if isinstance(part.get("text"), str):
                    parts.append({"type": "text", "chars": len(part["text"])})
                elif "image_url" in part:
                    ref = part["image_url"]
                    value = ref.get("url", "") if isinstance(ref, dict) else str(ref)
                    parts.append({"type": "image", **_media_log_info(value)})
                elif "input_audio" in part:
                    ref = part["input_audio"]
                    value = ref.get("data", "") if isinstance(ref, dict) else str(ref)
                    parts.append({"type": "audio", **_media_log_info(value)})
            entry["content"] = parts
        elif message.get("tool_calls"):
            entry["tool_calls"] = [
                str(((call or {}).get("function") or {}).get("name") or "")
                for call in message["tool_calls"]
            ]
        logged_messages.append(entry)
    summary["messages"] = logged_messages
    if payload.get("tools"):
        summary["tools"] = [tool.get("function", {}).get("name") for tool in payload["tools"]]
    return summary


def _full_request_log_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Complete payload including message texts — for ``LOG_PROMPTS`` only.

    Deliberately dangerous: the log becomes a plaintext archive of the
    conversation. Only used when the operator opted in and DEBUG is enabled.
    """
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
        elif message.get("tool_calls"):
            entry["tool_calls"] = [
                str(((call or {}).get("function") or {}).get("name") or "")
                for call in message["tool_calls"]
            ]
        logged_messages.append(entry)
    summary["messages"] = logged_messages
    if payload.get("tools"):
        summary["tools"] = [tool.get("function", {}).get("name") for tool in payload["tools"]]
    return summary


def _log_request_payload(request_id: str, endpoint: str, payload: dict[str, Any], *, timeout: float | None = None) -> None:
    """Log an outgoing request: metadata at INFO, full text only when opted in."""
    if config.log_prompts and logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "AI API request id=%s path=%s timeout=%ss full_payload=%s",
            request_id,
            endpoint,
            timeout if timeout is not None else "-",
            json.dumps(_full_request_log_payload(payload), ensure_ascii=False),
        )
    elif logger.isEnabledFor(logging.INFO):
        logger.info(
            "AI API request id=%s path=%s timeout=%ss payload=%s",
            request_id,
            endpoint,
            timeout if timeout is not None else "-",
            json.dumps(_request_log_payload(payload), ensure_ascii=False),
        )


# Model catalog for the /models menu. It lives in the MODEL_1_* ... MODEL_6_*
# and FALLBACK_MODEL_* sets of the .env file (see ``bot.config``): internal id,
# display name, provider URL, provider model id and the vision flag.
#
# Sets the account cannot use are hidden at startup (``GET /models`` of each
# provider).


def is_vision_model(model: str) -> bool:
    """Return True if the model with this internal id accepts images.

    The flag comes from the model's ``MODEL_<n>_VISION`` setting; an unknown id
    is treated like the default model.
    """
    return bool(config.resolve_model(model).get("vision"))


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

# Простой фон облегчает сегментацию; объект должен целиком помещаться в кадр.
_STICKER_PROMPT_SUFFIX = (
    ", die-cut sticker style, single centered subject, bold clean outlines, "
    "vibrant flat colors, thick white border around the subject, isolated on "
    "a plain solid light gray background, no background patterns, no cast shadow, "
    "entire subject visible with space around it, "
    "no text, no watermark"
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
    ``errors`` содержит безопасную диагностику только текущего вызова.
    """

    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    elapsed: float = 0.0
    usage: dict[str, Any] | None = None
    # Model that actually produced the answer (may differ from the requested
    # one when the fallback chain was used).
    model: str | None = None
    # Только ошибки этого вызова; общего last_error у клиента нет.
    errors: list[AIRequestError] = field(default_factory=list)

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
    # Безопасный внутренний код причины: сырой ответ провайдера не передаётся в чат.
    error: str | None = None


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


# Fields added to user-visible requests by bot.services.thinking. A provider
# that does not understand them answers 400/422; the request is retried once
# without them and the model is remembered for the rest of the process.
_THINKING_FIELDS = frozenset({"enable_thinking", "thinking_budget", "reasoning_effort"})


def _has_thinking(payload: dict[str, Any]) -> bool:
    return any(field in payload for field in _THINKING_FIELDS)


def _without_thinking(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key not in _THINKING_FIELDS}


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
        # Provider model IDs reported as available, per provider base URL
        # (a missing / ``None`` entry = unknown).
        self._available_ids: dict[str, set[str] | None] = {}
        # Models whose endpoint rejected thinking parameters; the rest of the
        # process sends them without thinking instead of repeating the 400.
        self._thinking_unsupported: set[str] = set()

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

    async def close(self) -> None:
        """Close the underlying HTTP session."""
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None

    # --- Model catalog -----------------------------------------------------

    async def list_models(
        self, base_url: str | None = None, api_key: str | None = None
    ) -> set[str] | None:
        """Return the set of model IDs available to this API key.

        Uses the OpenAI-compatible ``GET {base_url}/models`` endpoint of the
        given provider (default: ``AI_BASE_URL``). Returns ``None`` when the
        request fails so callers can fall back gracefully.
        """
        url = f"{(base_url or self.base_url).rstrip('/')}/models"
        logger.info("AI API request: method=GET path=/models")
        try:
            session = await self._get_session()
            async with session.get(
                url, headers=self._auth_headers(api_key), allow_redirects=False
            ) as resp:
                body = await _read_response_text(resp, _MAX_PROVIDER_BODY)
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

    def _auth_headers(self, api_key: str | None) -> dict[str, str] | None:
        """Per-request Authorization override for a model's own provider key."""
        if not api_key or api_key == self.api_key:
            return None
        return {"Authorization": f"Bearer {api_key}"}

    def _route(self, model_id: str | None) -> tuple[str, dict[str, str] | None]:
        """``(chat/completions URL, header override)`` for an internal model id."""
        spec = config.resolve_model(model_id)
        base = (spec.get("base_url") or self.base_url).rstrip("/")
        return f"{base}/chat/completions", self._auth_headers(spec.get("api_key"))

    def _error_secrets(self) -> tuple[str, ...]:
        return (
            self.api_key, config.ai_api_key, config.bot_token,
            *(spec.get("api_key", "") for spec in (*config.ai_models, *config.ai_fallback_models)),
        )

    def _stream_errors(
        self, payload: dict[str, Any], exc: AIStreamError,
    ) -> list[AIRequestError]:
        return exc.errors or [request_error(
            str(payload["model"]), str(exc.status) if exc.status else exc.code,
            str(exc), secrets=self._error_secrets(),
        )]

    async def refresh_available_models(self) -> list[dict[str, Any]]:
        """Fetch every provider's live model list and cache the usable subset."""
        providers = {
            (spec.get("base_url", ""), spec.get("api_key", ""))
            for spec in (*config.ai_models, *config.ai_fallback_models)
        }
        self._available_ids = {}
        for base_url, api_key in providers:
            self._available_ids[base_url] = await self.list_models(base_url, api_key)
        return self.available_models()

    def _provider_has(self, spec: dict[str, Any]) -> bool | None:
        """Whether the provider lists the model: True/False/``None`` (unknown)."""
        ids = self._available_ids.get(spec.get("base_url", ""))
        if not ids:
            return None
        return spec.get("model", spec["id"]) in ids

    def available_models(self) -> list[dict[str, Any]]:
        """Catalog models, filtered to those the account can actually use."""
        catalog = config.ai_models
        filtered = [m for m in catalog if self._provider_has(m) is not False]
        return filtered or list(catalog)

    def is_model_available(self, model: str) -> bool | None:
        """Availability of an internal model id: True/False, or ``None`` if unknown."""
        return self._provider_has(config.resolve_model(model))

    def model_chain(self, model: str | None, *, require_vision: bool = False) -> list[str]:
        """Ordered internal model ids to try for one request.

        Starts with the requested model (an id without settings is replaced by
        the default model), then every catalog entry *after* it, and finally
        the ``FALLBACK_MODEL_*`` model. Models the provider does not list are
        skipped. With ``require_vision`` every non-vision model is skipped,
        **including the requested one**: an image must never be sent to a model
        that cannot process it. An empty list means no vision-capable model is
        configured; the caller must not send the image at all.
        """
        primary = config.resolve_model(model)["id"]
        catalog = [spec["id"] for spec in config.ai_models]
        following = (
            catalog[catalog.index(primary) + 1 :] if primary in catalog else catalog
        )
        candidates = [primary, *following]
        candidates.extend(spec["id"] for spec in config.ai_fallback_models)

        chain: list[str] = []
        for model_id in candidates:
            if not model_id or model_id in chain:
                continue
            if require_vision and not is_vision_model(model_id):
                continue
            if model_id != primary and self._provider_has(
                config.resolve_model(model_id)
            ) is False:
                continue
            chain.append(model_id)
        if chain:
            return chain
        return [] if require_vision else [primary]

    # --- Chat completions --------------------------------------------------

    async def chat_completion(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        *,
        public: bool = False,
    ) -> str | None:
        """Send a chat completion request and return the assistant text."""
        result = await self.chat_completion_result(
            messages, model, temperature, max_tokens, public=public
        )
        return result.content

    async def chat_completion_result(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        *,
        require_vision: bool = False,
        public: bool = False,
    ) -> ChatResult:
        """Chat completion returning the text plus timing and token usage.

        When a model does not answer (API error, quota exhausted, empty
        reply), the next model in :meth:`model_chain` is tried automatically.
        ``require_vision`` is for multimodal messages: it keeps the request
        away from models whose ``VISION`` flag is off. ``public`` marks a
        request whose answer is shown to a user: it receives the global
        thinking settings (see :mod:`bot.services.thinking`).
        """
        started = time.monotonic()
        errors: list[AIRequestError] = []
        for attempt_model in self.model_chain(model, require_vision=require_vision):
            payload = self._chat_payload(
                messages,
                attempt_model,
                temperature,
                max_tokens,
                public=public and attempt_model not in self._thinking_unsupported,
            )
            status, data, _elapsed = await self._post_chat(
                payload, retries=2, model_id=attempt_model, errors=errors
            )
            if status in {400, 422} and _has_thinking(payload):
                self._thinking_unsupported.add(attempt_model)
                logger.warning(
                    "Model %s rejected thinking parameters; retrying without them",
                    attempt_model,
                )
                payload = _without_thinking(payload)
                status, data, _elapsed = await self._post_chat(
                    payload, retries=2, model_id=attempt_model, errors=errors
                )
            if status == 200 and data is not None:
                content = self._finalize(self._extract_content(data))
                if content:
                    return ChatResult(
                        content=content,
                        elapsed=time.monotonic() - started,
                        usage=_extract_usage(data),
                        model=attempt_model,
                        errors=errors,
                    )
                errors.append(request_error(
                    str(payload["model"]), "EMPTY_RESPONSE", "HTTP 200: no usable answer",
                    secrets=self._error_secrets(),
                ))
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
        return ChatResult(elapsed=time.monotonic() - started, errors=errors)

    async def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        public: bool = False,
    ) -> ChatResult:
        """Chat completion with function calling enabled.

        Falls back to a plain completion if the model/region rejects tools,
        then moves on to the next model in the chain. ``public`` applies the
        global thinking settings (user-visible answers only).
        """
        started = time.monotonic()
        errors: list[AIRequestError] = []
        for attempt_model in self.model_chain(model):
            offered = tools if tools is not None else [IMAGE_TOOL]
            payload = self._chat_payload(
                messages,
                attempt_model,
                temperature,
                max_tokens,
                public=public and attempt_model not in self._thinking_unsupported,
                tools=offered,
            )
            payload["tools"] = offered

            status, data, _elapsed = await self._post_chat(
                payload, retries=2, model_id=attempt_model, errors=errors
            )
            # Some models do not support the tools parameter -> retry the
            # same model without it, but only for a tool-shaped rejection.
            # Tools go first so a tools-only rejection does not needlessly
            # disable thinking for this model.
            if status in {400, 404, 422}:
                payload.pop("tools", None)
                payload["messages"] = self._without_hint(
                    payload["messages"], TOOL_CALLS_HINT
                )
                status, data, _elapsed = await self._post_chat(
                    payload, retries=2, model_id=attempt_model, errors=errors
                )
            if status in {400, 422} and _has_thinking(payload):
                self._thinking_unsupported.add(attempt_model)
                logger.warning(
                    "Model %s rejected thinking parameters; retrying without them",
                    attempt_model,
                )
                payload = _without_thinking(payload)
                status, data, _elapsed = await self._post_chat(
                    payload, retries=2, model_id=attempt_model, errors=errors
                )

            if status == 200 and data is not None:
                result = self._parse_result(data, finalize=False)
                self._recover_search_dsml(result, payload.get("tools") or [])
                retry_payload = self._text_tool_retry_payload(payload, result)
                if retry_payload is not None:
                    errors.append(request_error(
                        str(payload["model"]), "TEXT_TOOL_CALL", "Text instead of structured tool_calls",
                        secrets=self._error_secrets(),
                    ))
                    payload = retry_payload
                    status, data, _elapsed = await self._post_chat(
                        payload, retries=1, model_id=attempt_model, errors=errors
                    )
                    result = (
                        self._parse_result(data, finalize=False)
                        if status == 200 and data is not None else ChatResult()
                    )
                    self._recover_search_dsml(result, payload.get("tools") or [])
                result.content = self._finalize(
                    result.content, tools_offered=bool(payload.get("tools"))
                )
                if result.content or result.tool_calls:
                    result.elapsed = time.monotonic() - started
                    result.model = attempt_model
                    result.errors = errors
                    return result
                if status == 200 and data is not None:
                    errors.append(request_error(
                        str(payload["model"]), "EMPTY_RESPONSE", "HTTP 200: no usable answer",
                        secrets=self._error_secrets(),
                    ))

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
        return ChatResult(elapsed=time.monotonic() - started, errors=errors)

    async def chat_completion_with_image(
        self,
        text: str,
        image_bytes: bytes,
        messages: list[dict[str, Any]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        *,
        public: bool = False,
    ) -> str | None:
        """Send a chat completion that includes a single image."""
        result = await self.chat_with_image_result(
            text, image_bytes, messages, model, temperature, max_tokens, public=public
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
        *,
        public: bool = False,
    ) -> ChatResult:
        """Vision completion returning the text plus timing and token usage.

        Only vision-capable models are tried, so a text-only fallback never
        receives an image request. ``public`` applies the global thinking
        settings when the analysis itself is shown to the user.
        """
        started = time.monotonic()
        errors: list[AIRequestError] = []
        for attempt_model in self.model_chain(model, require_vision=True):
            payload = self._chat_payload(
                self.vision_messages(messages, text, image_bytes),
                attempt_model,
                temperature,
                max_tokens,
                public=public and attempt_model not in self._thinking_unsupported,
            )
            status, data, _elapsed = await self._post_chat(
                payload, retries=2, model_id=attempt_model, errors=errors
            )
            if status in {400, 422} and _has_thinking(payload):
                self._thinking_unsupported.add(attempt_model)
                logger.warning(
                    "Vision model %s rejected thinking parameters; "
                    "retrying without them",
                    attempt_model,
                )
                payload = _without_thinking(payload)
                status, data, _elapsed = await self._post_chat(
                    payload, retries=2, model_id=attempt_model, errors=errors
                )
            if status == 200 and data is not None:
                content = self._finalize(self._extract_content(data))
                if content:
                    return ChatResult(
                        content=content,
                        elapsed=time.monotonic() - started,
                        usage=_extract_usage(data),
                        model=attempt_model,
                        errors=errors,
                    )
                errors.append(request_error(
                    str(payload["model"]), "EMPTY_RESPONSE", "HTTP 200: no usable answer",
                    secrets=self._error_secrets(),
                ))
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
        return ChatResult(elapsed=time.monotonic() - started, errors=errors)

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
        require_vision: bool = False,
        public: bool = False,
    ) -> ChatResult:
        """Streaming chat completion over SSE.

        ``on_delta`` is awaited with the text accumulated so far after every
        chunk so the caller can show the answer progressively. A model that
        fails is skipped in favour of the next one from :meth:`model_chain`;
        when every model failed, :class:`AIStreamError` is raised and the
        caller should fall back to a regular single-shot request.
        ``require_vision`` keeps multimodal messages away from text-only
        models; it is passed straight to :meth:`model_chain`. ``public``
        applies the global thinking settings to user-visible answers.
        """
        started = time.monotonic()
        last_error: AIStreamError | None = None
        errors: list[AIRequestError] = []
        for attempt_model in self.model_chain(model, require_vision=require_vision):
            payload = self._chat_payload(
                messages,
                attempt_model,
                temperature,
                max_tokens,
                public=public and attempt_model not in self._thinking_unsupported,
                tools=tools,
            )
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}
            if tools:
                payload["tools"] = list(tools)
            try:
                result = await self._stream_attempt_with_thinking_fallback(
                    payload,
                    on_delta,
                    attempt_model,
                    allow_retry=not tools,
                )
                self._recover_search_dsml(result, payload.get("tools") or [])
                retry_payload = self._text_tool_retry_payload(payload, result)
                if retry_payload is not None:
                    errors.extend(result.errors)
                    errors.append(request_error(
                        str(payload["model"]), "TEXT_TOOL_CALL", "Text instead of structured tool_calls",
                        secrets=self._error_secrets(),
                    ))
                    payload = retry_payload
                    result = await self._stream_attempt_with_thinking_fallback(
                        payload, on_delta, attempt_model, allow_retry=False,
                    )
                    self._recover_search_dsml(result, payload.get("tools") or [])
                result.content = self._finalize(
                    result.content, tools_offered=bool(tools)
                )
            except AIStreamError as exc:
                errors.extend(self._stream_errors(payload, exc))
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
                errors.extend(result.errors)
                last_error = AIStreamError("HTTP 200: no usable answer", code="EMPTY_RESPONSE")
                errors.extend(self._stream_errors(payload, last_error))
                logger.warning(
                    "Model %s returned an empty streamed answer; "
                    "trying the next model",
                    attempt_model,
                )
                continue
            result.elapsed = time.monotonic() - started
            result.model = attempt_model
            result.errors = [*errors, *result.errors]
            return result

        last_error = last_error or AIStreamError("no model produced an answer")
        last_error.errors = errors
        raise last_error

    async def _stream_attempt_with_thinking_fallback(
        self,
        payload: dict[str, Any],
        on_delta: Any,
        model_id: str,
        *,
        allow_retry: bool,
    ) -> ChatResult:
        """One streaming request, retried without thinking fields on 400/422.

        ``allow_retry`` is False when tools are attached: a tool-shaped
        rejection must not be mistaken for an unsupported thinking parameter
        (the caller's single-shot fallback handles that case).
        """
        try:
            return await self._stream_attempt(payload, on_delta, model_id)
        except AIStreamError as exc:
            if (
                not allow_retry
                or exc.status not in {400, 422}
                or not _has_thinking(payload)
            ):
                raise
            self._thinking_unsupported.add(model_id)
            logger.warning(
                "Model %s rejected thinking parameters in a stream; "
                "retrying without them",
                model_id,
            )
            errors = self._stream_errors(payload, exc)
            try:
                result = await self._stream_attempt(
                    _without_thinking(payload), on_delta, model_id
                )
            except AIStreamError as retry_error:
                retry_error.errors = [*errors, *self._stream_errors(payload, retry_error)]
                raise
            result.errors = [*errors, *result.errors]
            return result

    async def _stream_attempt(
        self,
        payload: dict[str, Any],
        on_delta: Any = None,
        model_id: str | None = None,
    ) -> ChatResult:
        """One streaming request; raises :class:`AIStreamError` on failure."""
        request_id = uuid4().hex[:12]
        started = time.monotonic()
        _log_request_payload(request_id, "/chat/completions", payload)

        parts: list[str] = []
        tool_acc: dict[int, dict[str, str]] = {}
        usage: dict[str, Any] | None = None
        saw_done = False
        malformed = 0

        async for line in self._stream_lines(payload, model_id):
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
                raise AIStreamError("API stream error", errors=[provider_error(
                    str(payload["model"]), 200, data, secrets=self._error_secrets(),
                )])
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
                            preview = tool_stream_preview(
                                "".join(parts),
                                tools_offered=bool(payload.get("tools")),
                            )
                            if preview:
                                await on_delta(preview)
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
            content=content,
            tool_calls=tool_calls,
            elapsed=time.monotonic() - started,
            usage=usage,
        )

    async def _stream_lines(
        self, payload: dict[str, Any], model_id: str | None = None
    ) -> AsyncIterator[str]:
        """POST the streaming request and yield its raw SSE lines."""
        target, headers = self._route(model_id)
        timeout = aiohttp.ClientTimeout(
            total=config.request_timeout,
            sock_read=max(10, config.stream_stall_timeout),
        )
        try:
            session = await self._get_session()
            async with session.post(
                target,
                json=payload,
                timeout=timeout,
                headers=headers,
                allow_redirects=False,
            ) as resp:
                if resp.status != 200:
                    body = await _read_response_text(
                        resp, _MAX_PROVIDER_ERROR_BODY
                    )
                    logger.error(
                        "AI stream error (status %s): %s", resp.status, body[:500]
                    )
                    raise AIStreamError(f"HTTP {resp.status}", status=resp.status, errors=[
                        provider_error(str(payload["model"]), resp.status, body, secrets=self._error_secrets()),
                    ])
                async for line in _sse_lines(resp.content):
                    yield line
        except AIStreamError:
            raise
        except TimeoutError as exc:
            raise AIStreamError("Request timed out", code="TIMEOUT") from exc
        except aiohttp.ClientError as exc:
            raise AIStreamError("Network request failed", code="NETWORK_ERROR") from exc

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
            async with (
                aiohttp.ClientSession() as session,
                session.ws_connect(
                    url, headers=headers, timeout=timeout,
                ) as ws,
            ):
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

        status, data, _elapsed = await self._post_chat(
            payload,
            url=url,
            timeout=config.image_timeout,
            fail_fast_on_quota=False,
        )
        if status != 200 or not data or not isinstance(data, dict):
            logger.error("Image generation failed (status %s)", status)
            error = {
                None: "generation_connection",
                200: "generation_invalid_response",
                400: "generation_rejected",
                401: "generation_auth",
                403: "generation_forbidden",
                404: "generation_not_found",
                429: "generation_rate_limit",
            }.get(status, "generation_unavailable" if status is not None and 500 <= status < 600 else "generation_failed")
            return ImageResult(elapsed=time.monotonic() - started, error=error)

        try:
            image_url = self._extract_image_url(data)
        except (AttributeError, TypeError, IndexError):
            logger.exception("Malformed image generation response")
            return ImageResult(elapsed=time.monotonic() - started, error="generation_invalid_response")
        if image_url is not None and not isinstance(image_url, str):
            logger.error("Image response contains a non-string URL")
            return ImageResult(elapsed=time.monotonic() - started, error="generation_invalid_response")
        if not image_url:
            logger.error("Image response has no URL: %s", str(data)[:300])
            return ImageResult(elapsed=time.monotonic() - started, error="generation_no_image")

        image_bytes = await self.download_image(image_url) if download else None
        return ImageResult(
            url=image_url,
            data=image_bytes,
            elapsed=time.monotonic() - started,
            error="image_download_failed" if download and image_bytes is None else None,
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
        """Сгенерировать изображение, удалить фон и вписать в прозрачный WebP 512×512."""
        full_prompt = prompt.strip() + _STICKER_PROMPT_SUFFIX
        result = await self.generate_image_result(
            full_prompt, model=model, size=size, download=True,
        )
        if result.data is None:
            if result.error is None:
                result.error = "image_download_failed" if result.url else "generation_failed"
            return result
        started = time.monotonic()
        processed = await prepare_sticker_result(result.data)
        elapsed = (result.elapsed or 0) + time.monotonic() - started
        return ImageResult(url=result.url, data=processed.data, elapsed=elapsed, error=processed.error)

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
        Every hop is DNS-pinned to public IPs on web ports; the body is read
        incrementally under a byte limit, even without Content-Length.
        """
        if not url:
            return None

        from bot.services.documents import _fetch_body

        # Never forward the API key to an external image/storage host.
        try:
            source = urlsplit(url)
        except ValueError:
            return None
        trusted_origins = {
            (parsed.scheme, parsed.netloc)
            for parsed in (urlsplit(self.base_url), urlsplit(self.native_base_url))
        }
        header_variants = [None]
        if (source.scheme, source.netloc) in trusted_origins:
            header_variants.append(self._headers)
        for headers in header_variants:
            body, _final_url, _content_type = await _fetch_body(
                url, max_bytes=MAX_DOWNLOAD_BYTES, headers=headers,
                text_only=False, timeout_seconds=config.image_timeout,
            )
            if body:
                return body
        logger.error("Image download failed")
        return None

    # --- Internals ---------------------------------------------------------

    @staticmethod
    def _text_tool_retry_payload(
        payload: dict[str, Any], result: ChatResult,
    ) -> dict[str, Any] | None:
        """Один повтор JSON-имитации через штатный протокол, без исполнения текста."""
        offered = {
            str((tool.get("function") or {}).get("name", ""))
            for tool in payload.get("tools") or []
        }
        if not offered or result.tool_calls or not has_json_tool_calls(result.content, allowed_names=offered):
            return None
        logger.warning("Model returned JSON tool calls in content; retrying with native tools")
        # Не добавляем имитацию в messages: её аргументы не доверены, а даты
        # могут быть выдуманы. Исходные messages и tools остаются неизменными.
        retry = {
            **payload,
            "messages": AIClient._with_hint(payload["messages"], TOOL_CALLS_RETRY_HINT),
        }
        if "search_web" in offered and has_json_tool_calls(result.content, allowed_names={"search_web"}):
            # Попросить провайдера именно о штатном поисковом вызове. Запрос
            # модель строит заново — аргументы текстового JSON не исполняются.
            retry["tool_choice"] = {"type": "function", "function": {"name": "search_web"}}
        return retry

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
        *,
        public: bool = False,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        rendered = AIClient._with_style_hint(messages)
        if tools:
            # Модель должна вызывать инструменты штатно, а не «рисовать»
            # вызов текстом в ответе (см. bot/services/tool_markup.py).
            rendered = AIClient._with_hint(rendered, TOOL_CALLS_HINT)
        payload = {
            "model": (spec := config.resolve_model(model)).get("model", spec["id"]),
            "messages": rendered,
            "temperature": (
                config.default_temperature if temperature is None else temperature
            ),
            "max_tokens": (
                config.default_max_tokens if max_tokens is None else max_tokens
            ),
        }
        if public:
            payload.update(thinking.payload_fields())
        return payload

    @staticmethod
    def _with_hint(
        messages: list[dict[str, Any]], hint: str
    ) -> list[dict[str, Any]]:
        """Append ``hint`` to the system prompt of a copy of ``messages``."""
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
    def _without_hint(
        messages: list[dict[str, Any]], hint: str
    ) -> list[dict[str, Any]]:
        """Remove ``hint`` from a copy of ``messages`` (tools were rejected)."""
        result = [dict(entry) for entry in messages]
        for entry in result:
            if entry.get("role") != "system":
                continue
            content = entry.get("content")
            if isinstance(content, str) and hint in content:
                entry["content"] = content.replace(
                    f"\n\n{hint}", ""
                ).replace(hint, "").strip()
        return result

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
        return AIClient._with_hint(messages, hint)

    @staticmethod
    def _finalize(
        content: str | None, *, tools_offered: bool = False,
    ) -> str | None:
        """Clean the answer and normalize scientific notation for Telegram.

        Text-form tool calls (search_notes, DSML, …) are removed first: the
        bot executes only structured tool calls, so such markup must never
        reach the user. ``tools_offered`` включает строгое удаление всего
        ответа при JSON-имитации вызова: это нарушение протокола только когда
        модель должна была вызвать инструмент штатно. В обычном чате, vision,
        OCR и прочих путях без инструментов удаляется лишь сам JSON.
        With rich messages the raw LaTeX survives for Telegram's native
        renderer, so the Unicode rewrite only matters on the legacy path.
        """
        content = strip_tool_markup(content, drop_json_calls=tools_offered)
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
        model_id: str | None = None,
        errors: list[AIRequestError] | None = None,
    ) -> tuple[int | None, dict[str, Any] | None, float]:
        """POST a JSON payload with retry on transient errors.

        Returns ``(status, json, elapsed)``; ``status`` is ``None`` on network
        failure and ``elapsed`` is the wall time in seconds, retries included.
        ``fail_fast_on_quota`` returns a quota/rate-limit answer immediately so
        the chat fallback chain can switch models; set it to ``False`` for
        requests without an alternative model (e.g. image generation).
        ``errors``, если передан, пополняется причиной каждой неудачной попытки.
        """
        if url:
            target, headers = url, None
        elif model_id:
            target, headers = self._route(model_id)
        else:
            target, headers = f"{self.base_url}/chat/completions", None
        request_timeout = aiohttp.ClientTimeout(total=timeout or config.request_timeout)
        started = time.monotonic()
        request_id = uuid4().hex[:12]
        endpoint = urlsplit(target).path
        _log_request_payload(
            request_id, endpoint, payload, timeout=request_timeout.total
        )

        def _elapsed() -> float:
            return time.monotonic() - started

        for attempt in range(1, retries + 1):
            try:
                logger.info("AI API send id=%s attempt=%s/%s", request_id, attempt, retries)
                session = await self._get_session()
                async with session.post(
                    target,
                    json=payload,
                    timeout=request_timeout,
                    headers=headers,
                    allow_redirects=False,
                ) as resp:
                    body = await _read_response_text(
                        resp,
                        _MAX_PROVIDER_BODY
                        if resp.status == 200
                        else _MAX_PROVIDER_ERROR_BODY,
                    )
                    logger.info(
                        "AI API response id=%s attempt=%s status=%s elapsed=%.2fs",
                        request_id, attempt, resp.status, _elapsed(),
                    )

                    if resp.status == 200:
                        try:
                            data = json.loads(body)
                        except json.JSONDecodeError:
                            logger.error("AI API returned invalid JSON: %s", body[:300])
                            if errors is not None:
                                errors.append(request_error(
                                    str(payload["model"]), "INVALID_RESPONSE", "HTTP 200: invalid JSON",
                                    secrets=self._error_secrets(),
                                ))
                            return resp.status, None, _elapsed()
                        if not isinstance(data, dict) or data.get("error"):
                            if errors is not None:
                                errors.append(
                                    provider_error(str(payload["model"]), 200, body, secrets=self._error_secrets())
                                    if isinstance(data, dict) else request_error(
                                        str(payload["model"]), "INVALID_RESPONSE", "HTTP 200: expected a JSON object",
                                        secrets=self._error_secrets(),
                                    )
                                )
                            return resp.status, None, _elapsed()
                        return 200, data, _elapsed()

                    if errors is not None:
                        errors.append(provider_error(
                            str(payload["model"]), resp.status, body, secrets=self._error_secrets(),
                        ))

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
                if errors is not None:
                    errors.append(request_error(
                        str(payload["model"]), "TIMEOUT", "Request timed out",
                        secrets=self._error_secrets(),
                    ))
                logger.warning(
                    "AI API request timed out id=%s (attempt %s/%s)",
                    request_id, attempt, retries,
                )
                if attempt >= retries:
                    return None, None, _elapsed()
                await asyncio.sleep(min(2**attempt, 8))
            except aiohttp.ClientError as exc:
                if errors is not None:
                    errors.append(request_error(
                        str(payload["model"]), "NETWORK_ERROR", "Network request failed",
                        secrets=self._error_secrets(),
                    ))
                logger.error("AI API request failed id=%s: %s", request_id, exc)
                if attempt >= retries:
                    return None, None, _elapsed()
                await asyncio.sleep(min(2**attempt, 8))

        return None, None, _elapsed()

    @staticmethod
    def _recover_search_dsml(result: ChatResult, tools: list[dict[str, Any]]) -> None:
        """Восстановить текстовый DSML-поиск только при явно предложенном search_web.

        Принимается только целое сообщение с вызовами поиска, не цитата внутри
        ответа. Другие инструменты не исполняются; лимиты остаются в tool-loop.
        Повреждённый DSML не является ответом: вызывающий код попробует fallback.
        """
        content = (result.content or "").strip()
        if not content.startswith("<｜DSML｜"):
            return
        result.content = None
        if result.tool_calls:
            return
        allowed = any(tool.get("function", {}).get("name") == "search_web" for tool in tools)
        if not allowed or len(content) > 65_536 or "<!" in content:
            logger.warning("Ignoring DSML response: search unavailable or invalid envelope")
            return
        xml = re.sub(r"<(/?)｜DSML｜\s*(\w+)", r"<\1\2", content)
        try:
            root = ElementTree.fromstring(xml)
            if root.tag != "calls" or root.attrib or (root.text or "").strip() or not len(root):
                raise ValueError("invalid calls envelope")
            calls: list[ToolCall] = []
            for invoke in root:
                if (
                    invoke.tag != "invoke" or invoke.attrib != {"name": "search_web"}
                    or (invoke.text or "").strip() or (invoke.tail or "").strip()
                ):
                    raise ValueError("unexpected tool")
                arguments: dict[str, Any] = {}
                for parameter in invoke:
                    name = parameter.get("name")
                    if (
                        parameter.tag != "parameter" or len(parameter)
                        or (parameter.tail or "").strip()
                        or set(parameter.attrib) != {"name", "string"}
                        or name not in {"query", "max_results"} or name in arguments
                        or parameter.get("string") not in {"true", "false"}
                    ):
                        raise ValueError("invalid parameter")
                    value = parameter.text or ""
                    arguments[name] = value if parameter.get("string") == "true" else json.loads(value)
                if not isinstance(arguments.get("query"), str) or not arguments["query"].strip():
                    raise ValueError("missing query")
                if "max_results" in arguments and type(arguments["max_results"]) is not int:
                    raise ValueError("invalid max_results")
                calls.append(ToolCall(id=f"call_{uuid4().hex}", name="search_web", arguments=arguments))
        except (ElementTree.ParseError, ValueError):
            logger.warning("Ignoring malformed or unsupported DSML tool response")
            return
        result.tool_calls = calls
        logger.info("Recovered %s DSML search calls from model content", len(calls))

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
    def _parse_result(
        cls, data: dict[str, Any], elapsed: float = 0.0, *, finalize: bool = True
    ) -> ChatResult:
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

        content = cls._normalize_content(message.get("content"))
        if finalize:
            content = cls._finalize(content)
        return ChatResult(
            content=content,
            tool_calls=calls,
            elapsed=elapsed,
            usage=_extract_usage(data),
        )


# Shared singleton used across handlers.
ai_client = AIClient()
