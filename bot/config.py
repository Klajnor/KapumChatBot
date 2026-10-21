from __future__ import annotations

import logging
import os
import math
import re
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv

load_dotenv()


def _env(*names: str, default: str = "") -> str:
    """Return the first non-empty environment variable among ``names``."""
    for name in names:
        value = os.getenv(name)
        if value:
            return value.strip()
    return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw.strip())
        return value if math.isfinite(value) else default
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "y", "да"}


# Default catalog for the /models menu. Override it in .env without touching
# the code (see AI_MODELS below).
DEFAULT_AI_MODELS: list[dict[str, Any]] = [
    {"id": "qwen3.8-max", "name": "Qwen 3.8 Max", "vision": True},
    {"id": "qwen3.8-flash", "name": "Qwen 3.8 Flash", "vision": True},
    {"id": "qwen3.8-omni-flash", "name": "Qwen 3.8 Omni Flash", "vision": True},
    {"id": "qwen3.5-ocr", "name": "Qwen 3.5 OCR", "vision": True},
    {"id": "deepseek-v4-pro-0813", "name": "DeepSeek V4 Pro", "vision": False},
    {"id": "deepseek-v4.1-flash-1017", "name": "DeepSeek V4.1 Flash", "vision": True},
]

_VISION_FLAGS = {"vision", "image", "multimodal", "yes", "true", "1"}
_NO_VISION_FLAGS = {"novision", "no-vision", "text", "no", "false", "0"}

# Notice shown while a slow AI request is still running (see PROGRESS_TEXT).
# Placeholders: {spinner} — animated frame, {seconds} — elapsed whole seconds.
DEFAULT_PROGRESS_TEXT = "{spinner} Waiting for the AI response… {seconds} s"

# Footer appended to an answer with the AI API processing time.
# Placeholders: {seconds}, {model}, {tokens}.
DEFAULT_TIMING_FORMAT = "⏱️ {seconds} с · {model}"

# Sent instead of answers Telegram cannot render (Markdown tables, raw LaTeX).
# Placeholder: {url}.
DEFAULT_TELEGRAPH_NOTICE = (
    "📄 This answer contains tables or formulas Telegram cannot display — "
    "the full version is published on Telegraph: {url}"
)

# Appended to the system prompt so the model answers in a Telegram-friendly way
# (see ENABLE_STYLE_HINT / STYLE_HINT).
DEFAULT_STYLE_HINT = (
    "Your answer is displayed in Telegram, which has no LaTeX renderer and no "
    "subscript/superscript formatting. Write formulas with Unicode characters: "
    "H₂SO₄, CO₂, Ca(OH)₂, m/s², E = mc², 6.022×10²³, 25 °C, Ω, α, β, ΔG, ×, "
    "≈, ≤, ≥, ±, √2, →. Never output LaTeX ($...$, \\frac{}{}, \\alpha, ^{}, "
    "_{}) or mhchem \\ce{}. Format with Telegram Markdown: **bold**, *italic*, "
    "`code`, ```fenced blocks```, - lists. Do not use tables — use short lines "
    "or lists instead."
)

FORMULA_STYLE_HINT = (
    "Your answer is displayed in Telegram. Use Markdown for text and Unicode "
    "for short inline formulas and chemical formulas (H₂SO₄, CO₂, x², 25 °C). "
    "Write EVERY calculation and equation as a LaTeX display block delimited by "
    "\\[ and \\], each on its own line, so it is rendered as an image: for "
    "example \\[ n = \\frac{m}{M} = \\frac{18.0}{208} = 0.0865\\ \\text{mol} \\]. "
    "Never put calculations inline in a sentence, and never use $...$ or $$...$$ "
    "delimiters. Use a decimal point or comma as a plain character (never "
    "{,}). Use standard LaTeX only: \\frac, \\sqrt, \\sum, \\int, \\cdot, \\times, "
    "\\approx, Greek letters and \\text{...} for words and units. For several "
    "steps use one block per step, or one block with lines separated by \\\\ "
    "(do not use matrices, tables or \\boxed). Do not use Markdown tables with "
    "LaTeX inside; put plain text or Unicode formulas in table cells. Keep code "
    "inside code fences."
)


def _parse_models(raw: str) -> list[dict[str, Any]]:
    """Parse the ``AI_MODELS`` environment value into a model catalog.

    Entries are separated by commas and/or newlines and have the form::

        model-id
        model-id:Display Name
        model-id:Display Name:vision      # image-capable
        model-id:Display Name:novision    # text-only

    Without the third field, vision support is detected automatically from the
    model id (Qwen-VL / Omni / OCR and the Qwen3.5+ families).
    """
    models: list[dict[str, Any]] = []
    for entry in re.split(r"[\n,]+", raw or ""):
        entry = entry.strip()
        if not entry or entry.startswith("#"):
            continue
        parts = [part.strip() for part in entry.split(":")]
        model_id = parts[0]
        if not model_id:
            continue
        name = parts[1] if len(parts) > 1 and parts[1] else model_id
        vision: bool | None = None
        if len(parts) > 2 and parts[2]:
            flag = parts[2].lower()
            if flag in _VISION_FLAGS:
                vision = True
            elif flag in _NO_VISION_FLAGS:
                vision = False
        models.append({"id": model_id, "name": name, "vision": vision})
    return models


@dataclass
class Config:
    """Application configuration loaded from environment variables."""

    # --- Telegram ---
    bot_token: str = field(default_factory=lambda: os.getenv("BOT_TOKEN", "").strip())
    # Optional: bot username without "@". Resolved automatically at startup when omitted.
    bot_username: str = field(
        default_factory=lambda: os.getenv("BOT_USERNAME", "").strip().lstrip("@")
    )

    # --- AI provider (Qwen / any OpenAI-compatible endpoint) ---
    # "Qwen Token Plan" is served through Alibaba Cloud Model Studio over an
    # OpenAI-compatible API. Only the API key, base URL and model name differ
    # from OpenRouter, so the client stays provider agnostic.
    ai_api_key: str = field(
        default_factory=lambda: _env(
            "AI_API_KEY",
            "QWEN_API_KEY",
            "DASHSCOPE_API_KEY",
        )
    )
    ai_base_url: str = field(
        default_factory=lambda: _env(
            "AI_BASE_URL",
            "QWEN_BASE_URL",
            default="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        )
    )
    # Model catalog for the /models menu, parsed from AI_MODELS (see .env).
    # Falls back to DEFAULT_AI_MODELS when the variable is absent or empty.
    ai_models: list[dict[str, Any]] = field(default_factory=list)

    # --- Database ---
    # SQLite (aiosqlite) by default — a single file, no server to maintain.
    # A PostgreSQL URL keeps working: postgresql+asyncpg://user:pass@host/db
    database_url: str = field(
        default_factory=lambda: os.getenv(
            "DATABASE_URL", "sqlite+aiosqlite:///chat_bot.db"
        ).strip()
    )

    # --- Defaults ---
    default_model: str = field(
        default_factory=lambda: os.getenv("DEFAULT_MODEL", "qwen3.8-flash").strip()
    )
    default_system_prompt: str = field(
        default_factory=lambda: os.getenv(
            "DEFAULT_SYSTEM_PROMPT",
            "You are a helpful AI assistant. Answer questions concisely and accurately.",
        )
    )
    default_temperature: float = field(
        default_factory=lambda: _env_float("DEFAULT_TEMPERATURE", 0.7)
    )
    default_max_tokens: int = field(
        default_factory=lambda: _env_int("DEFAULT_MAX_TOKENS", 2048)
    )
    request_timeout: int = field(
        default_factory=lambda: _env_int("REQUEST_TIMEOUT", 300)
    )
    # Reply formatting for AI answers: "markdown" (MarkdownV2), "html", "plain".
    parse_mode: str = field(
        default_factory=lambda: os.getenv("PARSE_MODE", "markdown").strip().lower()
    )

    # --- Speech recognition (voice messages) ---
    # OpenAI-compatible ASR model (Qwen3-ASR-Flash accepts OGG/Opus directly).
    asr_model: str = field(
        default_factory=lambda: os.getenv("ASR_MODEL", "qwen3-asr-flash").strip()
    )

    # --- Image generation ---
    # DashScope native image-synthesis model and (optional) explicit endpoint.
    image_model: str = field(
        default_factory=lambda: os.getenv(
            "IMAGE_MODEL", "qwen-image-3.0-pro"
        ).strip()
    )
    image_api_url: str = field(
        default_factory=lambda: os.getenv("IMAGE_API_URL", "").strip()
    )
    image_timeout: int = field(
        default_factory=lambda: _env_int("IMAGE_TIMEOUT", 300)
    )

    # --- Admin ---
    admin_ids: list[int] = field(default_factory=list)

    # --- Limits ---
    max_history_length: int = field(
        default_factory=lambda: _env_int("MAX_HISTORY_LENGTH", 30)
    )
    # Daily question limit for whitelisted (non-admin) users.
    daily_limit: int = field(default_factory=lambda: _env_int("DAILY_LIMIT", 10))
    # Lifetime free-trial messages for anyone who is not whitelisted.
    guest_message_limit: int = field(
        default_factory=lambda: _env_int("GUEST_MESSAGE_LIMIT", 5)
    )
    # Feature flags.
    allow_guest_access: bool = field(
        default_factory=lambda: _env_bool("ALLOW_GUEST_ACCESS", True)
    )
    # Guest mode (Bot API 10.0): the bot can be summoned with @username in any
    # chat or group without being a member, and replies via answerGuestQuery.
    enable_guest_mode: bool = field(
        default_factory=lambda: _env_bool("ENABLE_GUEST_MODE", True)
    )
    # A guest query allows exactly one reply, so bound the work before answering.
    guest_timeout: int = field(
        default_factory=lambda: _env_int("GUEST_TIMEOUT", 300)
    )
    # Recognize voice messages and answer with text or a generated image.
    enable_voice: bool = field(
        default_factory=lambda: _env_bool("ENABLE_VOICE", True)
    )
    # Let the AI answer with a generated image when the user asks for one.
    enable_image_generation: bool = field(
        default_factory=lambda: _env_bool("ENABLE_IMAGE_GENERATION", True)
    )
    # Maximum voice-message size accepted for transcription (bytes).
    voice_max_bytes: int = field(
        default_factory=lambda: _env_int("VOICE_MAX_BYTES", 7_500_000)
    )

    # Simultaneous AI requests allowed per user (protects the paid API).
    max_concurrent_per_user: int = field(
        default_factory=lambda: _env_int("MAX_CONCURRENT_PER_USER", 2)
    )
    # Telegraph pages are public: by default the page title is generic instead
    # of the user's question.
    telegraph_title_from_question: bool = field(
        default_factory=lambda: _env_bool("TELEGRAPH_TITLE_FROM_QUESTION", False)
    )

    # --- Slow-request feedback ---
    # If an AI request runs longer than ``progress_delay`` seconds, a "waiting"
    # message is posted and refreshed every ``progress_interval`` seconds until
    # the answer arrives; the notice is deleted right before the reply is sent.
    # Requests that finish sooner never produce a message.
    enable_progress: bool = field(
        default_factory=lambda: _env_bool("ENABLE_PROGRESS", True)
    )
    progress_delay: float = field(
        default_factory=lambda: _env_float("PROGRESS_DELAY", 5.0)
    )
    progress_interval: float = field(
        default_factory=lambda: _env_float("PROGRESS_INTERVAL", 1.0)
    )
    progress_text: str = field(
        default_factory=lambda: _env("PROGRESS_TEXT", default=DEFAULT_PROGRESS_TEXT)
    )
    # Minimum seconds between two outgoing messages in the same chat. Telegram
    # allows roughly one message per second per chat; exceeding it returns 429.
    # Set to 0 to disable the throttle.
    send_min_interval: float = field(
        default_factory=lambda: _env_float("SEND_MIN_INTERVAL", 1.05)
    )

    # --- Timing footer ---
    # Append the AI API processing time to every answer.
    show_timing: bool = field(
        default_factory=lambda: _env_bool("SHOW_TIMING", True)
    )
    timing_format: str = field(
        default_factory=lambda: _env("TIMING_FORMAT", default=DEFAULT_TIMING_FORMAT)
    )

    # --- Telegraph (answers Telegram cannot render) ---
    # Markdown tables and raw LaTeX have no representation in Telegram
    # messages. Such answers are published as Telegraph pages and the user
    # receives a link; every failure falls back to normal delivery.
    # The API stopped accepting anonymous createPage calls (ACCESS_TOKEN_-
    # INVALID), so an account is required: set TELEGRAPH_SHORT_NAME +
    # TELEGRAPH_ACCESS_TOKEN (from createAccount), or leave both empty and
    # the bot creates an anonymous account itself and caches the token in
    # TELEGRAPH_ACCOUNT_FILE. The file holds a secret: keep it out of git.
    enable_telegraph: bool = field(
        default_factory=lambda: _env_bool("TELEGRAPH_ENABLED", True)
    )
    telegraph_api_url: str = field(
        default_factory=lambda: _env(
            "TELEGRAPH_API_URL", default="https://api.telegra.ph"
        ).rstrip("/")
        or "https://api.telegra.ph"
    )
    telegraph_author: str = field(
        default_factory=lambda: _env("TELEGRAPH_AUTHOR", default="Kapum ChatBot")
    )
    telegraph_short_name: str = field(
        default_factory=lambda: os.getenv("TELEGRAPH_SHORT_NAME", "").strip()
    )
    telegraph_access_token: str = field(
        default_factory=lambda: os.getenv("TELEGRAPH_ACCESS_TOKEN", "").strip()
    )
    telegraph_account_file: str = field(
        default_factory=lambda: _env(
            "TELEGRAPH_ACCOUNT_FILE", default="telegraph_account.json"
        )
    )
    telegraph_notice: str = field(
        default_factory=lambda: _env(
            "TELEGRAPH_NOTICE", default=DEFAULT_TELEGRAPH_NOTICE
        )
    )
    telegraph_timeout: int = field(
        default_factory=lambda: _env_int("TELEGRAPH_TIMEOUT", 20)
    )

    # --- Streaming answers ---
    # The reply is shown progressively in one message that keeps being edited.
    # Telegram tolerates roughly one edit per second per chat, so edits are
    # throttled to STREAM_EDIT_INTERVAL seconds. Guest replies stay single-shot
    # (a guest query allows exactly one result).
    stream_responses: bool = field(
        default_factory=lambda: _env_bool("STREAM_RESPONSES", False)
    )
    stream_edit_interval: float = field(
        default_factory=lambda: _env_float("STREAM_EDIT_INTERVAL", 3.0)
    )
    # Abort a stalled stream (no data for this many seconds) and fall back to
    # a regular single-shot request.
    stream_stall_timeout: int = field(
        default_factory=lambda: _env_int("STREAM_STALL_TIMEOUT", 60)
    )

    # --- "Ask another model" button ---
    # After an answer in a private chat, offer a button to re-ask the last
    # question with a different model (one-time, the default model stays).
    enable_ask_other_model: bool = field(
        default_factory=lambda: _env_bool("ASK_OTHER_MODEL", True)
    )

    # --- Scientific notation ---
    # Telegram cannot show LaTeX or sub/superscript formatting, so the model's
    # answer is rewritten with Unicode: H_2SO_4 → H₂SO₄, \alpha → α,
    # 10^{23} → 10²³, \frac{s}{t} → s/t. Code blocks are left untouched.
    normalize_formulas: bool = field(
        default_factory=lambda: _env_bool("NORMALIZE_FORMULAS", True)
    )
    render_formula_images: bool = field(
        default_factory=lambda: _env_bool("RENDER_FORMULA_IMAGES", True)
    )
    # Ask the model to answer in Telegram-friendly notation in the first place.
    enable_style_hint: bool = field(
        default_factory=lambda: _env_bool("ENABLE_STYLE_HINT", True)
    )
    style_hint: str = field(
        default_factory=lambda: _env("STYLE_HINT", default=DEFAULT_STYLE_HINT)
    )

    def __post_init__(self) -> None:
        """Parse admin IDs after initialization."""
        raw = os.getenv("ADMIN_IDS", "")
        self.admin_ids = []
        for chunk in raw.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            try:
                self.admin_ids.append(int(chunk))
            except ValueError:
                continue

        if self.parse_mode not in {"markdown", "html", "plain"}:
            self.parse_mode = "markdown"

        if self.progress_delay < 0:
            self.progress_delay = 0.0
        # Telegram tolerates roughly one message edit per second per chat.
        if self.progress_interval < 0.5:
            self.progress_interval = 0.5
        if self.send_min_interval < 0 or self.send_min_interval > 10:
            self.send_min_interval = 0.0 if self.send_min_interval < 0 else 10.0
        # Telegraph page creation is a courtesy call: bound it tightly.
        if self.telegraph_timeout < 5:
            self.telegraph_timeout = 5
        elif self.telegraph_timeout > 120:
            self.telegraph_timeout = 120
        # Keep streamed edits away from Telegram's ~1 edit/second per chat.
        if self.stream_edit_interval < 1.0:
            self.stream_edit_interval = 1.0
        elif self.stream_edit_interval > 30.0:
            self.stream_edit_interval = 30.0
        if self.stream_stall_timeout < 10:
            self.stream_stall_timeout = 10
        # Base64 must fit the ASR API's 10 MB input limit.
        self.voice_max_bytes = min(max(1, self.voice_max_bytes), 7_500_000)

        self.ai_models = _parse_models(os.getenv("AI_MODELS", "")) or [
            dict(model) for model in DEFAULT_AI_MODELS
        ]

    def is_admin(self, telegram_id: int) -> bool:
        """Return True if the given Telegram ID belongs to an administrator."""
        return telegram_id in self.admin_ids

    def validate(self) -> None:
        """Validate that required config values are present."""
        if not self.admin_ids:
            logging.getLogger(__name__).warning(
                "ADMIN_IDS is empty or invalid: admin commands are unavailable"
            )
        missing: list[str] = []
        if not self.bot_token:
            missing.append("BOT_TOKEN")
        if not self.ai_api_key:
            missing.append("AI_API_KEY (or QWEN_API_KEY / DASHSCOPE_API_KEY)")
        if not self.ai_base_url:
            missing.append("AI_BASE_URL")
        if missing:
            raise ValueError(
                "Missing required environment variables: "
                + ", ".join(missing)
                + ". Check your .env file."
            )


config = Config()
