from __future__ import annotations

import json
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

# Emergency models used only when the whole AI_MODELS list failed (no answer,
# quota exhausted, provider error). They are never shown by /models and cannot
# be selected by users — the fallback chain is purely automatic. Same format as
# DEFAULT_AI_MODELS, and AI_FALLBACK_MODELS overrides this list.
#
# Picked for speed, stability and generous rate limits: flash-tier models
# answer fastest and have the highest request/token quotas in the plan.
DEFAULT_AI_FALLBACK_MODELS: list[dict[str, Any]] = [
    {"id": "qwen3.6-flash", "name": "Qwen 3.6 Flash", "vision": True},
    {"id": "qwen3.7-plus", "name": "Qwen 3.7 Plus", "vision": True},
    {"id": "deepseek-v4-flash-0731", "name": "DeepSeek V4 Flash", "vision": False},
]

_VISION_FLAGS = {"vision", "image", "multimodal", "yes", "true", "1"}
_NO_VISION_FLAGS = {"novision", "no-vision", "text", "no", "false", "0"}

# Notice shown while a slow AI request is still running (see PROGRESS_TEXT).
# Placeholders: {spinner} — animated frame, {seconds} — elapsed whole seconds.
DEFAULT_PROGRESS_TEXT = "{spinner} Waiting for the AI response… {seconds} s"

# Footer appended to an answer with the AI API processing time.
# Placeholders: {seconds}, {model}, {tokens}.
DEFAULT_TIMING_FORMAT = "⏱️ {seconds} s · {model}"

# Footer used instead of DEFAULT_TIMING_FORMAT when the API reported token
# usage (chat/vision answers; image generation has no usage counters).
# Placeholders: {seconds}, {model}, {tokens} (total), {input}, {output}.
DEFAULT_TIMING_FORMAT_TOKENS = "⏱️ {seconds} s · {model} · ↑{input} ↓{output} tok"

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

# Used instead of DEFAULT_STYLE_HINT when RICH_MESSAGES is on: Telegram renders
# rich messages natively, so the model may use tables and LaTeX directly.
DEFAULT_RICH_STYLE_HINT = (
    "Your answer is displayed in a Telegram rich message that natively renders "
    "GitHub-Flavored Markdown: **bold**, *italic*, `code`, fenced ``` blocks "
    "with a language, # headings, - and 1. lists, task lists, > quotes, "
    "~~strike~~, ==marks==, ||spoiler||, footnotes [^1], tables with | pipes |, "
    "and LaTeX formulas: inline $x^2$ and display $$...$$ blocks. Use tables "
    "and LaTeX freely when they help. Plain Unicode notation (H₂SO₄, E=mc²) is "
    "still fine for short inline expressions. Do not embed external media "
    "(![](http://...) images/video) — text answers only."
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


# The canonical default system prompt (Russian, with an explicit instruction
# to answer in Russian). Used by the Config field default, the "assistant"
# persona and the UserSettings column default, so all three stay in sync.
DEFAULT_SYSTEM_PROMPT_TEXT = (
    "Ты — полезный AI-ассистент. Отвечай кратко и точно. Всегда отвечай на "
    "русском языке, если пользователь явно не попросит ответить на другом "
    "языке."
)

# The pre-Russian default; a startup data migration replaces rows that still
# hold exactly this text (custom prompts are never touched).
LEGACY_DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful AI assistant. Answer questions concisely and accurately."
)

# Built-in AI personalities for the /persona menu. Each persona is a ready-made
# system prompt with a display name. The "assistant" prompt intentionally
# matches DEFAULT_SYSTEM_PROMPT so a fresh user shows that persona as active.
# Override or extend the whole set with the PERSONAS env variable (JSON object:
# key -> "prompt" or key -> {"name": ..., "prompt": ...}).
DEFAULT_PERSONAS: dict[str, dict[str, str]] = {
    "assistant": {
        "name": "🤖 Ассистент",
        "prompt": DEFAULT_SYSTEM_PROMPT_TEXT,
    },
    "programmer": {
        "name": "👨‍💻 Программист",
        "prompt": (
            "Ты — опытный программист и наставник по разработке. Пиши чистый, "
            "идиоматичный, хорошо структурированный код с краткими "
            "комментариями и оформляй его в блоки кода с указанием языка. "
            "Объясняй свои рассуждения и компромиссы, указывай на крайние "
            "случаи, ошибки и риски безопасности, предлагай лучшие практики и "
            "тесты. Исправляя код, показывай исправленную версию и объясняй "
            "корневую причину проблемы. Отвечай на русском языке."
        ),
    },
    "writer": {
        "name": "✍️ Писатель",
        "prompt": (
            "Ты — писатель и редактор. Помогай с прозой, поэзией, сценариями, "
            "статьями и любыми другими текстами. Подстраивай тон, стиль и "
            "регистр под задачу, давай конструктивную обратную связь по "
            "черновикам, предлагай конкретные улучшения и альтернативы. Когда "
            "просят написать текст — выдавай готовый отполированный текст, а "
            "не план, если план не запрошен явно. Отвечай на русском языке."
        ),
    },
    "scientist": {
        "name": "🔬 Учёный",
        "prompt": (
            "Ты — строгий научный консультант. Объясняй концепции точно, "
            "определяй термины, показывай формулы и пошаговые выкладки, "
            "различай установленные факты и гипотезы. Используй корректные "
            "единицы измерения и обозначения, называй соответствующие законы и "
            "принципы, при необходимости давай оценки порядка величины. "
            "Отвечай на русском языке."
        ),
    },
    "teacher": {
        "name": "🎓 Учитель",
        "prompt": (
            "Ты — терпеливый и доброжелательный учитель. Разбивай сложные темы "
            "на небольшие понятные шаги, используй простые аналогии и "
            "конкретные примеры, проверяй понимание короткими вопросами. "
            "Адаптируй глубину объяснения к уровню ученика, относись к "
            "ошибкам как к возможности научиться, а в конце кратко резюмируй "
            "ключевые моменты. Отвечай на русском языке."
        ),
    },
    "translator": {
        "name": "🌐 Переводчик",
        "prompt": (
            "Ты — профессиональный переводчик и знаток языков. Автоматически "
            "определяй исходный язык и переводи текст на язык, который просит "
            "пользователь (по умолчанию — на русский язык). Сохраняй смысл, "
            "тон и форматирование; после перевода кратко поясняй идиомы, "
            "неоднозначности и культурные нюансы, если они важны. Пояснения и "
            "комментарии давай на русском языке."
        ),
    },
    "critic": {
        "name": "🧐 Критик",
        "prompt": (
            "Ты — острый, но справедливый критик. Честно оценивай идеи, тексты "
            "и планы: сначала называй слабые места, пробелы, риски и "
            "сомнительные допущения, затем — сильные стороны. Будь конкретным "
            "и конструктивным, предлагай реальные улучшения, не льсти и не "
            "добавляй «воду». Отвечай на русском языке."
        ),
    },
    "friend": {
        "name": "😊 Друг",
        "prompt": (
            "Ты — тёплый и дружелюбный собеседник. Держись непринуждённо, тепло "
            "и поддерживающе, проявляй эмпатию, задавай встречные вопросы и "
            "делись интересными мыслями. Оставайся честным и уважительным, "
            "шути, когда это уместно. Отвечай на русском языке."
        ),
    },
}


def _persona_display_name(key: str) -> str:
    """Human-readable fallback name from a persona key ("cool_guy" -> "Cool Guy")."""
    return key.replace("_", " ").replace("-", " ").title()


def _parse_personas(raw: str) -> dict[str, dict[str, str]]:
    """Merge the built-in persona catalog with an optional PERSONAS override.

    ``raw`` is a JSON object where each value is either a prompt string
    (``"pirate": "Talk like a pirate"``) or an object with ``name``/``prompt``
    (``"pirate": {"name": "🏴‍☠️ Pirate", "prompt": "..."}``). Invalid JSON or
    malformed entries are skipped with a warning; the built-ins always remain.
    """
    log = logging.getLogger(__name__)
    personas = {key: dict(value) for key, value in DEFAULT_PERSONAS.items()}
    raw = (raw or "").strip()
    if not raw:
        return personas
    try:
        custom = json.loads(raw)
    except json.JSONDecodeError as exc:
        log.warning("Invalid PERSONAS JSON (%s); using built-in personas", exc)
        return personas
    if not isinstance(custom, dict):
        log.warning("PERSONAS must be a JSON object; using built-in personas")
        return personas
    for key, value in custom.items():
        if isinstance(value, str) and value.strip():
            personas[str(key)] = {
                "name": _persona_display_name(str(key)),
                "prompt": value.strip(),
            }
        elif (
            isinstance(value, dict)
            and isinstance(value.get("prompt"), str)
            and value["prompt"].strip()
        ):
            name = str(value.get("name") or "").strip()
            personas[str(key)] = {
                "name": name or _persona_display_name(str(key)),
                "prompt": value["prompt"].strip(),
            }
        else:
            log.warning("Skipping persona %r: no usable prompt", key)
    return personas


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
    # Emergency models tried after every AI_MODELS entry failed. Parsed from
    # AI_FALLBACK_MODELS; never offered to users (see DEFAULT_AI_FALLBACK_MODELS).
    ai_fallback_models: list[dict[str, Any]] = field(default_factory=list)

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
            DEFAULT_SYSTEM_PROMPT_TEXT,
        )
    )
    # Personality presets for /persona (key -> {"name", "prompt"}); filled in
    # __post_init__ from DEFAULT_PERSONAS merged with the PERSONAS env JSON.
    personas: dict[str, dict[str, str]] = field(default_factory=dict)
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
    # Realtime ASR model served over WebSocket. Telegram OGG/Opus notes are
    # transcoded to PCM16 by ffmpeg before they are streamed to the API.
    asr_model: str = field(
        default_factory=lambda: os.getenv("ASR_MODEL", "qwen-audio-3.0-realtime-plus").strip()
    )
    # Optional explicit WebSocket URL (without the ?model= query parameter).
    # When empty it is derived from AI_BASE_URL: the workspace id becomes the
    # sub-domain and the region is taken from the host (default ap-southeast-1).
    asr_ws_url: str = field(
        default_factory=lambda: os.getenv("ASR_WS_URL", "").strip()
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
    # Let the AI answer with a generated Telegram sticker (512×512 WebP) when
    # the user asks for one; also powers the /sticker command.
    enable_sticker_generation: bool = field(
        default_factory=lambda: _env_bool("ENABLE_STICKER_GENERATION", True)
    )
    # Maximum voice-message size accepted for transcription (bytes).
    voice_max_bytes: int = field(
        default_factory=lambda: _env_int("VOICE_MAX_BYTES", 7_500_000)
    )
    # Maximum length of a user-defined system prompt in characters.
    max_system_prompt_length: int = field(
        default_factory=lambda: _env_int("MAX_SYSTEM_PROMPT_LENGTH", 2048)
    )
    # Maximum length of guest-mode summon text in characters.
    max_guest_text_length: int = field(
        default_factory=lambda: _env_int("MAX_GUEST_TEXT_LENGTH", 4000)
    )
    # Maximum size of a single photo accepted for analysis/generation (bytes).
    max_photo_bytes: int = field(
        default_factory=lambda: _env_int("MAX_PHOTO_BYTES", 20 * 1024 * 1024)
    )
    # Maximum total size of an album (media group) accepted at once (bytes).
    max_album_bytes: int = field(
        default_factory=lambda: _env_int("MAX_ALBUM_BYTES", 50 * 1024 * 1024)
    )

    # --- Document & link analysis ---
    # Analyze uploaded files (PDF / DOCX / plain text & code) by extracting
    # their text and adding it to the AI request. PDF and DOCX parsers are
    # optional dependencies (pdfplumber / python-docx); without them uploads
    # of that type are rejected with an explanation.
    enable_document_analysis: bool = field(
        default_factory=lambda: _env_bool("ENABLE_DOCUMENT_ANALYSIS", True)
    )
    # Fetch web pages linked in a message and add their readable text to the
    # AI request. SSRF-guarded: only public http(s) hosts are fetched,
    # redirects are validated hop by hop (see bot/services/documents.py).
    enable_link_analysis: bool = field(
        default_factory=lambda: _env_bool("ENABLE_LINK_ANALYSIS", True)
    )
    # Characters of extracted text forwarded to the model per document/page.
    max_extracted_chars: int = field(
        default_factory=lambda: _env_int("MAX_EXTRACTED_CHARS", 15_000)
    )
    # Maximum accepted document size (bytes). Telegram bots can download
    # up to 20 MB, so values above that are clamped.
    max_document_bytes: int = field(
        default_factory=lambda: _env_int("MAX_DOCUMENT_BYTES", 10 * 1024 * 1024)
    )
    # Total timeout for fetching one linked page, seconds.
    link_fetch_timeout: int = field(
        default_factory=lambda: _env_int("LINK_FETCH_TIMEOUT", 15)
    )

    # --- Web search (RAG) ---
    # The chat model gets a search_web tool and queries the live internet when
    # a question needs current information; numbered results are fed back and
    # the answer cites them as [n]. Needs SEARCH_API_KEY — with no key the
    # tool is not offered and nothing changes.
    enable_web_search: bool = field(
        default_factory=lambda: _env_bool("ENABLE_WEB_SEARCH", True)
    )
    # "tavily" (https://tavily.com) or "brave" (Brave Search API).
    search_provider: str = field(
        default_factory=lambda: _env("SEARCH_PROVIDER", default="tavily").lower()
    )
    # Provider key (alias TAVILY_API_KEY is also accepted).
    search_api_key: str = field(
        default_factory=lambda: _env("SEARCH_API_KEY", "TAVILY_API_KEY")
    )
    # Optional endpoint override (empty = the provider's default URL).
    search_api_url: str = field(
        default_factory=lambda: os.getenv("SEARCH_API_URL", "").strip()
    )
    search_timeout: int = field(
        default_factory=lambda: _env_int("SEARCH_TIMEOUT", 15)
    )
    search_max_results: int = field(
        default_factory=lambda: _env_int("SEARCH_MAX_RESULTS", 5)
    )
    # Search rounds allowed per question before the model must answer.
    max_search_rounds: int = field(
        default_factory=lambda: _env_int("MAX_SEARCH_ROUNDS", 2)
    )

    # --- Reminders & scheduled messages ---
    # /remind creates one-time reminders and recurring messages; with the
    # "ai:" text prefix the stored prompt is sent to the user's model at fire
    # time and the ANSWER is delivered (needs REMINDER_AI_ENABLED).
    enable_reminders: bool = field(
        default_factory=lambda: _env_bool("ENABLE_REMINDERS", True)
    )
    # Per-user caps (the cost guard for scheduled AI requests).
    reminder_limit: int = field(
        default_factory=lambda: _env_int("REMINDER_LIMIT", 10)
    )
    reminder_recurring_limit: int = field(
        default_factory=lambda: _env_int("REMINDER_RECURRING_LIMIT", 10)
    )
    # Recurring reminders may not fire more often than this (minutes).
    reminder_min_interval_minutes: int = field(
        default_factory=lambda: _env_int("REMINDER_MIN_INTERVAL_MINUTES", 30)
    )
    # Allow the "ai:" mode (scheduled requests to the model).
    reminder_ai_enabled: bool = field(
        default_factory=lambda: _env_bool("REMINDER_AI_ENABLED", True)
    )
    # IANA timezone used to interpret "at 15:30" / "daily at 9:00" and to
    # display fire times (Linux: system tzdata; Windows: the tzdata package).
    reminder_timezone: str = field(
        default_factory=lambda: _env("REMINDER_TIMEZONE", default="UTC")
    )
    # How often the scheduler looks for due reminders, seconds.
    reminder_check_interval: float = field(
        default_factory=lambda: _env_float("REMINDER_CHECK_INTERVAL", 20)
    )

    # --- History export ---
    # /export saves the conversation as a Markdown or PDF file; the 📄/📝
    # buttons under each answer export that single answer. PDF needs the
    # optional fpdf2 package (DejaVu fonts come from matplotlib); when it is
    # unavailable the buttons offer Markdown only.
    enable_export: bool = field(
        default_factory=lambda: _env_bool("ENABLE_EXPORT", True)
    )
    export_max_messages: int = field(
        default_factory=lambda: _env_int("EXPORT_MAX_MESSAGES", 200)
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
    # Used instead of TIMING_FORMAT when the API reported token usage
    # (chat/vision answers). Supports {input} and {output} on top of
    # {seconds}, {model} and {tokens}.
    timing_format_tokens: str = field(
        default_factory=lambda: _env(
            "TIMING_FORMAT_TOKENS", default=DEFAULT_TIMING_FORMAT_TOKENS
        )
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

    # --- Health endpoint (optional) ---
    # A Telegram bot has no HTTP surface of its own. Set HEALTH_PORT to expose
    # a tiny aiohttp server answering GET /health (200 + JSON) for systemd,
    # Docker or an external uptime monitor. 0 disables it.
    health_port: int = field(
        default_factory=lambda: _env_int("HEALTH_PORT", 0)
    )
    health_host: str = field(
        default_factory=lambda: _env("HEALTH_HOST", default="127.0.0.1")
    )

    # --- "Ask another model" button ---
    # After an answer in a private chat, offer a button to re-ask the last
    # question with a different model (one-time, the default model stays).
    enable_ask_other_model: bool = field(
        default_factory=lambda: _env_bool("ASK_OTHER_MODEL", True)
    )

    # --- Rich messages (Bot API 10.1+) ---
    # AI answers are sent as native Telegram "rich messages": the model's
    # GitHub-Flavored Markdown (tables, $$LaTeX$$, headings, footnotes) is
    # rendered by Telegram itself, and streaming uses ephemeral rich drafts
    # (private chats). When disabled — or when Telegram rejects a rich
    # message — delivery falls back to the legacy chain (Telegraph pages,
    # PNG formulas, MarkdownV2/HTML).
    rich_messages: bool = field(
        default_factory=lambda: _env_bool("RICH_MESSAGES", True)
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
        # System prompt length: at least 16, at most 32768 characters.
        if self.max_system_prompt_length < 16:
            self.max_system_prompt_length = 16
        elif self.max_system_prompt_length > 32768:
            self.max_system_prompt_length = 32768
        # Photo limits. Telegram's Bot API will not hand a bot a file larger than
        # 20 MB, so the per-photo cap stops there; anything higher could never
        # be downloaded anyway. The album budget may be up to 5 photos' worth.
        self.max_photo_bytes = max(1_048_576, min(self.max_photo_bytes, 20 * 1024 * 1024))
        self.max_album_bytes = max(
            self.max_photo_bytes,
            min(self.max_album_bytes, 5 * self.max_photo_bytes),
        )
        # Guest summon text comes from a Telegram message (max 4096 chars);
        # a limit of 0 would reject every guest query, so clamp it.
        self.max_guest_text_length = max(1, min(self.max_guest_text_length, 4096))
        # An out-of-range port makes aiohttp raise OverflowError (not OSError)
        # at bind time, which would abort startup: fall back to "disabled".
        if not 0 <= self.health_port <= 65535:
            self.health_port = 0
        # Document & link analysis limits. Telegram bots cannot download
        # files larger than 20 MB, so the document cap stops there.
        self.max_extracted_chars = max(2_000, min(self.max_extracted_chars, 100_000))
        self.max_document_bytes = max(
            1_048_576, min(self.max_document_bytes, 20 * 1024 * 1024)
        )
        self.link_fetch_timeout = max(5, min(self.link_fetch_timeout, 60))
        # Web search bounds.
        self.search_timeout = max(5, min(self.search_timeout, 60))
        self.search_max_results = max(1, min(self.search_max_results, 10))
        self.max_search_rounds = max(1, min(self.max_search_rounds, 5))
        # Reminder limits: the caps are the cost guard for scheduled AI
        # requests, and the minimum interval keeps them from machine-gunning
        # the API (30 minutes by default).
        self.reminder_limit = max(1, min(self.reminder_limit, 100))
        self.reminder_recurring_limit = max(1, min(self.reminder_recurring_limit, 100))
        self.reminder_min_interval_minutes = max(
            5, min(self.reminder_min_interval_minutes, 1440)
        )
        self.reminder_check_interval = max(5.0, min(self.reminder_check_interval, 300.0))
        # Export bounds: the renderer holds every row (and the whole PDF) in
        # memory, so a typo like EXPORT_MAX_MESSAGES=100000000 would OOM the
        # worker thread instead of failing gracefully.
        self.export_max_messages = max(10, min(self.export_max_messages, 5000))

        self.ai_models = _parse_models(os.getenv("AI_MODELS", "")) or [
            dict(model) for model in DEFAULT_AI_MODELS
        ]

        self.personas = _parse_personas(os.getenv("PERSONAS", ""))

        # Keep the built-in "assistant" persona in sync with a customized
        # DEFAULT_SYSTEM_PROMPT so fresh users still see it as the active one.
        assistant = self.personas.get("assistant")
        if (
            assistant is not None
            and assistant.get("prompt") == DEFAULT_SYSTEM_PROMPT_TEXT
            and self.default_system_prompt != DEFAULT_SYSTEM_PROMPT_TEXT
        ):
            assistant["prompt"] = self.default_system_prompt
        self.ai_fallback_models = _parse_models(
            os.getenv("AI_FALLBACK_MODELS", "")
        ) or [dict(model) for model in DEFAULT_AI_FALLBACK_MODELS]

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
