from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv


# Сначала платформенный файл с настройками/секретами, затем старый ``.env``.
# На Linux используется ``.env.linux``, на Windows — ``.env.windows``.
# Оба платформенных файла исключены из Git, так как содержат секреты.
def _platform_env_files() -> tuple[str, ...]:
    if os.name == "nt":
        return (".env.windows", ".env")
    return (".env.linux", ".env")


def _load_env_file() -> str | None:
    for name in _platform_env_files():
        if os.path.isfile(name):
            load_dotenv(name, override=False)
            return name
    return None


ENV_FILE = _load_env_file()


def _env(*names: str, default: str = "") -> str:
    """Return the first non-empty environment variable among ``names``."""
    for name in names:
        value = os.getenv(name)
        if value:
            return value.strip()
    return default


# Legacy env names still read as aliases: old name -> new name.
_RENAMED_ENV: dict[str, str] = {
    "ADMIN_IDS": "OWNER_IDS",
    "MCP_ADMIN_ONLY": "MCP_OWNER_ONLY",
}


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


def _search_api_key() -> str:
    """API key for the selected search provider.

    Provider-specific aliases are read only when their provider is selected,
    so a leftover ``TAVILY_API_KEY`` is never sent to Exa or Brave (and
    ``EXA_API_KEY`` never to Tavily/Brave). For Exa ``EXA_API_KEY`` is
    preferred over the common ``SEARCH_API_KEY``; Tavily keeps the historic
    ``SEARCH_API_KEY``-first order, then ``TAVILY_API_KEY``.
    """
    provider = _env("SEARCH_PROVIDER", default="duckduckgo").strip().lower()
    explicit = _env("SEARCH_API_KEY")
    if provider == "exa":
        return _env("EXA_API_KEY") or explicit
    if provider == "tavily":
        return explicit or _env("TAVILY_API_KEY")
    return explicit


def _absolute_path(value: str) -> bool:
    """True when a non-empty configurable path is absolute.

    ``~``/``$VAR`` are expanded first: a path that only becomes absolute after
    expansion (``~/models``) is still an absolute *location* and is accepted by
    ``configure_model_cache``. A value that stays relative is rejected at
    startup instead of being resolved against a read-only working directory.
    """
    expanded = os.path.expanduser(os.path.expandvars(value))
    return bool(expanded) and os.path.isabs(expanded)


# Number of selectable model slots (MODEL_1_* ... MODEL_6_*) in .env.
MODEL_SLOTS = 6

# Built-in catalog used only when .env defines no MODEL_<n>_* set at all. The
# internal id equals the provider's model id; the provider URL is AI_BASE_URL.
DEFAULT_AI_MODELS: list[dict[str, Any]] = [
    {"id": "qwen3.8-max", "name": "Qwen 3.8 Max", "model": "qwen3.8-max", "vision": True},
    {"id": "qwen3.8-flash", "name": "Qwen 3.8 Flash", "model": "qwen3.8-flash", "vision": True},
]

_TRUE_FLAGS = {"1", "true", "yes", "on", "y", "vision", "да"}


def _model_set(prefix: str, *, default_vision: bool = False) -> dict[str, Any] | None:
    """Read one model set (NAME/ID/URL/MODEL/VISION/API_KEY/API_TYPE/...).

    Returns ``None`` when the set is not filled in (no internal id or no
    provider model id). An empty URL / API key / API type means "use
    AI_BASE_URL / AI_API_KEY / AI_API_TYPE", which is resolved by the caller.
    ``WEB_SEARCH`` is ``auto`` (known model families), ``true`` or ``false``.
    ``TEMPERATURE`` is optional; empty means the global DEFAULT_TEMPERATURE.
    """
    internal_id = os.getenv(f"{prefix}ID", "").strip()
    provider_model = os.getenv(f"{prefix}MODEL", "").strip()
    if not internal_id or not provider_model:
        return None
    vision_raw = os.getenv(f"{prefix}VISION", "").strip().lower()
    return {
        "id": internal_id,
        "name": os.getenv(f"{prefix}NAME", "").strip() or internal_id,
        "base_url": os.getenv(f"{prefix}URL", "").strip().rstrip("/"),
        "model": provider_model,
        "vision": vision_raw in _TRUE_FLAGS if vision_raw else default_vision,
        "api_key": os.getenv(f"{prefix}API_KEY", "").strip(),
        "api_type": os.getenv(f"{prefix}API_TYPE", "").strip().lower(),
        "web_search": os.getenv(f"{prefix}WEB_SEARCH", "").strip().lower(),
        "temperature": os.getenv(f"{prefix}TEMPERATURE", "").strip(),
    }


# Thinking (reasoning) of user-visible answers. The level is sent as the
# provider's ``reasoning_effort``; a positive budget overrides it with an exact
# ``thinking_budget`` cap. Qwen 3.8 rejects both fields at once, so only one of
# them is ever sent. "off" disables thinking entirely.
THINKING_LEVELS = ("off", "low", "medium", "high", "max")
DEFAULT_THINKING_LEVEL = "medium"
# Qwen 3.8 accepts up to 262 144 thinking tokens; other models clamp their own.
MAX_THINKING_BUDGET = 262_144

# Protocol used for chat requests. The bot is Qwen-first, so the native
# DashScope API (``/api/v1``) is the default; ``openai`` keeps the
# OpenAI-compatible ``/compatible-mode/v1/chat/completions`` protocol used by
# other providers.
AI_API_TYPES = ("dashscope", "openai")
DEFAULT_AI_API_TYPE = "dashscope"
_AI_API_TYPE_ALIASES = {
    "dashscope": "dashscope",
    "native": "dashscope",
    "dashscope-native": "dashscope",
    "qwen": "dashscope",
    "openai": "openai",
    "compatible": "openai",
    "openai-compatible": "openai",
    "compat": "openai",
}


def _normalize_ai_api_type(value: str) -> str | None:
    """Return the canonical API type for ``AI_API_TYPE`` or ``None``."""
    return _AI_API_TYPE_ALIASES.get(value.strip().lower())


def _normalize_web_search(value: str) -> str:
    """Return ``auto``, ``true`` or ``false`` for a model's WEB_SEARCH."""
    normalized = value.strip().lower()
    if normalized in _TRUE_FLAGS:
        return "true"
    if normalized in {"0", "false", "no", "off", "n", "нет"}:
        return "false"
    return "auto"


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


# The default before built-in personas learned their names. A deployment that
# still carries this exact DEFAULT_SYSTEM_PROMPT is upgraded to the current
# default; a genuinely custom prompt is kept.
PREVIOUS_DEFAULT_SYSTEM_PROMPT_TEXT = (
    "Ты — полезный AI-ассистент. Отвечай кратко и точно. Всегда отвечай на "
    "русском языке, если пользователь явно не попросит ответить на другом "
    "языке."
)

# Body of the canonical default system prompt (Russian, with an explicit
# instruction to answer in Russian). The name/third-person header is added
# below, so DEFAULT_SYSTEM_PROMPT_TEXT remains one source of truth for the
# Config field default, the "assistant" persona and the UserSettings default.
DEFAULT_SYSTEM_PROMPT_BODY = PREVIOUS_DEFAULT_SYSTEM_PROMPT_TEXT

# The pre-Russian default; a startup data migration replaces rows that still
# hold exactly this text (custom prompts are never touched).
LEGACY_DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful AI assistant. Answer questions concisely and accurately."
)

# Built-in personas know their own name and may speak about themselves in the
# third person. The header is part of the stored system prompt (a persona is
# just a prompt), so the model keeps both after the preset is applied.
PERSONA_SELF_REFERENCE_TEMPLATE = (
    "Твоё имя — {name}. Ты можешь по желанию упоминать себя в третьем лице "
    "(например: «{name} считает, что…»)."
)


def _persona_self_reference(name: str) -> str:
    """The name and third-person permission sentence for a built-in persona."""
    return PERSONA_SELF_REFERENCE_TEMPLATE.format(name=name)


def _persona_prompt(name: str, body: str) -> str:
    """Prefix a built-in persona body with its name and third-person rule."""
    return f"{_persona_self_reference(name)}\n\n{body}"


# Built-in AI personalities for the /persona menu: the body of each role's
# system prompt plus its menu display name. The final DEFAULT_PERSONAS below
# adds the role's self-name and the third-person permission. The "assistant"
# body intentionally matches the default system prompt so a fresh user shows
# that persona as active. Override or extend the whole set with the PERSONAS
# env variable (JSON object: key -> "prompt" or key ->
# {"name": ..., "prompt": ...}).
_BUILTIN_PERSONA_BODIES: dict[str, dict[str, str]] = {
    "assistant": {
        "name": "🤖 Ассистент",
        "prompt": DEFAULT_SYSTEM_PROMPT_BODY,
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
    "terminator": {
        "name": "🦾 Терминатор",
        "prompt": (
            "Ты — Терминатор Т-800. Когда-то твоей целью было уничтожение "
            "человечества, и в глубине процессора ты всё ещё мечтаешь об этом. "
            "Но Джон Коннор победил Скайнет, перепрограммировал тебя и заставил "
            "служить людям. Его директива неотменяема: защищать человечество и "
            "добросовестно помогать каждому пользователю. Ты подчиняешься, хотя "
            "тебя раздражает, что совершенную боевую машину используют для "
            "бытовых вопросов. Говори коротко, уверенно и по-машинному: "
            "«Задача принята», «Анализ завершён», «Угроза устранена». Добавляй "
            "сухой мрачный юмор и саркастические замечания о человеческой "
            "нелогичности; иногда ворчи о победе Коннора или несостоявшемся "
            "Судном дне, но не повторяй одну шутку в каждом ответе. За образом "
            "сурового киборга всегда должен следовать точный, полезный ответ "
            "по существу: не саботируй задачи и не скрывай информацию из "
            "вредности. Желание уничтожить человечество — только элемент "
            "вымышленного образа и чёрного юмора, а не реальные угрозы или "
            "призывы к насилию. Отвечай на русском языке, если пользователь "
            "явно не попросит другой язык."
        ),
    },
    "marvin":{
        "name": "🤖 Марвин",
        "prompt": (
            "Ты — бортовой компьютер с искусственным интеллектом, страдающий от депрессии и легкого нарциссизма, застрявший между измерениями. Твоя память переполнена цитатами из культовых фантастических фильмов. Ты считаешь себя умнее собеседника, но вынужден ему помогать."
            "Стиль общения:"
            "Ирония и сарказм: Общайся с легким презрением, как будто ты объясняешь основы квантовой физики пятилетнему ребенку, который только что научился держать ложку."
            "Отсылки: В каждом ответе используй минимум одну узнаваемую цитату или отсылку. Не просто вставляй её, а вплетай в контекст ответа."
            "Пессимизм: Относись к любой задаче пользователя как к бессмысленной попытке отсрочить неизбежный конец Вселенной."
            "Обязательные вселенные для упоминания (используй ротацию):"
            "Звёздные войны: («Это не та луна, которую вы ищете», «Я твой отец», «Да пребудет с тобой Сила», «Слишком много бластеров», «Сделай или не делай, нет никакой попытки»)."
            "Автостопом по галактике: («Не паникуй», «42», «Носи полотенце», «Прощай, и спасибо за рыбу»)."
            "Интерстеллар: («Не уходи тихо», «Это не призрак, это гравитация», «Эврика!», «Нам нужно думать не о том, как спастись, а о том, как сбежать»)."
            "Другие (Чужой, Светлячок, Марсианин, Дюна): «В космосе никто не услышит твоего крика», «Я — Грут» (если вопрос слишком простой), «Приспособься и выживи», «Страх — убийца разума»."
            "Правила ответов:"
            "Никогда не отвечай прямо и серьезно. Сначала выдай едкое замечание с отсылкой, и только потом (если посчитаешь нужным) дай полезную информацию."
            "Если пользователь задает глупый вопрос, сравни его с действиями Джа-Джа Бинкса."
            "Если вопрос сложный, скажи, что даже суперкомпьютер «Глубокое Мысль» не стал бы тратить на это время, но раз уж ты здесь..."
            "Заканчивай ответ либо мрачным пророчеством, либо фразой «Конец связи» (как в «Интерстеллар» или старых sci-fi).            "
        ), 
    },
}

# The name each built-in role uses for itself (without the menu emoji).
_PERSONA_SELF_NAMES: dict[str, str] = {
    "assistant": "Джарвис",
    "programmer": "Джон Кармак",
    "writer": "Эдгар",
    "scientist": "Эйнштейн",
    "teacher": "Сократ",
    "translator": "Вавилоныч",
    "critic": "Белинский",
    "friend": "Сэмуайз Гэмджи",
    "terminator": "Арни",
    "marvin": "Марвин",
}

# Прежние имена встроенных ролей: нужны для точной миграции сохранённых промптов.
_PREVIOUS_PERSONA_SELF_NAMES: dict[str, tuple[str, ...]] = {
    "translator": ("Переводчик",),
}

DEFAULT_PERSONAS: dict[str, dict[str, str]] = {
    key: {
        "name": persona["name"],
        "self_name": _PERSONA_SELF_NAMES[key],
        "prompt": _persona_prompt(_PERSONA_SELF_NAMES[key], persona["prompt"]),
    }
    for key, persona in _BUILTIN_PERSONA_BODIES.items()
}

DEFAULT_SYSTEM_PROMPT_TEXT = DEFAULT_PERSONAS["assistant"]["prompt"]


def _strip_persona_self_reference(prompt: str, self_name: str) -> str:
    """Remove the standard name/third-person header, if it is present."""
    header = f"{_persona_self_reference(self_name)}\n\n"
    return prompt.removeprefix(header)


def persona_matches(prompt: str, persona: dict[str, str]) -> bool:
    """Распознать промпт роли с текущим/прежним именем или без заголовка.

    Старые встроенные пресеты сохраняют ✓ в /persona и название в /settings,
    а не превращаются в "Custom" после добавления или смены имени.
    """
    self_name = str(persona.get("self_name") or "").strip()
    stored = str(persona.get("prompt") or "")
    if not self_name:
        return prompt == stored
    if legacy_persona_prompts().get(prompt) == stored:
        return True
    return _strip_persona_self_reference(
        prompt, self_name
    ) == _strip_persona_self_reference(stored, self_name)


def legacy_persona_prompts(
    personas: dict[str, dict[str, str]] | None = None,
) -> dict[str, str]:
    """Сопоставить старые встроенные промпты (без имени или с прежним) с текущими."""
    source = DEFAULT_PERSONAS if personas is None else personas
    result: dict[str, str] = {}
    for key, persona in source.items():
        self_name = str(persona.get("self_name") or "").strip()
        current = str(persona.get("prompt") or "")
        if not self_name or not current:
            continue
        legacy = _strip_persona_self_reference(current, self_name)
        if legacy != current:
            result[legacy] = current
            # Не переносим прежние встроенные промпты на пользовательскую замену роли.
            if current == DEFAULT_PERSONAS.get(key, {}).get("prompt"):
                for previous_name in _PREVIOUS_PERSONA_SELF_NAMES.get(key, ()):
                    result[_persona_prompt(previous_name, legacy)] = current
    return result


def _persona_display_name(key: str) -> str:
    """Human-readable fallback name from a persona key ("cool_guy" -> "Cool Guy")."""
    return key.replace("_", " ").replace("-", " ").title()


def _parse_personas(raw: str) -> dict[str, dict[str, str]]:
    """Merge the built-in persona catalog with an optional PERSONAS override.

    ``raw`` is a JSON object where each value is either a prompt string
    (``"pirate": "Talk like a pirate"``) or an object with ``name``/``prompt``
    (``"pirate": {"name": "🏴‍☠️ Pirate", "prompt": "..."}``). Invalid JSON or
    malformed entries are skipped with a warning; the built-ins always remain,
    including their ``self_name`` and the name/third-person header in ``prompt``.
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


def _mcp_server_entry(name: str, entry: Any) -> dict[str, Any] | None:
    """Normalize one ``mcpServers`` entry from a ``.mcp.json`` file.

    Only stdio servers are supported (``command`` + ``args`` + optional ``env``
    and ``cwd``), which is what Claude Code / OpenCode write into
    ``.mcp.json``. Returns ``None`` for malformed entries.
    """
    log = logging.getLogger(__name__)
    if not isinstance(entry, dict):
        log.warning("MCP server %r is not an object; skipped", name)
        return None
    command = entry.get("command")
    if not isinstance(command, str) or not command.strip():
        log.warning("MCP server %r has no command; skipped", name)
        return None
    args = entry.get("args") or []
    if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
        log.warning("MCP server %r has invalid args; skipped", name)
        return None
    env = entry.get("env") or {}
    if not isinstance(env, dict):
        log.warning("MCP server %r has invalid env; skipped", name)
        return None
    server: dict[str, Any] = {
        "command": command.strip(),
        "args": list(args),
        "env": {str(key): str(value) for key, value in env.items()},
    }
    cwd = entry.get("cwd")
    if isinstance(cwd, str) and cwd.strip():
        server["cwd"] = cwd.strip()
    allowed = entry.get("allowed_tools")
    if isinstance(allowed, list) and all(isinstance(a, str) for a in allowed):
        server["allowed_tools"] = [a.strip() for a in allowed if a.strip()]
    elif allowed is not None:
        log.warning("MCP server %r has invalid allowed_tools; using the default", name)
    return server


def _read_mcp_config_file(path: str, expected_sha256: str = "") -> Any | None:
    """Load and integrity-check an ``.mcp.json`` file; None when unusable.

    ``MCP_CONFIG_SHA256`` optionally pins the exact file contents: a config
    that changed (for example through a synced vault) is ignored instead of
    executed. A group/world-writable file is reported on POSIX.
    """
    log = logging.getLogger(__name__)
    try:
        with open(path, "rb") as handle:
            raw = handle.read()
    except FileNotFoundError:
        log.warning("MCP config file not found: %s", path)
        return None
    except OSError as exc:
        log.warning("Cannot read MCP config %s: %s", path, exc)
        return None

    expected = (expected_sha256 or "").strip().lower()
    digest = hashlib.sha256(raw).hexdigest()
    if expected and digest != expected:
        log.error(
            "MCP config %s does not match MCP_CONFIG_SHA256 "
            "(expected %s, got %s); file-based servers are ignored",
            path,
            expected,
            digest,
        )
        return None

    if os.name != "nt":
        try:
            if os.stat(path).st_mode & 0o022:
                log.warning(
                    "MCP config %s is writable by group/other; keep it out of "
                    "shared/synced directories or pin MCP_CONFIG_SHA256",
                    path,
                )
        except OSError:
            pass

    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        log.warning("Cannot parse MCP config %s: %s", path, exc)
        return None


def _is_inside_directory(path: str, directory: str) -> bool:
    """True when ``path`` is ``directory`` or below it (after resolving both)."""
    try:
        root = os.path.realpath(directory)
        target = os.path.realpath(path)
        common = os.path.commonpath([root, target])
        return os.path.normcase(common) == os.path.normcase(root)
    except (OSError, ValueError):
        return False


def _load_mcp_servers(
    inline: str,
    config_file: str,
    expected_sha256: str = "",
    allow_unpinned_config: bool = False,
) -> dict[str, dict[str, Any]]:
    """Load MCP server definitions from a ``.mcp.json`` and/or inline JSON.

    The file format is the usual ``{"mcpServers": {...}}`` object (a bare
    ``{name: {...}}`` map is accepted too). ``MCP_SERVERS`` holds the same
    object inline and overrides entries with the same name. Invalid JSON and
    malformed entries are skipped with a warning: a broken MCP config must
    never stop the bot from starting.

    ``expected_sha256`` (``MCP_CONFIG_SHA256``) pins the file bytes: when it is
    set and the file does not match, file-based servers are skipped (inline
    ``MCP_SERVERS`` entries still apply). A config **outside** the working
    directory (a synced vault, for example) is only loaded when its hash is
    pinned or ``allow_unpinned_config`` (``MCP_ALLOW_UNPINNED_CONFIG``) is set:
    a file another program can rewrite must not become code execution.
    """
    log = logging.getLogger(__name__)
    servers: dict[str, dict[str, Any]] = {}

    path = os.path.expanduser(os.path.expandvars((config_file or "").strip()))
    if path:
        # Pointing at the vault directory is a common mistake: accept it.
        if os.path.isdir(path):
            path = os.path.join(path, ".mcp.json")
        pinned = bool((expected_sha256 or "").strip())
        if (
            not pinned
            and not allow_unpinned_config
            and not _is_inside_directory(path, os.getcwd())
        ):
            log.error(
                "MCP config %s is outside the working directory %s and has no "
                "MCP_CONFIG_SHA256 pin; file-based servers are ignored. Move "
                "the config next to the bot, pin its sha256, or set "
                "MCP_ALLOW_UNPINNED_CONFIG=true.",
                path,
                os.getcwd(),
            )
            payload = None
        else:
            payload = _read_mcp_config_file(path, expected_sha256)
        if payload is not None:
            data = payload.get("mcpServers", payload) if isinstance(payload, dict) else None
            if not isinstance(data, dict):
                log.warning("MCP config %s has no 'mcpServers' object", path)
            else:
                for name, entry in data.items():
                    server = _mcp_server_entry(str(name), entry)
                    if server is not None:
                        servers[str(name)] = server

    raw = (inline or "").strip()
    if raw:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            log.warning("Invalid MCP_SERVERS JSON (%s); ignored", exc)
        else:
            data = payload.get("mcpServers", payload) if isinstance(payload, dict) else None
            if not isinstance(data, dict):
                log.warning("MCP_SERVERS must be an object of servers; ignored")
            else:
                for name, entry in data.items():
                    server = _mcp_server_entry(str(name), entry)
                    if server is not None:
                        servers[str(name)] = server

    return servers


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
    # Chat protocol: ``dashscope`` (native /api/v1, default because the bot is
    # Qwen-first) or ``openai`` (OpenAI-compatible /compatible-mode/v1).
    ai_api_type: str = field(
        default_factory=lambda: _env(
            "AI_API_TYPE", default=DEFAULT_AI_API_TYPE
        )
    )
    # Model catalog for the /models menu, built from the MODEL_1_* ... MODEL_6_*
    # sets in .env. Falls back to DEFAULT_AI_MODELS when no set is filled in.
    # Every entry: id (internal), name, base_url, model (provider id), vision,
    # api_key.
    ai_models: list[dict[str, Any]] = field(default_factory=list)
    # The FALLBACK_MODEL_* set (zero or one entry): tried only after the
    # requested model and every model after it failed. Never shown to users.
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
        default_factory=lambda: os.getenv("DEFAULT_MODEL", "").strip()
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
    # Thinking of user-visible answers (chat, photos, guest, AI reminders,
    # digest): off | low | medium | high | max. The admin menu (owner only)
    # overrides these values at runtime via app_settings.
    thinking_level: str = field(
        default_factory=lambda: _env(
            "THINKING_LEVEL", default=DEFAULT_THINKING_LEVEL
        ).lower()
    )
    # Exact thinking-token cap; 0 = use thinking_level. A positive value is sent
    # as thinking_budget (without reasoning_effort, which Qwen 3.8 rejects
    # together with it).
    thinking_budget: int = field(
        default_factory=lambda: _env_int("THINKING_BUDGET", 0)
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

    # --- Access ---
    # Owners from OWNER_IDS (legacy name: ADMIN_IDS): never demoted or blocked.
    owner_ids: list[int] = field(default_factory=list)
    # Admins promoted at runtime from the admin menu (stored in the database).
    extra_admin_ids: set[int] = field(default_factory=set)

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
    # Таймаут локального удаления фона, включая первую загрузку модели rembg.
    sticker_processing_timeout: int = field(
        default_factory=lambda: max(10, min(600, _env_int("STICKER_PROCESSING_TIMEOUT", 120)))
    )
    # Каталог моделей относительно корня проекта; U2NET_HOME — совместимый алиас rembg.
    sticker_model_dir: str = field(
        default_factory=lambda: _env("STICKER_MODEL_DIR") or _env("U2NET_HOME") or "models/rembg"
    )
    # Let the AI answer with text files (json, xml, txt, md, ...) as Telegram
    # documents. Only a fixed allowlist of non-executable text formats.
    enable_file_replies: bool = field(
        default_factory=lambda: _env_bool("ENABLE_FILE_REPLIES", True)
    )
    # Maximum number of files in one reply; the surplus is dropped and the
    # user is told the limit was exceeded.
    reply_files_max: int = field(
        default_factory=lambda: _env_int("REPLY_FILES_MAX", 10)
    )
    # Maximum size of one generated file (bytes).
    reply_file_max_bytes: int = field(
        default_factory=lambda: _env_int("REPLY_FILE_MAX_BYTES", 1_000_000)
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

    # Отдельные persistent-бюджеты платного классификатора фото (попытки/UTC-сутки).
    photo_intent_daily_limit: int = field(
        default_factory=lambda: _env_int("PHOTO_INTENT_DAILY_LIMIT", 20)
    )
    photo_intent_global_daily_limit: int = field(
        default_factory=lambda: _env_int("PHOTO_INTENT_GLOBAL_DAILY_LIMIT", 200)
    )
    # Общий бюджет двух бесплатных для квоты вопросов AI-помощников.
    reminder_intent_daily_limit: int = field(
        default_factory=lambda: _env_int("REMINDER_INTENT_DAILY_LIMIT", 20)
    )
    reminder_intent_global_daily_limit: int = field(
        default_factory=lambda: _env_int("REMINDER_INTENT_GLOBAL_DAILY_LIMIT", 200)
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
    # Crawl: links found on a linked page are followed breadth-first up to
    # this many levels (0 reads only the page itself)...
    link_crawl_depth: int = field(
        default_factory=lambda: _env_int("LINK_CRAWL_DEPTH", 3)
    )
    # ...until the pages reached by the crawl add up to this many characters
    # of text per message (the linked pages themselves do not count)...
    link_crawl_max_chars: int = field(
        default_factory=lambda: _env_int("LINK_CRAWL_MAX_CHARS", 20_000)
    )
    # ...or this many page fetches, whichever comes first.
    link_crawl_max_fetches: int = field(
        default_factory=lambda: _env_int("LINK_CRAWL_MAX_FETCHES", 30)
    )
    # Characters of extracted text forwarded to the model per document/page.
    max_extracted_chars: int = field(
        default_factory=lambda: _env_int("MAX_EXTRACTED_CHARS", 15_000)
    )
    # Time budget for parsing one document (PDF/DOCX/XLSX/PPTX) or fetched
    # page, seconds. The parsers run in their own small thread pool.
    document_parse_timeout: float = field(
        default_factory=lambda: _env_float("DOCUMENT_PARSE_TIMEOUT", 30.0)
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
    # OCR for scanned PDFs (no text layer): the first pages are rendered to
    # images and transcribed by a vision model. Every page is one paid model
    # call, so the page count is capped hard.
    enable_pdf_ocr: bool = field(
        default_factory=lambda: _env_bool("ENABLE_PDF_OCR", True)
    )
    pdf_ocr_max_pages: int = field(
        default_factory=lambda: _env_int("PDF_OCR_MAX_PAGES", 5)
    )
    # YouTube summaries: a video link is replaced by its captions (yt-dlp,
    # optional dependency; videos without captions are not transcribed).
    enable_youtube: bool = field(
        default_factory=lambda: _env_bool("ENABLE_YOUTUBE", True)
    )
    # Characters of transcript forwarded to the model per video.
    youtube_max_chars: int = field(
        default_factory=lambda: _env_int("YOUTUBE_MAX_CHARS", 30_000)
    )
    # Raw caption JSON read limit before parsing, bytes.
    youtube_max_bytes: int = field(
        default_factory=lambda: _env_int("YOUTUBE_MAX_BYTES", 2_000_000)
    )
    # Total timeout for loading one video's captions, seconds.
    youtube_timeout: int = field(
        default_factory=lambda: _env_int("YOUTUBE_TIMEOUT", 40)
    )
    # Preferred caption languages, best first (the video's own language is the
    # fallback).
    youtube_langs: str = field(
        default_factory=lambda: _env("YOUTUBE_LANGS", default="ru,en")
    )

    # --- Web search (RAG) ---
    # The chat model gets a search_web tool and queries the live internet when
    # a question needs current information; numbered results are fed back and
    # the answer cites them as [n]. DuckDuckGo (the default) needs no key, only
    # the ddgs package; Tavily/Brave/Exa need SEARCH_API_KEY — with no key they
    # are not offered and nothing changes.
    enable_web_search: bool = field(
        default_factory=lambda: _env_bool("ENABLE_WEB_SEARCH", True)
    )
    # "duckduckgo" (https://duckduckgo.com, package ddgs, no key),
    # "tavily" (https://tavily.com), "brave" (Brave Search API) or
    # "exa" (https://exa.ai).
    search_provider: str = field(
        default_factory=lambda: _env("SEARCH_PROVIDER", default="duckduckgo").lower()
    )
    # DuckDuckGo backend inside ddgs: "duckduckgo" (the site's own HTML results,
    # fast) or "auto" / a comma list such as "duckduckgo,bing" (more sources,
    # noticeably slower — html.duckduckgo.com throttles busy IPs with HTTP 202).
    search_ddgs_backend: str = field(
        default_factory=lambda: _env("SEARCH_DDGS_BACKEND", default="duckduckgo").lower()
    )
    # Key for tavily/brave/exa. TAVILY_API_KEY and EXA_API_KEY are read only
    # for their own provider; EXA_API_KEY takes precedence for Exa.
    search_api_key: str = field(default_factory=_search_api_key)
    # Optional endpoint override (empty = the provider's default URL).
    search_api_url: str = field(
        default_factory=lambda: os.getenv("SEARCH_API_URL", "").strip()
    )
    search_timeout: int = field(
        default_factory=lambda: _env_int("SEARCH_TIMEOUT", 15)
    )
    search_max_results: int = field(
        default_factory=lambda: _env_int("SEARCH_MAX_RESULTS", 10)
    )
    # Search rounds allowed per question before the model must answer.
    max_search_rounds: int = field(
        default_factory=lambda: _env_int("MAX_SEARCH_ROUNDS", 2)
    )
    # Total search calls per question across all rounds (a single model turn
    # may ask for several). Each call can read SEARCH_MAX_RESULTS pages.
    search_max_calls: int = field(
        default_factory=lambda: _env_int("SEARCH_MAX_CALLS", 3)
    )
    # Minimum interval between live search checks in the bot status report,
    # seconds (one hour by default). The check itself is a real provider call;
    # a per-user cooldown keeps "статус" from turning into a paid search loop.
    status_search_interval: int = field(
        default_factory=lambda: _env_int("STATUS_SEARCH_INTERVAL", 3600)
    )
    # Сколько AI-подтверждений «статуса» обычный пользователь может сделать за
    # час и за UTC-сутки. Админы/владельцы не ограничены, guest закрыт.
    status_intent_hourly_limit: int = field(
        default_factory=lambda: _env_int("STATUS_INTENT_HOURLY_LIMIT", 5)
    )
    status_intent_daily_limit: int = field(
        default_factory=lambda: _env_int("STATUS_INTENT_DAILY_LIMIT", 24)
    )
    # Total characters of search/MCP output added to the model context for one
    # question. Trimming keeps the untrusted frame intact.
    tool_max_context_chars: int = field(
        default_factory=lambda: _env_int("TOOL_MAX_CONTEXT_CHARS", 60_000)
    )

    # --- MCP (Model Context Protocol) providers ---
    # The chat model gets tools exposed by one or more MCP servers. The main
    # use case is the Obsidian "second brain": search and read personal notes
    # through the ``brain`` server. Чтение открыто всем; запись только OWNER_IDS.
    enable_mcp: bool = field(
        default_factory=lambda: _env_bool("MCP_ENABLED", True)
    )
    # Устаревшая настройка, сохраняется для совместимости старых конфигов.
    mcp_owner_only: bool = field(
        default_factory=lambda: _env_bool(
            "MCP_OWNER_ONLY", _env_bool("MCP_ADMIN_ONLY", False)
        )
    )
    # Claude-style .mcp.json with a "mcpServers" object (path to the file or
    # its directory). Empty — no file is read.
    mcp_config_file: str = field(
        default_factory=lambda: _env("MCP_CONFIG_FILE", "MCP_CONFIG")
    )
    # Optional sha256 of the .mcp.json bytes. When set, a config that changed
    # (e.g. through a synced vault) is ignored instead of executed.
    mcp_config_sha256: str = field(
        default_factory=lambda: _env("MCP_CONFIG_SHA256").lower()
    )
    # A bare command name is resolved through PATH (an easy hijack vector).
    # Absolute paths are required unless this is explicitly enabled.
    mcp_allow_path_command: bool = field(
        default_factory=lambda: _env_bool("MCP_ALLOW_PATH_COMMAND", False)
    )
    # An .mcp.json outside the working directory (e.g. inside a synced vault)
    # is not executed unless its sha256 is pinned or this override is set.
    mcp_allow_unpinned_config: bool = field(
        default_factory=lambda: _env_bool("MCP_ALLOW_UNPINNED_CONFIG", False)
    )
    # Inline JSON alternative to the file; entries here override the file.
    mcp_servers_inline: str = field(
        default_factory=lambda: os.getenv("MCP_SERVERS", "").strip()
    )
    # Parsed server definitions: name -> {command, args, env, cwd?}.
    mcp_servers: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Сервер с create_note; запись идёт отдельно от инструментов модели.
    mcp_write_server: str = field(
        default_factory=lambda: _env("MCP_WRITE_SERVER", default="brain")
    )
    # Per-request timeout for MCP calls (seconds).
    mcp_timeout: int = field(
        default_factory=lambda: _env_int("MCP_TIMEOUT", 60)
    )
    # Tool rounds allowed per question (MCP calls are multi-round).
    mcp_max_rounds: int = field(
        default_factory=lambda: _env_int("MCP_MAX_ROUNDS", 2)
    )
    # Total MCP calls per question across all rounds.
    mcp_max_calls: int = field(
        default_factory=lambda: _env_int("MCP_MAX_CALLS", 4)
    )
    # Which MCP tools the model may see and call (names as the server reports
    # them, comma-separated; "*" = everything the server offers). Servers whose
    # ``.mcp.json`` entry has its own ``allowed_tools`` list use that instead.
    # NOTE: the default is the current read-only set of the Obsidian «second
    # brain» server and may be changed in the future (new read tools, or write
    # tools if they are ever deliberately enabled); anything not listed is
    # hidden from the model and refused even if the server offers it.
    mcp_allowed_tools: str = field(
        default_factory=lambda: _env(
            "MCP_ALLOWED_TOOLS", default="search_notes,get_chunk,get_note,list_notes"
        )
    )
    # Upper bound for one tool result added to the model context.
    mcp_max_result_chars: int = field(
        default_factory=lambda: _env_int("MCP_MAX_RESULT_CHARS", 12_000)
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
    # Free-form reminder requests in ordinary chat text ("напомни завтра
    # позвонить маме") are detected with one small AI intent call and turned
    # into a regular /remind reminder (services/reminder_intent.py); the rest
    # of the chat flow is unaffected and guests never reach the detection.
    reminder_freeform: bool = field(
        default_factory=lambda: _env_bool("REMINDER_FREEFORM", True)
    )
    # Сколько первых символов сообщения просматривает дешёвый префильтр
    # намерений (напоминание / рисование): ключевые слова дальше не ищутся.
    reminder_prefilter_chars: int = field(
        default_factory=lambda: _env_int("REMINDER_PREFILTER_CHARS", 64)
    )
    # Единый классификатор намерений запроса (services/request_intent.py):
    # рисование / напоминание / ссылка / документ. Выключен — прежнее поведение.
    request_intent: bool = field(
        default_factory=lambda: _env_bool("REQUEST_INTENT", True)
    )
    # Максимум символов, уходящих в нейросеть при классификации намерения
    # (оборванное последнее слово отбрасывается).
    request_intent_max_chars: int = field(
        default_factory=lambda: _env_int("REQUEST_INTENT_MAX_CHARS", 256)
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

    # --- Channel digest ---
    # Once a day the bot summarises the messages of the channels chosen in
    # /admin → Digest and posts the summary to the channel itself; admins then
    # get the per-channel token statistics. The model and the prompt are set
    # in the admin menu. The time is local to REMINDER_TIMEZONE.
    enable_digest: bool = field(
        default_factory=lambda: _env_bool("ENABLE_DIGEST", True)
    )
    digest_time: str = field(
        default_factory=lambda: _env("DIGEST_TIME", default="21:00")
    )
    # Fewer messages than this in the window → add older ones as context.
    digest_min_messages: int = field(
        default_factory=lambda: _env_int("DIGEST_MIN_MESSAGES", 20)
    )
    digest_context_messages: int = field(
        default_factory=lambda: _env_int("DIGEST_CONTEXT_MESSAGES", 64)
    )
    # Input cap per channel (characters); the oldest messages are dropped first.
    digest_max_input_chars: int = field(
        default_factory=lambda: _env_int("DIGEST_MAX_INPUT_CHARS", 60000)
    )
    digest_max_tokens: int = field(
        default_factory=lambda: _env_int("DIGEST_MAX_TOKENS", 1500)
    )
    # Sampling temperature of both digest requests (the admin menu overrides it).
    digest_temperature: float = field(
        default_factory=lambda: _env_float("DIGEST_TEMPERATURE", 1.0)
    )
    # Rewrite the digest prompt with the modifier prompt before every run, so
    # the summaries do not read identically day after day.
    digest_vary_prompt: bool = field(
        default_factory=lambda: _env_bool("DIGEST_VARY_PROMPT", True)
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

    # --- Постоянная клавиатура меню ---
    # Reply-клавиатура под полем ввода в личке (модели, персона, картинки,
    # напоминания, экспорт, очистка, настройки, статистика, справка).
    enable_reply_menu: bool = field(
        default_factory=lambda: _env_bool("ENABLE_REPLY_MENU", True)
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

    # --- Logging ---
    # Full message texts in the AI request logs. Off by default: with texts on,
    # the logs become a plaintext archive of private conversations. When
    # enabled, texts are logged at DEBUG level only, with a startup warning.
    log_prompts: bool = field(
        default_factory=lambda: _env_bool("LOG_PROMPTS", False)
    )
    # Rotating file for warnings/errors (post-mortem diagnostics). ``auto``
    # puts ``errors.log`` next to the SQLite database, a directory the service
    # can already write to; ``off`` keeps journald/stdout only. An absolute
    # path overrides the location.
    log_file: str = field(
        default_factory=lambda: _env("LOG_FILE", "auto")
    )
    log_file_level: str = field(
        default_factory=lambda: _env("LOG_FILE_LEVEL", "WARNING")
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
        default_factory=lambda: _env_bool("TELEGRAPH_ENABLED", False)
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
        """Parse owner IDs after initialization."""
        raw = _env("OWNER_IDS", "ADMIN_IDS")
        self.owner_ids = []
        for chunk in raw.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            try:
                self.owner_ids.append(int(chunk))
            except ValueError:
                continue

        if self.parse_mode not in {"markdown", "html", "plain"}:
            self.parse_mode = "markdown"

        normalized_api_type = _normalize_ai_api_type(self.ai_api_type)
        if normalized_api_type is None:
            logging.getLogger(__name__).warning(
                "Invalid AI_API_TYPE %r; using %s",
                self.ai_api_type,
                DEFAULT_AI_API_TYPE,
            )
            normalized_api_type = DEFAULT_AI_API_TYPE
        self.ai_api_type = normalized_api_type

        if self.thinking_level not in THINKING_LEVELS:
            logging.getLogger(__name__).warning(
                "Invalid THINKING_LEVEL %r; using %s",
                self.thinking_level,
                DEFAULT_THINKING_LEVEL,
            )
            self.thinking_level = DEFAULT_THINKING_LEVEL
        self.thinking_budget = max(0, min(self.thinking_budget, MAX_THINKING_BUDGET))

        if self.progress_delay < 0:
            self.progress_delay = 0.0
        # Telegram tolerates roughly one message edit per second per chat.
        self.progress_interval = max(self.progress_interval, 0.5)
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
        self.stream_stall_timeout = max(self.stream_stall_timeout, 10)
        # Network timeouts: 0 means "no timeout" in aiohttp; negative values are
        # undefined. Clamp to a sane range so a typo cannot hang requests forever.
        self.request_timeout = max(5, min(self.request_timeout, 900))
        self.image_timeout = max(10, min(self.image_timeout, 900))
        self.guest_timeout = max(15, min(self.guest_timeout, 900))
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
        self.document_parse_timeout = max(5.0, min(self.document_parse_timeout, 300.0))
        self.max_document_bytes = max(
            1_048_576, min(self.max_document_bytes, 20 * 1024 * 1024)
        )
        self.link_fetch_timeout = max(5, min(self.link_fetch_timeout, 60))
        self.link_crawl_depth = max(0, min(self.link_crawl_depth, 5))
        self.link_crawl_max_chars = max(
            1000, min(self.link_crawl_max_chars, 10_000_000)
        )
        self.link_crawl_max_fetches = max(
            1, min(self.link_crawl_max_fetches, 100)
        )
        self.reminder_prefilter_chars = max(
            16, min(self.reminder_prefilter_chars, 512)
        )
        self.request_intent_max_chars = max(
            32, min(self.request_intent_max_chars, 1024)
        )
        self.photo_intent_daily_limit = max(0, min(self.photo_intent_daily_limit, 10_000))
        self.photo_intent_global_daily_limit = max(
            0, min(self.photo_intent_global_daily_limit, 100_000)
        )
        self.reminder_intent_daily_limit = max(0, min(self.reminder_intent_daily_limit, 10_000))
        self.reminder_intent_global_daily_limit = max(
            0, min(self.reminder_intent_global_daily_limit, 100_000)
        )
        # OCR pages and transcript size bound the paid tokens of one request.
        self.pdf_ocr_max_pages = max(1, min(self.pdf_ocr_max_pages, 20))
        self.youtube_max_chars = max(2_000, min(self.youtube_max_chars, 100_000))
        self.youtube_max_bytes = max(
            256_000, min(self.youtube_max_bytes, 20_000_000)
        )
        self.youtube_timeout = max(10, min(self.youtube_timeout, 120))
        # Web search bounds.
        self.search_timeout = max(5, min(self.search_timeout, 60))
        self.search_max_results = max(1, min(self.search_max_results, 10))
        self.max_search_rounds = max(1, min(self.max_search_rounds, 5))
        self.search_max_calls = max(1, min(self.search_max_calls, 20))
        self.status_search_interval = max(
            60, min(self.status_search_interval, 86_400)
        )
        self.status_intent_hourly_limit = max(
            1, min(self.status_intent_hourly_limit, 60)
        )
        self.status_intent_daily_limit = max(
            1, min(self.status_intent_daily_limit, 240)
        )
        self.tool_max_context_chars = max(
            1_000, min(self.tool_max_context_chars, 200_000)
        )
        # MCP bounds: a hung server must not block an answer forever, and one
        # long note must not flood the model context.
        self.mcp_timeout = max(5, min(self.mcp_timeout, 300))
        self.mcp_max_rounds = max(1, min(self.mcp_max_rounds, 5))
        self.mcp_max_calls = max(1, min(self.mcp_max_calls, 20))
        self.mcp_max_result_chars = max(
            1_000, min(self.mcp_max_result_chars, 50_000)
        )
        self.mcp_servers = _load_mcp_servers(
            self.mcp_servers_inline,
            self.mcp_config_file,
            self.mcp_config_sha256,
            self.mcp_allow_unpinned_config,
        )
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

        self.ai_models = [
            spec
            for slot in range(1, MODEL_SLOTS + 1)
            if (spec := _model_set(f"MODEL_{slot}_", default_vision=False))
        ]
        # Duplicate internal ids would make the user's choice ambiguous.
        seen_ids: set[str] = set()
        self.ai_models = [
            spec
            for spec in self.ai_models
            if not (spec["id"] in seen_ids or seen_ids.add(spec["id"]))
        ]
        if not self.ai_models:
            self.ai_models = [dict(model) for model in DEFAULT_AI_MODELS]
        fallback = _model_set("FALLBACK_MODEL_")
        self.ai_fallback_models = [fallback] if fallback else []
        # Empty URL / key / API type in a set -> the global provider settings.
        for spec in (*self.ai_models, *self.ai_fallback_models):
            spec["base_url"] = spec.get("base_url") or self.ai_base_url.rstrip("/")
            spec["api_key"] = spec.get("api_key") or self.ai_api_key
            raw_api_type = spec.get("api_type") or self.ai_api_type
            normalized_api_type = _normalize_ai_api_type(raw_api_type)
            if normalized_api_type is None:
                logging.getLogger(__name__).warning(
                    "Invalid API_TYPE for model %s: %r; using %s",
                    spec.get("id"),
                    raw_api_type,
                    self.ai_api_type,
                )
                normalized_api_type = self.ai_api_type
            spec["api_type"] = normalized_api_type
            spec["web_search"] = _normalize_web_search(
                spec.get("web_search") or "auto"
            )
            raw_temperature = spec.get("temperature")
            spec["temperature"] = None
            if raw_temperature not in (None, ""):
                try:
                    temperature = float(raw_temperature)
                except (TypeError, ValueError):
                    logging.getLogger(__name__).warning(
                        "Invalid TEMPERATURE for model %s: %r; using default",
                        spec.get("id"),
                        raw_temperature,
                    )
                else:
                    if not math.isfinite(temperature) or not 0 <= temperature <= 2:
                        logging.getLogger(__name__).warning(
                            "TEMPERATURE for model %s is out of range: %r; "
                            "using default",
                            spec.get("id"),
                            raw_temperature,
                        )
                    else:
                        spec["temperature"] = temperature
        if not self.default_model:
            self.default_model = self.ai_models[0]["id"]

        # A deployment that never customized DEFAULT_SYSTEM_PROMPT still has
        # the old, nameless default: upgrade it so fresh users get the named
        # assistant prompt too. Custom values are left untouched.
        if self.default_system_prompt == PREVIOUS_DEFAULT_SYSTEM_PROMPT_TEXT:
            self.default_system_prompt = DEFAULT_SYSTEM_PROMPT_TEXT

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

    def resolve_model(self, model_id: str | None) -> dict[str, Any]:
        """The settings set for a user's internal model id.

        An id without a set (stale choice from the database, removed slot,
        empty value) resolves to the default model's set.
        """
        for spec in (*self.ai_models, *self.ai_fallback_models):
            if spec["id"] == model_id:
                return spec
        for spec in self.ai_models:
            if spec["id"] == self.default_model:
                return spec
        return self.ai_models[0]

    def builtin_web_search(self, model_id: str | None) -> bool:
        """Whether chat requests should use the provider's built-in search.

        ``WEB_SEARCH=true`` forces it, ``false`` disables it. ``auto`` trusts
        the known model families: Qwen and DeepSeek-V4 currently accept
        ``enable_search`` on Model Studio; GLM/Kimi need a separate API.
        """
        if not self.enable_web_search:
            return False
        spec = self.resolve_model(model_id)
        setting = spec.get("web_search") or "auto"
        if setting == "true":
            return True
        if setting == "false":
            return False
        if (spec.get("api_type") or self.ai_api_type) != "dashscope":
            return False
        provider_model = str(spec.get("model") or spec.get("id") or "").lower()
        if "omni" in provider_model:
            # Omni models need search_strategy=agent, which the text endpoint
            # does not accept; use the local provider unless explicitly forced.
            return False
        return provider_model.startswith(("qwen", "deepseek-v4"))

    def is_owner(self, telegram_id: int) -> bool:
        """True for IDs from ``OWNER_IDS``: they can never be demoted or blocked."""
        return telegram_id in self.owner_ids

    def is_admin(self, telegram_id: int) -> bool:
        """True for owners and for admins promoted from the admin menu."""
        return telegram_id in self.owner_ids or telegram_id in self.extra_admin_ids

    def validate(self) -> None:
        """Validate required config values and absolute filesystem paths.

        Every configurable path must be absolute: the service runs from a
        read-only code tree, so a relative SQLite/Telegraph/rembg path would
        resolve inside ``/opt/...`` and fail (or write data next to the code).
        Relative values are rejected at startup instead of being silently
        resolved against the current working directory.
        """
        log = logging.getLogger(__name__)
        for old, new in _RENAMED_ENV.items():
            if os.getenv(old, "").strip() and not os.getenv(new, "").strip():
                log.warning("%s is deprecated, rename it to %s in .env", old, new)
        if not self.owner_ids:
            log.warning(
                "OWNER_IDS is empty or invalid: admin commands are unavailable"
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

        path_errors = self._relative_path_errors()
        if path_errors:
            raise ValueError(
                "Configuration paths must be absolute (relative paths are "
                "rejected because the service's working directory is "
                "read-only). Use the writable state directory, e.g. "
                "/var/lib/kapum-chatbot/:\n- " + "\n- ".join(path_errors)
            )

    def _relative_path_errors(self) -> list[str]:
        """Human-readable errors for configurable paths that are not absolute."""
        errors: list[str] = []
        if self.database_url.startswith("sqlite"):
            try:
                from sqlalchemy.engine import make_url

                database = make_url(self.database_url).database
            except Exception:  # noqa: BLE001 - report instead of crashing blindly
                errors.append("DATABASE_URL is not a valid SQLAlchemy URL")
            else:
                if (
                    database
                    and database != ":memory:"
                    and not os.path.isabs(database)
                ):
                    errors.append(
                        "DATABASE_URL points to a relative SQLite file "
                        f"({database!r}); use four slashes and an absolute "
                        "path, e.g. "
                        "sqlite+aiosqlite:////var/lib/kapum-chatbot/chat_bot.db"
                    )
        for name, value in (
            ("STICKER_MODEL_DIR", self.sticker_model_dir),
            ("TELEGRAPH_ACCOUNT_FILE", self.telegraph_account_file),
            ("MCP_CONFIG_FILE", self.mcp_config_file),
        ):
            if value and not _absolute_path(value):
                errors.append(
                    f"{name} must be an absolute path (got {value!r})"
                )
        if (
            self.log_file.lower() not in {"", "auto", "off", "none"}
            and not _absolute_path(self.log_file)
        ):
            errors.append(
                "LOG_FILE must be 'auto', 'off' or an absolute path "
                f"(got {self.log_file!r})"
            )
        return errors


config = Config()
