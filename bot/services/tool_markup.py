"""Очистка текстовых вызовов инструментов из ответа модели.

Некоторые модели вместо нативного ``tool_calls`` пишут вызов инструмента
текстом: оборачивают имя инструмента в теги (search_notes, tool_call,
function_call) или присылают DSML-конверт. Бот исполняет только
структурированные вызовы, поэтому такая разметка иначе уходит пользователю
как обычный ответ. :func:`strip_tool_markup` вырезает её из готового текста.

Функция намеренно узкая: удаляются только известные имена инструментов бота,
обёртки tool/function-call и DSML. Код в тройных и одинарных backtick-блоках
не трогается, чтобы не портить ответы про HTML/XML.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Добавляется к системному промпту, когда модели предложены инструменты.
# Текстовые вызовы не исполняются, поэтому модель должна пользоваться
# штатным function calling, а не «рисовать» вызов в ответе.
TOOL_CALLS_HINT = (
    "Инструменты вызывай только штатным механизмом function calling "
    "(структурированное поле tool_calls), а не текстом. Не пиши в ответе "
    "вызовы инструментов в тегах (search_notes, tool_call, function_call) "
    "и не присылай DSML: такой текст не исполняется, а попадает пользователю. "
    "Если нужного инструмента нет, ответь обычным текстом."
)

# Имена инструментов, которые бот может предложить модели. Новый инструмент
# стоит добавить сюда: иначе его текстовая форма не будет вырезана.
_TOOL_NAMES = (
    "search_web",
    "search_notes",
    "get_chunk",
    "get_note",
    "list_notes",
    "generate_image",
    "generate_sticker",
    "send_files",
)

# Обёртки, в которые модели заворачивают вызовы (Anthropic, DeepSeek, Qwen).
_ENVELOPE_NAMES = (
    "tool_call",
    "tool_calls",
    "function_call",
    "function_calls",
    "invoke",
    "antml:invoke",
    "calls",
    "parameter",
    "antml:parameter",
)

# ``\x3c`` в шаблонах — это символ «<»; re раскрывает escape.
_TOOL_PATTERN = (
    r"(?:mcp__[A-Za-z0-9_-]+|"
    + "|".join(re.escape(name) for name in _TOOL_NAMES)
    + r")"
)
_ENVELOPE_PATTERN = (
    r"(?:" + "|".join(re.escape(name) for name in _ENVELOPE_NAMES) + r")"
)

# Парные блоки: тег инструмента с содержимым и обёртка tool/function-call.
# Содержимое — это аргументы вызова (query/path), а не ответ, поэтому блок
# удаляется целиком.
_PAIRED_ENVELOPE_RE = re.compile(
    rf"\x3c({_ENVELOPE_PATTERN})\b[^>]*>.*?\x3c/\1\s*>",
    re.DOTALL | re.IGNORECASE,
)
_PAIRED_TOOL_RE = re.compile(
    rf"\x3c({_TOOL_PATTERN})\b[^>]*>.*?\x3c/\1\s*>",
    re.DOTALL | re.IGNORECASE,
)

# Одиночные теги: само-закрывающиеся, открывающие без пары и «осиротевшие»
# закрывающие (именно так выглядел реальный случай после рендера Telegram).
_STRAY_RE = re.compile(
    rf"\x3c/?(?:{_TOOL_PATTERN}|{_ENVELOPE_PATTERN})\b[^>]*?/?>",
    re.IGNORECASE,
)

# DSML DeepSeek: теги вида <｜DSML｜calls>, <｜DSML｜invoke name="…">.
_DSML_START_RE = re.compile(r"\x3c[|｜]{1,2}DSML[|｜]{1,2}", re.IGNORECASE)
_DSML_TAG_RE = re.compile(
    r"\x3c(/?)[|｜]{1,2}DSML[|｜]{1,2}\s*([A-Za-z_][\w-]*)",
    re.IGNORECASE,
)

# Код не чистим: ответ может объяснять HTML/XML и приводить такие теги
# как пример. Разделитель сохраняет чётные сегменты (обычный текст) и
# нечётные (код) на своих местах.
_CODE_SPLIT_RE = re.compile(
    r"(```.*?```|~~~.*?~~~|`[^`\n]+`)",
    re.DOTALL,
)


def _strip_segment(segment: str) -> str:
    """Remove text-form tool calls from one non-code text segment."""
    if "\x3c" not in segment:
        return segment
    original = segment
    segment = _DSML_TAG_RE.sub(
        lambda match: "\x3c" + match.group(1) + match.group(2), segment
    )
    segment = _PAIRED_ENVELOPE_RE.sub("", segment)
    segment = _PAIRED_TOOL_RE.sub("", segment)
    segment = _STRAY_RE.sub("", segment)
    if segment == original:
        # A lone «less than» in prose is not a tool call; keep the layout.
        return segment
    return re.sub(r"\n{3,}", "\n\n", segment)


def strip_tool_markup(text: str | None) -> str | None:
    """Remove text-form tool calls from an answer.

    Returns ``None`` when nothing but such markup was left: an answer made
    only of ``search_notes``/DSML calls carries no useful text, and the caller
    may then try the next model. Plain answers are returned untouched.
    """
    if not text or "\x3c" not in text:
        return text
    if _DSML_START_RE.match(text.lstrip()):
        # Неподдержанный/повреждённый DSML не является ответом. Восстановление
        # search_web делает _recover_search_dsml; всё остальное — мусор.
        logger.warning("Dropped an unsupported text-form tool call (DSML)")
        return None

    parts = _CODE_SPLIT_RE.split(text)
    for index in range(0, len(parts), 2):
        parts[index] = _strip_segment(parts[index])
    cleaned = "".join(parts).strip()
    if cleaned == text.strip():
        return text
    if not cleaned:
        logger.info("Dropped an answer made only of text-form tool calls")
        return None
    logger.info("Stripped text-form tool-call markup from the answer")
    return cleaned
