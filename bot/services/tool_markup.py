"""Очистка текстовых вызовов инструментов из ответа модели.

Некоторые модели вместо нативного ``tool_calls`` пишут вызов инструмента
текстом: JSON name/arguments, теги (search_notes, tool_call, function_call)
или DSML-конверт. Бот исполняет только
структурированные вызовы, поэтому такая разметка иначе уходит пользователю
как обычный ответ. :func:`strip_tool_markup` вырезает её из готового текста.

Функция намеренно узкая: удаляются только известные имена инструментов бота,
обёртки tool/function-call и DSML. Код в тройных и одинарных backtick-блоках
не трогается, чтобы не портить ответы про HTML/XML.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Добавляется к системному промпту, когда модели предложены инструменты.
# Текстовые вызовы не исполняются, поэтому модель должна пользоваться
# штатным function calling, а не «рисовать» вызов в ответе.
TOOL_CALLS_HINT = (
    "Инструменты вызывай только штатным механизмом function calling "
    "(структурированное поле tool_calls), а не текстом. Не пиши в ответе "
    "вызовы инструментов в тегах (search_notes, tool_call, function_call) "
    "или JSON (name/arguments), не присылай DSML: такой текст не исполняется. "
    "Веб-поиск называется search_web, а не web_search. "
    "Если нужного инструмента нет, ответь обычным текстом."
)

TOOL_CALLS_RETRY_HINT = (
    "Предыдущая попытка вернула текстовую имитацию вызова инструмента, а не "
    "результат работы. Повтори запрос через штатное поле tool_calls, используя "
    "только инструменты из переданного tools. Не печатай JSON с name/arguments "
    "в content и не ограничивайся обещанием выполнить задачу. Инструмент "
    "веб-поиска называется search_web, не web_search."
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

_JSON_TOOL_START_RE = re.compile(r'\{\s*"(?:name|function|tool_calls|id|type)"\s*:')
_JSON_VALUE_START_RE = re.compile(r"[{[]")
_JSON_DECODER = json.JSONDecoder()
_JSON_STREAM_START_RE = re.compile(r'\{\s*(?:$|")')


def _is_json_tool_call(value: Any, allowed_names: set[str] | None) -> bool:
    """Распознать известный конверт вызова, не исполняя его аргументы."""
    if isinstance(value, list):
        return bool(value) and all(
            _is_json_tool_call(item, allowed_names) for item in value
        )
    if not isinstance(value, dict):
        return False
    if set(value) == {"tool_calls"}:
        calls = value["tool_calls"]
        return isinstance(calls, list) and bool(calls) and all(
            _is_json_tool_call(call, allowed_names) for call in calls
        )
    if "function" in value and set(value) <= {"id", "index", "type", "function"}:
        value = value["function"]
        if not isinstance(value, dict):
            return False
    name = value.get("name")
    return (
        set(value) == {"name", "arguments"}
        and isinstance(name, str)
        and (name in (*_TOOL_NAMES, "web_search") or name.startswith("mcp__"))
        and (allowed_names is None or ("search_web" if name == "web_search" else name) in allowed_names)
        and isinstance(value["arguments"], (dict, str))
    )


def has_json_tool_calls(text: str | None, *, allowed_names: set[str] | None = None) -> bool:
    """Найти JSON-имитацию вызова вне примеров кода; ничего не исполнять."""
    if not text or "{" not in text:
        return False
    for segment in _CODE_SPLIT_RE.split(text)[::2]:
        end = 0
        for match in _JSON_TOOL_START_RE.finditer(segment):
            if match.start() < end:
                continue
            try:
                value, end = _JSON_DECODER.raw_decode(segment, match.start())
                if _is_json_tool_call(value, allowed_names):
                    return True
            except (ValueError, RecursionError):
                continue
    return False


def _strip_json_tool_calls(text: str) -> str | None:
    """Удалить только сами JSON-вызовы, сохранив остальной текст ответа.

    Используется когда инструменты модели не предлагались: текстовая имитация
    тогда не является нарушением протокола, но показывать её пользователю
    всё равно не стоит. Аргументы не разбираются и не исполняются.
    """
    parts = _CODE_SPLIT_RE.split(text)
    for index in range(0, len(parts), 2):
        segment = parts[index]
        kept: list[str] = []
        end = 0
        for match in _JSON_VALUE_START_RE.finditer(segment):
            if match.start() < end:
                continue
            try:
                value, stop = _JSON_DECODER.raw_decode(segment, match.start())
            except (ValueError, RecursionError):
                continue
            if not _is_json_tool_call(value, None):
                continue
            kept.append(segment[end:match.start()])
            end = stop
        kept.append(segment[end:])
        parts[index] = "".join(kept)
    cleaned = "".join(parts).strip()
    return cleaned or None


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


def strip_tool_markup(
    text: str | None, *, drop_json_calls: bool = False,
) -> str | None:
    """Remove text-form tool calls from an answer.

    ``drop_json_calls=True`` ставится только там, где модели были предложены
    инструменты: JSON-имитация вне кода тогда делает весь ответ невалидным,
    включая обещание «начинаю поиск», и AIClient повторяет запрос штатным
    протоколом. В обычном чате без инструментов имитация не нарушает протокол,
    поэтому удаляется только сам JSON — остальной ответ сохраняется.

    Returns ``None`` when nothing but such markup was left: an answer made
    only of ``search_notes``/DSML calls carries no useful text, and the caller
    may then try the next model. Plain answers are returned untouched.
    """
    if has_json_tool_calls(text):
        if drop_json_calls:
            # Обещание «запускаю поиск» тоже не результат: оставив его, мы не
            # запустим fallback и выдадим обещание вместо выполненной задачи.
            logger.warning("Dropped an answer containing text-form JSON tool calls")
            return None
        cleaned = _strip_json_tool_calls(text or "")
        logger.info("Removed text-form JSON tool calls, kept the rest of the answer")
        if not cleaned:
            return None
        text = cleaned
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


def tool_stream_preview(text: str) -> str | None:
    """Придержать JSON вне кода до конца ответа, не показывая частичный вызов."""
    # Предпросмотр относится к путям с инструментами: полную имитацию здесь
    # показывать нельзя (см. drop_json_calls в strip_tool_markup).
    cleaned = strip_tool_markup(text, drop_json_calls=True)
    if not cleaned:
        return None
    parts = _CODE_SPLIT_RE.split(cleaned)
    for index in range(0, len(parts), 2):
        start = _JSON_STREAM_START_RE.search(parts[index])
        if start:
            return ("".join(parts[:index]) + parts[index][:start.start()]).rstrip() or None
    return cleaned
