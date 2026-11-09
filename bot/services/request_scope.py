"""Классификация новых запросов при изоляции недоверенной истории.

``search.isolate_search_history`` и ``mcp.isolate_mcp_history`` вырезают прежнюю
переписку из контекста, чтобы внешний материал не попал в аргументы
инструментов. Изоляцию нельзя применять к уточнениям: модель потеряет сам
документ, о котором её спрашивают. Оба вида изоляции должны одинаково понимать,
что считается новым самостоятельным запросом, и одинаково распознавать явные
указания на интернет и на второй мозг — иначе один и тот же текст маршрутизируется
по-разному в зависимости от порядка вызовов.

Модуль не зависит от конфигурации, БД и Telegram-объектов: это чистые строки.
"""

from __future__ import annotations

import re
from typing import Literal

Route = Literal["web", "notes", "general"]

# Дейксис/анафора: сообщение явно ссылается на прежний контекст. Такие запросы
# изолировать нельзя — им нужна история для ответа, поэтому для них остаётся
# обычный защитный гейт (инструменты не предлагаются).
_FOLLOWUP_RE = re.compile(
    r"\b(?:"
    r"(?:ещё|еще)\s+по\s+этой\s+теме|по\s+этой\s+теме|"
    r"второй\s+пункт|первый\s+пункт|этот\s+пункт|эту\s+часть|этот\s+раздел|"
    r"предыдущ\w*|вышеуказанн\w*|вышеупомянут\w*|сказанн\w*\s+выше|"
    r"об\s+этом|про\s+это|продолжи|продолжай|"
    r"этот\s+(?:документ|файл|текст)|эта\s+страница|из\s+него|из\s+неё|"
    r"tell\s+me\s+more|more\s+about\s+(?:this|it)|what\s+about\s+(?:this|it)|"
    r"the\s+(?:document|file|page)\b|as\s+(?:i|we)\s+(?:said|mentioned)"
    r")\b",
    re.IGNORECASE,
)

# Самостоятельный запрос: вопрос, действие или развёрнутая просьба. Список
# намеренно шире, чем «глагол + слово интернет»: новый вопрос про погоду или
# курс тоже должен получить изолированный контекст вместо тихого отключения
# поиска из-за старого документа в истории.
_FRESH_RE = re.compile(
    r"\b(?:найди(?:те)?|найти|поищи(?:те)?|ищи(?:те)?|поиск\w*|"
    r"проверь(?:те)?|узнай(?:те)?|посмотри(?:те)?|покажи(?:те)?|"
    r"прочитай(?:те)?|открой(?:те)?|расскажи(?:те)?|объясни(?:те)?|"
    r"сравни(?:те)?|подскажи(?:те)?|посоветуй(?:те)?|переведи(?:те)?|"
    r"что|кто|где|когда|какой|какая|какие|какое|как|почему|зачем|сколько|"
    r"актуальн\w*|сегодня|сейчас|последн\w*|свеж\w*|новост\w*|"
    r"курс|погод\w*|цен\w*|верси\w*|"
    r"search|find|look\s+up|check|show|read|list|tell|explain|compare|"
    r"what|who|where|when|which|why|how|is|are|does|do|can|could|"
    r"current|latest|today|news)\b",
    re.IGNORECASE,
)

# Явное указание на интернет.
WEB_MARKER_RE = re.compile(
    r"\b(?:интернет\w*|сети|онлайн|гугл\w*|online|internet|web)\b",
    re.IGNORECASE,
)

# Явное указание на заметки / второй мозг (русские и английские формулировки).
NOTES_TARGET_RE = re.compile(
    r"\b(?:во?\s+(?:(?:мо[её]м|тво[её]м|сво[её]м)\s+)?втором\s+мозге|"
    r"(?:в|из)\s+(?:(?:моих|твоих|своих|сохран[её]нных|личных)\s+)?замет(?:ках|ок)|"
    r"по\s+(?:(?:моим|твоим|своим|сохран[её]нным|личным)\s+)?заметкам|"
    r"в\s+obsidian|(?:in|from|through)\s+(?:(?:my|your|the|our|saved)\s+)?"
    r"(?:second\s+brain|notes|obsidian|vault)|"
    r"(?:my|your|our|saved)\s+(?:second\s+brain|notes|vault))\b",
    re.IGNORECASE,
)

# Короткие реплики («привет», «спасибо», «hi») не считаем новым запросом:
# изолировать из-за них всю переписку не нужно.
_MIN_FRESH_CHARS = 30


def is_fresh_request(text: str | None) -> bool:
    """True для нового самостоятельного запроса, не уточнения к материалу."""
    if not isinstance(text, str) or not text.strip():
        return False
    stripped = text.strip()
    if _FOLLOWUP_RE.search(stripped):
        return False
    return bool(
        "?" in stripped
        or _FRESH_RE.search(stripped)
        or WEB_MARKER_RE.search(stripped)
        or NOTES_TARGET_RE.search(stripped)
        or len(stripped) >= _MIN_FRESH_CHARS
    )


def first_web_at(text: str) -> int | None:
    """Позиция первого явного указания на интернет, если оно есть."""
    match = WEB_MARKER_RE.search(text)
    return match.start() if match else None


def first_notes_at(text: str) -> int | None:
    """Позиция первого явного указания на заметки, если оно есть."""
    match = NOTES_TARGET_RE.search(text)
    return match.start() if match else None


def has_notes_target(text: str) -> bool:
    """True, когда запрос явно адресован заметкам/второму мозгу."""
    return first_notes_at(text) is not None


def route(text: str) -> Route | None:
    """Куда адресован новый запрос: ``web``, ``notes`` или ``general``.

    ``None`` означает, что запрос не самостоятельный (уточнение/пусто) и
    историю изолировать нельзя. Если в тексте есть оба указания, побеждает то,
    которое стоит раньше: «найди в интернете … в Obsidian» — веб-поиск, а
    «найди во втором мозге … про интернет» — заметки.
    """
    if not is_fresh_request(text):
        return None
    web_at, notes_at = first_web_at(text), first_notes_at(text)
    if web_at is not None and notes_at is not None:
        return "web" if web_at < notes_at else "notes"
    if notes_at is not None:
        return "notes"
    if web_at is not None:
        return "web"
    return "general"
