"""Единый классификатор намерения запроса: что хочет пользователь.

Метки (:class:`Intent`): обычный вопрос, генерация картинки, напоминание,
анализ документа, анализ сайта. Слои, от дешёвого к дорогому:

1. вложение-документ и ссылка определяются без модели;
2. слова-триггеры в первых ``REMINDER_PREFILTER_CHARS`` символах (напоминание —
   префильтр из ``reminder_intent``, рисование — ``_IMAGE_WORDS``);
3. только при срабатывании триггера рисования один маленький AI-вызов
   подтверждает намерение. В нейросеть уходит не больше
   ``REQUEST_INTENT_MAX_CHARS`` (256) символов, оборванное последнее слово
   отбрасывается (:func:`truncate_for_intent`).

Любой сбой модели → ``Intent.CHAT``: поведение «как раньше». Для напоминаний
окончательное решение и разбор времени остаются за
``reminder_intent.extract_freeform``; здесь только маршрутизация.
"""

from __future__ import annotations

import enum
import logging
import re

from bot.config import config
from bot.services.ai_client import ai_client
from bot.services.reminder_intent import looks_like_reminder

logger = logging.getLogger(__name__)


class Intent(enum.StrEnum):
    """Намерение запроса."""

    CHAT = "chat"
    IMAGE = "image"
    REMINDER = "reminder"
    DOCUMENT = "document"
    WEBSITE = "website"


_IMAGE_WORDS = re.compile(
    r"нарису|рису[йюе]|рисунок|сгенер|картинк|изображени|иллюстраци|логотип"
    r"|стикер|постер|обои|аватар|портрет|как\s+выгляд"
    r"|\bdraw|paint|image|picture|generate|sticker|logo|illustrat|wallpaper",
    re.IGNORECASE,
)

_URL = re.compile(r"https?://\S+", re.IGNORECASE)
# Остаток текста после ссылок: длиннее — ссылка лишь упомянута, а не предмет
# запроса (то же правило, что _LINK_REMAINDER_LIMIT в handlers/chat.py).
_URL_REMAINDER_LIMIT = 300

_IMAGE_SYSTEM_PROMPT = (
    "You classify a user message for a chat bot. Answer with EXACTLY ONE "
    "word:\n"
    "- 'image' — the user asks to DRAW / GENERATE / CREATE a picture, "
    "illustration, logo, wallpaper, avatar or sticker;\n"
    "- 'chat' — anything else: a question about images, a description, "
    "advice, ordinary conversation.\n"
    "Nothing else."
)


def truncate_for_intent(text: str, limit: int | None = None) -> str:
    """Обрезать текст до ``limit`` символов, не оставляя обрубок слова.

    Текст короче лимита возвращается как есть. Иначе берётся ``text[:limit]``;
    если сразу за срезом не пробел (слово оборвано), оборванное слово
    отбрасывается целиком. Единственное слово длиннее лимита режется жёстко.
    """
    limit = limit or config.request_intent_max_chars
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    head = text[:limit]
    if not text[limit].isspace():
        cut = max(head.rfind(" "), head.rfind("\n"), head.rfind("\t"))
        if cut > 0:
            head = head[:cut]
    return head.strip()


def looks_like_image_request(text: str) -> bool:
    """Дешёвый префильтр: слова про рисование в первых символах сообщения."""
    if not text:
        return False
    return bool(_IMAGE_WORDS.search(text[: config.reminder_prefilter_chars]))


def _has_link_request(text: str) -> bool:
    """Ссылка есть, а остального текста мало: предмет запроса — страница."""
    if not _URL.search(text):
        return False
    return len(_URL.sub("", text).strip()) <= _URL_REMAINDER_LIMIT


async def _confirm_image(text: str, model: str | None) -> bool:
    """Один маленький AI-вызов: действительно ли просят нарисовать."""
    try:
        result = await ai_client.chat_completion_result(
            messages=[
                {"role": "system", "content": _IMAGE_SYSTEM_PROMPT},
                {"role": "user", "content": truncate_for_intent(text)},
            ],
            model=model,
            temperature=0.0,
            max_tokens=10,
        )
    except Exception:
        logger.warning("Image intent call failed; treating as chat", exc_info=True)
        return False
    return (result.content or "").strip().lower().startswith("image")


async def classify_request(
    text: str,
    model: str | None = None,
    *,
    has_document: bool = False,
    from_voice: bool = False,
    skip_reminder: bool = False,
) -> Intent:
    """Определить намерение запроса.

    ``from_voice``: распознанная речь может исказить адрес, поэтому ссылки в
    ней сайтом не считаются. ``skip_reminder``: напоминание уже проверено
    (и отвергнуто) отдельно — не предлагать его повторно.
    """
    if has_document:
        return Intent.DOCUMENT
    if not skip_reminder and looks_like_reminder(text):
        return Intent.REMINDER
    if not from_voice and _has_link_request(text):
        return Intent.WEBSITE
    if looks_like_image_request(text) and await _confirm_image(text, model):
        return Intent.IMAGE
    return Intent.CHAT
