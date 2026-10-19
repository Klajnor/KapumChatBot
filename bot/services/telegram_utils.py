from __future__ import annotations

import logging
import re

from aiogram.exceptions import TelegramAPIError, TelegramBadRequest
from aiogram.types import CallbackQuery, Message

from bot.bot import get_bot_id, get_bot_username
from bot.config import config
from bot.services.markdown import render, split_chunks
from bot.services.science import normalize_formulas

logger = logging.getLogger(__name__)

# Mapping of our internal mode names to Telegram parse modes.
PARSE_MODES = {"markdown": "MarkdownV2", "html": "HTML"}


async def safe_answer_callback(
    callback: CallbackQuery,
    text: str | None = None,
    *,
    show_alert: bool = False,
    **kwargs,
) -> bool:
    """Answer a callback query, tolerating expired/invalid queries.

    Telegram rejects ``answerCallbackQuery`` when the query is too old (for
    example when the user tapped a button long ago, or tapped several times).
    Such failures must never crash the handler.
    """
    try:
        await callback.answer(text, show_alert=show_alert, **kwargs)
        return True
    except TelegramBadRequest as exc:
        logger.warning("Callback answer failed (expired?): %s", exc)
        return False


async def safe_edit_text(message: Message | None, text: str, **kwargs) -> bool:
    """Edit a message, tolerating 'not modified' / expired / inaccessible errors."""
    if message is None or not hasattr(message, "edit_text"):
        return False
    try:
        await message.edit_text(text, **kwargs)
        return True
    except TelegramBadRequest as exc:
        logger.warning("Message edit failed: %s", exc)
        return False
    except Exception as exc:  # noqa: BLE001 - UI helper must never crash a handler
        logger.warning("Message edit failed: %s", exc)
        return False


# --- Bot mentions ----------------------------------------------------------


def strip_bot_mention(text: str | None) -> str:
    """Remove ``@bot_username`` occurrences from a message text."""
    if not text:
        return ""
    username = get_bot_username()
    if username:
        text = re.sub(rf"@{re.escape(username)}", "", text, flags=re.IGNORECASE)
    return text.strip()


def mentions_bot(text: str | None) -> bool:
    """Return True if the text mentions the bot by ``@username``."""
    if not text:
        return False
    username = get_bot_username()
    return bool(username) and f"@{username.lower()}" in text.lower()


def is_reply_to_bot(message: Message) -> bool:
    """Return True if the message replies to one of the bot's own messages."""
    reply = message.reply_to_message
    bot_id = get_bot_id()
    return (
        reply is not None
        and reply.from_user is not None
        and bot_id != 0
        and reply.from_user.id == bot_id
    )


# --- Guest mode ------------------------------------------------------------


async def safe_answer_guest(message: Message, result) -> bool:
    """Reply to a guest-mode summon, tolerating expired/invalid queries.

    A guest query allows exactly **one** reply, so callers must invoke this a
    single time. Failures are logged and never raised.
    """
    guest_query_id = getattr(message, "guest_query_id", None)
    if not guest_query_id:
        logger.warning("Guest message without guest_query_id; cannot reply")
        return False
    try:
        await message.bot.answer_guest_query(
            guest_query_id=guest_query_id, result=result
        )
        return True
    except TelegramAPIError as exc:
        logger.warning("Guest query answer failed: %s", exc)
        return False


# --- Formatted sending -----------------------------------------------------


async def send_formatted(message: Message, text: str, *, limit: int = 4000) -> bool:
    """Send an AI answer rendered from Markdown, splitting long replies.

    The configured parse mode is tried first; if Telegram rejects the entities,
    the same chunk is re-sent as HTML and finally as plain text, so a reply is
    never lost because of formatting.
    """
    # Scientific notation is normalized when the answer arrives from the model;
    # doing it again here is a cheap, idempotent guarantee that nothing raw
    # reaches the user even if a caller bypassed the AI client.
    if config.normalize_formulas:
        text = normalize_formulas(text) or text

    chunks = split_chunks(text, config.parse_mode, limit)
    if not chunks:
        return False

    modes = [config.parse_mode]
    for extra in ("html", "plain"):
        if extra not in modes:
            modes.append(extra)

    total = len(chunks)
    for index, raw in enumerate(chunks, start=1):
        body = f"(Part {index}/{total})\n\n{raw}" if total > 1 else raw
        delivered = False
        for mode in modes:
            try:
                await message.answer(
                    render(body, mode), parse_mode=PARSE_MODES.get(mode)
                )
                delivered = True
                break
            except TelegramBadRequest as exc:
                logger.warning(
                    "Sending with parse_mode=%s failed, trying next: %s",
                    PARSE_MODES.get(mode) or "plain",
                    exc,
                )
        if not delivered:
            return False
    return True