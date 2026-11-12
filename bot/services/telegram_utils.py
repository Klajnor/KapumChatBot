from __future__ import annotations

import asyncio
import logging
import re
import time
from string import Formatter
from typing import Any

from aiogram.enums import ChatType
from aiogram.exceptions import (
    TelegramAPIError,
    TelegramBadRequest,
    TelegramForbiddenError,
    TelegramRetryAfter,
)
from aiogram.types import BufferedInputFile, CallbackQuery, Message

from bot.bot import get_bot_id, get_bot_username
from bot.config import DEFAULT_TIMING_FORMAT, DEFAULT_TIMING_FORMAT_TOKENS, config
from bot.services.formulas import (
    MAX_FORMULA_IMAGES,
    MAX_TABLE_IMAGES,
    promote_complex_inline,
    render_formula,
    render_table,
    split_blocks,
)
from bot.services.markdown import render, split_chunks
from bot.services.response_metadata import ResponseMetadata
from bot.services.science import normalize_formulas

logger = logging.getLogger(__name__)

# Mapping of our internal mode names to Telegram parse modes.
PARSE_MODES = {"markdown": "MarkdownV2", "html": "HTML"}

# Telegram allows roughly one message per second per chat (and ~20/min in a
# group). Sending chunks — or a progress notice plus the answer — faster than
# that produces 429, which used to make replies disappear. The interval is
# configurable (SEND_MIN_INTERVAL, default 1.05 s; 0 disables the throttle).
_MAX_429_RETRIES = 3


class SendThrottle:
    """Serialize outgoing messages per chat to stay under Telegram's limit."""

    def __init__(self, interval: float | None = None) -> None:
        self.interval = (
            config.send_min_interval if interval is None else float(interval)
        )
        self._locks: dict[int, asyncio.Lock] = {}
        self._last: dict[int, float] = {}

    def _lock(self, chat_id: int) -> asyncio.Lock:
        lock = self._locks.get(chat_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[chat_id] = lock
        return lock

    async def wait(self, chat_id: int) -> None:
        """Sleep until at least ``interval`` has passed since the last send."""
        async with self._lock(chat_id):
            delay = self._last.get(chat_id, 0.0) + self.interval - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            self._last[chat_id] = time.monotonic()
            if len(self._last) > 1000:
                self._prune()

    def _prune(self) -> None:
        """Forget chats that have been idle for a while."""
        cutoff = time.monotonic() - 600
        stale = [chat for chat, stamp in self._last.items() if stamp < cutoff]
        for chat in stale:
            self._last.pop(chat, None)
            self._locks.pop(chat, None)


throttle = SendThrottle()


def _retry_delay(exc: Exception) -> float:
    """How long to wait after a 429, from Telegram's ``retry_after``."""
    value = getattr(exc, "retry_after", 1) or 1
    try:
        return max(0.0, float(value)) + 0.1
    except (TypeError, ValueError):
        return 1.1


# --- Timing footer ---------------------------------------------------------


def _model_label(model: str | None) -> str:
    """Display name of a configured model id; any other value is shown as is."""
    if not model:
        return config.resolve_model(None).get("name") or config.default_model
    for spec in (*config.ai_models, *config.ai_fallback_models):
        if spec["id"] == model:
            return spec.get("name") or model
    return model


def format_timing(
    elapsed: float | None,
    *,
    model: str | None = None,
    tokens: int | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    metadata: ResponseMetadata | None = None,
) -> str:
    """Footer with the AI API processing time, or "" when it should not show.

    ``TIMING_FORMAT`` supports ``{seconds}``, ``{model}`` and ``{tokens}``.
    When the API reported token usage (any of ``tokens``/``input_tokens``/
    ``output_tokens`` is known), ``TIMING_FORMAT_TOKENS`` is used instead:
    it additionally supports ``{input}`` and ``{output}``. Requests without
    usage counters (e.g. image generation) keep the plain format. ``{search}``
    и ``{reasoning}`` доступны в обоих шаблонах. Если они не указаны, известные
    метрики дописываются автоматически; неизвестный reasoning не показывается.
    """
    if not config.show_timing or not elapsed or elapsed <= 0:
        return ""
    has_usage = (
        tokens is not None or input_tokens is not None or output_tokens is not None
        or (metadata is not None and metadata.reasoning_tokens is not None)
    )
    if has_usage:
        template = config.timing_format_tokens or DEFAULT_TIMING_FORMAT_TOKENS
    else:
        template = config.timing_format or DEFAULT_TIMING_FORMAT
    values = {
        "seconds": f"{elapsed:.1f}",
        "model": _model_label(model),
        "tokens": tokens if tokens is not None else "—",
        "input": input_tokens if input_tokens is not None else "—",
        "output": output_tokens if output_tokens is not None else "—",
        "search": _search_label(metadata),
        "reasoning": (
            (f"≥{metadata.reasoning_tokens}" if metadata.reasoning_partial else metadata.reasoning_tokens)
            if metadata is not None and metadata.reasoning_tokens is not None else "—"
        ),
    }
    try:
        rendered = template.format(**values)
        fields = {name for _, name, _, _ in Formatter().parse(template)}
    except (KeyError, IndexError, ValueError):
        logger.warning("Invalid TIMING_FORMAT %r; using the default", template)
        fallback = (
            DEFAULT_TIMING_FORMAT_TOKENS if has_usage else DEFAULT_TIMING_FORMAT
        )
        rendered = fallback.format(**values)
        fields = {name for _, name, _, _ in Formatter().parse(fallback)}
    if metadata is not None:
        if "search" not in fields:
            rendered += f" · 🔎 {values['search']}"
        if metadata.reasoning_tokens is not None and "reasoning" not in fields:
            rendered += f" · 🧠 {values['reasoning']} tok"
    return rendered


def _search_label(metadata: ResponseMetadata | None) -> str:
    """Не выдавать разрешённый, но неподтверждённый поиск за выполненный."""
    if metadata is None:
        return "unknown"
    if metadata.model_search_used is True:
        return "model + API" if metadata.api_search_used else "model"
    if metadata.model_search_used is None:
        return "API; model unknown" if metadata.api_search_used else "unknown"
    return "API" if metadata.api_search_used else "none"


def with_timing(text: str, footer: str) -> str:
    """Append the timing footer to an answer body."""
    if not footer:
        return text
    return f"{text}\n\n{footer}" if text else footer


# --- Safe answers ----------------------------------------------------------


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
        if "not modified" in str(exc).lower():
            return True  # the text is already exactly what we wanted
        logger.warning("Message edit failed: %s", exc)
        return False
    except Exception as exc:  # noqa: BLE001 - UI helper must never crash a handler
        logger.warning("Message edit failed: %s", exc)
        return False


async def safe_delete(message: Message | None) -> bool:
    """Delete a message, tolerating 'too old' / missing-access errors."""
    if message is None or not hasattr(message, "delete"):
        return False
    try:
        await message.delete()
        return True
    except Exception as exc:  # noqa: BLE001 - cosmetic, must never break a reply
        logger.debug("Message delete failed: %s", exc)
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
    reply = getattr(message, "reply_to_message", None)
    bot_id = get_bot_id()
    author = getattr(reply, "from_user", None)
    if (
        reply is not None
        and author is not None
        and bot_id != 0
        and getattr(author, "id", None) == bot_id
    ):
        return True
    # Reply из другого топика/чата может прийти без reply_to_message. Только
    # проверенный Telegram user id происхождения, никогда имя/текст цитаты.
    external = getattr(message, "external_reply", None)
    origin = getattr(external, "origin", None)
    author = getattr(origin, "sender_user", None)
    return bot_id != 0 and author is not None and getattr(author, "id", None) == bot_id


def is_forwarded(message: Message) -> bool:
    """True for a forwarded message: its text was written by somebody else."""
    return (
        getattr(message, "forward_origin", None) is not None
        or getattr(message, "forward_date", None) is not None
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


async def send_formatted(
    message: Message, text: str, *, limit: int = 4000, reply_markup: Any = None
) -> bool:
    """Send text and display formulas in their original order.

    ``reply_markup`` (when given) is attached to the last text chunk only.
    """
    if not config.render_formula_images:
        return await _send_text_formatted(
            message, text, limit=limit, reply_markup=reply_markup
        )

    delivered = True
    sent_any = False
    image_count = 0
    table_count = 0
    parts = list(split_blocks(promote_complex_inline(text)))
    last_text_index = -1
    for index, (kind, body) in enumerate(parts):
        if body.strip() and kind == "text":
            last_text_index = index

    for index, (kind, body) in enumerate(parts):
        if not body.strip():
            continue
        if kind == "table":
            images = None
            if table_count < MAX_TABLE_IMAGES:
                table_count += 1
                images = await render_table(body)
            if images and await _send_table_photos(message, images):
                sent_any = True
                continue
            # Rendering failed: keep the table readable as a plain list.
            body = _table_as_list(body)
            markup = reply_markup if index == last_text_index else None
            delivered = (
                await _send_text_formatted(message, body, limit=limit, reply_markup=markup)
                and delivered
            )
            sent_any = True
            continue
        image = None
        if kind == "math" and image_count < MAX_FORMULA_IMAGES:
            image_count += 1
            image = await render_formula(body)
        if image is not None and await _send_formula_photo(message, image):
            sent_any = True
            continue
        # Unsupported math and failed uploads retain the existing text fallback.
        if kind == "math":
            body = normalize_formulas(body) or body
        markup = reply_markup if index == last_text_index else None
        delivered = (
            await _send_text_formatted(message, body, limit=limit, reply_markup=markup)
            and delivered
        )
        sent_any = True
    return delivered and sent_any


async def _send_table_photos(message: Message, images: list[bytes]) -> bool:
    """Send every table image; True only if all of them were delivered."""
    for number, image in enumerate(images, start=1):
        if not await _send_formula_photo(message, image, filename=f"table{number}.png"):
            return False
    return True


def _table_as_list(table: str) -> str:
    """Markdown table -> bullet list (fallback when the picture failed)."""
    lines = table.splitlines()
    header = [c.strip() for c in lines[0].strip().strip("|").split("|")]
    out = []
    for line in lines[2:]:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        pairs = [
            f"{title}: {value}" if title else value
            for title, value in zip(header[1:], cells[1:])
            if value
        ]
        out.append(f"- **{cells[0]}**" + (" — " + "; ".join(pairs) if pairs else ""))
    return "\n".join(out)


async def _send_formula_photo(
    message: Message, image: bytes, filename: str = "formula.png"
) -> bool:
    for attempt in range(_MAX_429_RETRIES + 1):
        try:
            await throttle.wait(message.chat.id)
            await message.answer_photo(
                BufferedInputFile(image, filename=filename)
            )
            return True
        except TelegramRetryAfter as exc:
            if attempt >= _MAX_429_RETRIES:
                return False
            await asyncio.sleep(_retry_delay(exc))
        except Exception as exc:  # noqa: BLE001 - image delivery falls back to text
            logger.warning("Could not send formula image; using text: %s", exc)
            return False
    return False


async def _send_text_formatted(
    message: Message, text: str, *, limit: int = 4000, reply_markup: Any = None
) -> bool:
    """Send an AI answer rendered from Markdown, splitting long replies.

    Delivery is retried and degraded rather than lost:

    * the configured parse mode is tried first, then HTML, then plain text;
    * ``429`` (rate limit) waits ``retry_after`` and retries the same variant;
    * a chunk rejected as "too long" is truncated and re-sent as plain text;
    * sending is throttled to Telegram's ~1 message/second per chat.

    Returns True only when every chunk was delivered.
    """
    # Scientific notation is normalized when the answer arrives from the model;
    # doing it again here is a cheap, idempotent guarantee that nothing raw
    # reaches the user even if a caller bypassed the AI client.
    if config.normalize_formulas:
        text = normalize_formulas(text) or text

    chunks = split_chunks(text, config.parse_mode, limit)
    if not chunks:
        logger.warning("Nothing to send: the answer rendered to an empty string")
        return False

    modes = [config.parse_mode]
    for extra in ("html", "plain"):
        if extra not in modes:
            modes.append(extra)

    total = len(chunks)
    delivered_all = True
    for index, raw in enumerate(chunks, start=1):
        body = f"(Part {index}/{total})\n\n{raw}" if total > 1 else raw
        markup = reply_markup if index == total else None
        if not await _send_chunk(message, body, modes, limit, reply_markup=markup):
            delivered_all = False
            logger.error(
                "Chunk %d/%d was NOT delivered to chat %s", index, total, message.chat.id
            )
    return delivered_all


async def _send_chunk(
    message: Message,
    body: str,
    modes: list[str],
    limit: int,
    *,
    reply_markup: Any = None,
) -> bool:
    """Try every parse mode (with rate-limit retries) to deliver one chunk."""
    for mode in modes:
        rendered = render(body, mode)
        parse_mode = PARSE_MODES.get(mode)
        retries = 0
        while True:
            try:
                await throttle.wait(message.chat.id)
                await message.answer(rendered, parse_mode=parse_mode, reply_markup=reply_markup)
                return True
            except TelegramRetryAfter as exc:
                retries += 1
                if retries > _MAX_429_RETRIES:
                    logger.error("Rate limit persists after %d retries", retries - 1)
                    break
                delay = _retry_delay(exc)
                logger.warning("Telegram rate limit (429); retrying in %.1fs", delay)
                await asyncio.sleep(delay)
            except TelegramBadRequest as exc:
                logger.warning(
                    "Sending with parse_mode=%s failed: %s",
                    parse_mode or "plain",
                    exc,
                )
                if "too long" in str(exc).lower():
                    return await _send_truncated(
                        message, body, limit, reply_markup=reply_markup
                    )
                break
            except TelegramForbiddenError as exc:
                # The bot was blocked or kicked: retrying cannot help.
                logger.warning("Chat is unreachable for the bot: %s", exc)
                return False
            except TelegramAPIError as exc:
                logger.error("Sending the answer failed (%s)", exc)
                break
            except Exception:
                logger.exception("Unexpected error while sending the answer")
                break

    return await _send_truncated(message, body, limit, reply_markup=reply_markup)


async def _send_truncated(
    message: Message, body: str, limit: int, *, reply_markup: Any = None
) -> bool:
    """Last resort: plain text, truncated to the hard limit, with 429 retries."""
    payload = body if len(body) <= limit else body[: max(limit - 1, 1)] + "…"
    for attempt in range(_MAX_429_RETRIES + 1):
        try:
            await throttle.wait(message.chat.id)
            await message.answer(payload, parse_mode=None, reply_markup=reply_markup)
            logger.info("Answer delivered as truncated plain text")
            return True
        except TelegramRetryAfter as exc:
            if attempt >= _MAX_429_RETRIES:
                logger.error("Rate limit persists; the answer was lost")
                return False
            await asyncio.sleep(_retry_delay(exc))
        except Exception as exc:  # noqa: BLE001 - a reply must not be lost
            logger.error("Last-resort plain send failed: %s", exc)
            return False
    return False


async def edit_formatted(
    message: Message, raw: str, *, reply_markup: Any = None
) -> bool:
    """Edit one existing message into the final formatted answer.

    Tries the same parse-mode fallback chain as :func:`send_formatted`, but the
    whole text must fit one message: when it does not (or editing is rejected),
    the caller is expected to fall back to a multi-message send.
    """
    if message is None or not hasattr(message, "edit_text"):
        return False

    if config.normalize_formulas:
        raw = normalize_formulas(raw) or raw

    modes = [config.parse_mode]
    for extra in ("html", "plain"):
        if extra not in modes:
            modes.append(extra)

    for mode in modes:
        rendered = render(raw, mode)
        if not rendered:
            continue
        if len(rendered) > 4096:
            logger.info(
                "Formatted answer does not fit one message (%d chars); resending",
                len(rendered),
            )
            return False
        parse_mode = PARSE_MODES.get(mode)
        try:
            await throttle.wait(message.chat.id)
            await message.edit_text(
                rendered, parse_mode=parse_mode, reply_markup=reply_markup
            )
            return True
        except TelegramBadRequest as exc:
            if "not modified" in str(exc).lower():
                return True
            logger.warning(
                "Editing with parse_mode=%s failed: %s", parse_mode or "plain", exc
            )
        except TelegramRetryAfter:
            # The streamed edits are throttled, so this is rare: give up on
            # the in-place edit and let the caller re-send robustly.
            logger.warning("Telegram rate limit hit while finalizing the answer")
            return False
        except Exception as exc:  # noqa: BLE001 - the caller has a fallback
            logger.warning("Could not edit the message in place: %s", exc)
            return False
    return False


async def safe_answer(message: Message, text: str, **kwargs) -> bool:
    """Send a message, retrying on a rate limit; never raises.

    ``text`` is interpreted with the bot's default parse mode (HTML), so escape
    ``<``, ``>`` and ``&`` in dynamic parts.
    """
    for attempt in range(_MAX_429_RETRIES + 1):
        try:
            await throttle.wait(message.chat.id)
            await message.answer(text, **kwargs)
            return True
        except TelegramRetryAfter as exc:
            if attempt >= _MAX_429_RETRIES:
                logger.error(
                    "Rate limit persists; a message could not be sent (%d chars)",
                    len(text),
                )
                return False
            await asyncio.sleep(_retry_delay(exc))
        except TelegramForbiddenError as exc:
            logger.warning("Chat is unreachable for the bot: %s", exc)
            return False
        except Exception as exc:  # noqa: BLE001 - a reply must not be lost
            logger.error("Could not send the message: %s", exc)
            return False
    return False


async def require_private_chat(message: Message, *, what: str = "This command") -> bool:
    """True in a private chat; otherwise explain and return False.

    Commands that print personal settings or conversation content must not run
    in groups — the reply would be visible to every participant.
    """
    if message.chat.type == ChatType.PRIVATE:
        return True
    await safe_answer(
        message,
        f"🔒 <b>{what} is available in a private chat only.</b>\n\n"
        "Send it to me in a direct message.",
    )
    return False
