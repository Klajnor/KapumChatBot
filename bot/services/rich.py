"""Rich messages (Bot API 10.1-10.3): native rendering of model Markdown.

Telegram now parses GitHub-Flavored Markdown *inside messages*: headings,
tables, task lists, footnotes, `~~strike~~`, `==mark==` and — crucially for
this bot — LaTeX formulas in `$…$` / `$$…$$` / ```math fences. The old
workarounds (Telegraph pages, PNG formula images, MarkdownV2 escaping) are no
longer needed on the main path; they stay as fallbacks when the API refuses a
rich message.

Two delivery forms are used here:

* :func:`send_rich` — the final answer as one or more `sendRichMessage`
  calls (limit: 32768 characters per message, we split at paragraph
  boundaries);
* :func:`send_rich_draft` — a streaming preview via `sendRichMessageDraft`.
  Drafts work only in private chats, are ephemeral (a 30-second preview), and
  the complete message must still be sent with `sendRichMessage` afterwards.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import re
from typing import Any

from aiogram.enums import ChatType
from aiogram.exceptions import TelegramForbiddenError, TelegramRetryAfter
from aiogram.types import InputRichMessage, ReplyParameters

from bot.services.markdown import _raw_paragraphs
from bot.services.science import (  # internal helpers of the formula detector
    _CODE_RE,
    _INLINE_MATH_RE,
    _MATH_MARKER_RE,
    _VARIABLE_RE,
)
from bot.services.telegram_utils import _retry_delay, throttle

logger = logging.getLogger(__name__)

# The hard limit is 32768 UTF-8 characters of rich text (including formula
# source); keep a safety margin, like elsewhere in this project.
RICH_LIMIT = 32000

_MAX_429_RETRIES = 3

# ``\[ … \]`` / ``\( … \)`` are the LaTeX delimiters the old PNG pipeline asked
# the model to use; rich Markdown only understands the dollar forms.
_DISPLAY_LATEX_RE = re.compile(r"\\\[\s*(.+?)\s*\\\]", re.DOTALL)
_INLINE_LATEX_RE = re.compile(r"\\\(\s*(.+?)\s*\\\)")


def _looks_formulaic(body: str) -> bool:
    """Same heuristic as the formula normalizer: prices are not math."""
    stripped = body.strip()
    if not stripped:
        return False
    if _MATH_MARKER_RE.search(stripped) or _VARIABLE_RE.fullmatch(stripped):
        return True
    return bool(
        stripped == body
        and re.search(r"[A-Za-zА-Яа-я]", stripped)
        and not (stripped[0].isdigit() and " " in stripped)
    )


def _protect_dollars(segment: str) -> str:
    """Escape ``$…$`` pairs that are not formulas (``$5 and $10``).

    Left alone: real math (kept for the native renderer) and ``$`` signs that
    form no pair at all.
    """

    def repl(match: re.Match[str]) -> str:
        body = match.group(1)
        if _looks_formulaic(body or ""):
            return match.group(0)
        return rf"\${body}\$"

    return _INLINE_MATH_RE.sub(repl, segment)


def _protect_dollars_outside_code(text: str) -> str:
    parts = _CODE_RE.split(text)
    return "".join(
        part if index % 2 else _protect_dollars(part)
        for index, part in enumerate(parts)
    )


def adapt_markdown(text: str) -> str:
    """Rewrite the answer into syntax the rich Markdown parser understands.

    Only delimiters are touched: ``\\[…\\]`` → ``$$…$$``, ``\\(…\\)`` → ``$…$``,
    and price-like ``$…$`` pairs are escaped. Code stays untouched, and the
    function never raises.
    """
    if not text:
        return text
    try:
        parts = _CODE_RE.split(text)
        for index in range(0, len(parts), 2):
            part = parts[index]
            part = _DISPLAY_LATEX_RE.sub(lambda m: f"$${m.group(1)}$$", part)
            part = _INLINE_LATEX_RE.sub(lambda m: f"${m.group(1)}$", part)
            parts[index] = part
        return _protect_dollars_outside_code("".join(parts))
    except Exception as exc:  # noqa: BLE001 - adaptation must never eat an answer
        logger.warning("Rich Markdown adaptation failed (%s); raw text used", exc)
        return text


def split_rich(text: str, limit: int = RICH_LIMIT) -> list[str]:
    """Split adapted Markdown so every part fits one rich message."""
    if not text or not text.strip():
        return []
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    current = ""
    for piece in _raw_paragraphs(text):
        candidate = f"{current}\n\n{piece}" if current else piece
        if len(candidate) <= limit:
            current = candidate
            continue
        if current:
            chunks.append(current)
            current = ""
        while len(piece) > limit:
            chunks.append(piece[:limit])
            piece = piece[limit:]
        current = piece
    if current:
        chunks.append(current)
    return chunks


def _send_error(exc: Exception) -> None:
    logger.warning("Rich message rejected (%s); using the legacy path", exc)


async def send_rich(
    message: Any, text: str, *, reply_markup: Any = None
) -> bool:
    """Deliver ``text`` (model Markdown) as native rich message(s).

    Returns True only when every part was sent; any rejection makes the caller
    fall back to the legacy delivery chain, so nothing can be lost here.
    """
    body = adapt_markdown(text)
    parts = split_rich(body)
    if not parts:
        logger.warning("Nothing to send: the answer adapted to an empty string")
        return False

    total = len(parts)
    extra: dict[str, Any] = {}
    if message.chat.type != ChatType.PRIVATE:
        extra["reply_parameters"] = ReplyParameters(message_id=message.message_id)
    thread_id = getattr(message, "message_thread_id", None)
    if thread_id:
        extra["message_thread_id"] = thread_id

    for index, part in enumerate(parts, start=1):
        payload = f"Part {index}/{total}\n\n{part}" if total > 1 else part
        rich = InputRichMessage(markdown=payload)
        markup = reply_markup if index == total else None
        if not await _send_one(message, rich, markup, extra):
            return False
    return True


async def _send_one(
    message: Any, rich: InputRichMessage, reply_markup: Any, extra: dict[str, Any]
) -> bool:
    retries = 0
    while True:
        try:
            await throttle.wait(message.chat.id)
            await message.bot.send_rich_message(
                chat_id=message.chat.id,
                rich_message=rich,
                reply_markup=reply_markup,
                **extra,
            )
            return True
        except TelegramRetryAfter as exc:
            retries += 1
            if retries > _MAX_429_RETRIES:
                _send_error(exc)
                return False
            delay = _retry_delay(exc)
            logger.warning("Telegram rate limit (429); retrying rich send in %.1fs", delay)
            await asyncio.sleep(delay)
        except TelegramForbiddenError as exc:
            logger.warning("Chat is unreachable for the bot: %s", exc)
            return False
        except Exception as exc:  # noqa: BLE001 - the caller has a fallback chain
            _send_error(exc)
            return False


# --- Drafts (streaming preview) ---------------------------------------------

_draft_ids = itertools.count(1)


def next_draft_id() -> int:
    """A fresh non-zero draft identifier (same id = animated replacement)."""
    return next(_draft_ids)


async def send_rich_draft(
    bot: Any, chat_id: int, draft_id: int, text: str, *, message_thread_id: int | None = None
) -> bool:
    """Stream a partial rich message; False when Telegram refused the draft.

    Partial Markdown can hold unclosed syntax (``**bold`` cut mid-stream),
    which the parser rejects — the caller then degrades to a plain text draft.
    """
    extra: dict[str, Any] = {}
    if message_thread_id:
        extra["message_thread_id"] = message_thread_id
    try:
        await bot.send_rich_message_draft(
            chat_id=chat_id,
            draft_id=draft_id,
            rich_message=InputRichMessage(markdown=adapt_markdown(text)),
            **extra,
        )
        return True
    except Exception as exc:  # noqa: BLE001 - a lost draft must not break streaming
        logger.debug("Rich draft update refused: %s", exc)
        return False


async def send_text_draft(
    bot: Any, chat_id: int, draft_id: int, text: str, *, message_thread_id: int | None = None
) -> bool:
    """Plain text preview through sendMessageDraft (rich draft fallback)."""
    extra: dict[str, Any] = {}
    if message_thread_id:
        extra["message_thread_id"] = message_thread_id
    try:
        await bot.send_message_draft(
            chat_id=chat_id,
            draft_id=draft_id,
            text=text,
            parse_mode=None,
            **extra,
        )
        return True
    except Exception as exc:  # noqa: BLE001 - previews are cosmetic
        logger.debug("Text draft update refused: %s", exc)
        return False
