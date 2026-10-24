"""Conversation export: the /export command and the 📄/📝 answer buttons.

Two scopes share one renderer (``bot/services/export.py``):

* a single answer — the buttons under each assistant message in private chats
  (callbacks ``expans:<fmt>:<message_id>``); the stored row is reloaded with
  an **owner check**, so a guessed id can never export somebody else's answer;
* the whole conversation — ``/export [md|pdf]`` or its format-picker buttons
  (callbacks ``expall:<fmt>``); only the requesting user's own thread in that
  chat is included.

Exports never consume the daily quota: no AI calls are made, rendering is
local (Markdown, or PDF via fpdf2 + matplotlib's DejaVu fonts) and runs in a
worker thread so a big history does not block the event loop.

Two protections surround that local work:

* **authorized users only** (``check_access(..., consume=False,
  require_quota=False)``, guests excluded) — a free command must not become a
  CPU amplifier for accounts with no access at all;
* a **per-user cooldown** plus the per-user concurrency slot bound repeated
  renders, which would otherwise let anyone pin the CPU by hammering a PDF
  button.

The resulting file is always delivered to the requesting user's **private
chat**, even when the command was issued in a group: a conversation export is
personal data and must not be posted where other participants can read it.
"""
from __future__ import annotations

import asyncio
import logging
import math
import time
from typing import Any

from aiogram import F, Router
from aiogram.enums import ChatType
from aiogram.exceptions import TelegramAPIError, TelegramRetryAfter
from aiogram.filters import Command
from aiogram.types import BufferedInputFile, CallbackQuery, Message

from bot.config import config
from bot.keyboards.inline import get_export_format_keyboard
from bot.services import export, limits
from bot.services.access import check_access
from bot.services.export import ExportEntry
from bot.services.telegram_utils import safe_answer, safe_answer_callback
from bot.services.user_service import user_service

logger = logging.getLogger(__name__)
router = Router()

_FORMATS = ("pdf", "md")
_EXPORT_DISABLED = "🚫 <b>Export is disabled</b> on this bot (ENABLE_EXPORT=false)."
_EXPORT_DENIED = "⛔ Export is available to authorized users only."
_DM_REQUIRED = (
    "🔒 <b>Export is delivered to a private chat.</b>\n\n"
    "Open the bot in DM, send /start and try again."
)
_BUSY = "⏳ <b>Please wait.</b> Another request is still being processed."

# Rendering up to EXPORT_MAX_MESSAGES messages is not free (PDF: font loading +
# matplotlib in a worker thread), so one export per user per 30 seconds is
# plenty and stops a button-spammer from pinning the CPU.
_EXPORT_COOLDOWN = 30.0
_export_last: dict[int, float] = {}
_EXPORT_REGISTRY_CAP = 500


def _export_wait_seconds(user_id: int) -> int:
    """Seconds left before this user may export again (0 = allowed)."""
    remaining = _EXPORT_COOLDOWN - (time.monotonic() - _export_last.get(user_id, 0.0))
    return max(0, math.ceil(remaining)) if remaining > 0 else 0


def _mark_export(user_id: int) -> None:
    """Start the cooldown, keeping the registry bounded."""
    _export_last.pop(user_id, None)
    _export_last[user_id] = time.monotonic()
    while len(_export_last) > _EXPORT_REGISTRY_CAP:
        _export_last.pop(next(iter(_export_last)))


async def _export_denial(user_id: int, *, reserve: bool) -> str | None:
    """Why this user may not export now, or ``None`` when the export may run.

    ``reserve=True`` takes the cooldown in the same synchronous step as the
    check (no ``await`` in between), so two rapid taps cannot both pass.
    ``reserve=False`` is used when only the format picker is about to be shown.
    """
    async with user_service() as service:
        access = await check_access(
            service, user_id, consume=False, require_quota=False
        )
    if not access.allowed or access.role == "guest":
        return _EXPORT_DENIED
    wait = _export_wait_seconds(user_id)
    if wait:
        return f"⏳ Too many exports. Try again in {wait} s."
    if reserve:
        _mark_export(user_id)
    return None


async def _send_document(
    bot: Any, chat_id: int, data: bytes, filename: str, caption: str
) -> bool:
    """``send_document`` to ``chat_id`` with one rate-limit retry; never raises."""
    for attempt in range(2):
        try:
            await bot.send_document(
                chat_id=chat_id,
                document=BufferedInputFile(data, filename=filename),
                caption=caption,
            )
            return True
        except TelegramRetryAfter as exc:
            if attempt >= 1:
                break
            await asyncio.sleep(float(exc.retry_after) + 0.1)
        except TelegramAPIError as exc:
            logger.error("Sending the export file failed: %s", exc)
            break
    return False


def _fallback_note(fmt: str, filename: str) -> str:
    """Note when PDF was requested but Markdown was produced instead."""
    if fmt == "pdf" and filename.endswith(".md"):
        return "\n<i>PDF is unavailable on this server — sent Markdown instead.</i>"
    return ""


async def _export_history(message: Message, fmt: str, user_id: int) -> None:
    """Export the requesting user's thread in this chat, delivered via DM."""
    private = message.chat.type == ChatType.PRIVATE
    with limits.slot(user_id) as granted:
        if not granted:
            await safe_answer(message, _BUSY)
            return
        async with user_service() as service:
            rows = await service.get_export_history(message.chat.id, user_id)
            entries = export.entries_from_messages(rows)
        if not entries:
            await safe_answer(
                message,
                "📭 <b>Nothing to export yet.</b>\n\n"
                "Ask me something first — then /export saves the conversation.",
            )
            return
        result = await asyncio.to_thread(
            export.render_file,
            fmt,
            entries,
            title="Kapum ChatBot — Conversation History",
            meta=export.build_meta(
                message.chat.id, len(entries), scope="full history"
            ),
        )
        if result is None:
            await safe_answer(message, "❌ <b>Could not build the export file.</b>")
            return
        data, filename, _mime = result
        caption = (
            f"📤 <b>Conversation export</b> · {len(entries)} messages"
            + _fallback_note(fmt, filename)
        )
        if not private:
            await safe_answer(
                message, "📤 <b>Sending the export to your private chat…</b>"
            )
        if not await _send_document(message.bot, user_id, data, filename, caption):
            if private:
                await safe_answer(message, "❌ <b>Could not send the export file.</b>")
            else:
                await safe_answer(message, _DM_REQUIRED)


async def _export_single_answer(
    callback: CallbackQuery, fmt: str, message_id: int
) -> None:
    """Export one stored answer (plus the question it answered) via DM."""
    user = callback.from_user
    if user is None or callback.message is None:
        await safe_answer_callback(callback)
        return

    with limits.slot(user.id) as granted:
        if not granted:
            await safe_answer_callback(
                callback,
                "⏳ Another request is still running. Please wait.",
                show_alert=True,
            )
            return

        async with user_service() as service:
            answer = await service.get_message_for_export(message_id, user.id)
            if answer is None or answer.role != "assistant":
                await safe_answer_callback(
                    callback,
                    "⚠️ This message is no longer available for export.",
                    show_alert=True,
                )
                return
            question = await service.get_question_for(answer)
            # Snapshot into plain entries inside the session.
            entries: list[ExportEntry] = []
            if question is not None:
                entries.append(
                    ExportEntry(
                        role="user",
                        content=question.content or "",
                        created_at=question.created_at,
                    )
                )
            entries.append(
                ExportEntry(
                    role="assistant",
                    content=answer.content or "",
                    created_at=answer.created_at,
                    model=answer.model,
                )
            )
            chat_id = answer.chat_id or callback.message.chat.id

        result = await asyncio.to_thread(
            export.render_file,
            fmt,
            entries,
            title="Kapum ChatBot — AI Answer",
            meta=export.build_meta(chat_id, len(entries), scope="single answer"),
        )
        if result is None:
            await safe_answer_callback(
                callback, "⚠️ Could not build the file.", show_alert=True
            )
            return
        data, filename, _mime = result
        caption = "📤 <b>Answer export</b>" + _fallback_note(fmt, filename)
        if await _send_document(callback.message.bot, user.id, data, filename, caption):
            await safe_answer_callback(callback, "✅ File sent to your private chat")
        else:
            await safe_answer_callback(
                callback,
                "⚠️ Could not send the file to your private chat. "
                "Send /start to the bot in DM and retry.",
                show_alert=True,
            )


@router.message(Command("export"))
async def cmd_export(message: Message) -> None:
    """Handle /export [md|pdf] — save this conversation as a file."""
    if not config.enable_export:
        await safe_answer(message, _EXPORT_DISABLED)
        return
    user = message.from_user
    if user is None:
        return

    parts = (message.text or "").split()
    fmt = parts[1].strip().lower() if len(parts) > 1 else ""
    if fmt and fmt not in _FORMATS:
        await safe_answer(
            message,
            "⚠️ <b>Unknown format.</b>\n\n"
            "Use <code>/export md</code> or <code>/export pdf</code> — "
            "or plain <code>/export</code> to choose with buttons.",
        )
        return

    if not fmt:
        denial = await _export_denial(user.id, reserve=False)
        if denial:
            await safe_answer(message, denial)
            return
        await safe_answer(
            message,
            "📤 <b>Export this conversation?</b>\n\n"
            f"Your thread in this chat (up to {config.export_max_messages} "
            "latest messages) will be sent to your <b>private chat</b> as a file.",
            reply_markup=get_export_format_keyboard(),
        )
        return

    denial = await _export_denial(user.id, reserve=True)
    if denial:
        await safe_answer(message, denial)
        return
    await _export_history(message, fmt, user.id)


@router.callback_query(F.data.startswith("expall:"))
async def cb_export_all(callback: CallbackQuery) -> None:
    """Format-picker callback for the full-history export."""
    if not config.enable_export:
        await safe_answer_callback(callback, "🚫 Export is disabled.", show_alert=True)
        return
    parts = (callback.data or "").split(":", 1)
    fmt = parts[1] if len(parts) == 2 else ""
    if fmt not in _FORMATS or callback.message is None or callback.from_user is None:
        await safe_answer_callback(callback)
        return

    denial = await _export_denial(callback.from_user.id, reserve=True)
    if denial:
        await safe_answer_callback(callback, denial, show_alert=True)
        return
    await safe_answer_callback(callback, "📤 Preparing the export…")
    await _export_history(callback.message, fmt, callback.from_user.id)


@router.callback_query(F.data.startswith("expans:"))
async def cb_export_answer(callback: CallbackQuery) -> None:
    """📄/📝 button under an answer: export that single Q&A pair via DM."""
    if not config.enable_export:
        await safe_answer_callback(callback, "🚫 Export is disabled.", show_alert=True)
        return
    parts = (callback.data or "").split(":")
    if len(parts) != 3 or parts[1] not in _FORMATS or not parts[2].isdigit():
        await safe_answer_callback(callback)
        return

    denial = await _export_denial(callback.from_user.id, reserve=True)
    if denial:
        await safe_answer_callback(callback, denial, show_alert=True)
        return
    await _export_single_answer(callback, parts[1], int(parts[2]))
