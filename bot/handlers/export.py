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
"""

from __future__ import annotations

import asyncio
import logging

from aiogram import F, Router
from aiogram.exceptions import TelegramAPIError, TelegramRetryAfter
from aiogram.filters import Command
from aiogram.types import BufferedInputFile, CallbackQuery, Message

from bot.config import config
from bot.keyboards.inline import get_export_format_keyboard
from bot.services import export
from bot.services.export import ExportEntry
from bot.services.telegram_utils import safe_answer, safe_answer_callback
from bot.services.user_service import user_service

logger = logging.getLogger(__name__)
router = Router()

_FORMATS = ("pdf", "md")
_EXPORT_DISABLED = "🚫 <b>Export is disabled</b> on this bot (ENABLE_EXPORT=false)."


async def _send_document(
    message: Message, data: bytes, filename: str, caption: str
) -> bool:
    """``answer_document`` with one rate-limit retry; never raises."""
    for attempt in range(2):
        try:
            await message.answer_document(
                BufferedInputFile(data, filename=filename), caption=caption
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
    """Export the requesting user's thread in this chat."""
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
        meta=export.build_meta(message.chat.id, len(entries), scope="full history"),
    )
    if result is None:
        await safe_answer(message, "❌ <b>Could not build the export file.</b>")
        return
    data, filename, _mime = result
    caption = (
        f"📤 <b>Conversation export</b> · {len(entries)} messages"
        + _fallback_note(fmt, filename)
    )
    if not await _send_document(message, data, filename, caption):
        await safe_answer(message, "❌ <b>Could not send the export file.</b>")


async def _export_single_answer(
    callback: CallbackQuery, fmt: str, message_id: int
) -> None:
    """Export one stored answer (plus the question it answered)."""
    user = callback.from_user
    if user is None or callback.message is None:
        await safe_answer_callback(callback)
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
    if await _send_document(callback.message, data, filename, caption):
        await safe_answer_callback(callback, "✅ File sent")
    else:
        await safe_answer_callback(
            callback, "⚠️ Could not send the file.", show_alert=True
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
        await safe_answer(
            message,
            "📤 <b>Export this conversation?</b>\n\n"
            f"Your thread in this chat (up to {config.export_max_messages} "
            "latest messages) will be saved as a file.",
            reply_markup=get_export_format_keyboard(),
        )
        return
    await _export_history(message, fmt, user.id)


@router.callback_query(F.data.startswith("expall:"))
async def cb_export_all(callback: CallbackQuery) -> None:
    """Format-picker callback for the full-history export."""
    if not config.enable_export:
        await safe_answer_callback(callback, "🚫 Export is disabled.", show_alert=True)
        return
    fmt = (callback.data or "").split(":", 1)[1]
    if fmt not in _FORMATS or callback.message is None or callback.from_user is None:
        await safe_answer_callback(callback)
        return
    await safe_answer_callback(callback, "📤 Preparing the export…")
    await _export_history(callback.message, fmt, callback.from_user.id)


@router.callback_query(F.data.startswith("expans:"))
async def cb_export_answer(callback: CallbackQuery) -> None:
    """📄/📝 button under an answer: export that single Q&A pair."""
    if not config.enable_export:
        await safe_answer_callback(callback, "🚫 Export is disabled.", show_alert=True)
        return
    parts = (callback.data or "").split(":")
    if len(parts) != 3 or parts[1] not in _FORMATS or not parts[2].isdigit():
        await safe_answer_callback(callback)
        return
    await _export_single_answer(callback, parts[1], int(parts[2]))
