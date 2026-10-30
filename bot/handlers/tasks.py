"""Монитор активных операций для администраторов в личном чате."""

from __future__ import annotations

import html
import time

from aiogram import F, Router
from aiogram.filters import Command
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from bot.config import config
from bot.services.operations import active_operations
from bot.services.telegram_utils import safe_answer_callback, safe_edit_text

router = Router()
PAGE_SIZE = 5


def render_tasks(viewer_id: int, page: int = 0) -> tuple[str, InlineKeyboardMarkup]:
    """Сформировать ограниченную страницу без текстов запросов и файлов."""
    operations = active_operations()
    pages = max(1, (len(operations) + PAGE_SIZE - 1) // PAGE_SIZE)
    page = max(0, min(page, pages - 1))
    lines = [f"🔄 <b>Active tasks: {len(operations)}</b>", f"Page {page + 1}/{pages}"]
    for operation in operations[page * PAGE_SIZE:(page + 1) * PAGE_SIZE]:
        owner = "Scheduled" if operation.user_id is None else str(operation.user_id)
        if operation.user_name:
            owner = f"{operation.user_name[:60]} ({owner})"
        if operation.user_id == viewer_id:
            owner += " · Your task"
        seconds = max(0, int(time.monotonic() - operation.started))
        location = html.escape(operation.location[:80])
        lines.append(
            f"\n<b>#{operation.id} · {html.escape(operation.kind)}</b> · {seconds}s\n"
            f"👤 {html.escape(owner)}\n"
            f"📍 {location} · <code>{operation.chat_id}</code>\n"
            f"{html.escape(operation.stage)}"
        )
    if not operations:
        lines.append("\nNo active tasks.")
    navigation = []
    if page:
        navigation.append(InlineKeyboardButton(text="◀", callback_data=f"tasks:{page - 1}"))
    navigation.append(InlineKeyboardButton(text="🔄 Refresh", callback_data=f"tasks:{page}"))
    if page + 1 < pages:
        navigation.append(InlineKeyboardButton(text="▶", callback_data=f"tasks:{page + 1}"))
    return "\n".join(lines), InlineKeyboardMarkup(inline_keyboard=[
        navigation,
        [InlineKeyboardButton(text="◀ Admin menu", callback_data="adm:home")],
    ])


@router.message(Command("tasks"))
async def cmd_tasks(message: Message) -> None:
    """Показать задачи только действующему админу в личке."""
    user = message.from_user
    if user is None or not config.is_admin(user.id):
        await message.answer("⛔ Admin access required.")
        return
    if message.chat.type != "private":
        await message.answer("Open /tasks in a private chat with the bot.")
        return
    text, markup = render_tasks(user.id)
    await message.answer(text, reply_markup=markup)


@router.callback_query(F.data.startswith("tasks:"))
async def cb_tasks(callback: CallbackQuery) -> None:
    """Перепроверить права и место показа при каждом обновлении."""
    message = callback.message
    if not config.is_admin(callback.from_user.id):
        await safe_answer_callback(callback, "Admin access required.", show_alert=True)
        return
    if message is None or message.chat.type != "private":
        await safe_answer_callback(callback, "Open /tasks in a private chat.", show_alert=True)
        return
    raw = "0" if callback.data == "adm:tasks" else (callback.data or "").partition(":")[2]
    if not raw.isascii() or not raw.isdecimal() or len(raw) > 8:
        await safe_answer_callback(callback)
        return
    text, markup = render_tasks(callback.from_user.id, int(raw))
    await safe_answer_callback(callback)
    await safe_edit_text(message, text, reply_markup=markup)
