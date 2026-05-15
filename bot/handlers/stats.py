from __future__ import annotations

import html

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from bot.database.session import async_session_factory
from bot.services.user_service import UserService

router = Router()


@router.message(Command("stats"))
async def cmd_stats(message: Message) -> None:
    """Handle /stats command — show usage statistics."""
    user = message.from_user
    if user is None:
        return

    async with async_session_factory() as session:
        service = UserService(session)
        stats = await service.get_stats(user.id)

    await message.answer(
        f"📊 <b>Your Statistics</b>\n\n"
        f"💬 <b>Messages sent:</b> {stats['message_count']}\n"
        f"🤖 <b>Current model:</b> <code>{html.escape(stats['model'])}</code>\n"
        f"📝 <b>System prompt:</b>\n"
        f"<code>{html.escape(stats['system_prompt'][:100])}</code>\n"
        f"{'<i>...(truncated)</i>' if len(stats['system_prompt']) > 100 else ''}"
    )