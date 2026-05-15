from __future__ import annotations

import datetime
import html

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from bot.config import config
from bot.database.session import async_session_factory
from bot.services.user_service import UserService

router = Router()

DAILY_LIMIT = 10


@router.message(Command("stats"))
async def cmd_stats(message: Message) -> None:
    """Handle /stats command — show usage statistics."""
    user = message.from_user
    if user is None:
        return

    is_admin = user.id in config.admin_ids

    async with async_session_factory() as session:
        service = UserService(session)
        stats = await service.get_stats(user.id)

    # Build base stats
    lines = [
        f"📊 <b>Your Statistics</b>\n",
        f"💬 <b>Messages sent:</b> {stats['message_count']}\n",
        f"🤖 <b>Current model:</b> <code>{html.escape(stats['model'])}</code>\n",
        f"📝 <b>System prompt:</b>\n",
        f"<code>{html.escape(stats['system_prompt'][:100])}</code>\n",
    ]
    if len(stats['system_prompt']) > 100:
        lines.append("<i>...(truncated)</i>\n")

    # Show daily usage for non-admin users
    if not is_admin:
        async with async_session_factory() as session:
            service = UserService(session)
            remaining = await service.get_daily_usage_remaining(user.id, DAILY_LIMIT)

        today = datetime.date.today().isoformat()
        lines.append(f"\n📅 <b>Today ({today}):</b>")
        lines.append(f"⚠️ <b>Remaining questions:</b> {remaining}/{DAILY_LIMIT}")

    await message.answer("\n".join(lines))