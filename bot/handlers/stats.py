from __future__ import annotations

import datetime
import html

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from bot.config import config
from bot.services.telegram_utils import require_private_chat
from bot.services.user_service import user_service

router = Router()


@router.message(Command("stats"))
async def cmd_stats(message: Message) -> None:
    """Handle /stats command — show usage statistics."""
    user = message.from_user
    if user is None:
        return
    # The system prompt and per-chat message counts are personal: never print
    # them where other group participants can read them.
    if not await require_private_chat(message, what="/stats"):
        return

    async with user_service() as service:
        stats = await service.get_stats(user.id)

        lines = [
            "📊 <b>Your Statistics</b>\n",
            f"💬 <b>Messages sent:</b> {stats['message_count']}\n",
            f"🤖 <b>Current model:</b> <code>{html.escape(stats['model'])}</code>\n",
            "📝 <b>System prompt:</b>\n",
            f"<code>{html.escape(stats['system_prompt'][:100])}</code>\n",
        ]
        if len(stats["system_prompt"]) > 100:
            lines.append("<i>...(truncated)</i>\n")

        if config.is_admin(user.id):
            lines.append("\n👑 <b>Role:</b> admin (unlimited)")
        elif await service.is_blocked(user.id):
            lines.append("\n🚫 <b>Role:</b> blocked")
        else:
            is_allowed = await service.is_allowed_user(user.id)
            today = datetime.date.today().isoformat()
            if is_allowed:
                remaining = await service.get_daily_usage_remaining(user.id)
                lines.append("\n👤 <b>Role:</b> authorized user")
                lines.append(f"\n📅 <b>Today ({today}):</b>")
                lines.append(
                    f"⚠️ <b>Remaining questions:</b> "
                    f"{remaining}/{await service.get_daily_limit(user.id)}"
                )
            else:
                remaining = await service.get_guest_messages_remaining(user.id)
                lines.append("\n👤 <b>Role:</b> guest (free trial)")
                lines.append(
                    f"🎁 <b>Free messages left:</b> "
                    f"{remaining}/{config.guest_message_limit}"
                )

    await message.answer("\n".join(lines))