from __future__ import annotations

import html

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from bot.bot import get_bot_username
from bot.config import config
from bot.database.session import async_session_factory
from bot.services.user_service import UserService

router = Router()

_CHAT_SECTION = (
    "<b>Chat:</b>\n"
    "• <b>Send any text</b> — chat with the AI\n"
    "• <b>Send an image</b> — analyze it with a vision model\n"
    "• <b>Send a voice message</b> — I'll recognize it and reply\n\n"
    "<b>Commands:</b>\n"
    "• /start — welcome message and registration\n"
    "• /help — this help message\n"
    "• /models — select an AI model\n"
    "• /model — show your current model\n"
    "• /image &lt;prompt&gt; — generate an image\n"
    "• /prompt — view the current system prompt\n"
    "• /prompt &lt;text&gt; — set a new system prompt\n"
    "• /clear — clear the history of this chat\n"
    "• /stats — your usage statistics\n"
    "• /settings — your current settings"
)

_TIPS = (
    "<b>💡 Tips:</b>\n"
    "• Customize the AI with /prompt\n"
    "• Switch models with /models (models marked 🖼️ support images)\n"
    "• Ask to draw something and I'll reply with a generated image\n"
    "• Use /clear to start a fresh conversation\n"
    "• In groups, mention me or reply to my message to get an answer"
)


def _guest_mode_tip() -> str:
    username = get_bot_username()
    if not config.enable_guest_mode or not username:
        return ""
    return (
        f"\n\n<b>🌍 Works in any chat:</b> mention "
        f"<code>@{html.escape(username)}</code> in any group or private "
        f"conversation — even where I'm not a member — and I'll reply there."
    )


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    """Handle /help command — show all available commands."""
    user = message.from_user
    if user is None:
        return

    is_admin = config.is_admin(user.id)

    is_allowed = False
    if not is_admin:
        async with async_session_factory() as session:
            service = UserService(session)
            is_allowed = await service.is_allowed_user(user.id)

    if is_admin:
        help_text = (
            "<b>🤖 Available Commands</b>\n\n"
            f"{_CHAT_SECTION}\n\n"
            "<b>👑 Admin Commands:</b>\n"
            "• /add_user &lt;id&gt; — add a user to the allowed list\n"
            "• /remove_user &lt;id&gt; — remove a user from the allowed list\n"
            "• /list_users — show all allowed users\n"
            "• /admin_stats — usage statistics for the last 14 days\n"
            "• /reset_daily_usage &lt;id&gt; — reset a user's daily usage"
            f"{_guest_mode_tip()}\n\n"
            f"{_TIPS}"
        )
    elif is_allowed:
        help_text = (
            "<b>🤖 Available Commands</b>\n\n"
            f"{_CHAT_SECTION}\n\n"
            f"⚠️ <b>Daily limit:</b> {config.daily_limit} questions per day"
            f"{_guest_mode_tip()}\n\n"
            f"{_TIPS}"
        )
    else:
        guest_note = (
            f"🎁 <b>Free trial:</b> up to {config.guest_message_limit} questions."
            if config.allow_guest_access
            else ""
        )
        help_text = (
            "<b>🤖 Available Commands</b>\n\n"
            "<b>Chat:</b>\n"
            "• <b>Send any text</b> — chat with the AI\n"
            "• <b>Send an image</b> — analyze it (vision models)\n\n"
            "<b>Commands:</b>\n"
            "• /start — welcome message and registration\n"
            "• /help — this help message\n\n"
            f"{guest_note}"
            f"{_guest_mode_tip()}\n\n"
            "<b>💡 Note:</b> this bot is limited to authorized personnel. "
            "Contact an administrator if you need full access."
        )

    await message.answer(help_text)