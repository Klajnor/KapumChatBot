from __future__ import annotations

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from bot.config import config
from bot.database.session import async_session_factory
from bot.services.user_service import UserService

router = Router()

DAILY_LIMIT = 10


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    """Handle /help command — show all available commands."""
    user = message.from_user

    if user is None:
        return

    is_admin = user.id in config.admin_ids

    # Check if user is in allowed list
    is_allowed = False
    if not is_admin:
        async with async_session_factory() as session:
            service = UserService(session)
            is_allowed = await service.is_allowed_user(user.id)

    if is_admin:
        help_text = (
            "<b>🤖 Available Commands</b>\n\n"
            "<b>Chat:</b>\n"
            "• <b>Send any text</b> — Chat with AI\n"
            "• <b>Send an image</b> — Analyze image with AI (vision models)\n\n"
            "<b>Commands:</b>\n"
            "• /start — Welcome message and registration\n"
            "• /help — This help message\n"
            "• /models — Select an AI model from the list\n"
            "• /model — Show your current AI model\n"
            "• /prompt — View current system prompt\n"
            "• /prompt &lt;text&gt; — Set a new system prompt\n"
            "• /clear — Clear conversation history\n"
            "• /stats — View your usage statistics\n"
            "• /settings — View your current settings\n\n"
            "<b>👑 Admin Commands:</b>\n"
            "• /add_user &lt;id&gt; — Add a user to the allowed list\n"
            "• /remove_user &lt;id&gt; — Remove a user from allowed list\n"
            "• /list_users — Show all allowed users\n"
            "• /admin_stats — Usage statistics for last 14 days\n"
            "• /reset_daily_usage &lt;id&gt; — Reset daily usage for a user\n\n"
            "<b>💡 Tips:</b>\n"
            "• Change the system prompt to customize AI behavior\n"
            "• Use /models to switch between different AI models\n"
            "• Clear history with /clear to start a fresh conversation\n"
            "• Send images to use vision capabilities"
        )
    elif is_allowed:
        help_text = (
            "<b>🤖 Available Commands</b>\n\n"
            "<b>Chat:</b>\n"
            "• <b>Send any text</b> — Chat with AI\n"
            "• <b>Send an image</b> — Analyze image with AI (vision models)\n\n"
            "<b>Commands:</b>\n"
            "• /start — Welcome message and registration\n"
            "• /help — This help message\n"
            "• /models — Select an AI model from the list\n"
            "• /model — Show your current AI model\n"
            "• /prompt — View current system prompt\n"
            "• /prompt &lt;text&gt; — Set a new system prompt\n"
            "• /clear — Clear conversation history\n"
            "• /stats — View your usage statistics\n"
            "• /settings — View your current settings\n\n"
            f"<b>⚠️ Daily limit:</b> {DAILY_LIMIT} questions per day\n\n"
            "<b>💡 Tips:</b>\n"
            "• Change the system prompt to customize AI behavior\n"
            "• Use /models to switch between different AI models\n"
            "• Clear history with /clear to start a fresh conversation\n"
            "• Send images to use vision capabilities"
        )
    else:
        help_text = (
            "<b>🤖 Available Commands</b>\n\n"
            "<b>Chat:</b>\n"
            "• <b>Send any text</b> — 🚫 <i>Restricted access</i>\n"
            "• <b>Send an image</b> — 🚫 <i>Restricted access</i>\n\n"
            "<b>Commands:</b>\n"
            "• /start — Welcome message and registration\n"
            "• /help — This help message\n\n"
            "<b>💡 Note:</b>\n"
            "This bot is restricted to authorized personnel. "
            "If you believe you should have access, contact an administrator."
        )
    await message.answer(help_text)
