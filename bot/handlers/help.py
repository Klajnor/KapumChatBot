from __future__ import annotations

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from bot.config import config

router = Router()


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    """Handle /help command — show all available commands."""
    user = message.from_user
    
    if user is None:
        return   
    
    is_admin = user.id in config.admin_ids

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
            "• <b>Send any text</b> — 🚫 <i>Admin-only channel</i>\n"
            "• <b>Send an image</b> — 🚫 <i>Admin-only channel</i>\n\n"
            "<b>Commands:</b>\n"
            "• /start — Welcome message and registration\n"
            "• /help — This help message\n\n"
            "<b>💡 Note:</b>\n"
            "This bot is restricted to authorized personnel. "
            "If you believe you should have access, contact an administrator."
        )
    await message.answer(help_text)
