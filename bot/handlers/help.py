from __future__ import annotations

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

router = Router()


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    """Handle /help command — show all available commands."""
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
        "• /prompt <text> — Set a new system prompt\n"
        "• /clear — Clear conversation history\n"
        "• /stats — View your usage statistics\n"
        "• /settings — View your current settings\n\n"
        "<b>💡 Tips:</b>\n"
        "• Change the system prompt to customize AI behavior\n"
        "• Use /models to switch between different AI models\n"
        "• Clear history with /clear to start a fresh conversation\n"
        "• Send images to use vision capabilities"
    )
    await message.answer(help_text)