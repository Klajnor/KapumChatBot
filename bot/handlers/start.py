from __future__ import annotations

from aiogram import Router
from aiogram.filters import CommandStart
from aiogram.types import Message

from bot.database.session import async_session_factory
from bot.services.user_service import UserService

router = Router()


@router.message(CommandStart())
async def cmd_start(message: Message) -> None:
    """Handle /start command — register user and show welcome message."""
    user = message.from_user
    if user is None:
        return

    async with async_session_factory() as session:
        service = UserService(session)
        await service.register_user(
            telegram_id=user.id,
            username=user.username,
            first_name=user.first_name,
            last_name=user.last_name,
        )

    welcome_text = (
        f"👋 <b>Hello, {user.first_name or 'there'}!</b>\n\n"
        f"I'm an AI-powered bot connected to <b>OpenRouter</b>. "
        f"I can chat with you using various AI models, analyze images, "
        f"and adapt to your preferences.\n\n"
        f"<b>Available commands:</b>\n"
        f"💬 <b>Just send me a message</b> — I'll reply using AI\n"
        f"🖼️ <b>Send an image</b> — I'll analyze it (vision models)\n"
        f"🤖 /models — Choose an AI model\n"
        f"📝 /prompt — View or set system prompt\n"
        f"🧹 /clear — Clear conversation history\n"
        f"📊 /stats — Your usage statistics\n"
        f"⚙️ /settings — Current settings\n"
        f"❓ /help — All commands\n\n"
        f"<i>Let's get started! 🚀</i>"
    )
    await message.answer(welcome_text)