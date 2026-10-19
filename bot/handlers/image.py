from __future__ import annotations

import logging

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from bot.config import config
from bot.database.session import async_session_factory
from bot.handlers.chat import generate_and_send_image
from bot.services.access import check_access, guest_hint
from bot.services.themed_responses import get_random_themed_response
from bot.services.user_service import UserService

logger = logging.getLogger(__name__)

router = Router()


@router.message(Command("image"))
async def cmd_image(message: Message) -> None:
    """Handle /image <prompt> — generate an image and send it."""
    user = message.from_user
    if user is None:
        return

    if not config.enable_image_generation:
        await message.answer("🚫 <b>Image generation is disabled.</b>")
        return

    parts = (message.text or "").split(maxsplit=1)
    if len(parts) < 2 or not parts[1].strip():
        await message.answer(
            "🎨 <b>Generate an image</b>\n\n"
            "Usage: <code>/image &lt;description&gt;</code>\n"
            "Example: <code>/image a red fox in a snowy forest</code>"
        )
        return

    prompt = parts[1].strip()

    async with async_session_factory() as session:
        service = UserService(session)
        await service.register_user(
            telegram_id=user.id,
            username=user.username,
            first_name=user.first_name,
            last_name=user.last_name,
        )

        access = await check_access(service, user.id)
        if not access.allowed:
            await message.answer(access.message or get_random_themed_response())
            return

        settings = await service.get_user_settings(user.id)
        sent = await generate_and_send_image(
            message,
            prompt,
            service=service,
            user_id=user.id,
            model=settings["model"],
        )

        hint = guest_hint(access)
        if sent and hint:
            await message.answer(hint)