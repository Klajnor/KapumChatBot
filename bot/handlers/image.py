from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from bot.config import config
from bot.handlers.chat import generate_and_send_image
from bot.services import limits
from bot.services.access import check_access, guest_hint, refund_access
from bot.services.operations import tracked
from bot.services.themed_responses import get_random_themed_response
from bot.services.user_service import user_service

logger = logging.getLogger(__name__)

router = Router()

# A sender produces the image from a prompt and sends it, returning True on
# success.
Sender = Callable[..., Awaitable[bool]]


@router.message(Command("image"))
async def cmd_image(message: Message) -> None:
    """Handle /image <prompt> — generate an image and send it."""
    if not config.enable_image_generation:
        await message.answer("🚫 <b>Image generation is disabled.</b>")
        return
    await _handle_generation_command(
        message,
        generate_and_send_image,
        usage=(
            "🎨 <b>Generate an image</b>\n\n"
            "Usage: <code>/image &lt;description&gt;</code>\n"
            "Example: <code>/image a red fox in a snowy forest</code>"
        ),
    )


async def _handle_generation_command(
    message: Message, sender: Sender, *, usage: str
) -> None:
    """Shared body of generation commands: validate, then generate under a slot."""
    user = message.from_user
    if user is None:
        return

    parts = (message.text or "").split(maxsplit=1)
    if len(parts) < 2 or not parts[1].strip():
        await message.answer(usage)
        return

    await _run_generation(message, sender, parts[1].strip())


async def generate_image_from_prompt(message: Message, prompt: str) -> None:
    """Generate an image for ``prompt`` (the reply-menu dialog's second step)."""
    if not config.enable_image_generation:
        await message.answer("🚫 <b>Image generation is disabled.</b>")
        return
    await _run_generation(message, generate_and_send_image, prompt.strip())


@tracked("Image generation")
async def _run_generation(message: Message, sender: Sender, prompt: str) -> None:
    """Take the user's concurrency slot and generate under it."""
    user = message.from_user
    if user is None:
        return
    with limits.slot(user.id) as granted:
        if not granted:
            await message.answer(limits.BUSY_MESSAGE)
            return
        await _generate(message, user, prompt, sender)


async def _generate(message: Message, user, prompt: str, sender: Sender) -> None:
    """Register the user, check access, run ``sender`` and settle the quota."""
    async with user_service() as service:
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
        sent = await sender(
            message,
            prompt,
            service=service,
            user_id=user.id,
            model=settings["model"],
            # Выбранная персона становится личностью художника.
            persona=settings["system_prompt"],
        )

        if not sent:
            await refund_access(service, user.id, access)

        hint = guest_hint(access)
        if sent and hint:
            await message.answer(hint)
