from __future__ import annotations

import html

from aiogram import Router, F
from aiogram.filters import Command
from aiogram.types import CallbackQuery, Message

from bot.database.session import async_session_factory
from bot.keyboards.inline import get_model_selection_keyboard
from bot.services.user_service import UserService

router = Router()


@router.message(Command("prompt"))
async def cmd_prompt(message: Message) -> None:
    """Handle /prompt command — view or set system prompt."""
    user = message.from_user
    if user is None:
        return

    # Extract prompt text if provided
    text = message.text or ""
    parts = text.split(maxsplit=1)

    async with async_session_factory() as session:
        service = UserService(session)

        if len(parts) > 1:
            # Set new prompt
            new_prompt = parts[1]
            await service.set_system_prompt(user.id, new_prompt)
            await message.answer(
                f"✅ <b>System prompt updated!</b>\n\n"
                f"<b>New prompt:</b>\n<code>{html.escape(new_prompt)}</code>"
            )
        else:
            # Show current prompt
            settings = await service.get_user_settings(user.id)
            await message.answer(
                f"📝 <b>Current System Prompt:</b>\n\n"
                f"<code>{html.escape(settings['system_prompt'])}</code>\n\n"
                f"<i>To change it, use:</i>\n"
                f"<code>/prompt Your new system prompt here</code>"
            )


@router.message(Command("settings"))
async def cmd_settings(message: Message) -> None:
    """Handle /settings command — show current settings."""
    user = message.from_user
    if user is None:
        return

    async with async_session_factory() as session:
        service = UserService(session)
        settings = await service.get_user_settings(user.id)

    await message.answer(
        f"⚙️ <b>Your Settings</b>\n\n"
        f"🤖 <b>Model:</b> <code>{html.escape(settings['model'])}</code>\n"
        f"🌡️ <b>Temperature:</b> <code>{settings['temperature']}</code>\n"
        f"📏 <b>Max Tokens:</b> <code>{settings['max_tokens']}</code>\n\n"
        f"📝 <b>System Prompt:</b>\n"
        f"<code>{html.escape(settings['system_prompt'][:200])}</code>\n"
        f"{'<i>...(truncated)</i>' if len(settings['system_prompt']) > 200 else ''}"
    )