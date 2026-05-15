from __future__ import annotations

import html

from aiogram import Router, F
from aiogram.filters import Command
from aiogram.types import CallbackQuery, Message

from bot.database.session import async_session_factory
from bot.keyboards.inline import get_model_selection_keyboard
from bot.services.openrouter import AVAILABLE_MODELS
from bot.services.user_service import UserService

router = Router()


@router.message(Command("models"))
async def cmd_models(message: Message) -> None:
    """Handle /models command — show model selection keyboard."""
    keyboard = get_model_selection_keyboard()
    await message.answer(
        "🤖 <b>Select an AI Model:</b>\n\n"
        "Choose a model from the list below. "
        "Different models have different strengths:\n"
        "• <b>GPT-4o</b> — Best all-rounder, supports vision\n"
        "• <b>Claude 3.5 Sonnet</b> — Excellent for reasoning\n"
        "• <b>DeepSeek</b> — Great for coding\n"
        "• <b>Gemini</b> — Fast and capable\n\n"
        "<i>Tap a model to select it:</i>",
        reply_markup=keyboard,
    )


@router.message(Command("model"))
async def cmd_model(message: Message) -> None:
    """Handle /model command — show current model."""
    user = message.from_user
    if user is None:
        return

    async with async_session_factory() as session:
        service = UserService(session)
        settings = await service.get_user_settings(user.id)

    await message.answer(
        f"🤖 <b>Current Model:</b> <code>{html.escape(settings['model'])}</code>"
    )


@router.callback_query(F.data.startswith("model:"))
async def cb_model_selected(callback: CallbackQuery) -> None:
    """Handle model selection callback."""
    if callback.data is None or callback.from_user is None:
        return

    model_id = callback.data.replace("model:", "")

    # Find model name
    model_name = model_id
    for m in AVAILABLE_MODELS:
        if m["id"] == model_id:
            model_name = m["name"]
            break

    async with async_session_factory() as session:
        service = UserService(session)
        await service.set_model(callback.from_user.id, model_id)

    await callback.message.edit_text(  # type: ignore[union-attr]
        f"✅ <b>Model changed to:</b> <code>{html.escape(model_name)}</code>\n\n"
        f"Model ID: <code>{html.escape(model_id)}</code>"
    )
    await callback.answer(f"Selected: {model_name}")