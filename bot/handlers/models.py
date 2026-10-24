from __future__ import annotations

import html

from aiogram import F, Router
from aiogram.enums import ChatType
from aiogram.filters import Command
from aiogram.types import CallbackQuery, Message

from bot.keyboards.inline import get_model_selection_keyboard
from bot.services.ai_client import ai_client, is_vision_model
from bot.services.telegram_utils import (
    require_private_chat,
    safe_answer_callback,
    safe_edit_text,
)
from bot.services.user_service import user_service

router = Router()


@router.message(Command("models"))
async def cmd_models(message: Message) -> None:
    """Handle /models command — show the model selection keyboard."""
    lines = ["🤖 <b>Select an AI Model:</b>\n"]
    for model in ai_client.available_models():
        vision = " 🖼️" if is_vision_model(model["id"]) else ""
        lines.append(f"• <code>{html.escape(model['id'])}</code>{vision}")
    lines.append("\n🖼️ — supports images (vision)")
    lines.append("<i>Tap a model to select it:</i>")

    await message.answer(
        "\n".join(lines), reply_markup=get_model_selection_keyboard()
    )


@router.message(Command("model"))
async def cmd_model(message: Message) -> None:
    """Handle /model command — show the current model."""
    user = message.from_user
    if user is None:
        return
    # The selected model is a personal setting: don't expose it in a group.
    if not await require_private_chat(message, what="/model"):
        return

    async with user_service() as service:
        settings = await service.get_user_settings(user.id)

    await message.answer(
        f"🤖 <b>Current Model:</b> <code>{html.escape(settings['model'])}</code>"
    )


@router.callback_query(F.data.startswith("model:"))
async def cb_model_selected(callback: CallbackQuery) -> None:
    """Handle model selection callback."""
    if callback.data is None or callback.from_user is None:
        return
    # The confirmation names the model, so keep the picker private as well.
    if callback.message is not None and callback.message.chat.type != ChatType.PRIVATE:
        await safe_answer_callback(
            callback,
            "🔒 Model selection is available in a private chat only.",
            show_alert=True,
        )
        return

    model_id = callback.data.removeprefix("model:")
    available = {m["id"]: m["name"] for m in ai_client.available_models()}

    if model_id not in available:
        await safe_answer_callback(
            callback,
            "⚠️ This model is not available for your account.",
            show_alert=True,
        )
        return

    model_name = available[model_id]

    async with user_service() as service:
        await service.set_model(callback.from_user.id, model_id)

    vision = "\n🖼️ Supports images (vision)." if is_vision_model(model_id) else ""
    await safe_edit_text(
        callback.message,  # type: ignore[arg-type]
        f"✅ <b>Model changed to:</b> <code>{html.escape(model_name)}</code>"
        f"{vision}\n\nModel ID: <code>{html.escape(model_id)}</code>",
    )
    await safe_answer_callback(callback, f"Selected: {model_name}")