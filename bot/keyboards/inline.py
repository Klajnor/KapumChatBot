from __future__ import annotations

from aiogram.types import InlineKeyboardMarkup
from aiogram.utils.keyboard import InlineKeyboardBuilder

from bot.services.ai_client import ai_client


def get_model_selection_keyboard() -> InlineKeyboardMarkup:
    """Build an inline keyboard for selecting an AI model."""
    builder = InlineKeyboardBuilder()
    for model in ai_client.available_models():
        builder.button(
            text=model["name"],
            callback_data=f"model:{model['id']}",
        )
    # 2 columns for better layout
    builder.adjust(2)
    return builder.as_markup()


def get_clear_confirmation_keyboard() -> InlineKeyboardMarkup:
    """Build a confirmation keyboard for clearing history."""
    builder = InlineKeyboardBuilder()
    builder.button(text="✅ Yes, clear history", callback_data="clear:confirm")
    builder.button(text="❌ Cancel", callback_data="clear:cancel")
    builder.adjust(1)
    return builder.as_markup()


def get_settings_keyboard() -> InlineKeyboardMarkup:
    """Build a settings inline keyboard."""
    builder = InlineKeyboardBuilder()
    builder.button(text="🤖 Change Model", callback_data="settings:model")
    builder.button(text="📝 Change Prompt", callback_data="settings:prompt")
    builder.button(text="📊 Statistics", callback_data="settings:stats")
    builder.adjust(1)
    return builder.as_markup()