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


def get_enhance_choice_keyboard() -> InlineKeyboardMarkup:
    """Ask where to enhance the photo: local server pipeline or AI redraw."""
    builder = InlineKeyboardBuilder()
    builder.button(text="🖥️ Server — fast, content untouched", callback_data="enh:local")
    builder.button(text="🧠 Neural network — AI redraw", callback_data="enh:ai")
    builder.adjust(1)
    return builder.as_markup()


def get_ask_other_model_keyboard() -> InlineKeyboardMarkup:
    """Keyboard shown under an answer: re-ask the question with another model."""
    builder = InlineKeyboardBuilder()
    builder.button(text="🔀 Ask another model", callback_data="askother")
    builder.adjust(1)
    return builder.as_markup()


def get_answer_keyboard(
    export_id: int | None = None, *, ask_other: bool = False
) -> InlineKeyboardMarkup | None:
    """Action row under an AI answer: 📄 PDF / 📝 MD export + re-ask.

    ``export_id`` is the database id of the saved assistant message; the
    export callbacks carry it so the handler can reload the exact answer
    (with an owner check) instead of trusting callback text. Returns None
    when there would be no buttons at all.
    """
    if export_id is None and not ask_other:
        return None
    builder = InlineKeyboardBuilder()
    if export_id is not None:
        builder.button(text="📄 PDF", callback_data=f"expans:pdf:{export_id}")
        builder.button(text="📝 MD", callback_data=f"expans:md:{export_id}")
    if ask_other:
        builder.button(text="🔀 Ask another model", callback_data="askother")
    builder.adjust(2)
    return builder.as_markup()


def get_export_format_keyboard() -> InlineKeyboardMarkup:
    """Format picker for a full-history ``/export`` (PDF or Markdown)."""
    builder = InlineKeyboardBuilder()
    builder.button(text="📄 PDF", callback_data="expall:pdf")
    builder.button(text="📝 Markdown", callback_data="expall:md")
    builder.adjust(2)
    return builder.as_markup()


def get_resend_model_keyboard() -> InlineKeyboardMarkup:
    """Model picker used to re-ask the previous question once."""
    builder = InlineKeyboardBuilder()
    for model in ai_client.available_models():
        builder.button(text=model["name"], callback_data=f"resend:{model['id']}")
    builder.adjust(2)
    return builder.as_markup()


def get_persona_keyboard(
    personas: dict[str, dict[str, str]], current_prompt: str
) -> InlineKeyboardMarkup:
    """Persona picker for /persona; the active preset is marked with ✓."""
    builder = InlineKeyboardBuilder()
    for key, persona in personas.items():
        marker = " ✓" if persona.get("prompt") == current_prompt else ""
        builder.button(
            text=f"{persona.get('name', key)}{marker}",
            callback_data=f"persona:{key}",
        )
    builder.adjust(1)
    return builder.as_markup()