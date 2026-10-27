from __future__ import annotations

from aiogram.types import KeyboardButton, ReplyKeyboardMarkup

from bot.config import config
from bot.services.user_service import user_service

# Тексты кнопок — единственный источник: клавиатура и обработчики
# (bot/handlers/menu.py) сравнивают текст сообщения с этими константами.
BTN_MODELS = "🤖 Models"
BTN_PERSONA = "🎭 Persona"
BTN_IMAGE = "🎨 Image"
BTN_REMINDERS = "⏰ Reminders"
BTN_EXPORT = "📤 Export"
BTN_CLEAR = "🧹 Clear"
BTN_SETTINGS = "⚙️ Settings"
BTN_STATS = "📊 Stats"
BTN_HELP = "❓ Help"
BTN_ADMIN = "👑 Admin"

ALL_BUTTONS = frozenset(
    {
        BTN_MODELS,
        BTN_PERSONA,
        BTN_IMAGE,
        BTN_REMINDERS,
        BTN_EXPORT,
        BTN_CLEAR,
        BTN_SETTINGS,
        BTN_STATS,
        BTN_HELP,
        BTN_ADMIN,
    }
)

_PLACEHOLDER = "Ask anything or pick an action…"


async def menu_for_user(user_id: int) -> ReplyKeyboardMarkup | None:
    """The menu for ``user_id``, or None when it is switched off or not allowed.

    Only admins and whitelisted users get it: guests and strangers can only
    chat, so the buttons would lead them to refusals.
    """
    if not config.enable_reply_menu:
        return None
    is_admin = config.is_admin(user_id)
    if not is_admin:
        async with user_service() as service:
            if not await service.is_allowed_user(user_id):
                return None
    return get_main_menu_keyboard(is_admin)


def get_main_menu_keyboard(is_admin: bool = False) -> ReplyKeyboardMarkup:
    """Persistent reply keyboard with the main actions (2 columns).

    Buttons of disabled features (images, reminders, export) are left out.
    """
    buttons = [BTN_MODELS, BTN_PERSONA]
    if config.enable_image_generation:
        buttons.append(BTN_IMAGE)
    if config.enable_reminders:
        buttons.append(BTN_REMINDERS)
    if config.enable_export:
        buttons.append(BTN_EXPORT)
    buttons += [BTN_CLEAR, BTN_SETTINGS, BTN_STATS, BTN_HELP]
    if is_admin:
        buttons.append(BTN_ADMIN)

    rows = [
        [KeyboardButton(text=text) for text in buttons[i : i + 2]]
        for i in range(0, len(buttons), 2)
    ]
    return ReplyKeyboardMarkup(
        keyboard=rows,
        resize_keyboard=True,
        is_persistent=True,
        input_field_placeholder=_PLACEHOLDER,
    )
