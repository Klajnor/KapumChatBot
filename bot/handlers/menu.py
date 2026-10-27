"""Постоянная reply-клавиатура: кнопки меню и диалоги «Image» / «New reminder».

Нажатие кнопки приходит как обычное текстовое сообщение, поэтому роутер
подключается раньше catch-all ``chat.router``. Кнопки без аргументов вызывают
обработчики соответствующих команд; для /image и /remind запускается диалог
(FSM): бот спрашивает описание, следующее сообщение пользователя
обрабатывается как аргумент команды.
"""

from __future__ import annotations

import logging
import time

from aiogram import F, Router
from aiogram.dispatcher.event.bases import SkipHandler
from aiogram.enums import ChatType
from aiogram.filters import Command, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import CallbackQuery, Message, ReplyKeyboardRemove

from bot.config import config
from bot.handlers import admin_panel
from bot.handlers.clear import cmd_clear
from bot.handlers.export import cmd_export
from bot.handlers.help import cmd_help
from bot.handlers.image import generate_image_from_prompt
from bot.handlers.models import cmd_models
from bot.handlers.prompt import cmd_persona, cmd_settings
from bot.handlers.reminders import cmd_reminders, create_reminder_from_text
from bot.handlers.stats import cmd_stats
from bot.keyboards.inline import get_dialog_cancel_keyboard
from bot.keyboards.reply import (
    ALL_BUTTONS,
    BTN_ADMIN,
    BTN_CLEAR,
    BTN_EXPORT,
    BTN_HELP,
    BTN_IMAGE,
    BTN_MODELS,
    BTN_PERSONA,
    BTN_REMINDERS,
    BTN_SETTINGS,
    BTN_STATS,
    menu_for_user,
)
from bot.services.reminders import get_timezone
from bot.services.telegram_utils import (
    is_forwarded,
    safe_answer,
    safe_answer_callback,
    safe_edit_text,
)
from bot.services.user_service import user_service

logger = logging.getLogger(__name__)

router = Router()
# Клавиатура существует только в личке; в группах этот роутер молчит.
router.message.filter(F.chat.type == ChatType.PRIVATE)

_DIALOG_TTL = 600.0  # секунд, как у /timezone
_CANCEL_WORDS = frozenset({"cancel", "отмена", "стоп", "не надо"})

_IMAGE_PROMPT = (
    "🎨 <b>What should I draw?</b>\n\n"
    "Describe the image, e.g. <i>a red fox in a snowy forest</i>.\n"
    "Say <code>cancel</code> or press the button to stop."
)
_REMIND_PROMPT = (
    "⏰ <b>New reminder</b> — send <b>when</b> and <b>what</b> in one message:\n\n"
    "• <code>in 30 minutes check the oven</code>\n"
    "• <code>tomorrow at 9:00 standup</code>\n"
    "• <code>every 2 hours stretch</code>\n"
    "• <code>daily at 9:00 ai: five main AI news</code> — a scheduled AI request\n\n"
    "Times are interpreted in <b>{tz}</b> (change with /timezone).\n"
    "Say <code>cancel</code> or press the button to stop."
)
_NON_TEXT_HINT = (
    "Please send the answer as text, or say <code>cancel</code> to stop."
)


class MenuDialog(StatesGroup):
    """Диалоги, которые открывают кнопки меню."""

    image = State()
    remind = State()


def _is_cancel(text: str) -> bool:
    return text.strip().lower().rstrip("!.") in _CANCEL_WORDS


async def _start_dialog(state: FSMContext, dialog: State) -> None:
    await state.set_state(dialog)
    await state.update_data(started=time.monotonic())


async def _dialog_alive(state: FSMContext) -> bool:
    """False (and the state is dropped) when the dialog has timed out."""
    data = await state.get_data()
    if time.monotonic() - float(data.get("started", 0.0)) > _DIALOG_TTL:
        await state.clear()
        return False
    return True


def _as_command(message: Message, command: str) -> Message:
    """Copy of ``message`` whose text is ``command`` (handlers parse arguments)."""
    return message.model_copy(update={"text": command})


# --- Кнопки без диалога -------------------------------------------------------


@router.message(F.text.in_(ALL_BUTTONS))
async def on_menu_button(message: Message, state: FSMContext) -> None:
    """Route a menu button to the handler of the matching command."""
    user = message.from_user
    text = message.text or ""
    if user is None or is_forwarded(message):
        return  # a forwarded "button" is just someone else's text
    # A button always abandons whatever dialog was open before.
    await state.clear()
    admin_panel._awaiting.pop(user.id, None)

    if text == BTN_IMAGE:
        await _open_image_dialog(message, state)
    elif text == BTN_MODELS:
        await cmd_models(message)
    elif text == BTN_PERSONA:
        await cmd_persona(message)
    elif text == BTN_REMINDERS:
        await cmd_reminders(message)
    elif text == BTN_EXPORT:
        await cmd_export(_as_command(message, "/export"))
    elif text == BTN_CLEAR:
        await cmd_clear(message)
    elif text == BTN_SETTINGS:
        await cmd_settings(message)
    elif text == BTN_STATS:
        await cmd_stats(message)
    elif text == BTN_HELP:
        await cmd_help(message)
    elif text == BTN_ADMIN:
        await admin_panel.cmd_admin(message)


# --- Диалог «Image» ------------------------------------------------------------


async def _open_image_dialog(message: Message, state: FSMContext) -> None:
    if not config.enable_image_generation:
        await safe_answer(message, "🚫 <b>Image generation is disabled.</b>")
        return
    await _start_dialog(state, MenuDialog.image)
    await safe_answer(
        message, _IMAGE_PROMPT, reply_markup=get_dialog_cancel_keyboard()
    )


@router.message(MenuDialog.image, F.text, ~F.text.startswith("/"))
async def on_image_prompt(message: Message, state: FSMContext) -> None:
    """Second step of the image dialog: the text is the picture description."""
    if not await _dialog_alive(state):
        raise SkipHandler  # timed out — treat the text as an ordinary question
    text = message.text or ""
    if is_forwarded(message):
        await safe_answer(message, "⚠️ Type the description yourself, please.")
        return
    if _is_cancel(text):
        await state.clear()
        await safe_answer(message, "Cancelled.")
        return
    await state.clear()
    await generate_image_from_prompt(message, text)


# --- Диалог «New reminder» -----------------------------------------------------


async def _open_remind_dialog(message: Message, state: FSMContext, user_id: int) -> None:
    async with user_service() as service:
        tz = get_timezone(await service.get_reminder_timezone(user_id))
    await _start_dialog(state, MenuDialog.remind)
    await safe_answer(
        message,
        _REMIND_PROMPT.format(tz=tz),
        reply_markup=get_dialog_cancel_keyboard(),
    )


@router.message(MenuDialog.remind, F.text, ~F.text.startswith("/"))
async def on_remind_text(message: Message, state: FSMContext) -> None:
    """Second step of the reminder dialog: parse «when + what» like /remind."""
    if not await _dialog_alive(state):
        raise SkipHandler
    text = message.text or ""
    if is_forwarded(message):
        await safe_answer(
            message,
            "⚠️ <b>Reminders cannot be created from a forwarded message.</b>\n\n"
            "Type it yourself, or say <code>cancel</code>.",
        )
        return
    if _is_cancel(text):
        await state.clear()
        await safe_answer(message, "Cancelled.")
        return
    # On a parse error the user has already been told why; the dialog stays
    # open so the next message can be a corrected attempt.
    if await create_reminder_from_text(message, text):
        await state.clear()


# --- Выход из диалога ----------------------------------------------------------


@router.message(StateFilter(MenuDialog.image, MenuDialog.remind), F.text.startswith("/"))
async def on_command_in_dialog(message: Message, state: FSMContext) -> None:
    """A slash command abandons the dialog and then runs as usual."""
    await state.clear()
    raise SkipHandler


@router.message(StateFilter(MenuDialog.image, MenuDialog.remind), ~F.text)
async def on_non_text_in_dialog(message: Message) -> None:
    await safe_answer(message, _NON_TEXT_HINT)


@router.callback_query(F.data == "menu:cancel")
async def cb_dialog_cancel(callback: CallbackQuery, state: FSMContext) -> None:
    await state.clear()
    await safe_edit_text(callback.message, "Cancelled.")  # type: ignore[arg-type]
    await safe_answer_callback(callback)


@router.callback_query(F.data.startswith("rem:new:"))
async def cb_new_reminder(callback: CallbackQuery, state: FSMContext) -> None:
    """«➕ New reminder» under /reminders starts the reminder dialog."""
    message = callback.message
    try:
        owner = int((callback.data or "").rsplit(":", 1)[1])
    except (IndexError, ValueError):
        await safe_answer_callback(callback)
        return
    if callback.from_user is None or callback.from_user.id != owner:
        await safe_answer_callback(callback, "This is not your list.", show_alert=True)
        return
    if not config.enable_reminders:
        await safe_answer_callback(callback, "Reminders are disabled.", show_alert=True)
        return
    if message is None or message.chat.type != ChatType.PRIVATE:
        await safe_answer_callback(
            callback,
            "Use /remind here, or open a private chat with the bot.",
            show_alert=True,
        )
        return
    await safe_answer_callback(callback)
    await _open_remind_dialog(message, state, owner)  # type: ignore[arg-type]


# --- /menu, /hide_menu ---------------------------------------------------------


@router.message(Command("menu"))
async def cmd_menu(message: Message) -> None:
    """Show the persistent button menu."""
    user = message.from_user
    if user is None:
        return
    markup = await menu_for_user(user.id)
    if markup is None:
        await safe_answer(
            message,
            "ℹ️ The button menu is not available for you (or is switched off).",
        )
        return
    await safe_answer(message, "🧭 <b>Menu</b> is on — use the buttons below.", reply_markup=markup)


@router.message(Command("hide_menu"))
async def cmd_hide_menu(message: Message) -> None:
    """Remove the persistent button menu (bring it back with /menu)."""
    await safe_answer(
        message,
        "Menu hidden. Send /menu to bring it back.",
        reply_markup=ReplyKeyboardRemove(),
    )
