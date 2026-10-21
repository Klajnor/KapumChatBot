from __future__ import annotations

from aiogram import Router, F
from aiogram.enums import ChatType
from aiogram.filters import Command
from aiogram.types import CallbackQuery, Message

from bot.database.session import async_session_factory
from bot.keyboards.inline import get_clear_confirmation_keyboard
from bot.services.telegram_utils import safe_answer_callback, safe_edit_text
from bot.services.user_service import UserService

router = Router()


@router.message(Command("clear"))
async def cmd_clear(message: Message) -> None:
    """Handle /clear command — ask for confirmation."""
    keyboard = get_clear_confirmation_keyboard()
    await message.answer(
        "🧹 <b>Clear conversation history?</b>\n\n"
        "This will delete <b>your</b> history in <b>this chat</b> and start a fresh "
        "conversation. This action cannot be undone.",
        reply_markup=keyboard,
    )


@router.callback_query(F.data == "clear:confirm")
async def cb_clear_confirm(callback: CallbackQuery) -> None:
    """Handle clear confirmation."""
    if callback.from_user is None or callback.message is None:
        return

    # In a group the history is shared: a participant clears only their own part.
    chat = callback.message.chat
    own_only = None if chat.type == ChatType.PRIVATE else callback.from_user.id
    async with async_session_factory() as session:
        service = UserService(session)
        await service.clear_history(chat.id, user_id=own_only)

    await safe_edit_text(
        callback.message,  # type: ignore[arg-type]
        "✅ <b>Conversation history cleared!</b>\n\n"
        "Your chat history has been deleted. "
        "You can start a fresh conversation now.",
    )
    await safe_answer_callback(callback, "History cleared")


@router.callback_query(F.data == "clear:cancel")
async def cb_clear_cancel(callback: CallbackQuery) -> None:
    """Handle clear cancellation."""
    await safe_edit_text(
        callback.message,  # type: ignore[arg-type]
        "❌ <b>Clear cancelled.</b>\n\n"
        "Your conversation history is preserved.",
    )
    await safe_answer_callback(callback, "Cancelled")