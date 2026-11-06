"""ACL samo_helper: только рабочий владелец и только его личный чат/группа."""

from __future__ import annotations

from typing import Protocol

from aiogram import Bot
from aiogram.types import CallbackQuery, Message


class OwnerIdentity(Protocol):
    owner_id: int


def private_owner_message(message: Message, config: OwnerIdentity) -> bool:
    return bool(
        message.from_user
        and message.from_user.id == config.owner_id
        and message.chat.type == "private"
        and message.chat.id == config.owner_id
    )


def samo_callback(
    callback: CallbackQuery,
    owner_id: int,
    notification_chat_id: int,
    bot: Bot,
    *,
    allow_legacy: bool = False,
) -> bool:
    """Кнопки samo_helper: только владелец и только на сообщениях этого бота."""
    data = callback.data or ""
    supported = data == "samo:done" or data.startswith("samo:rm:")
    if allow_legacy:
        supported = supported or data == "done" or data.startswith("rm:")
    if not supported:
        return False
    message = callback.message
    return bool(
        callback.from_user.id == owner_id
        and isinstance(message, Message)
        and message.from_user
        and message.from_user.id == bot.id
        and (
            (message.chat.type == "private" and message.chat.id == owner_id)
            or (
                message.chat.type in {"group", "supergroup"}
                and message.chat.id == notification_chat_id
            )
        )
    )
