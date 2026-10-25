"""Collects the posts of digest channels (see ``services/digest.py``).

Channel posts arrive as ``channel_post`` updates (the bot must be a channel
admin). Group messages are observed by an outer middleware, so the normal
handlers still see them. Nothing is answered here.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from aiogram import BaseMiddleware, Router
from aiogram.types import Message

from bot.services import digest

logger = logging.getLogger(__name__)

router = Router()


def _author(message: Message) -> str:
    if message.from_user is not None and not message.sender_chat:
        return message.from_user.full_name
    if message.author_signature:
        return message.author_signature
    return ""  # an anonymous channel post: the channel itself is the author


async def _record(message: Message, *, edited: bool = False) -> None:
    try:
        await digest.record_message(
            message.chat.id,
            message.message_id,
            _author(message),
            message.text or message.caption,
            message.edit_date if edited and message.edit_date else message.date,
            edited=edited,
        )
    except Exception:  # noqa: BLE001 - observing must never break the bot
        logger.exception("Could not store a message for the digest")


@router.channel_post()
async def on_channel_post(message: Message) -> None:
    await _record(message)


@router.edited_channel_post()
async def on_edited_channel_post(message: Message) -> None:
    await _record(message, edited=True)


class _GroupObserver(BaseMiddleware):
    async def __call__(
        self,
        handler: Callable[[Message, dict[str, Any]], Awaitable[Any]],
        event: Message,
        data: dict[str, Any],
    ) -> Any:
        if event.chat.type in ("group", "supergroup"):
            await _record(event)
        return await handler(event, data)


router.message.outer_middleware(_GroupObserver())
