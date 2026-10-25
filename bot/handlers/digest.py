"""Collects the posts of digest channels (see ``services/digest.py``).

Channel posts arrive as ``channel_post`` updates (the bot must be a channel
admin). Group messages (and their edits) are observed by outer middlewares, so
the normal handlers still see them. A bot admin turns a group on or off with
``/digest_on`` / ``/digest_off`` written in the group itself (a group has no
forwardable posts to identify it by). In a group the bot only sees ordinary
messages when it is an admin there or has privacy mode disabled in BotFather.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from aiogram import BaseMiddleware, F, Router
from aiogram.filters import Command
from aiogram.types import Message

from bot.bot import get_bot_id
from bot.config import config
from bot.services import digest

logger = logging.getLogger(__name__)

router = Router()


def _author(message: Message) -> str:
    if message.from_user is not None and not message.sender_chat:
        return message.from_user.full_name
    if message.author_signature:
        return message.author_signature
    return ""  # an anonymous channel post: the channel itself is the author


def _is_own(message: Message) -> bool:
    """True for the bot's own messages (its answers and the posted summaries)."""
    bot_id = get_bot_id()
    if not bot_id:
        return False
    sender = message.from_user
    via = message.via_bot
    return (sender is not None and sender.id == bot_id) or (
        via is not None and via.id == bot_id
    )


async def _record(message: Message, *, edited: bool = False) -> None:
    if _is_own(message):
        return
    try:
        await digest.record_message(
            message.chat.id,
            message.message_id,
            _author(message),
            message.text or message.caption,
            message.date,
            edited=edited,
        )
    except Exception:
        logger.exception("Could not store a message for the digest")


@router.channel_post()
async def on_channel_post(message: Message) -> None:
    await _record(message)


@router.edited_channel_post()
async def on_edited_channel_post(message: Message) -> None:
    await _record(message, edited=True)


_GROUPS = ("group", "supergroup")


@router.message(Command("digest_on", "digest_off"), F.chat.type.in_(_GROUPS))
async def on_digest_toggle(message: Message) -> None:
    user = message.from_user
    if user is None or not config.is_admin(user.id):
        return  # stay silent for everyone else
    command = (message.text or "").split()[0].split("@")[0].lower()
    if command == "/digest_on":
        title = message.chat.title or str(message.chat.id)
        await digest.add_channel(message.chat.id, title, user.id)
        await message.answer(
            "📰 This group is now in the daily digest. I see ordinary messages "
            "only if I am an admin here or privacy mode is off in BotFather."
        )
    elif await digest.remove_channel(message.chat.id):
        await message.answer("📰 This group was removed from the daily digest.")


@router.message(F.migrate_to_chat_id)
async def on_group_migrated(message: Message) -> None:
    """A group became a supergroup: keep it in the digest under the new ID."""
    new_id = message.migrate_to_chat_id
    if new_id is None:
        return
    for chat_id, title in await digest.list_channels():
        if chat_id == message.chat.id:
            await digest.add_channel(new_id, title, None)
            await digest.remove_channel(chat_id)


class _GroupObserver(BaseMiddleware):
    def __init__(self, *, edited: bool = False) -> None:
        self.edited = edited

    async def __call__(
        self,
        handler: Callable[[Message, dict[str, Any]], Awaitable[Any]],
        event: Message,
        data: dict[str, Any],
    ) -> Any:
        if event.chat.type in _GROUPS:
            await _record(event, edited=self.edited)
        return await handler(event, data)


router.message.outer_middleware(_GroupObserver())
router.edited_message.outer_middleware(_GroupObserver(edited=True))
