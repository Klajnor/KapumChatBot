from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from contextlib import contextmanager

from aiogram import F, Router
from aiogram.exceptions import TelegramAPIError, TelegramBadRequest, TelegramRetryAfter
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InputSticker,
)

from bot.bot import get_bot_username
from bot.config import config
from bot.services import limits, sticker_sets
from bot.services.access import check_access
from bot.services.telegram_utils import safe_answer_callback, safe_edit_text
from bot.services.user_service import user_service

logger = logging.getLogger(__name__)
router = Router()
_CALLBACK = re.compile(r"stk:(menu|new|add):([0-9a-f]{16})(?::([0-9]{1,9}))?\Z")
_active_users: set[tuple[int, int]] = set()


@contextmanager
def _set_slot(user_id: int, bot_id: int) -> Iterator[bool]:
    """Сериализовать изменения наборов даже при MAX_CONCURRENT_PER_USER > 1."""
    key = (user_id, bot_id)
    with limits.slot(user_id) as granted:
        if not granted or key in _active_users:
            yield False
            return
        _active_users.add(key)
        try:
            yield True
        finally:
            _active_users.discard(key)


@router.callback_query(F.data.startswith("stk:"))
async def sticker_set_action(callback: CallbackQuery) -> None:
    """Управлять только наборами и стикерами нажавшего, не расходуя квоту."""
    match = _CALLBACK.fullmatch(callback.data or "")
    if match is None or callback.message is None or not hasattr(callback.message, "edit_text"):
        await safe_answer_callback(callback, "This button is no longer available.", show_alert=True)
        return
    if not config.enable_sticker_generation:
        await safe_answer_callback(callback, "Sticker generation is disabled.", show_alert=True)
        return
    action, token, number = match.groups()
    if (action in {"menu", "add"}) != (number is not None):
        await safe_answer_callback(callback, "Invalid button.", show_alert=True)
        return
    user = callback.from_user
    sticker = await sticker_sets.get_sticker(token, user.id, callback.bot.id)
    if sticker is None:
        await safe_answer_callback(callback, "This is not your sticker or it is no longer available.", show_alert=True)
        return
    async with user_service() as service:
        access = await check_access(service, user.id, consume=False, require_quota=False)
    if not access.allowed:
        await safe_answer_callback(callback, "Access restricted.", show_alert=True)
        return
    with _set_slot(user.id, callback.bot.id) as granted:
        if not granted:
            await safe_answer_callback(callback, "Please wait for your current request to finish.", show_alert=True)
            return
        await safe_answer_callback(callback)
        try:
            if action == "menu":
                page = min(int(number), 100_000)
                packs = await sticker_sets.list_packs(user.id, callback.bot.id, page)
                await safe_edit_text(
                    callback.message,
                    "Choose one of your sets or create a new one.\n"
                    "Only sets created by this bot are listed.\n"
                    "New sets receive an automatic name; stickers use 🙂.",
                    reply_markup=sticker_sets.packs_keyboard(token, packs, page),
                )
                return
            if action == "new":
                username = get_bot_username()
                if not username:
                    await callback.message.answer("Bot identity is unavailable. Please try again later.")
                    return
                title = f"{user.first_name[:40]}'s stickers · {token[:6]}"
                pack = await sticker_sets.reserve_pack(sticker, username, title)
                try:
                    remote = await callback.bot.get_sticker_set(pack.name)
                except TelegramBadRequest as exc:
                    if "STICKERSET_INVALID" not in exc.message.upper():
                        raise
                    await callback.bot.create_new_sticker_set(
                        user_id=user.id, name=pack.name, title=pack.title,
                        stickers=[InputSticker(sticker=sticker.file_id, format="static", emoji_list=["🙂"])],
                        sticker_type="regular",
                    )
                else:
                    # Повтор после сетевого сбоя: набор уже мог быть создан.
                    if not any(s.file_unique_id == sticker.file_unique_id for s in remote.stickers):
                        await callback.bot.add_sticker_to_set(
                            user_id=user.id, name=pack.name,
                            sticker=InputSticker(sticker=sticker.file_id, format="static", emoji_list=["🙂"]),
                        )
                await sticker_sets.mark_created(pack.id)
            else:
                pack = await sticker_sets.get_pack(int(number), user.id, callback.bot.id)
                if pack is None:
                    await callback.message.answer("This set is unavailable or does not belong to you.")
                    return
                remote = await callback.bot.get_sticker_set(pack.name)
                # Telegram file_unique_id позволяет не добавлять дубликат при повторном нажатии.
                if not any(s.file_unique_id == sticker.file_unique_id for s in remote.stickers):
                    await callback.bot.add_sticker_to_set(
                        user_id=user.id, name=pack.name,
                        sticker=InputSticker(sticker=sticker.file_id, format="static", emoji_list=["🙂"]),
                    )
            await safe_edit_text(
                callback.message,
                "✅ Sticker saved to your set.",
                reply_markup=InlineKeyboardMarkup(inline_keyboard=[[
                    InlineKeyboardButton(text="Open sticker set", url=f"https://t.me/addstickers/{pack.name}"),
                ], [
                    InlineKeyboardButton(text="Add to another set", callback_data=f"stk:menu:{token}:0"),
                ]]),
            )
        except TelegramRetryAfter as exc:
            await callback.message.answer(f"Telegram rate limit. Please try again in {exc.retry_after} seconds.")
        except TelegramAPIError as exc:
            logger.warning("Sticker set operation failed: %s", exc)
            await callback.message.answer(
                "Could not update the sticker set. It may be full, deleted or unavailable. "
                "Please try again or create a new set. If this happened in a group, "
                "open the bot privately and press Start first."
            )
