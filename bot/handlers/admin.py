from __future__ import annotations

import asyncio
import datetime
import functools
import html
import logging
from typing import Any

from aiogram import F, Router
from aiogram.exceptions import TelegramForbiddenError, TelegramRetryAfter
from aiogram.filters import Command
from aiogram.types import CallbackQuery, Message
from aiogram.utils.keyboard import InlineKeyboardBuilder

from bot.config import config
from bot.services.telegram_utils import safe_answer_callback, safe_edit_text
from bot.services.user_service import user_service

logger = logging.getLogger(__name__)

router = Router()
USERS_PER_PAGE = 15


def _admin_required(func):
    """Decorator to restrict commands to admins only."""

    @functools.wraps(func)
    async def wrapper(message: Message, *args, **kwargs):
        user = message.from_user
        if user is None or not config.is_admin(user.id):
            await message.answer(
                "⛔ <b>Access denied.</b> This command is for admins only."
            )
            return
        return await func(message, *args, **kwargs)

    return wrapper


@router.message(Command("add_user"))
@_admin_required
async def cmd_add_user(message: Message) -> None:
    """Handle /add_user &lt;id&gt; — add a user by ID to the allowed list."""
    user = message.from_user
    if user is None:
        return

    text = message.text or ""
    parts = text.split(maxsplit=1)

    if len(parts) < 2:
        await message.answer(
            "❌ <b>Usage:</b> <code>/add_user &lt;telegram_id&gt;</code>\n\n"
            "Example: <code>/add_user 123456789</code>"
        )
        return

    try:
        target_id = int(parts[1].strip())
    except ValueError:
        await message.answer(
            "❌ <b>Invalid ID.</b> Please provide a valid numeric Telegram ID.\n\n"
            "Example: <code>/add_user 123456789</code>"
        )
        return

    if config.is_admin(target_id):
        await message.answer("⚠️ <b>Admins already have access.</b> No need to add them manually.")
        return

    async with user_service() as service:
        await service.add_allowed_user(telegram_id=target_id, added_by=user.id)

    await message.answer(
        f"✅ <b>User added to allowed list.</b>\n\n"
        f"👤 <b>ID:</b> <code>{target_id}</code>\n"
        f"📊 <b>Daily limit:</b> {config.daily_limit} questions"
    )


@router.message(Command("remove_user"))
@_admin_required
async def cmd_remove_user(message: Message) -> None:
    """Handle /remove_user &lt;id&gt; — remove a user from the allowed list."""
    text = message.text or ""
    parts = text.split(maxsplit=1)

    if len(parts) < 2:
        await message.answer(
            "❌ <b>Usage:</b> <code>/remove_user &lt;telegram_id&gt;</code>\n\n"
            "Example: <code>/remove_user 123456789</code>"
        )
        return

    try:
        target_id = int(parts[1].strip())
    except ValueError:
        await message.answer(
            "❌ <b>Invalid ID.</b> Please provide a valid numeric Telegram ID.\n\n"
            "Example: <code>/remove_user 123456789</code>"
        )
        return

    async with user_service() as service:
        removed = await service.remove_allowed_user(target_id)

    if removed:
        await message.answer(
            f"✅ <b>User removed from allowed list.</b>\n\n"
            f"👤 <b>ID:</b> <code>{target_id}</code>"
        )
    else:
        await message.answer(
            f"⚠️ <b>User not found.</b>\n\n"
            f"User with ID <code>{target_id}</code> was not in the allowed list."
        )


@router.message(Command("list_users"))
@_admin_required
async def cmd_list_users(message: Message) -> None:
    """Handle /list_users — list all allowed users with their info."""
    text = message.text or ""
    parts = text.split()
    page = 0
    if len(parts) > 1:
        try:
            page = max(0, int(parts[1]) - 1)
        except ValueError:
            pass

    async with user_service() as service:
        allowed_users = await service.get_allowed_users()

    if not allowed_users:
        await message.answer(
            "📭 <b>Allowed users list is empty.</b>\n\n"
            "Use <code>/add_user &lt;id&gt;</code> to add users."
        )
        return

    total = len(allowed_users)
    total_pages = (total + USERS_PER_PAGE - 1) // USERS_PER_PAGE
    page = min(page, total_pages - 1)
    subset = allowed_users[page * USERS_PER_PAGE : (page + 1) * USERS_PER_PAGE]

    lines = [
        (
            f"📋 <b>Allowed Users</b> "
            f"(page {page + 1}/{total_pages}, {total} total)\n"
        )
    ]
    start_num = page * USERS_PER_PAGE
    for i, au in enumerate(subset, start_num + 1):
        name_parts = []
        if au["first_name"]:
            name_parts.append(html.escape(au["first_name"]))
        if au["username"]:
            name_parts.append(f"@{html.escape(au['username'])}")
        name_str = f" ({' '.join(name_parts)})" if name_parts else ""

        added_at_str = au["added_at"].strftime("%Y-%m-%d %H:%M") if au["added_at"] else "?"
        lines.append(
            f"{i}. <code>{au['telegram_id']}</code>{name_str}\n"
            f"   🕐 Added: {added_at_str}"
        )

    if total_pages > 1:
        lines.append(f"\n<i>Page {page + 1}/{total_pages}. "
                      f"Use /list_users &lt;number&gt; for another page.</i>")

    await message.answer("\n".join(lines))


@router.message(Command("admin_stats"))
@_admin_required
async def cmd_admin_stats(message: Message) -> None:
    """Handle /admin_stats — show usage statistics for last 14 days."""
    async with user_service() as service:
        stats_data = await service.get_usage_stats_last_14_days()

    if not stats_data:
        await message.answer(
            "📊 <b>Usage Statistics (last 14 days)</b>\n\n"
            "No usage data available yet."
        )
        return

    # Aggregate by user
    user_stats: dict[int, dict] = {}
    for entry in stats_data:
        tid = entry["telegram_id"]
        if tid not in user_stats:
            user_stats[tid] = {
                "telegram_id": tid,
                "total": 0,
                "dates": {},
            }
        user_stats[tid]["total"] += entry["query_count"]
        date_str = entry["date"].strftime("%Y-%m-%d") if isinstance(entry["date"], (datetime.date, datetime.datetime)) else str(entry["date"])
        user_stats[tid]["dates"][date_str] = entry["query_count"]

    # Build response
    lines = ["📊 <b>Usage Statistics (last 14 days)</b>\n"]
    sorted_users = sorted(user_stats.values(), key=lambda x: x["total"], reverse=True)

    for i, us in enumerate(sorted_users, 1):
        lines.append(
            f"{i}. <code>{us['telegram_id']}</code> — <b>{us['total']}</b> queries"
        )
        # Show daily breakdown for top users (up to 15)
        if i <= 15:
            date_lines = []
            for date_str in sorted(us["dates"].keys()):
                date_lines.append(f"{date_str}: {us['dates'][date_str]}")
            if date_lines:
                lines.append("   " + " | ".join(date_lines))

    lines.append(f"\n👥 <b>Total active users:</b> {len(sorted_users)}")
    lines.append(f"📝 <b>Total queries:</b> {sum(us['total'] for us in sorted_users)}")

    await message.answer("\n".join(lines))


@router.message(Command("reset_daily_usage"))
@_admin_required
async def cmd_reset_daily_usage(message: Message) -> None:
    """Handle /reset_daily_usage &lt;id&gt; — reset daily usage for a user."""
    text = message.text or ""
    parts = text.split(maxsplit=1)

    if len(parts) < 2:
        await message.answer(
            "❌ <b>Usage:</b> <code>/reset_daily_usage &lt;telegram_id&gt;</code>\n\n"
            "Example: <code>/reset_daily_usage 123456789</code>"
        )
        return

    try:
        target_id = int(parts[1].strip())
    except ValueError:
        await message.answer(
            "❌ <b>Invalid ID.</b> Please provide a valid numeric Telegram ID."
        )
        return

    async with user_service() as service:
        reset = await service.reset_daily_usage(target_id)

    if reset:
        await message.answer(
            f"✅ <b>Daily usage reset for user</b> <code>{target_id}</code>."
        )
    else:
        await message.answer(
            f"ℹ️ <b>No usage data found</b> for user <code>{target_id}</code> today."
        )


# --- Broadcast --------------------------------------------------------------

# Pending broadcasts await an explicit confirmation, keyed by the admin's ID.
_pending_broadcasts: dict[int, str] = {}
# Strong references to in-flight broadcast tasks: the event loop keeps only
# weak references, so an unreferenced task can be garbage-collected mid-send.
_broadcast_tasks: set[asyncio.Task[None]] = set()
_BROADCAST_DELAY = 0.05  # ~20 messages/second, under Telegram's global limit
_MAX_BROADCAST_LENGTH = 4096  # Telegram's hard message limit


def _broadcast_keyboard(owner_id: int) -> Any:
    """Confirmation keyboard bound to the admin who started the broadcast."""
    builder = InlineKeyboardBuilder()
    builder.button(text="✅ Send", callback_data=f"broadcast:send:{owner_id}")
    builder.button(text="❌ Cancel", callback_data=f"broadcast:cancel:{owner_id}")
    builder.adjust(2)
    return builder.as_markup()


@router.message(Command("broadcast"))
@_admin_required
async def cmd_broadcast(message: Message) -> None:
    """Handle /broadcast <text> — send an announcement to every registered user.

    A confirmation step protects against typos: the message is only sent after
    the *same* admin presses the button. Replying to a message with
    ``/broadcast`` uses that message's text.
    """
    user = message.from_user
    if user is None:
        return

    parts = (message.text or "").split(maxsplit=1)
    if len(parts) > 1 and parts[1].strip():
        text = parts[1].strip()
    elif message.reply_to_message is not None:
        text = (message.reply_to_message.text or message.reply_to_message.caption or "").strip()
    else:
        text = ""

    if not text:
        await message.answer(
            "❌ <b>Usage:</b> <code>/broadcast &lt;text&gt;</code>\n\n"
            "You can also reply to any message with <code>/broadcast</code>."
        )
        return

    if len(text) > _MAX_BROADCAST_LENGTH:
        await message.answer(
            f"❌ <b>The announcement is too long</b> "
            f"({len(text)} characters; Telegram's limit is {_MAX_BROADCAST_LENGTH})."
        )
        return

    async with user_service() as service:
        recipients = await service.get_all_user_ids()

    if not recipients:
        await message.answer("ℹ️ <b>There are no registered users to notify.</b>")
        return

    # A new draft replaces the previous one so a stale confirmation cannot send
    # text the admin has already discarded.
    _pending_broadcasts[user.id] = text

    preview = text if len(text) <= 500 else text[:500] + "…"
    await message.answer(
        f"📣 <b>Broadcast preview</b> ({len(recipients)} recipients):\n\n"
        f"<blockquote>{html.escape(preview)}</blockquote>\n\n"
        "<i>The message is sent as plain text. Confirm?</i>",
        reply_markup=_broadcast_keyboard(user.id),
    )


def _pending_owner(callback: CallbackQuery, action: str) -> int | None:
    """Return the owner id when the pressing admin owns this broadcast.

    Callback data can be forged by a raw MTProto client, so the owner id is
    carried in the payload and compared with the pressing user: only the admin
    who drafted the announcement may confirm or cancel it.
    """
    data = callback.data or ""
    prefix = f"broadcast:{action}:"
    if not data.startswith(prefix) or callback.from_user is None:
        return None
    try:
        owner = int(data[len(prefix) :])
    except ValueError:
        return None
    if owner != callback.from_user.id or not config.is_admin(owner):
        return None
    return owner


@router.callback_query(F.data.startswith("broadcast:cancel:"))
async def cb_broadcast_cancel(callback: CallbackQuery) -> None:
    """Discard the pending broadcast (owner only)."""
    owner = _pending_owner(callback, "cancel")
    if owner is None:
        await safe_answer_callback(callback)
        return
    _pending_broadcasts.pop(owner, None)
    if callback.message is not None:
        await safe_edit_text(callback.message, "❌ <b>Broadcast cancelled.</b>")
    await safe_answer_callback(callback, "Cancelled")


@router.callback_query(F.data.startswith("broadcast:send:"))
async def cb_broadcast_send(callback: CallbackQuery) -> None:
    """Deliver the pending broadcast to every registered user (owner only).

    The delivery loop runs as a background task: with thousands of recipients
    it takes minutes, which must not hold up the callback handler.
    """
    owner = _pending_owner(callback, "send")
    if owner is None:
        await safe_answer_callback(callback)
        return

    text = _pending_broadcasts.pop(owner, None)
    if not text:
        await safe_answer_callback(
            callback, "Nothing to send — start with /broadcast", show_alert=True
        )
        return

    await safe_answer_callback(callback, "Sending…")

    async with user_service() as service:
        recipients = await service.get_all_user_ids()

    notice = callback.message
    if notice is not None:
        await safe_edit_text(
            notice, f"📣 <b>Sending the broadcast to {len(recipients)} users…</b>"
        )

    task = asyncio.create_task(
        _deliver_broadcast(callback.bot, recipients, text, notice),
        name=f"broadcast-{owner}",
    )
    _broadcast_tasks.add(task)
    task.add_done_callback(_broadcast_tasks.discard)


async def _deliver_broadcast(
    bot: Any, recipients: list[int], text: str, notice: Message | None
) -> None:
    """Send ``text`` to every recipient, tolerating individual failures."""
    sent = 0
    failed = 0
    for chat_id in recipients:
        for attempt in range(3):
            try:
                await bot.send_message(chat_id, text, parse_mode=None)
                sent += 1
                break
            except TelegramRetryAfter as exc:
                # Global flood control: honour the requested delay and retry.
                if attempt >= 2:
                    failed += 1
                    break
                await asyncio.sleep(float(exc.retry_after) + 0.1)
            except TelegramForbiddenError:
                # The user blocked the bot: nothing to retry.
                failed += 1
                break
            except Exception as exc:  # noqa: BLE001 - one bad recipient must not stop the rest
                failed += 1
                logger.warning("Broadcast to %s failed: %s", chat_id, exc)
                break
        await asyncio.sleep(_BROADCAST_DELAY)

    logger.info("Broadcast finished: %d sent, %d failed", sent, failed)
    if notice is not None:
        await safe_edit_text(
            notice,
            f"📣 <b>Broadcast finished.</b>\n\n"
            f"✅ Delivered: {sent}\n"
            f"⚠️ Failed: {failed}\n"
            f"👥 Total: {len(recipients)}",
        )
