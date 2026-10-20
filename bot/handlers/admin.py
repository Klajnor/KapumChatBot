from __future__ import annotations

import datetime
import functools
import html
import logging

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from bot.config import config
from bot.database.session import async_session_factory
from bot.services.user_service import UserService

logger = logging.getLogger(__name__)

router = Router()


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

    async with async_session_factory() as session:
        service = UserService(session)
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

    async with async_session_factory() as session:
        service = UserService(session)
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
    async with async_session_factory() as session:
        service = UserService(session)
        allowed_users = await service.get_allowed_users()

    if not allowed_users:
        await message.answer(
            "📭 <b>Allowed users list is empty.</b>\n\n"
            "Use <code>/add_user &lt;id&gt;</code> to add users."
        )
        return

    lines = ["📋 <b>Allowed Users</b>\n"]
    for i, au in enumerate(allowed_users, 1):
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

    await message.answer("\n".join(lines))


@router.message(Command("admin_stats"))
@_admin_required
async def cmd_admin_stats(message: Message) -> None:
    """Handle /admin_stats — show usage statistics for last 14 days."""
    async with async_session_factory() as session:
        service = UserService(session)
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

    async with async_session_factory() as session:
        service = UserService(session)
        reset = await service.reset_daily_usage(target_id)

    if reset:
        await message.answer(
            f"✅ <b>Daily usage reset for user</b> <code>{target_id}</code>."
        )
    else:
        await message.answer(
            f"ℹ️ <b>No usage data found</b> for user <code>{target_id}</code> today."
        )
