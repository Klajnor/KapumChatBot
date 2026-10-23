"""Reminder commands: /remind, /reminders, /delreminder.

Creating a reminder is free (no daily quota is consumed), but only admins and
whitelisted users may create them — guests cannot. The number of one-time and
recurring reminders is capped per user (REMINDER_LIMIT /
REMINDER_RECURRING_LIMIT), and recurring ones cannot fire more often than
REMINDER_MIN_INTERVAL_MINUTES.
"""

from __future__ import annotations

import asyncio
import datetime
import html
import logging

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from bot.config import config
from bot.database.models import Reminder
from bot.services.access import check_access
from bot.services.reminders import (
    ParsedReminder,
    describe_schedule,
    get_timezone,
    parse_reminder,
    remind_usage,
)
from bot.services.telegram_utils import safe_answer
from bot.services.user_service import user_service

logger = logging.getLogger(__name__)
router = Router()

# Serialises "count then insert" per user. The cap check and the insert are two
# statements, so without a lock two concurrent /remind commands can both pass
# the check and overshoot REMINDER_LIMIT.
_create_locks: dict[int, asyncio.Lock] = {}
_CREATE_LOCKS_CAP = 500


def _create_lock(user_id: int) -> asyncio.Lock:
    """Per-user lock for reminder creation (evicts idle locks, never held ones)."""
    lock = _create_locks.get(user_id)
    if lock is None:
        lock = asyncio.Lock()
        _create_locks[user_id] = lock
    if len(_create_locks) > _CREATE_LOCKS_CAP:
        for key in list(_create_locks):
            candidate = _create_locks[key]
            if not candidate.locked():
                _create_locks.pop(key, None)
            if len(_create_locks) <= _CREATE_LOCKS_CAP:
                break
    return lock


def _format_fire_time(fire_at: datetime.datetime) -> str:
    """Fire time in the configured display timezone."""
    if fire_at.tzinfo is None:
        # SQLite hands back naive datetimes; stored times are always UTC.
        fire_at = fire_at.replace(tzinfo=datetime.UTC)
    return fire_at.astimezone(get_timezone()).strftime("%d.%m.%Y %H:%M %Z")


def _confirmation(reminder_id: int, parsed: ParsedReminder) -> str:
    """The 'reminder created' reply."""
    icon = "🔁" if parsed.schedule else "⏰"
    lines = [
        f"✅ <b>Reminder #{reminder_id} created</b> {icon}",
        "",
        f"<i>{html.escape(parsed.text[:300])}</i>",
        f"🕒 First fire: {_format_fire_time(parsed.fire_at)}",
    ]
    if parsed.schedule:
        lines.append(f"🔁 Schedule: {html.escape(describe_schedule(parsed.schedule))}")
    if parsed.ai_mode:
        lines.append(
            "🤖 <b>AI request</b> — at fire time this prompt is sent to your "
            "model and the answer is delivered here."
        )
    lines.append("")
    lines.append("Manage: /reminders · /delreminder " + str(reminder_id))
    return "\n".join(lines)


@router.message(Command("remind"))
async def cmd_remind(message: Message) -> None:
    """Create a one-time or recurring reminder (optionally an AI request)."""
    user = message.from_user
    if user is None:
        return
    if not config.enable_reminders:
        await safe_answer(message, "⚠️ Reminders are disabled on this bot.")
        return

    parts = (message.text or "").split(maxsplit=1)
    if len(parts) < 2 or not parts[1].strip():
        await safe_answer(message, remind_usage())
        return

    parsed = parse_reminder(parts[1])
    if isinstance(parsed, str):  # an error/help message for the user
        await safe_answer(message, parsed)
        return

    async with user_service() as service:
        await service.register_user(
            telegram_id=user.id,
            username=user.username,
            first_name=user.first_name,
            last_name=user.last_name,
        )
        # Creation is free, but only for authorized users (no guests):
        # consume=False + require_quota=False checks the *role* without touching
        # the daily budget — a user who spent today's questions can still create
        # a reminder, because it costs nothing.
        access = await check_access(
            service, user.id, consume=False, require_quota=False
        )
        if not access.allowed or access.role == "guest":
            await safe_answer(
                message,
                access.message
                or (
                    "⚠️ <b>Reminders are for authorized users only.</b>\n\n"
                    "Guests cannot create reminders."
                ),
            )
            return

        recurring = parsed.schedule is not None
        limit = (
            config.reminder_recurring_limit if recurring else config.reminder_limit
        )
        async with _create_lock(user.id):
            count = await service.count_reminders(user.id, recurring=recurring)
            if count >= limit:
                kind = "Recurring reminders" if recurring else "Reminders"
                await safe_answer(
                    message,
                    f"⚠️ <b>Limit reached:</b> {kind} {count}/{limit}.\n\n"
                    "Delete some first — see /reminders and /delreminder.",
                )
                return

            created = await service.add_reminder(
                Reminder(
                    user_id=user.id,
                    chat_id=message.chat.id,
                    text=parsed.text,
                    ai_mode=parsed.ai_mode,
                    fire_at=parsed.fire_at,
                    schedule=parsed.schedule,
                )
            )
            reminder_id = created.id

    await safe_answer(message, _confirmation(reminder_id, parsed))


@router.message(Command("reminders"))
async def cmd_reminders(message: Message) -> None:
    """List the user's active reminders."""
    user = message.from_user
    if user is None:
        return
    if not config.enable_reminders:
        await safe_answer(message, "⚠️ Reminders are disabled on this bot.")
        return

    async with user_service() as service:
        reminders = await service.list_reminders(user.id)
        # Snapshot the rows into plain data inside the session.
        rows = [
            {
                "id": r.id,
                "text": r.text,
                "fire_at": r.fire_at,
                "schedule": r.schedule,
                "ai_mode": r.ai_mode,
            }
            for r in reminders
        ]

    if not rows:
        await safe_answer(
            message,
            "📭 <b>No active reminders.</b>\n\n"
            "Create one with /remind — the full format list is in /help.",
        )
        return

    one_time = sum(1 for row in rows if row["schedule"] is None)
    recurring = len(rows) - one_time
    lines = [
        (
            f"⏰ <b>Your reminders</b> — one-time: {one_time}/{config.reminder_limit}, "
            f"recurring: {recurring}/{config.reminder_recurring_limit}"
        ),
        "",
    ]
    for row in rows:
        icon = "🔁" if row["schedule"] else "⏰"
        ai = " · 🤖 AI" if row["ai_mode"] else ""
        when = _format_fire_time(row["fire_at"])
        schedule = f" · {describe_schedule(row['schedule'])}" if row["schedule"] else ""
        lines.append(
            f"<b>#{row['id']}</b> {icon}{ai}{schedule}\n"
            f"Next: {when}\n"
            f"<i>{html.escape(row['text'][:150])}</i>"
        )
    lines.append("")
    lines.append(
        "Delete: <code>/delreminder N</code> or <code>/delreminder all</code>"
    )
    await safe_answer(message, "\n\n".join(lines))


@router.message(Command("delreminder"))
async def cmd_delreminder(message: Message) -> None:
    """Delete one reminder by id, or all of them: /delreminder 12 | all."""
    user = message.from_user
    if user is None:
        return
    if not config.enable_reminders:
        await safe_answer(message, "⚠️ Reminders are disabled on this bot.")
        return

    parts = (message.text or "").split()
    if len(parts) < 2 or not parts[1].strip():
        await safe_answer(
            message,
            "Usage: <code>/delreminder N</code> or <code>/delreminder all</code>"
            "\n\nSee your reminders with /reminders.",
        )
        return

    argument = parts[1].strip().lower()
    async with user_service() as service:
        if argument in ("all", "*"):
            count = await service.delete_all_reminders(user.id)
            await safe_answer(message, f"✅ Deleted {count} reminder(s).")
            return
        if not argument.isdigit():
            await safe_answer(
                message,
                "⚠️ Pass the reminder number: <code>/delreminder 12</code> "
                "or <code>/delreminder all</code>.",
            )
            return
        deleted = await service.delete_reminder(int(argument), user.id)

    if deleted:
        await safe_answer(message, f"✅ Reminder <b>#{html.escape(argument)}</b> deleted.")
    else:
        await safe_answer(
            message,
            f"⚠️ Reminder <b>#{html.escape(argument)}</b> not found — "
            "see /reminders for the list.",
        )
