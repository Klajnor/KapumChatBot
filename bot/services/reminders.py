"""Reminder scheduling: parsing, next-occurrence math and firing.

One-time reminders (``schedule IS NULL``) fire once and are deleted;
recurring ones are advanced after every fire according to ``schedule``:
``every:<minutes>``, ``daily:<HH>:<MM>`` or ``weekly:<dow>:<HH>:<MM>``
(dow: 0 = Monday). All stored times are UTC; user-facing times use
``REMINDER_TIMEZONE``.

The scheduler *claims* due reminders in the database (advances or deletes
them) **before** firing, so a slow or failing delivery can never cause a
double fire on the next tick. Fires run as independent tasks: one slow AI
request does not hold up the other reminders.

A reminder in ``ai_mode`` sends its stored prompt to the user's model at
fire time and delivers the answer. Scheduled AI requests do NOT consume the
interactive daily quota — their cost is bounded by the reminder limits
(``REMINDER_LIMIT`` / ``REMINDER_RECURRING_LIMIT`` / minimum interval).
"""

from __future__ import annotations

import asyncio
import datetime
import html
import logging
import re
from dataclasses import dataclass

from aiogram.exceptions import (
    TelegramAPIError,
    TelegramBadRequest,
    TelegramForbiddenError,
    TelegramRetryAfter,
)

from bot.bot import bot
from bot.config import config
from bot.database.repository import ReminderRepository, UserRepository
from bot.database.session import async_session_factory
from bot.services.access import check_access
from bot.services.ai_client import ai_client
from bot.services.markdown import render, split_chunks
from bot.services.telegram_utils import format_timing, throttle
from bot.services.user_service import UserService

logger = logging.getLogger(__name__)

# --- Time specification parsing ----------------------------------------------

# Accepted units for "in N <unit>" / "every N <unit>", in minutes.
_UNITS_MINUTES = {
    "m": 1, "min": 1, "mins": 1, "minute": 1, "minutes": 1,
    "h": 60, "hr": 60, "hrs": 60, "hour": 60, "hours": 60,
    "d": 1440, "day": 1440, "days": 1440,
    "w": 10080, "week": 10080, "weeks": 10080,
}

_WEEKDAYS = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}

# Upper bound for a parsed interval, in minutes (~10 years). The regexes accept
# arbitrarily many digits, and ``timedelta`` raises OverflowError far below
# ``sys.maxsize`` (already at ~10**12 minutes), which would escape the parser as
# an unhandled exception instead of a friendly message.
_MAX_INTERVAL_MINUTES = 5_259_600

# In-flight delivery tasks (strong references, see ``process_due``).
_fire_tasks: set[asyncio.Task[None]] = set()

# Every pattern is anchored at the start of the argument and eats its match,
# so the remainder is the reminder text.
_RE_EVERY_N = re.compile(
    r"^every\s+(\d+)\s*(minutes?|mins?|m|hours?|hrs?|h|days?|d)\b[\s,]*",
    re.IGNORECASE,
)
_RE_EVERY_HOUR = re.compile(r"^every\s+hour\b[\s,]*", re.IGNORECASE)
_RE_DAILY = re.compile(r"^daily\s+at\s+(\d{1,2}):(\d{2})\b[\s,]*", re.IGNORECASE)
_RE_WEEKLY = re.compile(
    r"^weekly\s+(mon|tue|wed|thu|fri|sat|sun)\w*\s+at\s+(\d{1,2}):(\d{2})\b[\s,]*",
    re.IGNORECASE,
)
_RE_AT = re.compile(r"^(tomorrow\s+)?at\s+(\d{1,2}):(\d{2})\b[\s,]*", re.IGNORECASE)
_RE_IN = re.compile(
    r"^in\s+(\d+)\s*(minutes?|mins?|m|hours?|hrs?|h|days?|d|weeks?|w)\b[\s,]*",
    re.IGNORECASE,
)


@dataclass(slots=True)
class ParsedReminder:
    """A successfully parsed ``/remind`` argument."""

    text: str                    # reminder text, or the AI prompt in ai_mode
    fire_at: datetime.datetime   # first fire, UTC (aware)
    schedule: str | None         # None = one-time
    ai_mode: bool = False


def get_timezone(name: str | None = None) -> datetime.tzinfo:
    """Resolve an IANA zone name (validated) or fall back to UTC.

    ``name=None`` selects the global ``REMINDER_TIMEZONE`` default; users
    override it individually with ``/timezone`` (stored in the DB).
    """
    name = (name or "").strip()
    if not name:
        name = (config.reminder_timezone or "").strip()
    if not name or name.upper() == "UTC":
        return datetime.UTC
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo(name)
    except Exception:  # noqa: BLE001 - unknown name or missing tzdata
        logger.warning("Unknown REMINDER_TIMEZONE %r; using UTC", name)
        return datetime.UTC


def _validate_hhmm(hour_text: str, minute_text: str) -> tuple[int, int] | None:
    hour, minute = int(hour_text), int(minute_text)
    if hour > 23 or minute > 59:
        return None
    return hour, minute


def _interval_too_long_error() -> str:
    """Friendly rejection for an interval that ``timedelta`` cannot represent."""
    return (
        "⚠️ <b>That interval is too far in the future.</b>\n\n"
        "Reminders can be scheduled at most ~10 years ahead."
    )


def _next_at(
    hour: int, minute: int, force_tomorrow: bool,
    now: datetime.datetime, tz: datetime.tzinfo,
) -> datetime.datetime:
    """Next occurrence of a wall-clock time in ``tz`` (UTC result)."""
    local_now = now.astimezone(tz)
    candidate = local_now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if force_tomorrow or candidate <= local_now:
        candidate += datetime.timedelta(days=1)
    return candidate.astimezone(datetime.UTC)


def _next_weekly(
    dow: int, hour: int, minute: int,
    now: datetime.datetime, tz: datetime.tzinfo,
) -> datetime.datetime:
    """Next occurrence of a weekday + wall-clock time in ``tz`` (UTC result)."""
    local_now = now.astimezone(tz)
    candidate = local_now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    candidate += datetime.timedelta(days=(dow - candidate.weekday()) % 7)
    if candidate <= local_now:
        candidate += datetime.timedelta(days=7)
    return candidate.astimezone(datetime.UTC)


def remind_usage(user_tz_name: str | None = None) -> str:
    """The /remind help text (also the error for an empty argument)."""
    tz = user_tz_name or config.reminder_timezone or "UTC"
    return (
        "⏰ <b>Set a reminder</b> — <code>/remind &lt;when&gt; &lt;text&gt;</code>\n\n"
        "<b>One-time:</b>\n"
        "• <code>/remind in 30 minutes check the oven</code>\n"
        "• <code>/remind in 2 hours call mom</code>\n"
        "• <code>/remind at 15:30 review the report</code>\n"
        "• <code>/remind tomorrow at 9:00 standup</code>\n\n"
        f"<b>Recurring</b> (not more often than every "
        f"{config.reminder_min_interval_minutes} minutes):\n"
        "• <code>/remind every 2 hours stretch</code>\n"
        "• <code>/remind daily at 10:00 drink water</code>\n"
        "• <code>/remind weekly mon at 10:00 weekly report</code>\n\n"
        "<b>🤖 AI requests:</b> prefix the text with <code>ai:</code> — at fire "
        "time the prompt is sent to your model and the <b>answer</b> is "
        "delivered:\n"
        "• <code>/remind daily at 9:00 ai: five main AI news of the day</code>\n\n"
        f"Limits: {config.reminder_limit} one-time, "
        f"{config.reminder_recurring_limit} recurring. "
        f"Times are interpreted in <b>{html.escape(tz)}</b>.\n"
        "Manage: /reminders · /delreminder N · /delreminder all"
    )


def parse_reminder(
    argument: str,
    now: datetime.datetime | None = None,
    tz: datetime.tzinfo | None = None,
) -> ParsedReminder | str:
    """Parse the text after ``/remind``.

    ``tz`` overrides the display timezone (the user's personal zone chosen
    with ``/timezone``); by default the global ``REMINDER_TIMEZONE`` applies.
    Returns a :class:`ParsedReminder`, or a user-facing HTML error string.
    """
    now = now or datetime.datetime.now(datetime.UTC)
    tz = tz or get_timezone()
    spec = argument.strip()
    if not spec:
        return remind_usage()

    fire_at: datetime.datetime
    schedule: str | None = None

    if (match := _RE_EVERY_N.match(spec)):
        minutes = int(match.group(1)) * _UNITS_MINUTES[match.group(2).lower()]
        if minutes > _MAX_INTERVAL_MINUTES:
            return _interval_too_long_error()
        if minutes < config.reminder_min_interval_minutes:
            return (
                "⚠️ <b>Too often.</b> Recurring reminders cannot fire more "
                f"often than every {config.reminder_min_interval_minutes} "
                "minutes."
            )
        schedule = f"every:{minutes}"
        fire_at = now + datetime.timedelta(minutes=minutes)
        spec = spec[match.end():]
    elif (match := _RE_EVERY_HOUR.match(spec)):
        schedule = "every:60"
        fire_at = now + datetime.timedelta(minutes=60)
        spec = spec[match.end():]
    elif (match := _RE_DAILY.match(spec)):
        hhmm = _validate_hhmm(match.group(1), match.group(2))
        if hhmm is None:
            return "⚠️ Invalid time — use a 24-hour <code>HH:MM</code> value."
        schedule = f"daily:{hhmm[0]:02d}:{hhmm[1]:02d}"
        fire_at = _next_at(hhmm[0], hhmm[1], False, now, tz)
        spec = spec[match.end():]
    elif (match := _RE_WEEKLY.match(spec)):
        hhmm = _validate_hhmm(match.group(2), match.group(3))
        if hhmm is None:
            return "⚠️ Invalid time — use a 24-hour <code>HH:MM</code> value."
        dow = _WEEKDAYS[match.group(1).lower()[:3]]
        schedule = f"weekly:{dow}:{hhmm[0]:02d}:{hhmm[1]:02d}"
        fire_at = _next_weekly(dow, hhmm[0], hhmm[1], now, tz)
        spec = spec[match.end():]
    elif (match := _RE_AT.match(spec)):
        hhmm = _validate_hhmm(match.group(2), match.group(3))
        if hhmm is None:
            return "⚠️ Invalid time — use a 24-hour <code>HH:MM</code> value."
        fire_at = _next_at(hhmm[0], hhmm[1], bool(match.group(1)), now, tz)
        spec = spec[match.end():]
    elif (match := _RE_IN.match(spec)):
        minutes = int(match.group(1)) * _UNITS_MINUTES[match.group(2).lower()]
        if minutes > _MAX_INTERVAL_MINUTES:
            return _interval_too_long_error()
        fire_at = now + datetime.timedelta(minutes=minutes)
        spec = spec[match.end():]
    else:
        return (
            "⚠️ <b>Could not find a time</b> in your command.\n\n"
            "Start with <code>in 30 minutes</code>, <code>at 15:30</code>, "
            "<code>tomorrow at 9:00</code>, <code>every 2 hours</code>, "
            "<code>daily at 10:00</code> or <code>weekly mon at 10:00</code> "
            "— full help: <code>/remind</code>."
        )

    ai_mode = False
    text = spec.strip()
    if text[:3].lower() == "ai:":
        ai_mode = True
        text = text[3:].strip()
    if ai_mode and not config.reminder_ai_enabled:
        return (
            "⚠️ <b>AI requests are disabled</b> for reminders on this bot "
            "(REMINDER_AI_ENABLED=false). Create a plain text reminder instead."
        )
    if not text:
        return (
            "⚠️ <b>Add the reminder text</b> after the time — or an "
            "<code>ai:</code> prompt. Examples: <code>/remind</code>."
        )

    return ParsedReminder(text=text, fire_at=fire_at, schedule=schedule, ai_mode=ai_mode)


# --- Next occurrence (scheduler advancement) ----------------------------------


def next_occurrence(
    schedule: str,
    previous_fire: datetime.datetime,
    now: datetime.datetime,
    tz: datetime.tzinfo | None = None,
) -> datetime.datetime | None:
    """When a recurring reminder should fire next; None on a broken schedule.

    Interval schedules advance from ``previous_fire`` (no drift); fires missed
    while the bot was down are skipped, not queued. Wall-clock schedules
    (daily/weekly) simply take the next matching local time after ``now`` —
    computed in ``tz`` (the reminder owner's personal zone when they have one,
    ``REMINDER_TIMEZONE`` otherwise; wall-clock schedules fire at the same
    local hour for their owner).
    """
    tz = tz or get_timezone()
    if previous_fire.tzinfo is None:
        # SQLite hands back naive datetimes; stored times are always UTC.
        previous_fire = previous_fire.replace(tzinfo=datetime.UTC)
    parts = schedule.split(":")
    try:
        if parts[0] == "every" and len(parts) == 2:
            minutes = int(parts[1])
            if minutes < 1:
                return None
            candidate = previous_fire + datetime.timedelta(minutes=minutes)
            while candidate <= now:
                candidate += datetime.timedelta(minutes=minutes)
            return candidate
        if parts[0] == "daily" and len(parts) == 3:
            return _next_at(int(parts[1]), int(parts[2]), False, now, tz)
        if parts[0] == "weekly" and len(parts) == 4:
            return _next_weekly(
                int(parts[1]), int(parts[2]), int(parts[3]), now, tz
            )
    except (ValueError, IndexError):
        pass
    logger.error("Broken reminder schedule %r", schedule)
    return None


def describe_schedule(schedule: str | None) -> str:
    """Human-readable recurrence for the /reminders list."""
    if not schedule:
        return "one-time"
    parts = schedule.split(":")
    try:
        if parts[0] == "every":
            minutes = int(parts[1])
            if minutes % 1440 == 0:
                return f"every {minutes // 1440} day(s)"
            if minutes % 60 == 0:
                return f"every {minutes // 60} h"
            return f"every {minutes} min"
        if parts[0] == "daily":
            return f"daily at {parts[1]}:{parts[2]}"
        if parts[0] == "weekly":
            names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            return f"weekly on {names[int(parts[1]) % 7]} at {parts[2]}:{parts[3]}"
    except (ValueError, IndexError):
        pass
    return schedule


# --- Scheduler ----------------------------------------------------------------


@dataclass(slots=True)
class DueReminder:
    """A claimed reminder, detached from the database session."""

    id: int
    user_id: int
    chat_id: int
    text: str
    ai_mode: bool
    schedule: str | None


async def reminder_scheduler_loop() -> None:
    """Background task: fire due reminders every ``REMINDER_CHECK_INTERVAL`` s."""
    logger.info(
        "Reminder scheduler started (tick %.0fs, timezone %s)",
        config.reminder_check_interval,
        config.reminder_timezone or "UTC",
    )
    while True:
        try:
            await process_due()
        except asyncio.CancelledError:
            raise
        except Exception:  # the loop must survive any single tick failure
            logger.exception("Reminder scheduler tick failed")
        await asyncio.sleep(config.reminder_check_interval)


async def process_due() -> list[DueReminder]:
    """Claim every due reminder and fire them as independent tasks.

    Claiming (advance recurring / delete one-time) happens in one committed
    transaction before any delivery, so a crash or a slow AI call cannot make
    the next tick fire the same reminder twice. Returns the claimed list.
    """
    now = datetime.datetime.now(datetime.UTC)
    claimed: list[DueReminder] = []
    async with async_session_factory() as session:
        repo = ReminderRepository(session)
        user_repo = UserRepository(session)
        due_all = await repo.get_due(now)
        # One query per tick (not per reminder) for owners' personal zones.
        personal_tz = await user_repo.get_reminder_timezones(
            [r.user_id for r in due_all]
        )
        tz_cache: dict[int, datetime.tzinfo] = {}
        for reminder in due_all:
            tz = tz_cache.get(reminder.user_id)
            if tz is None:
                tz = get_timezone(personal_tz.get(reminder.user_id))
                tz_cache[reminder.user_id] = tz
            if reminder.schedule:
                next_fire = next_occurrence(
                    reminder.schedule, reminder.fire_at, now, tz=tz
                )
                if next_fire is None:
                    # A broken schedule would fire forever — stop it for good.
                    await repo.delete_by_id(reminder.id)
                    continue
                await repo.reschedule(reminder, next_fire)
            else:
                await repo.delete_by_id(reminder.id)
            claimed.append(
                DueReminder(
                    id=reminder.id,
                    user_id=reminder.user_id,
                    chat_id=reminder.chat_id,
                    text=reminder.text,
                    ai_mode=reminder.ai_mode,
                    schedule=reminder.schedule,
                )
            )
        await session.commit()

    for due in claimed:
        task = asyncio.create_task(fire_reminder(due), name=f"reminder-{due.id}")
        # Keep a strong reference: the event loop only holds weak references to
        # tasks, so an unreferenced one can be garbage-collected mid-delivery.
        _fire_tasks.add(task)
        task.add_done_callback(_fire_tasks.discard)
    return claimed


async def cancel_pending_fires() -> None:
    """Cancel in-flight deliveries (shutdown); reminders already claimed are lost."""
    tasks = list(_fire_tasks)
    _fire_tasks.clear()
    for task in tasks:
        task.cancel()
    for task in tasks:
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception as exc:  # noqa: BLE001 - shutdown must not raise
            logger.debug("Reminder task failed during shutdown: %s", exc)


async def fire_reminder(due: DueReminder) -> None:
    """Deliver one claimed reminder. Never raises."""
    try:
        if due.ai_mode:
            await _fire_ai(due)
        else:
            await _send_html(
                due.chat_id,
                f"⏰ <b>Reminder</b>\n\n{html.escape(due.text)}",
            )
    except TelegramForbiddenError:
        # The bot was blocked or kicked: stop this reminder for good.
        logger.info(
            "Chat %s is unreachable; disabling reminder %s", due.chat_id, due.id
        )
        await _disable_reminder(due.id)
    except Exception:  # one bad fire must not kill the others
        logger.exception("Reminder %s failed", due.id)


async def _disable_reminder(reminder_id: int) -> None:
    try:
        async with async_session_factory() as session:
            await ReminderRepository(session).disable(reminder_id)
            await session.commit()
    except Exception as exc:  # noqa: BLE001 - cleanup is best-effort
        logger.warning("Could not disable reminder %s: %s", reminder_id, exc)


async def _fire_ai(due: DueReminder) -> None:
    """Run the stored prompt through the user's model and deliver the answer."""
    async with async_session_factory() as session:
        service = UserService(session)
        settings = await service.get_user_settings(due.user_id)
        # Access is re-checked at fire time, not only when the reminder is
        # created: a user who was removed from the allow-list (or an admin
        # dropped from ADMIN_IDS) must not keep spending money on scheduled AI
        # calls forever. ``require_quota=False`` matters here — a reminder is a
        # scheduled request, not a new question, so an exhausted *daily*
        # question budget must NOT disable it (only losing authorization may).
        gate = await check_access(
            service, due.user_id, consume=False, require_quota=False
        )

    if not gate.allowed or gate.role == "guest":
        # Guests may not hold scheduled AI reminders at all (see /remind).
        logger.info(
            "Reminder %s: user %s is no longer authorized (role=%s); disabling it",
            due.id,
            due.user_id,
            gate.role,
        )
        await _disable_reminder(due.id)
        return

    result = await ai_client.chat_completion_result(
        messages=[
            {"role": "system", "content": settings["system_prompt"]},
            {"role": "user", "content": due.text},
        ],
        model=settings["model"],
        temperature=settings["temperature"],
        max_tokens=settings["max_tokens"],
    )
    if not result.content:
        await _send_html(
            due.chat_id,
            "❌ <b>Scheduled AI request failed</b> — the model returned no "
            "answer.\n"
            f"<i>Prompt: {html.escape(due.text[:300])}</i>",
        )
        return

    footer = format_timing(
        result.elapsed,
        model=settings["model"],
        tokens=result.total_tokens,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
    )
    markdown = f"🔔 **Scheduled AI request**\n_{due.text[:300]}_\n\n{result.content}"
    if footer:
        markdown = f"{markdown}\n\n{footer}"
    await _send_markdown(due.chat_id, markdown)


# --- Delivery helpers ----------------------------------------------------------


_MAX_429_RETRIES = 3


async def _send_once(chat_id: int, text: str, *, parse_mode: str | None) -> None:
    """Send one message, honouring Telegram's 429 ``retry_after``.

    ``throttle`` serialises per chat only, while a scheduler tick can fire many
    reminders across many different chats at once — exactly when Telegram's
    global limit kicks in. A reminder is claimed (deleted or rescheduled)
    *before* delivery, so an unretried 429 would lose it permanently.
    """
    for attempt in range(_MAX_429_RETRIES + 1):
        try:
            await throttle.wait(chat_id)
            await bot.send_message(chat_id, text, parse_mode=parse_mode)
            return
        except TelegramRetryAfter as exc:
            if attempt >= _MAX_429_RETRIES:
                logger.error("Rate limit persists for chat %s", chat_id)
                raise
            delay = float(getattr(exc, "retry_after", 1) or 1) + 0.1
            logger.warning(
                "Telegram rate limit (429) for chat %s; retrying in %.1fs",
                chat_id,
                delay,
            )
            await asyncio.sleep(delay)


async def _send_html(chat_id: int, text: str) -> None:
    """Send one HTML message (raises TelegramForbiddenError when blocked)."""
    await _send_once(chat_id, text, parse_mode="HTML")


async def _send_markdown(chat_id: int, markdown_text: str) -> bool:
    """Deliver a Markdown answer chunk by chunk: HTML first, plain on reject."""
    delivered = True
    for chunk in split_chunks(markdown_text, "html", 4000):
        if not chunk.strip():
            continue
        try:
            await _send_once(chat_id, render(chunk, "html"), parse_mode="HTML")
        except TelegramBadRequest as exc:
            logger.warning("Scheduled answer rejected as HTML (%s); plain retry", exc)
            try:
                await _send_once(chat_id, chunk, parse_mode=None)
            except TelegramForbiddenError:
                raise
            except TelegramAPIError as retry_exc:
                logger.error("Plain retry failed: %s", retry_exc)
                delivered = False
        except TelegramForbiddenError:
            raise
        except TelegramAPIError as exc:
            logger.error("Scheduled message delivery failed: %s", exc)
            delivered = False
    return delivered
