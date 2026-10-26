"""Reminder commands: /remind, /reminders (inline controls), /delreminder.

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
import time
from dataclasses import dataclass

from aiogram import F, Router
from aiogram.filters import Command
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)
from aiogram.utils.keyboard import InlineKeyboardBuilder

from bot.config import config
from bot.database.models import Reminder
from bot.services.access import check_access
from bot.services.reminder_intent import extract_freeform, looks_like_reminder
from bot.services.reminders import (
    ParsedReminder,
    describe_schedule,
    get_timezone,
    parse_reminder,
    remind_usage,
)
from bot.services.telegram_utils import (
    safe_answer,
    safe_answer_callback,
    safe_edit_text,
)
from bot.services.timezone_intent import detect_timezone, is_valid_zone_name
from bot.services.user_service import UserService, user_service

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


def _format_fire_time(fire_at: datetime.datetime, tz: datetime.tzinfo) -> str:
    """Fire time in the given (European) display timezone."""
    if fire_at.tzinfo is None:
        # SQLite hands back naive datetimes; stored times are always UTC.
        fire_at = fire_at.replace(tzinfo=datetime.UTC)
    return fire_at.astimezone(tz).strftime("%d.%m.%Y %H:%M %Z")


def _confirmation(
    reminder_id: int, parsed: ParsedReminder, tz: datetime.tzinfo
) -> str:
    """The 'reminder created' reply."""
    icon = "🔁" if parsed.schedule else "⏰"
    lines = [
        f"✅ <b>Reminder #{reminder_id} created</b> {icon}",
        "",
        f"<i>{html.escape(parsed.text[:300])}</i>",
        f"🕒 First fire: {_format_fire_time(parsed.fire_at, tz)}",
    ]
    if parsed.schedule:
        lines.append(f"🔁 Schedule: {html.escape(describe_schedule(parsed.schedule))}")
    if parsed.ai_mode:
        lines.append(
            "🤖 <b>AI request</b> — at fire time this prompt is sent to your "
            "model and the answer is delivered here."
        )
    lines.append("")
    lines.append("Manage: /reminders (pause / resume / delete buttons) · /delreminder " + str(reminder_id))
    return "\n".join(lines)


async def create_reminder_record(
    service: UserService,
    user_id: int,
    chat_id: int,
    parsed: ParsedReminder,
) -> tuple[int | None, str | None]:
    """Insert a reminder honouring the per-user cap (capture race-safe).

    Returns ``(reminder_id, None)`` on success or ``(None, error)`` when the
    one-time / recurring limit of the user is exhausted. Shared by
    ``/remind`` and free-form reminder requests.
    """
    recurring = parsed.schedule is not None
    limit = config.reminder_recurring_limit if recurring else config.reminder_limit
    async with _create_lock(user_id):
        count = await service.count_reminders(user_id, recurring=recurring)
        if count >= limit:
            kind = "Recurring reminders" if recurring else "Reminders"
            return None, (
                f"⚠️ <b>Limit reached:</b> {kind} {count}/{limit}.\n\n"
                "Delete some first — see /reminders and /delreminder."
            )
        created = await service.add_reminder(
            Reminder(
                user_id=user_id,
                chat_id=chat_id,
                text=parsed.text,
                ai_mode=parsed.ai_mode,
                fire_at=parsed.fire_at,
                schedule=parsed.schedule,
            )
        )
        return created.id, None


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
        # The help shows *that* user's personal zone when they have one.
        async with user_service() as service:
            await service.register_user(
                telegram_id=user.id,
                username=user.username,
                first_name=user.first_name,
                last_name=user.last_name,
            )
            tz = get_timezone(await service.get_reminder_timezone(user.id))
            await safe_answer(message, remind_usage(user_tz_name=str(tz)))
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

        # The wall-clock slots (at / tomorrow at / daily / weekly) are
        # interpreted in the user's personal zone (/timezone) when they set
        # one, otherwise in REMINDER_TIMEZONE.
        tz = get_timezone(await service.get_reminder_timezone(user.id))
        parsed = parse_reminder(parts[1], tz=tz)
        if isinstance(parsed, str):  # an error/help message for the user
            await safe_answer(message, parsed)
            return

        reminder_id, error = await create_reminder_record(
            service, user.id, message.chat.id, parsed
        )
        if error is not None:
            await safe_answer(message, error)
            return

    await safe_answer(message, _confirmation(reminder_id, parsed, tz))


async def maybe_freeform_reminder(message: Message, text: str) -> bool:
    """Catch free-form reminder requests inside ordinary text («напомни…»).

    Called from the chat flow before the message becomes a chat question.
    Returns ``True`` when the message was fully handled as a reminder request
    (a reminder was created, or a friendly error was sent) and the caller
    must not answer it again. Creation is free (see ``/remind``), does not
    consume the daily quota, and is offered only to non-guest users.
    """
    user = message.from_user
    if user is None:
        return False
    if not config.enable_reminders or not config.reminder_freeform:
        return False
    # Layer 1: the cheap pre-filter. Messages without any reminder-like word
    # continue as chat without paying for detection.
    if not looks_like_reminder(text):
        return False

    async with user_service() as service:
        await service.register_user(
            telegram_id=user.id,
            username=user.username,
            first_name=user.first_name,
            last_name=user.last_name,
        )
        # Reminder creation is free (nothing is consumed), but it is an
        # authorized-only feature; guests get the normal chat answer instead.
        access = await check_access(
            service, user.id, consume=False, require_quota=False
        )
        if not access.allowed or access.role == "guest":
            return False
        settings = await service.get_user_settings(user.id)
        tz = get_timezone(await service.get_reminder_timezone(user.id))

    # Layer 2: one small AI call converts the request into a /remind slot;
    # layer 3 inside the service re-validates it with the plain regexes.
    outcome = await extract_freeform(text, settings["model"], tz=tz)
    if outcome is None:
        # Not a reminder request (or detection failed): answer as chat.
        return False

    if isinstance(outcome, str):
        await safe_answer(message, outcome)
        return True

    async with user_service() as service:
        reminder_id, error = await create_reminder_record(
            service, user.id, message.chat.id, outcome
        )
    if error is not None:
        await safe_answer(message, error)
        return True
    await safe_answer(message, _confirmation(reminder_id, outcome, tz))
    return True


PER_PAGE = 6


async def _render_list(
    user_id: int, page: int = 0
) -> tuple[str, InlineKeyboardMarkup | None]:
    """The /reminders text and its control keyboard (None markup when empty).

    Every callback carries the owner id so that, in a group, only the person
    who opened the list can press its buttons.
    """
    async with user_service() as service:
        reminders = await service.list_reminders(user_id)
        owner_tz = get_timezone(await service.get_reminder_timezone(user_id))
        # Snapshot the rows into plain data inside the session.
        rows = [
            {
                "id": r.id,
                "text": r.text,
                "fire_at": r.fire_at,
                "schedule": r.schedule,
                "ai_mode": r.ai_mode,
                "enabled": r.enabled,
            }
            for r in reminders
        ]

    if not rows:
        return (
            ("📭 <b>No active reminders.</b>\n\n"
            "Create one with /remind — the full format list is in /help."),
            None,
        )

    pages = (len(rows) + PER_PAGE - 1) // PER_PAGE
    page = max(0, min(page, pages - 1))
    shown = rows[page * PER_PAGE : (page + 1) * PER_PAGE]

    one_time = sum(1 for row in rows if row["schedule"] is None)
    recurring = len(rows) - one_time
    header = (
        f"⏰ <b>Your reminders</b> — one-time: {one_time}/{config.reminder_limit}, "
        f"recurring: {recurring}/{config.reminder_recurring_limit}"
    )
    if pages > 1:
        header += f" (page {page + 1}/{pages})"
    blocks = [header]
    builder = InlineKeyboardBuilder()
    for row in shown:
        icon = "🔁" if row["schedule"] else "⏰"
        ai = " · 🤖 AI" if row["ai_mode"] else ""
        paused = "" if row["enabled"] else " · ⏸ <b>paused</b>"
        schedule = f" · {describe_schedule(row['schedule'])}" if row["schedule"] else ""
        next_line = (
            f"Next: {_format_fire_time(row['fire_at'], owner_tz)}"
            if row["enabled"]
            else "Paused"
        )
        blocks.append(
            f"<b>#{row['id']}</b> {icon}{ai}{schedule}{paused}\n"
            f"{next_line}\n"
            f"<i>{html.escape(row['text'][:150])}</i>"
        )
        toggle = (
            f"⏸ Pause #{row['id']}" if row["enabled"] else f"▶️ Resume #{row['id']}"
        )
        builder.row(
            InlineKeyboardButton(
                text=toggle, callback_data=f"rem:tg:{user_id}:{row['id']}:{page}"
            ),
            InlineKeyboardButton(
                text=f"🗑 #{row['id']}",
                callback_data=f"rem:del:{user_id}:{row['id']}:{page}",
            ),
        )
    if pages > 1:
        nav = []
        if page > 0:
            nav.append(
                InlineKeyboardButton(text="◀", callback_data=f"rem:pg:{user_id}:{page - 1}")
            )
        nav.append(InlineKeyboardButton(text=f"{page + 1}/{pages}", callback_data="rem:noop"))
        if page < pages - 1:
            nav.append(
                InlineKeyboardButton(text="▶", callback_data=f"rem:pg:{user_id}:{page + 1}")
            )
        builder.row(*nav)
    builder.row(
        InlineKeyboardButton(text="🗑 Delete all", callback_data=f"rem:all:{user_id}")
    )
    return "\n\n".join(blocks), builder.as_markup()


@router.message(Command("reminders"))
async def cmd_reminders(message: Message) -> None:
    """List the user's reminders with Pause / Resume / Delete buttons."""
    user = message.from_user
    if user is None:
        return
    if not config.enable_reminders:
        await safe_answer(message, "⚠️ Reminders are disabled on this bot.")
        return

    text, markup = await _render_list(user.id)
    await safe_answer(message, text, reply_markup=markup)


def _confirm_all_markup(user_id: int) -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    builder.button(text="✅ Yes, delete all", callback_data=f"rem:allok:{user_id}")
    builder.button(text="✖ Cancel", callback_data=f"rem:pg:{user_id}:0")
    builder.adjust(2)
    return builder.as_markup()


@router.callback_query(F.data == "rem:noop")
async def cb_reminders_noop(callback: CallbackQuery) -> None:
    await safe_answer_callback(callback)


@router.callback_query(F.data.startswith("rem:"))
async def cb_reminders(callback: CallbackQuery) -> None:
    """Pause/resume/delete/paginate; only the list's owner may press."""
    parts = (callback.data or "").split(":")
    try:
        action, owner = parts[1], int(parts[2])
    except (IndexError, ValueError):
        await safe_answer_callback(callback)
        return
    if callback.from_user is None or callback.from_user.id != owner:
        await safe_answer_callback(callback, "This is not your list.", show_alert=True)
        return
    if not config.enable_reminders:
        await safe_answer_callback(callback, "Reminders are disabled.", show_alert=True)
        return

    page = 0
    toast: str | None = None
    try:
        if action == "pg":
            page = int(parts[3])
        elif action in ("tg", "del"):
            reminder_id, page = int(parts[3]), int(parts[4])
            async with user_service() as service:
                if action == "tg":
                    reminder = await service.toggle_reminder(reminder_id, owner)
                    if reminder is None:
                        toast = "Reminder not found."
                    else:
                        toast = "Resumed" if reminder.enabled else "Paused"
                else:
                    deleted = await service.delete_reminder(reminder_id, owner)
                    toast = "Deleted" if deleted else "Reminder not found."
        elif action == "all":
            await safe_edit_text(
                callback.message,
                "🗑 <b>Delete all your reminders?</b> This cannot be undone.",
                reply_markup=_confirm_all_markup(owner),
            )
            await safe_answer_callback(callback)
            return
        elif action == "allok":
            async with user_service() as service:
                count = await service.delete_all_reminders(owner)
            toast = f"Deleted {count}"
        else:
            await safe_answer_callback(callback)
            return
    except (IndexError, ValueError):
        await safe_answer_callback(callback)
        return

    text, markup = await _render_list(owner, page)
    await safe_edit_text(callback.message, text, reply_markup=markup)
    await safe_answer_callback(callback, toast)


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


# --- Personal reminders timezone (/timezone) -----------------------------------

@dataclass(slots=True)
class _TzDialog:
    """One in-flight /timezone dialog.

    ``candidate`` is the currently proposed IANA zone (None = waiting for the
    user's city/zone description). The dialog lives only in the chat where
    /timezone was sent and expires after ``_TZ_DIALOG_TTL`` seconds.
    """

    chat_id: int
    candidate: str | None
    expires_at: float


_TZ_DIALOG_TTL = 600.0
_tz_dialogs: dict[int, _TzDialog] = {}
_TZ_DIALOGS_CAP = 200


def _start_dialog(user_id: int, chat_id: int, candidate: str | None = None) -> None:
    _tz_dialogs[user_id] = _TzDialog(
        chat_id=chat_id,
        candidate=candidate,
        expires_at=time.monotonic() + _TZ_DIALOG_TTL,
    )


def _get_dialog(user_id: int, chat_id: int | None = None) -> _TzDialog | None:
    """The user's live dialog; expired ones are dropped, and a dialog started
    in another chat is invisible (``chat_id`` given)."""
    dialog = _tz_dialogs.get(user_id)
    if dialog is None:
        return None
    if dialog.expires_at <= time.monotonic():
        _tz_dialogs.pop(user_id, None)
        return None
    if chat_id is not None and dialog.chat_id != chat_id:
        return None
    return dialog


_TZ_AUTH_MESSAGE = (
    "⚠️ <b>Time zones are for authorized users only.</b>\n\n"
    "Reminder times are interpreted per user in their own zone; guests "
    "always use the server timezone."
)
_TZ_DIALOG_PROMPT = (
    "🌍 <b>What time zone are you in?</b>\n\n"
    "Write a city or zone — for example <i>I'm in Moscow</i>, "
    "<i>novosibirsk</i> or <i>Europe/Moscow</i>. I will guess the exact IANA "
    "zone, show you the current time there and ask you to confirm.\n\n"
    "Say <code>cancel</code> to stop without saving."
)
_TZ_COULD_NOT_DETECT = (
    "🤔 <b>Could not tell where that is.</b>\n\n"
    "Name a city or a known zone (e.g. <code>Europe/Moscow</code>), or say "
    "<code>cancel</code> to stop."
)
_TZ_INVALID_TEMPLATE = (
    '⚠️ <b>"{name}" is not a valid IANA time zone.</b>\n\n'
    "Use the <code>Area/City</code> form — for example "
    "<code>/timezone Europe/Moscow</code> — or send <code>/timezone</code> "
    "and I will guess the zone in a short dialog."
)
_TZ_SAVED_TEMPLATE = (
    "✅ <b>Time zone saved: {zone}.</b>\n\n"
    "It is now {now:%H:%M} there. New <code>/remind</code> commands "
    "(<i>at 15:30</i>, <i>tomorrow</i>, <i>daily</i>, <i>weekly</i>) are "
    "interpreted in this zone and <code>/reminders</code> shows fire times "
    "in it. One-time reminders created earlier keep their stored time.\n"
    "Change it any time with <code>/timezone</code>."
)
_TZ_CANDIDATE_TEMPLATE = (
    "🌍 Your time zone looks like <b>{zone}</b>.\n"
    "It is now {now:%H:%M}, {now:%A} there.\n\n"
    "Is that the same as the time on your clock?"
)
_TZ_WRONG = (
    " okay — name another city or zone (or say <code>cancel</code>)."
)


@router.message(Command("timezone"))
async def cmd_timezone(message: Message) -> None:
    """Set the personal reminders timezone: strict form or a guided dialog.

    ``/timezone Europe/Moscow`` saves the exact IANA name; a bare
    ``/timezone`` starts the dialog: the bot asks where the user lives, the
    user's model guesses the IANA zone, and the choice is confirmed by
    showing the current time in that zone before anything is saved.
    """
    user = message.from_user
    if user is None:
        return
    if not config.enable_reminders:
        await safe_answer(message, "⚠️ Reminders are disabled on this bot.")
        return

    parts = (message.text or "").split(maxsplit=1)
    argument = parts[1].strip() if len(parts) > 1 else ""

    async with user_service() as service:
        await service.register_user(
            telegram_id=user.id,
            username=user.username,
            first_name=user.first_name,
            last_name=user.last_name,
        )
        # Free, but authorized-only — like /remind: the zone is only used to
        # interpret reminder times.
        access = await check_access(
            service, user.id, consume=False, require_quota=False
        )
        if not access.allowed or access.role == "guest":
            await safe_answer(message, access.message or _TZ_AUTH_MESSAGE)
            return

        if not argument:
            _start_dialog(user.id, message.chat.id)
            await safe_answer(message, _TZ_DIALOG_PROMPT)
            return

        if not is_valid_zone_name(argument):
            await safe_answer(
                message,
                _TZ_INVALID_TEMPLATE.format(name=html.escape(argument)),
            )
            return
        await service.set_reminder_timezone(user.id, argument)

    tz = get_timezone(argument)
    local_now = datetime.datetime.now(datetime.UTC).astimezone(tz)
    await safe_answer(
        message,
        _TZ_SAVED_TEMPLATE.format(zone=html.escape(argument), now=local_now),
    )


async def maybe_timezone_dialog(message: Message, text: str) -> bool:
    """Feed the user's next message into an active ``/timezone`` dialog.

    Called from the chat flow before anything else. Returns ``True`` when
    the message was consumed by the dialog (or it was cancelled) and the
    caller must not answer again.
    """
    user = message.from_user
    if user is None or _get_dialog(user.id, message.chat.id) is None:
        return False

    lowered = text.strip().lower().rstrip("!.")
    if lowered in ("cancel", "отмена", "не надо", "стоп"):
        _tz_dialogs.pop(user.id, None)
        await safe_answer(message, "Cancelled — the time zone was not changed.")
        return True

    # Evict expired dialogs, then the oldest ones over the cap.
    if len(_tz_dialogs) > _TZ_DIALOGS_CAP:
        for key in list(_tz_dialogs):
            _get_dialog(key)
        for key in list(_tz_dialogs)[: max(0, len(_tz_dialogs) - _TZ_DIALOGS_CAP)]:
            _tz_dialogs.pop(key, None)

    async with user_service() as service:
        settings = await service.get_user_settings(user.id)

    detected = await detect_timezone(text, settings["model"])
    if detected is None:
        await safe_answer(message, _TZ_COULD_NOT_DETECT)
        return True  # keep waiting for a usable answer

    _start_dialog(user.id, message.chat.id, detected)
    local_now = datetime.datetime.now(datetime.UTC).astimezone(get_timezone(detected))
    builder = InlineKeyboardBuilder()
    builder.row(
        InlineKeyboardButton(text="✅ Yes, it matches", callback_data=f"tz:ok:{detected}"),
        InlineKeyboardButton(text="❌ No, wrong", callback_data=f"tz:no:{detected}"),
    )
    await safe_answer(
        message,
        _TZ_CANDIDATE_TEMPLATE.format(
            zone=html.escape(detected), now=local_now
        ),
        reply_markup=builder.as_markup(),
    )
    return True


@router.callback_query(F.data.startswith("tz:"))
async def cb_timezone(callback: CallbackQuery) -> None:
    """Confirm / reject the detected zone; only the dialog owner may answer."""
    user = callback.from_user
    action = (callback.data or "").split(":")[1] if callback.data else ""
    if user is None:
        await safe_answer_callback(callback)
        return
    chat_id = callback.message.chat.id if callback.message is not None else None
    dialog = _get_dialog(user.id, chat_id)
    candidate = dialog.candidate if dialog is not None else None
    # The button carries the zone it was shown for: a stale button of an
    # earlier guess must not confirm the newest candidate.
    shown = (callback.data or "").split(":", 2)[2:]
    if candidate is None or (shown and shown[0] != candidate):
        await safe_answer_callback(
            callback, "The dialog is over — send /timezone to start again.",
            show_alert=True,
        )
        return

    if action == "ok":
        async with user_service() as service:
            await service.set_reminder_timezone(user.id, candidate)
        _tz_dialogs.pop(user.id, None)
        local_now = datetime.datetime.now(datetime.UTC).astimezone(
            get_timezone(candidate)
        )
        if callback.message is not None:
            await safe_edit_text(
                callback.message,
                _TZ_SAVED_TEMPLATE.format(zone=html.escape(candidate), now=local_now),
            )
        await safe_answer_callback(callback)
        return

    if action == "no":
        # Back to "tell me where you live": a wrong guess means the model
        # needs a clearer description.
        _start_dialog(user.id, dialog.chat_id)
        if callback.message is not None:
            await safe_edit_text(
                callback.message,
                "❌ Not that one." + _TZ_WRONG,
            )
        await safe_answer_callback(callback)
        return

    await safe_answer_callback(callback)
