"""Inline-keyboard admin menu: access levels, per-user limits and settings.

Opened with ``/admin``. Every button is a callback ``adm:<action>[:<id>[:<arg>]]``
and is re-checked against the *live* admin list, so a forged callback or a
demoted admin's old menu cannot change anything.

Levels: guest < user (whitelisted) < admin < owner (``ADMIN_IDS``). Admins manage
guests and users; only owners promote/demote admins. Owners and other admins
cannot be limited or blocked by an admin.
"""

from __future__ import annotations

import datetime
import html
import logging
from typing import Any

from aiogram import F, Router
from aiogram.filters import Command
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)
from aiogram.utils.keyboard import InlineKeyboardBuilder

from bot.bot import bot, get_bot_id
from bot.config import config
from bot.handlers.admin import _fetch_profile, _name_html, _resolve_target
from bot.services import digest
from bot.services.telegram_utils import safe_answer_callback, safe_edit_text
from bot.services.user_service import user_service

logger = logging.getLogger(__name__)

router = Router()

PER_PAGE = 8
MAX_LIMIT = 100_000

_LEVEL_ICON = {"owner": "👑", "admin": "🛡", "user": "✅", "guest": "👤", "blocked": "🚫"}
_LEVEL_TITLE = {
    "owner": "Owner",
    "admin": "Admin",
    "user": "User",
    "guest": "Guest",
    "blocked": "Blocked",
}
_FILTERS = {  # key -> (button label, levels shown; None = all)
    "a": ("All", None),
    "u": ("Users", {"user"}),
    "g": ("Guests", {"guest"}),
    "b": ("Blocked", {"blocked"}),
    "m": ("Admins", {"admin", "owner"}),
}

# Per-admin menu state (in memory: a restart just closes the menu context).
_view: dict[int, tuple[str, int]] = {}  # admin -> (list filter, page)
_awaiting: dict[int, tuple[str, int | None]] = {}  # admin -> (kind, target id)


# --- Helpers ---------------------------------------------------------------


def _is_admin_cb(callback: CallbackQuery) -> bool:
    return callback.from_user is not None and config.is_admin(callback.from_user.id)


def _can_manage(actor_id: int, level: str) -> bool:
    """May ``actor_id`` change a user who currently has ``level``?"""
    if level == "owner":
        return False
    if level == "admin":
        return config.is_owner(actor_id)
    return True


def _label(row: dict[str, Any]) -> str:
    name = _name_html(row["first_name"], row["last_name"], row["username"], "")
    return name or "unknown"


def _plain_label(row: dict[str, Any]) -> str:
    name = " ".join(p for p in (row["first_name"], row["last_name"]) if p)
    if row["username"]:
        name = f"{name} @{row['username']}".strip()
    return (name or "unknown")[:24]


async def _show(
    callback: CallbackQuery, text: str, markup: InlineKeyboardMarkup
) -> None:
    await safe_edit_text(callback.message, text, reply_markup=markup)


async def _find(service: Any, telegram_id: int) -> dict[str, Any]:
    for row in await service.list_known_users():
        if row["telegram_id"] == telegram_id:
            return row
    return {
        "telegram_id": telegram_id,
        "level": await service.get_level(telegram_id),
        "username": None,
        "first_name": None,
        "last_name": None,
    }


# --- Screens ---------------------------------------------------------------


def _home_markup() -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    b.button(text="👥 Users", callback_data="adm:list:a:0")
    b.button(text="📊 Statistics", callback_data="adm:stats")
    b.button(text="➕ Add user", callback_data="adm:add")
    b.button(text="📰 Digest", callback_data="adm:dg")
    b.button(text="✖ Close", callback_data="adm:close")
    b.adjust(2, 2, 1)
    return b.as_markup()


async def _home_text() -> str:
    async with user_service() as service:
        rows = await service.list_known_users()
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["level"]] = counts.get(row["level"], 0) + 1
    return (
        "🛠 <b>Admin menu</b>\n\n"
        f"👑 Owners: <b>{counts.get('owner', 0)}</b>  "
        f"🛡 Admins: <b>{counts.get('admin', 0)}</b>\n"
        f"✅ Users: <b>{counts.get('user', 0)}</b>  "
        f"👤 Guests: <b>{counts.get('guest', 0)}</b>  "
        f"🚫 Blocked: <b>{counts.get('blocked', 0)}</b>\n\n"
        f"📊 Default daily limit: <b>{config.daily_limit}</b> "
        "(<code>DAILY_LIMIT</code>); each user can get a personal one.\n"
        "Pick <b>Users</b> to open a card and manage limits, access and model."
    )


async def _render_list(admin_id: int, flt: str, page: int) -> tuple[str, InlineKeyboardMarkup]:
    flt = flt if flt in _FILTERS else "a"
    levels = _FILTERS[flt][1]
    async with user_service() as service:
        rows = await service.list_known_users()
    order = {"owner": 0, "admin": 1, "user": 2, "blocked": 3, "guest": 4}
    rows = [r for r in rows if levels is None or r["level"] in levels]
    rows.sort(key=lambda r: order[r["level"]])

    pages = max(1, (len(rows) + PER_PAGE - 1) // PER_PAGE)
    page = max(0, min(page, pages - 1))
    _view[admin_id] = (flt, page)

    b = InlineKeyboardBuilder()
    for key, (title, _) in _FILTERS.items():
        mark = "• " if key == flt else ""
        b.button(text=f"{mark}{title}", callback_data=f"adm:list:{key}:0")
    b.adjust(len(_FILTERS))

    for row in rows[page * PER_PAGE : (page + 1) * PER_PAGE]:
        b.row(
            _button(
                f"{_LEVEL_ICON[row['level']]} {_plain_label(row)} · {row['telegram_id']}",
                f"adm:u:{row['telegram_id']}",
            )
        )

    nav = InlineKeyboardBuilder()
    if page > 0:
        nav.button(text="◀", callback_data=f"adm:list:{flt}:{page - 1}")
    nav.button(text=f"{page + 1}/{pages}", callback_data="adm:noop")
    if page < pages - 1:
        nav.button(text="▶", callback_data=f"adm:list:{flt}:{page + 1}")
    b.attach(nav)
    b.row(_button("🏠 Menu", "adm:home"), _button("➕ Add user", "adm:add"))

    text = f"👥 <b>{_FILTERS[flt][0]}</b> — {len(rows)} total"
    if not rows:
        text += "\n\n<i>Nobody here yet.</i>"
    return text, b.as_markup()


def _button(text: str, data: str) -> InlineKeyboardButton:
    return InlineKeyboardButton(text=text, callback_data=data)


async def _render_card(
    actor_id: int, telegram_id: int, notice: str = ""
) -> tuple[str, InlineKeyboardMarkup]:
    async with user_service() as service:
        row = await _find(service, telegram_id)
        level = row["level"]
        settings = await service.get_user_settings(telegram_id)
        limit = await service.get_daily_limit(telegram_id)
        used = await service.get_daily_usage_used(telegram_id)
        guest_left = await service.get_guest_messages_remaining(telegram_id)
        personal = (await service.user_repo.get_daily_limit_override(telegram_id)) is not None

    lines = [
        f"{_LEVEL_ICON[level]} <b>{_LEVEL_TITLE[level]}</b>  <code>{telegram_id}</code>",
        f"🏷 {_label(row)}",
        f"🤖 Model: <code>{html.escape(settings['model'])}</code>",
    ]
    if level in ("owner", "admin"):
        lines.append("📊 Requests: <b>unlimited</b>")
    elif level == "user":
        kind = "personal" if personal else "default"
        lines.append(f"📊 Today: <b>{used}/{limit}</b> ({kind} limit)")
    elif level == "guest":
        lines.append(
            f"🎁 Free trial left: <b>{guest_left}/{config.guest_message_limit}</b>"
        )
    else:
        lines.append("⛔ Access is blocked")
    if notice:
        lines.append(f"\n{notice}")

    b = InlineKeyboardBuilder()
    if _can_manage(actor_id, level):
        if level == "user":
            for delta in (-5, -1, 1, 5):
                b.button(
                    text=f"{delta:+d}", callback_data=f"adm:lim:{telegram_id}:{delta}"
                )
            b.adjust(4)
            b.row(
                _button("✏️ Set limit", f"adm:lim:{telegram_id}:i"),
                _button("♻️ Default", f"adm:lim:{telegram_id}:d"),
            )
        b.row(
            _button("🔄 Reset today", f"adm:rst:{telegram_id}"),
            _button("🤖 Model", f"adm:mdl:{telegram_id}"),
        )
        b.row(_button("🧹 Reset prompt", f"adm:rsp:{telegram_id}"))
        if level == "guest":
            b.row(_button("✅ Grant access", f"adm:grant:{telegram_id}"))
        elif level == "user":
            b.row(_button("⬇️ Revoke access", f"adm:revoke:{telegram_id}"))
        if level == "blocked":
            b.row(_button("🔓 Unblock", f"adm:blk:{telegram_id}"))
        elif level != "admin":
            b.row(_button("🚫 Block", f"adm:blk:{telegram_id}"))
        if config.is_owner(actor_id):
            if level == "admin":
                b.row(_button("⬇️ Remove admin", f"adm:adm:{telegram_id}"))
            else:
                b.row(_button("🛡 Make admin", f"adm:adm:{telegram_id}"))
    else:
        lines.append("\n<i>You cannot change this account.</i>")
    flt, page = _view.get(actor_id, ("a", 0))
    b.row(
        _button("◀ List", f"adm:list:{flt}:{page}"), _button("🏠 Menu", "adm:home")
    )
    return "\n".join(lines), b.as_markup()


async def _refresh_card(callback: CallbackQuery, telegram_id: int, notice: str = "") -> None:
    text, markup = await _render_card(callback.from_user.id, telegram_id, notice)
    await _show(callback, text, markup)


# --- Entry points ------------------------------------------------------------


@router.message(Command("admin"))
async def cmd_admin(message: Message) -> None:
    user = message.from_user
    if user is None or not config.is_admin(user.id):
        await message.answer("⛔ <b>Access denied.</b> This command is for admins only.")
        return
    if message.chat.type != "private":
        await message.answer("ℹ️ Open the admin menu in a private chat with the bot.")
        return
    _awaiting.pop(user.id, None)
    await message.answer(await _home_text(), reply_markup=_home_markup())


@router.callback_query(F.data == "adm:noop")
async def cb_noop(callback: CallbackQuery) -> None:
    await safe_answer_callback(callback)


@router.callback_query(F.data.startswith("adm:"))
async def cb_admin(callback: CallbackQuery) -> None:
    """Single dispatcher for the whole menu."""
    if not _is_admin_cb(callback):
        await safe_answer_callback(callback, "Admins only", show_alert=True)
        return

    actor = callback.from_user.id
    parts = (callback.data or "").split(":")
    action = parts[1] if len(parts) > 1 else ""
    # Leaving an input prompt by pressing any other button cancels it.
    _awaiting.pop(actor, None)

    try:
        target = int(parts[2]) if len(parts) > 2 and action not in ("list",) else None
    except ValueError:
        await safe_answer_callback(callback)
        return

    if action == "home":
        await _show(callback, await _home_text(), _home_markup())
    elif action == "close":
        if callback.message is not None and hasattr(callback.message, "delete"):
            try:
                await callback.message.delete()
            except Exception:  # noqa: BLE001 - an old message may be undeletable
                pass
    elif action == "list":
        flt = parts[2] if len(parts) > 2 else "a"
        try:
            page = int(parts[3]) if len(parts) > 3 else 0
        except ValueError:
            page = 0
        text, markup = await _render_list(actor, flt, page)
        await _show(callback, text, markup)
    elif action == "stats":
        await _show_stats(callback)
    elif action == "add":
        _awaiting[actor] = ("add", None)
        await _show(
            callback,
            "➕ <b>Add user</b>\n\nSend a numeric Telegram ID or <code>@username</code> "
            "(the user must have written to the bot once).",
            _cancel_markup(),
        )
    elif action.startswith("dg"):
        await _digest_action(callback, actor, action, parts)
        return
    elif target is not None:
        await _user_action(callback, actor, action, target, parts)
        return
    await safe_answer_callback(callback)


def _cancel_markup() -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    b.button(text="✖ Cancel", callback_data="adm:home")
    return b.as_markup()


async def _show_stats(callback: CallbackQuery) -> None:
    async with user_service() as service:
        data = await service.get_usage_stats_last_14_days()
    today = datetime.date.today()
    totals: dict[int, int] = {}
    today_total = 0
    for entry in data:
        totals[entry["telegram_id"]] = totals.get(entry["telegram_id"], 0) + entry["query_count"]
        if entry["date"] == today:
            today_total += entry["query_count"]
    lines = [
        "📊 <b>Statistics</b>\n",
        f"Today: <b>{today_total}</b> queries",
        f"14 days: <b>{sum(totals.values())}</b> queries, "
        f"<b>{len(totals)}</b> active users\n",
    ]
    for i, (tid, total) in enumerate(
        sorted(totals.items(), key=lambda kv: kv[1], reverse=True)[:10], 1
    ):
        lines.append(f"{i}. <code>{tid}</code> — {total}")
    b = InlineKeyboardBuilder()
    b.button(text="🏠 Menu", callback_data="adm:home")
    await _show(callback, "\n".join(lines), b.as_markup())


# --- Per-user actions --------------------------------------------------------


async def _user_action(
    callback: CallbackQuery, actor: int, action: str, target: int, parts: list[str]
) -> None:
    async with user_service() as service:
        level = await service.get_level(target)

    if action == "u":
        await _refresh_card(callback, target)
        await safe_answer_callback(callback)
        return

    if not _can_manage(actor, level):
        await safe_answer_callback(callback, "You cannot change this account", show_alert=True)
        return

    arg = parts[3] if len(parts) > 3 else ""
    notice = ""

    if action == "lim":
        if level != "user":
            await safe_answer_callback(callback, "Only for users with access", show_alert=True)
            return
        if arg == "i":
            _awaiting[actor] = ("limit", target)
            await _show(
                callback,
                f"✏️ <b>Daily limit for</b> <code>{target}</code>\n\n"
                "Send a number from 0 to "
                f"{MAX_LIMIT} (0 = no questions allowed).",
                _cancel_markup(),
            )
            await safe_answer_callback(callback)
            return
        async with user_service() as service:
            if arg == "d":
                await service.set_daily_limit(target, None)
                notice = "♻️ Limit reset to the default."
            else:
                try:
                    delta = int(arg)
                except ValueError:
                    await safe_answer_callback(callback)
                    return
                current = await service.get_daily_limit(target)
                new = max(0, min(MAX_LIMIT, current + delta))
                await service.set_daily_limit(target, new)
                notice = f"📊 Limit is now {new}."
    elif action == "rst":
        async with user_service() as service:
            done = await service.reset_daily_usage(target)
        notice = "🔄 Today's usage reset." if done else "ℹ️ No usage today."
    elif action == "rsp":
        async with user_service() as service:
            await service.set_system_prompt(target, config.default_system_prompt)
        notice = "🧹 System prompt reset to the default."
    elif action == "grant":
        profile = await _fetch_profile(target)
        async with user_service() as service:
            await service.add_allowed_user(target, added_by=actor, **profile)
        notice = f"✅ Access granted ({config.daily_limit}/day by default)."
    elif action == "revoke":
        async with user_service() as service:
            await service.remove_allowed_user(target)
        notice = "⬇️ Access revoked."
    elif action == "blk":
        async with user_service() as service:
            await service.set_blocked(target, level != "blocked")
        notice = "🔓 Unblocked." if level == "blocked" else "🚫 Blocked."
    elif action == "adm":
        if not config.is_owner(actor):
            await safe_answer_callback(callback, "Owners only", show_alert=True)
            return
        async with user_service() as service:
            await service.set_admin(target, level != "admin")
        notice = "⬇️ Admin rights removed." if level == "admin" else "🛡 Promoted to admin."
    elif action == "mdl":
        await _show_models(callback, target)
        await safe_answer_callback(callback)
        return
    elif action == "mset":
        try:
            model = config.ai_models[int(arg)]["id"]
        except (ValueError, IndexError):
            await safe_answer_callback(callback)
            return
        async with user_service() as service:
            await service.set_model(target, model)
        notice = f"🤖 Model set: <code>{html.escape(model)}</code>"
    else:
        await safe_answer_callback(callback)
        return

    await _refresh_card(callback, target, notice)
    await safe_answer_callback(callback)


async def _show_models(callback: CallbackQuery, target: int) -> None:
    async with user_service() as service:
        current = (await service.get_user_settings(target))["model"]
    b = InlineKeyboardBuilder()
    for index, model in enumerate(config.ai_models):
        mark = "✅ " if model["id"] == current else ""
        b.button(text=f"{mark}{model['name']}", callback_data=f"adm:mset:{target}:{index}")
    b.adjust(1)
    b.row(_button("◀ Back", f"adm:u:{target}"))
    await _show(
        callback,
        f"🤖 <b>Model for</b> <code>{target}</code>",
        b.as_markup(),
    )


# --- Typed input (add user / custom limit) -----------------------------------


def _expecting_input(message: Message) -> bool:
    user = message.from_user
    return (
        user is not None
        and user.id in _awaiting
        and message.chat.type == "private"
        and bool(message.text)
        and not message.text.startswith("/")
        and config.is_admin(user.id)
    )


@router.message(_expecting_input)
async def on_admin_input(message: Message) -> None:
    actor = message.from_user.id
    kind, target = _awaiting.pop(actor)
    text = (message.text or "").strip()

    if kind in ("dgadd", "dgprompt"):
        await _digest_input(message, actor, kind, text)
        return

    if kind == "add":
        found, error = await _resolve_target(text)
        if found is None:
            _awaiting[actor] = ("add", None)
            await message.answer(
                error or "❌ Send a numeric ID or @username.", reply_markup=_cancel_markup()
            )
            return
        async with user_service() as service:
            level = await service.get_level(found)
        if level in ("owner", "admin"):
            await message.answer("⚠️ Admins already have full access.")
            return
        profile = await _fetch_profile(found)
        async with user_service() as service:
            await service.add_allowed_user(found, added_by=actor, **profile)
        target, notice = found, "✅ Added to the allowed list."
    elif kind == "limit" and target is not None:
        try:
            value = int(text)
            if not 0 <= value <= MAX_LIMIT:
                raise ValueError
        except ValueError:
            _awaiting[actor] = ("limit", target)
            await message.answer(
                f"❌ Send a whole number from 0 to {MAX_LIMIT}.",
                reply_markup=_cancel_markup(),
            )
            return
        async with user_service() as service:
            level = await service.get_level(target)
            if not _can_manage(actor, level) or level != "user":
                await message.answer("⚠️ This account cannot be limited.")
                return
            await service.set_daily_limit(target, value)
        notice = f"📊 Limit is now {value}."
    else:
        return

    card, markup = await _render_card(actor, target, notice)
    await message.answer(card, reply_markup=markup)


# --- Channel digest ----------------------------------------------------------------

MAX_PROMPT_CHARS = 3000
_CHANNEL_TYPES = ("channel", "supergroup", "group")


async def _digest_screen(notice: str = "") -> tuple[str, InlineKeyboardMarkup]:
    channels = await digest.list_channels()
    model_id = await digest.get_model()
    model = config.resolve_model(model_id)
    prompt = await digest.get_prompt()
    custom = prompt != digest.DEFAULT_DIGEST_PROMPT
    preview = html.escape(prompt if len(prompt) <= 400 else prompt[:400] + "…")
    state = "on" if config.enable_digest else "off (ENABLE_DIGEST=false)"
    lines = [
        "📰 <b>Channel digest</b>",
        (
            f"Daily at <b>{html.escape(config.digest_time)}</b> "
            f"({html.escape(config.reminder_timezone or 'UTC')}) — {state}"
        ),
        f"🤖 Model: <code>{html.escape(model['name'])}</code>",
        f"📝 Prompt{' (custom)' if custom else ' (default)'}:\n<blockquote>{preview}</blockquote>",
        f"📡 Channels: <b>{len(channels)}</b>",
        (
            "<i>The bot must be an admin of the channel to read posts and publish "
            f"the summary. Under {config.digest_min_messages} messages up to "
            f"{config.digest_context_messages} earlier ones are added as context.</i>"
        ),
    ]
    if notice:
        lines.append(f"\n{notice}")

    b = InlineKeyboardBuilder()
    for chat_id, title in channels:
        label = (title or str(chat_id))[:30]
        b.row(_button(f"📡 {label}", f"adm:dgc:{chat_id}"))
    b.row(_button("➕ Add channel", "adm:dga"), _button("▶️ Run now", "adm:dgrun"))
    b.row(_button("🤖 Model", "adm:dgm"), _button("✏️ Prompt", "adm:dgp"))
    if custom:
        b.row(_button("♻️ Default prompt", "adm:dgd"))
    b.row(_button("🏠 Menu", "adm:home"))
    return "\n".join(lines), b.as_markup()


async def _digest_show(callback: CallbackQuery, notice: str = "") -> None:
    text, markup = await _digest_screen(notice)
    await _show(callback, text, markup)


def _digest_cancel() -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    b.button(text="✖ Cancel", callback_data="adm:dg")
    return b.as_markup()


async def _digest_action(
    callback: CallbackQuery, actor: int, action: str, parts: list[str]
) -> None:
    arg = parts[2] if len(parts) > 2 else ""
    if action == "dg":
        await _digest_show(callback)
    elif action == "dga":
        _awaiting[actor] = ("dgadd", None)
        await _show(
            callback,
            "➕ <b>Add channel</b>\n\nForward any post from the channel, or send its "
            "numeric ID (<code>-100…</code>) or <code>@username</code>.\n"
            "Add the bot to the channel as an admin first.",
            _digest_cancel(),
        )
    elif action == "dgc":
        try:
            chat_id = int(arg)
        except ValueError:
            await safe_answer_callback(callback)
            return
        title = dict(await digest.list_channels()).get(chat_id)
        if title is None:
            await _digest_show(callback, "ℹ️ Channel not found.")
        else:
            b = InlineKeyboardBuilder()
            b.row(_button("🗑 Remove", f"adm:dgx:{chat_id}"), _button("◀ Digest", "adm:dg"))
            await _show(
                callback,
                f"📡 <b>{html.escape(title or str(chat_id))}</b>\n<code>{chat_id}</code>\n\n"
                "Removing it also deletes the stored messages.",
                b.as_markup(),
            )
    elif action == "dgx":
        try:
            removed = await digest.remove_channel(int(arg))
        except ValueError:
            await safe_answer_callback(callback)
            return
        await _digest_show(callback, "🗑 Channel removed." if removed else "ℹ️ Not found.")
    elif action == "dgm":
        current = await digest.get_model()
        b = InlineKeyboardBuilder()
        for index, model in enumerate(config.ai_models):
            mark = "✅ " if model["id"] == current else ""
            b.button(text=f"{mark}{model['name']}", callback_data=f"adm:dgs:{index}")
        b.adjust(1)
        b.row(_button("◀ Digest", "adm:dg"))
        await _show(callback, "🤖 <b>Model for the digest</b>", b.as_markup())
    elif action == "dgs":
        try:
            model = config.ai_models[int(arg)]
        except (ValueError, IndexError):
            await safe_answer_callback(callback)
            return
        await digest.set_setting(digest.SETTING_MODEL, model["id"])
        await _digest_show(callback, f"🤖 Model set: <code>{html.escape(model['name'])}</code>")
    elif action == "dgp":
        _awaiting[actor] = ("dgprompt", None)
        await _show(
            callback,
            "✏️ <b>Digest prompt</b>\n\nCurrent:\n"
            f"<blockquote>{html.escape((await digest.get_prompt())[:1500])}</blockquote>\n"
            f"Send the new system prompt (up to {MAX_PROMPT_CHARS} characters).",
            _digest_cancel(),
        )
    elif action == "dgd":
        await digest.set_setting(digest.SETTING_PROMPT, None)
        await _digest_show(callback, "♻️ Prompt reset to the default.")
    elif action == "dgrun":
        await safe_answer_callback(callback, "Running…")
        result = await digest.run_digest(notify=actor)
        if result.skipped:
            notice = "⏳ A digest run is already in progress."
        elif not result.reports:
            notice = "ℹ️ No channels yet."
        else:
            posted = sum(r.posted for r in result.reports)
            notice = f"✅ Done: {posted}/{len(result.reports)} channel(s) posted."
        await _digest_show(callback, notice)
        return
    await safe_answer_callback(callback)


async def _resolve_channel(message: Message, text: str) -> tuple[Any, str]:
    """Chat object from a forwarded post, a numeric ID or a @username."""
    origin = getattr(message, "forward_origin", None)
    chat = getattr(origin, "chat", None)
    ref: Any
    if chat is not None:
        ref = chat.id
    elif text.lstrip("-").isdigit():
        ref = int(text)
    elif text.startswith("@") and len(text) > 1:
        ref = text.split()[0]
    else:
        return None, "❌ Forward a post from the channel, or send its ID / @username."
    try:
        found = await bot.get_chat(ref)
    except Exception:  # noqa: BLE001 - unknown chat, bot not a member, bad ID
        return None, "❌ Telegram does not know this chat, or the bot cannot see it."
    if found.type not in _CHANNEL_TYPES:
        return None, "❌ This is not a channel or group."
    return found, ""


async def _digest_input(message: Message, actor: int, kind: str, text: str) -> None:
    if kind == "dgprompt":
        if not text or len(text) > MAX_PROMPT_CHARS:
            _awaiting[actor] = ("dgprompt", None)
            await message.answer(
                f"❌ Send a prompt of 1–{MAX_PROMPT_CHARS} characters.",
                reply_markup=_digest_cancel(),
            )
            return
        await digest.set_setting(digest.SETTING_PROMPT, text)
        screen, markup = await _digest_screen("✅ Prompt saved.")
        await message.answer(screen, reply_markup=markup)
        return

    chat, error = await _resolve_channel(message, text)
    if chat is None:
        _awaiting[actor] = ("dgadd", None)
        await message.answer(error, reply_markup=_digest_cancel())
        return
    await digest.add_channel(chat.id, chat.title or chat.username or str(chat.id), actor)
    notice = f"✅ Added: <b>{html.escape(chat.title or str(chat.id))}</b>"
    try:
        member = await bot.get_chat_member(chat.id, get_bot_id())
        if member.status not in ("administrator", "creator"):
            raise ValueError
    except Exception:  # noqa: BLE001 - not a member / not an admin
        notice += "\n⚠️ The bot is not an admin there yet — it will not see posts."
    screen, markup = await _digest_screen(notice)
    await message.answer(screen, reply_markup=markup)
