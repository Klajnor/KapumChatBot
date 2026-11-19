from __future__ import annotations

import asyncio
import html
import logging
import time

from aiogram import Router
from aiogram.enums import ChatType
from aiogram.exceptions import TelegramAPIError, TelegramRetryAfter
from aiogram.filters import CommandStart
from aiogram.types import InlineKeyboardMarkup, Message
from aiogram.utils.keyboard import InlineKeyboardBuilder

from bot.bot import get_bot_username
from bot.config import config
from bot.keyboards.reply import menu_for_user
from bot.services.themed_responses import get_random_themed_response
from bot.services.user_service import user_service

logger = logging.getLogger(__name__)

router = Router()

# Повторное нажатие /start зарегистрированным пользователем не дублирует заявку:
# кроме проверки «первый раз ли» (строка в users), держим короткий guard на
# случай гонки двух одновременных /start одного пользователя.
_REQUEST_GUARD_SECONDS = 300
_recent_requests: dict[int, float] = {}

_BASE_COMMANDS = (
    "💬 <b>Send any text</b> — chat with the AI\n"
    "🖼️ <b>Send an image</b> — analyze it (vision models)\n"
    "🎤 <b>Send a voice message</b> — I'll recognize it and reply\n"
    "📄 <b>Send a document</b> (PDF/DOCX/XLSX/PPTX/TXT) — I'll read and analyze it\n"
    "🔗 <b>Send a link</b> — I'll fetch the page and summarize it\n"
    "↩️ <b>Reply to a message</b> — I'll answer about the quoted text or photo\n"
    "🤖 /models — choose an AI model\n"
    "🎨 /image &lt;prompt&gt; — generate an image\n"
    "⏰ /remind &lt;when&gt; &lt;text&gt; — reminder or scheduled AI request\n"
    "🎭 /persona — choose an AI personality\n"
    "📤 /export — save the conversation (MD/PDF, sent to DM)\n"
    "📝 /prompt — view or set the system prompt (private chat only)\n"
    "🧹 /clear — clear conversation history\n"
    "📊 /stats — your usage statistics (private chat only)\n"
    "⚙️ /settings — current settings (private chat only)\n"
    "🧭 /menu — show the button menu\n"
    "❓ /help — all commands"
)


def _guest_mode_hint() -> str:
    username = get_bot_username()
    if not config.enable_guest_mode or not username:
        return ""
    return (
        f"\n\n💡 <b>Summon me in any chat:</b> mention "
        f"<code>@{html.escape(username)}</code> in any group or private "
        f"conversation — even one I'm not a member of — and I'll reply right "
        f"there."
    )


def _guest_hint() -> str:
    if not config.allow_guest_access:
        return ""
    return (
        f"\n\n🎁 <b>Free trial:</b> you can send up to "
        f"{config.guest_message_limit} questions for free."
    )


def _request_keyboard(user_id: int) -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    b.button(text="✅ Approve", callback_data=f"appr:a:{user_id}")
    b.button(text="🚫 Block", callback_data=f"appr:b:{user_id}")
    b.adjust(2)
    return b.as_markup()


def _admin_recipients() -> list[int]:
    """Owners plus promoted admins, deduplicated, in a stable order."""
    return list(dict.fromkeys([*config.owner_ids, *config.extra_admin_ids]))


def _request_text(user_id: int, name: str, username: str | None, when: str) -> str:
    who = html.escape(name) if name else "—"
    nick = f" · @{html.escape(username)}" if username else ""
    return (
        "🆕 <b>New user started the bot</b>\n\n"
        f"👤 <b>{who}</b>{nick}\n"
        f"🆔 <code>{user_id}</code>\n"
        f"🕐 First <code>/start</code>: {html.escape(when)}\n\n"
        "Approve full access or block this user."
    )


async def _notify_admins_about_new_user(message: Message) -> None:
    """Ask every admin to approve (or block) a first-time /start user.

    The guard set suppresses duplicates when two /start updates of the same
    user race; the DB existence check is the primary filter. Delivery errors
    (admin blocked the bot, flood limits) are logged, never raised: the
    welcome answer to the newcomer must go out regardless.
    """
    user = message.from_user
    if user is None:
        return
    recipients = _admin_recipients()
    if not recipients:
        logger.warning("Access request: no admins configured (OWNER_IDS empty)")
        return

    now = time.monotonic()
    if _recent_requests.get(user.id, 0.0) > now:
        return
    _recent_requests[user.id] = now + _REQUEST_GUARD_SECONDS
    # Не копим записи вечно: подчищаем просроченные на каждом обходе.
    for stale in [uid for uid, until in _recent_requests.items() if until <= now]:
        _recent_requests.pop(stale, None)

    username = user.username
    name = " ".join(
        part for part in (user.first_name, user.last_name) if part
    ).strip()
    when = getattr(message, "date", None)
    when_text = when.strftime("%Y-%m-%d %H:%M UTC") if when else "—"
    text = _request_text(user.id, name, username, when_text)
    keyboard = _request_keyboard(user.id)

    sent_any = False
    for admin_id in recipients:
        if admin_id == user.id:
            continue
        try:
            await message.bot.send_message(admin_id, text, reply_markup=keyboard)
            sent_any = True
        except TelegramRetryAfter as exc:
            await asyncio.sleep(exc.retry_after + 1)
            try:
                await message.bot.send_message(admin_id, text, reply_markup=keyboard)
                sent_any = True
            except TelegramAPIError as retry_exc:
                logger.warning(
                    "Access request to admin %s failed after retry: %s",
                    admin_id, retry_exc,
                )
        except TelegramAPIError as exc:
            logger.warning("Access request to admin %s failed: %s", admin_id, exc)
    if not sent_any:
        logger.warning(
            "Access request for user %s could not be delivered to any admin",
            user.id,
        )


@router.message(CommandStart())
async def cmd_start(message: Message) -> None:
    """Handle /start command — register user and show welcome message."""
    user = message.from_user
    if user is None:
        return

    async with user_service() as service:
        # Заявка админам — только на самом первом /start (раньше строки в users
        # не было): повторные /start известного пользователя ничего не рассылают.
        first_time = not await service.is_known_user(user.id)
        await service.register_user(
            telegram_id=user.id,
            username=user.username,
            first_name=user.first_name,
            last_name=user.last_name,
        )
        is_allowed = await service.is_allowed_user(user.id)
        daily_limit = await service.get_daily_limit(user.id) if is_allowed else None

    if first_time and not is_allowed and not config.is_admin(user.id):
        await _notify_admins_about_new_user(message)

    safe_name = html.escape(user.first_name or "there")
    # Persistent button menu: private chats of admins / whitelisted users only.
    menu = (
        await menu_for_user(user.id)
        if message.chat.type == ChatType.PRIVATE
        else None
    )

    if config.is_admin(user.id):
        welcome_text = (
            f"👋 <b>Hello, Admin {safe_name}!</b>\n\n"
            f"I'm an AI-powered bot connected to <b>Qwen</b> (Alibaba Cloud "
            f"Model Studio). I can chat, analyze images and adapt to your "
            f"preferences.\n\n"
            f"<b>Your commands:</b>\n"
            f"{_BASE_COMMANDS}\n\n"
            f"<b>👑 Admin commands:</b>\n"
            f"➕ /add_user &lt;id|@username&gt; — allow a user to use the bot\n"
            f"➖ /remove_user &lt;id|@username&gt; — remove a user from the allowed list\n"
            f"📋 /list_users — show all allowed users\n"
            f"📈 /admin_stats — usage statistics (14 days)\n"
            f"📣 /broadcast &lt;text&gt; — message every user\n"
            f"🔄 /reset_daily_usage &lt;id&gt; — reset a user's daily usage"
            f"{_guest_mode_hint()}\n\n"
            f"<i>Let's get started! 🚀</i>"
        )
        await message.answer(welcome_text, reply_markup=menu)
    elif is_allowed:
        await message.answer(
            f"👋 <b>Welcome back, {safe_name}!</b>\n\n"
            f"You have been granted access to use this AI bot.\n\n"
            f"<b>Available commands:</b>\n"
            f"{_BASE_COMMANDS}\n\n"
            f"⚠️ <b>Daily limit:</b> {daily_limit} questions per day"
            f"{_guest_mode_hint()}\n\n"
            f"<i>Happy chatting! 🚀</i>",
            reply_markup=menu,
        )
    else:
        await message.answer(
            f"🌌 <b>Greetings, {safe_name}.</b>\n\n"
            f"This communication channel is reserved for authorized personnel. "
            f"Your transmission has been received and logged — but your ID is "
            f"not on the admin manifest."
            f"{_guest_hint()}"
            f"{_guest_mode_hint()}\n\n"
            f"{get_random_themed_response()}\n\n"
            f"<i>— Transmission ends.</i>"
        )
