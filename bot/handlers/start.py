from __future__ import annotations

import html

from aiogram import Router
from aiogram.filters import CommandStart
from aiogram.types import Message

from bot.bot import get_bot_username
from bot.config import config
from bot.services.themed_responses import get_random_themed_response
from bot.services.user_service import user_service

router = Router()

_BASE_COMMANDS = (
    "💬 <b>Send any text</b> — chat with the AI\n"
    "🖼️ <b>Send an image</b> — analyze it (vision models)\n"
    "🎤 <b>Send a voice message</b> — I'll recognize it and reply\n"
    "📄 <b>Send a document</b> (PDF/DOCX/XLSX/PPTX/TXT) — I'll read and analyze it\n"
    "🔗 <b>Send a link</b> — I'll fetch the page and summarize it\n"
    "↩️ <b>Reply to a message</b> — I'll answer about the quoted text or photo\n"
    "🤖 /models — choose an AI model\n"
    "🎨 /image &lt;prompt&gt; — generate an image\n"
    "🩹 /sticker &lt;prompt&gt; — generate a Telegram sticker\n"
    "⏰ /remind &lt;when&gt; &lt;text&gt; — reminder or scheduled AI request\n"
    "🎭 /persona — choose an AI personality\n"
    "📤 /export — save the conversation (MD/PDF, sent to DM)\n"
    "📝 /prompt — view or set the system prompt (private chat only)\n"
    "🧹 /clear — clear conversation history\n"
    "📊 /stats — your usage statistics (private chat only)\n"
    "⚙️ /settings — current settings (private chat only)\n"
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


@router.message(CommandStart())
async def cmd_start(message: Message) -> None:
    """Handle /start command — register user and show welcome message."""
    user = message.from_user
    if user is None:
        return

    async with user_service() as service:
        await service.register_user(
            telegram_id=user.id,
            username=user.username,
            first_name=user.first_name,
            last_name=user.last_name,
        )
        is_allowed = await service.is_allowed_user(user.id)

    safe_name = html.escape(user.first_name or "there")

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
        await message.answer(welcome_text)
    elif is_allowed:
        await message.answer(
            f"👋 <b>Welcome back, {safe_name}!</b>\n\n"
            f"You have been granted access to use this AI bot.\n\n"
            f"<b>Available commands:</b>\n"
            f"{_BASE_COMMANDS}\n\n"
            f"⚠️ <b>Daily limit:</b> {config.daily_limit} questions per day"
            f"{_guest_mode_hint()}\n\n"
            f"<i>Happy chatting! 🚀</i>"
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