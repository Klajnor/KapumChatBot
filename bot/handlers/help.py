from __future__ import annotations

import html

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from bot.bot import get_bot_username
from bot.config import config
from bot.services.user_service import user_service

router = Router()

_CHAT_SECTION = (
    "<b>Chat:</b>\n"
    "• <b>Send any text</b> — chat with the AI\n"
    "• <b>Send an image</b> — analyze it with a vision model\n"
    "• <b>Send a voice message</b> — I'll recognize it and reply\n"
    "• <b>Send a document</b> (PDF/DOCX/XLSX/PPTX/TXT/code) — I'll read and analyze it\n"
    "• <b>Send a link</b> — I'll fetch the page and summarize it\n"
    "• <b>Reply to a message</b> — I'll answer about the quoted text or photo\n\n"
    "<b>Commands:</b>\n"
    "• /start — welcome message and registration\n"
    "• /help — this help message\n"
    "• /models — select an AI model\n"
    "• /model — show your current model (private chat only)\n"
    "• /image &lt;prompt&gt; — generate an image\n"
    "• /sticker &lt;prompt&gt; — generate a Telegram sticker\n"
    "• /remind &lt;when&gt; &lt;text&gt; — reminder or scheduled AI request\n"
    "• /reminders — list your reminders\n"
    "• /delreminder &lt;N|all&gt; — delete a reminder\n"
    "• /export — save this conversation as Markdown or PDF (sent to DM)\n"
    "• /persona — choose an AI personality (preset system prompt)\n"
    "• /prompt — view the current system prompt (private chat only)\n"
    "• /prompt &lt;text&gt; — set a new system prompt (private chat only)\n"
    "• /clear — clear the history of this chat\n"
    "• /stats — your usage statistics (private chat only)\n"
    "• /settings — your current settings (private chat only)"
)

_TIPS = (
    "<b>💡 Tips:</b>\n"
    "• Pick a ready-made personality with /persona, or write your own with /prompt\n"
    "• Switch models with /models (models marked 🖼️ support images)\n"
    "• Ask to draw something and I'll reply with a generated image\n"
    "• <code>/remind daily at 9:00 ai: …</code> — a scheduled request to the AI\n"
    "• Use /clear to start a fresh conversation\n"
    "• In groups, mention me or reply to my message to get an answer"
)


def _guest_mode_tip() -> str:
    username = get_bot_username()
    if not config.enable_guest_mode or not username:
        return ""
    return (
        f"\n\n<b>🌍 Works in any chat:</b> mention "
        f"<code>@{html.escape(username)}</code> in any group or private "
        f"conversation — even where I'm not a member — and I'll reply there."
    )


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    """Handle /help command — show all available commands."""
    user = message.from_user
    if user is None:
        return

    is_admin = config.is_admin(user.id)

    is_allowed = False
    if not is_admin:
        async with user_service() as service:
            is_allowed = await service.is_allowed_user(user.id)

    if is_admin:
        help_text = (
            "<b>🤖 Available Commands</b>\n\n"
            f"{_CHAT_SECTION}\n\n"
            "<b>👑 Admin Commands:</b>\n"
            "• /admin — admin menu: users, access levels, daily limits, "
            "📰 Digest (daily channel summary: channels, model, prompt, time)\n"
            "• /digest_on — <i>in a group</i>: add this group to the daily digest\n"
            "• /digest_off — <i>in a group</i>: remove this group from the digest\n"
            "• /add_user &lt;id|@username&gt; — add a user to the allowed list\n"
            "• /remove_user &lt;id|@username&gt; — remove a user from the allowed list\n"
            "• /list_users — show all allowed users\n"
            "• /admin_stats — usage statistics for the last 14 days\n"
            "• /broadcast &lt;text&gt; — send an announcement to every user\n"
            "• /mcp — MCP servers (the Obsidian «second brain») and their tools\n"
            "• /reset_daily_usage &lt;id&gt; — reset a user's daily usage"
            f"{_guest_mode_tip()}\n\n"
            f"{_TIPS}"
        )
    elif is_allowed:
        help_text = (
            "<b>🤖 Available Commands</b>\n\n"
            f"{_CHAT_SECTION}\n\n"
            f"⚠️ <b>Daily limit:</b> {config.daily_limit} questions per day"
            f"{_guest_mode_tip()}\n\n"
            f"{_TIPS}"
        )
    else:
        guest_note = (
            f"🎁 <b>Free trial:</b> up to {config.guest_message_limit} questions."
            if config.allow_guest_access
            else ""
        )
        help_text = (
            "<b>🤖 Available Commands</b>\n\n"
            "<b>Chat:</b>\n"
            "• <b>Send any text</b> — chat with the AI\n"
            "• <b>Send an image</b> — analyze it (vision models)\n\n"
            "<b>Commands:</b>\n"
            "• /start — welcome message and registration\n"
            "• /help — this help message\n\n"
            f"{guest_note}"
            f"{_guest_mode_tip()}\n\n"
            "<b>💡 Note:</b> this bot is limited to authorized personnel. "
            "Contact an administrator if you need full access."
        )

    await message.answer(help_text)