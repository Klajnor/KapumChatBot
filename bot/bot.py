from __future__ import annotations

import logging

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.types import BotCommand

from bot.config import config

logger = logging.getLogger(__name__)

# Initialize bot with HTML parse mode by default.
bot = Bot(
    token=config.bot_token,
    default=DefaultBotProperties(parse_mode=ParseMode.HTML),
)

# Initialize dispatcher.
dp = Dispatcher()

# Resolved at startup (see ``setup_bot``).
bot_id: int = 0
bot_username: str = config.bot_username

_COMMANDS = [
    BotCommand(command="start", description="Start / welcome"),
    BotCommand(command="help", description="Show all commands"),
    BotCommand(command="menu", description="Show the button menu"),
    BotCommand(command="models", description="Choose an AI model"),
    BotCommand(command="model", description="Show the current model"),
    BotCommand(command="image", description="Generate an image from a prompt"),
    BotCommand(command="sticker", description="Generate a Telegram sticker"),
    BotCommand(command="remind", description="Set a reminder or scheduled AI request"),
    BotCommand(command="reminders", description="List your reminders"),
    BotCommand(command="delreminder", description="Delete a reminder"),
    BotCommand(command="export", description="Export the conversation (MD/PDF)"),
    BotCommand(command="persona", description="Choose an AI personality"),
    BotCommand(command="prompt", description="View or set the system prompt"),
    BotCommand(command="clear", description="Clear conversation history"),
    BotCommand(command="stats", description="Your usage statistics"),
    BotCommand(command="settings", description="Your current settings"),
]


async def setup_bot() -> None:
    """Resolve the bot identity and register the command list."""
    global bot_id, bot_username

    me = await bot.get_me()
    bot_id = me.id
    bot_username = me.username or config.bot_username
    logger.info("Bot identity resolved: @%s (id=%s)", bot_username, bot_id)

    try:
        await bot.set_my_commands(_COMMANDS)
    except Exception as exc:  # pragma: no cover - network dependent  # noqa: BLE001
        logger.warning("Failed to register bot commands: %s", exc)


def get_bot_id() -> int:
    """Current bot user ID (0 until ``setup_bot`` has run)."""
    return bot_id


def get_bot_username() -> str:
    """Current bot username without ``@`` (empty until resolved).

    Use this accessor instead of importing ``bot_username`` directly: the value
    is assigned at startup, so a plain ``from ... import`` would capture the
    empty initial value.
    """
    return bot_username