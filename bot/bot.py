from __future__ import annotations

import asyncio
import logging

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramNetworkError, TelegramServerError
from aiogram.types import BotCommand

from bot.config import config

logger = logging.getLogger(__name__)

# Transient Telegram failures: a request timeout or a 5xx/restart must never
# stop the bot. Startup and polling are retried with capped exponential
# backoff; a bad token (401) or a blocked chat still fails fast.
RETRYABLE_TELEGRAM_ERRORS = (TelegramNetworkError, TelegramServerError)
RETRY_INITIAL_DELAY = 1.0
RETRY_MAX_DELAY = 60.0

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


def next_retry_delay(delay: float) -> float:
    """Double a retry delay, capped so recovery stays reasonably quick."""
    return min(delay * 2, RETRY_MAX_DELAY)


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
    """Resolve the bot identity and register the command list.

    ``getMe`` is retried on transient network/server failures: a request
    timeout while Telegram is unreachable must not stop the bot before polling
    even starts. ``bot.me()`` caches the result, so the polling loop does not
    repeat the call.
    """
    global bot_id, bot_username

    delay = RETRY_INITIAL_DELAY
    while True:
        try:
            me = await bot.me()
            break
        except RETRYABLE_TELEGRAM_ERRORS as exc:
            logger.warning(
                "Telegram is unreachable while resolving the bot identity "
                "(%s); retrying in %.0f s",
                exc,
                delay,
            )
            await asyncio.sleep(delay)
            delay = next_retry_delay(delay)

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