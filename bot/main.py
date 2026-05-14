from __future__ import annotations

import asyncio
import logging

from bot.bot import bot, dp
from bot.config import config
from bot.database.session import init_db

# Import all handlers to register them
from bot.handlers import start, help, models, prompt, clear, chat, stats  # noqa: F401

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def main() -> None:
    """Entry point: validate config, init DB, start polling."""
    config.validate()
    logger.info("Configuration validated successfully")

    await init_db()
    logger.info("Database initialized")

    logger.info("Starting bot polling...")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())