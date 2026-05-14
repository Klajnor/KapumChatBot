from __future__ import annotations

import asyncio
import logging

from bot.bot import bot, dp
from bot.config import config
from bot.database.session import init_db

# Import all handlers and register their routers with the dispatcher
from bot.handlers import chat, clear, help, models, prompt, start, stats  # noqa: F401

dp.include_router(start.router)
dp.include_router(help.router)
dp.include_router(models.router)
dp.include_router(prompt.router)
dp.include_router(clear.router)
dp.include_router(chat.router)
dp.include_router(stats.router)

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