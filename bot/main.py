from __future__ import annotations

import asyncio
import logging

from bot.bot import bot, dp, setup_bot
from bot.config import config
from bot.database.session import engine, init_db
from bot.services.ai_client import ai_client

# Import all handlers and register their routers with the dispatcher.
from bot.handlers import (  # noqa: F401
    admin,
    chat,
    clear,
    guest,
    help,
    image,
    models,
    prompt,
    start,
    stats,
)

# Order matters: command routers are registered before the catch-all chat router.
dp.include_router(start.router)
dp.include_router(help.router)
dp.include_router(models.router)
dp.include_router(image.router)
dp.include_router(prompt.router)
dp.include_router(clear.router)
dp.include_router(stats.router)
dp.include_router(admin.router)
if config.enable_guest_mode:
    dp.include_router(guest.router)
dp.include_router(chat.router)

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

    await setup_bot()

    # Hide curated models the account cannot access via the API.
    available = await ai_client.refresh_available_models()
    logger.info(
        "Available models: %s",
        ", ".join(m["id"] for m in available),
    )
    if ai_client.is_model_available(config.default_model) is False:
        logger.warning(
            "DEFAULT_MODEL %r is not in the account's model list; "
            "pick another one with /models",
            config.default_model,
        )

    logger.info(
        "Starting bot polling (guest access: %s, guest mode: %s)",
        config.allow_guest_access,
        config.enable_guest_mode,
    )

    try:
        await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())
    finally:
        await ai_client.close()
        await bot.session.close()
        await engine.dispose()
        logger.info("Bot stopped")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Shutdown requested")