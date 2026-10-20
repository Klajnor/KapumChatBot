from __future__ import annotations

import asyncio
import logging

from aiogram.types import (
    ErrorEvent,
    InlineQueryResultArticle,
    InputTextMessageContent,
)

from bot.bot import bot, dp, setup_bot
from bot.config import config
from bot.database.session import engine, init_db
from bot.services.ai_client import ai_client
from bot.services.telegram_utils import safe_answer_guest

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

ERROR_REPLY = (
    "⚠️ <b>Something went wrong while preparing the answer.</b>\n\n"
    "The error has been logged. Please try again in a moment — if it keeps "
    "happening, the database or the AI API may be unavailable."
)


@dp.error()
async def on_unhandled_error(event: ErrorEvent) -> None:
    """Last line of defence: log the failure and tell the user.

    Without this handler an exception anywhere in a pipeline (database down,
    Telegram rate limit, unexpected API shape) is only logged, and the user
    simply never gets a reply.
    """
    update = event.update
    logger.error(
        "Unhandled error while processing update %s: %r",
        getattr(update, "update_id", "?"),
        event.exception,
        exc_info=event.exception,
    )

    message = None
    if update is not None:
        message = update.message or update.edited_message
        if message is None and update.callback_query is not None:
            message = update.callback_query.message
        if message is None and getattr(update, "guest_message", None) is not None:
            # A guest query allows exactly one reply: answer it with the error.
            await safe_answer_guest(
                update.guest_message,
                InlineQueryResultArticle(
                    id="error",
                    title="The bot could not answer",
                    input_message_content=InputTextMessageContent(
                        message_text="⚠️ Something went wrong. Please try again."
                    ),
                ),
            )
            return

    # Old callback messages and channel posts have no usable chat to answer to.
    if message is None or not hasattr(message, "chat"):
        return

    try:
        await message.answer(ERROR_REPLY)
    except Exception as exc:  # noqa: BLE001 - notifying must never raise again
        logger.warning("Could not notify the user about the error: %s", exc)


async def main() -> None:
    """Entry point: validate config, init DB, start polling."""
    config.validate()
    logger.info("Configuration validated successfully")

    try:
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