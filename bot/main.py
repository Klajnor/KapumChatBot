from __future__ import annotations

import asyncio
import logging

from aiogram.types import (
    ErrorEvent,
    InlineQueryResultArticle,
    InputTextMessageContent,
)

from bot.bot import (
    RETRY_INITIAL_DELAY,
    RETRYABLE_TELEGRAM_ERRORS,
    bot,
    dp,
    next_retry_delay,
    setup_bot,
)
from bot.config import config
from bot.database.session import engine, init_db

# Import all handlers and register their routers with the dispatcher.
from bot.handlers import (
    admin,
    admin_panel,
    brain,
    chat,
    clear,
    export,
    guest,
    help,
    image,
    menu,
    models,
    prompt,
    reminders,
    start,
    stats,
    sticker_sets,
    tasks,
)
from bot.handlers import (
    digest as digest_handler,
)
from bot.handlers.chat import _album_cleanup
from bot.logging_setup import setup_file_logging
from bot.services import telegraph, thinking
from bot.services.ai_client import ai_client, ffmpeg_available
from bot.services.digest import digest_scheduler_loop
from bot.services.health import start_health_server, stop_health_server
from bot.services.mcp import mcp_manager
from bot.services.reminders import cancel_pending_fires, reminder_scheduler_loop
from bot.services.telegram_utils import safe_answer_guest
from bot.services.user_service import user_service

# Order matters: command routers are registered before the catch-all chat router.
# The digest router only observes channel/group posts and passes them on.
dp.include_router(digest_handler.router)
# Форвард владельца — данные, а не команда (даже если текст начинается с /clear).
dp.include_router(brain.router)
dp.include_router(start.router)
dp.include_router(help.router)
dp.include_router(models.router)
dp.include_router(image.router)
dp.include_router(sticker_sets.router)
dp.include_router(prompt.router)
dp.include_router(reminders.router)
dp.include_router(export.router)
dp.include_router(clear.router)
dp.include_router(stats.router)
# Menu buttons arrive as plain text: they must be matched before admin_panel
# (admin text input) and the catch-all chat router.
dp.include_router(menu.router)
dp.include_router(admin.router)
dp.include_router(tasks.router)
dp.include_router(admin_panel.router)
if config.enable_guest_mode:
    dp.include_router(guest.router)
dp.include_router(chat.router)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Full prompts are an explicit DEBUG-level opt-in: raise the client logger only,
# so aiohttp/Telegram libraries do not flood the journal at DEBUG.
if config.log_prompts:
    logging.getLogger("bot.services.ai_client").setLevel(logging.DEBUG)

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
    if isinstance(event.exception, RETRYABLE_TELEGRAM_ERRORS):
        # Telegram is unreachable: the user cannot be reached either, and
        # trying to send an explanation would only block the handler for one
        # more request timeout.
        logger.warning(
            "Telegram is unreachable while handling update %s: %s",
            getattr(update, "update_id", "?"),
            event.exception,
        )
        return

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


async def _poll_forever() -> None:
    """Run polling until shutdown, surviving transient Telegram failures.

    aiogram retries failed ``getUpdates`` calls itself, but its polling task
    first calls ``getMe`` and runs startup callbacks outside that protection.
    A single request timeout there used to propagate out of ``start_polling``
    and terminate the process, so the whole loop is retried here. A normal
    return means a shutdown signal, not a failure.
    """
    delay = RETRY_INITIAL_DELAY
    while True:
        try:
            await dp.start_polling(
                bot,
                allowed_updates=dp.resolve_used_update_types(),
                close_bot_session=False,
            )
            return
        except RETRYABLE_TELEGRAM_ERRORS as exc:
            logger.warning(
                "Polling stopped by a transient Telegram error (%s); "
                "restarting in %.0f s",
                exc,
                delay,
            )
            await asyncio.sleep(delay)
            delay = next_retry_delay(delay)


async def main() -> None:
    """Entry point: validate config, init DB, start polling."""
    config.validate()
    setup_file_logging()
    logger.info("Configuration validated successfully")
    if config.log_prompts:
        logger.warning(
            "LOG_PROMPTS=true: full conversation texts (prompts, history, "
            "tool results) are written to the DEBUG log. Keep the log files "
            "private and turn this off when debugging is finished."
        )

    cleanup_task: asyncio.Task[None] | None = None
    reminder_task: asyncio.Task[None] | None = None
    digest_task: asyncio.Task[None] | None = None
    health_runner = None
    try:
        await init_db()
        logger.info("Database initialized")

        # Admin overrides for thinking of user-visible answers (.env defaults
        # apply when no value is stored).
        await thinking.load()

        async with user_service() as service:
            await service.load_granted_admins()

        await setup_bot()

        # Optional HTTP liveness endpoint (HEALTH_PORT; disabled by default).
        health_runner = await start_health_server()

        # Start the periodic album-cleanup background task (prevents memory
        # leak from stale media-group entries).
        cleanup_task = asyncio.create_task(_album_cleanup())

        # Start the reminder scheduler (fires due reminders / AI requests).
        if config.enable_reminders:
            reminder_task = asyncio.create_task(reminder_scheduler_loop())

        # Daily channel digest (channels, model and prompt: /admin → Digest).
        if config.enable_digest:
            digest_task = asyncio.create_task(digest_scheduler_loop())

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

        # Voice recognition needs ffmpeg to convert OGG/Opus → PCM16 for the
        # Realtime ASR WebSocket.  Warn loudly but keep the bot running: text,
        # photos and images still work, and the voice handler degrades to the
        # "could not recognize" reply.
        if config.enable_voice and not ffmpeg_available():
            logger.warning(
                "ffmpeg not found on PATH: voice messages cannot be recognized. "
                "Install it (see README → «Установка ffmpeg») or set "
                "ENABLE_VOICE=false to silence the feature."
            )

        if config.enable_mcp and config.mcp_servers:
            logger.info(
                "MCP: %d server(s) configured (%s), owner-only=%s",
                len(config.mcp_servers),
                ", ".join(config.mcp_servers),
                config.mcp_owner_only,
            )
        elif config.enable_mcp:
            logger.info(
                "MCP is enabled but no servers are configured "
                "(set MCP_CONFIG_FILE or MCP_SERVERS)"
            )

        logger.info(
            "Starting bot polling (guest access: %s, guest mode: %s)",
            config.allow_guest_access,
            config.enable_guest_mode,
        )

        await _poll_forever()
    finally:
        for task in (cleanup_task, reminder_task, digest_task):
            if task is not None:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        # Stop in-flight reminder deliveries before the engine goes away:
        # they still touch the database (disable/reschedule on failure).
        await cancel_pending_fires()
        await stop_health_server(health_runner)
        await mcp_manager.close()
        await ai_client.close()
        await telegraph.close()
        await bot.session.close()
        await engine.dispose()
        logger.info("Bot stopped")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Shutdown requested")
