"""Optional HTTP health endpoint for infrastructure monitoring.

A Telegram bot has no HTTP surface of its own, so an operator running it under
systemd, Docker or an external uptime monitor has nothing to probe. When
``HEALTH_PORT`` is non-zero a tiny aiohttp server answers ``GET /health`` with
HTTP 200 and a small JSON payload.

Disabled by default (``HEALTH_PORT=0``). The host defaults to ``127.0.0.1`` so
the endpoint is not exposed publicly unless explicitly configured.
"""

from __future__ import annotations

import logging
import time

from aiohttp import web

from bot.config import config

logger = logging.getLogger(__name__)

_STARTED = time.monotonic()


def uptime_seconds() -> float:
    """Seconds since the bot process imported this module."""
    return max(0.0, time.monotonic() - _STARTED)


async def _health(_request: web.Request) -> web.Response:
    """Liveness probe payload.

    Deliberately minimal: the endpoint is unauthenticated, so it reports only
    liveness and uptime — no configuration or version details.
    """
    return web.json_response(
        {
            "status": "ok",
            "uptime_seconds": round(uptime_seconds(), 1),
        }
    )


async def start_health_server() -> web.AppRunner | None:
    """Start the health server when configured; return its runner or ``None``."""
    if config.health_port <= 0:
        return None

    app = web.Application()
    app.router.add_get("/health", _health)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, config.health_host, config.health_port)
    try:
        await site.start()
    except Exception as exc:  # noqa: BLE001 - a broken health endpoint must not
        # stop the bot. aiohttp/asyncio raise OSError for a busy port and
        # OverflowError/ValueError for an out-of-range one, so catch broadly.
        logger.error(
            "Health endpoint could not bind %s:%s (%s); continuing without it",
            config.health_host,
            config.health_port,
            exc,
        )
        await runner.cleanup()
        return None
    logger.info(
        "Health endpoint listening on http://%s:%s/health",
        config.health_host,
        config.health_port,
    )
    return runner


async def stop_health_server(runner: web.AppRunner | None) -> None:
    """Shut the health server down; safe to call with ``None``."""
    if runner is None:
        return
    await runner.cleanup()
    logger.info("Health endpoint stopped")