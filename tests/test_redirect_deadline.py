"""Редиректы: единый absolute deadline на всю цепочку, не на каждый hop."""

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import patch

from aiohttp import web

from bot.services import documents


class RedirectDeadlineTests(unittest.IsolatedAsyncioTestCase):
    async def test_slow_redirect_chain_stops_at_the_total_deadline(self):
        served: list[str] = []

        async def hop(request: web.Request):
            served.append(request.path)
            await asyncio.sleep(0.12)
            number = int(request.match_info["n"])
            if number >= 5:
                return web.Response(text="done", content_type="text/plain")
            raise web.HTTPFound(f"/r{number + 1}")

        app = web.Application()
        app.router.add_get("/r{n}", hop)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        try:
            port = next(iter(runner.addresses))[1]
            real_resolve = documents._resolve_public_ips

            def fake_resolve(host: str):
                if host == "127.0.0.1":
                    return ["127.0.0.1"]
                return real_resolve(host)

            with (
                patch.object(documents, "_resolve_public_ips", fake_resolve),
                patch.object(documents, "_ALLOWED_PORTS", frozenset({port})),
            ):
                body, _url, _type = await documents._fetch_body(
                    f"http://127.0.0.1:{port}/r0", timeout_seconds=0.2
                )
        finally:
            await runner.cleanup()

        self.assertIsNone(body)
        # Без общего дедлайна все шесть медленных hop успели бы ответить.
        self.assertLess(len(served), 6)