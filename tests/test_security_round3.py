"""Регрессии аудита: внешние изображения, секреты MCP и допуск инструментов."""

from __future__ import annotations

import os
import unittest
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from bot.config import Config, config
from bot.handlers import chat
from bot.services import documents, mcp
from bot.services.access import AccessResult
from bot.services.ai_client import AIClient, ChatResult, ToolCall


@asynccontextmanager
async def no_wait(*args, **kwargs):
    yield


class SecurityTests(unittest.IsolatedAsyncioTestCase):
    def test_model_slot_secrets_cannot_be_expanded_in_mcp_config(self):
        for name in ("MODEL_1_API_KEY", "MODEL_6_API_KEY", "FALLBACK_MODEL_API_KEY"):
            with patch.dict(os.environ, {name: "private-key"}):
                self.assertEqual(mcp._expand_env("${" + name + "}"), "")
                self.assertEqual(mcp._expand_env("${" + name + ":-safe}"), "safe")
                self.assertNotIn(name, mcp._child_environment())
        with patch.dict(os.environ, {"VAULT_PATH": "notes"}):
            self.assertEqual(mcp._expand_env("${VAULT_PATH}"), "notes")

    def test_telegraph_is_opt_in(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(Config().enable_telegraph)

    async def test_image_download_refuses_internal_hosts_before_connection(self):
        client = AIClient(api_key="test-key")
        with patch.object(documents.aiohttp, "ClientSession") as session:
            for url in (
                "http://127.0.0.1/private", "http://169.254.169.254/latest/meta-data/",
                "http://100.100.100.200/", "http://[::1]/", "https://example.test:6379/",
                "https://[broken/",
            ):
                self.assertIsNone(await client.download_image(url), url)
            session.assert_not_called()

    async def test_image_download_streams_and_drops_auth_on_cross_origin_redirect(self):
        client = AIClient(api_key="test-key", base_url="https://api.example.test/v1")
        requested = []

        async def chunks():
            yield b"ab"
            yield b"cd"
            yield b"ef"
            self.fail("Download continued after reaching its size limit")

        def get(url, **kwargs):
            requested.append((url, kwargs))
            response = MagicMock()
            response.url = url
            response.content_length = None
            if url.startswith("https://api.example.test"):
                # Сначала анонимный отказ, затем авторизованный редирект.
                response.status = 302 if kwargs["headers"].get("Authorization") else 403
                response.headers = {"Location": "https://storage.example.test/image"}
            else:
                response.status = 200
                response.headers = {"Content-Type": "image/png"}
                response.content.iter_chunked.return_value = chunks()
            request = MagicMock()
            request.__aenter__ = AsyncMock(return_value=response)
            return request

        session = MagicMock()
        session.get.side_effect = get
        session.__aenter__.return_value = session
        with (
            patch.object(documents, "_resolve_public_ips", return_value=["93.184.216.34"]),
            patch.object(documents.aiohttp, "ClientSession", return_value=session),
            patch("bot.services.ai_client.MAX_DOWNLOAD_BYTES", 4),
        ):
            self.assertIsNone(await client.download_image("https://api.example.test/image"))
        self.assertEqual(len(requested), 3)
        self.assertEqual(requested[1][1]["headers"]["Authorization"], "Bearer test-key")
        self.assertNotIn("Authorization", requested[2][1]["headers"])
        self.assertTrue(all(not kwargs["allow_redirects"] for _, kwargs in requested))

    async def test_image_redirect_to_metadata_is_refused(self):
        response = MagicMock()
        response.status = 302
        response.headers = {"Location": "http://169.254.169.254/latest/meta-data/"}
        request = MagicMock()
        request.__aenter__ = AsyncMock(return_value=response)
        session = MagicMock()
        session.__aenter__.return_value = session
        session.get.return_value = request
        with (
            patch.object(documents, "_resolve_public_ips", side_effect=[
                ["93.184.216.34"], None,
            ]),
            patch.object(documents.aiohttp, "ClientSession", return_value=session),
        ):
            self.assertIsNone(await AIClient(api_key="test-key").download_image(
                "https://storage.example.test/image"
            ))
        session.get.assert_called_once()

    async def test_public_image_is_returned_without_text_mime_restriction(self):
        async def chunks():
            yield b"image"

        response = MagicMock()
        response.status = 200
        response.headers = {"Content-Type": "image/png"}
        response.content_length = 5
        response.url = "https://storage.example.test/image"
        response.content.iter_chunked.return_value = chunks()
        request = MagicMock()
        request.__aenter__ = AsyncMock(return_value=response)
        session = MagicMock()
        session.__aenter__.return_value = session
        session.get.return_value = request
        with (
            patch.object(documents, "_resolve_public_ips", return_value=["93.184.216.34"]),
            patch.object(documents.aiohttp, "ClientSession", return_value=session),
        ):
            self.assertEqual(await AIClient(api_key="test-key").download_image(
                response.url
            ), b"image")

    async def test_unoffered_generation_tools_are_not_executed(self):
        message = SimpleNamespace(
            chat=SimpleNamespace(id=123, type="private"),
            from_user=SimpleNamespace(id=123), answer=AsyncMock(),
        )
        service = SimpleNamespace(
            get_user_settings=AsyncMock(return_value={
                "model": "m", "system_prompt": "sys", "temperature": 0.5, "max_tokens": 100,
            }), get_history=AsyncMock(return_value=[]), add_message=AsyncMock(),
        )
        for name in ("generate_image", "generate_sticker", "send_files"):
            with (
                patch.object(config, "stream_responses", False),
                patch.object(config, "enable_image_generation", False),
                patch.object(config, "enable_sticker_generation", False),
                patch.object(config, "enable_file_replies", False),
                patch.object(chat, "search_configured", return_value=False),
                patch.object(chat, "mcp_available", return_value=False),
                patch.object(chat, "waiting", no_wait),
                patch.object(chat, "_single_shot", AsyncMock(return_value=ChatResult(
                    content="answer", tool_calls=[ToolCall(id="c", name=name, arguments={})],
                ))),
                patch.object(chat, "deliver_answer", AsyncMock()),
                patch.object(chat, "_retry_markup", return_value=None),
                patch.object(chat, "generate_and_send_image", AsyncMock()) as image,
                patch.object(chat, "generate_and_send_sticker", AsyncMock()) as sticker,
                patch.object(chat, "send_reply_files", AsyncMock()) as files,
            ):
                await chat._answer_with_ai(
                    message, service, "hello", AccessResult(allowed=True, role="admin"),
                )
                image.assert_not_awaited()
                sticker.assert_not_awaited()
                files.assert_not_awaited()
