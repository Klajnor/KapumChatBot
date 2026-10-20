"""Offline regression checks: no Telegram, AI or database connections."""
from __future__ import annotations

import os
import asyncio
import unittest
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

# Keep local credentials out of the tests and never contact configured services.
os.environ["BOT_TOKEN"] = "123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghi"
os.environ["AI_API_KEY"] = "test-key"
os.environ["DATABASE_URL"] = "postgresql+asyncpg://test:test@localhost/test"

from bot.config import Config, config, _env_float
from bot.database.repository import UserRepository
from bot.database.session import _run_migrations
from bot.handlers import chat, guest
from bot.services.ai_client import AIClient, ImageResult, _request_log_payload
from bot.services.telegram_utils import _retry_delay
from bot.services.user_service import UserService


@asynccontextmanager
async def no_wait(*args, **kwargs):
    yield


class RegressionTests(unittest.IsolatedAsyncioTestCase):
    def test_api_log_includes_text_but_not_media_or_credentials(self):
        import json
        payload = {
            "model": "test-model", "max_tokens": 16384, "api_key": "secret-key",
            "messages": [
                {"role": "user", "content": "Полный текст запроса"},
                {"role": "user", "content": [
                    {"type": "text", "text": "Описание картинки"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}},
                    {"type": "input_audio", "input_audio": {"data": "data:audio/ogg;base64,aGVsbG8="}},
                    {"type": "image_url", "image_url": {"url": "https://example.test/image?signature=secret-signature"}},
                ]},
            ],
        }
        summary = _request_log_payload(payload)
        logged = json.dumps(summary, ensure_ascii=False)
        self.assertIn("Полный текст запроса", logged)
        self.assertIn("Описание картинки", logged)
        self.assertNotIn("aGVsbG8=", logged)
        self.assertNotIn("secret-key", logged)
        self.assertNotIn("secret-signature", logged)
        parts = summary["messages"][1]["content"]
        self.assertEqual(parts[1], {"type": "image", "source": "inline", "mime": "image/png", "bytes": 5})
        self.assertEqual(parts[2]["type"], "audio")
        self.assertEqual(parts[2]["bytes"], 5)

    async def test_actual_api_request_is_logged(self):
        client = AIClient(api_key="secret-key")
        response = MagicMock()
        response.status = 200
        response.text = AsyncMock(return_value='{"choices": []}')
        context = MagicMock()
        context.__aenter__ = AsyncMock(return_value=response)
        session = MagicMock()
        session.post.return_value = context
        with (
            patch.object(client, "_get_session", AsyncMock(return_value=session)),
            self.assertLogs("bot.services.ai_client", level="INFO") as logs,
        ):
            status, _, _ = await client._post_chat({
                "model": "test", "messages": [{"role": "user", "content": "test prompt"}]
            })
        self.assertEqual(status, 200)
        logged = "\n".join(logs.output)
        self.assertIn("test prompt", logged)
        self.assertIn("attempt=1/3", logged)
        self.assertIn("status=200", logged)
        self.assertNotIn("secret-key", logged)

    async def test_existing_user_uses_configured_token_limit(self):
        service = UserService(MagicMock())
        saved = SimpleNamespace(
            selected_model="custom-model", system_prompt="Custom prompt",
            temperature=0.4, max_tokens=2048,
        )
        with (
            patch.object(service.user_repo, "get_user_settings", AsyncMock(return_value=saved)),
            patch.object(config, "default_max_tokens", 16384),
        ):
            settings = await service.get_user_settings(1)
        self.assertEqual(settings["max_tokens"], 16384)
        self.assertEqual(settings["model"], "custom-model")
        self.assertEqual(settings["system_prompt"], "Custom prompt")
        self.assertEqual(settings["temperature"], 0.4)

    async def test_guest_timeout_returns_explanation_and_cancels_work(self):
        cancelled = asyncio.Event()
        async def slow_reply(*args):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        message = SimpleNamespace(
            guest_query_id="test", from_user=SimpleNamespace(id=1),
            chat=SimpleNamespace(id=42),
        )
        session = SimpleNamespace(close=AsyncMock())
        with (
            patch.object(config, "enable_guest_mode", True),
            patch.object(config, "guest_timeout", 0.01),
            patch.object(guest, "async_session_factory", return_value=session),
            patch.object(guest, "UserService"),
            patch.object(guest, "_build_reply", side_effect=slow_reply),
            patch.object(guest, "safe_answer_guest", AsyncMock(return_value=True)) as answer,
            patch.object(guest.logger, "exception") as error,
        ):
            await guest.handle_guest_message(message)
        self.assertTrue(cancelled.is_set())
        session.close.assert_awaited_once()
        answer.assert_awaited_once()
        self.assertIn("/image", answer.call_args.args[1].input_message_content.message_text)
        error.assert_not_called()

    async def test_guest_external_cancellation_is_not_swallowed(self):
        message = SimpleNamespace(
            guest_query_id="test", from_user=SimpleNamespace(id=1),
            chat=SimpleNamespace(id=42),
        )
        session = SimpleNamespace(close=AsyncMock())
        with (
            patch.object(config, "enable_guest_mode", True),
            patch.object(guest, "async_session_factory", return_value=session),
            patch.object(guest, "UserService"),
            patch.object(guest, "_build_reply", AsyncMock(side_effect=asyncio.CancelledError)),
            patch.object(guest, "safe_answer_guest", AsyncMock()) as answer,
        ):
            with self.assertRaises(asyncio.CancelledError):
                await guest.handle_guest_message(message)
        session.close.assert_awaited_once()
        answer.assert_not_awaited()

    async def test_failed_photo_is_not_saved_as_success(self):
        message = SimpleNamespace(
            bot=SimpleNamespace(send_chat_action=AsyncMock()),
            chat=SimpleNamespace(id=42),
        )
        service = SimpleNamespace(add_message=AsyncMock())
        image = ImageResult(data=b"png", url="https://example.test/image", elapsed=1)
        with (
            patch.object(chat, "waiting", no_wait),
            patch.object(chat.ai_client, "generate_image_result", AsyncMock(return_value=image)),
            patch.object(chat, "_safe_answer_photo", AsyncMock(return_value=False)),
        ):
            self.assertFalse(await chat.generate_and_send_image(
                message, "fox", service=service, user_id=1
            ))
        service.add_message.assert_not_awaited()

    async def test_url_generation_does_not_download(self):
        client = AIClient(api_key="test-key")
        result = ImageResult(url="https://example.test/image")
        with patch.object(client, "generate_image_result", AsyncMock(return_value=result)) as generate:
            self.assertEqual(await client.generate_image_url("fox"), result.url)
        self.assertFalse(generate.call_args.kwargs["download"])

    async def test_external_image_host_never_receives_api_key(self):
        client = AIClient(api_key="test-key", base_url="https://api.example.test/compatible-mode/v1")
        response = MagicMock()
        response.status = 403
        response.read = AsyncMock(return_value=b"denied")
        request = MagicMock()
        request.__aenter__ = AsyncMock(return_value=response)
        session = MagicMock()
        session.get.return_value = request
        with patch.object(client, "_get_download_session", AsyncMock(return_value=session)):
            self.assertIsNone(await client.download_image("https://storage.example.test/image"))
        session.get.assert_called_once()
        self.assertIsNone(session.get.call_args.kwargs["headers"])

    async def test_large_audio_is_rejected_before_request(self):
        client = AIClient(api_key="test-key")
        with (
            patch.object(config, "voice_max_bytes", 3),
            patch.object(client, "_post_chat", AsyncMock()) as request,
        ):
            self.assertIsNone(await client.transcribe_audio(b"1234"))
        request.assert_not_awaited()

    async def test_long_voice_rejected_before_access_is_consumed(self):
        message = SimpleNamespace(from_user=SimpleNamespace(id=1), voice=SimpleNamespace(duration=301))
        with (
            patch.object(config, "enable_voice", True),
            patch.object(chat, "_is_addressed", return_value=True),
            patch.object(chat, "safe_answer", AsyncMock()) as answer,
            patch.object(chat, "_open_registered", AsyncMock()) as register,
        ):
            await chat.handle_voice(message)
        answer.assert_awaited_once()
        register.assert_not_awaited()

    async def test_settings_register_new_user_first(self):
        repo = UserRepository(MagicMock())
        settings = SimpleNamespace(selected_model="test")
        with (
            patch.object(repo, "get_user_settings", AsyncMock(side_effect=[None, settings])),
            patch.object(repo, "get_or_create_user", AsyncMock()) as register,
        ):
            self.assertIs(await repo._ensure_settings(1), settings)
        register.assert_awaited_once_with(1)

    async def test_new_settings_use_configured_defaults(self):
        session = MagicMock()
        session.get = AsyncMock(return_value=None)
        session.flush = AsyncMock()
        repo = UserRepository(session)
        with (
            patch.object(repo, "get_user_settings", AsyncMock(return_value=None)),
            patch.object(config, "default_temperature", 0.2),
            patch.object(config, "default_max_tokens", 321),
        ):
            await repo.get_or_create_user(1)
        settings = session.add.call_args_list[-1].args[0]
        self.assertEqual(settings.temperature, 0.2)
        self.assertEqual(settings.max_tokens, 321)

    async def test_migrations_isolate_failed_statement(self):
        conn = MagicMock()
        conn.begin_nested.side_effect = no_wait
        conn.execute = AsyncMock(side_effect=[RuntimeError("migration failed")] + [None] * 20)
        await _run_migrations(conn)
        self.assertGreater(conn.execute.await_count, 1)
        self.assertEqual(conn.begin_nested.call_count, conn.execute.await_count)

    async def test_startup_failure_closes_resources(self):
        from bot import main
        with (
            patch.object(config, "validate"),
            patch.object(main, "init_db", AsyncMock(side_effect=RuntimeError("DB unavailable"))),
            patch.object(main.ai_client, "close", AsyncMock()) as ai_close,
            patch.object(main.bot.session, "close", AsyncMock()) as bot_close,
            patch.object(main, "engine", SimpleNamespace(dispose=AsyncMock())) as db,
        ):
            with self.assertRaises(RuntimeError):
                await main.main()
        ai_close.assert_awaited_once()
        bot_close.assert_awaited_once()
        db.dispose.assert_awaited_once()

    def test_guest_long_reply_has_complete_markup(self):
        result = guest._to_result(guest.GuestReply(text="**" + "x" * 6000 + "**"), mode="html")
        body = result.input_message_content.message_text
        self.assertLessEqual(len(body), guest.MAX_TEXT_LENGTH)
        self.assertEqual(body.count("<b>"), body.count("</b>"))

    def test_guest_photo_caption_uses_plain_text(self):
        result = guest._to_result(guest.GuestReply(photo_url="https://example.test/image", caption="x < y"))
        content = result.input_message_content
        self.assertIsNone(content.parse_mode)
        self.assertIn("x < y", content.message_text)
        self.assertIn("https://example.test/image", content.message_text)
        self.assertTrue(content.link_preview_options.is_disabled)

    async def test_failed_guest_answer_is_not_retried(self):
        message = SimpleNamespace(
            guest_query_id="test", from_user=SimpleNamespace(id=1),
            chat=SimpleNamespace(id=42),
        )
        session = SimpleNamespace(close=AsyncMock())
        with (
            patch.object(config, "enable_guest_mode", True),
            patch.object(guest, "async_session_factory", return_value=session),
            patch.object(guest, "UserService"),
            patch.object(guest, "_build_reply", AsyncMock(return_value=guest.GuestReply(text="hello"))),
            patch.object(guest, "safe_answer_guest", AsyncMock(return_value=False)) as answer,
        ):
            await guest.handle_guest_message(message)
        answer.assert_awaited_once()

    def test_guest_image_link_is_complete_when_caption_is_long(self):
        url = "https://example.test/image?signature=" + "x" * 3000
        result = guest._to_result(guest.GuestReply(photo_url=url, caption="c" * 2000))
        text = result.input_message_content.message_text
        self.assertTrue(text.endswith(url))
        self.assertLessEqual(len(text), guest.MAX_TEXT_LENGTH)

    def test_asr_limit_accounts_for_base64(self):
        with patch.dict(os.environ, {"VOICE_MAX_BYTES": "10485760"}):
            value = Config().voice_max_bytes
        self.assertLessEqual(4 * ((value + 2) // 3), 10_000_000)

    def test_retry_waits_full_server_delay(self):
        self.assertGreaterEqual(_retry_delay(SimpleNamespace(retry_after=90)), 90)

    def test_nonfinite_float_uses_default(self):
        with patch.dict(os.environ, {"TEST_FLOAT": "nan"}):
            self.assertEqual(_env_float("TEST_FLOAT", 0.7), 0.7)


if __name__ == "__main__":
    unittest.main()
