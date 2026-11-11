"""Диагностика неудачных AI-запросов без сети, Telegram и рабочей БД."""

from __future__ import annotations

import asyncio
import json
import unittest
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp

from bot.config import config
from bot.handlers import chat
from bot.services.access import AccessResult
from bot.services.ai_client import IMAGE_TOOL, AIClient, AIStreamError, ChatResult
from bot.services.ai_errors import (
    AIRequestError,
    format_request_errors,
    provider_error,
    request_error,
)


def response(status: int, body: str) -> SimpleNamespace:
    return SimpleNamespace(
        status=status, reason="test", headers={}, charset="utf-8",
        content=SimpleNamespace(read=AsyncMock(side_effect=[body.encode(), b""])),
    )


def http_context(status: int, body: str) -> MagicMock:
    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=response(status, body))
    return context


def error_body(code: str, message: str) -> str:
    return json.dumps({"error": {"code": code, "message": message}})


class ErrorFormattingTests(unittest.TestCase):
    def test_provider_formats_and_missing_descriptions(self):
        cases = [
            (429, error_body("insufficient_quota", "Quota exhausted"), "429 / insufficient_quota", "Quota exhausted"),
            (400, '{"code":"InvalidParameter","message":"Bad model"}', "400 / InvalidParameter", "Bad model"),
            (400, '{"error":{"type":"invalid_request_error","message":"Bad tools"}}', "400 / invalid_request_error", "Bad tools"),
            (503, '{"error":"Busy"}', "503", "Busy"),
            (422, '{"detail":"Invalid input"}', "422", "Invalid input"),
            (502, "Bad gateway upstream", "502", "Bad gateway upstream"),
            (503, "<html>private proxy configuration</html>", "503", "Service Unavailable"),
            (401, '{"error":{"code":"InvalidKey"}}', "401 / InvalidKey", "Unauthorized"),
            (599, "", "599", ""),
            (500, '{"request":{"messages":["private"]}}', "500", "Internal Server Error"),
        ]
        for status, body, code, message in cases:
            with self.subTest(body=body):
                self.assertEqual(provider_error("m", status, body), AIRequestError("m", code, message))

    def test_redacts_before_truncating_and_removes_url_credentials(self):
        secret = "provider-key-" + "a" * 300
        message = (
            f"Invalid key {secret}; Bearer other-private-key "
            'api_key="unconfigured-key" password=pwd '
            "https://user:proxy-password@proxy.test/path?token=url-token "
            "sk-other-secret-key 123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghi"
        )
        error = request_error("m", "401", message, secrets=[secret])
        rendered = format_request_errors([error])
        for value in (secret[:30], "other-private-key", "unconfigured-key", "pwd", "proxy-password", "url-token", "sk-other", "ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
            self.assertNotIn(value, rendered)
        self.assertIn("[redacted]", rendered)
        self.assertLessEqual(len(error.message), 180)
        self.assertNotIn(";", rendered)

    def test_control_characters_and_long_fields_are_bounded(self):
        error = request_error("m" * 200, "c" * 200, "First;\nsecond\u202ereversed " + "x" * 300)
        self.assertEqual(len(error.model), 96)
        self.assertEqual(len(error.code), 80)
        self.assertEqual(len(error.message), 180)
        self.assertNotIn(";", error.message)
        self.assertNotIn("\n", error.message)
        self.assertNotIn("\u202e", error.message)

    def test_each_generic_secret_shape_is_redacted_without_truncation(self):
        cases = [
            "Bearer unknown-private-key", 'api_key="unknown-private-key"',
            "password=unknown-private-key", "token: unknown-private-key",
            "https://user:unknown-private-key@proxy.test/path?token=secret",
            "sk-unknown-private-key", "123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghi",
        ]
        for text in cases:
            with self.subTest(text=text):
                error = request_error("m", "401", text)
                self.assertNotIn("unknown-private-key", error.message)
                self.assertNotIn("ABCDEFGHIJKLMNOPQRSTUVWXYZ", error.message)
                self.assertNotIn("proxy.test", error.message)

    def test_semicolon_separator_order_and_repeat_count(self):
        first = request_error("a", "429", "Quota exhausted")
        second = request_error("b", "503", "Busy")
        changed = request_error("a", "429", "Rate limit")
        self.assertEqual(
            format_request_errors([first, second, first, changed]),
            "a — 429 (Quota exhausted) ×2; b — 503 (Busy); a — 429 (Rate limit)",
        )
        self.assertEqual(format_request_errors([]), "")
        self.assertEqual(format_request_errors([AIRequestError("m", "599")]), "m — 599")


class ClientErrorTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.client = AIClient(api_key="client-secret")
        self.session = MagicMock()
        patches = [
            patch.object(config, "ai_models", [
                {"id": "a", "model": "provider-a", "name": "A", "vision": True, "api_key": "slot-secret"},
                {"id": "b", "model": "provider-b", "name": "B", "vision": True},
            ]),
            patch.object(config, "ai_fallback_models", []),
            patch.object(config, "default_model", "a"),
            patch.object(self.client, "_get_session", AsyncMock(return_value=self.session)),
            patch("bot.services.ai_client.asyncio.sleep", AsyncMock()),
        ]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)

    async def test_collects_each_http_retry_and_next_model(self):
        self.session.post.side_effect = [
            http_context(429, error_body("Quota", "Quota exhausted")),
            http_context(503, error_body("Unavailable", "Busy")),
            http_context(503, error_body("Unavailable", "Busy")),
        ]
        result = await self.client.chat_completion_result([], model="a")
        self.assertIsNone(result.content)
        self.assertEqual([error.model for error in result.errors], ["provider-a", "provider-b", "provider-b"])
        self.assertEqual([error.code for error in result.errors], ["429 / Quota", "503 / Unavailable", "503 / Unavailable"])
        self.assertIn("provider-b — 503 / Unavailable (Busy) ×2", format_request_errors(result.errors))
        self.assertEqual(self.session.post.call_count, 3)

    async def test_fatal_error_stops_chain_and_redacts_all_provider_keys(self):
        self.session.post.return_value = http_context(
            401, error_body("InvalidKey", "Invalid client-secret / slot-secret"),
        )
        result = await self.client.chat_completion_result([], model="a")
        self.assertEqual(result.errors, [AIRequestError("provider-a", "401 / InvalidKey", "Invalid [redacted] / [redacted]")])
        self.assertEqual(self.session.post.call_count, 1)

    async def test_redacts_fallback_key_and_telegram_token(self):
        self.session.post.return_value = http_context(
            401, error_body("InvalidKey", "Invalid fallback-secret and bot-secret"),
        )
        with (
            patch.object(config, "ai_fallback_models", [{"id": "rescue", "api_key": "fallback-secret"}]),
            patch.object(config, "bot_token", "bot-secret"),
        ):
            result = await self.client.chat_completion_result([], model="a")
        self.assertEqual(result.errors[0].message, "Invalid [redacted] and [redacted]")

    async def test_invalid_and_empty_success_responses_have_local_codes(self):
        cases = [
            ("not JSON", "INVALID_RESPONSE"),
            ("[]", "INVALID_RESPONSE"),
            ('{"choices":[]}', "EMPTY_RESPONSE"),
            (error_body("ProviderFailure", "Failed"), "200 / ProviderFailure"),
        ]
        with patch.object(self.client, "model_chain", return_value=["a"]):
            for body, code in cases:
                with self.subTest(body=body):
                    self.session.post.return_value = http_context(200, body)
                    result = await self.client.chat_completion_result([], model="a")
                    self.assertIsNone(result.content)
                    self.assertEqual(len(result.errors), 1)
                    self.assertEqual(result.errors[0].code, code)

    async def test_tools_and_thinking_rejection_retries_are_kept(self):
        self.session.post.side_effect = [
            http_context(400, error_body("Tools", "Unsupported tools")),
            http_context(422, error_body("Thinking", "Unsupported thinking")),
            http_context(429, error_body("Quota", "Quota exhausted")),
        ]
        with (
            patch.object(self.client, "model_chain", return_value=["a"]),
            patch("bot.services.ai_client.thinking.payload_fields", return_value={"enable_thinking": True}),
        ):
            result = await self.client.chat_with_tools([], model="a", tools=[IMAGE_TOOL], public=True)
        self.assertEqual([error.code for error in result.errors], ["400 / Tools", "422 / Thinking", "429 / Quota"])

    async def test_vision_requests_also_carry_errors(self):
        self.session.post.side_effect = [
            http_context(429, error_body("Quota", "Quota exhausted")),
            http_context(404, error_body("ModelMissing", "Not found")),
        ]
        result = await self.client.chat_with_image_result("describe", b"image", [], model="a")
        self.assertEqual([error.model for error in result.errors], ["provider-a", "provider-b"])

    async def test_timeout_and_network_retries_have_distinct_codes(self):
        for exception, code in [(TimeoutError(), "TIMEOUT"), (aiohttp.ClientConnectionError("private URL"), "NETWORK_ERROR")]:
            with self.subTest(code=code), patch.object(self.client, "model_chain", return_value=["a"]):
                context = MagicMock()
                context.__aenter__ = AsyncMock(side_effect=exception)
                self.session.post.return_value = context
                result = await self.client.chat_completion_result([], model="a")
                self.assertEqual([error.code for error in result.errors], [code, code])
                self.assertNotIn("private URL", format_request_errors(result.errors))

    async def test_stream_preserves_http_bodies_from_entire_chain(self):
        self.session.post.side_effect = [
            http_context(429, error_body("Quota", "Quota exhausted")),
            http_context(503, error_body("Busy", "Try later")),
        ]
        with self.assertRaises(AIStreamError) as raised:
            await self.client.stream_chat_completion([], model="a")
        self.assertEqual(raised.exception.status, 503)
        self.assertEqual(raised.exception.errors, [
            AIRequestError("provider-a", "429 / Quota", "Quota exhausted"),
            AIRequestError("provider-b", "503 / Busy", "Try later"),
        ])

    async def test_stream_timeout_and_network_error_codes(self):
        for exception, code in [(TimeoutError(), "TIMEOUT"), (aiohttp.ClientConnectionError(), "NETWORK_ERROR")]:
            with self.subTest(code=code), patch.object(self.client, "model_chain", return_value=["a"]):
                context = MagicMock()
                context.__aenter__ = AsyncMock(side_effect=exception)
                self.session.post.return_value = context
                with self.assertRaises(AIStreamError) as raised:
                    await self.client.stream_chat_completion([], model="a")
                self.assertEqual([error.code for error in raised.exception.errors], [code])

    async def test_stream_thinking_retry_preserves_first_rejection(self):
        self.session.post.side_effect = [
            http_context(400, error_body("Thinking", "Unsupported thinking")),
            http_context(503, error_body("Busy", "Try later")),
        ]
        with (
            patch.object(self.client, "model_chain", return_value=["a"]),
            patch("bot.services.ai_client.thinking.payload_fields", return_value={"enable_thinking": True}),
            self.assertRaises(AIStreamError) as raised,
        ):
            await self.client.stream_chat_completion([], model="a", public=True)
        self.assertEqual([error.code for error in raised.exception.errors], ["400 / Thinking", "503 / Busy"])

    async def test_successful_stream_thinking_retry_keeps_diagnostic(self):
        with (
            patch.object(self.client, "model_chain", return_value=["a"]),
            patch("bot.services.ai_client.thinking.payload_fields", return_value={"enable_thinking": True}),
            patch.object(self.client, "_stream_attempt", AsyncMock(side_effect=[
                AIStreamError("Unsupported thinking", status=422), ChatResult(content="answer"),
            ])),
        ):
            result = await self.client.stream_chat_completion([], model="a", public=True)
        self.assertEqual(result.content, "answer")
        self.assertEqual(result.errors, [AIRequestError("provider-a", "422", "Unsupported thinking")])

    async def test_stream_fatal_error_does_not_invent_uncalled_models(self):
        self.session.post.return_value = http_context(401, error_body("InvalidKey", "Invalid key"))
        with self.assertRaises(AIStreamError) as raised:
            await self.client.stream_chat_completion([], model="a")
        self.assertEqual([error.model for error in raised.exception.errors], ["provider-a"])
        self.assertEqual(self.session.post.call_count, 1)

    async def test_sse_error_event_and_empty_answer(self):
        for line, code in [
            ("data: " + error_body("Quota", "Quota exhausted"), "200 / Quota"),
            ("data: [DONE]", "EMPTY_RESPONSE"),
        ]:
            async def lines(*args, event_line=line):
                yield event_line

            with (
                self.subTest(code=code),
                patch.object(self.client, "model_chain", return_value=["a"]),
                patch.object(self.client, "_stream_lines", lines),
                self.assertRaises(AIStreamError) as raised,
            ):
                await self.client.stream_chat_completion([], model="a")
            self.assertEqual([error.code for error in raised.exception.errors], [code])

    async def test_stream_protocol_error_gets_local_code(self):
        async def lines(*args):
            yield "data: invalid-json"
            for _ in range(10):
                yield "data: invalid-json"

        with (
            patch.object(self.client, "model_chain", return_value=["a"]),
            patch.object(self.client, "_stream_lines", lines),
            self.assertRaises(AIStreamError) as raised,
        ):
            await self.client.stream_chat_completion([], model="a")
        self.assertEqual(raised.exception.errors, [AIRequestError(
            "provider-a", "STREAM_ERROR", "too many malformed SSE chunks",
        )])

    async def test_errors_do_not_leak_between_concurrent_calls(self):
        ready = asyncio.Event()
        started = 0

        @asynccontextmanager
        async def post(*args, **kwargs):
            nonlocal started
            started += 1
            if started == 2:
                ready.set()
            await ready.wait()
            sent = kwargs["json"]
            messages = sent.get("messages") or (sent.get("input") or {}).get("messages") or []
            question = messages[-1]["content"]
            yield response(429, error_body("Quota", question))

        self.session.post.side_effect = post
        with patch.object(self.client, "model_chain", return_value=["a"]):
            first, second = await asyncio.wait_for(asyncio.gather(
                self.client.chat_completion_result([{"role": "user", "content": "first"}]),
                self.client.chat_completion_result([{"role": "user", "content": "second"}]),
            ), timeout=2)
        self.assertEqual([error.message for error in first.errors], ["first"])
        self.assertEqual([error.message for error in second.errors], ["second"])
        self.assertIsNot(first.errors, second.errors)

    async def test_successful_next_question_does_not_reuse_previous_errors(self):
        self.session.post.side_effect = [
            http_context(429, error_body("Quota", "Quota exhausted")),
            http_context(200, '{"choices":[{"message":{"content":"ok"}}]}'),
        ]
        with patch.object(self.client, "model_chain", return_value=["a"]):
            failed = await self.client.chat_completion_result([])
            answered = await self.client.chat_completion_result([])
        self.assertEqual(len(failed.errors), 1)
        self.assertEqual(answered.content, "ok")
        self.assertEqual(answered.errors, [])


@asynccontextmanager
async def no_wait(*args, **kwargs):
    yield SimpleNamespace()


class ChatErrorDeliveryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.message = SimpleNamespace(
            from_user=SimpleNamespace(id=10), chat=SimpleNamespace(id=10, type="private"),
        )
        self.service = SimpleNamespace(
            get_user_settings=AsyncMock(return_value={
                "model": "a", "system_prompt": "sys", "temperature": 0.5, "max_tokens": 100,
            }),
            get_history=AsyncMock(return_value=[]), add_message=AsyncMock(),
        )
        self.single = AsyncMock(return_value=ChatResult())
        self.send = AsyncMock(return_value=True)
        self.refund = AsyncMock()
        self.deliver = AsyncMock()
        patches = [
            patch.object(chat, "waiting", no_wait),
            patch.object(chat, "_single_shot", self.single),
            patch.object(chat, "safe_answer", self.send),
            patch.object(chat, "refund_access", self.refund),
            patch.object(chat, "deliver_answer", self.deliver),
            patch.object(chat, "search_configured", return_value=False),
            patch.object(chat, "mcp_available", return_value=False),
            patch.object(config, "stream_responses", False),
            patch.object(config, "enable_image_generation", False),
            patch.object(config, "enable_sticker_generation", False),
            patch.object(config, "enable_file_replies", False),
        ]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)

    async def answer(self):
        await chat._answer_with_ai(self.message, self.service, "question", AccessResult(True, "user"))

    async def test_combines_stream_and_regular_errors_with_semicolons(self):
        streaming = AIRequestError("a", "429 / Quota", "Quota exhausted")
        regular = AIRequestError("b", "503", "<b>Busy</b> & retry")
        self.single.return_value = ChatResult(errors=[streaming, regular, regular])
        printer = SimpleNamespace(on_delta=AsyncMock(), discard=AsyncMock())
        with (
            patch.object(config, "stream_responses", True),
            patch.object(chat, "_StreamPrinter", return_value=printer),
            patch.object(chat.ai_client, "stream_chat_completion", AsyncMock(
                side_effect=AIStreamError("failed", errors=[streaming]),
            )),
        ):
            await self.answer()
        self.refund.assert_awaited_once()
        self.assertEqual(self.send.await_count, 1)
        call = self.send.await_args
        self.assertIsNone(call.kwargs["parse_mode"])
        self.assertIn(
            "a — 429 / Quota (Quota exhausted) ×2; b — 503 (<b>Busy</b> & retry) ×2",
            call.args[1],
        )
        self.assertNotIn("Qwen account", call.args[1])
        printer.discard.assert_awaited_once()
        self.assertEqual(self.service.add_message.await_count, 1)
        self.assertEqual(self.service.add_message.await_args.kwargs["role"], "user")

    async def test_long_diagnostics_are_split_without_losing_models(self):
        errors = [request_error(f"model-{index}", "503", "Busy " + "😀" * 180) for index in range(40)]
        self.single.return_value = ChatResult(errors=errors)
        await self.answer()
        texts = [call.args[1] for call in self.send.await_args_list]
        self.assertGreater(len(texts), 1)
        self.assertTrue(all(len(text.encode("utf-16-le")) // 2 <= chat.MAX_MESSAGE_LENGTH for text in texts))
        self.assertTrue(all(call.kwargs["parse_mode"] is None for call in self.send.await_args_list))
        self.assertIn(format_request_errors(errors), "".join(texts))
        self.refund.assert_awaited_once()

    async def test_success_does_not_send_diagnostics_or_refund(self):
        self.single.return_value = ChatResult(
            content="answer", errors=[AIRequestError("a", "429", "Quota exhausted")],
        )
        with patch.object(chat, "guest_hint", return_value=None):
            await self.answer()
        self.deliver.assert_awaited_once()
        self.send.assert_not_awaited()
        self.refund.assert_not_awaited()

    async def test_empty_trace_keeps_generic_error_and_refunds(self):
        await self.answer()
        self.assertIn("Error communicating with the AI", self.send.await_args.args[1])
        self.assertNotIn("Model errors:", self.send.await_args.args[1])
        self.refund.assert_awaited_once()
