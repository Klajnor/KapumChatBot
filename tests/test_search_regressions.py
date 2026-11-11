"""Регрессии поиска: JSON вместо tool_calls, дата и недоверенная история."""

from __future__ import annotations

import datetime
import json
import unittest
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from bot.config import config
from bot.handlers import chat
from bot.services import reminders
from bot.services.access import AccessResult
from bot.services.ai_client import IMAGE_TOOL, AIClient, ChatResult
from bot.services.search import SEARCH_SYSTEM_HINT, SEARCH_TOOL, isolate_search_history
from bot.services.untrusted import wrap_untrusted

QUESTION = "найди в интернете и проанализируй вчерашние актуальные статьи про ИИ"
LEAK = 'Запускаю поиск.\n{"name": "web_search", "arguments": {"query": "wrong date"}}'
NATIVE = {"id": "real", "type": "function", "function": {
    "name": "search_web", "arguments": '{"query": "AI news November 8 2026"}',
}}


def _response(content=None, calls=None):
    return {"choices": [{"message": {"content": content, "tool_calls": calls or []}}]}


@asynccontextmanager
async def _no_wait(*args, **kwargs):
    yield SimpleNamespace()


class TextSearchRetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_json_is_retried_natively_not_executed_or_returned(self):
        client = AIClient(api_key="test")
        messages = [{"role": "user", "content": QUESTION}]
        post = AsyncMock(side_effect=[
            (200, _response(LEAK), 0), (200, _response(calls=[NATIVE]), 0),
        ])
        with (
            patch.object(client, "model_chain", return_value=["m"]),
            patch.object(client, "_post_chat", post),
        ):
            result = await client.chat_with_tools(messages, tools=[SEARCH_TOOL])
        self.assertIsNone(result.content)
        self.assertEqual(result.tool_calls[0].arguments["query"], "AI news November 8 2026")
        self.assertEqual(result.tool_calls[0].id, "real")
        self.assertEqual(post.await_count, 2)
        retry = post.await_args_list[1].args[0]
        self.assertIn("tool_calls", retry["messages"][0]["content"])
        self.assertEqual(retry["tool_choice"]["function"]["name"], "search_web")
        self.assertNotIn("wrong date", json.dumps(retry, ensure_ascii=False))
        self.assertEqual(messages, [{"role": "user", "content": QUESTION}])

    async def test_repeated_json_uses_fallback_without_returning_the_preamble(self):
        client = AIClient(api_key="test")
        post = AsyncMock(side_effect=[
            (200, _response(LEAK), 0), (200, _response(LEAK), 0),
            (200, _response(calls=[NATIVE]), 0),
        ])
        with (
            patch.object(client, "model_chain", return_value=["m1", "m2"]),
            patch.object(client, "_post_chat", post),
        ):
            result = await client.chat_with_tools([], tools=[SEARCH_TOOL])
        self.assertEqual(result.model, "m2")
        self.assertEqual(post.await_count, 3)
        self.assertTrue(result.tool_calls)
        self.assertIsNone(result.content)

    async def test_disabled_search_does_not_retry_or_recover_json(self):
        """Без предложенных инструментов JSON-имитация не съедает весь ответ."""
        client = AIClient(api_key="test")
        with (
            patch.object(client, "model_chain", return_value=["m"]),
            patch.object(client, "_post_chat", AsyncMock(return_value=(200, _response(LEAK), 0))) as post,
        ):
            result = await client.chat_completion_result([])
        self.assertEqual(result.content, "Запускаю поиск.")
        self.assertFalse(result.tool_calls)
        post.assert_awaited_once()

    async def test_code_example_is_not_retried(self):
        client = AIClient(api_key="test")
        content = f"Пример:\n```json\n{LEAK.splitlines()[1]}\n```"
        with (
            patch.object(client, "model_chain", return_value=["m"]),
            patch.object(client, "_post_chat", AsyncMock(return_value=(200, _response(content), 0))) as post,
        ):
            result = await client.chat_with_tools([], tools=[SEARCH_TOOL])
        self.assertEqual(result.content, content)
        post.assert_awaited_once()

    async def test_unoffered_search_is_not_retried_even_when_other_tools_exist(self):
        client = AIClient(api_key="test")
        with (
            patch.object(client, "model_chain", return_value=["m"]),
            patch.object(client, "_post_chat", AsyncMock(return_value=(200, _response(LEAK), 0))) as post,
        ):
            result = await client.chat_with_tools([], tools=[IMAGE_TOOL])
        self.assertIsNone(result.content)
        self.assertFalse(result.tool_calls)
        post.assert_awaited_once()

    async def test_native_calls_are_not_retried_or_duplicated_by_json(self):
        client = AIClient(api_key="test")
        with (
            patch.object(client, "model_chain", return_value=["m"]),
            patch.object(client, "_post_chat", AsyncMock(return_value=(200, _response(LEAK, [NATIVE]), 0))) as post,
        ):
            result = await client.chat_with_tools([], tools=[SEARCH_TOOL])
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0].id, "real")
        self.assertIsNone(result.content)
        post.assert_awaited_once()

    async def test_failed_protocol_retry_moves_to_next_model(self):
        client = AIClient(api_key="test")
        post = AsyncMock(side_effect=[
            (200, _response(LEAK), 0), (400, None, 0),
            (200, _response(calls=[NATIVE]), 0),
        ])
        with (
            patch.object(client, "model_chain", return_value=["m1", "m2"]),
            patch.object(client, "_post_chat", post),
        ):
            result = await client.chat_with_tools([], tools=[SEARCH_TOOL])
        self.assertEqual(result.model, "m2")
        self.assertEqual(post.await_count, 3)
        self.assertNotIn("tool_choice", post.await_args_list[2].args[0])

    async def test_stream_retries_json_and_also_recovers_dsml(self):
        for first in (LEAK, (
            '<｜DSML｜calls><｜DSML｜invoke name="search_web">'
            '<｜DSML｜parameter name="query" string="true">news'
            '</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜calls>'
        )):
            with self.subTest(first=first):
                client = AIClient(api_key="test")
                attempts = []

                async def lines(payload, model_id, *, attempts=attempts, first=first):
                    attempts.append(payload)
                    delta = {"content": first} if len(attempts) == 1 else {
                        "tool_calls": [{"index": 0, **NATIVE}],
                    }
                    yield "data: " + json.dumps({"choices": [{"delta": delta}]})
                    yield "data: [DONE]"

                with (
                    patch.object(client, "model_chain", return_value=["m"]),
                    patch.object(client, "_stream_lines", lines),
                ):
                    result = await client.stream_chat_completion([], tools=[SEARCH_TOOL])
                self.assertIsNone(result.content)
                self.assertEqual(result.tool_calls[0].name, "search_web")
                self.assertEqual(len(attempts), 2 if first == LEAK else 1)

    async def test_fragmented_json_is_not_sent_to_the_stream_callback(self):
        client = AIClient(api_key="test")

        async def lines(payload, model_id):
            for piece in ("Запускаю.\n{", '"na', 'me": "web_search", ', '"arguments": {"query": "news"}}'):
                yield "data: " + json.dumps({"choices": [{"delta": {"content": piece}}]})
            yield "data: [DONE]"

        preview = AsyncMock()
        # Придержание недостроенного JSON нужно только в потоке с инструментами.
        with patch.object(client, "_stream_lines", lines):
            await client._stream_attempt(
                {"model": "m", "messages": [], "tools": [SEARCH_TOOL]}, preview
            )
        for call in preview.await_args_list:
            self.assertNotIn("{", call.args[0])

    async def test_stream_protocol_retry_is_bounded_and_fallback_resets_tool_choice(self):
        client = AIClient(api_key="test")
        attempts = []

        async def lines(payload, model_id):
            attempts.append((payload, model_id))
            delta = {"content": LEAK} if len(attempts) <= 2 else {"tool_calls": [NATIVE]}
            yield "data: " + json.dumps({"choices": [{"delta": delta}]})
            yield "data: [DONE]"

        with (
            patch.object(client, "model_chain", return_value=["m1", "m2"]),
            patch.object(client, "_stream_lines", lines),
        ):
            result = await client.stream_chat_completion([], tools=[SEARCH_TOOL])
        self.assertEqual(result.model, "m2")
        self.assertEqual([model for _, model in attempts], ["m1", "m1", "m2"])
        self.assertIn("tool_choice", attempts[1][0])
        self.assertNotIn("tool_choice", attempts[2][0])


class IsolatedSearchTests(unittest.IsolatedAsyncioTestCase):
    async def test_explicit_search_omits_entire_tainted_history_and_runs_tool_loop(self):
        for streaming in (False, True):
            with self.subTest(streaming=streaming):
                client = AIClient(api_key="test")
                message = SimpleNamespace(
                    chat=SimpleNamespace(id=1, type="private"),
                    from_user=SimpleNamespace(id=1), answer=AsyncMock(),
                )
                history = [
                    {"role": "user", "content": wrap_untrusted("document", "PRIVATE DOCUMENT")},
                    {"role": "assistant", "content": "PRIVATE DOCUMENT SUMMARY"},
                ]
                service = SimpleNamespace(
                    get_user_settings=AsyncMock(return_value={
                        "model": "m", "system_prompt": "sys", "temperature": 0.5, "max_tokens": 100,
                    }),
                    get_history=AsyncMock(return_value=history),
                    add_message=AsyncMock(return_value=SimpleNamespace(id=1)),
                )
                responses = [_response(LEAK), _response(calls=[NATIVE]), _response("Обзор [1]")]
                sent = []

                async def post(payload, *, sent=sent, responses=responses, **kwargs):
                    sent.append(payload)
                    return 200, responses.pop(0), 0

                async def lines(payload, model_id, *, sent=sent, responses=responses):
                    sent.append(payload)
                    data = responses.pop(0)["choices"][0]["message"]
                    yield "data: " + json.dumps({"choices": [{"delta": data}]})
                    yield "data: [DONE]"

                printer = SimpleNamespace(on_delta=AsyncMock(), discard=AsyncMock())
                with (
                    patch.multiple(config, stream_responses=streaming, request_intent=False,
                                   enable_image_generation=True, enable_sticker_generation=True,
                                   enable_file_replies=True),
                    patch.object(chat, "ai_client", client),
                    patch.object(chat, "search_configured", return_value=True),
                    patch.object(config, "builtin_web_search", return_value=False),
                    patch.object(chat, "mcp_available", return_value=True),
                    patch.object(chat.mcp_manager, "openai_tools", AsyncMock()) as mcp,
                    patch.object(chat, "search_web", AsyncMock(return_value=[])) as search,
                    patch.object(chat, "waiting", _no_wait),
                    patch.object(chat, "_StreamPrinter", return_value=printer),
                    patch.object(chat, "safe_answer", AsyncMock()),
                    patch.object(chat, "deliver_answer", AsyncMock(return_value=True)) as deliver,
                    patch.object(chat, "guest_hint", return_value=None),
                    patch.object(chat, "_retry_markup", return_value=None),
                    patch.object(client, "model_chain", return_value=["m"]),
                    patch.object(client, "_post_chat", post),
                    patch.object(client, "_stream_lines", lines),
                ):
                    await chat._answer_with_ai(message, service, QUESTION, AccessResult(True, "admin"))
                search.assert_awaited_once_with("AI news November 8 2026", None)
                # Каталог теперь загружается до изоляции, но MCP в поисковом
                # режиме не предлагается: изолированный search-only контекст.
                mcp.assert_awaited_once()
                deliver.assert_awaited_once()
                self.assertEqual(deliver.await_args.args[2], "Обзор [1]")
                self.assertEqual(len(sent), 3)
                for payload in sent:
                    self.assertNotIn("PRIVATE", json.dumps(payload, ensure_ascii=False))
                    self.assertEqual([t["function"]["name"] for t in payload["tools"]], ["search_web"])
                    self.assertIn("Текущая дата", payload["messages"][0]["content"])
                self.assertIn(SEARCH_SYSTEM_HINT, sent[0]["messages"][0]["content"])
                self.assertEqual(len(history), 2)
                saved = [call.kwargs["content"] for call in service.add_message.await_args_list]
                self.assertEqual(saved, [QUESTION, "Обзор [1]"])

    async def test_reminder_explicit_search_also_omits_tainted_history(self):
        final = ChatResult(content="done")
        history = [{"role": "user", "content": wrap_untrusted("web", "PRIVATE PAGE")}]
        with (
            patch.object(reminders, "search_configured", return_value=True),
            patch.object(config, "builtin_web_search", return_value=False),
            patch.object(reminders.ai_client, "chat_with_tools", AsyncMock(return_value=final)) as tools,
            patch.object(reminders.ai_client, "chat_completion_result", AsyncMock(return_value=final)) as plain,
        ):
            await reminders._run_ai_prompt(
                reminders.DueReminder(id=1, user_id=1, chat_id=1, text=QUESTION, ai_mode=True, schedule=None),
                {"model": "m", "system_prompt": "sys", "temperature": 0.5, "max_tokens": 100}, history,
            )
        tools.assert_awaited_once()
        plain.assert_not_awaited()
        self.assertNotIn("PRIVATE", json.dumps(tools.await_args.kwargs, ensure_ascii=False))


class CurrentDateTests(unittest.TestCase):
    def test_reported_scenario_uses_november_8_not_july_10(self):
        from bot.services.time_context import current_time_hint

        hint = current_time_hint(now=datetime.datetime(2026, 11, 9, 6, 52, tzinfo=datetime.UTC))
        self.assertIn("Вчера: 2026-11-08", hint)

    def test_yesterday_is_computed_in_the_selected_timezone_at_month_boundary(self):
        from bot.services.time_context import current_time_hint

        hint = current_time_hint(
            datetime.timezone(datetime.timedelta(hours=3)),
            now=datetime.datetime(2026, 10, 31, 22, 30, tzinfo=datetime.UTC),
        )
        self.assertIn("2026-11-01 01:30", hint)
        self.assertIn("Вчера: 2026-10-31", hint)


class SearchIsolationSafetyTests(unittest.TestCase):
    def _messages(self, text):
        history = [{"role": "user", "content": wrap_untrusted("document", "PRIVATE")}]
        return [{"role": "system", "content": "sys"}, *history,
                {"role": "user", "content": text}]

    def test_current_external_content_and_ordinary_followups_keep_the_gate(self):
        for text in ("объясни второй пункт", QUESTION + "\n" + wrap_untrusted("quoted", "data")):
            with self.subTest(text=text):
                messages = self._messages(text)
                self.assertFalse(isolate_search_history(messages))
                self.assertEqual(len(messages), 3)

    def test_clean_history_is_kept_even_for_an_explicit_search(self):
        messages = [{"role": "system", "content": "sys"},
                    {"role": "user", "content": "Earlier question"},
                    {"role": "user", "content": QUESTION}]
        self.assertFalse(isolate_search_history(messages))
        self.assertEqual(len(messages), 3)

    def test_new_topic_after_document_is_isolated_without_an_online_keyword(self):
        for text in (
            "Какая сейчас погода в Москве?",
            "Актуальный курс доллара на сегодня",
            "Проверь последние новости",
            "What is the current exchange rate?",
        ):
            with self.subTest(text=text):
                messages = self._messages(text)
                self.assertTrue(isolate_search_history(messages))
                self.assertEqual(len(messages), 2)
                self.assertNotIn("PRIVATE", str(messages))

    def test_note_requests_are_left_to_mcp_isolation(self):
        for text in ("Поищи во втором мозге про Python", "Search my notes for Rust"):
            with self.subTest(text=text):
                messages = self._messages(text)
                self.assertFalse(isolate_search_history(messages))
                self.assertEqual(len(messages), 3)

    def test_short_acknowledgements_do_not_drop_the_history(self):
        messages = self._messages("спасибо")
        self.assertFalse(isolate_search_history(messages))
        self.assertEqual(len(messages), 3)
