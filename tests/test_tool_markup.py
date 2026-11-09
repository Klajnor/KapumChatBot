"""Регрессии очистки текстовых вызовов инструментов из ответа модели."""

from __future__ import annotations

import json
import unittest
from unittest.mock import AsyncMock, patch

from bot.config import config
from bot.services.ai_client import AIClient
from bot.services.search import SEARCH_TOOL
from bot.services.tool_markup import (
    TOOL_CALLS_HINT,
    strip_tool_markup,
    tool_stream_preview,
)

LT = "\x3c"


def tag(name: str, body: str = "", *, attrs: str = "") -> str:
    """Build a paired tag without a literal «less than» in the test source."""
    opening = f"{LT}{name}{(' ' + attrs) if attrs else ''}>"
    return f"{opening}{body}{LT}/{name}>"


class StripToolMarkupTests(unittest.TestCase):
    def test_json_search_calls_with_preamble_are_not_an_answer_when_tools_offered(self):
        call = json.dumps({"name": "web_search", "arguments": {"query": "AI news"}})
        text = "Запускаю поиск.\n" + call + " " + call
        self.assertIsNone(strip_tool_markup(text, drop_json_calls=True))
        # Без предложенных инструментов это обычный ответ: убираем только XML
        # вызовы, но не выбрасываем весь текст.
        self.assertEqual(strip_tool_markup(text), "Запускаю поиск.")

    def test_json_native_envelope_and_call_list_are_dropped_when_tools_offered(self):
        call = {"id": "c1", "type": "function", "function": {
            "name": "search_web", "arguments": '{"query": "news"}',
        }}
        for value in (call, [call], {"tool_calls": [call]}):
            with self.subTest(value=value):
                self.assertIsNone(
                    strip_tool_markup(json.dumps(value), drop_json_calls=True)
                )
                self.assertIsNone(strip_tool_markup(json.dumps(value)))

    def test_json_code_examples_and_unrelated_data_are_preserved(self):
        call = '{"name": "web_search", "arguments": {"query": "news"}}'
        for text in (
            f"Пример: `{call}`", f"Пример:\n```json\n{call}\n```",
            '{"name": "product", "arguments": {"color": "red"}}',
            '{"query": "AI news", "results": []}',
        ):
            with self.subTest(text=text):
                self.assertEqual(strip_tool_markup(text), text)

    def test_stream_preview_keeps_prose_formulas_and_completed_code_examples(self):
        for text in (
            "Обычный ответ.", r"Формула: $\frac{a}{b}$",
            'Пример: `{"name":"web_search","arguments":{"query":"news"}}`',
            '```json\n{"name":"web_search","arguments":{"query":"news"}}\n```',
        ):
            with self.subTest(text=text):
                self.assertEqual(tool_stream_preview(text), text)

    def test_plain_text_and_none_are_untouched(self):
        self.assertIsNone(strip_tool_markup(None))
        self.assertEqual(strip_tool_markup("Обычный ответ."), "Обычный ответ.")

    def test_observed_search_notes_leak_is_removed(self):
        text = tag("search_notes", "тревеллайн") + "\n" + tag(
            "search_notes", "Travelline"
        )
        self.assertIsNone(strip_tool_markup(text))

    def test_answer_survives_around_the_leak(self):
        text = "Вот что нашлось:\n\n" + tag("search_notes", "тревеллайн")
        self.assertEqual(strip_tool_markup(text), "Вот что нашлось:")

    def test_orphan_closing_tag_is_removed(self):
        self.assertIsNone(strip_tool_markup(f"{LT}/search_notes>"))

    def test_tool_call_and_anthropic_envelopes_are_removed(self):
        envelope = tag("tool_call", tag("search_web", attrs='query="news"'))
        self.assertIsNone(strip_tool_markup(envelope))
        anthropic = tag(
            "function_calls",
            tag(
                "invoke",
                tag("parameter", "q", attrs='name="query"'),
                attrs='name="search_notes"',
            ),
        )
        self.assertIsNone(strip_tool_markup(anthropic))

    def test_mcp_prefixed_names_are_removed(self):
        self.assertIsNone(strip_tool_markup(tag("mcp__brain__search_notes", "q")))

    def test_self_closing_tag_is_removed(self):
        self.assertIsNone(strip_tool_markup(f'{LT}search_notes query="q"/>'))

    def test_code_examples_are_preserved(self):
        fenced = "Пример:\n```\n" + tag("search_notes", "q") + "\n```"
        self.assertEqual(strip_tool_markup(fenced), fenced)
        inline = "Тег " + f"`{tag('search_notes', 'q')}`" + " — это вызов."
        self.assertEqual(strip_tool_markup(inline), inline)

    def test_plain_html_is_not_touched(self):
        html = f"{LT}div>привет{LT}/div>"
        self.assertEqual(strip_tool_markup(html), html)

    def test_lone_less_than_keeps_layout(self):
        text = "1 \x3c 2\n\n\nи ещё"
        self.assertEqual(strip_tool_markup(text), text)

    def test_dsml_envelope_is_dropped(self):
        dsml = (
            f'{LT}｜DSML｜calls>{LT}｜DSML｜invoke name="search_notes">'
            f'{LT}｜DSML｜parameter name="query" string="true">q'
            f"{LT}/｜DSML｜parameter>{LT}/｜DSML｜invoke>{LT}/｜DSML｜calls>"
        )
        self.assertIsNone(strip_tool_markup(dsml))

    def test_finalize_applies_the_sanitizer(self):
        text = "Ответ\n\n" + tag("search_notes", "q")
        with patch.object(config, "rich_messages", True):
            self.assertEqual(AIClient._finalize(text), "Ответ")


class StreamPreviewTests(unittest.TestCase):
    def test_benign_json_is_not_held_back(self):
        for tools_offered in (False, True):
            with self.subTest(tools_offered=tools_offered):
                text = 'Ответ: {"a": 1} и ещё текст'
                self.assertEqual(
                    tool_stream_preview(text, tools_offered=tools_offered), text
                )

    def test_partial_tool_json_is_held_only_when_tools_are_offered(self):
        text = 'Запускаю.\n{"na'
        self.assertEqual(tool_stream_preview(text, tools_offered=True), "Запускаю.")
        self.assertEqual(tool_stream_preview(text), text)

    def test_complete_json_call_is_hidden_only_when_tools_are_offered(self):
        text = 'Ищу...\n{"name": "web_search", "arguments": {"query": "x"}}'
        self.assertIsNone(tool_stream_preview(text, tools_offered=True))
        self.assertEqual(tool_stream_preview(text), "Ищу...")


class PlainStreamPreviewTests(unittest.IsolatedAsyncioTestCase):
    async def test_benign_json_streams_in_full_without_tools(self):
        client = AIClient(api_key="test")
        text = 'Ответ: {"a": 1} и ещё текст'

        async def lines(payload, model_id):
            for piece in ('Ответ: {"a": 1}', " и ещё текст"):
                yield "data: " + json.dumps(
                    {"choices": [{"delta": {"content": piece}}]}
                )
            yield "data: [DONE]"

        preview = AsyncMock()
        with patch.object(client, "_stream_lines", lines):
            await client._stream_attempt({"model": "m", "messages": []}, preview)
        self.assertEqual(preview.await_args_list[-1].args[0], text)


class PlainChatJsonTests(unittest.IsolatedAsyncioTestCase):
    async def test_plain_chat_keeps_answer_with_a_json_tool_example(self):
        """Ответ без инструментов не должен проваливать всю fallback-цепочку."""
        text = 'Пример вызова: {"name": "get_note", "arguments": {"path": "x"}}'
        client = AIClient(api_key="test")
        with (
            patch.object(client, "model_chain", return_value=["m1", "m2"]),
            patch.object(client, "_post_chat", AsyncMock(return_value=(
                200, {"choices": [{"message": {"content": text}}]}, 0,
            ))) as post,
        ):
            result = await client.chat_completion_result([])
        self.assertEqual(result.content, "Пример вызова:")
        post.assert_awaited_once()

    async def test_vision_answer_with_json_is_not_dropped(self):
        client = AIClient(api_key="test")
        text = 'Объект: {"name": "search_web", "arguments": {}}'
        with (
            patch.object(client, "model_chain", return_value=["m"]),
            patch.object(client, "_post_chat", AsyncMock(return_value=(
                200, {"choices": [{"message": {"content": text}}]}, 0,
            ))),
        ):
            result = await client.chat_completion_result([])
        self.assertEqual(result.content, "Объект:")


class ToolCallsHintTests(unittest.IsolatedAsyncioTestCase):
    def test_hint_is_added_when_tools_are_offered(self):
        messages = [{"role": "system", "content": "sys"}]
        with patch.object(config, "enable_style_hint", False):
            payload = AIClient._chat_payload(
                messages, None, 0.5, 100, tools=[SEARCH_TOOL]
            )
        self.assertIn(TOOL_CALLS_HINT, payload["messages"][0]["content"])
        self.assertNotIn(TOOL_CALLS_HINT, messages[0]["content"])

    def test_hint_is_absent_without_tools(self):
        with patch.object(config, "enable_style_hint", False):
            payload = AIClient._chat_payload(
                [{"role": "system", "content": "sys"}], None, 0.5, 100
            )
        self.assertNotIn(TOOL_CALLS_HINT, payload["messages"][0]["content"])

    async def test_tools_rejection_retry_drops_the_hint(self):
        client = AIClient(api_key="test")
        sent = []

        async def post(payload, **kwargs):
            # Snapshot now: the caller mutates the same dict before the retry.
            sent.append(([dict(entry) for entry in payload["messages"]], "tools" in payload))
            if len(sent) == 1:
                return 400, None, 0.0
            return 200, {"choices": [{"message": {"content": "ok"}}]}, 0.0

        with (
            patch.object(client, "model_chain", return_value=["m"]),
            patch.object(client, "_post_chat", side_effect=post),
        ):
            result = await client.chat_with_tools(
                [{"role": "system", "content": "sys"}], tools=[SEARCH_TOOL]
            )
        self.assertEqual(result.content, "ok")
        self.assertTrue(sent[0][1])
        self.assertIn(TOOL_CALLS_HINT, sent[0][0][0]["content"])
        self.assertFalse(sent[1][1])
        self.assertNotIn(TOOL_CALLS_HINT, sent[1][0][0]["content"])


if __name__ == "__main__":
    unittest.main()
