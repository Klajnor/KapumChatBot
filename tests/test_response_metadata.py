"""Футер показывает только подтверждённый поиск и числовые reasoning-токены API."""

from __future__ import annotations

import copy
import json
import unittest
from unittest.mock import AsyncMock, patch

from bot.config import config
from bot.services.ai_client import AIClient, ChatResult, _native_response
from bot.services.response_metadata import (
    AnswerMetadata,
    ResponseMetadata,
    combine_metadata,
    model_search_used,
    reasoning_tokens,
)
from bot.services.search import SEARCH_TOOL
from bot.services.telegram_utils import format_timing


class ResponseCountersTests(unittest.TestCase):
    def test_reasoning_formats_and_explicit_zero(self):
        for usage, expected in (
            ({"completion_tokens_details": {"reasoning_tokens": 15}}, 15),
            ({"output_tokens_details": {"reasoning_tokens": 20}}, 20),
            ({"reasoning_tokens": 7}, 7),
            ({"output_tokens_details": {"reasoning_tokens": 0}}, 0),
            ({"completion_tokens": 100, "reasoning_content": "long thought"}, None),
        ):
            with self.subTest(usage=usage):
                self.assertEqual(reasoning_tokens(usage), expected)
                self.assertEqual(ChatResult(usage=usage).reasoning_tokens, expected)

    def test_invalid_or_unreported_counts_are_not_zero(self):
        for value in (None, True, -1, 1.5, "15", float("nan"), float("inf"), 10**1000):
            with self.subTest(value=str(value)[:20]):
                self.assertIsNone(reasoning_tokens({"reasoning_tokens": value}))

    def test_actual_provider_search_not_enable_flag_or_response_text(self):
        self.assertIsNone(model_search_used({"enable_search": True}, enabled=True))
        self.assertIsNone(model_search_used({"content": "I searched the web: https://example.org"}))
        self.assertFalse(model_search_used({}, enabled=False))
        self.assertTrue(model_search_used({"usage": {"plugins": {"search": {"count": 1}}}}))
        self.assertFalse(model_search_used({"usage": {"plugins": {"search": {"count": 0}}}}))
        self.assertTrue(model_search_used({"output": {"search_info": {"search_results": [{"url": "x"}]}}}))
        self.assertIsNone(model_search_used({"output": {"search_info": {"search_results": []}}}))

    def test_missing_reasoning_in_one_round_marks_reported_sum_as_partial(self):
        merged = combine_metadata([
            ResponseMetadata(model_search_used=True, reasoning_tokens=12),
            ResponseMetadata(model_search_used=False),
            ResponseMetadata(model_search_used=False, reasoning_tokens=5),
        ])
        self.assertTrue(merged.model_search_used)
        self.assertEqual(merged.reasoning_tokens, 17)
        self.assertTrue(merged.reasoning_partial)


class MetadataFooterTests(unittest.TestCase):
    def footer(self, metadata, **settings):
        with patch.multiple(config, show_timing=True, timing_format="{seconds} s · {model}",
                            timing_format_tokens="{seconds} s · ↑{input} ↓{output}", **settings):
            return format_timing(1.0, model="m", metadata=metadata)

    def test_search_sources_none_and_unknown(self):
        for model, api, label in (
            (True, False, "model"), (False, True, "API"), (True, True, "model + API"),
            (False, False, "none"), (None, False, "unknown"), (None, True, "API; model unknown"),
        ):
            with self.subTest(model=model, api=api):
                self.assertIn("🔎 " + label, self.footer(ResponseMetadata(model, api)))

    def test_reasoning_missing_zero_and_partial(self):
        self.assertNotIn("🧠", self.footer(ResponseMetadata(model_search_used=False)))
        self.assertIn("🧠 0 tok", self.footer(ResponseMetadata(reasoning_tokens=0)))
        self.assertIn("🧠 ≥17 tok", self.footer(ResponseMetadata(reasoning_tokens=17, reasoning_partial=True)))

    def test_custom_placeholders_do_not_duplicate_automatic_suffix(self):
        with patch.multiple(config, show_timing=True,
                            timing_format_tokens="{model} search={search} reasoning={reasoning}"):
            footer = format_timing(1, model="m", metadata=ResponseMetadata(True, True, 12))
        self.assertEqual(footer, "m search=model + API reasoning=12")

    def test_footer_can_be_disabled_and_invalid_template_keeps_metrics(self):
        with patch.object(config, "show_timing", False):
            self.assertEqual(format_timing(1, metadata=ResponseMetadata(True, False, 4)), "")
        with patch.multiple(config, show_timing=True, timing_format_tokens="{bad}"):
            footer = format_timing(1, metadata=ResponseMetadata(True, False, 4))
        self.assertIn("🔎 model", footer)
        self.assertIn("🧠 4 tok", footer)


class APIUsageMetadataTests(unittest.IsolatedAsyncioTestCase):
    async def test_regular_native_compatible_and_vision_responses(self):
        data = {"choices": [{"message": {"content": "answer"}}], "usage": {
            "output_tokens_details": {"reasoning_tokens": 25},
            "plugins": {"search": {"count": 1}},
        }}
        for mode in ("plain", "tools", "vision"):
            with self.subTest(mode=mode):
                client = AIClient(api_key="test")
                with (
                    patch.object(client, "model_chain", return_value=["m"]),
                    patch.object(client, "_post_chat", AsyncMock(return_value=(200, copy.deepcopy(data), 0))),
                ):
                    if mode == "vision":
                        result = await client.chat_with_image_result("q", b"image", [], model="m")
                    elif mode == "tools":
                        result = await client.chat_with_tools([], model="m", tools=[SEARCH_TOOL])
                    else:
                        result = await client.chat_completion_result([], model="m")
                self.assertTrue(result.metadata.model_search_used)
                self.assertEqual(result.reasoning_tokens, 25)
        native = _native_response({"output": {"text": "a", "search_info": {
            "search_results": [{"url": "https://example.org"}],
        }}, "usage": {"reasoning_tokens": 3}})
        result = AIClient._parse_result(native)
        self.assertTrue(result.metadata.model_search_used)
        self.assertEqual(result.reasoning_tokens, 3)

    async def test_sse_cumulative_usage_is_not_double_counted(self):
        client = AIClient(api_key="test")

        async def lines(payload, model_id):
            for event in (
                {"choices": [{"delta": {"content": "answer"}}], "usage": {
                    "completion_tokens_details": {"reasoning_tokens": 10},
                    "plugins": {"search": {"count": 1}},
                }},
                {"choices": [], "usage": {"completion_tokens_details": {"reasoning_tokens": 15}}},
                {"choices": [], "usage": {"completion_tokens": 50}},
            ):
                yield "data: " + json.dumps(event)
            yield "data: [DONE]"

        with patch.object(client, "_stream_lines", lines):
            result = await client._stream_attempt({"model": "m", "enable_search": True})
        self.assertTrue(result.metadata.model_search_used)
        self.assertEqual(result.reasoning_tokens, 15)
        self.assertEqual(result.output_tokens, 50)

    async def test_native_sse_keeps_search_info_and_usage_after_finish(self):
        client = AIClient(api_key="test")

        async def upstream():
            for event in (
                {"output": {"choices": [{"message": {"content": "answer"}, "finish_reason": "stop"}],
                            "search_info": {"search_results": [{"url": "https://example.org"}]}}},
                {"usage": {"output_tokens_details": {"reasoning_tokens": 7}, "output_tokens": 20}},
            ):
                yield ("data: " + json.dumps(event) + "\n\n").encode()

        async def lines(payload, model_id):
            async for line in client._native_stream_lines(upstream()):
                yield line

        with patch.object(client, "_stream_lines", lines):
            result = await client._stream_attempt({"model": "m", "enable_search": True})
        self.assertTrue(result.metadata.model_search_used)
        self.assertEqual(result.reasoning_tokens, 7)

    async def test_model_fallback_preserves_search_and_reasoning_from_earlier_attempt(self):
        client = AIClient(api_key="test")
        with (
            patch.object(client, "model_chain", return_value=["m1", "m2"]),
            patch.object(client, "_post_chat", AsyncMock(side_effect=[
                (200, {"choices": [], "usage": {"reasoning_tokens": 11, "plugins": {"search": {"count": 1}}}}, 0),
                (200, {"choices": [{"message": {"content": "answer"}}], "usage": {"reasoning_tokens": 4}}, 0),
            ])),
        ):
            result = await client.chat_completion_result([])
        self.assertEqual(result.reasoning_tokens, 15)
        self.assertTrue(result.metadata.model_search_used)
        stats = AnswerMetadata()
        stats.add(result)
        stats.api_search_used = True
        self.assertEqual(stats.summary(), ResponseMetadata(True, True, 15))
