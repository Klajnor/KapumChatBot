"""Provider-side web search is preferred over the local search_web tool."""

from __future__ import annotations

import copy
import unittest
from unittest.mock import AsyncMock, patch

from bot.config import Config, config
from bot.services.ai_client import (
    BUILTIN_SEARCH_LANGUAGE_HINT,
    AIClient,
    _native_payload,
)
from bot.services.search import SEARCH_TOOL
from bot.services.tool_markup import TOOL_CALLS_HINT


class BuiltinSearchConfigTests(unittest.TestCase):
    def test_auto_detection_by_model_family(self):
        catalog = [
            {
                "id": "qwen",
                "model": "qwen3.8-flash",
                "api_type": "dashscope",
                "web_search": "auto",
            },
            {
                "id": "deepseek",
                "model": "deepseek-v4.1-flash-1017",
                "api_type": "dashscope",
                "web_search": "auto",
            },
            {
                "id": "glm",
                "model": "glm-5.3",
                "api_type": "dashscope",
                "web_search": "auto",
            },
            {
                "id": "forced",
                "model": "glm-5.3",
                "api_type": "dashscope",
                "web_search": "true",
            },
            {
                "id": "disabled",
                "model": "qwen3.8-flash",
                "api_type": "dashscope",
                "web_search": "false",
            },
            {
                "id": "openai-auto",
                "model": "qwen3.8-flash",
                "api_type": "openai",
                "web_search": "auto",
            },
        ]
        with (
            patch.object(config, "ai_models", catalog),
            patch.object(config, "ai_fallback_models", []),
        ):
            self.assertTrue(config.builtin_web_search("qwen"))
            self.assertTrue(config.builtin_web_search("deepseek"))
            self.assertFalse(config.builtin_web_search("glm"))
            self.assertTrue(config.builtin_web_search("forced"))
            self.assertFalse(config.builtin_web_search("disabled"))
            self.assertFalse(config.builtin_web_search("openai-auto"))

    def test_global_web_search_switch_disables_provider_search(self):
        with patch.object(config, "enable_web_search", False):
            self.assertFalse(config.builtin_web_search("qwen3.8-flash"))
            self.assertFalse(config.builtin_web_search("glm-5.3"))

    def test_web_search_setting_is_read_from_env(self):
        import os

        env = {
            "MODEL_1_ID": "qwen-test",
            "MODEL_1_MODEL": "qwen3.8-flash",
            "MODEL_1_WEB_SEARCH": "false",
        }
        with patch.dict(os.environ, env):
            settings = Config()
        spec = next(s for s in settings.ai_models if s["id"] == "qwen-test")
        self.assertEqual(spec["web_search"], "false")
        self.assertFalse(settings.builtin_web_search("qwen-test"))


class BuiltinSearchPayloadTests(unittest.TestCase):
    def test_public_payload_enables_provider_search(self):
        client = AIClient(api_key="test")
        with patch.object(config, "builtin_web_search", return_value=True):
            payload = client._chat_payload(
                [{"role": "user", "content": "news?"}],
                "qwen3.8-flash",
                0.2,
                100,
                public=True,
            )
        self.assertTrue(payload["enable_search"])
        self.assertEqual(payload["search_options"]["enable_source"], True)
        native = _native_payload(payload)
        self.assertTrue(native["parameters"]["enable_search"])
        self.assertEqual(
            native["parameters"]["search_options"]["search_strategy"], "turbo"
        )

    def test_explicit_internet_request_forces_provider_search(self):
        client = AIClient(api_key="test")
        with patch.object(config, "builtin_web_search", return_value=True):
            payload = client._chat_payload(
                [{"role": "user", "content": "Найди в интернете погоду в Москве"}],
                "qwen3.8-flash",
                0.2,
                100,
                public=True,
            )
        self.assertTrue(payload["search_options"]["forced_search"])
        native = _native_payload(payload)
        self.assertTrue(native["parameters"]["search_options"]["forced_search"])

    def test_internal_payload_does_not_enable_provider_search(self):
        client = AIClient(api_key="test")
        with patch.object(config, "builtin_web_search", return_value=True):
            payload = client._chat_payload(
                [{"role": "user", "content": "classify"}],
                "qwen3.8-flash",
                0.2,
                100,
                public=False,
            )
        self.assertNotIn("enable_search", payload)

    def test_disabled_model_uses_local_search_instead(self):
        client = AIClient(api_key="test")
        with patch.object(config, "builtin_web_search", return_value=False):
            payload = client._chat_payload(
                [{"role": "user", "content": "news?"}],
                "glm-5.3",
                0.2,
                100,
                public=True,
            )
        self.assertNotIn("enable_search", payload)


class BuiltinSearchLanguageTests(unittest.IsolatedAsyncioTestCase):
    def test_english_query_hint_keeps_original_text_and_model_search_options(self):
        for question in (
            "Найди в интернете прогноз погоды в Москве на 13 ноября 2026",
            "Search the web for Rust 1.91 release notes",
        ):
            with self.subTest(question=question):
                messages = [{"role": "system", "content": "Отвечай по-русски."},
                            {"role": "user", "content": question}]
                original = copy.deepcopy(messages)
                with (
                    patch.object(config, "builtin_web_search", return_value=True),
                    patch.object(config, "enable_style_hint", False),
                ):
                    payload = AIClient._chat_payload(messages, "qwen", 0.2, 100, public=True)
                self.assertEqual(messages, original)
                self.assertEqual(payload["messages"][-1]["content"], question)
                self.assertIn(BUILTIN_SEARCH_LANGUAGE_HINT, payload["messages"][0]["content"])
                self.assertIn("Отвечай по-русски.", payload["messages"][0]["content"])
                self.assertTrue(payload["enable_search"])
                self.assertTrue(payload["search_options"]["forced_search"])
                self.assertTrue(payload["search_options"]["enable_source"])

    def test_hint_creates_system_message_when_missing(self):
        messages = [{"role": "user", "content": "Какая сейчас погода?"}]
        with (
            patch.object(config, "builtin_web_search", return_value=True),
            patch.object(config, "enable_style_hint", False),
        ):
            payload = AIClient._chat_payload(messages, "qwen", 0.2, 100, public=True)
        self.assertEqual(payload["messages"][0], {"role": "system", "content": BUILTIN_SEARCH_LANGUAGE_HINT})
        self.assertNotIn("forced_search", payload["search_options"])
        native = _native_payload(payload)
        self.assertIn(BUILTIN_SEARCH_LANGUAGE_HINT, native["input"]["messages"][0]["content"])
        self.assertEqual(len(messages), 1)

    def test_hint_is_not_duplicated_and_keeps_tool_hint_and_mcp_arguments(self):
        messages = [
            {"role": "system", "content": "sys\n\n" + BUILTIN_SEARCH_LANGUAGE_HINT},
            {"role": "user", "content": "Найди в моих заметках план проекта"},
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "c1", "type": "function", "function": {
                    "name": "mcp__brain__search_notes", "arguments": '{"query":"план проекта"}',
                },
            }]},
            {"role": "tool", "tool_call_id": "c1", "content": "Найден план"},
        ]
        original = copy.deepcopy(messages)
        with (
            patch.object(config, "builtin_web_search", return_value=True),
            patch.object(config, "enable_style_hint", False),
        ):
            payload = AIClient._chat_payload(messages, "qwen", 0.2, 100, tools=[SEARCH_TOOL], public=True)
        system = payload["messages"][0]["content"]
        self.assertEqual(system.count(BUILTIN_SEARCH_LANGUAGE_HINT), 1)
        self.assertIn(TOOL_CALLS_HINT, system)
        self.assertEqual(payload["messages"][1:], original[1:])
        self.assertEqual(messages, original)

    def test_internal_or_local_search_does_not_get_translation_hint(self):
        for public, builtin in ((False, True), (True, False), (False, False)):
            with self.subTest(public=public, builtin=builtin):
                with patch.object(config, "builtin_web_search", return_value=builtin):
                    payload = AIClient._chat_payload(
                        [{"role": "system", "content": "sys"}, {"role": "user", "content": "Текст"}],
                        "m", 0.2, 100, public=public,
                    )
                self.assertNotIn(BUILTIN_SEARCH_LANGUAGE_HINT, payload["messages"][0]["content"])

    async def test_single_shot_and_stream_use_hint_without_a_translation_request(self):
        for streaming in (False, True):
            with self.subTest(streaming=streaming):
                client = AIClient(api_key="test")
                seen = []

                async def post(payload, *, seen=seen, **kwargs):
                    seen.append(copy.deepcopy(payload))
                    return 200, {"choices": [{"message": {"content": "Прогноз найден."}}]}, 0

                async def lines(payload, model_id, *, seen=seen):
                    seen.append(copy.deepcopy(payload))
                    yield 'data: {"choices":[{"delta":{"content":"Прогноз найден."}}]}'
                    yield "data: [DONE]"

                with (
                    patch.object(config, "builtin_web_search", return_value=True),
                    patch.object(client, "model_chain", return_value=["qwen"]),
                    patch.object(client, "_post_chat", AsyncMock(side_effect=post)),
                    patch.object(client, "_stream_lines", lines),
                ):
                    messages = [{"role": "user", "content": "Найди в интернете прогноз погоды"}]
                    if streaming:
                        result = await client.stream_chat_completion(messages, model="qwen", public=True)
                    else:
                        result = await client.chat_completion_result(messages, model="qwen", public=True)
                self.assertEqual(len(seen), 1)
                self.assertIn(BUILTIN_SEARCH_LANGUAGE_HINT, seen[0]["messages"][0]["content"])
                self.assertEqual(result.content, "Прогноз найден.")


if __name__ == "__main__":
    unittest.main()
