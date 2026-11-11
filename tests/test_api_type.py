from __future__ import annotations

import json
import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from bot.config import Config, config
from bot.services.ai_client import (
    AIClient,
    _derive_compatible_base_url,
    _derive_native_base_url,
    _native_payload,
    _native_response,
)


class _AsyncContent:
    """Minimal async byte stream for a fake aiohttp response."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    def __aiter__(self):
        self._iterator = iter(self._chunks)
        return self

    async def __anext__(self) -> bytes:
        try:
            return next(self._iterator)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


def _stream_response(chunks: list[bytes], status: int = 200) -> MagicMock:
    response = MagicMock()
    response.status = status
    response.charset = "utf-8"
    response.reason = "OK"
    response.headers = {}
    response.content = _AsyncContent(chunks)
    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=response)
    context.__aexit__ = AsyncMock(return_value=False)
    return context


def _json_response(body: dict) -> MagicMock:
    response = MagicMock()
    response.status = 200
    response.charset = "utf-8"
    response.reason = "OK"
    response.headers = {}
    response.content.read = AsyncMock(
        side_effect=[json.dumps(body).encode(), b""]
    )
    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=response)
    context.__aexit__ = AsyncMock(return_value=False)
    return context


class ApiTypeConfigTests(unittest.TestCase):
    def test_default_is_native_dashscope(self):
        with patch.dict(os.environ, {"AI_API_TYPE": ""}):
            self.assertEqual(Config().ai_api_type, "dashscope")

    def test_native_and_openai_aliases(self):
        for value in ("native", "DashScope", "qwen", "dashscope-native"):
            with self.subTest(value=value), patch.dict(
                os.environ, {"AI_API_TYPE": value}
            ):
                self.assertEqual(Config().ai_api_type, "dashscope")
        for value in ("openai", "compatible", "openai-compatible", "compat"):
            with self.subTest(value=value), patch.dict(
                os.environ, {"AI_API_TYPE": value}
            ):
                self.assertEqual(Config().ai_api_type, "openai")

    def test_unknown_value_falls_back_to_dashscope_with_warning(self):
        with (
            patch.dict(os.environ, {"AI_API_TYPE": "something-else"}),
            self.assertLogs("bot.config", level="WARNING") as logs,
        ):
            settings = Config()
        self.assertEqual(settings.ai_api_type, "dashscope")
        self.assertIn("Invalid AI_API_TYPE", "\n".join(logs.output))

    def test_each_model_can_override_the_global_api_type(self):
        env = {
            "AI_API_TYPE": "dashscope",
            "MODEL_1_ID": "native-model",
            "MODEL_1_MODEL": "native-provider",
            "MODEL_1_API_TYPE": "",
            "MODEL_2_ID": "openai-model",
            "MODEL_2_MODEL": "openai-provider",
            "MODEL_2_API_TYPE": "openai",
            "FALLBACK_MODEL_ID": "fallback-model",
            "FALLBACK_MODEL_MODEL": "fallback-provider",
            "FALLBACK_MODEL_API_TYPE": "compatible",
        }
        with patch.dict(os.environ, env):
            settings = Config()
        by_id = {spec["id"]: spec for spec in settings.ai_models}
        self.assertEqual(by_id["native-model"]["api_type"], "dashscope")
        self.assertEqual(by_id["openai-model"]["api_type"], "openai")
        self.assertEqual(
            settings.ai_fallback_models[0]["api_type"], "openai"
        )

    def test_invalid_per_model_api_type_falls_back_to_global(self):
        env = {
            "AI_API_TYPE": "dashscope",
            "MODEL_1_ID": "bad-model",
            "MODEL_1_MODEL": "bad-provider",
            "MODEL_1_API_TYPE": "unknown",
        }
        with (
            patch.dict(os.environ, env),
            self.assertLogs("bot.config", level="WARNING") as logs,
        ):
            settings = Config()
        self.assertEqual(settings.ai_models[0]["api_type"], "dashscope")
        self.assertIn("Invalid API_TYPE for model bad-model", "\n".join(logs.output))


class BaseUrlTests(unittest.TestCase):
    def test_native_and_compatible_bases_are_interchangeable(self):
        compatible = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        native = "https://dashscope-intl.aliyuncs.com/api/v1"
        self.assertEqual(_derive_native_base_url(compatible), native)
        self.assertEqual(_derive_native_base_url(native), native)
        self.assertEqual(_derive_compatible_base_url(native), compatible)
        self.assertEqual(_derive_compatible_base_url(compatible), compatible)

    def test_custom_base_is_not_rewritten_for_compatible_mode(self):
        self.assertEqual(
            _derive_compatible_base_url("https://api.example.test/v1"),
            "https://api.example.test/v1",
        )
        self.assertEqual(
            _derive_native_base_url("https://api.example.test/v1"),
            "https://api.example.test/v1/api/v1",
        )


class NativePayloadTests(unittest.TestCase):
    def test_tools_thinking_and_images_move_to_native_fields(self):
        payload = {
            "model": "qwen3.8-max",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Что на фото?"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,AAA"},
                        },
                    ],
                }
            ],
            "temperature": 0.4,
            "max_tokens": 100,
            "tools": [{"type": "function", "function": {"name": "search_web"}}],
            "tool_choice": {
                "type": "function",
                "function": {"name": "search_web"},
            },
            "enable_thinking": True,
            "thinking_budget": 1024,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        native = _native_payload(payload)
        self.assertEqual(native["model"], "qwen3.8-max")
        self.assertEqual(
            native["input"]["messages"][0]["content"],
            [
                {"text": "Что на фото?"},
                {"image": "data:image/png;base64,AAA"},
            ],
        )
        parameters = native["parameters"]
        self.assertEqual(parameters["result_format"], "message")
        self.assertTrue(parameters["incremental_output"])
        self.assertTrue(parameters["enable_thinking"])
        self.assertEqual(parameters["thinking_budget"], 1024)
        self.assertEqual(parameters["tools"], payload["tools"])
        self.assertEqual(parameters["tool_choice"], payload["tool_choice"])
        self.assertNotIn("stream_options", native)
        self.assertNotIn("messages", native)

    def test_assistant_tool_call_arguments_become_a_json_string(self):
        native = _native_payload({
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "search_web",
                                "arguments": {"query": "x"},
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": "result",
                },
            ],
        })
        assistant = native["input"]["messages"][0]
        self.assertEqual(assistant["content"], "")
        self.assertEqual(
            assistant["tool_calls"][0]["function"]["arguments"],
            '{"query": "x"}',
        )
        self.assertEqual(
            native["input"]["messages"][1],
            {"role": "tool", "tool_call_id": "call_1", "content": "result"},
        )


class NativeResponseTests(unittest.TestCase):
    def test_native_choices_are_wrapped_as_openai_choices(self):
        native = {
            "output": {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": "Привет",
                        },
                    }
                ]
            },
            "usage": {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
        }
        converted = _native_response(native)
        self.assertEqual(converted["choices"][0]["message"]["content"], "Привет")
        self.assertEqual(converted["usage"]["total_tokens"], 3)

    def test_text_result_format_is_supported(self):
        converted = _native_response({
            "output": {"text": "Ответ", "finish_reason": "stop"},
        })
        self.assertEqual(converted["choices"][0]["message"]["content"], "Ответ")

    def test_tool_call_without_content_gets_an_empty_string(self):
        converted = _native_response({
            "output": {
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "search_web",
                                        "arguments": "{}",
                                    },
                                }
                            ],
                        },
                    }
                ]
            }
        })
        message = converted["choices"][0]["message"]
        self.assertEqual(message["content"], "")
        self.assertEqual(message["tool_calls"][0]["id"], "call_1")


class NativeRouteTests(unittest.TestCase):
    def test_vision_models_use_multimodal_and_text_models_use_text_endpoint(self):
        client = AIClient(api_key="test")
        client.api_type = "dashscope"
        catalog = [
            {
                "id": "vision",
                "name": "Vision",
                "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
                "model": "qwen3.8-max",
                "vision": True,
                "api_key": "key",
            },
            {
                "id": "text",
                "name": "Text",
                "base_url": "https://dashscope-intl.aliyuncs.com/api/v1",
                "model": "qwen-plus",
                "vision": False,
                "api_key": "key",
            },
        ]
        with (
            patch.object(config, "ai_models", catalog),
            patch.object(config, "ai_fallback_models", []),
        ):
            vision_url, _ = client._route(
                "vision", native=True, payload={"messages": []}
            )
            text_url, _ = client._route(
                "text", native=True, payload={"messages": []}
            )
            image_url, _ = client._route(
                "text",
                native=True,
                payload={
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": "data:image/png;base64,AA"},
                                }
                            ],
                        }
                    ]
                },
            )
        self.assertTrue(
            vision_url.endswith(
                "/api/v1/services/aigc/multimodal-generation/generation"
            )
        )
        self.assertTrue(
            text_url.endswith(
                "/api/v1/services/aigc/text-generation/generation"
            )
        )
        self.assertTrue(
            image_url.endswith(
                "/api/v1/services/aigc/multimodal-generation/generation"
            )
        )

    def test_compatible_mode_keeps_chat_completions_url(self):
        client = AIClient(api_key="test")
        client.api_type = "openai"
        url, _ = client._route("qwen3.8-max", payload={"messages": []})
        self.assertTrue(url.endswith("/compatible-mode/v1/chat/completions"))

    def test_protocol_is_selected_per_model(self):
        client = AIClient(api_key="test")
        catalog = [
            {
                "id": "native",
                "name": "Native",
                "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
                "model": "qwen3.8-max",
                "vision": True,
                "api_key": "key",
                "api_type": "dashscope",
            },
            {
                "id": "openai",
                "name": "OpenAI",
                "base_url": "https://dashscope-intl.aliyuncs.com/api/v1",
                "model": "other-model",
                "vision": False,
                "api_key": "key",
                "api_type": "openai",
            },
        ]
        with (
            patch.object(config, "ai_models", catalog),
            patch.object(config, "ai_fallback_models", []),
        ):
            self.assertTrue(client._uses_native("native"))
            self.assertFalse(client._uses_native("openai"))
            native_url, _ = client._route(
                "native", native=True, payload={"messages": []}
            )
            openai_url, _ = client._route("openai", payload={"messages": []})
        self.assertIn("/api/v1/services/aigc/", native_url)
        self.assertTrue(openai_url.endswith("/compatible-mode/v1/chat/completions"))


class ModelListTests(unittest.IsolatedAsyncioTestCase):
    async def test_native_base_still_lists_models_via_compatible_endpoint(self):
        client = AIClient(
            api_key="test",
            base_url="https://dashscope-intl.aliyuncs.com/api/v1",
        )
        context = _json_response({"data": [{"id": "qwen3.8-max"}]})
        session = MagicMock()
        session.get.return_value = context
        with patch.object(client, "_get_session", AsyncMock(return_value=session)):
            ids = await client.list_models()
        self.assertEqual(ids, {"qwen3.8-max"})
        self.assertEqual(
            session.get.call_args.args[0],
            "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/models",
        )


class NativeRequestTests(unittest.IsolatedAsyncioTestCase):
    async def test_post_chat_sends_native_envelope(self):
        client = AIClient(api_key="test")
        client.api_type = "dashscope"
        context = _json_response({
            "output": {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "ok"},
                    }
                ]
            },
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        })
        session = MagicMock()
        session.post.return_value = context
        payload = {
            "model": "qwen3.8-max",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.2,
            "max_tokens": 10,
        }
        with patch.object(
            client, "_get_session", AsyncMock(return_value=session)
        ):
            status, data, _ = await client._post_chat(
                payload, retries=1, model_id="qwen3.8-max"
            )
        self.assertEqual(status, 200)
        self.assertEqual(data["choices"][0]["message"]["content"], "ok")
        url = session.post.call_args.args[0]
        self.assertIn("/api/v1/services/aigc/", url)
        sent = session.post.call_args.kwargs["json"]
        self.assertIn("input", sent)
        self.assertEqual(sent["input"]["messages"][0]["content"], "hi")
        self.assertEqual(sent["parameters"]["max_tokens"], 10)
        self.assertNotIn("messages", sent)

    async def test_compatible_mode_sends_the_original_payload(self):
        client = AIClient(api_key="test")
        context = _json_response({
            "choices": [{"message": {"role": "assistant", "content": "ok"}}]
        })
        session = MagicMock()
        session.post.return_value = context
        payload = {
            "model": "qwen3.8-max",
            "messages": [{"role": "user", "content": "hi"}],
        }
        catalog = [{
            "id": "qwen3.8-max",
            "name": "Qwen",
            "base_url": config.ai_base_url,
            "model": "qwen3.8-max",
            "vision": True,
            "api_key": "test",
            "api_type": "openai",
        }]
        with (
            patch.object(config, "ai_models", catalog),
            patch.object(
                client, "_get_session", AsyncMock(return_value=session)
            ),
        ):
            status, data, _ = await client._post_chat(
                payload, retries=1, model_id="qwen3.8-max"
            )
        self.assertEqual(status, 200)
        self.assertEqual(data["choices"][0]["message"]["content"], "ok")
        url = session.post.call_args.args[0]
        self.assertTrue(url.endswith("/compatible-mode/v1/chat/completions"))
        self.assertEqual(session.post.call_args.kwargs["json"], payload)


class NativeStreamTests(unittest.IsolatedAsyncioTestCase):
    async def test_native_events_are_translated_and_terminated(self):
        client = AIClient(api_key="test")
        client.api_type = "dashscope"
        events = [
            {
                "output": {
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "Hel"},
                            "finish_reason": "null",
                        }
                    ]
                },
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
            {
                "output": {
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "lo"},
                            "finish_reason": None,
                        }
                    ]
                }
            },
            {
                "output": {
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": ""},
                            "finish_reason": "stop",
                        }
                    ]
                },
                "usage": {
                    "input_tokens": 1,
                    "output_tokens": 2,
                    "total_tokens": 3,
                },
            },
        ]
        chunks = [
            ("data: " + json.dumps(event) + "\n").encode()
            for event in events
        ]
        context = _stream_response(chunks)
        session = MagicMock()
        session.post.return_value = context
        with patch.object(
            client, "_get_session", AsyncMock(return_value=session)
        ):
            lines = [
                line
                async for line in client._stream_lines(
                    {
                        "model": "qwen3.8-max",
                        "messages": [{"role": "user", "content": "hi"}],
                        "stream": True,
                    },
                    "qwen3.8-max",
                )
            ]
        self.assertEqual(lines[-1], "data: [DONE]")
        self.assertEqual(session.post.call_args.kwargs["headers"]["X-DashScope-SSE"], "enable")
        sent = session.post.call_args.kwargs["json"]
        self.assertTrue(sent["parameters"]["incremental_output"])
        parsed = [
            json.loads(line[len("data:"):].strip())
            for line in lines
            if line.startswith("data:") and line != "data: [DONE]"
        ]
        self.assertEqual(parsed[0]["choices"][0]["delta"]["content"], "Hel")
        self.assertEqual(parsed[1]["choices"][0]["delta"]["content"], "lo")
        self.assertEqual(parsed[-1]["usage"]["total_tokens"], 3)

    async def test_native_tool_call_fragments_are_forwarded(self):
        client = AIClient(api_key="test")
        chunks = [
            (
                "data: "
                + json.dumps({
                    "output": {
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": "call_1",
                                            "function": {
                                                "name": "search_web",
                                                "arguments": '{"query":',
                                            },
                                        }
                                    ],
                                },
                                "finish_reason": None,
                            }
                        ]
                    }
                })
                + "\n"
            ).encode(),
            (
                "data: "
                + json.dumps({
                    "output": {
                        "choices": [
                            {
                                "message": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "function": {"arguments": ' "x"}'},
                                        }
                                    ],
                                },
                                "finish_reason": "tool_calls",
                            }
                        ]
                    }
                })
                + "\n"
            ).encode(),
        ]
        lines = [
            line async for line in client._native_stream_lines(_AsyncContent(chunks))
        ]
        parsed = [
            json.loads(line[len("data:"):].strip())
            for line in lines
            if line.startswith("data:") and line != "data: [DONE]"
        ]
        calls = parsed[0]["choices"][0]["delta"]["tool_calls"]
        self.assertEqual(calls[0]["index"], 0)
        self.assertEqual(calls[0]["function"]["name"], "search_web")
        self.assertEqual(calls[0]["function"]["arguments"], '{"query":')
        self.assertEqual(parsed[1]["choices"][0]["delta"]["tool_calls"][0]["index"], 0)
        self.assertEqual(lines[-1], "data: [DONE]")


if __name__ == "__main__":
    unittest.main()
