"""Provider-side web search is preferred over the local search_web tool."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from bot.config import Config, config
from bot.services.ai_client import AIClient, _native_payload


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


if __name__ == "__main__":
    unittest.main()
