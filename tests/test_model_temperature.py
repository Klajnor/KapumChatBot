"""Per-model temperature overrides the caller/user value."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from bot.config import Config, config
from bot.services.ai_client import AIClient


class ModelTemperatureConfigTests(unittest.TestCase):
    def test_temperature_is_read_from_the_model_set(self):
        env = {
            "MODEL_1_ID": "temp-model",
            "MODEL_1_MODEL": "provider-model",
            "MODEL_1_TEMPERATURE": "1.0",
            "MODEL_2_ID": "default-model",
            "MODEL_2_MODEL": "provider-model-2",
            "MODEL_2_TEMPERATURE": "",
        }
        with patch.dict(os.environ, env):
            settings = Config()
        by_id = {spec["id"]: spec for spec in settings.ai_models}
        self.assertEqual(by_id["temp-model"]["temperature"], 1.0)
        self.assertIsNone(by_id["default-model"]["temperature"])

    def test_invalid_temperature_uses_default_and_warns(self):
        env = {
            "MODEL_1_ID": "bad-temp",
            "MODEL_1_MODEL": "provider-model",
            "MODEL_1_TEMPERATURE": "3.5",
        }
        with (
            patch.dict(os.environ, env),
            self.assertLogs("bot.config", level="WARNING") as logs,
        ):
            settings = Config()
        self.assertIsNone(settings.ai_models[0]["temperature"])
        self.assertIn("out of range", "\n".join(logs.output))


class ModelTemperaturePayloadTests(unittest.TestCase):
    def test_model_value_overrides_caller_temperature(self):
        client = AIClient(api_key="test")
        catalog = [{
            "id": "temp-model",
            "name": "Temp",
            "base_url": "https://example.test/v1",
            "model": "provider-model",
            "vision": False,
            "api_key": "key",
            "api_type": "openai",
            "web_search": "false",
            "temperature": 1.0,
        }]
        with (
            patch.object(config, "ai_models", catalog),
            patch.object(config, "ai_fallback_models", []),
        ):
            payload = client._chat_payload(
                [{"role": "user", "content": "hi"}],
                "temp-model",
                0.2,
                100,
            )
        self.assertEqual(payload["temperature"], 1.0)

    def test_empty_model_value_keeps_caller_temperature(self):
        client = AIClient(api_key="test")
        catalog = [{
            "id": "plain-model",
            "name": "Plain",
            "base_url": "https://example.test/v1",
            "model": "provider-model",
            "vision": False,
            "api_key": "key",
            "api_type": "openai",
            "web_search": "false",
            "temperature": None,
        }]
        with (
            patch.object(config, "ai_models", catalog),
            patch.object(config, "ai_fallback_models", []),
        ):
            payload = client._chat_payload(
                [{"role": "user", "content": "hi"}],
                "plain-model",
                0.2,
                100,
            )
        self.assertEqual(payload["temperature"], 0.2)


if __name__ == "__main__":
    unittest.main()
