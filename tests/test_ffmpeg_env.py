"""ffmpeg must not inherit the bot's secrets."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from bot.services import ai_client


class FfmpegEnvironmentTests(unittest.TestCase):
    def test_bot_secrets_are_not_inherited(self):
        with patch.dict(
            os.environ,
            {
                "BOT_TOKEN": "secret-token",
                "AI_API_KEY": "secret-key",
                "DATABASE_URL": "postgresql://u:p@host/db",
                "PATH": os.environ.get("PATH", ""),
                "TZ": "UTC",
            },
            clear=False,
        ):
            env = ai_client._ffmpeg_environment()
        self.assertNotIn("BOT_TOKEN", env)
        self.assertNotIn("AI_API_KEY", env)
        self.assertNotIn("DATABASE_URL", env)
        self.assertIn("PATH", env)
        self.assertEqual(env.get("TZ"), "UTC")


if __name__ == "__main__":
    unittest.main()
