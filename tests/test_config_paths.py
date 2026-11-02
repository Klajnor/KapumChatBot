"""Regression tests: every configurable filesystem path must be absolute."""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import bot.config as bot_config
from bot.config import (
    Config,
    _load_env_file,
    _platform_env_files,
)


def _sqlite_url(path: Path) -> str:
    """SQLAlchemy URL for an absolute SQLite path on the current platform."""
    return f"sqlite+aiosqlite:///{path.as_posix()}"


class ConfigAbsolutePathTests(unittest.TestCase):
    def setUp(self):
        self.temp = Path(tempfile.mkdtemp(prefix="kapum-paths-"))
        self.addCleanup(shutil.rmtree, self.temp, True)
        self.env = {
            "BOT_TOKEN": "123456:test-token",
            "AI_API_KEY": "test-key",
            "AI_BASE_URL": "https://example.invalid/v1",
            "OWNER_IDS": "1",
            "DATABASE_URL": _sqlite_url(self.temp / "chat_bot.db"),
            "STICKER_MODEL_DIR": str(self.temp / "rembg"),
            "TELEGRAPH_ACCOUNT_FILE": str(self.temp / "telegraph_account.json"),
            "MCP_CONFIG_FILE": "",
            "MCP_SERVERS": "",
        }

    def _validate(self, **overrides):
        env = {**self.env, **overrides}
        with patch.dict(os.environ, env, clear=False):
            cfg = Config()
            cfg.validate()

    def test_absolute_paths_are_accepted(self):
        self._validate()

    def test_platform_env_file_order(self):
        for platform, expected in (
            ("nt", (".env.windows", ".env")),
            ("posix", (".env.linux", ".env")),
        ):
            with self.subTest(platform=platform), patch.object(os, "name", platform):
                self.assertEqual(_platform_env_files(), expected)

    def test_platform_specific_env_file_wins(self):
        specific = ".env.windows" if os.name == "nt" else ".env.linux"
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / specific).write_text("SPECIFIC=1", encoding="utf-8")
            (root / ".env").write_text("BASE=1", encoding="utf-8")
            previous = os.getcwd()
            os.chdir(root)
            try:
                with patch.object(bot_config, "load_dotenv") as load:
                    self.assertEqual(_load_env_file(), specific)
            finally:
                os.chdir(previous)
        load.assert_called_once_with(specific, override=False)

    def test_env_fallback_when_platform_file_is_absent(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / ".env").write_text("BASE=1", encoding="utf-8")
            previous = os.getcwd()
            os.chdir(root)
            try:
                with patch.object(bot_config, "load_dotenv") as load:
                    self.assertEqual(_load_env_file(), ".env")
            finally:
                os.chdir(previous)
        load.assert_called_once_with(".env", override=False)

    def test_in_memory_sqlite_is_accepted(self):
        self._validate(DATABASE_URL="sqlite+aiosqlite://")

    def test_postgres_url_is_not_a_filesystem_path(self):
        self._validate(DATABASE_URL="postgresql+asyncpg://u:p@localhost/db")

    def test_relative_sqlite_path_is_rejected(self):
        with self.assertRaises(ValueError) as caught:
            self._validate(DATABASE_URL="sqlite+aiosqlite:///chat_bot.db")
        self.assertIn("DATABASE_URL", str(caught.exception))

    def test_relative_sticker_directory_is_rejected(self):
        with self.assertRaises(ValueError) as caught:
            self._validate(STICKER_MODEL_DIR="models/rembg")
        self.assertIn("STICKER_MODEL_DIR", str(caught.exception))

    def test_relative_telegraph_file_is_rejected(self):
        with self.assertRaises(ValueError) as caught:
            self._validate(TELEGRAPH_ACCOUNT_FILE="telegraph_account.json")
        self.assertIn("TELEGRAPH_ACCOUNT_FILE", str(caught.exception))

    def test_relative_mcp_config_is_rejected(self):
        with self.assertRaises(ValueError) as caught:
            self._validate(MCP_CONFIG_FILE=".mcp.json")
        self.assertIn("MCP_CONFIG_FILE", str(caught.exception))

    def test_home_relative_paths_are_expanded_and_accepted(self):
        self._validate(
            STICKER_MODEL_DIR="~/rembg",
            TELEGRAPH_ACCOUNT_FILE="~/telegraph_account.json",
            MCP_CONFIG_FILE="~/.mcp.json",
        )

    def test_error_names_every_relative_path(self):
        with self.assertRaises(ValueError) as caught:
            self._validate(
                DATABASE_URL="sqlite+aiosqlite:///chat_bot.db",
                STICKER_MODEL_DIR="models/rembg",
                TELEGRAPH_ACCOUNT_FILE="telegraph_account.json",
                MCP_CONFIG_FILE=".mcp.json",
            )
        message = str(caught.exception)
        for name in (
            "DATABASE_URL",
            "STICKER_MODEL_DIR",
            "TELEGRAPH_ACCOUNT_FILE",
            "MCP_CONFIG_FILE",
        ):
            self.assertIn(name, message)


if __name__ == "__main__":
    unittest.main()
