"""Error log: path resolution, level filtering and rotation setup."""

from __future__ import annotations

import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bot import logging_setup
from bot.config import config


class ErrorLogSetupTests(unittest.TestCase):
    def _close_handlers(self) -> None:
        root = logging.getLogger()
        for handler in list(root.handlers):
            if getattr(handler, "_kapum_log_file", None) is not None:
                root.removeHandler(handler)
                handler.close()

    def tearDown(self) -> None:
        self._close_handlers()

    def test_auto_writes_next_to_the_database(self):
        with tempfile.TemporaryDirectory() as tmp:
            database = Path(tmp) / "chat_bot.db"
            with (
                patch.object(
                    config,
                    "database_url",
                    f"sqlite+aiosqlite:///{database.as_posix()}",
                ),
                patch.object(config, "log_file", "auto"),
                patch.object(config, "log_file_level", "WARNING"),
            ):
                path = logging_setup.setup_file_logging()
                self.assertEqual(path, Path(tmp) / "errors.log")
                logging.getLogger("test.error_log").warning("boom")
                for handler in logging.getLogger().handlers:
                    handler.flush()
                text = (Path(tmp) / "errors.log").read_text(encoding="utf-8")
            self._close_handlers()
        self.assertIn("boom", text)
        self.assertIn("WARNING", text)

    def test_off_and_auto_without_absolute_database_disable_the_file(self):
        with (
            patch.object(config, "log_file", "off"),
            patch.object(config, "database_url", "sqlite+aiosqlite:///:memory:"),
        ):
            self.assertIsNone(logging_setup.setup_file_logging())
        with (
            patch.object(config, "log_file", "auto"),
            patch.object(config, "database_url", "sqlite+aiosqlite:///chat_bot.db"),
        ):
            self.assertIsNone(logging_setup.setup_file_logging())

    def test_explicit_path_overrides_the_database_location(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "nested" / "bot.log"
            with (
                patch.object(config, "log_file", str(target)),
                patch.object(config, "log_file_level", "ERROR"),
            ):
                path = logging_setup.setup_file_logging()
                self.assertEqual(path, target)
                logging.getLogger("test.error_log").warning("hidden")
                logging.getLogger("test.error_log").error("shown")
                for handler in logging.getLogger().handlers:
                    handler.flush()
                text = target.read_text(encoding="utf-8")
            self._close_handlers()
        self.assertIn("shown", text)
        self.assertNotIn("hidden", text)
