"""Эксплуатация: restart.sh не отдаёт код сервису и совпадает с README."""

from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class RestartHardeningTests(unittest.TestCase):
    def test_code_and_venv_stay_root_owned(self):
        script = (ROOT / "restart.sh").read_text(encoding="utf-8")
        self.assertIn('chown -R root:root "$APP_DIR"', script)
        self.assertIn('chmod -R go-w "$APP_DIR"', script)
        # Прежняя ошибка: весь каталог, включая код и venv, отдавался сервису.
        self.assertNotIn('chown -R "$OWNER" "$APP_DIR"', script)

    def test_configs_are_root_owned_but_group_readable(self):
        script = (ROOT / "restart.sh").read_text(encoding="utf-8")
        self.assertIn('CONFIG_OWNER="root:brain"', script)
        self.assertIn('"$APP_DIR/.env.linux"', script)
        self.assertIn('chown "$CONFIG_OWNER" -- "$f"', script)
        self.assertIn('chmod 640 -- "$f"', script)

    def test_mutable_data_belongs_to_the_service(self):
        script = (ROOT / "restart.sh").read_text(encoding="utf-8")
        self.assertIn('OWNER="brain:brain"', script)
        self.assertIn('chown -R "$OWNER" "$STATE_DIR"', script)
        self.assertIn('chmod 700 "$STATE_DIR"', script)
        self.assertIn('chown "$OWNER" -- "$f"', script)
        self.assertIn('chmod 600 -- "$f"', script)

    def test_restart_removes_stale_root_owned_pycache(self):
        script = (ROOT / "restart.sh").read_text(encoding="utf-8")
        self.assertIn("__pycache__", script)
        self.assertIn("rm -rf", script)
        self.assertIn("systemctl daemon-reload", script)

    def test_no_pycache_cleanup_runs_as_the_service_user(self):
        unit = (ROOT / "chat_bot.service").read_text(encoding="utf-8")
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        # root-owned code cannot be cleaned by brain; the unit must not try.
        self.assertNotIn("ExecStartPre", unit)
        self.assertIn("PYTHONDONTWRITEBYTECODE=1", unit)
        self.assertIn("clean_pycache", readme)
        self.assertIn("root:root", readme)
        self.assertIn("root:brain", readme)