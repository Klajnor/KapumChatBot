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
        self.assertIn('secure_data_permissions.py', script)
        self.assertIn('--user brain "$STATE_DIR" "${DATA_GLOBS[@]}"', script)
        self.assertNotIn('chown "$OWNER" -- "$f"', script)

    def test_restart_removes_stale_root_owned_pycache(self):
        script = (ROOT / "restart.sh").read_text(encoding="utf-8")
        self.assertIn("__pycache__", script)
        self.assertIn("rm -rf", script)
        self.assertIn("systemctl daemon-reload", script)

    def test_restart_does_not_change_the_callers_shell(self):
        script = (ROOT / "restart.sh").read_text(encoding="utf-8")
        # Sourcing the script used to override INT/TERM with exit 0: Ctrl+C
        # closed the SSH session. It must be run as a separate process.
        self.assertNotIn("trap ", script)
        self.assertNotIn("exit 0", script)
        self.assertIn("BASH_SOURCE[0]", script)
        self.assertIn("journalctl", script)

    def test_no_pycache_cleanup_runs_as_the_service_user(self):
        unit = (ROOT / "chat_bot.service").read_text(encoding="utf-8")
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        # root-owned code cannot be cleaned by brain; the unit must not try.
        self.assertNotIn("ExecStartPre", unit)
        self.assertIn("PYTHONDONTWRITEBYTECODE=1", unit)
        self.assertIn("clean_pycache", readme)
        self.assertIn("root:root", readme)
        self.assertIn("root:brain", readme)
