"""The Telegraph account token is written atomically with 0600 permissions."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from bot.services import telegraph


class TelegraphAccountFileTests(unittest.TestCase):
    def test_account_file_is_replaced_atomically(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "telegraph_account.json"
            path.write_text('{"old": true}', encoding="utf-8")
            telegraph._write_account_file(path, "short", "secret")
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(
                data, {"short_name": "short", "access_token": "secret"}
            )
            leftovers = [p.name for p in Path(temp).iterdir() if p.name != path.name]
            self.assertEqual(leftovers, [])
            if os.name != "nt":
                self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_existing_symlink_is_replaced_not_followed(self):
        if os.name == "nt":
            self.skipTest("symlinks need privileges on Windows")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            target = root / "target.json"
            target.write_text("original", encoding="utf-8")
            link = root / "telegraph_account.json"
            link.symlink_to(target)
            telegraph._write_account_file(link, "short", "secret")
            self.assertFalse(link.is_symlink())
            self.assertEqual(target.read_text(encoding="utf-8"), "original")
            self.assertIn("access_token", link.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
