"""Защита привилегированного исправления прав от подмены файлов."""

from __future__ import annotations

import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.secure_data_permissions import repair


@unittest.skipUnless(os.name == "posix", "Нужны Linux/POSIX-дескрипторы")
class SecureDataPermissionsTests(unittest.TestCase):
    def test_links_do_not_change_external_targets(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            outside = root / "outside"
            outside.write_text("secret")
            outside.chmod(0o640)
            data = root / "data"
            data.mkdir()
            (data / "link").symlink_to(outside)
            os.link(outside, data / "hardlink")
            (data / "directory").symlink_to(root, target_is_directory=True)
            normal = data / "normal"
            normal.write_text("data")
            repair(str(data), os.getuid(), os.getgid())
            self.assertEqual(stat.S_IMODE(outside.stat().st_mode), 0o640)
            self.assertEqual(stat.S_IMODE(normal.stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(data.stat().st_mode), 0o700)

    def test_replacement_after_open_cannot_redirect_chmod(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            target = root / "target"
            target.write_text("secret")
            target.chmod(0o640)
            data = root / "data"
            data.write_text("data")
            original = os.fchown

            def replace(fd, uid, gid):
                data.unlink()
                data.symlink_to(target)
                original(fd, uid, gid)

            with patch("os.fchown", side_effect=replace):
                repair(str(data), os.getuid(), os.getgid())
            self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o640)
