from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from bot.config import config
from bot.handlers import chat
from bot.services.outgoing_files import prepare_files, safe_filename


def _args(*names: str, content: str = "data") -> dict:
    return {"files": [{"filename": n, "content": content} for n in names]}


class SafeFilenameTests(unittest.TestCase):
    def test_allowed_extensions(self) -> None:
        for name in ("a.json", "B.XML", "n.txt", "r.md", "t.csv", "c.yaml"):
            self.assertIsNotNone(safe_filename(name), name)

    def test_executables_and_active_content_rejected(self) -> None:
        for name in (
            "a.exe", "a.bat", "a.sh", "a.ps1", "a.py", "a.js", "a.vbs",
            "a.html", "a.svg", "a.zip", "a.lnk", "a.jar", "noext", "",
            "a.txt.exe", "a.exe ", "a.exe.",
        ):
            self.assertIsNone(safe_filename(name), name)

    def test_path_and_double_extension_sanitized(self) -> None:
        self.assertEqual(safe_filename("../../etc/x.txt"), "x.txt")
        self.assertEqual(safe_filename("C:\\dir\\y.md"), "y.md")
        self.assertEqual(safe_filename("report.exe.txt"), "report_exe.txt")

    def test_bidi_override_stripped(self) -> None:
        # U+202E разворачивает отображение: «txt.exe» не должно проскочить.
        self.assertIsNone(safe_filename("a\u202etxt.exe"))


class PrepareFilesTests(unittest.TestCase):
    def test_limit_keeps_first_ten_and_counts_rest(self) -> None:
        names = [f"f{i}.txt" for i in range(13)]
        prepared = prepare_files(_args(*names))
        self.assertEqual(len(prepared.files), 10)
        self.assertEqual(prepared.over_limit, 3)
        self.assertEqual(prepared.files[0][0], "f0.txt")

    def test_duplicate_names_made_unique(self) -> None:
        prepared = prepare_files(_args("a.txt", "a.txt", "A.txt"))
        self.assertEqual(
            [n for n, _ in prepared.files], ["a.txt", "a_2.txt", "A_3.txt"]
        )

    def test_invalid_entries_are_skipped(self) -> None:
        arguments = {
            "files": [
                {"filename": "ok.json", "content": "{}"},
                {"filename": "bad.exe", "content": "x"},
                {"filename": "empty.txt", "content": "  "},
                {"filename": "nul.txt", "content": "a\x00b"},
                {"filename": "num.txt", "content": 5},
                "junk",
            ]
        }
        prepared = prepare_files(arguments)
        self.assertEqual([n for n, _ in prepared.files], ["ok.json"])
        self.assertEqual(len(prepared.rejected), 4)

    def test_size_limit(self) -> None:
        with patch.object(config, "reply_file_max_bytes", 10):
            prepared = prepare_files(_args("a.txt", content="x" * 11))
        self.assertEqual(prepared.files, [])

    def test_malformed_arguments(self) -> None:
        self.assertEqual(prepare_files({}).files, [])
        self.assertEqual(prepare_files({"files": "a.txt"}).files, [])


class SendReplyFilesTests(unittest.IsolatedAsyncioTestCase):
    def _message(self) -> SimpleNamespace:
        return SimpleNamespace(
            chat=SimpleNamespace(id=1),
            bot=SimpleNamespace(send_chat_action=AsyncMock()),
            answer_document=AsyncMock(),
        )

    async def test_sends_each_file_and_warns_over_limit(self) -> None:
        message = self._message()
        names = [f"f{i}.txt" for i in range(12)]
        with (
            patch.object(chat, "safe_answer", AsyncMock()) as notice,
            patch.object(chat.throttle, "wait", AsyncMock()),
        ):
            ok = await chat.send_reply_files(message, _args(*names))
        self.assertTrue(ok)
        self.assertEqual(message.answer_document.await_count, 10)
        self.assertIn("limit exceeded", notice.await_args.args[1])

    async def test_nothing_safe_sends_nothing(self) -> None:
        message = self._message()
        with patch.object(chat, "safe_answer", AsyncMock()) as notice:
            ok = await chat.send_reply_files(message, _args("virus.exe"))
        self.assertFalse(ok)
        message.answer_document.assert_not_awaited()
        notice.assert_awaited_once()
