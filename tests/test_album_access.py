"""Album collection checks access before downloading any photo."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from bot.handlers import chat


class AlbumAccessGateTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        chat._album_images.clear()
        chat._album_total_bytes.clear()
        chat._album_download_bytes.clear()
        self.answer = patch.object(chat, "safe_answer", AsyncMock())
        self.answer.start()
        self.addCleanup(self.answer.stop)
        collect = patch.object(chat, "_ALBUM_COLLECT_SECONDS", 0)
        collect.start()
        self.addCleanup(collect.stop)

    def tearDown(self):
        chat._album_images.clear()
        chat._album_total_bytes.clear()
        chat._album_download_bytes.clear()

    def _message(self, *, user_id: int = 1, chat_type: str = "private"):
        return SimpleNamespace(
            media_group_id="group-1",
            chat=SimpleNamespace(id=user_id, type=chat_type),
            from_user=SimpleNamespace(id=user_id),
            caption=None,
            text=None,
            reply_to_message=None,
            photo=[SimpleNamespace(file_id="f1", file_size=10)],
            bot=SimpleNamespace(
                get_file=AsyncMock(),
                download_file=AsyncMock(),
            ),
            answer=AsyncMock(),
        )

    async def test_denied_user_album_is_never_downloaded(self):
        message = self._message()
        with patch.object(
            chat,
            "_album_access_allowed",
            AsyncMock(return_value=(False, "Access blocked.")),
        ):
            await chat._handle_album_photo(message)

        message.bot.get_file.assert_not_awaited()
        message.bot.download_file.assert_not_awaited()
        chat.safe_answer.assert_awaited_once()
        self.assertIn("Access blocked", chat.safe_answer.call_args.args[1])
        self.assertEqual(chat._album_images, {})

    async def test_group_album_without_mention_is_not_collected(self):
        message = self._message(chat_type="group")
        with patch.object(chat, "_handle_album_photo", AsyncMock()) as collect:
            await chat.handle_photo(message)
        collect.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
