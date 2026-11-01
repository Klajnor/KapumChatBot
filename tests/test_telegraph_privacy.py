"""Private answers must not be published to Telegraph."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from bot.config import config
from bot.handlers import chat, guest


class TelegraphPrivacyTests(unittest.IsolatedAsyncioTestCase):
    def _message(self):
        return SimpleNamespace(chat=SimpleNamespace(id=1, type="private"))

    async def test_deliver_answer_can_forbid_telegraph(self):
        with (
            patch.object(config, "rich_messages", False),
            patch.object(config, "render_formula_images", False),
            patch.object(config, "enable_telegraph", True),
            patch.object(chat, "needs_telegraph", return_value=True),
            patch.object(chat, "publish_answer", AsyncMock()) as publish,
            patch.object(chat, "send_long_message", AsyncMock(return_value=True)),
        ):
            delivered = await chat.deliver_answer(
                self._message(), None, "text", "", allow_telegraph=False
            )
        self.assertTrue(delivered)
        publish.assert_not_awaited()

    async def test_guest_reply_skips_telegraph_for_external_context(self):
        with (
            patch.object(config, "rich_messages", False),
            patch.object(config, "enable_telegraph", True),
            patch.object(guest, "needs_telegraph", return_value=True),
            patch.object(guest, "publish_answer", AsyncMock()) as publish,
        ):
            result = await guest._maybe_telegraph_reply(
                "q", "answer", "", allow=False
            )
        self.assertIsNone(result)
        publish.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
