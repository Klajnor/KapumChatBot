"""`/clear` also drops in-memory question and pending-photo state."""

from __future__ import annotations

import time
import unittest
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from bot.handlers import chat, clear


@asynccontextmanager
async def _service():
    yield SimpleNamespace(clear_history=AsyncMock())


class ClearStateTests(unittest.IsolatedAsyncioTestCase):
    async def test_clear_drops_memory_state(self):
        callback = SimpleNamespace(
            from_user=SimpleNamespace(id=1),
            message=SimpleNamespace(
                chat=SimpleNamespace(id=1, type="private")
            ),
        )
        chat._last_questions[1] = "old question"
        chat._pending_enhance[1] = {
            "expires": time.monotonic() + 60,
            "file_ids": ["f"],
        }
        with (
            patch.object(clear, "user_service", _service),
            patch.object(clear, "safe_edit_text", AsyncMock()),
            patch.object(clear, "safe_answer_callback", AsyncMock()),
        ):
            await clear.cb_clear_confirm(callback)

        self.assertNotIn(1, chat._last_questions)
        self.assertNotIn(1, chat._pending_enhance)


if __name__ == "__main__":
    unittest.main()
