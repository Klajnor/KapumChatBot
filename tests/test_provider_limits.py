"""Provider responses are read under explicit size limits."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from bot.services import ai_client


class ProviderBodyLimitTests(unittest.IsolatedAsyncioTestCase):
    async def test_response_is_capped_and_not_read_past_the_limit(self):
        read = AsyncMock(return_value=b"abcdef")
        response = SimpleNamespace(
            content=SimpleNamespace(read=read), charset="utf-8"
        )
        text = await ai_client._read_response_text(response, 4)
        self.assertEqual(text, "abcd")
        read.assert_awaited_once_with(5)

    async def test_oversized_sse_line_is_rejected(self):
        async def chunks():
            yield b"x" * (ai_client._MAX_SSE_BUFFER + 1)

        with self.assertRaises(ai_client.AIStreamError):
            async for _line in ai_client._sse_lines(chunks()):
                pass


if __name__ == "__main__":
    unittest.main()
