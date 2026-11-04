"""Provider responses are read under explicit size limits."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from bot.services import ai_client
from bot.services.http_body import read_bounded


class ProviderBodyLimitTests(unittest.IsolatedAsyncioTestCase):
    async def test_fragmented_json_and_utf8_are_read_to_eof(self):
        payload = '{"answer": "Привет"}'.encode()
        stream = SimpleNamespace(read=AsyncMock(side_effect=[
            payload[:13], payload[13:14], payload[14:], b"",
        ]))
        response = SimpleNamespace(content=stream, charset="utf-8")
        self.assertEqual(await ai_client._read_response_text(response, 100), payload.decode())

    async def test_fragmented_body_stops_at_overflow_before_eof(self):
        stream = SimpleNamespace(read=AsyncMock(side_effect=[b"ab", b"cd", b"e"]))
        self.assertEqual(await read_bounded(stream, 4), b"abcde")
        self.assertEqual([call.args[0] for call in stream.read.await_args_list], [5, 3, 1])

    async def test_exact_limit_requires_eof_check(self):
        stream = SimpleNamespace(read=AsyncMock(side_effect=[b"abcd", b""]))
        self.assertEqual(await read_bounded(stream, 4), b"abcd")
        self.assertEqual(stream.read.await_count, 2)

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
