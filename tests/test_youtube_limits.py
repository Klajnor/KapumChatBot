"""YouTube: ограниченное чтение субтитров, ранняя обрезка и отмена."""

from __future__ import annotations

import json
import threading
import time
import unittest
from io import BytesIO
from unittest.mock import patch

from bot.config import config
from bot.services import youtube


class CaptionReadLimitTests(unittest.TestCase):
    def test_read_is_capped_and_reports_truncation(self):
        body, truncated = youtube._read_capped(BytesIO(b"x" * 100), 10)
        self.assertEqual(body, b"x" * 10)
        self.assertTrue(truncated)

    def test_small_body_is_not_marked_truncated(self):
        body, truncated = youtube._read_capped(BytesIO(b"abcd"), 10)
        self.assertEqual(body, b"abcd")
        self.assertFalse(truncated)

    def test_cancel_event_stops_the_reader(self):
        event = threading.Event()
        event.set()
        token = youtube._CANCEL.set(event)
        try:
            with self.assertRaises(youtube._Cancelled):
                youtube._read_capped(BytesIO(b"x" * 100), 10)
        finally:
            youtube._CANCEL.reset(token)

    def test_parse_stops_early_instead_of_building_the_whole_text(self):
        events = [
            {"tStartMs": i * 1000, "segs": [{"utf8": "word " * 20}]}
            for i in range(100)
        ]
        raw = json.dumps({"events": events})
        full = youtube.parse_json3(raw)
        limited = youtube.parse_json3(raw, max_chars=200)
        self.assertGreater(len(full), 1_000)
        self.assertLess(len(limited), len(full))
        self.assertTrue(limited.startswith("[0:00]"))
        # Последняя минута не должна попасть в раннюю остановку.
        self.assertNotIn("[1:39]", limited)


class TimeoutCancellationTests(unittest.IsolatedAsyncioTestCase):
    async def test_timeout_sets_the_shared_cancel_event(self):
        seen: list[threading.Event | None] = []

        def slow(_video_id: str):
            seen.append(youtube._CANCEL.get())
            time.sleep(0.2)
            return youtube.Transcript(
                video_id="", title="", channel="", duration=0,
                language="en", automatic=True, text="late",
            )

        with (
            patch.object(youtube, "_load_transcript", slow),
            patch.object(config, "youtube_timeout", 0.05),
            self.assertRaises(youtube.YouTubeError) as ctx,
        ):
            await youtube.fetch_transcript("dQw4w9WgXcQ")

        self.assertIn("too long", str(ctx.exception))
        self.assertEqual(len(seen), 1)
        self.assertIsNotNone(seen[0])
        self.assertTrue(seen[0].is_set())