"""PDF OCR and YouTube transcripts: parsing, track choice, caps and wiring."""

from __future__ import annotations

import json
import unittest
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from bot.config import config
from bot.services import pdf_ocr, youtube
from bot.services.ai_client import ChatResult


class VideoIdTests(unittest.TestCase):
    def test_accepted_forms(self):
        vid = "dQw4w9WgXcQ"
        for url in (
            f"https://www.youtube.com/watch?v={vid}",
            f"https://youtube.com/watch?feature=share&v={vid}&t=10",
            f"https://m.youtube.com/watch?v={vid}",
            f"https://youtu.be/{vid}?si=abc",
            f"https://www.youtube.com/shorts/{vid}",
            f"https://www.youtube.com/live/{vid}",
        ):
            self.assertEqual(youtube.youtube_video_id(url), vid, url)

    def test_rejected_forms(self):
        for url in (
            "https://evil.com/watch?v=dQw4w9WgXcQ",
            "https://youtube.com.evil.com/watch?v=dQw4w9WgXcQ",
            "https://www.youtube.com/watch?v=short",
            "https://www.youtube.com/playlist?list=PL123",
            "https://www.youtube.com/watch?v=dQw4w9WgXcQxx",
            "ftp://youtu.be/dQw4w9WgXcQ",
        ):
            self.assertIsNone(youtube.youtube_video_id(url), url)


class CaptionParsingTests(unittest.TestCase):
    def test_json3_groups_by_minute(self):
        raw = json.dumps(
            {
                "events": [
                    {"tStartMs": 0, "segs": [{"utf8": "Hello "}, {"utf8": "world"}]},
                    {"tStartMs": 5000, "segs": [{"utf8": "\n"}]},
                    {"tStartMs": 20000, "segs": [{"utf8": "second  line"}]},
                    {"tStartMs": 61000, "segs": [{"utf8": "next minute"}]},
                    {"tStartMs": 70000},
                    {"tStartMs": 3725000, "segs": [{"utf8": "later"}]},
                ]
            }
        )
        self.assertEqual(
            youtube.parse_json3(raw),
            "[0:00] Hello world second line\n[1:01] next minute\n[1:02:05] later",
        )

    def test_track_choice(self):
        def fmt(lang):
            return [{"ext": "json3", "url": f"u/{lang}"}]

        info = {
            "language": "de",
            "subtitles": {"de": fmt("de"), "en": fmt("en")},
            "automatic_captions": {"ru": fmt("ru"), "ru-orig": fmt("ru-orig")},
        }
        lang, auto, _ = youtube._pick_track(info, ["ru", "en"])
        self.assertEqual((lang, auto), ("en", False))  # manual beats automatic

        info["subtitles"] = {}
        lang, auto, _ = youtube._pick_track(info, ["fr", "ru"])
        self.assertEqual((lang, auto), ("ru", True))

        info["automatic_captions"] = {"xx-orig": fmt("xx-orig")}
        self.assertEqual(youtube._pick_track(info, ["ru"])[0], "xx-orig")
        self.assertIsNone(youtube._pick_track({}, ["ru"]))
        # a track without a json3 format is unusable
        self.assertIsNone(
            youtube._pick_track({"subtitles": {"ru": [{"ext": "vtt", "url": "x"}]}}, ["ru"])
        )


class TranscriptFlowTests(unittest.IsolatedAsyncioTestCase):
    async def test_errors_become_user_messages(self):
        with (
            patch.object(youtube, "_load_transcript", side_effect=youtube.YouTubeError("no")),
            self.assertRaises(youtube.YouTubeError),
        ):
            await youtube.fetch_transcript("dQw4w9WgXcQ")

    async def test_timeout(self):
        import time

        with (
            patch.object(youtube, "_load_transcript", lambda _id: time.sleep(0.3)),
            patch.object(config, "youtube_timeout", 0.05),
            self.assertRaises(youtube.YouTubeError) as ctx,
        ):
            await youtube.fetch_transcript("dQw4w9WgXcQ")
        self.assertIn("too long", str(ctx.exception))


def _blank_pdf(pages: int) -> bytes:
    from PIL import Image

    image = Image.new("RGB", (200, 280), "white")
    buffer = BytesIO()
    image.save(buffer, format="PDF", save_all=True, append_images=[image] * (pages - 1))
    return buffer.getvalue()


class PdfOcrTests(unittest.IsolatedAsyncioTestCase):
    async def test_page_cap_and_empty_pages(self):
        answers = iter(["First page text", "EMPTY", "Third page"])

        async def fake(prompt, image, messages, **kwargs):
            self.assertEqual(kwargs["temperature"], 0)
            self.assertTrue(image.startswith(b"\xff\xd8"))  # JPEG
            return ChatResult(content=next(answers))

        fake_client = SimpleNamespace(chat_with_image_result=AsyncMock(side_effect=fake))
        with (
            patch.object(pdf_ocr, "ai_client", fake_client),
            patch.object(config, "pdf_ocr_max_pages", 3),
        ):
            result = await pdf_ocr.ocr_pdf(_blank_pdf(7))
        self.assertEqual(fake_client.chat_with_image_result.await_count, 3)
        self.assertEqual((result.pages_read, result.total_pages), (3, None))
        self.assertTrue(result.partial)
        self.assertIn("## Page 1\nFirst page text", result.text)
        self.assertNotIn("EMPTY", result.text)
        self.assertIn("## Page 3\nThird page", result.text)

    async def test_nothing_read_returns_none(self):
        fake_client = SimpleNamespace(
            chat_with_image_result=AsyncMock(return_value=ChatResult(content=None))
        )
        with patch.object(pdf_ocr, "ai_client", fake_client):
            self.assertIsNone(await pdf_ocr.ocr_pdf(_blank_pdf(1)))
        self.assertIsNone(await pdf_ocr.ocr_pdf(b"not a pdf"))


if __name__ == "__main__":
    unittest.main()
