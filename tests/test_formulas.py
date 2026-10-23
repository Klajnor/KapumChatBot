from __future__ import annotations

import io
import os
import unittest
from unittest.mock import AsyncMock, patch

os.environ["BOT_TOKEN"] = "123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghi"
os.environ["AI_API_KEY"] = "test-key"
os.environ["DATABASE_URL"] = "postgresql+asyncpg://test:test@localhost/test"

from PIL import Image

from bot.config import DEFAULT_STYLE_HINT, FORMULA_STYLE_HINT, config
from bot.handlers.guest import GuestReply, _to_result
from bot.services.ai_client import AIClient
from bot.services.formulas import MAX_FORMULA_CHARS, render_formula, split_display_math
from bot.services import telegram_utils as sending


class FormulaTests(unittest.IsolatedAsyncioTestCase):
    def test_display_math_is_extracted_in_order(self):
        text = r"Before \[\frac{1}{2}\] after $$x^2$$ end"
        self.assertEqual(split_display_math(text), [
            ("text", "Before "), ("math", r"\[\frac{1}{2}\]"),
            ("text", " after "), ("math", "$$x^2$$"), ("text", " end"),
        ])

    def test_code_and_inline_math_are_preserved(self):
        for text in (
            r"`\[x\]`", "```python\n$$x$$\n```",
            "```\n$$unfinished$$", r"Inline \(x^2\) and $5",
        ):
            with self.subTest(text=text):
                self.assertEqual(split_display_math(text), [("text", text)])

    def test_code_followed_by_display_math(self):
        text = r"`$$code$$` \[x\]"
        self.assertEqual(split_display_math(text)[-1], ("math", r"\[x\]"))

    async def test_real_fraction_and_integral_render_as_png(self):
        for expression in (r"\[\frac{-b+\sqrt{b^2-4ac}}{2a}\]",
                           r"$$\int_0^1 x^2\,dx = \frac{1}{3}$$"):
            with self.subTest(expression=expression):
                png = await render_formula(expression)
                self.assertIsNotNone(png)
                self.assertTrue(png.startswith(b"\x89PNG"))
                with Image.open(io.BytesIO(png)) as image:
                    self.assertGreater(image.width, 48)
                    self.assertGreater(image.height, 48)
                    self.assertEqual(image.getpixel((0, 0)), (255, 255, 255))

    async def test_unsupported_and_oversized_math_falls_back(self):
        self.assertIsNone(await render_formula(r"\[\unknowncommand{x}\]"))
        self.assertIsNone(await render_formula(r"\[" + "x" * MAX_FORMULA_CHARS + r"\]"))
        self.assertIsNone(await render_formula(r"\[\]"))

    async def test_output_order(self):
        calls = []
        async def text(message, body, **kwargs):
            calls.append("text:" + body.strip())
            return True
        async def photo(message, image):
            calls.append("photo")
            return True
        with (
            patch.object(config, "render_formula_images", True),
            patch.object(sending, "render_formula", AsyncMock(return_value=b"png")),
            patch.object(sending, "_send_text_formatted", side_effect=text),
            patch.object(sending, "_send_formula_photo", side_effect=photo),
        ):
            self.assertTrue(await sending.send_formatted(object(), r"Before \[x\] After"))
        self.assertEqual(calls, ["text:Before", "photo", "text:After"])

    async def test_failed_photo_uses_unicode_text(self):
        with (
            patch.object(config, "render_formula_images", True),
            patch.object(sending, "render_formula", AsyncMock(return_value=b"png")),
            patch.object(sending, "_send_formula_photo", AsyncMock(return_value=False)),
            patch.object(sending, "_send_text_formatted", AsyncMock(return_value=True)) as send,
        ):
            self.assertTrue(await sending.send_formatted(object(), r"\[\alpha\]"))
        self.assertEqual(send.call_args.args[1].strip(), "α")

    async def test_render_attempts_are_bounded(self):
        with (
            patch.object(config, "render_formula_images", True),
            patch.object(sending, "render_formula", AsyncMock(return_value=None)) as render,
            patch.object(sending, "_send_text_formatted", AsyncMock(return_value=True)),
        ):
            await sending.send_formatted(object(), " ".join([r"\[x\]"] * 10))
        self.assertEqual(render.await_count, sending.MAX_FORMULA_IMAGES)

    async def test_disabled_setting_uses_text_only(self):
        with (
            patch.object(config, "render_formula_images", False),
            patch.object(sending, "render_formula", AsyncMock()) as render,
            patch.object(sending, "_send_text_formatted", AsyncMock(return_value=True)),
        ):
            self.assertTrue(await sending.send_formatted(object(), r"\[x\]"))
        render.assert_not_awaited()

    def test_finalize_keeps_latex_for_renderer(self):
        with patch.object(config, "render_formula_images", True):
            self.assertEqual(AIClient._finalize(r"\[\frac{1}{2}\]"), r"\[\frac{1}{2}\]")

    def test_default_prompt_requests_display_math(self):
        with (
            patch.object(config, "rich_messages", False),
            patch.object(config, "render_formula_images", True),
            patch.object(config, "enable_style_hint", True),
            patch.object(config, "style_hint", DEFAULT_STYLE_HINT),
        ):
            messages = AIClient._with_style_hint([{"role": "system", "content": "Help"}])
        self.assertIn(FORMULA_STYLE_HINT, messages[0]["content"])

    def test_guest_reply_has_unicode_fallback(self):
        with patch.object(config, "render_formula_images", True):
            result = _to_result(GuestReply(text=r"\[\alpha\]"), mode="plain")
        self.assertEqual(result.input_message_content.message_text.strip(), "α")


if __name__ == "__main__":
    unittest.main()
