from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from bot.config import config
from bot.handlers import chat
from bot.services.ai_client import ImageResult
from bot.services.artist_persona import (
    MAX_ARTIST_PERSONA_CHARS,
    artist_style,
    with_artist_persona,
)


class _NoWait:
    async def __aenter__(self):
        return SimpleNamespace()

    async def __aexit__(self, *exc_info):
        return False


def _no_wait(*args, **kwargs):
    return _NoWait()


class ArtistPersonaTests(unittest.TestCase):
    """Выбранная персона становится личностью художника в промпте генерации."""

    def test_empty_persona_keeps_the_prompt(self):
        self.assertEqual(with_artist_persona("кот", None), "кот")
        self.assertEqual(with_artist_persona("кот", ""), "кот")
        self.assertEqual(with_artist_persona("кот", "   \n "), "кот")
        self.assertEqual(artist_style(None), "")

    def test_default_persona_adds_nothing(self):
        """Стандартный «Ассистент» не описывает художника."""
        self.assertEqual(
            with_artist_persona("кот", config.default_system_prompt), "кот"
        )

    def test_persona_is_appended_as_artist_style(self):
        prompt = with_artist_persona(
            "  рыжий лис в лесу  ", "Ты — мрачный художник-сюрреалист."
        )
        self.assertTrue(prompt.startswith("рыжий лис в лесу\n\n"))
        self.assertIn("Artist persona", prompt)
        self.assertIn("мрачный художник-сюрреалист", prompt)
        self.assertTrue(prompt.rstrip().endswith("мрачный художник-сюрреалист."))

    def test_persona_is_whitespace_collapsed_and_bounded(self):
        style = artist_style("Ты —\n\nхудожник\tакварелист.")
        self.assertEqual(style, "Ты — художник акварелист.")
        long_style = artist_style("я" * (MAX_ARTIST_PERSONA_CHARS + 500))
        self.assertEqual(len(long_style), MAX_ARTIST_PERSONA_CHARS)


class ImagePersonaDeliveryTests(unittest.IsolatedAsyncioTestCase):
    """Промпт генерации получает персону, подпись и история — нет."""

    async def _run(self, persona):
        message = SimpleNamespace(
            chat=SimpleNamespace(id=1),
            bot=SimpleNamespace(send_chat_action=AsyncMock()),
            answer_photo=AsyncMock(),
        )
        result = ImageResult(url=None, data=b"\x89PNG", elapsed=1.0)
        with (
            patch.object(chat, "waiting", _no_wait),
            patch.object(
                chat.ai_client,
                "generate_image_result",
                AsyncMock(return_value=result),
            ) as generate,
        ):
            ok = await chat.generate_and_send_image(
                message, "рыжий лис в лесу", persona=persona
            )
        return ok, message, generate

    async def test_persona_reaches_generation_prompt_only(self):
        ok, message, generate = await self._run("Ты — мрачный художник-сюрреалист.")
        self.assertTrue(ok)
        sent_prompt = generate.await_args.args[0]
        self.assertIn("рыжий лис в лесу", sent_prompt)
        self.assertIn("мрачный художник-сюрреалист", sent_prompt)
        caption = message.answer_photo.await_args.kwargs["caption"]
        self.assertIn("рыжий лис в лесу", caption)
        self.assertNotIn("мрачный художник", caption)

    async def test_no_persona_keeps_the_original_prompt(self):
        ok, _message, generate = await self._run(None)
        self.assertTrue(ok)
        self.assertEqual(generate.await_args.args[0], "рыжий лис в лесу")


if __name__ == "__main__":
    unittest.main()