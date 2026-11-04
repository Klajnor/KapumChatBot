"""Классификатор намерений запроса: обрезка до 256 символов и маршрутизация."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from bot.services.request_intent import (
    Intent,
    classify_request,
    looks_like_status_request,
    truncate_for_intent,
)


def _client(content: str | None = "chat", error: Exception | None = None):
    mock = AsyncMock(
        return_value=SimpleNamespace(content=content), side_effect=error
    )
    return SimpleNamespace(chat_completion_result=mock)


class TestTruncate(unittest.TestCase):
    def test_short_text_unchanged(self):
        self.assertEqual(truncate_for_intent("нарисуй кота", 256), "нарисуй кота")

    def test_cuts_broken_last_word(self):
        text = "слово " * 60  # 360 символов
        out = truncate_for_intent(text + "обрыв", 256)
        self.assertLessEqual(len(out), 256)
        self.assertTrue(out.endswith("слово"))

    def test_boundary_on_space_keeps_last_word(self):
        text = "a" * 9 + " " + "b" * 10  # пробел сразу после среза по 9 символам
        self.assertEqual(truncate_for_intent(text, 9), "a" * 9)

    def test_long_word_without_spaces_is_hard_cut(self):
        self.assertEqual(truncate_for_intent("x" * 500, 256), "x" * 256)

    def test_cyrillic_limit(self):
        out = truncate_for_intent("привет мир " * 100, 256)
        self.assertLessEqual(len(out), 256)
        self.assertTrue(out.endswith(("мир", "привет")))


class TestClassify(unittest.IsolatedAsyncioTestCase):
    async def test_document_without_ai(self):
        client = _client()
        with patch("bot.services.request_intent.ai_client", client):
            intent = await classify_request("посмотри", has_document=True)
        self.assertIs(intent, Intent.DOCUMENT)
        client.chat_completion_result.assert_not_awaited()

    async def test_reminder_without_ai(self):
        client = _client()
        with patch("bot.services.request_intent.ai_client", client):
            intent = await classify_request("поставь напоминалку на завтра")
        self.assertIs(intent, Intent.REMINDER)
        client.chat_completion_result.assert_not_awaited()

    async def test_skip_reminder(self):
        client = _client()
        with patch("bot.services.request_intent.ai_client", client):
            intent = await classify_request(
                "напомни завтра", skip_reminder=True
            )
        self.assertIs(intent, Intent.CHAT)

    async def test_website_without_ai(self):
        client = _client()
        with patch("bot.services.request_intent.ai_client", client):
            intent = await classify_request("о чём этот сайт https://example.com/a")
        self.assertIs(intent, Intent.WEBSITE)
        client.chat_completion_result.assert_not_awaited()

    async def test_link_in_voice_is_not_a_website(self):
        client = _client()
        with patch("bot.services.request_intent.ai_client", client):
            intent = await classify_request(
                "о чём этот сайт https://example.com/a", from_voice=True
            )
        self.assertIs(intent, Intent.CHAT)

    async def test_ordinary_chat_does_not_call_ai(self):
        client = _client()
        with patch("bot.services.request_intent.ai_client", client):
            intent = await classify_request("почему небо синее")
        self.assertIs(intent, Intent.CHAT)
        client.chat_completion_result.assert_not_awaited()

    async def test_status_prefilter(self):
        for text in ("статус", "как дела?", "How are you?", "what's up"):
            self.assertTrue(looks_like_status_request(text), text)
        for text in ("почему небо синее", "статус проекта в другом чате"):
            # The second phrase still reaches the cheap prefilter; the AI
            # confirmation must reject it (tested below).
            if text.startswith("статус"):
                self.assertTrue(looks_like_status_request(text))
            else:
                self.assertFalse(looks_like_status_request(text))

    async def test_status_confirmed_by_model(self):
        client = _client("status")
        with patch("bot.services.request_intent.ai_client", client):
            intent = await classify_request("как дела")
        self.assertIs(intent, Intent.STATUS)

    async def test_status_words_but_model_says_chat(self):
        client = _client("chat")
        with patch("bot.services.request_intent.ai_client", client):
            intent = await classify_request("статус проекта в другом чате")
        self.assertIs(intent, Intent.CHAT)

    async def test_image_intent_wins_over_status_words(self):
        client = _client("image")
        with patch("bot.services.request_intent.ai_client", client):
            intent = await classify_request("нарисуй статус")
        self.assertIs(intent, Intent.IMAGE)

    async def test_image_confirmed_by_model(self):
        client = _client("image")
        with patch("bot.services.request_intent.ai_client", client):
            intent = await classify_request("нарисуй кота в космосе")
        self.assertIs(intent, Intent.IMAGE)

    async def test_image_words_but_model_says_chat(self):
        client = _client("chat")
        with patch("bot.services.request_intent.ai_client", client):
            intent = await classify_request("что такое изображение в физике")
        self.assertIs(intent, Intent.CHAT)

    async def test_model_failures_fall_back_to_chat(self):
        for client in (
            _client(None),
            _client("мусор"),
            _client(error=RuntimeError("down")),
        ):
            with patch("bot.services.request_intent.ai_client", client):
                intent = await classify_request("нарисуй кота")
            self.assertIs(intent, Intent.CHAT)

    async def test_status_model_failure_falls_back_to_chat(self):
        client = _client(error=RuntimeError("down"))
        with patch("bot.services.request_intent.ai_client", client):
            intent = await classify_request("статус")
        self.assertIs(intent, Intent.CHAT)

    async def test_at_most_256_chars_reach_the_model(self):
        client = _client("image")
        text = "нарисуй " + "кота " * 200
        with patch("bot.services.request_intent.ai_client", client):
            await classify_request(text)
        sent = client.chat_completion_result.await_args.kwargs["messages"][1]
        self.assertLessEqual(len(sent["content"]), 256)


if __name__ == "__main__":
    unittest.main()
