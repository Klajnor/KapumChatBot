"""Free-form reminder requests: pre-filter, JSON parsing, slot validation.

The provider-facing AI call is replaced by a stub; everything downstream
(the strict /remind parser, recurring floor and interval ceiling) runs for
real, so the tests also verify that a hallucinated slot cannot break the
limits enforced by ``parse_reminder``.
"""

from __future__ import annotations

import datetime
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from bot.services.reminder_intent import (
    _extract_json,
    extract_freeform,
    looks_like_reminder,
)

# 12:00 UTC on a Tuesday; the "tomorrow at 17:00" expectation assumes a UTC
# dev environment (REMINDER_TIMEZONE default).
NOW = datetime.datetime(2026, 10, 27, 12, 0, tzinfo=datetime.UTC)


def _model_reply(content: str):
    """Patch the intent call to return ``content``."""
    client = SimpleNamespace(
        chat_completion_result=AsyncMock(return_value=SimpleNamespace(content=content))
    )
    return patch("bot.services.reminder_intent.ai_client", client)


class TestPrefilter(unittest.IsolatedAsyncioTestCase):
    def test_matches_ru_and_en(self):
        for text in (
            "напомни завтра позвонить маме",
            "Поставь будильник на 7 утра",
            "Разбуди меня в выходные",
            "remind me in 30 minutes",
            "Set a reminder about the report",
            "What is my reminder status?",
            "поставь напоминалку на завтра",
            "устнови напоминалку в 18:00",
            "создай напоминание про отчёт",
        ):
            self.assertTrue(looks_like_reminder(text), text)

    def test_matches_schedule_without_reminder_word(self):
        for text in (
            "завтра в 9 утра покажи погоду в москве",
            "каждый день в 8 присылай новости",
            "через 2 часа пришли курс доллара",
            "tomorrow at 9 show me the weather",
        ):
            self.assertTrue(looks_like_reminder(text), text)

    def test_only_first_chars_are_inspected(self):
        text = "а" * 70 + " напомни завтра"
        self.assertFalse(looks_like_reminder(text))
        self.assertTrue(looks_like_reminder(text, limit=200))

    def test_ignores_ordinary_chat(self):
        for text in (
            "Как приготовить борщ?",
            "What is the capital of France?",
            "почему небо синее",
            "",
        ):
            self.assertFalse(looks_like_reminder(text), text)


class TestJsonExtraction(unittest.IsolatedAsyncioTestCase):
    def test_plain_json(self):
        answer = _extract_json(
            '{"reminder": true, "spec": "tomorrow at 17:00", "text": "позвонить маме"}'
        )
        self.assertTrue(answer.reminder)
        self.assertEqual(answer.spec, "tomorrow at 17:00")
        self.assertEqual(answer.text, "позвонить маме")

    def test_json_in_code_fences(self):
        answer = _extract_json('```json\n{"reminder": false}\n```')
        self.assertFalse(answer.reminder)

    def test_garbage_returns_none(self):
        for content in ("not json at all", "", '{"reminder": tru'):
            self.assertIsNone(_extract_json(content))

    def test_not_a_dict(self):
        self.assertIsNone(_extract_json("[1, 2, 3]"))


class TestExtractFreeform(unittest.IsolatedAsyncioTestCase):
    async def test_not_a_reminder_returns_none(self):
        with _model_reply('{"reminder": false}'):
            outcome = await extract_freeform("Как дела?", "model-x", now=NOW)
        self.assertIsNone(outcome)

    async def test_not_reminder_content_with_reminder_words(self):
        # A question about the feature is not a reminder request: the cheap
        # pre-filter matches it, the AI call must answer "false".
        with _model_reply('{"reminder": false}'):
            outcome = await extract_freeform(
                "How do I delete a reminder?", "model-x", now=NOW
            )
        self.assertIsNone(outcome)

    async def test_broken_json_returns_none(self):
        for content in ("good-old-days", '{"reminder": true, "spec": }'):
            with _model_reply(content):
                outcome = await extract_freeform("напомни завтра…", "model-x", now=NOW)
            self.assertIsNone(outcome, content)

    async def test_ai_failure_returns_none(self):
        client = SimpleNamespace(
            chat_completion_result=AsyncMock(side_effect=RuntimeError("down"))
        )
        with patch("bot.services.reminder_intent.ai_client", client):
            outcome = await extract_freeform("напомни завтра", "model-x", now=NOW)
        self.assertIsNone(outcome)

    async def test_one_time_tomorrow(self):
        content = (
            '{"reminder": true, "spec": "tomorrow at 17:00", '
            '"text": "позвонить маме"}'
        )
        # Pin the display timezone to UTC so the expectation does not
        # depend on the REMINDER_TIMEZONE of the machine running the tests;
        # both the prompt context and parse_reminder must see the same tz.
        with _model_reply(content), patch(
            "bot.services.reminder_intent.get_timezone", return_value=datetime.UTC
        ), patch("bot.services.reminders.get_timezone", return_value=datetime.UTC):
            parsed = await extract_freeform(
                "напомни завтра в 5 позвонить маме", "model-x", now=NOW
            )
        self.assertIsNotNone(parsed)
        self.assertIsNone(parsed.schedule)
        self.assertEqual(parsed.text, "позвонить маме")
        expected = datetime.datetime(2026, 10, 28, 17, 0, tzinfo=datetime.UTC)
        self.assertEqual(parsed.fire_at, expected)

    async def test_recurring_daily(self):
        content = (
            '{"reminder": true, "spec": "daily at 09:00", "text": "drink water"}'
        )
        with _model_reply(content):
            parsed = await extract_freeform(
                "напоминай каждое утро пить воду", "model-x", now=NOW
            )
        self.assertEqual(parsed.schedule, "daily:09:00")
        self.assertEqual(parsed.text, "drink water")

    async def test_ai_prefix_passthrough(self):
        content = (
            '{"reminder": true, "spec": "daily at 09:00", '
            '"text": "ai: main AI news of the day"}'
        )
        with _model_reply(content):
            parsed = await extract_freeform(
                "каждое утро напоминай мне новости через ai", "model-x", now=NOW
            )
        self.assertTrue(parsed.ai_mode)
        self.assertEqual(parsed.text, "main AI news of the day")

    async def test_hallucinated_too_frequent_slot_is_rejected(self):
        with _model_reply(
            '{"reminder": true, "spec": "every 1 minute", "text": "x"}'
        ):
            outcome = await extract_freeform(
                "напоминай каждые 20 секунд", "model-x", now=NOW
            )
        self.assertIsInstance(outcome, str)  # friendly error, not a reminder

    async def test_hallucinated_every_n_seconds_is_rejected(self):
        # Vocabulary violation that still parses as an interval.
        with _model_reply('{"reminder": true, "spec": "in 5 seconds", "text": "x"}'):
            outcome = await extract_freeform("напомни через 5 секунд", "model-x", now=NOW)
        self.assertIsInstance(outcome, str)


if __name__ == "__main__":
    unittest.main()
