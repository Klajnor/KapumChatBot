"""Channel digest: message intake, the 40/20 window rule, token statistics."""

from __future__ import annotations

import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from bot.config import config
from bot.database.repository import DigestRepository
from bot.services import digest
from bot.services.ai_client import ChatResult
from tests.test_database import DBTest

CHAT = -1001
NOW = datetime.datetime(2026, 10, 25, 18, 0, tzinfo=datetime.UTC)


class TestDigest(DBTest):
    def _run(self, coro_fn):
        patches = [
            patch("bot.services.digest.async_session_factory", self._factory),
            patch("bot.services.digest._channel_ids", None),
        ]
        for p in patches:
            p.start()
        try:
            return self._loop.run_until_complete(coro_fn())
        finally:
            for p in reversed(patches):
                p.stop()

    async def _seed(self, hours_ago: list[float]) -> None:
        await digest.add_channel(CHAT, "News", 1)
        for i, h in enumerate(hours_ago):
            await digest.record_message(
                CHAT, i + 1, "Ann", f"post {i + 1}", NOW - datetime.timedelta(hours=h)
            )

    @staticmethod
    def _fake_ai():
        seen: dict = {}

        async def fake(messages, **kwargs):
            seen["messages"] = messages
            return ChatResult(
                content="summary",
                usage={"prompt_tokens": 120, "completion_tokens": 30},
                model="m",
            )

        return seen, fake

    def test_untracked_chat_and_empty_text_are_ignored(self):
        async def go() -> None:
            await digest.add_channel(CHAT, "News", 1)
            self.assertFalse(await digest.record_message(-5, 1, "x", "hi", NOW))
            self.assertFalse(await digest.record_message(CHAT, 2, "x", "  ", NOW))
            self.assertTrue(await digest.record_message(CHAT, 3, "x", "hi", NOW))
            # A repeated delivery is stored once; an edit replaces the text.
            self.assertFalse(await digest.record_message(CHAT, 3, "x", "hi", NOW))
            await digest.record_message(CHAT, 3, "x", "fixed", NOW, edited=True)
            async with self._factory() as s:
                rows = await DigestRepository(s).messages_up_to(CHAT, NOW, 10)
            self.assertEqual([r.text for r in rows], ["fixed"])

        self._run(go)

    def test_few_messages_get_up_to_20_context_messages(self):
        async def go() -> None:
            # 5 inside the 24 h window, 30 older ones.
            await self._seed([1, 2, 3, 4, 5] + [30 + i for i in range(30)])
            seen, fake = self._fake_ai()
            with (
                patch.object(digest.ai_client, "chat_completion_result", fake),
                patch.object(digest, "_send_markdown", AsyncMock(return_value=True)),
                patch.object(digest, "notify_admins", AsyncMock()),
                patch.object(digest.config, "digest_context_messages", 20),
            ):
                until = NOW
                report = await digest._digest_channel(CHAT, "News", until, "m", "p")
            self.assertEqual((report.messages, report.context), (5, 20))
            self.assertEqual((report.input_tokens, report.output_tokens), (120, 30))
            self.assertTrue(report.posted)
            body = seen["messages"][1]["content"]
            self.assertIn("<untrusted_content", body)
            self.assertIn("Контекст", body)

        self._run(go)

    def test_enough_messages_get_no_context(self):
        async def go() -> None:
            await self._seed([0.1 + i / 100 for i in range(config.digest_min_messages)] + [40])
            _, fake = self._fake_ai()
            with (
                patch.object(digest.ai_client, "chat_completion_result", fake),
                patch.object(digest, "_send_markdown", AsyncMock(return_value=True)),
            ):
                report = await digest._digest_channel(CHAT, "News", NOW, "m", "p")
            self.assertEqual((report.messages, report.context), (config.digest_min_messages, 0))

        self._run(go)

    def test_window_starts_at_previous_digest_and_advances(self):
        async def go() -> None:
            await self._seed([30, 10, 2])
            async with self._factory() as s:
                await DigestRepository(s).mark_digest(CHAT, NOW - datetime.timedelta(hours=5))
                await s.commit()
            _, fake = self._fake_ai()
            with (
                patch.object(digest.ai_client, "chat_completion_result", fake),
                patch.object(digest, "_send_markdown", AsyncMock(return_value=True)),
            ):
                first = await digest._digest_channel(CHAT, "News", NOW, "m", "p")
                second = await digest._digest_channel(CHAT, "News", NOW, "m", "p")
            self.assertEqual(first.messages, 1)  # only the post 2 h ago
            # A quiet period still gets a summary of the latest stored messages.
            self.assertTrue(second.recent_only)
            self.assertEqual((second.messages, second.context), (3, 0))
            self.assertTrue(second.posted)

        self._run(go)

    def test_channel_without_any_messages_is_skipped(self):
        async def go() -> None:
            await digest.add_channel(CHAT, "News", 1)
            report = await digest._digest_channel(CHAT, "News", NOW, "m", "p")
            self.assertEqual(report.error, digest.NO_MESSAGES)
            self.assertFalse(report.posted)

        self._run(go)

    def test_failed_post_keeps_the_window_open(self):
        async def go() -> None:
            await self._seed([1])
            _, fake = self._fake_ai()
            with (
                patch.object(digest.ai_client, "chat_completion_result", fake),
                patch.object(digest, "_send_markdown", AsyncMock(return_value=False)),
            ):
                report = await digest._digest_channel(CHAT, "News", NOW, "m", "p")
            self.assertFalse(report.posted)
            async with self._factory() as s:
                channel = await DigestRepository(s).get_channel(CHAT)
            self.assertIsNone(channel.last_digest_at)

        self._run(go)

    def test_settings_roundtrip_and_defaults(self):
        async def go() -> None:
            self.assertEqual(await digest.get_prompt(), digest.DEFAULT_DIGEST_PROMPT)
            await digest.set_setting(digest.SETTING_PROMPT, "custom")
            await digest.set_setting(digest.SETTING_MODEL, config.ai_models[-1]["id"])
            self.assertEqual(await digest.get_prompt(), "custom")
            self.assertEqual(await digest.get_model(), config.ai_models[-1]["id"])
            await digest.set_setting(digest.SETTING_PROMPT, None)
            self.assertEqual(await digest.get_prompt(), digest.DEFAULT_DIGEST_PROMPT)

        self._run(go)

    def test_scheduled_run_is_claimed_once_per_day(self):
        async def go() -> None:
            fixed = SimpleNamespace(
                now=lambda tz=None: NOW.astimezone(tz) if tz else NOW,
            )
            with (
                patch.object(digest.config, "digest_time", "10:00"),
                patch.object(digest.config, "reminder_timezone", "UTC"),
                patch.object(digest.datetime, "datetime", fixed),
            ):
                self.assertTrue(await digest._scheduled_run_due())
                self.assertFalse(await digest._scheduled_run_due())

        self._run(go)

    def test_run_time_setting_overrides_the_env_default(self):
        async def go() -> None:
            with patch.object(digest.config, "digest_time", "21:00"):
                self.assertEqual(await digest.get_time(), datetime.time(21, 0))
                await digest.set_setting(digest.SETTING_TIME, "08:30")
                self.assertEqual(await digest.get_time(), datetime.time(8, 30))
                await digest.set_setting(digest.SETTING_TIME, None)
                self.assertEqual(await digest.get_time(), datetime.time(21, 0))
            self.assertIsNone(digest.parse_time("25:00"))
            self.assertIsNone(digest.parse_time("soon"))

        self._run(go)

    def test_manual_run_notice_explains_each_channel(self):
        from bot.handlers.admin_panel import _run_notice

        reports = [
            digest.ChannelReport(1, "A", 5, 0, 100, 20, "m", True),
            digest.ChannelReport(2, "B", error=digest.NO_MESSAGES),
            digest.ChannelReport(3, "C", error="boom"),
        ]
        text = _run_notice(digest.DigestResult(reports))
        self.assertIn("A: 5 msg", text)
        self.assertIn("nothing stored", text)
        self.assertIn("boom", text)
        self.assertIn("already in progress", _run_notice(digest.DigestResult(skipped=True)))

    def test_stats_report_both_directions_and_total(self):
        reports = [
            digest.ChannelReport(CHAT, "News", 5, 2, 1200, 300, "m", True),
            digest.ChannelReport(-2, "Chat", 1, 0, None, None, "m", False, "boom"),
            digest.ChannelReport(-3, "More", 9, 0, 800, 200, "m", True),
        ]
        text = digest.format_stats(reports)
        self.assertIn("↑ 1 200 ↓ 300", text)
        self.assertIn("boom", text)
        self.assertIn("Σ ↑ 2 000 ↓ 500", text)
