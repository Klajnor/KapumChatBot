"""Personal reminders timezone (/timezone).

Covers the pieces the repo DB drives for real: the resolution chain
(personal zone → REMINDER_TIMEZONE → UTC), persistence on the user row,
per-owner wall-clock rescheduling in the scheduler, the free-form reminder
context zone, and the AI zone detection with its re-validation.
"""

from __future__ import annotations

import datetime
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from zoneinfo import ZoneInfo

from bot.config import config
from bot.database.models import Reminder
from bot.database.repository import UserRepository
from bot.services import reminders as service
from bot.services.reminder_intent import extract_freeform
from bot.services.reminders import get_timezone, parse_reminder
from bot.services.timezone_intent import detect_timezone, is_valid_zone_name
from bot.services.user_service import user_service
from tests.test_database import DBTest

U1, U2 = 500, 501
MOSCOW = ZoneInfo("Europe/Moscow")


def _model_reply(content: str, target: str):
    client = SimpleNamespace(
        chat_completion_result=AsyncMock(return_value=SimpleNamespace(content=content))
    )
    return patch(target, client)


class TestZoneResolution(unittest.TestCase):
    def test_utc_names_resolve_to_utc(self):
        self.assertEqual(get_timezone("UTC"), datetime.UTC)

    def test_valid_zone_resolves(self):
        self.assertIsNotNone(get_timezone("Europe/Moscow"))

    def test_invalid_name_falls_back_to_utc(self):
        # A warning is logged; UTC must be returned, never raised.
        self.assertEqual(get_timezone("Europe/Nowhere"), datetime.UTC)

    def test_none_and_empty_use_the_global_default(self):
        self.assertEqual(
            get_timezone(None), get_timezone(config.reminder_timezone)
        )
        self.assertEqual(
            get_timezone("   "), get_timezone(config.reminder_timezone)
        )

    def test_iana_zone_names_validated(self):
        self.assertTrue(is_valid_zone_name("Europe/Moscow"))
        self.assertFalse(is_valid_zone_name("Europe/Nowhere"))
        self.assertFalse(is_valid_zone_name("Mordor/Barad-dur"))
        self.assertFalse(is_valid_zone_name("Europe/Moscow; DROP TABLE users"))


class TestParseWithTz(unittest.TestCase):
    NOW = datetime.datetime(2026, 10, 27, 12, 0, tzinfo=datetime.UTC)

    def test_at_is_interpreted_in_the_given_zone(self):
        parsed = parse_reminder(
            "at 15:30 feed the cat", now=self.NOW, tz=get_timezone("Europe/Moscow")
        )
        # 15:30 in Moscow (+3) is 12:30 UTC.
        self.assertEqual(
            parsed.fire_at,
            datetime.datetime(2026, 10, 27, 12, 30, tzinfo=datetime.UTC),
        )

    def test_none_tz_keeps_the_global_default(self):
        parsed = parse_reminder("tomorrow at 09:00 x", now=self.NOW)
        explicit = parse_reminder(
            "tomorrow at 09:00 x", now=self.NOW, tz=get_timezone()
        )
        self.assertEqual(parsed.fire_at, explicit.fire_at)


class TestPerUserPersistence(DBTest):
    def _patch_factories(self):
        p = patch("bot.database.session.async_session_factory", self._factory)
        p2 = patch("bot.services.reminders.async_session_factory", self._factory)
        return p, p2

    def test_set_get_reset_zone(self):
        async def go() -> None:
            async with user_service() as s:
                await s.register_user(U1)
                self.assertIsNone(await s.get_reminder_timezone(U1))
                await s.set_reminder_timezone(U1, "Europe/Moscow")
                self.assertEqual(
                    await s.get_reminder_timezone(U1), "Europe/Moscow"
                )
                await s.set_reminder_timezone(U1, None)
                self.assertIsNone(await s.get_reminder_timezone(U1))
            async with user_service() as s2:
                self.assertIsNone(await s2.get_reminder_timezone(U1))

        p, p2 = self._patch_factories()
        p.start()
        p2.start()
        try:
            self._loop.run_until_complete(go())
        finally:
            p2.stop()
            p.stop()

    def test_bulk_lookup_skips_users_without_override(self):
        async def go() -> None:
            async with user_service() as s:
                await s.register_user(U1)
                await s.register_user(U2)
                await s.set_reminder_timezone(U1, "Europe/Kaliningrad")
            async with user_service() as s2:
                repo = UserRepository(s2.session)
                zones = await repo.get_reminder_timezones([U1, U2, 9999])
            self.assertEqual(zones, {U1: "Europe/Kaliningrad"})

        p, p2 = self._patch_factories()
        p.start()
        p2.start()
        try:
            self._loop.run_until_complete(go())
        finally:
            p2.stop()
            p.stop()


class TestSchedulerUsesOwnerZone(DBTest):
    """process_due must compute daily/weekly times in the OWNER's zone."""

    def _patch_and_run(self, coro_fn) -> None:
        p = patch("bot.database.session.async_session_factory", self._factory)
        p2 = patch("bot.services.reminders.async_session_factory", self._factory)
        no_fire = patch("bot.services.reminders.fire_reminder", AsyncMock())
        p.start()
        p2.start()
        no_fire.start()
        try:
            self._loop.run_until_complete(coro_fn())
        finally:
            no_fire.stop()
            p2.stop()
            p.stop()

    def test_daily_reminder_reschedules_in_owners_zone(self):
        async def go() -> None:
            real_now = datetime.datetime.now(datetime.UTC)
            moscow_now = real_now.astimezone(MOSCOW)
            target_local = moscow_now + datetime.timedelta(
                hours=5
            )  # wall-clock target; safe: strictly after "now"
            schedule = f"daily:{target_local.hour:02d}:{target_local.minute:02d}"

            overdue = real_now - datetime.timedelta(minutes=30)
            async with user_service() as s:
                await s.register_user(U1)
                await s.set_reminder_timezone(U1, "Europe/Moscow")
                await s.add_reminder(
                    Reminder(
                        user_id=U1,
                        chat_id=U1,
                        text="standup",
                        fire_at=overdue,
                        schedule=schedule,
                    )
                )

            await service.process_due()

            expected = datetime.datetime(
                target_local.year,
                target_local.month,
                target_local.day,
                target_local.hour,
                target_local.minute,
                tzinfo=MOSCOW,
            ).astimezone(datetime.UTC)
            async with user_service() as s2:
                rows = await s2.list_reminders(U1)
            self.assertEqual(len(rows), 1)
            actual = rows[0].fire_at
            if actual.tzinfo is None:
                actual = actual.replace(tzinfo=datetime.UTC)
            self.assertEqual(actual, expected)

        self._patch_and_run(go)


class TestFreeformContextZone(unittest.IsolatedAsyncioTestCase):
    NOW = datetime.datetime(2026, 10, 27, 12, 0, tzinfo=datetime.UTC)

    async def test_owner_zone_feeds_the_intent_prompt(self):
        captured = {}

        async def fake_completion(messages, **kwargs):
            captured["context"] = messages[1]["content"]
            return SimpleNamespace(
                content='{"reminder": true, "spec": "tomorrow at 10:00", "text": "x"}'
            )

        client = SimpleNamespace(chat_completion_result=fake_completion)
        with patch("bot.services.reminder_intent.ai_client", client):
            outcome = await extract_freeform(
                "напомни завтра",
                "model-x",
                now=self.NOW,
                tz=get_timezone("Europe/Moscow"),
            )
        self.assertIn("Europe/Moscow", captured["context"])
        # 10:00 MSK on Wednesday == 07:00 UTC.
        self.assertEqual(
            outcome.fire_at,
            datetime.datetime(2026, 10, 28, 7, 0, tzinfo=datetime.UTC),
        )


class TestDetectTimezone(unittest.IsolatedAsyncioTestCase):
    async def test_valid_proposal_returned(self):
        with _model_reply(
            '{"zone": "Europe/Moscow"}', "bot.services.timezone_intent.ai_client"
        ):
            self.assertEqual(
                await detect_timezone("я живу в Казани", "model-x"), "Europe/Moscow"
            )

    async def test_hallucinated_zone_rejected(self):
        with _model_reply(
            '{"zone": "Europe/Neverland"}', "bot.services.timezone_intent.ai_client"
        ):
            self.assertIsNone(await detect_timezone("где-то в Европе", "model-x"))

    async def test_not_a_place_returns_none(self):
        for content in ('{"zone": null}', "not json"):
            with _model_reply(content, "bot.services.timezone_intent.ai_client"):
                self.assertIsNone(await detect_timezone("выключи свет", "model-x"))

    async def test_failure_returns_none(self):
        client = SimpleNamespace(
            chat_completion_result=AsyncMock(side_effect=RuntimeError("down"))
        )
        with patch("bot.services.timezone_intent.ai_client", client):
            self.assertIsNone(await detect_timezone("Москва", "model-x"))

    async def test_empty_text_returns_none(self):
        self.assertIsNone(await detect_timezone("", "model-x"))


if __name__ == "__main__":
    unittest.main()


class TestNextOccurrenceHonoursZone(unittest.TestCase):
    """Regression: next_occurrence used to overwrite ``tz`` with the global zone."""

    def test_daily_uses_passed_zone(self):
        now = datetime.datetime(2026, 1, 10, 0, 0, tzinfo=datetime.UTC)
        with patch.object(config, "reminder_timezone", "UTC"):
            nxt = service.next_occurrence("daily:09:00", now, now, tz=MOSCOW)
        # 09:00 Moscow (UTC+3) == 06:00 UTC, not 09:00 UTC.
        self.assertEqual(nxt, datetime.datetime(2026, 1, 10, 6, 0, tzinfo=datetime.UTC))

    def test_weekly_uses_passed_zone(self):
        now = datetime.datetime(2026, 1, 10, 0, 0, tzinfo=datetime.UTC)  # Saturday
        with patch.object(config, "reminder_timezone", "UTC"):
            nxt = service.next_occurrence("weekly:0:09:00", now, now, tz=MOSCOW)
        self.assertEqual(nxt, datetime.datetime(2026, 1, 12, 6, 0, tzinfo=datetime.UTC))


class TestResumeUsesOwnerZone(DBTest):
    def test_resume_daily_in_owners_zone(self):
        async def go() -> None:
            now = datetime.datetime.now(datetime.UTC)
            target = now.astimezone(MOSCOW) + datetime.timedelta(hours=5)
            schedule = f"daily:{target.hour:02d}:{target.minute:02d}"
            async with user_service() as s:
                await s.register_user(U1)
                await s.set_reminder_timezone(U1, "Europe/Moscow")
                created = await s.add_reminder(
                    Reminder(user_id=U1, chat_id=U1, text="x", fire_at=now,
                             schedule=schedule, enabled=False)
                )
                rid = created.id
            async with user_service() as s:
                resumed = await s.toggle_reminder(rid, U1)
            expected = datetime.datetime(
                target.year, target.month, target.day, target.hour, target.minute,
                tzinfo=MOSCOW,
            ).astimezone(datetime.UTC)
            fire = resumed.fire_at
            if fire.tzinfo is None:
                fire = fire.replace(tzinfo=datetime.UTC)
            self.assertEqual(fire, expected)

        p = patch("bot.database.session.async_session_factory", self._factory)
        p.start()
        try:
            self._loop.run_until_complete(go())
        finally:
            p.stop()


class TestTimezoneDialogScope(unittest.TestCase):
    def setUp(self):
        from bot.handlers import reminders as handler

        self.h = handler
        handler._tz_dialogs.clear()
        self.addCleanup(handler._tz_dialogs.clear)

    def test_only_in_same_chat(self):
        self.h._start_dialog(1, chat_id=10)
        self.assertIsNotNone(self.h._get_dialog(1, 10))
        self.assertIsNone(self.h._get_dialog(1, 99))

    def test_expires(self):
        self.h._start_dialog(1, chat_id=10)
        self.h._tz_dialogs[1].expires_at = 0.0
        self.assertIsNone(self.h._get_dialog(1, 10))
        self.assertNotIn(1, self.h._tz_dialogs)

    def test_expired_dialog_does_not_swallow_message(self):
        import asyncio

        self.h._start_dialog(1, chat_id=10)
        self.h._tz_dialogs[1].expires_at = 0.0
        msg = SimpleNamespace(from_user=SimpleNamespace(id=1), chat=SimpleNamespace(id=10))
        self.assertFalse(asyncio.run(self.h.maybe_timezone_dialog(msg, "hello")))
