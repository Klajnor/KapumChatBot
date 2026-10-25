"""Inline reminder controls: pause/resume, ownership and the rendered keyboard."""

from __future__ import annotations

import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from bot.config import config
from bot.database.models import Reminder
from bot.handlers import reminders as handler
from bot.services.user_service import user_service
from tests.test_database import DBTest

OWNER, OTHER = 10, 11


class TestReminderControls(DBTest):
    def _run(self, coro_fn):
        p = patch("bot.database.session.async_session_factory", self._factory)
        p2 = patch("bot.services.reminders.async_session_factory", self._factory)
        p.start()
        p2.start()
        try:
            self._loop.run_until_complete(coro_fn())
        finally:
            p2.stop()
            p.stop()

    async def _add(self, user_id: int, schedule: str | None, minutes: int) -> int:
        async with user_service() as s:
            await s.register_user(user_id)
            created = await s.add_reminder(
                Reminder(
                    user_id=user_id,
                    chat_id=user_id,
                    text="drink water",
                    fire_at=datetime.datetime.now(datetime.UTC)
                    + datetime.timedelta(minutes=minutes),
                    schedule=schedule,
                )
            )
            return created.id

    def test_pause_keeps_slot_and_resume_skips_missed_fires(self):
        async def go() -> None:
            rid = await self._add(OWNER, "every:60", -300)  # 5 h overdue
            async with user_service() as s:
                paused = await s.toggle_reminder(rid, OWNER)
                self.assertFalse(paused.enabled)
                self.assertEqual(await s.count_reminders(OWNER), 1)  # still counted
                self.assertEqual(len(await s.list_reminders(OWNER)), 1)  # still listed
            async with user_service() as s:
                resumed = await s.toggle_reminder(rid, OWNER)
            self.assertTrue(resumed.enabled)
            fire = resumed.fire_at
            if fire.tzinfo is None:
                fire = fire.replace(tzinfo=datetime.UTC)
            self.assertGreater(fire, datetime.datetime.now(datetime.UTC))

        self._run(go)

    def test_cannot_toggle_someone_elses_reminder(self):
        async def go() -> None:
            rid = await self._add(OWNER, None, 30)
            await self._add(OTHER, None, 30)
            async with user_service() as s:
                self.assertIsNone(await s.toggle_reminder(rid, OTHER))
                self.assertFalse(await s.delete_reminder(rid, OTHER))

        self._run(go)

    def test_paused_reminder_is_not_claimed_by_scheduler(self):
        from bot.services.reminders import process_due

        async def go() -> None:
            rid = await self._add(OWNER, None, -5)
            async with user_service() as s:
                await s.toggle_reminder(rid, OWNER)
            self.assertEqual(await process_due(), [])

        self._run(go)

    def test_list_keyboard_and_owner_guard(self):
        async def go() -> None:
            rid = await self._add(OWNER, None, 30)
            text, markup = await handler._render_list(OWNER)
            data = [b.callback_data for row in markup.inline_keyboard for b in row]
            self.assertIn(f"rem:tg:{OWNER}:{rid}:0", data)
            self.assertIn(f"rem:del:{OWNER}:{rid}:0", data)
            self.assertIn(f"rem:all:{OWNER}", data)

            # Another user pressing the owner's button changes nothing.
            callback = SimpleNamespace(
                data=f"rem:del:{OWNER}:{rid}:0",
                from_user=SimpleNamespace(id=OTHER),
                message=SimpleNamespace(),
            )
            with (
                patch.object(handler, "safe_answer_callback", AsyncMock()) as answer,
                patch.object(config, "enable_reminders", True),
            ):
                await handler.cb_reminders(callback)
            self.assertIn("not your list", answer.await_args.args[1])
            async with user_service() as s:
                self.assertEqual(await s.count_reminders(OWNER), 1)

        self._run(go)
