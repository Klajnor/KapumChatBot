"""H-6: отзыв доступа останавливает обычные доставки и старые resume-кнопки."""

from __future__ import annotations

import asyncio
import datetime
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from sqlalchemy import update
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from bot.config import config
from bot.database.models import Base, Reminder
from bot.handlers import reminders as handler
from bot.services import reminders
from bot.services.user_service import user_service


class ReminderRevocationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        self.engine = create_async_engine("sqlite+aiosqlite://")
        self.factory = async_sessionmaker(self.engine, expire_on_commit=False)
        async with self.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        stack.enter_context(patch("bot.database.session.async_session_factory", self.factory))
        stack.enter_context(patch.object(reminders, "async_session_factory", self.factory))
        stack.enter_context(patch.object(reminders, "_fire_tasks", set()))
        stack.enter_context(patch.object(config, "owner_ids", []))
        stack.enter_context(patch.object(config, "extra_admin_ids", set()))
        stack.enter_context(patch.object(config, "allow_guest_access", False))
        stack.enter_context(patch.object(config, "enable_reminders", True))
        stack.enter_context(patch.object(config, "daily_limit", 1))
        self.send = stack.enter_context(patch.object(reminders, "_send_html", AsyncMock()))
        self.answer = stack.enter_context(patch.object(handler, "safe_answer", AsyncMock()))
        self.callback_answer = stack.enter_context(patch.object(handler, "safe_answer_callback", AsyncMock()))
        stack.enter_context(patch.object(handler, "safe_edit_text", AsyncMock()))
        async with user_service() as service:
            await service.register_user(1)
            await service.register_user(2)
            await service.add_allowed_user(1)

    async def asyncTearDown(self):
        await reminders.cancel_pending_fires()
        await self.engine.dispose()

    async def add(self, *, enabled=True, schedule="every:30"):
        async with user_service() as service:
            row = await service.add_reminder(Reminder(
                user_id=1, chat_id=-100, text="check <oven>", enabled=enabled,
                schedule=schedule,
                fire_at=datetime.datetime.now(datetime.UTC) - datetime.timedelta(minutes=1),
            ))
        return row.id

    def due(self, rid, *, schedule="every:30"):
        return reminders.DueReminder(
            id=rid, user_id=1, chat_id=-100, text="check <oven>",
            ai_mode=False, schedule=schedule,
        )

    async def enabled(self, rid):
        async with self.factory() as session:
            row = await session.get(Reminder, rid)
            return row.enabled if row is not None else None

    def callback(self, rid, *, action="tg", user=1):
        return SimpleNamespace(
            data=f"rem:{action}:1:{rid}:0", from_user=SimpleNamespace(id=user),
            message=SimpleNamespace(chat=SimpleNamespace(id=1, type="private")),
        )

    async def test_blocked_user_never_delivers_and_recurring_is_disabled(self):
        rid = await self.add()
        async with user_service() as service:
            await service.set_blocked(1, True)
        await reminders.fire_reminder(self.due(rid))
        self.send.assert_not_awaited()
        self.assertFalse(await self.enabled(rid))

    async def test_removed_user_never_delivers(self):
        rid = await self.add()
        async with user_service() as service:
            await service.remove_allowed_user(1)
        await reminders.fire_reminder(self.due(rid))
        self.send.assert_not_awaited()
        self.assertFalse(await self.enabled(rid))

    async def test_guest_trial_cannot_keep_old_reminders_alive(self):
        rid = await self.add()
        async with user_service() as service:
            await service.remove_allowed_user(1)
        with (
            patch.object(config, "allow_guest_access", True),
            patch.object(config, "guest_message_limit", 5),
        ):
            await reminders.fire_reminder(self.due(rid))
        self.send.assert_not_awaited()
        self.assertFalse(await self.enabled(rid))

    async def test_exhausted_questions_do_not_disable_free_delivery_or_consume_more(self):
        rid = await self.add()
        async with user_service() as service:
            self.assertEqual(await service.consume_daily_usage(1), 1)
        await reminders.fire_reminder(self.due(rid))
        self.send.assert_awaited_once()
        self.assertIn("&lt;oven&gt;", self.send.call_args.args[1])
        self.assertTrue(await self.enabled(rid))
        async with user_service() as service:
            self.assertEqual(await service.get_daily_usage_used(1), 1)

    async def test_removed_owner_without_allowlist_loses_scheduled_delivery(self):
        rid = await self.add()
        async with user_service() as service:
            await service.remove_allowed_user(1)
        with patch.object(config, "owner_ids", [1]):
            await reminders.fire_reminder(self.due(rid))
        self.send.assert_awaited_once()
        self.send.reset_mock()
        await reminders.fire_reminder(self.due(rid))
        self.send.assert_not_awaited()
        self.assertFalse(await self.enabled(rid))

    async def test_scheduler_claim_then_revocation_check_disables_recurring(self):
        rid = await self.add()
        async with user_service() as service:
            await service.set_blocked(1, True)
        claimed = await reminders.process_due()
        self.assertEqual([due.id for due in claimed], [rid])
        await asyncio.gather(*list(reminders._fire_tasks))
        self.send.assert_not_awaited()
        self.assertFalse(await self.enabled(rid))
        # Даже принудительно просроченная отключённая запись больше не заявляется.
        async with self.factory() as session:
            await session.execute(update(Reminder).where(Reminder.id == rid).values(
                fire_at=datetime.datetime.now(datetime.UTC) - datetime.timedelta(minutes=1),
            ))
            await session.commit()
        self.assertEqual(await reminders.process_due(), [])

    async def test_one_time_revoked_reminder_is_not_delivered_after_claim_deletion(self):
        rid = await self.add(schedule=None)
        async with user_service() as service:
            await service.set_blocked(1, True)
        await reminders.process_due()
        await asyncio.gather(*list(reminders._fire_tasks))
        self.send.assert_not_awaited()
        self.assertIsNone(await self.enabled(rid))

    async def test_database_failure_is_fail_closed_without_permanent_disable(self):
        rid = await self.add()
        with patch.object(reminders, "check_access", AsyncMock(side_effect=RuntimeError("DB down"))):
            await reminders.fire_reminder(self.due(rid))
        self.send.assert_not_awaited()
        self.assertTrue(await self.enabled(rid))

    async def test_old_resume_button_fails_after_block(self):
        rid = await self.add(enabled=False)
        async with user_service() as service:
            await service.set_blocked(1, True)
        await handler.cb_reminders(self.callback(rid))
        self.assertFalse(await self.enabled(rid))
        self.assertIn("authorization", self.callback_answer.call_args.args[1])

    async def test_old_resume_button_fails_after_removal_even_with_guest_trial(self):
        rid = await self.add(enabled=False)
        async with user_service() as service:
            await service.remove_allowed_user(1)
        with patch.object(config, "allow_guest_access", True):
            await handler.cb_reminders(self.callback(rid))
        self.assertFalse(await self.enabled(rid))

    async def test_authorized_user_can_pause_and_resume_with_no_questions_left(self):
        rid = await self.add()
        async with user_service() as service:
            await service.consume_daily_usage(1)
        await handler.cb_reminders(self.callback(rid))
        self.assertFalse(await self.enabled(rid))
        await handler.cb_reminders(self.callback(rid))
        self.assertTrue(await self.enabled(rid))
        async with user_service() as service:
            self.assertEqual(await service.get_daily_usage_used(1), 1)

    async def test_restoring_access_does_not_automatically_resume(self):
        rid = await self.add()
        async with user_service() as service:
            await service.set_blocked(1, True)
        await reminders.fire_reminder(self.due(rid))
        async with user_service() as service:
            await service.set_blocked(1, False)
        self.assertFalse(await self.enabled(rid))
        await handler.cb_reminders(self.callback(rid))
        self.assertTrue(await self.enabled(rid))

    async def test_owner_guard_precedes_access_check(self):
        rid = await self.add(enabled=False)
        with patch.object(handler, "check_access", AsyncMock()) as gate:
            await handler.cb_reminders(self.callback(rid, user=2))
        gate.assert_not_awaited()
        self.assertFalse(await self.enabled(rid))

    async def test_blocked_user_can_view_and_delete_own_records(self):
        rid = await self.add(enabled=False)
        async with user_service() as service:
            await service.set_blocked(1, True)
        message = SimpleNamespace(from_user=SimpleNamespace(id=1))
        await handler.cmd_reminders(message)
        self.answer.assert_awaited_once()
        await handler.cb_reminders(self.callback(rid, action="del"))
        self.assertIsNone(await self.enabled(rid))

    async def test_blocked_user_cannot_delete_another_owners_record(self):
        rid = await self.add(enabled=False)
        async with user_service() as service:
            await service.set_blocked(2, True)
        await handler.cb_reminders(self.callback(rid, action="del", user=2))
        self.assertIsNotNone(await self.enabled(rid))
