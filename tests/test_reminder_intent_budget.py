"""H-5: общий бюджет AI-помощников, live-доступ и слот до текстовых helper-ов."""

from __future__ import annotations

import asyncio
import datetime
import tempfile
import unittest
from collections import defaultdict
from contextlib import ExitStack, asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from bot.config import config
from bot.database.models import Base, DailyUsage, ReminderIntentUsage
from bot.database.repository import UserRepository
from bot.handlers import chat
from bot.handlers import reminders as handler
from bot.services import limits
from bot.services.access import AccessResult
from bot.services.user_service import UserService

DAY = datetime.date(2026, 10, 31)


class ReminderBudgetDatabaseTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="reminder-intent-")
        self.url = "sqlite+aiosqlite:///" + self.directory.name.replace("\\", "/") + "/test.db"
        self.engine = create_async_engine(self.url)
        self.factory = async_sessionmaker(self.engine, expire_on_commit=False)
        async with self.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

    async def asyncTearDown(self):
        await self.engine.dispose()
        self.directory.cleanup()

    async def reserve(self, user=1, user_limit=2, global_limit=10, day=DAY):
        async with self.factory() as session:
            accepted = await UserRepository(session).try_consume_reminder_intent(
                user, day, user_limit, global_limit,
            )
            await session.commit()
            return accepted

    async def counts(self, day=DAY):
        async with self.factory() as session:
            rows = await session.scalars(select(ReminderIntentUsage).where(ReminderIntentUsage.date == day))
            return {row.scope_id: row.call_count for row in rows}

    async def test_user_cap_and_rollback_of_global_increment(self):
        self.assertEqual([await self.reserve() for _ in range(4)], [True, True, False, False])
        self.assertEqual(await self.counts(), {0: 2, 1: 2})

    async def test_concurrent_user_cap(self):
        accepted = await asyncio.gather(*(self.reserve() for _ in range(12)))
        self.assertEqual(sum(accepted), 2)
        self.assertEqual(await self.counts(), {0: 2, 1: 2})

    async def test_concurrent_global_cap(self):
        accepted = await asyncio.gather(*(
            self.reserve(user=i, global_limit=3) for i in range(1, 13)
        ))
        self.assertEqual(sum(accepted), 3)
        counts = await self.counts()
        self.assertEqual(counts.pop(0), 3)
        self.assertEqual(sum(counts.values()), 3)

    async def test_zero_disables_and_next_day_resets_budget(self):
        self.assertFalse(await self.reserve(user_limit=0))
        self.assertFalse(await self.reserve(global_limit=0))
        self.assertEqual(await self.counts(), {})
        self.assertTrue(await self.reserve(user_limit=1))
        self.assertFalse(await self.reserve(user_limit=1))
        self.assertTrue(await self.reserve(user_limit=1, day=DAY + datetime.timedelta(days=1)))

    async def test_restart_does_not_reset_budget(self):
        self.assertTrue(await self.reserve(user_limit=1))
        await self.engine.dispose()
        self.engine = create_async_engine(self.url)
        self.factory = async_sessionmaker(self.engine, expire_on_commit=False)
        self.assertFalse(await self.reserve(user_limit=1))

    async def test_photo_and_reminder_budgets_are_independent(self):
        self.assertTrue(await self.reserve(user_limit=1))
        async with self.factory() as session:
            self.assertTrue(await UserRepository(session).try_consume_photo_intent(1, DAY, 1, 1))
            await session.commit()
        self.assertFalse(await self.reserve(user_limit=1))

    async def test_service_commits_and_does_not_charge_questions(self):
        with (
            patch.object(config, "reminder_intent_daily_limit", 1),
            patch.object(config, "reminder_intent_global_daily_limit", 2),
        ):
            async with self.factory() as session:
                self.assertTrue(await UserService(session).consume_reminder_intent_budget(1))
            async with self.factory() as session:
                self.assertFalse(await UserService(session).consume_reminder_intent_budget(1))
                self.assertEqual(list(await session.scalars(select(DailyUsage))), [])


class ReminderHelperGateTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.object(config, "enable_reminders", True))
        stack.enter_context(patch.object(config, "reminder_freeform", True))
        stack.enter_context(patch.object(config, "max_concurrent_per_user", 1))
        stack.enter_context(patch.object(handler, "_tz_dialogs", {}))
        stack.enter_context(patch.object(limits, "_inflight", defaultdict(int)))
        self.service = SimpleNamespace(
            register_user=AsyncMock(), get_user_settings=AsyncMock(return_value={"model": "m"}),
            get_reminder_timezone=AsyncMock(return_value="UTC"),
            consume_reminder_intent_budget=AsyncMock(return_value=True),
            set_reminder_timezone=AsyncMock(),
        )

        @asynccontextmanager
        async def service():
            yield self.service

        stack.enter_context(patch.object(handler, "user_service", service))
        stack.enter_context(patch.object(handler, "is_forwarded", return_value=False))
        self.gate = stack.enter_context(patch.object(
            handler, "check_access", AsyncMock(return_value=AccessResult(True, "allowed")),
        ))
        self.answer = stack.enter_context(patch.object(handler, "safe_answer", AsyncMock()))
        self.callback_answer = stack.enter_context(patch.object(handler, "safe_answer_callback", AsyncMock()))
        stack.enter_context(patch.object(handler, "safe_edit_text", AsyncMock()))
        self.freeform = stack.enter_context(patch.object(handler, "extract_freeform", AsyncMock(return_value=None)))
        self.timezone = stack.enter_context(patch.object(handler, "detect_timezone", AsyncMock(return_value=None)))
        self.message = SimpleNamespace(
            from_user=SimpleNamespace(id=1, username=None, first_name="Test", last_name=None),
            chat=SimpleNamespace(id=1, type="private"), text="remind me tomorrow", reply_to_message=None,
        )

    async def test_repeated_non_reminder_outputs_still_use_attempts(self):
        self.service.consume_reminder_intent_budget.side_effect = [True, True, False]
        results = [await handler.maybe_freeform_reminder(self.message, "remind me tomorrow") for _ in range(3)]
        self.assertEqual(results, [False, False, True])
        self.assertEqual(self.freeform.await_count, 2)
        self.assertIn("interpretation limit", self.answer.call_args.args[1])
        for call in self.gate.call_args_list:
            self.assertEqual(call.kwargs, {"consume": False, "require_quota": False})

    async def test_alternating_helpers_cannot_get_separate_allowances(self):
        self.service.consume_reminder_intent_budget.side_effect = [True, True, False, False]
        handler._start_dialog(1, 1)
        await handler.maybe_freeform_reminder(self.message, "remind me tomorrow")
        await handler.maybe_timezone_dialog(self.message, "unknown place")
        await handler.maybe_freeform_reminder(self.message, "remind me tomorrow")
        await handler.maybe_timezone_dialog(self.message, "unknown place")
        self.assertEqual(self.freeform.await_count, 1)
        self.assertEqual(self.timezone.await_count, 1)

    async def test_blocked_active_timezone_dialog_does_not_call_ai(self):
        handler._start_dialog(1, 1)
        self.gate.return_value = AccessResult(False, "blocked")
        self.assertTrue(await handler.maybe_timezone_dialog(self.message, "Moscow"))
        self.timezone.assert_not_awaited()
        self.service.consume_reminder_intent_budget.assert_not_awaited()
        self.assertNotIn(1, handler._tz_dialogs)

    async def test_guest_active_dialog_is_rejected(self):
        handler._start_dialog(1, 1)
        self.gate.return_value = AccessResult(True, "guest")
        await handler.maybe_timezone_dialog(self.message, "Moscow")
        self.timezone.assert_not_awaited()
        self.service.consume_reminder_intent_budget.assert_not_awaited()

    async def test_denied_freeform_never_uses_budget(self):
        self.gate.return_value = AccessResult(False, "blocked")
        self.assertFalse(await handler.maybe_freeform_reminder(self.message, "remind me tomorrow"))
        self.freeform.assert_not_awaited()
        self.service.consume_reminder_intent_budget.assert_not_awaited()

    async def test_failed_model_does_not_refund_helper_attempt(self):
        self.service.consume_reminder_intent_budget.side_effect = [True, False]
        # None — штатный результат сервиса и при ошибке/невалидном JSON модели.
        for _ in range(2):
            await handler.maybe_freeform_reminder(self.message, "remind me tomorrow")
        self.freeform.assert_awaited_once()

    async def test_database_failure_is_fail_closed(self):
        handler._start_dialog(1, 1)
        self.service.consume_reminder_intent_budget.side_effect = RuntimeError("database down")
        with self.assertRaises(RuntimeError):
            await handler.maybe_timezone_dialog(self.message, "Moscow")
        self.timezone.assert_not_awaited()

    async def test_revocation_during_freeform_prevents_creation(self):
        self.gate.side_effect = [AccessResult(True, "allowed"), AccessResult(False, "blocked")]
        self.freeform.return_value = object()
        with patch.object(handler, "create_reminder_record", AsyncMock()) as create:
            self.assertTrue(await handler.maybe_freeform_reminder(self.message, "remind me tomorrow"))
        create.assert_not_awaited()

    async def test_timezone_callback_rechecks_access(self):
        handler._start_dialog(1, 1, "Europe/Moscow")
        self.gate.return_value = AccessResult(False, "blocked")
        callback = SimpleNamespace(
            from_user=self.message.from_user, message=self.message, data="tz:ok:Europe/Moscow",
        )
        await handler.cb_timezone(callback)
        self.service.set_reminder_timezone.assert_not_awaited()
        self.assertNotIn(1, handler._tz_dialogs)

    async def test_strict_timezone_is_available_without_helper_budget(self):
        self.message.text = "/timezone Europe/Moscow"
        self.service.consume_reminder_intent_budget.return_value = False
        await handler.cmd_timezone(self.message)
        self.service.set_reminder_timezone.assert_awaited_once_with(1, "Europe/Moscow")
        self.service.consume_reminder_intent_budget.assert_not_awaited()

    async def test_strict_remind_is_available_without_helper_budget(self):
        self.service.consume_reminder_intent_budget.return_value = False
        with patch.object(handler, "create_reminder_record", AsyncMock(return_value=(1, None))) as create:
            self.assertTrue(await handler.create_reminder_from_text(
                self.message, "in 30 minutes drink water",
            ))
        create.assert_awaited_once()
        self.service.consume_reminder_intent_budget.assert_not_awaited()
        self.freeform.assert_not_awaited()

    async def test_timezone_confirmation_does_not_need_another_helper_attempt(self):
        handler._start_dialog(1, 1, "Europe/Moscow")
        self.service.consume_reminder_intent_budget.return_value = False
        callback = SimpleNamespace(
            from_user=self.message.from_user, message=self.message, data="tz:ok:Europe/Moscow",
        )
        await handler.cb_timezone(callback)
        self.service.set_reminder_timezone.assert_awaited_once_with(1, "Europe/Moscow")
        self.service.consume_reminder_intent_budget.assert_not_awaited()

    async def test_helpers_run_under_text_slot_and_busy_update_does_not_reach_them(self):
        started = asyncio.Event()

        async def slow(*args):
            self.assertEqual(limits._inflight[1], 1)
            started.set()
            await asyncio.Event().wait()

        with (
            patch.object(chat, "_is_addressed", return_value=True),
            patch.object(chat, "maybe_timezone_dialog", AsyncMock(return_value=False)),
            patch.object(chat, "maybe_freeform_reminder", AsyncMock(side_effect=slow)) as helper,
            patch.object(chat, "safe_answer", AsyncMock()) as answer,
        ):
            first = asyncio.create_task(chat.handle_text(self.message))
            try:
                await asyncio.wait_for(started.wait(), 2)
                await chat.handle_text(self.message)
                helper.assert_awaited_once()
                self.assertEqual(answer.call_args.args[1], limits.BUSY_MESSAGE)
            finally:
                first.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await first
        self.assertEqual(dict(limits._inflight), {})

    async def test_fallthrough_releases_helper_slot_before_main_slot(self):
        with (
            patch.object(chat, "_is_addressed", return_value=True),
            patch.object(chat, "maybe_timezone_dialog", AsyncMock(return_value=False)),
            patch.object(chat, "maybe_freeform_reminder", AsyncMock(return_value=False)),
            patch.object(chat, "_open_registered", AsyncMock(return_value=SimpleNamespace(
                session=SimpleNamespace(close=AsyncMock()),
            ))),
            patch.object(chat, "_consume_access", AsyncMock(return_value=None)) as consume,
        ):
            await chat.handle_text(self.message)
        consume.assert_awaited_once()
        self.assertEqual(dict(limits._inflight), {})

    async def test_voice_caller_slot_is_not_acquired_twice(self):
        with limits.slot(1) as granted:
            self.assertTrue(granted)
            await handler.maybe_freeform_reminder(self.message, "remind me tomorrow")
        self.freeform.assert_awaited_once()
