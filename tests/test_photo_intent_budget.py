"""H-4: отдельный persistent-бюджет фото-классификатора, без двойной квоты."""

from __future__ import annotations

import asyncio
import datetime
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from bot.config import config
from bot.database.models import Base, DailyUsage, PhotoIntentUsage
from bot.database.repository import UserRepository
from bot.handlers import chat, guest
from bot.services.access import AccessResult
from bot.services.user_service import UserService

DAY = datetime.date(2026, 10, 31)


class PhotoBudgetDatabaseTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="photo-intent-")
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
            accepted = await UserRepository(session).try_consume_photo_intent(
                user, day, user_limit, global_limit,
            )
            await session.commit()
            return accepted

    async def counts(self, day=DAY):
        async with self.factory() as session:
            rows = await session.scalars(select(PhotoIntentUsage).where(PhotoIntentUsage.date == day))
            return {row.scope_id: row.call_count for row in rows}

    async def test_user_limit_and_no_global_charge_on_rejection(self):
        self.assertEqual([await self.reserve() for _ in range(4)], [True, True, False, False])
        self.assertEqual(await self.counts(), {0: 2, 1: 2})

    async def test_global_limit_and_no_user_row_on_rejection(self):
        results = [await self.reserve(user=i, global_limit=2) for i in (1, 2, 3)]
        self.assertEqual(results, [True, True, False])
        self.assertEqual(await self.counts(), {0: 2, 1: 1, 2: 1})

    async def test_concurrent_last_user_units_are_atomic(self):
        results = await asyncio.gather(*(self.reserve() for _ in range(12)))
        self.assertEqual(sum(results), 2)
        self.assertEqual(await self.counts(), {0: 2, 1: 2})

    async def test_concurrent_global_cap_is_atomic(self):
        results = await asyncio.gather(*(
            self.reserve(user=i, global_limit=3) for i in range(1, 13)
        ))
        self.assertEqual(sum(results), 3)
        counts = await self.counts()
        self.assertEqual(counts.pop(0), 3)
        self.assertEqual(sum(counts.values()), 3)

    async def test_new_day_has_independent_budget(self):
        self.assertTrue(await self.reserve(user_limit=1))
        self.assertFalse(await self.reserve(user_limit=1))
        self.assertTrue(await self.reserve(user_limit=1, day=DAY + datetime.timedelta(days=1)))

    async def test_zero_limits_never_charge(self):
        self.assertFalse(await self.reserve(user_limit=0))
        self.assertFalse(await self.reserve(global_limit=0))
        self.assertEqual(await self.counts(), {})

    async def test_budget_survives_engine_restart(self):
        self.assertTrue(await self.reserve(user_limit=1))
        await self.engine.dispose()
        self.engine = create_async_engine(self.url)
        self.factory = async_sessionmaker(self.engine, expire_on_commit=False)
        self.assertFalse(await self.reserve(user_limit=1))

    async def test_service_commits_without_touching_question_quota(self):
        with (
            patch.object(config, "photo_intent_daily_limit", 1),
            patch.object(config, "photo_intent_global_daily_limit", 2),
        ):
            async with self.factory() as session:
                service = UserService(session)
                self.assertTrue(await service.consume_photo_intent_budget(1))
            async with self.factory() as session:
                self.assertFalse(await UserService(session).consume_photo_intent_budget(1))
                self.assertEqual(list(await session.scalars(select(DailyUsage))), [])


class PhotoBudgetRoutingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.message = SimpleNamespace(
            from_user=SimpleNamespace(id=1), caption="enhance this photo",
            chat=SimpleNamespace(id=1, type="private"), answer=AsyncMock(),
        )
        self.service = SimpleNamespace(
            get_user_settings=AsyncMock(return_value={"model": "text-model"}),
            consume_photo_intent_budget=AsyncMock(return_value=False),
            session=SimpleNamespace(close=AsyncMock()),
        )

    async def route(self):
        with (
            patch.object(chat, "_is_addressed", return_value=True),
            patch.object(chat, "_open_registered", AsyncMock(return_value=self.service)),
            patch.object(chat, "check_access", AsyncMock(return_value=AccessResult(True, "allowed"))),
            patch.object(chat, "detect_intent", AsyncMock(return_value=("enhance", []))) as detect,
            patch.object(chat, "_handle_enhance_photo", AsyncMock()) as enhance,
            patch.object(chat, "_handle_photo", AsyncMock()) as analyze,
        ):
            await chat._route_single_photo(self.message)
        return detect, enhance, analyze

    async def test_exhausted_budget_defaults_to_analysis_without_paid_call(self):
        detect, enhance, analyze = await self.route()
        detect.assert_not_awaited()
        enhance.assert_not_awaited()
        analyze.assert_awaited_once()

    async def test_clarify_path_reserves_before_each_paid_call(self):
        self.service.consume_photo_intent_budget.side_effect = [True, True, False]
        for accepted in (True, True, False):
            detect, enhance, analyze = await self.route()
            self.assertEqual(detect.await_count, int(accepted))
            self.assertEqual(enhance.await_count, int(accepted))
            self.assertEqual(analyze.await_count, int(not accepted))
        self.assertEqual(self.service.consume_photo_intent_budget.await_count, 3)

    async def test_empty_caption_does_not_use_helper_budget(self):
        self.message.caption = ""
        detect, _, analyze = await self.route()
        detect.assert_not_awaited()
        self.service.consume_photo_intent_budget.assert_not_awaited()
        analyze.assert_awaited_once()

    async def test_real_no_vision_branch_does_not_allow_unlimited_classification(self):
        self.message.reply_to_message = None
        self.service.consume_photo_intent_budget.side_effect = [True, False, False]
        with (
            patch.object(chat, "_is_addressed", return_value=True),
            patch.object(chat, "_open_registered", AsyncMock(return_value=self.service)),
            patch.object(chat, "check_access", AsyncMock(return_value=AccessResult(True, "guest"))),
            patch.object(chat, "detect_intent", AsyncMock(return_value=("analyze", []))) as detect,
            patch.object(chat, "is_vision_model", return_value=False),
            patch.object(chat, "_consume_access", AsyncMock()) as main_quota,
        ):
            for _ in range(3):
                await chat._route_single_photo(self.message)
        self.assertEqual(detect.await_count, 1)
        self.assertEqual(self.message.answer.await_count, 3)
        main_quota.assert_not_awaited()

    async def test_real_clarify_dialog_still_requires_budget(self):
        self.message.photo = [SimpleNamespace(file_id="f", file_size=1)]
        self.service.consume_photo_intent_budget.side_effect = [True, False]
        with (
            patch.dict(chat._pending_enhance, {}, clear=True),
            patch.object(chat, "_is_addressed", return_value=True),
            patch.object(chat, "_open_registered", AsyncMock(return_value=self.service)),
            patch.object(chat, "check_access", AsyncMock(return_value=AccessResult(True, "allowed"))),
            patch.object(chat, "detect_intent", AsyncMock(return_value=("enhance", []))) as detect,
            patch.object(chat, "_handle_photo", AsyncMock()),
            patch.object(chat, "safe_answer", AsyncMock()) as answer,
            patch.object(config, "enable_image_generation", True),
        ):
            for _ in range(2):
                await chat._route_single_photo(self.message)
            self.assertEqual(chat._pending_enhance[1]["file_ids"], ["f"])
        self.assertEqual(detect.await_count, 1)
        answer.assert_awaited_once()

    async def test_reservation_failure_is_fail_closed(self):
        self.service.consume_photo_intent_budget.side_effect = RuntimeError("database down")
        with self.assertRaises(RuntimeError):
            await self.route()

    async def test_guest_text_only_refund_cannot_bypass_helper_budget(self):
        service = SimpleNamespace(
            register_user=AsyncMock(), get_user_settings=AsyncMock(return_value={
                "model": "text-model", "system_prompt": "test",
            }), get_history=AsyncMock(return_value=[]), add_message=AsyncMock(),
            consume_photo_intent_budget=AsyncMock(side_effect=[True, False]),
        )
        caller = SimpleNamespace(id=1, username=None, first_name=None, last_name=None)
        with (
            patch.object(guest, "_extract_text", return_value="analyze this"),
            patch.object(guest, "_voice_of", return_value=None),
            patch.object(guest, "_document_of", return_value=None),
            patch.object(guest, "_download_photo", AsyncMock(return_value=b"photo")),
            patch.object(guest, "is_vision_model", return_value=False),
            patch.object(guest, "check_access", AsyncMock(return_value=AccessResult(True, "guest"))),
            patch.object(guest, "refund_access", AsyncMock()) as refund,
            patch.object(guest, "detect_intent", AsyncMock(return_value=("analyze", []))) as detect,
            patch.object(config, "enable_image_generation", True),
        ):
            for _ in range(2):
                await guest._build_reply(SimpleNamespace(), service, caller, 1)
        self.assertEqual(refund.await_count, 2)
        self.assertEqual(detect.await_count, 1)
        self.assertEqual(service.consume_photo_intent_budget.await_count, 2)
