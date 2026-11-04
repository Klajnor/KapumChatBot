"""Database integration tests (real SQLite in-memory).

These test the atomic quota operations and CRUD flows that currently have
no coverage at all. Each test gets its own fresh in-memory database.
"""

from __future__ import annotations

import asyncio
import datetime
from unittest import TestCase

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from bot.config import config
from bot.database.models import Base
from bot.database.repository import MessageRepository, UserRepository


def _make_session_factory():
    """Create an isolated in-memory SQLite session factory."""
    engine = create_async_engine(
        "sqlite+aiosqlite://", echo=False, poolclass=None, connect_args={"check_same_thread": False}
    )
    return engine, async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class DBTest(TestCase):
    """Base class: fresh in-memory SQLite per test."""

    def setUp(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._engine, self._factory = None, None
        self._loop.run_until_complete(self._setup())

    async def _setup(self) -> None:
        self._engine, self._factory = _make_session_factory()
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    def tearDown(self) -> None:
        async def _cleanup() -> None:
            async with self._engine.begin() as conn:
                await conn.run_sync(Base.metadata.drop_all)
            await self._engine.dispose()

        try:
            self._loop.run_until_complete(_cleanup())
        finally:
            self._loop.close()


class TestAtomicQuotas(DBTest):
    """Verify that concurrent quota consumption cannot exceed the limit."""

    def test_consume_daily_usage_up_to_limit(self):
        async def _run() -> None:
            async with self._factory() as session:
                repo = UserRepository(session)
                await repo.get_or_create_user(100)
                await repo.add_allowed_user(100)
                await session.commit()

            today = datetime.date.today()
            for _ in range(3):
                async with self._factory() as session:
                    repo = UserRepository(session)
                    result = await repo.try_consume_daily_usage(100, today, 3)
                    assert result is not None
                    await session.commit()

            async with self._factory() as session:
                repo = UserRepository(session)
                result = await repo.try_consume_daily_usage(100, today, 3)
                assert result is None
                await session.commit()

        self._loop.run_until_complete(_run())

    def test_release_and_re_consume_daily_usage(self):
        async def _run() -> None:
            async with self._factory() as session:
                repo = UserRepository(session)
                await repo.get_or_create_user(200)
                await repo.add_allowed_user(200)
                await session.commit()

            today = datetime.date.today()
            async with self._factory() as session:
                repo = UserRepository(session)
                result = await repo.try_consume_daily_usage(200, today, 1)
                assert result is not None
                await repo.release_daily_usage(200, today)
                await session.commit()

            async with self._factory() as session:
                repo = UserRepository(session)
                result = await repo.try_consume_daily_usage(200, today, 1)
                assert result is not None
                await session.commit()

        self._loop.run_until_complete(_run())

    def test_status_search_cooldown_is_persistent_and_atomic(self):
        async def _run() -> None:
            start = datetime.datetime(2026, 1, 1, 12, 0, tzinfo=datetime.UTC)
            async with self._factory() as session:
                repo = UserRepository(session)
                self.assertIsNone(
                    await repo.try_reserve_status_search(700, start, 3600)
                )
                await session.commit()

            async with self._factory() as session:
                repo = UserRepository(session)
                previous = await repo.try_reserve_status_search(
                    700, start + datetime.timedelta(minutes=30), 3600
                )
                self.assertIsNotNone(previous)
                await session.commit()

            async with self._factory() as session:
                repo = UserRepository(session)
                self.assertIsNone(
                    await repo.try_reserve_status_search(
                        700, start + datetime.timedelta(minutes=61), 3600
                    )
                )
                await session.commit()

        self._loop.run_until_complete(_run())

    def test_guest_messages_limited(self):
        async def _run() -> None:
            async with self._factory() as session:
                repo = UserRepository(session)
                await repo.get_or_create_user(300)
                await session.commit()

            for _ in range(2):
                async with self._factory() as session:
                    repo = UserRepository(session)
                    result = await repo.try_consume_guest_message(300, 2)
                    assert result is not None
                    await session.commit()

            async with self._factory() as session:
                repo = UserRepository(session)
                result = await repo.try_consume_guest_message(300, 2)
                assert result is None
                await session.commit()

        self._loop.run_until_complete(_run())

    def test_guest_release_and_re_consume(self):
        async def _run() -> None:
            async with self._factory() as session:
                repo = UserRepository(session)
                await repo.get_or_create_user(400)
                await session.commit()

            async with self._factory() as session:
                repo = UserRepository(session)
                result = await repo.try_consume_guest_message(400, 1)
                assert result is not None
                await repo.release_guest_message(400)
                await session.commit()

            async with self._factory() as session:
                repo = UserRepository(session)
                result = await repo.try_consume_guest_message(400, 1)
                assert result is not None
                await session.commit()

        self._loop.run_until_complete(_run())

    def test_parallel_races_daily_limit(self):
        """Simulate concurrent requests racing for the last quota unit."""
        async def _run() -> None:
            async with self._factory() as session:
                repo = UserRepository(session)
                await repo.get_or_create_user(500)
                await repo.add_allowed_user(500)
                await session.commit()

            today = datetime.date.today()

            async def consume() -> int | None:
                async with self._factory() as session:
                    repo = UserRepository(session)
                    result = await repo.try_consume_daily_usage(500, today, 2)
                    await session.commit()
                    return result

            results = await asyncio.gather(consume(), consume(), consume(), consume())
            successes = [r for r in results if r is not None]
            self.assertEqual(len(successes), 2)

        self._loop.run_until_complete(_run())


class TestUserCRUD(DBTest):
    """Basic user and settings CRUD flows."""

    def test_register_and_get_settings(self):
        async def _run() -> None:
            async with self._factory() as session:
                repo = UserRepository(session)
                user = await repo.get_or_create_user(
                    111, username="tester", first_name="Test", last_name="User"
                )
                await session.commit()
                self.assertEqual(user.telegram_id, 111)
                self.assertEqual(user.username, "tester")

            async with self._factory() as session:
                repo = UserRepository(session)
                settings = await repo.get_user_settings(111)
                self.assertIsNotNone(settings)
                self.assertEqual(settings.selected_model, config.default_model)

        self._loop.run_until_complete(_run())

    def test_update_model_and_prompt(self):
        async def _run() -> None:
            async with self._factory() as session:
                repo = UserRepository(session)
                await repo.get_or_create_user(222)
                await session.commit()

            async with self._factory() as session:
                repo = UserRepository(session)
                await repo.update_model(222, "qwen3.8-max")
                await repo.update_system_prompt(222, "Be brief.")
                await session.commit()

            async with self._factory() as session:
                repo = UserRepository(session)
                settings = await repo.get_user_settings(222)
                self.assertEqual(settings.selected_model, "qwen3.8-max")
                self.assertEqual(settings.system_prompt, "Be brief.")

        self._loop.run_until_complete(_run())

    def test_allowed_users_list(self):
        async def _run() -> None:
            async with self._factory() as session:
                repo = UserRepository(session)
                await repo.add_allowed_user(1001, added_by=1)
                await repo.add_allowed_user(1002, added_by=1)
                await session.commit()

            async with self._factory() as session:
                repo = UserRepository(session)
                self.assertTrue(await repo.is_allowed_user(1001))
                self.assertTrue(await repo.is_allowed_user(1002))
                self.assertFalse(await repo.is_allowed_user(9999))

                all_users = await repo.get_allowed_users()
                self.assertEqual(len(all_users), 2)

                await repo.remove_allowed_user(1001)
                await session.commit()

            async with self._factory() as session:
                repo = UserRepository(session)
                self.assertFalse(await repo.is_allowed_user(1001))

        self._loop.run_until_complete(_run())

    def test_allowed_users_profile_cache(self):
        """Nicknames survive on allowed_users until the user registers."""

        async def _run() -> None:
            async with self._factory() as session:
                repo = UserRepository(session)
                await repo.add_allowed_user(
                    2001, added_by=1, username="nick", first_name="Nik"
                )
                await repo.add_allowed_user(2002, added_by=1)
                await session.commit()

            # A user who starts the bot refreshes the cached profile.
            async with self._factory() as session:
                repo = UserRepository(session)
                await repo.get_or_create_user(
                    2002, username="later", first_name="Later"
                )
                await session.commit()

            async with self._factory() as session:
                repo = UserRepository(session)
                rows = {
                    row.telegram_id: row
                    for row in await repo.get_allowed_users()
                }
                self.assertEqual(rows[2001].username, "nick")
                self.assertEqual(rows[2001].first_name, "Nik")
                self.assertEqual(rows[2002].username, "later")

                found = await repo.find_user_by_username("@LATER")
                self.assertIsNotNone(found)
                self.assertEqual(found.telegram_id, 2002)

        self._loop.run_until_complete(_run())


class TestMessages(DBTest):
    """Message history CRUD."""

    def test_add_and_get_history(self):
        async def _run() -> None:
            async with self._factory() as session:
                repo = MessageRepository(session)
                await repo.add_message(chat_id=42, user_id=700, role="user",
                                       content="Hello", model="qwen")
                await repo.add_message(chat_id=42, user_id=700, role="assistant",
                                       content="Hi there!", model="qwen")
                await session.commit()

                history = await repo.get_history(42, user_id=700)
                self.assertEqual(len(history), 2)
                self.assertEqual(history[0]["role"], "user")
                self.assertEqual(history[1]["role"], "assistant")

                # clear
                await repo.clear_history(42, user_id=700)
                await session.commit()
                history = await repo.get_history(42, user_id=700)
                self.assertEqual(len(history), 0)

        self._loop.run_until_complete(_run())

    def test_history_respects_limit(self):
        async def _run() -> None:
            async with self._factory() as session:
                repo = MessageRepository(session)
                for i in range(50):
                    await repo.add_message(chat_id=43, user_id=701, role="user",
                                           content=f"msg {i}")
                await session.commit()

                history = await repo.get_history(43, user_id=701, limit=5)
                self.assertEqual(len(history), 5)
                self.assertEqual(history[0]["content"], "msg 45")
                self.assertEqual(history[-1]["content"], "msg 49")

        self._loop.run_until_complete(_run())


class TestDailyUsageStats(DBTest):
    """14-day usage statistics aggregation."""

    def test_stats_aggregation(self):
        async def _run() -> None:
            async with self._factory() as session:
                repo = UserRepository(session)
                await repo.get_or_create_user(800)
                await repo.add_allowed_user(800)
                await session.commit()

            today = datetime.date.today()
            yesterday = today - datetime.timedelta(days=1)

            async with self._factory() as session:
                repo = UserRepository(session)
                await repo.try_consume_daily_usage(800, today, 100)
                await repo.try_consume_daily_usage(800, today, 100)
                await repo.try_consume_daily_usage(800, yesterday, 100)
                await session.commit()

            async with self._factory() as session:
                repo = UserRepository(session)
                stats = await repo.get_usage_stats_last_14_days()
                self.assertGreaterEqual(len(stats), 2)

                total = await repo.get_user_total_usage_last_14_days(800)
                self.assertEqual(total, 3)

        self._loop.run_until_complete(_run())

    def test_reset_daily_usage(self):
        async def _run() -> None:
            async with self._factory() as session:
                repo = UserRepository(session)
                await repo.get_or_create_user(900)
                await repo.add_allowed_user(900)
                await session.commit()

            today = datetime.date.today()
            async with self._factory() as session:
                repo = UserRepository(session)
                await repo.try_consume_daily_usage(900, today, 100)
                await session.commit()

            async with self._factory() as session:
                repo = UserRepository(session)
                usage = await repo.get_daily_usage(900, today)
                self.assertEqual(usage.query_count, 1)

                await repo.reset_daily_usage(900, today)
                await session.commit()

                usage = await repo.get_daily_usage(900, today)
                self.assertEqual(usage.query_count, 0)

        self._loop.run_until_complete(_run())


class TestUserServiceHelper(DBTest):
    """The user_service() context manager must yield a working service."""

    def test_context_manager_yields_working_service(self):
        from unittest.mock import patch

        from bot.services.user_service import user_service

        async def _run() -> None:
            with patch("bot.database.session.async_session_factory", self._factory):
                async with user_service() as service:
                    await service.register_user(1234, username="ctx")

                async with user_service() as service:
                    settings = await service.get_user_settings(1234)
                    self.assertIsNotNone(settings)
                    self.assertEqual(settings["model"], config.default_model)

        self._loop.run_until_complete(_run())


class TestReminders(DBTest):
    """Reminder persistence: limits, ownership and the scheduler claim step."""

    def test_count_separates_one_time_and_recurring(self):
        from bot.database.models import Reminder
        from bot.database.repository import ReminderRepository

        async def _run() -> None:
            now = datetime.datetime.now(datetime.UTC)
            async with self._factory() as session:
                repo = UserRepository(session)
                await repo.get_or_create_user(1000)
                await session.commit()

            async with self._factory() as session:
                repo = ReminderRepository(session)
                await repo.add(Reminder(
                    user_id=1000, chat_id=1000, text="one-time", fire_at=now,
                ))
                await repo.add(Reminder(
                    user_id=1000, chat_id=1000, text="recurring", fire_at=now,
                    schedule="daily:09:00",
                ))
                await session.commit()

                self.assertEqual(await repo.count(1000), 2)
                self.assertEqual(await repo.count(1000, recurring=True), 1)
                self.assertEqual(await repo.count(1000, recurring=False), 1)

        self._loop.run_until_complete(_run())

    def test_delete_is_scoped_to_owner(self):
        from bot.database.models import Reminder
        from bot.database.repository import ReminderRepository

        async def _run() -> None:
            now = datetime.datetime.now(datetime.UTC)
            async with self._factory() as session:
                repo = UserRepository(session)
                await repo.get_or_create_user(2000)
                await repo.get_or_create_user(2001)
                await session.commit()

            async with self._factory() as session:
                repo = ReminderRepository(session)
                created = await repo.add(Reminder(
                    user_id=2000, chat_id=2000, text="mine", fire_at=now,
                ))
                await session.commit()
                reminder_id = created.id

            async with self._factory() as session:
                repo = ReminderRepository(session)
                # Another user cannot delete it.
                self.assertFalse(await repo.delete(reminder_id, 2001))
                # The owner can.
                self.assertTrue(await repo.delete(reminder_id, 2000))
                await session.commit()

        self._loop.run_until_complete(_run())

    def test_get_due_returns_only_enabled_past_reminders(self):
        from bot.database.models import Reminder
        from bot.database.repository import ReminderRepository

        async def _run() -> None:
            now = datetime.datetime.now(datetime.UTC)
            past = now - datetime.timedelta(minutes=5)
            future = now + datetime.timedelta(minutes=5)
            async with self._factory() as session:
                repo = UserRepository(session)
                await repo.get_or_create_user(3000)
                await session.commit()

            async with self._factory() as session:
                repo = ReminderRepository(session)
                await repo.add(Reminder(
                    user_id=3000, chat_id=3000, text="due", fire_at=past,
                ))
                await repo.add(Reminder(
                    user_id=3000, chat_id=3000, text="later", fire_at=future,
                ))
                disabled = await repo.add(Reminder(
                    user_id=3000, chat_id=3000, text="off", fire_at=past,
                ))
                await repo.disable(disabled.id)
                await session.commit()

            async with self._factory() as session:
                repo = ReminderRepository(session)
                due = await repo.get_due(now)
                self.assertEqual([r.text for r in due], ["due"])

        self._loop.run_until_complete(_run())

    def test_reschedule_advances_recurring_reminder(self):
        from bot.database.models import Reminder
        from bot.database.repository import ReminderRepository

        async def _run() -> None:
            now = datetime.datetime.now(datetime.UTC)
            async with self._factory() as session:
                repo = UserRepository(session)
                await repo.get_or_create_user(4000)
                await session.commit()

            async with self._factory() as session:
                repo = ReminderRepository(session)
                created = await repo.add(Reminder(
                    user_id=4000, chat_id=4000, text="hourly", fire_at=now,
                    schedule="every:60",
                ))
                await session.commit()
                reminder_id = created.id

            next_fire = now + datetime.timedelta(minutes=60)
            async with self._factory() as session:
                repo = ReminderRepository(session)
                reminder = await repo.get(reminder_id, 4000)
                await repo.reschedule(reminder, next_fire)
                await session.commit()

            async with self._factory() as session:
                repo = ReminderRepository(session)
                # Not due at the original time any more.
                self.assertEqual(await repo.get_due(now), [])
                reminder = await repo.get(reminder_id, 4000)
                # SQLite drops tzinfo on round-trip; normalize like the
                # scheduler does before comparing.
                stored = reminder.fire_at
                if stored.tzinfo is None:
                    stored = stored.replace(tzinfo=datetime.UTC)
                self.assertEqual(stored, next_fire)

        self._loop.run_until_complete(_run())

class TestExportQueries(DBTest):
    """Repository queries backing the conversation export."""

    def test_export_history_queries(self):
        from bot.database.repository import MessageRepository

        async def _run() -> None:
            async with self._factory() as session:
                urepo = UserRepository(session)
                await urepo.get_or_create_user(5000)
                await urepo.get_or_create_user(5001)
                mrepo = MessageRepository(session)
                q1 = await mrepo.add_message(
                    chat_id=77, user_id=5000, role="user", content="q1"
                )
                a1 = await mrepo.add_message(
                    chat_id=77, user_id=5000, role="assistant", content="a1"
                )
                q2 = await mrepo.add_message(
                    chat_id=77, user_id=5000, role="user", content="q2"
                )
                other = await mrepo.add_message(
                    chat_id=77, user_id=5001, role="user", content="other"
                )
                await session.commit()
                ids = {"q1": q1.id, "a1": a1.id, "q2": q2.id, "other": other.id}

            async with self._factory() as session:
                mrepo = MessageRepository(session)
                row = await mrepo.get_message(ids["a1"])
                self.assertIsNotNone(row)
                self.assertEqual(row.content, "a1")

                prev = await mrepo.get_previous_user_message(77, 5000, ids["a1"])
                self.assertIsNotNone(prev)
                self.assertEqual(prev.content, "q1")

                rows = await mrepo.get_messages_for_export(77, 5000, 10)
                self.assertEqual([r.content for r in rows], ["q1", "a1", "q2"])

                rows = await mrepo.get_messages_for_export(77, 5000, 2)
                self.assertEqual([r.content for r in rows], ["a1", "q2"])

        self._loop.run_until_complete(_run())

    def test_get_message_for_export_is_owner_scoped(self):
        from bot.database.repository import MessageRepository
        from bot.services.user_service import UserService

        async def _run() -> None:
            async with self._factory() as session:
                urepo = UserRepository(session)
                await urepo.get_or_create_user(6000)
                await urepo.get_or_create_user(6001)
                mrepo = MessageRepository(session)
                saved = await mrepo.add_message(
                    chat_id=6000, user_id=6000, role="assistant", content="secret"
                )
                await session.commit()
                message_id = saved.id

            async with self._factory() as session:
                service = UserService(session)
                own = await service.get_message_for_export(message_id, 6000)
                self.assertIsNotNone(own)
                foreign = await service.get_message_for_export(message_id, 6001)
                self.assertIsNone(foreign)
                missing = await service.get_message_for_export(999999, 6000)
                self.assertIsNone(missing)

        self._loop.run_until_complete(_run())


class TestDefaultPromptMigration(DBTest):
    """The legacy English default prompt is replaced; custom prompts survive."""

    def test_legacy_default_prompt_is_replaced(self):
        from sqlalchemy import text as sql_text

        from bot.config import LEGACY_DEFAULT_SYSTEM_PROMPT
        from bot.database.session import _run_data_migrations

        async def _run() -> None:
            async with self._factory() as session:
                repo = UserRepository(session)
                await repo.get_or_create_user(7000)
                await repo.get_or_create_user(7001)
                await session.execute(
                    sql_text(
                        "UPDATE user_settings SET system_prompt = :p "
                        "WHERE user_id = 7000"
                    ),
                    {"p": LEGACY_DEFAULT_SYSTEM_PROMPT},
                )
                await session.execute(
                    sql_text(
                        "UPDATE user_settings SET system_prompt = :p "
                        "WHERE user_id = 7001"
                    ),
                    {"p": "Мой собственный промпт"},
                )
                await session.commit()

            async with self._engine.begin() as conn:
                await _run_data_migrations(conn)

            async with self._factory() as session:
                repo = UserRepository(session)
                migrated = await repo.get_user_settings(7000)
                custom = await repo.get_user_settings(7001)
                self.assertEqual(migrated.system_prompt, config.default_system_prompt)
                self.assertNotEqual(migrated.system_prompt, LEGACY_DEFAULT_SYSTEM_PROMPT)
                self.assertEqual(custom.system_prompt, "Мой собственный промпт")

        self._loop.run_until_complete(_run())


class TestPersonaPromptMigration(DBTest):
    """Pre-name prompts are upgraded to their named built-in versions."""

    def test_previous_default_prompt_gets_name(self):
        from sqlalchemy import text as sql_text

        from bot.config import PREVIOUS_DEFAULT_SYSTEM_PROMPT_TEXT
        from bot.database.session import _run_data_migrations

        self.assertNotEqual(
            PREVIOUS_DEFAULT_SYSTEM_PROMPT_TEXT, config.default_system_prompt
        )

        async def _run() -> None:
            async with self._factory() as session:
                repo = UserRepository(session)
                await repo.get_or_create_user(7100)
                await repo.get_or_create_user(7101)
                await session.execute(
                    sql_text(
                        "UPDATE user_settings SET system_prompt = :p "
                        "WHERE user_id = 7100"
                    ),
                    {"p": PREVIOUS_DEFAULT_SYSTEM_PROMPT_TEXT},
                )
                await session.execute(
                    sql_text(
                        "UPDATE user_settings SET system_prompt = :p "
                        "WHERE user_id = 7101"
                    ),
                    {"p": "Мой собственный промпт"},
                )
                await session.commit()

            async with self._engine.begin() as conn:
                await _run_data_migrations(conn)

            async with self._factory() as session:
                repo = UserRepository(session)
                migrated = await repo.get_user_settings(7100)
                custom = await repo.get_user_settings(7101)
                self.assertEqual(migrated.system_prompt, config.default_system_prompt)
                self.assertEqual(custom.system_prompt, "Мой собственный промпт")

        self._loop.run_until_complete(_run())

    def test_old_builtin_persona_prompt_gets_name(self):
        from sqlalchemy import text as sql_text

        from bot.config import DEFAULT_PERSONAS, legacy_persona_prompts
        from bot.database.session import _run_data_migrations

        persona = DEFAULT_PERSONAS["programmer"]
        legacy = next(
            old
            for old, new in legacy_persona_prompts().items()
            if new == persona["prompt"]
        )

        async def _run() -> None:
            async with self._factory() as session:
                repo = UserRepository(session)
                await repo.get_or_create_user(7200)
                await repo.get_or_create_user(7201)
                await session.execute(
                    sql_text(
                        "UPDATE user_settings SET system_prompt = :p "
                        "WHERE user_id = 7200"
                    ),
                    {"p": legacy},
                )
                await session.execute(
                    sql_text(
                        "UPDATE user_settings SET system_prompt = :p "
                        "WHERE user_id = 7201"
                    ),
                    {"p": "Мой собственный промпт"},
                )
                await session.commit()

            async with self._engine.begin() as conn:
                await _run_data_migrations(conn)

            async with self._factory() as session:
                repo = UserRepository(session)
                migrated = await repo.get_user_settings(7200)
                custom = await repo.get_user_settings(7201)
                self.assertEqual(migrated.system_prompt, persona["prompt"])
                self.assertEqual(custom.system_prompt, "Мой собственный промпт")

        self._loop.run_until_complete(_run())
