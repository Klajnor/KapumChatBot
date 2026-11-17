"""Статистика использования: имена вместо ID и rich-таблица."""

from __future__ import annotations

import datetime
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import AsyncMock, patch

from aiogram.types import Chat, Message, User

from bot.config import config
from bot.handlers import admin
from bot.handlers import admin_panel as panel
from bot.services.usage_report import (
    MAX_NAME_LENGTH,
    aggregate_usage,
    table_html,
    table_markdown,
    usage_name,
)
from bot.services.user_service import user_service
from tests.test_database import DBTest

OWNER = 1
NOW = datetime.datetime(2026, 11, 17, 12, 0, tzinfo=datetime.UTC)


def _entry(telegram_id: int, date: datetime.date, count: int) -> dict:
    return {"telegram_id": telegram_id, "date": date, "query_count": count}


class TestUsageNames(TestCase):
    def test_name_then_username_then_id(self):
        self.assertEqual(
            usage_name(
                {"first_name": "Анна", "last_name": "Петрова", "username": "ann"}, 1
            ),
            "Анна Петрова",
        )
        self.assertEqual(
            usage_name({"first_name": None, "last_name": None, "username": "ann"}, 1),
            "@ann",
        )
        self.assertEqual(usage_name(None, 42), "42")

    def test_long_name_is_clipped(self):
        name = usage_name({"first_name": "Я" * (MAX_NAME_LENGTH + 10)}, 1)
        self.assertEqual(len(name), MAX_NAME_LENGTH)
        self.assertTrue(name.endswith("…"))

    def test_aggregate_sums_sorts_and_skips_empty_days(self):
        today = datetime.date.today()
        yesterday = today - datetime.timedelta(days=1)
        rows = aggregate_usage(
            [
                _entry(1, today, 2),
                _entry(1, yesterday, 3),
                _entry(2, today, 10),
                _entry(3, today, 0),
            ],
            [
                {"telegram_id": 1, "first_name": "Анна", "last_name": "", "username": None},
                {"telegram_id": 2, "first_name": None, "last_name": None, "username": "bob"},
                {"telegram_id": 3, "first_name": "Пустой", "last_name": None, "username": None},
            ],
        )
        self.assertEqual([row["telegram_id"] for row in rows], [2, 1])
        self.assertEqual(rows[0]["name"], "@bob")
        self.assertEqual(rows[1]["name"], "Анна")
        self.assertEqual(rows[1]["total"], 5)
        self.assertEqual(rows[1]["today"], 2)
        self.assertEqual(rows[1]["days"], 2)

    def test_markdown_table_has_header_and_escapes_pipes(self):
        rows = [{"name": "A | B", "today": 1, "total": 2, "days": 1}]
        markdown = table_markdown(rows)
        self.assertIn("| # | User | Today | 14 days | Days |", markdown)
        self.assertIn(r"A \| B", markdown)

    def test_markdown_limit_mentions_hidden_users(self):
        rows = [
            {"name": f"User {i}", "today": 0, "total": i, "days": 1}
            for i in range(3)
        ]
        markdown = table_markdown(rows, limit=1)
        self.assertIn("_2 more users not shown._", markdown)

    def test_html_fallback_escapes_and_limits(self):
        rows = [
            {"name": "<b>x</b>", "today": 1, "total": 2, "days": 1},
            {"name": "Second", "today": 0, "total": 1, "days": 1},
        ]
        rendered = table_html(rows, limit=1)
        self.assertIn("&lt;b&gt;x&lt;/b&gt;", rendered)
        self.assertIn("... and 1 more users.", rendered)


class TestUsageReportDB(DBTest):
    def test_report_uses_profile_names(self):
        async def go() -> None:
            today = datetime.date.today()
            with patch("bot.database.session.async_session_factory", self._factory):
                async with user_service() as service:
                    await service.register_user(
                        100, first_name="Анна", last_name="Петрова"
                    )
                    await service.add_allowed_user(100)
                    await service.register_user(200, username="bob")
                    await service.add_allowed_user(200)
                    await service.user_repo.try_consume_daily_usage(100, today, 100)
                    await service.user_repo.try_consume_daily_usage(100, today, 100)
                    await service.user_repo.try_consume_daily_usage(200, today, 100)
                    await service.session.commit()

                    rows = await service.get_usage_report()

            self.assertEqual([row["name"] for row in rows], ["Анна Петрова", "@bob"])
            self.assertEqual([row["total"] for row in rows], [2, 1])
            self.assertEqual(rows[0]["today"], 2)
            self.assertEqual(rows[1]["today"], 1)

        self._loop.run_until_complete(go())

    def test_empty_report(self):
        async def go() -> None:
            with patch("bot.database.session.async_session_factory", self._factory):
                async with user_service() as service:
                    self.assertEqual(await service.get_usage_report(), [])

        self._loop.run_until_complete(go())


class TestUsageScreens(DBTest):
    async def _seed(self) -> None:
        today = datetime.date.today()
        with patch("bot.database.session.async_session_factory", self._factory):
            async with user_service() as service:
                await service.register_user(
                    100, first_name="Анна", last_name="Петрова"
                )
                await service.add_allowed_user(100)
                await service.user_repo.try_consume_daily_usage(100, today, 100)
                await service.user_repo.try_consume_daily_usage(100, today, 100)
                await service.session.commit()

    def _dm(self) -> Message:
        return Message(
            message_id=1,
            date=NOW,
            chat=Chat(id=OWNER, type="private"),
            from_user=User(id=OWNER, is_bot=False, first_name="Ann"),
            text="/admin_stats",
        )

    def test_command_uses_rich_table_then_html_fallback(self):
        async def go() -> None:
            await self._seed()
            message = self._dm()
            rich = AsyncMock(return_value=True)
            answer = AsyncMock()
            with (
                patch("bot.database.session.async_session_factory", self._factory),
                patch.object(config, "owner_ids", [OWNER]),
                patch.object(
                    admin, "require_private_chat", AsyncMock(return_value=True)
                ),
                patch.object(admin, "send_rich", rich),
                patch.object(Message, "answer", answer),
            ):
                await admin.cmd_admin_stats(message)

            rich.assert_awaited_once()
            markdown = rich.await_args.args[1]
            self.assertIn("| # | User |", markdown)
            self.assertIn("Анна Петрова", markdown)
            self.assertIn("Today: **2**", markdown)
            answer.assert_not_awaited()

            answer.reset_mock()
            with (
                patch("bot.database.session.async_session_factory", self._factory),
                patch.object(config, "owner_ids", [OWNER]),
                patch.object(config, "rich_messages", False),
                patch.object(
                    admin, "require_private_chat", AsyncMock(return_value=True)
                ),
                patch.object(Message, "answer", answer),
            ):
                await admin.cmd_admin_stats(message)

            answer.assert_awaited_once()
            text = answer.await_args.args[0]
            self.assertIn("Анна Петрова", text)
            self.assertIn("Today: <b>2</b>", text)

        self._loop.run_until_complete(go())

    def test_panel_uses_names_and_a_rich_table(self):
        async def go() -> None:
            await self._seed()
            callback = SimpleNamespace(
                from_user=SimpleNamespace(id=OWNER),
                message=SimpleNamespace(
                    chat=SimpleNamespace(id=OWNER, type="private"),
                    delete=AsyncMock(),
                ),
            )
            rich = AsyncMock(return_value=True)
            show = AsyncMock()
            with (
                patch("bot.database.session.async_session_factory", self._factory),
                patch.object(config, "owner_ids", [OWNER]),
                patch.object(config, "rich_messages", True),
                patch.object(panel, "send_rich", rich),
                patch.object(panel, "_show", show),
                patch.object(panel, "safe_delete", AsyncMock()),
            ):
                await panel._show_stats(callback)

            rich.assert_awaited_once()
            markdown = rich.await_args.args[1]
            self.assertIn("| # | User |", markdown)
            self.assertIn("Анна Петрова", markdown)
            self.assertIn("reply_markup", rich.await_args.kwargs)
            show.assert_not_awaited()

        self._loop.run_until_complete(go())