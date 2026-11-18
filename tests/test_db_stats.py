"""Карточка «🗄 Database»: сбор статистики и owner-гейт.

Сбор гоняется против реального временного файла SQLite со схемой проекта;
обработчик проверяется на SimpleNamespace-callback'ах по образцу
``test_admin_safety``.
"""

from __future__ import annotations

import dataclasses
import datetime
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from sqlalchemy import create_engine, insert

from bot.config import config
from bot.database.models import (
    AllowedUser,
    Base,
    DailyUsage,
    Message,
    User,
)
from bot.handlers import admin_panel as panel
from bot.services import db_stats

OWNER = 1
USER = 3
NOW = datetime.datetime(2026, 11, 19, 12, 0, tzinfo=datetime.UTC)


def _callback(data: str, *, owner: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        data=data,
        from_user=SimpleNamespace(id=OWNER if owner else USER),
        message=SimpleNamespace(
            chat=SimpleNamespace(id=OWNER, type="private")
        ),
        answer=AsyncMock(),
    )


class _TempFileDB:
    """Временный SQLite-файл со схемой проекта и парой строк данных."""

    def __init__(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.path = str(Path(self._dir.name) / "chat_bot.db")
        engine = create_engine(f"sqlite:///{self.path}")
        Base.metadata.create_all(engine)
        now = datetime.datetime.now(datetime.UTC)
        with engine.begin() as conn:
            conn.execute(
                insert(User).values(
                    telegram_id=USER,
                    first_name="Ann",
                    created_at=now,
                )
            )
            conn.execute(
                insert(User).values(
                    telegram_id=OWNER,
                    first_name="Ow",
                    created_at=now,
                )
            )
            conn.execute(
                insert(Message).values(
                    user_id=USER,
                    chat_id=USER,
                    role="user",
                    content="hello",
                    created_at=now,
                )
            )
            conn.execute(
                insert(AllowedUser).values(telegram_id=USER, added_at=now)
            )
            conn.execute(
                insert(DailyUsage).values(
                    telegram_id=USER, date=now.date(), query_count=2
                )
            )
        engine.dispose()

    def cleanup(self) -> None:
        self._dir.cleanup()


class CollectTests(unittest.TestCase):
    """Сбор статистики против реального файла БД."""

    def setUp(self) -> None:
        db_stats.reset_cache()
        self.db = _TempFileDB()
        self._patch = patch.object(config, "database_url", f"sqlite+aiosqlite:///{self.db.path}")
        self._patch.start()

    def tearDown(self) -> None:
        self._patch.stop()
        db_stats.reset_cache()
        self.db.cleanup()

    def test_counts_and_notes(self) -> None:
        stats = db_stats.collect_file(self.db.path)
        self.assertIsNone(stats.error)
        rows = {table.name: table for table in stats.tables}
        self.assertEqual(rows["messages"].rows, 1)
        self.assertEqual(rows["users"].rows, 2)
        self.assertEqual(rows["allowed_users"].rows, 1)
        self.assertEqual(rows["daily_usage"].rows, 1)
        # Каждая таблица моделей описана; неизвестных нет.
        model_tables = set(Base.metadata.tables)
        self.assertTrue(model_tables.issubset(db_stats._TABLE_NOTES))
        for name in model_tables:
            self.assertNotEqual(rows[name].note, db_stats._UNKNOWN_NOTE, name)
        self.assertGreater(stats.file_bytes, 0)
        self.assertEqual(stats.page_count, stats.file_bytes // stats.page_size)
        self.assertIn(stats.journal_mode.lower(), ("delete", "wal", "truncate"))

    def test_unknown_table_is_flagged(self) -> None:
        conn = sqlite3.connect(self.db.path)
        conn.execute("CREATE TABLE mystery_table (id INTEGER)")
        conn.execute("INSERT INTO mystery_table VALUES (7)")
        conn.commit()
        conn.close()
        stats = db_stats.collect_file(self.db.path)
        rows = {table.name: table for table in stats.tables}
        self.assertIn("mystery_table", rows)
        self.assertEqual(rows["mystery_table"].rows, 1)
        self.assertEqual(rows["mystery_table"].note, db_stats._UNKNOWN_NOTE)

    def test_reports_render(self) -> None:
        stats = db_stats.collect_file(self.db.path)
        markdown = db_stats.markdown_report(stats)
        html_text = db_stats.html_report(stats)
        self.assertIn("chat_bot.db", markdown)
        self.assertIn("| 1 |", markdown)
        self.assertIn("messages", html_text)
        self.assertIn("chat history", html_text)

    def test_error_report(self) -> None:
        broken = dataclasses.replace(
            db_stats.DatabaseStats(path=self.db.path),
            error="boom",
        )
        self.assertEqual(db_stats.markdown_report(broken), "⚠️ boom")
        self.assertEqual(db_stats.html_report(broken), "⚠️ boom")

    def test_fmt_bytes(self) -> None:
        self.assertEqual(db_stats.fmt_bytes(None), "—")
        self.assertEqual(db_stats.fmt_bytes(512), "512 B")
        self.assertEqual(db_stats.fmt_bytes(2048), "2.0 KB")
        self.assertEqual(db_stats.fmt_bytes(5 * 1024 * 1024), "5.0 MB")


class AsyncCollectTests(unittest.IsolatedAsyncioTestCase):
    """Кэш и force-обновление через async-обёртку."""

    async def test_cache_and_force(self) -> None:
        db = _TempFileDB()
        db_stats.reset_cache()
        try:
            with patch.object(
                config, "database_url", f"sqlite+aiosqlite:///{db.path}"
            ):
                first = await db_stats.collect()
                second = await db_stats.collect()
                self.assertIs(first, second)
                # Новая строка без force не видна из кэша…
                conn = sqlite3.connect(db.path)
                conn.execute(
                    "INSERT INTO users (telegram_id, is_admin, guest_messages_used)"
                    " VALUES (9, 0, 0)"
                )
                conn.commit()
                conn.close()
                cached = await db_stats.collect()
                rows = {t.name: t.rows for t in cached.tables}
                self.assertEqual(rows["users"], 2)

                forced = await db_stats.collect(force=True)
                rows = {t.name: t.rows for t in forced.tables}
                self.assertEqual(rows["users"], 3)
        finally:
            db_stats.reset_cache()
            db.cleanup()

    async def test_memory_database_reports_error(self) -> None:
        db_stats.reset_cache()
        with patch.object(config, "database_url", "sqlite+aiosqlite:///:memory:"):
            stats = await db_stats.collect(force=True)
            self.assertIsNotNone(stats.error)
            self.assertIn("not a SQLite file path", stats.error or "")
        db_stats.reset_cache()


class HandlerGateTests(unittest.IsolatedAsyncioTestCase):
    """Карточка доступна только владельцу; rich/HTML-пути не падают."""

    def setUp(self) -> None:
        db_stats.reset_cache()
        self._patch = patch.object(config, "owner_ids", [OWNER])
        self._patch.start()

    def tearDown(self) -> None:
        self._patch.stop()
        db_stats.reset_cache()

    async def test_owner_gate(self) -> None:
        answered = AsyncMock()
        with (
            patch.object(panel, "safe_answer_callback", answered),
            patch.object(
                panel,
                "send_rich",
                AsyncMock(side_effect=AssertionError("must not render")),
            ),
        ):
            await panel._show_database(_callback("adm:db", owner=False))
        answered.assert_awaited_once()
        self.assertEqual(answered.await_args.args[1], "Owners only")

    async def test_owner_card_html_path(self) -> None:
        stats = dataclasses.replace(
            db_stats.DatabaseStats(path="/tmp/x.db"),
            tables=[
                db_stats.TableStat(
                    name="messages", rows=3, bytes=100, note="chat history"
                )
            ],
        )
        shown = AsyncMock()
        with (
            patch.object(db_stats, "collect", AsyncMock(return_value=stats)),
            patch.object(panel, "_show", shown),
            patch.object(panel, "send_rich", AsyncMock(return_value=False)),
            patch.object(config, "rich_messages", True),
        ):
            await panel._show_database(_callback("adm:db"))
        shown.assert_awaited_once()
        _, text, markup = shown.await_args.args
        self.assertIn("Database", text)
        self.assertIn("messages", text)
        # Кнопки: Refresh и Menu.
        self.assertEqual(len(markup.inline_keyboard), 1)
        labels = [b.text for b in markup.inline_keyboard[0]]
        self.assertEqual(labels, ["♻ Refresh", "🏠 Menu"])

    async def test_owner_card_rich_path(self) -> None:
        stats = dataclasses.replace(
            db_stats.DatabaseStats(path="/tmp/x.db"), tables=[]
        )
        rich = AsyncMock(return_value=True)
        deleted = AsyncMock()
        with (
            patch.object(db_stats, "collect", AsyncMock(return_value=stats)),
            patch.object(panel, "send_rich", rich),
            patch.object(panel, "safe_delete", deleted),
            patch.object(config, "rich_messages", True),
        ):
            await panel._show_database(_callback("adm:db"))
        rich.assert_awaited_once()
        deleted.assert_awaited_once()
        text = rich.await_args.args[1]
        self.assertTrue(text.startswith("### 🗄 Database"))


if __name__ == "__main__":
    unittest.main()
