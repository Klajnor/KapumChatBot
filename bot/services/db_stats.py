"""Статистика SQLite-базы для owner-карточки «🗄 Database» в /admin.

Только чтение: открывается отдельное соединение sqlite3 к файлу из
``DATABASE_URL`` (бот продолжает работать через aiosqlite), выполняются
``PRAGMA page_size/page_count/freelist_count/journal_mode``, список таблиц из
``sqlite_master`` и ``SELECT COUNT(*)`` по каждой из них в одной read-транзакции
(WAL даёт согласованный снапшот и не блокирует писателей).

Размер по таблицам считается модулем ``dbstat``, если он собран в libsqlite3
(доступность определяется в рантайме): страницы таблицы плюс её индексы
(соответствие через ``sqlite_master.tbl_name``). Без ``dbstat`` размер
показывается только у файла целиком — оценок «на глаз» модуль не придумывает.

Назначение таблиц — статический словарь ``_TABLE_NOTES``; неизвестная таблица
помечается явно, а не молча пропускается, чтобы словарь не разъехался со схемой
(тест ``test_db_stats`` дополнительно проверяет, что каждая таблица моделей
описана). Результат кэшируется на ``DB_STATS_CACHE_SECONDS``: кнопка «Refresh»
запрашивает свежий сбор принудительно.
"""

from __future__ import annotations

import asyncio
import dataclasses
import html
import logging
import os
import shutil
import sqlite3
import time
from typing import Any

logger = logging.getLogger(__name__)

DB_STATS_CACHE_SECONDS = 60.0

# Назначение таблиц БД (Kapum): расшифровка в карточке. Новую таблицу моделей
# надо добавлять сюда же — тест проверяет покрытие.
_TABLE_NOTES: dict[str, str] = {
    "users": "user profiles, roles, guest trials",
    "user_settings": "model, system prompt, personal name, timezone",
    "messages": "chat history (shrinks only via /clear)",
    "allowed_users": "access whitelist, limits, blocks",
    "daily_usage": "daily question counters",
    "photo_intent_usage": "free photo-intent classifier calls",
    "reminder_intent_usage": "free reminder-intent parser calls",
    "status_intent_hour_usage": "anti-spam counters for status (hourly)",
    "status_intent_day_usage": "anti-spam counters for status (daily)",
    "status_search_checks": "last live search check per user",
    "reminders": "reminders and scheduled AI requests",
    "app_settings": "runtime settings (thinking, digest model/prompt)",
    "digest_channels": "channels/groups summarized daily",
    "channel_messages": "digest source posts (pruned: 7 days)",
    "sqlite_sequence": "internal AUTOINCREMENT counters",
}

_UNKNOWN_NOTE = "❓ undocumented table"


@dataclasses.dataclass(slots=True)
class TableStat:
    """Одна таблица: число строк, размер (если известен) и назначение."""

    name: str
    rows: int
    bytes: int | None
    note: str


@dataclasses.dataclass(slots=True)
class DatabaseStats:
    """Снимок состояния файла БД на момент сбора."""

    path: str | None
    file_bytes: int = 0
    wal_bytes: int = 0
    shm_bytes: int = 0
    page_size: int = 0
    page_count: int = 0
    freelist_count: int = 0
    journal_mode: str = ""
    sqlite_version: str = ""
    free_disk_bytes: int | None = None
    dbstat_available: bool = False
    tables: list[TableStat] = dataclasses.field(default_factory=list)
    duration_ms: float = 0.0
    collected_at: float = 0.0
    error: str | None = None

    @property
    def freelist_bytes(self) -> int:
        """Сколько места вернул бы VACUUM (freelist страницы)."""
        return self.freelist_count * self.page_size


# --- Форматирование -----------------------------------------------------------


def fmt_bytes(size: int | None) -> str:
    """Человекочитаемый размер: B/KB/MB/GB."""
    if size is None:
        return "—"
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            if unit == "B":
                return f"{int(value)} B"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def _md_cell(value: Any) -> str:
    """Экранировать ячейку GFM-таблицы (по образцу usage_report._cell)."""
    text = " ".join(str(value).split())
    for char in "\\`*_[]()<>~#|=$@:/":
        text = text.replace(char, f"\\{char}")
    return text


def markdown_report(stats: DatabaseStats) -> str:
    """GFM-тело карточки для rich-сообщения (без заголовка)."""
    if stats.error:
        return f"⚠️ {stats.error}"
    lines = [
        (
            f"**File:** `{stats.path}` — **{fmt_bytes(stats.file_bytes)}**"
            f" (+ WAL {fmt_bytes(stats.wal_bytes)}, SHM {fmt_bytes(stats.shm_bytes)})"
        ),
        (
            f"**Pages:** {stats.page_count:,} × {fmt_bytes(stats.page_size)};"
            f" freelist {stats.freelist_count:,} pages"
            f" (~{fmt_bytes(stats.freelist_bytes)} reclaimable by VACUUM)"
        ),
        f"**Engine:** {stats.journal_mode or '?'} journal · SQLite "
        f"{stats.sqlite_version or '?'}"
        + (f" · disk free {fmt_bytes(stats.free_disk_bytes)}" if stats.free_disk_bytes is not None else "")
        + f" · collected in {stats.duration_ms:.2f} s",
        "",
        "| # | Table | Rows | Bytes | Purpose |",
        "|---:|:---|---:|---:|:---|",
    ]
    for index, table in enumerate(stats.tables, start=1):
        lines.append(
            f"| {index} | {_md_cell(table.name)} | {table.rows:,} | "
            f"{fmt_bytes(table.bytes) if stats.dbstat_available else '—'} | "
            f"{_md_cell(table.note)} |"
        )
    return "\n".join(lines)


def html_report(stats: DatabaseStats) -> str:
    """Запасной HTML-вариант для чата без rich-сообщений."""
    if stats.error:
        return f"⚠️ {html.escape(stats.error)}"
    lines = [
        (
            f"📄 <code>{html.escape(os.path.basename(stats.path or 'db'))}</code> — "
            f"<b>{fmt_bytes(stats.file_bytes)}</b> "
            f"(+ WAL {fmt_bytes(stats.wal_bytes)}, SHM {fmt_bytes(stats.shm_bytes)})"
        ),
        (
            f"{stats.page_count:,} pages × {fmt_bytes(stats.page_size)};"
            f" freelist {stats.freelist_count:,} pages "
            f"(~{fmt_bytes(stats.freelist_bytes)} reclaimable)"
        ),
        f"{html.escape(stats.journal_mode or '?')} journal · SQLite "
        f"{html.escape(stats.sqlite_version or '?')}"
        + (f" · disk free {fmt_bytes(stats.free_disk_bytes)}" if stats.free_disk_bytes is not None else ""),
        f"<i>Collected in {stats.duration_ms:.2f} s</i>",
        "",
    ]
    for index, table in enumerate(stats.tables, start=1):
        size = f" · {fmt_bytes(table.bytes)}" if stats.dbstat_available else ""
        lines.append(
            f"{index}. <b><code>{html.escape(table.name)}</code></b> — "
            f"<b>{table.rows:,}</b> rows{size} — {html.escape(table.note)}"
        )
    return "\n".join(lines)


# --- Сбор ---------------------------------------------------------------------


def database_path() -> str | None:
    """Файл SQLite из ``DATABASE_URL``; None для in-memory и не-SQLite URL."""
    try:
        from sqlalchemy.engine import make_url

        url = make_url(config_url())
    except Exception as exc:  # noqa: BLE001 - некорректный URL не должен ронять карточку
        logger.warning("Cannot parse DATABASE_URL: %s", exc)
        return None
    if not url.drivername.startswith("sqlite"):
        return None
    database = url.database or ""
    if database.startswith("file:"):
        from urllib.parse import unquote, urlparse

        parsed = urlparse(database)
        database = unquote(parsed.path or "")
        # mode/immutable из query — только чтение: игнорируем их.
    if not database or database == ":memory:":
        return None
    return os.path.abspath(database)


def config_url() -> str:
    """DATABASE_URL как строка (вынесено для удобных патчей в тестах)."""
    from bot.config import config

    return config.database_url


def _stat_size(path: str) -> int:
    """Размер файла или 0, если его ещё нет (WAL/SHM создаются лениво)."""
    try:
        return os.stat(path).st_size
    except OSError:
        return 0


def _quoted(name: str) -> str:
    """Идентификатор SQLite в двойных кавычках."""
    return '"' + name.replace('"', '""') + '"'


def _collect_sync(path: str) -> DatabaseStats:
    """Блокирующий сбор; вызывается из потока через ``asyncio.to_thread``."""
    stats = DatabaseStats(
        path=path, collected_at=time.time(), dbstat_available=True
    )
    started = time.perf_counter()
    conn: sqlite3.Connection | None = None
    try:
        stats.file_bytes = _stat_size(path)
        stats.wal_bytes = _stat_size(path + "-wal")
        stats.shm_bytes = _stat_size(path + "-shm")
        try:
            stats.free_disk_bytes = shutil.disk_usage(
                os.path.dirname(path) or "."
            ).free
        except OSError:  # отсутствие statvfs не должно ломать карточку
            stats.free_disk_bytes = None

        conn = sqlite3.connect(path, timeout=15)
        conn.isolation_level = None  # транзакции управляются вручную
        conn.execute("PRAGMA busy_timeout=15000")
        stats.journal_mode = str(
            conn.execute("PRAGMA journal_mode").fetchone()[0]
        )
        stats.page_size = int(conn.execute("PRAGMA page_size").fetchone()[0])
        stats.page_count = int(conn.execute("PRAGMA page_count").fetchone()[0])
        stats.freelist_count = int(
            conn.execute("PRAGMA freelist_count").fetchone()[0]
        )
        stats.sqlite_version = str(
            conn.execute("SELECT sqlite_version()").fetchone()[0]
        )

        rows = conn.execute(
            "SELECT type, name, tbl_name FROM sqlite_master "
            "WHERE type IN ('table','index')"
        ).fetchall()
        objects: dict[str, dict[str, Any]] = {}
        index_pages: dict[str, list[str]] = {}
        for kind, name, tbl_name in rows:
            if kind == "table":
                if not name.startswith("sqlite_"):
                    objects[name] = {"indexes": []}
            elif kind == "index":
                index_pages.setdefault(str(tbl_name), []).append(str(name))

        # sqlite_sequence — служебная таблица AUTOINCREMENT.
        if conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name = 'sqlite_sequence'"
        ).fetchone():
            objects["sqlite_sequence"] = {"indexes": []}

        # Согласованный снапшот на все подсчёты: WAL-читатели не блокируют
        # писателей, поэтому длинная read-транзакция безопасна для бота.
        conn.execute("BEGIN DEFERRED")
        try:
            # dbstat есть не в каждой сборке libsqlite3: определяем в рантайме.
            try:
                probe = _stat_pages(conn, sorted(objects))
            except sqlite3.OperationalError:
                stats.dbstat_available = False
            else:
                table_pages: dict[str, int] = dict(probe)

            for name in objects:
                count = conn.execute(
                    f"SELECT COUNT(*) FROM {_quoted(name)}"
                ).fetchone()[0]
                size: int | None = None
                if stats.dbstat_available:
                    size = table_pages.get(name, 0)
                    size += sum(
                        table_pages.get(index, 0)
                        for index in index_pages.get(name, [])
                    )
                stats.tables.append(
                    TableStat(
                        name=name,
                        rows=int(count),
                        bytes=size,
                        note=_TABLE_NOTES.get(name, _UNKNOWN_NOTE),
                    )
                )
        finally:
            conn.execute("COMMIT")

        stats.tables.sort(key=lambda table: (-table.rows, table.name))
    except Exception as exc:  # noqa: BLE001 - карточка не должна ломать /admin
        logger.warning("Database stats collection failed: %s", exc)
        stats.error = f"Could not collect database statistics: {exc}"
    finally:
        if conn is not None:
            try:
                conn.close()
            except sqlite3.Error:  # закрытие best-effort
                pass
        stats.duration_ms = (time.perf_counter() - started) * 1000
    return stats


def _stat_pages(
    conn: sqlite3.Connection, names: list[str]
) -> list[tuple[str, int]]:
    """Размер страниц по объектам БД через виртуальную таблицу ``dbstat``."""
    if not names:
        return []
    placeholders = ", ".join("?" for _ in names)
    rows = conn.execute(
        "SELECT name, SUM(pgsize) FROM dbstat WHERE name IN "
        f"({placeholders}) GROUP BY name",
        names,
    ).fetchall()
    return [(str(name), int(size or 0)) for name, size in rows]


# Кэш последнего сбора: кнопка Refresh запрашивает принудительно.
_cache: DatabaseStats | None = None


def collect_file(path: str) -> DatabaseStats:
    """Синхронный сбор по пути к файлу (публичная обёртка для тестов)."""
    return _collect_sync(path)


def _cache_fresh(now: float) -> bool:
    return (
        _cache is not None
        and _cache.error is None
        and now - _cache.collected_at < DB_STATS_CACHE_SECONDS
    )


async def collect(*, force: bool = False) -> DatabaseStats:
    """Собрать статистику (или вернуть кэш не старше 60 секунд).

    ``force=True`` — сбор заново (кнопка Refresh); ошибочный сбор не кэшируется,
    чтобы следующее открытие карточки повторило попытку.
    """
    global _cache
    now = time.time()
    if not force and _cache_fresh(now):
        return _cache
    path = database_path()
    if path is None:
        stats = DatabaseStats(
            path=None,
            collected_at=now,
            error=(
                "DATABASE_URL is not a SQLite file path "
                "(in-memory or a non-SQLite database)."
            ),
        )
    else:
        stats = await asyncio.to_thread(_collect_sync, path)
    if stats.error is None:
        _cache = stats
    return stats


def reset_cache() -> None:
    """Сбросить кэш (тесты и «Refresh» после внешних изменений)."""
    global _cache
    _cache = None
