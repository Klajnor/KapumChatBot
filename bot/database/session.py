from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from bot.config import config
from bot.database.models import Base

_IS_SQLITE = config.database_url.startswith("sqlite")


def _build_engine() -> Any:
    """Create the async engine for SQLite (aiosqlite) or PostgreSQL (asyncpg)."""
    url = config.database_url
    kwargs: dict[str, Any] = {"echo": False, "pool_pre_ping": True}

    if _IS_SQLITE:
        # The asyncio driver must not pin the connection to one thread, and an
        # in-memory database has to share a single connection (StaticPool),
        # otherwise every new connection starts with an empty schema.
        kwargs["connect_args"] = {"check_same_thread": False}
        if ":memory:" in url:
            kwargs["poolclass"] = StaticPool
    else:
        kwargs["pool_size"] = 5
        kwargs["max_overflow"] = 10

    engine = create_async_engine(url, **kwargs)

    if _IS_SQLITE:
        @event.listens_for(engine.sync_engine, "connect")
        def _sqlite_pragmas(dbapi_connection: Any, _record: Any) -> None:
            # WAL lets readers and one writer coexist; busy_timeout makes the
            # second concurrent write wait instead of erroring with "database
            # is locked".
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA busy_timeout=5000")
            # Enforce ON DELETE CASCADE (off by default in SQLite).
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return engine


engine = _build_engine()

async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def init_db() -> None:
    """Create all tables if they don't exist and apply lightweight migrations."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await _run_migrations(conn)


async def _run_migrations(conn: Any) -> None:
    """Apply idempotent migrations for databases created by older versions.

    The statements below are PostgreSQL syntax (SERIAL, TIMESTAMPTZ, ALTER …
    ADD COLUMN IF NOT EXISTS) repairing legacy Postgres databases. A SQLite
    file is always created by ``create_all`` with the current schema, so there
    is nothing to migrate there.
    """
    if conn.dialect.name == "sqlite":
        return

    statements = [
        # Guest free-trial counter on users.
        "ALTER TABLE users "
        "ADD COLUMN IF NOT EXISTS guest_messages_used INTEGER NOT NULL DEFAULT 0",
        # allowed_users table.
        "CREATE TABLE IF NOT EXISTS allowed_users ("
        "telegram_id BIGINT NOT NULL PRIMARY KEY, "
        "added_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), "
        "added_by BIGINT)",
        # daily_usage table + index + unique constraint.
        "CREATE TABLE IF NOT EXISTS daily_usage ("
        "id SERIAL PRIMARY KEY, "
        "telegram_id BIGINT NOT NULL REFERENCES users(telegram_id) ON DELETE CASCADE, "
        "date DATE NOT NULL, "
        "query_count INTEGER NOT NULL DEFAULT 0, "
        "UNIQUE (telegram_id, date))",
        "CREATE INDEX IF NOT EXISTS ix_daily_usage_telegram_id "
        "ON daily_usage (telegram_id)",
        # Conversation key on messages (needed for group/guest chats).
        "ALTER TABLE messages ADD COLUMN IF NOT EXISTS chat_id BIGINT",
        "CREATE INDEX IF NOT EXISTS ix_messages_chat_id ON messages (chat_id)",
        # Backfill existing rows: private-chat history was keyed by the user id.
        "UPDATE messages SET chat_id = user_id WHERE chat_id IS NULL",
    ]

    for statement in statements:
        try:
            async with conn.begin_nested():
                await conn.execute(text(statement))
        except Exception as exc:  # pragma: no cover - defensive, DB dependent
            # Never abort startup because of an idempotent migration.
            import logging

            logging.getLogger(__name__).warning(
                "Migration skipped (%s): %s", statement.split()[0:3], exc
            )


async def get_session() -> AsyncIterator[AsyncSession]:
    """Yield an async database session (dependency-injection helper)."""
    async with async_session_factory() as session:
        yield session
