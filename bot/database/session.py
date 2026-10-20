from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from bot.config import config
from bot.database.models import Base

engine = create_async_engine(
    config.database_url,
    echo=False,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
)

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
    """Apply idempotent migrations for databases created by older versions."""

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
