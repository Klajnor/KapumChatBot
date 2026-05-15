from __future__ import annotations

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from bot.config import config
from bot.database.models import Base

engine = create_async_engine(
    config.database_url,
    echo=False,
    pool_size=5,
    max_overflow=10,
)

async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def init_db() -> None:
    """Create all tables if they don't exist."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # Run migrations for existing tables
        await _run_migrations(conn)


async def _run_migrations(conn: Any) -> None:
    """Run database migrations to add new columns to existing tables."""
    from sqlalchemy import text

    # Add guest_messages_used column to users table if it doesn't exist
    try:
        await conn.execute(
            text(
                "ALTER TABLE users "
                "ADD COLUMN IF NOT EXISTS guest_messages_used "
                "INTEGER NOT NULL DEFAULT 0"
            )
        )
    except Exception:
        pass

    # Create allowed_users table if it doesn't exist
    try:
        await conn.execute(
            text(
                "CREATE TABLE IF NOT EXISTS allowed_users ("
                "telegram_id BIGINT NOT NULL PRIMARY KEY, "
                "added_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), "
                "added_by BIGINT"
                ")"
            )
        )
    except Exception:
        pass

    # Create daily_usage table if it doesn't exist
    try:
        await conn.execute(
            text(
                "CREATE TABLE IF NOT EXISTS daily_usage ("
                "id SERIAL PRIMARY KEY, "
                "telegram_id BIGINT NOT NULL REFERENCES users(telegram_id) ON DELETE CASCADE, "
                "date DATE NOT NULL, "
                "query_count INTEGER NOT NULL DEFAULT 0, "
                "UNIQUE (telegram_id, date)"
                ")"
            )
        )
        # Create index on telegram_id for daily_usage
        await conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_daily_usage_telegram_id "
                "ON daily_usage (telegram_id)"
            )
        )
    except Exception:
        pass


async def get_session() -> AsyncSession:  # type: ignore[misc]
    """Yield an async database session."""
    async with async_session_factory() as session:
        yield session