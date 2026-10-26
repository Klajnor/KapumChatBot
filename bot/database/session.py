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
        await _run_data_migrations(conn)


async def _run_data_migrations(conn: Any) -> None:
    """Dialect-agnostic data fixes (they run on SQLite too).

    Currently one fix: rows whose ``system_prompt`` still holds the *exact*
    legacy English default are switched to the current Russian default, so
    existing installations start answering in Russian after the update.
    Custom prompts and chosen personas differ from the legacy string and are
    therefore never touched.
    """
    from bot.config import LEGACY_DEFAULT_SYSTEM_PROMPT

    if config.default_system_prompt == LEGACY_DEFAULT_SYSTEM_PROMPT:
        return  # the deployment intentionally kept the English default
    try:
        await conn.execute(
            text(
                "UPDATE user_settings SET system_prompt = :new "
                "WHERE system_prompt = :legacy"
            ),
            {
                "new": config.default_system_prompt,
                "legacy": LEGACY_DEFAULT_SYSTEM_PROMPT,
            },
        )
    except Exception as exc:  # noqa: BLE001 - a data fix must never abort startup
        import logging

        logging.getLogger(__name__).warning(
            "Default-prompt data migration skipped: %s", exc
        )


async def _add_missing_allowed_user_columns_sqlite(conn: Any) -> None:
    """Add the profile-cache columns to a SQLite database created earlier.

    ``create_all`` never alters existing tables, and SQLite has no
    ``ADD COLUMN IF NOT EXISTS``, so the current columns are inspected first.
    """
    result = await conn.execute(text("PRAGMA table_info(allowed_users)"))
    existing = {str(row[1]) for row in result.fetchall()}
    for column in ("username", "first_name", "last_name"):
        if column in existing:
            continue
        try:
            async with conn.begin_nested():
                await conn.execute(
                    text(f"ALTER TABLE allowed_users ADD COLUMN {column} VARCHAR(128)")
                )
        except Exception as exc:  # pragma: no cover - defensive, DB dependent  # noqa: BLE001
            import logging

            logging.getLogger(__name__).warning(
                "Migration skipped (allowed_users.%s): %s", column, exc
            )


async def _add_missing_access_columns_sqlite(conn: Any) -> None:
    """Add the access-level columns (limits, blocking, admin grants) to SQLite."""
    wanted = {
        "allowed_users": [("daily_limit", "INTEGER")],
        "users": [
            ("granted_admin", "BOOLEAN NOT NULL DEFAULT 0"),
            ("blocked", "BOOLEAN NOT NULL DEFAULT 0"),
            ("reminder_timezone", "VARCHAR(64)"),
        ],
    }
    for table, columns in wanted.items():
        result = await conn.execute(text(f"PRAGMA table_info({table})"))
        existing = {str(row[1]) for row in result.fetchall()}
        for column, ddl in columns:
            if column in existing:
                continue
            try:
                async with conn.begin_nested():
                    await conn.execute(
                        text(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")
                    )
            except Exception as exc:  # pragma: no cover - defensive, DB dependent  # noqa: BLE001
                import logging

                logging.getLogger(__name__).warning(
                    "Migration skipped (%s.%s): %s", table, column, exc
                )


async def _run_migrations(conn: Any) -> None:
    """Apply idempotent migrations for databases created by older versions.

    The statements below are PostgreSQL syntax (SERIAL, TIMESTAMPTZ, ALTER …
    ADD COLUMN IF NOT EXISTS) repairing legacy Postgres databases. A SQLite
    file is always created by ``create_all`` with the current schema, so there
    is nothing to migrate there.
    """
    if conn.dialect.name == "sqlite":
        await _add_missing_allowed_user_columns_sqlite(conn)
        await _add_missing_access_columns_sqlite(conn)
        return

    statements = [
        # Guest free-trial counter on users.
        ("ALTER TABLE users "
        "ADD COLUMN IF NOT EXISTS guest_messages_used INTEGER NOT NULL DEFAULT 0"),
        # Access levels: runtime admins, blocking, per-user daily limit.
        ("ALTER TABLE users "
        "ADD COLUMN IF NOT EXISTS granted_admin BOOLEAN NOT NULL DEFAULT FALSE"),
        ("ALTER TABLE users "
        "ADD COLUMN IF NOT EXISTS blocked BOOLEAN NOT NULL DEFAULT FALSE"),
        # Personal reminders timezone (IANA name; NULL = REMINDER_TIMEZONE).
        ("ALTER TABLE users "
        "ADD COLUMN IF NOT EXISTS reminder_timezone VARCHAR(64)"),
        # allowed_users table.
        ("CREATE TABLE IF NOT EXISTS allowed_users ("
        "telegram_id BIGINT NOT NULL PRIMARY KEY, "
        "added_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), "
        "added_by BIGINT, "
        "username VARCHAR(128), "
        "first_name VARCHAR(128), "
        "last_name VARCHAR(128))"),
        # Cached Telegram profile on allowed_users (older installations).
        "ALTER TABLE allowed_users ADD COLUMN IF NOT EXISTS username VARCHAR(128)",
        "ALTER TABLE allowed_users ADD COLUMN IF NOT EXISTS first_name VARCHAR(128)",
        "ALTER TABLE allowed_users ADD COLUMN IF NOT EXISTS last_name VARCHAR(128)",
        "ALTER TABLE allowed_users ADD COLUMN IF NOT EXISTS daily_limit INTEGER",
        # daily_usage table + index + unique constraint.
        ("CREATE TABLE IF NOT EXISTS daily_usage ("
        "id SERIAL PRIMARY KEY, "
        "telegram_id BIGINT NOT NULL REFERENCES users(telegram_id) ON DELETE CASCADE, "
        "date DATE NOT NULL, "
        "query_count INTEGER NOT NULL DEFAULT 0, "
        "UNIQUE (telegram_id, date))"),
        ("CREATE INDEX IF NOT EXISTS ix_daily_usage_telegram_id "
        "ON daily_usage (telegram_id)"),
        # Conversation key on messages (needed for group/guest chats).
        "ALTER TABLE messages ADD COLUMN IF NOT EXISTS chat_id BIGINT",
        "CREATE INDEX IF NOT EXISTS ix_messages_chat_id ON messages (chat_id)",
        # Backfill existing rows: private-chat history was keyed by the user id.
        "UPDATE messages SET chat_id = user_id WHERE chat_id IS NULL",
        # reminders table (scheduled messages and AI requests).
        ("CREATE TABLE IF NOT EXISTS reminders ("
        "id SERIAL PRIMARY KEY, "
        "user_id BIGINT NOT NULL REFERENCES users(telegram_id) ON DELETE CASCADE, "
        "chat_id BIGINT NOT NULL, "
        "text TEXT NOT NULL, "
        "ai_mode BOOLEAN NOT NULL DEFAULT FALSE, "
        "fire_at TIMESTAMPTZ NOT NULL, "
        "schedule VARCHAR(64), "
        "enabled BOOLEAN NOT NULL DEFAULT TRUE, "
        "created_at TIMESTAMPTZ NOT NULL DEFAULT NOW())"),
        "CREATE INDEX IF NOT EXISTS ix_reminders_user_id ON reminders (user_id)",
        "CREATE INDEX IF NOT EXISTS ix_reminders_fire_at ON reminders (fire_at)",
    ]

    for statement in statements:
        try:
            async with conn.begin_nested():
                await conn.execute(text(statement))
        except Exception as exc:  # pragma: no cover - defensive, DB dependent  # noqa: BLE001
            # Never abort startup because of an idempotent migration.
            import logging

            logging.getLogger(__name__).warning(
                "Migration skipped (%s): %s", statement.split()[0:3], exc
            )


async def get_session() -> AsyncIterator[AsyncSession]:
    """Yield an async database session (dependency-injection helper)."""
    async with async_session_factory() as session:
        yield session
