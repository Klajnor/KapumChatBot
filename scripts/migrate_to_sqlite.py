"""One-time migration of the bot database into SQLite.

Usage (on the server):

    .venv/bin/python scripts/migrate_to_sqlite.py \
        --from postgresql+asyncpg://user:pass@localhost:5432/chat_bot \
        --to   sqlite+aiosqlite:///chat_bot.db [--force]

Rows are copied preserving their primary keys (history order and unique
constraints survive the move), five hundred rows per batch. Without --force
the script refuses to touch a non-empty target table. After a successful run
put the target URL into DATABASE_URL in .env and restart the bot service.
"""

from __future__ import annotations

import argparse
import asyncio
from typing import Any

from sqlalchemy import delete, func, insert, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from bot.database.models import (
    AllowedUser,
    Base,
    DailyUsage,
    Message,
    User,
    UserSettings,
)

# Copy order: users first keeps the file consistent even though SQLite does
# not enforce foreign keys during bulk inserts.
TABLES: list[Any] = [User, UserSettings, AllowedUser, DailyUsage, Message]

PAGE = 500  # rows per insert batch


def _engine_kwargs(url: str) -> dict[str, Any]:
    if url.startswith("sqlite"):
        return {"connect_args": {"check_same_thread": False}}
    return {"pool_pre_ping": True}


async def migrate(source_url: str, target_url: str, *, force: bool = False) -> dict[str, int]:
    """Copy all bot tables from ``source_url`` into ``target_url``.

    Returns a ``{table: rows copied}`` mapping. Raises RuntimeError when the
    target already holds rows and ``force`` was not requested.
    """
    src = create_async_engine(source_url, **_engine_kwargs(source_url))
    dst = create_async_engine(target_url, **_engine_kwargs(target_url))
    counts: dict[str, int] = {}
    try:
        async with dst.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        src_factory = async_sessionmaker(src, expire_on_commit=False)
        dst_factory = async_sessionmaker(dst, expire_on_commit=False)
        async with src_factory() as s_src, dst_factory() as s_dst:
            for model in TABLES:
                table = model.__table__
                if force:
                    await s_dst.execute(delete(table))
                else:
                    have = await s_dst.execute(select(func.count()).select_from(table))
                    existing = int(have.scalar() or 0)
                    if existing:
                        raise RuntimeError(
                            f"target table '{table.name}' already holds "
                            f"{existing} rows; re-run with --force to replace them"
                        )
                copied = 0
                # A streaming result: partitions() yields async pages of PAGE
                # rows, so a large history never lands in memory whole.
                result = await s_src.stream(select(table))
                async for partition in result.partitions(PAGE):
                    rows = [dict(record._mapping) for record in partition]
                    if rows:
                        await s_dst.execute(insert(table), rows)
                        copied += len(rows)
                counts[table.name] = copied
            await s_dst.commit()
    finally:
        await src.dispose()
        await dst.dispose()
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Copy the bot database (any async SQLAlchemy URL) into SQLite."
    )
    parser.add_argument(
        "--from",
        dest="source",
        required=True,
        help="Source URL, e.g. postgresql+asyncpg://user:pass@localhost:5432/chat_bot",
    )
    parser.add_argument(
        "--to",
        dest="target",
        default=None,
        help="Target URL (default: the DATABASE_URL currently in .env)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete existing rows in the target tables before copying",
    )
    args = parser.parse_args()

    target = args.target
    if not target:
        from bot.config import config

        target = config.database_url

    counts = asyncio.run(migrate(args.source, target, force=args.force))
    for name, n in counts.items():
        print(f"{name:15s} {n} rows")
    print(f"\nMigrated {sum(counts.values())} rows into: {target}")
    print("Set DATABASE_URL in .env to this URL and restart the bot service.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
