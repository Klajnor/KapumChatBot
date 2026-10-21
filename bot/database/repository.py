from __future__ import annotations

import datetime

from sqlalchemy import delete, func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from bot.config import config
from bot.database.models import AllowedUser, DailyUsage, Message, User, UserSettings


class UserRepository:
    """Repository for user-related database operations."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_or_create_user(
        self,
        telegram_id: int,
        username: str | None = None,
        first_name: str | None = None,
        last_name: str | None = None,
        is_admin: bool = False,
    ) -> User:
        """Get an existing user or create one with default settings."""
        user = await self.session.get(User, telegram_id)
        if user is None:
            user = User(
                telegram_id=telegram_id,
                username=username,
                first_name=first_name,
                last_name=last_name,
                is_admin=is_admin,
            )
            self.session.add(user)
            try:
                await self.session.flush()
            except IntegrityError:
                # A concurrent request registered the same user first.
                await self.session.rollback()
                user = await self.session.get(User, telegram_id)
                if user is None:
                    raise
        else:
            # Refresh profile fields that may have changed.
            if username is not None:
                user.username = username
            if first_name is not None:
                user.first_name = first_name
            if last_name is not None:
                user.last_name = last_name
            if is_admin:
                user.is_admin = True

        # Guarantee that settings exist (also for legacy rows).
        settings = await self.get_user_settings(telegram_id)
        if settings is None:
            self.session.add(
                UserSettings(
                    user_id=telegram_id,
                    selected_model=config.default_model,
                    system_prompt=config.default_system_prompt,
                    temperature=config.default_temperature,
                    max_tokens=config.default_max_tokens,
                )
            )
            try:
                await self.session.flush()
            except IntegrityError:
                await self.session.rollback()

        return user

    async def get_user(self, telegram_id: int) -> User | None:
        """Get user by Telegram ID."""
        return await self.session.get(User, telegram_id)

    async def get_guest_messages_used(self, telegram_id: int) -> int:
        """Get the number of free-trial messages used by a guest."""
        user = await self.session.get(User, telegram_id)
        return user.guest_messages_used if user is not None else 0

    async def increment_guest_messages(self, telegram_id: int) -> int:
        """Atomically increment and return the guest message counter.

        UPDATE + read-back (no RETURNING): SQLite only supports RETURNING on
        3.35+, which is not guaranteed on server distros; inside one
        transaction the read reflects our own update.
        """
        await self.session.execute(
            update(User)
            .where(User.telegram_id == telegram_id)
            .values(guest_messages_used=User.guest_messages_used + 1)
        )
        result = await self.session.execute(
            select(User.guest_messages_used).where(
                User.telegram_id == telegram_id
            )
        )
        value = result.scalar_one_or_none()
        return int(value) if value is not None else 0

    async def try_consume_guest_message(self, telegram_id: int, limit: int) -> int | None:
        """Atomically take one free-trial message; ``None`` when the limit is reached.

        The limit check lives in the UPDATE itself, so parallel requests cannot
        slip past it between a separate check and increment.
        """
        result = await self.session.execute(
            update(User)
            .where(User.telegram_id == telegram_id, User.guest_messages_used < limit)
            .values(guest_messages_used=User.guest_messages_used + 1)
        )
        if not result.rowcount:
            return None
        return await self.get_guest_messages_used(telegram_id)

    async def release_guest_message(self, telegram_id: int) -> None:
        """Give back a free-trial message taken for a request that failed."""
        await self.session.execute(
            update(User)
            .where(User.telegram_id == telegram_id, User.guest_messages_used > 0)
            .values(guest_messages_used=User.guest_messages_used - 1)
        )

    async def get_user_settings(self, telegram_id: int) -> UserSettings | None:
        """Get user settings."""
        result = await self.session.execute(
            select(UserSettings).where(UserSettings.user_id == telegram_id)
        )
        return result.scalar_one_or_none()

    async def _ensure_settings(self, telegram_id: int) -> UserSettings:
        """Return the user's settings row, creating it if necessary."""
        settings = await self.get_user_settings(telegram_id)
        if settings is None:
            await self.get_or_create_user(telegram_id)
            settings = await self.get_user_settings(telegram_id)
            assert settings is not None
        return settings

    async def update_model(self, telegram_id: int, model: str) -> None:
        """Update user's selected AI model."""
        settings = await self._ensure_settings(telegram_id)
        settings.selected_model = model
        await self.session.flush()

    async def update_system_prompt(self, telegram_id: int, prompt: str) -> None:
        """Update user's system prompt."""
        settings = await self._ensure_settings(telegram_id)
        settings.system_prompt = prompt
        await self.session.flush()

    async def update_settings(
        self,
        telegram_id: int,
        model: str | None = None,
        prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        """Update user settings."""
        settings = await self._ensure_settings(telegram_id)
        if model is not None:
            settings.selected_model = model
        if prompt is not None:
            settings.system_prompt = prompt
        if temperature is not None:
            settings.temperature = temperature
        if max_tokens is not None:
            settings.max_tokens = max_tokens
        await self.session.flush()

    # --- Allowed users management -----------------------------------------

    async def add_allowed_user(
        self, telegram_id: int, added_by: int | None = None
    ) -> AllowedUser:
        """Add a user to the allowed list."""
        existing = await self.session.get(AllowedUser, telegram_id)
        if existing:
            return existing
        allowed = AllowedUser(telegram_id=telegram_id, added_by=added_by)
        self.session.add(allowed)
        await self.session.flush()
        return allowed

    async def remove_allowed_user(self, telegram_id: int) -> bool:
        """Remove a user from the allowed list. Returns True if removed."""
        existing = await self.session.get(AllowedUser, telegram_id)
        if existing is None:
            return False
        await self.session.delete(existing)
        await self.session.flush()
        return True

    async def get_allowed_users(self) -> list[AllowedUser]:
        """Get all allowed users."""
        result = await self.session.execute(
            select(AllowedUser).order_by(AllowedUser.added_at.desc())
        )
        return list(result.scalars().all())

    async def is_allowed_user(self, telegram_id: int) -> bool:
        """Check if a user is in the allowed list."""
        existing = await self.session.get(AllowedUser, telegram_id)
        return existing is not None

    # --- Daily usage tracking ---------------------------------------------

    async def get_daily_usage(
        self, telegram_id: int, date: datetime.date
    ) -> DailyUsage | None:
        """Get daily usage record for a user on a specific date."""
        result = await self.session.execute(
            select(DailyUsage).where(
                DailyUsage.telegram_id == telegram_id,
                DailyUsage.date == date,
            )
        )
        return result.scalar_one_or_none()

    async def increment_daily_usage(self, telegram_id: int, date: datetime.date) -> int:
        """Atomically increment daily usage and return the new count.

        Uses an UPSERT so concurrent requests cannot lose increments; the
        dialect-correct statement is chosen at runtime (PostgreSQL keeps
        RETURNING, SQLite upserts and re-read inside the transaction).
        """
        if self.session.get_bind().dialect.name == "sqlite":
            stmt = (
                sqlite_insert(DailyUsage)
                .values(telegram_id=telegram_id, date=date, query_count=1)
                .on_conflict_do_update(
                    index_elements=["telegram_id", "date"],
                    set_={"query_count": DailyUsage.query_count + 1},
                )
            )
            await self.session.execute(stmt)
            result = await self.session.execute(
                select(DailyUsage.query_count).where(
                    DailyUsage.telegram_id == telegram_id,
                    DailyUsage.date == date,
                )
            )
            return int(result.scalar_one())

        stmt = (
            pg_insert(DailyUsage)
            .values(telegram_id=telegram_id, date=date, query_count=1)
            .on_conflict_do_update(
                index_elements=[DailyUsage.telegram_id, DailyUsage.date],
                set_={"query_count": DailyUsage.query_count + 1},
            )
            .returning(DailyUsage.query_count)
        )
        result = await self.session.execute(stmt)
        return result.scalar_one()

    async def try_consume_daily_usage(
        self, telegram_id: int, date: datetime.date, limit: int
    ) -> int | None:
        """Atomically count one query; ``None`` when today's limit is reached.

        Both dialects share the ``INSERT … ON CONFLICT DO UPDATE … WHERE`` form;
        the WHERE clause stops the increment at ``limit``.
        """
        insert_fn = (
            sqlite_insert
            if self.session.get_bind().dialect.name == "sqlite"
            else pg_insert
        )
        stmt = (
            insert_fn(DailyUsage)
            .values(telegram_id=telegram_id, date=date, query_count=1)
            .on_conflict_do_update(
                index_elements=["telegram_id", "date"],
                set_={"query_count": DailyUsage.query_count + 1},
                where=DailyUsage.query_count < limit,
            )
        )
        if limit <= 0:
            return None
        result = await self.session.execute(stmt)
        if not result.rowcount:
            return None
        usage = await self.get_daily_usage(telegram_id, date)
        return int(usage.query_count) if usage else 1

    async def release_daily_usage(self, telegram_id: int, date: datetime.date) -> None:
        """Give back a query taken for a request that failed."""
        await self.session.execute(
            update(DailyUsage)
            .where(
                DailyUsage.telegram_id == telegram_id,
                DailyUsage.date == date,
                DailyUsage.query_count > 0,
            )
            .values(query_count=DailyUsage.query_count - 1)
        )

    async def reset_daily_usage(self, telegram_id: int, date: datetime.date) -> bool:
        """Reset the daily usage counter. Returns True if a record existed."""
        usage = await self.get_daily_usage(telegram_id, date)
        if usage is None:
            return False
        usage.query_count = 0
        await self.session.flush()
        return True

    async def get_usage_stats_last_14_days(self) -> list[dict]:
        """Get usage statistics for all users over the last 14 days."""
        fourteen_days_ago = datetime.date.today() - datetime.timedelta(days=13)
        result = await self.session.execute(
            select(
                DailyUsage.telegram_id,
                DailyUsage.date,
                DailyUsage.query_count,
            )
            .where(DailyUsage.date >= fourteen_days_ago)
            .order_by(DailyUsage.date.desc(), DailyUsage.telegram_id)
        )
        return [
            {
                "telegram_id": row.telegram_id,
                "date": row.date,
                "query_count": row.query_count,
            }
            for row in result.all()
        ]

    async def get_user_total_usage_last_14_days(self, telegram_id: int) -> int:
        """Get total usage for a specific user over the last 14 days."""
        fourteen_days_ago = datetime.date.today() - datetime.timedelta(days=13)
        result = await self.session.execute(
            select(func.coalesce(func.sum(DailyUsage.query_count), 0)).where(
                DailyUsage.telegram_id == telegram_id,
                DailyUsage.date >= fourteen_days_ago,
            )
        )
        return int(result.scalar() or 0)


class MessageRepository:
    """Repository for message-related database operations."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add_message(
        self,
        chat_id: int,
        user_id: int,
        role: str,
        content: str,
        image_url: str | None = None,
        model: str | None = None,
    ) -> Message:
        """Save a message to the database."""
        message = Message(
            chat_id=chat_id,
            user_id=user_id,
            role=role,
            content=content,
            image_url=image_url,
            model=model,
        )
        self.session.add(message)
        await self.session.flush()
        return message

    async def get_history(
        self, chat_id: int, limit: int = 30, user_id: int | None = None
    ) -> list[dict[str, str]]:
        """Return the last ``limit`` messages of a conversation in OpenAI format.

        ``user_id`` restricts the history to one participant (assistant replies
        are stored under the asker's id), so users of a shared group chat do
        not see each other's questions in their context.
        """
        query = select(Message).where(Message.chat_id == chat_id)
        if user_id is not None:
            query = query.where(Message.user_id == user_id)
        result = await self.session.execute(
            query.order_by(Message.id.desc()).limit(limit)
        )
        messages = list(reversed(result.scalars().all()))
        return [{"role": m.role, "content": m.content} for m in messages]

    async def clear_history(self, chat_id: int, user_id: int | None = None) -> None:
        """Delete the messages of a conversation (only ``user_id``'s when given)."""
        stmt = delete(Message).where(Message.chat_id == chat_id)
        if user_id is not None:
            stmt = stmt.where(Message.user_id == user_id)
        await self.session.execute(stmt)
        await self.session.flush()

    async def get_message_count(self, user_id: int) -> int:
        """Get the total number of messages authored by a user."""
        result = await self.session.execute(
            select(func.count()).select_from(Message).where(Message.user_id == user_id)
        )
        return int(result.scalar() or 0)
