from __future__ import annotations

import datetime
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession

from bot.config import config
from bot.database.models import AllowedUser, Message, Reminder
from bot.database.repository import (
    MessageRepository,
    ReminderRepository,
    UserRepository,
)


class UserService:
    """Business logic layer for user operations."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.user_repo = UserRepository(session)
        self.message_repo = MessageRepository(session)
        self.reminder_repo = ReminderRepository(session)

    # --- Users -------------------------------------------------------------

    async def register_user(
        self,
        telegram_id: int,
        username: str | None = None,
        first_name: str | None = None,
        last_name: str | None = None,
    ) -> None:
        """Register or update a user in the database."""
        await self.user_repo.get_or_create_user(
            telegram_id=telegram_id,
            username=username,
            first_name=first_name,
            last_name=last_name,
            is_admin=config.is_admin(telegram_id),
        )
        await self.session.commit()

    async def get_user_settings(self, telegram_id: int) -> dict:
        """Get user settings as a dictionary (falls back to defaults)."""
        settings = await self.user_repo.get_user_settings(telegram_id)
        if settings is None:
            return {
                "model": config.default_model,
                "system_prompt": config.default_system_prompt,
                "temperature": config.default_temperature,
                "max_tokens": config.default_max_tokens,
            }
        return {
            "model": settings.selected_model,
            "system_prompt": settings.system_prompt,
            "temperature": settings.temperature,
            # No per-user token-limit control is exposed. Apply the configured
            # limit to existing users too, rather than freezing their signup value.
            "max_tokens": config.default_max_tokens,
        }

    async def set_model(self, telegram_id: int, model: str) -> None:
        """Set the AI model for a user."""
        await self.user_repo.update_model(telegram_id, model)
        await self.session.commit()

    async def set_system_prompt(self, telegram_id: int, prompt: str) -> None:
        """Set the system prompt for a user."""
        await self.user_repo.update_system_prompt(telegram_id, prompt)
        await self.session.commit()

    # --- History -----------------------------------------------------------

    async def add_message(
        self,
        chat_id: int,
        user_id: int,
        role: str,
        content: str,
        image_url: str | None = None,
        model: str | None = None,
    ) -> Message | None:
        """Save a message to the database and return the stored row.

        The returned row carries the database id, which the per-answer export
        buttons embed in their callback data.
        """
        saved = await self.message_repo.add_message(
            chat_id=chat_id,
            user_id=user_id,
            role=role,
            content=content,
            image_url=image_url,
            model=model,
        )
        await self.session.commit()
        return saved

    async def get_history(
        self, chat_id: int, limit: int | None = None, user_id: int | None = None
    ) -> list[dict[str, str]]:
        """Get the conversation history for a chat (one user's part if given)."""
        return await self.message_repo.get_history(
            chat_id, limit=limit or config.max_history_length, user_id=user_id
        )

    async def clear_history(self, chat_id: int, user_id: int | None = None) -> None:
        """Clear the messages of a conversation (only ``user_id``'s when given)."""
        await self.message_repo.clear_history(chat_id, user_id=user_id)
        await self.session.commit()

    # --- Export ----------------------------------------------------------------

    async def get_message_for_export(
        self, message_id: int, user_id: int
    ) -> Message | None:
        """One stored message, but only when it belongs to ``user_id``.

        The owner check lives here so a callback with a guessed id can never
        export somebody else's answer.
        """
        message = await self.message_repo.get_message(message_id)
        if message is None or message.user_id != user_id:
            return None
        return message

    async def get_question_for(self, answer: Message) -> Message | None:
        """The user message that preceded ``answer`` in the same thread."""
        if answer.chat_id is None:
            return None
        return await self.message_repo.get_previous_user_message(
            answer.chat_id, answer.user_id, answer.id
        )

    async def get_export_history(
        self, chat_id: int, user_id: int, limit: int | None = None
    ) -> list[Message]:
        """The user's thread in ``chat_id`` (oldest first) for /export."""
        rows = await self.message_repo.get_messages_for_export(
            chat_id, user_id, limit or config.export_max_messages
        )
        # Detach: attributes stay loaded after the session closes.
        await self.session.commit()
        return rows

    # --- Guest access ------------------------------------------------------

    async def get_guest_messages_used(self, telegram_id: int) -> int:
        """Get the number of free-trial messages used by a guest."""
        return await self.user_repo.get_guest_messages_used(telegram_id)

    async def increment_guest_messages(self, telegram_id: int) -> int:
        """Increment the guest message counter and return the new value."""
        count = await self.user_repo.increment_guest_messages(telegram_id)
        await self.session.commit()
        return count

    async def consume_guest_message(self, telegram_id: int) -> int | None:
        """Atomically take a free-trial message; the new count, or None at the limit."""
        count = await self.user_repo.try_consume_guest_message(
            telegram_id, config.guest_message_limit
        )
        await self.session.commit()
        return count

    async def release_guest_message(self, telegram_id: int) -> None:
        """Return a free-trial message after a failed request."""
        await self.user_repo.release_guest_message(telegram_id)
        await self.session.commit()

    async def get_guest_messages_remaining(self, telegram_id: int) -> int:
        """Get the number of remaining free-trial messages for a guest."""
        used = await self.get_guest_messages_used(telegram_id)
        return max(0, config.guest_message_limit - used)

    # --- Reminders -----------------------------------------------------------

    async def count_reminders(
        self, telegram_id: int, *, recurring: bool | None = None
    ) -> int:
        """Count a user's enabled reminders (all / one-time / recurring)."""
        return await self.reminder_repo.count(telegram_id, recurring=recurring)

    async def add_reminder(self, reminder: Reminder) -> Reminder:
        """Persist a new reminder and return it (with its assigned id)."""
        created = await self.reminder_repo.add(reminder)
        await self.session.commit()
        return created

    async def list_reminders(self, telegram_id: int) -> list[Reminder]:
        """All enabled reminders of a user, soonest first."""
        reminders = await self.reminder_repo.list_by_user(telegram_id)
        # Detach the plain data before the session closes.
        await self.session.commit()
        return reminders

    async def delete_reminder(self, reminder_id: int, telegram_id: int) -> bool:
        """Delete one reminder; True when it existed and belonged to the user."""
        deleted = await self.reminder_repo.delete(reminder_id, telegram_id)
        await self.session.commit()
        return deleted

    async def delete_all_reminders(self, telegram_id: int) -> int:
        """Delete every reminder of a user; returns how many were removed."""
        count = await self.reminder_repo.delete_all(telegram_id)
        await self.session.commit()
        return count

    # --- Statistics --------------------------------------------------------

    async def get_stats(self, telegram_id: int) -> dict:
        """Get usage statistics for a user."""
        settings = await self.get_user_settings(telegram_id)
        msg_count = await self.message_repo.get_message_count(telegram_id)
        return {
            "message_count": msg_count,
            "model": settings["model"],
            "system_prompt": settings["system_prompt"],
        }

    # --- Allowed users management -----------------------------------------

    async def add_allowed_user(
        self,
        telegram_id: int,
        added_by: int | None = None,
        username: str | None = None,
        first_name: str | None = None,
        last_name: str | None = None,
    ) -> None:
        """Add a user to the allowed list, caching any known profile data."""
        user = await self.user_repo.get_user(telegram_id)
        await self.user_repo.add_allowed_user(
            telegram_id,
            added_by,
            username=username if username is not None else (user.username if user else None),
            first_name=(
                first_name if first_name is not None else (user.first_name if user else None)
            ),
            last_name=last_name if last_name is not None else (user.last_name if user else None),
        )
        await self.session.commit()

    async def find_user_by_username(self, username: str) -> int | None:
        """Telegram ID of a registered user found by ``@username``, if any."""
        user = await self.user_repo.find_user_by_username(username)
        return user.telegram_id if user is not None else None

    async def remove_allowed_user(self, telegram_id: int) -> bool:
        """Remove a user from the allowed list. Returns True if removed."""
        result = await self.user_repo.remove_allowed_user(telegram_id)
        await self.session.commit()
        return result

    async def get_allowed_users(self) -> list[dict]:
        """Get all allowed users with their profile info."""
        allowed_users = await self.user_repo.get_allowed_users()
        return [await self._allowed_user_dict(au) for au in allowed_users]

    async def get_allowed_user(self, telegram_id: int) -> dict | None:
        """Get one allowed user with profile info, or ``None`` when not listed."""
        allowed = await self.user_repo.get_allowed_user(telegram_id)
        if allowed is None:
            return None
        return await self._allowed_user_dict(allowed)

    async def _allowed_user_dict(self, allowed: AllowedUser) -> dict:
        """Merge an allowed_users row with the live profile (if registered)."""
        telegram_id = allowed.telegram_id
        user = await self.user_repo.get_user(telegram_id)
        return {
            "telegram_id": telegram_id,
            "added_at": allowed.added_at,
            "added_by": allowed.added_by,
            # Live profile from ``users`` wins; the cached copy on the
            # allowed_users row covers people who never wrote to the bot.
            "username": user.username if user and user.username else allowed.username,
            "first_name": (
                user.first_name if user and user.first_name else allowed.first_name
            ),
            "last_name": (
                user.last_name if user and user.last_name else allowed.last_name
            ),
        }

    async def update_allowed_user_profile(
        self,
        telegram_id: int,
        username: str | None = None,
        first_name: str | None = None,
        last_name: str | None = None,
    ) -> None:
        """Refresh the cached profile of an allowed user."""
        await self.user_repo.update_allowed_user_profile(
            telegram_id,
            username=username,
            first_name=first_name,
            last_name=last_name,
        )
        await self.session.commit()

    async def is_allowed_user(self, telegram_id: int) -> bool:
        """Check if a user is in the allowed list."""
        return await self.user_repo.is_allowed_user(telegram_id)

    async def get_all_user_ids(self) -> list[int]:
        """Return the Telegram IDs of every registered user (for broadcasts)."""
        return await self.user_repo.get_all_user_ids()

    # --- Daily usage -------------------------------------------------------

    async def get_daily_usage_used(self, telegram_id: int) -> int:
        """Get the number of questions asked today."""
        usage = await self.user_repo.get_daily_usage(
            telegram_id, datetime.date.today()
        )
        return usage.query_count if usage else 0

    async def increment_daily_usage(self, telegram_id: int) -> int:
        """Increment today's usage counter and return the new value."""
        count = await self.user_repo.increment_daily_usage(
            telegram_id, datetime.date.today()
        )
        await self.session.commit()
        return count

    async def consume_daily_usage(self, telegram_id: int) -> int | None:
        """Atomically count a query; the new count, or None at the daily limit."""
        count = await self.user_repo.try_consume_daily_usage(
            telegram_id, datetime.date.today(), config.daily_limit
        )
        await self.session.commit()
        return count

    async def release_daily_usage(self, telegram_id: int) -> None:
        """Return a query after a failed request."""
        await self.user_repo.release_daily_usage(telegram_id, datetime.date.today())
        await self.session.commit()

    async def get_daily_usage_remaining(
        self, telegram_id: int, daily_limit: int | None = None
    ) -> int:
        """Get the number of remaining questions for today."""
        limit = config.daily_limit if daily_limit is None else daily_limit
        used = await self.get_daily_usage_used(telegram_id)
        return max(0, limit - used)

    async def reset_daily_usage(self, telegram_id: int) -> bool:
        """Reset today's usage counter. Returns True if a record existed."""
        result = await self.user_repo.reset_daily_usage(
            telegram_id, datetime.date.today()
        )
        await self.session.commit()
        return result

    async def get_usage_stats_last_14_days(self) -> list[dict]:
        """Get usage statistics for all users over the last 14 days."""
        return await self.user_repo.get_usage_stats_last_14_days()

    async def get_user_total_usage_last_14_days(self, telegram_id: int) -> int:
        """Get total usage for a specific user over the last 14 days."""
        return await self.user_repo.get_user_total_usage_last_14_days(telegram_id)


@asynccontextmanager
async def user_service() -> AsyncIterator[UserService]:
    """Yield a :class:`UserService` bound to a short-lived database session.

    Replaces the repeated boilerplate::

        async with async_session_factory() as session:
            service = UserService(session)
            ...

    with::

        async with user_service() as service:
            ...

    The import of the session factory is local to avoid a circular import
    (``bot.database.session`` must not import this module at load time).
    """
    from bot.database.session import async_session_factory

    async with async_session_factory() as session:
        yield UserService(session)
