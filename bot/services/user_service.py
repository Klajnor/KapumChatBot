from __future__ import annotations

import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from bot.config import config
from bot.database.repository import MessageRepository, UserRepository


class UserService:
    """Business logic layer for user operations."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.user_repo = UserRepository(session)
        self.message_repo = MessageRepository(session)

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
    ) -> None:
        """Save a message to the database."""
        await self.message_repo.add_message(
            chat_id=chat_id,
            user_id=user_id,
            role=role,
            content=content,
            image_url=image_url,
            model=model,
        )
        await self.session.commit()

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
        self, telegram_id: int, added_by: int | None = None
    ) -> None:
        """Add a user to the allowed list."""
        await self.user_repo.add_allowed_user(telegram_id, added_by)
        await self.session.commit()

    async def remove_allowed_user(self, telegram_id: int) -> bool:
        """Remove a user from the allowed list. Returns True if removed."""
        result = await self.user_repo.remove_allowed_user(telegram_id)
        await self.session.commit()
        return result

    async def get_allowed_users(self) -> list[dict]:
        """Get all allowed users with their profile info."""
        allowed_users = await self.user_repo.get_allowed_users()
        result = []
        for au in allowed_users:
            user = await self.user_repo.get_user(au.telegram_id)
            result.append(
                {
                    "telegram_id": au.telegram_id,
                    "added_at": au.added_at,
                    "added_by": au.added_by,
                    "username": user.username if user else None,
                    "first_name": user.first_name if user else None,
                }
            )
        return result

    async def is_allowed_user(self, telegram_id: int) -> bool:
        """Check if a user is in the allowed list."""
        return await self.user_repo.is_allowed_user(telegram_id)

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
