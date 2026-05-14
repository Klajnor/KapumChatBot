from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from bot.config import config
from bot.database.repository import MessageRepository, UserRepository


class UserService:
    """Business logic layer for user operations."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.user_repo = UserRepository(session)
        self.message_repo = MessageRepository(session)

    async def register_user(
        self,
        telegram_id: int,
        username: str | None = None,
        first_name: str | None = None,
        last_name: str | None = None,
    ) -> None:
        """Register or update a user in the database."""
        is_admin = telegram_id in config.admin_ids
        await self.user_repo.get_or_create_user(
            telegram_id=telegram_id,
            username=username,
            first_name=first_name,
            last_name=last_name,
            is_admin=is_admin,
        )
        await self.session.commit()

    async def get_user_settings(self, telegram_id: int) -> dict:
        """Get user settings as a dictionary."""
        settings = await self.user_repo.get_user_settings(telegram_id)
        if settings is None:
            return {
                "model": config.default_model,
                "system_prompt": config.default_system_prompt,
                "temperature": 0.7,
                "max_tokens": 2048,
            }
        return {
            "model": settings.selected_model,
            "system_prompt": settings.system_prompt,
            "temperature": settings.temperature,
            "max_tokens": settings.max_tokens,
        }

    async def set_model(self, telegram_id: int, model: str) -> None:
        """Set the AI model for a user."""
        await self.user_repo.update_model(telegram_id, model)
        await self.session.commit()

    async def set_system_prompt(self, telegram_id: int, prompt: str) -> None:
        """Set the system prompt for a user."""
        await self.user_repo.update_system_prompt(telegram_id, prompt)
        await self.session.commit()

    async def get_history(self, telegram_id: int) -> list[dict[str, str]]:
        """Get conversation history for a user."""
        return await self.message_repo.get_history(
            telegram_id, limit=config.max_history_length
        )

    async def add_message(
        self,
        telegram_id: int,
        role: str,
        content: str,
        image_url: str | None = None,
        model: str | None = None,
    ) -> None:
        """Save a message to the database."""
        await self.message_repo.add_message(
            user_id=telegram_id,
            role=role,
            content=content,
            image_url=image_url,
            model=model,
        )
        await self.session.commit()

    async def clear_history(self, telegram_id: int) -> None:
        """Clear all messages for a user."""
        await self.message_repo.clear_history(telegram_id)
        await self.session.commit()

    async def get_stats(self, telegram_id: int) -> dict:
        """Get usage statistics for a user."""
        settings = await self.get_user_settings(telegram_id)
        msg_count = await self.message_repo.get_message_count(telegram_id)
        return {
            "message_count": msg_count,
            "model": settings["model"],
            "system_prompt": settings["system_prompt"],
        }