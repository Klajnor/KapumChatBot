from __future__ import annotations

from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from bot.database.models import Message, User, UserSettings


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
        """Get existing user or create a new one with default settings."""
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
            await self.session.flush()

            # Create default settings for the new user
            settings = UserSettings(user_id=telegram_id)
            self.session.add(settings)
            await self.session.flush()
        else:
            # Update user info
            if username is not None:
                user.username = username
            if first_name is not None:
                user.first_name = first_name
            if last_name is not None:
                user.last_name = last_name

        return user

    async def get_user(self, telegram_id: int) -> User | None:
        """Get user by Telegram ID."""
        return await self.session.get(User, telegram_id)

    async def get_guest_messages_used(self, telegram_id: int) -> int:
        """Get the number of messages used by a guest user."""
        user = await self.session.get(User, telegram_id)
        if user is None:
            return 0
        return user.guest_messages_used

    async def increment_guest_messages(self, telegram_id: int) -> None:
        """Increment the guest message counter for a user."""
        user = await self.session.get(User, telegram_id)
        if user is not None:
            user.guest_messages_used += 1
            await self.session.flush()

    async def get_user_settings(self, telegram_id: int) -> UserSettings | None:
        """Get user settings."""
        result = await self.session.execute(
            select(UserSettings).where(UserSettings.user_id == telegram_id)
        )
        return result.scalar_one_or_none()

    async def update_model(self, telegram_id: int, model: str) -> None:
        """Update user's selected AI model."""
        settings = await self.get_user_settings(telegram_id)
        if settings:
            settings.selected_model = model
            await self.session.flush()

    async def update_system_prompt(self, telegram_id: int, prompt: str) -> None:
        """Update user's system prompt."""
        settings = await self.get_user_settings(telegram_id)
        if settings:
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
        settings = await self.get_user_settings(telegram_id)
        if settings:
            if model is not None:
                settings.selected_model = model
            if prompt is not None:
                settings.system_prompt = prompt
            if temperature is not None:
                settings.temperature = temperature
            if max_tokens is not None:
                settings.max_tokens = max_tokens
            await self.session.flush()


class MessageRepository:
    """Repository for message-related database operations."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add_message(
        self,
        user_id: int,
        role: str,
        content: str,
        image_url: str | None = None,
        model: str | None = None,
    ) -> Message:
        """Save a message to the database."""
        message = Message(
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
        self, user_id: int, limit: int = 30
    ) -> list[dict[str, str]]:
        """Get the last N messages for a user as a list of dicts for OpenAI format."""
        result = await self.session.execute(
            select(Message)
            .where(Message.user_id == user_id)
            .order_by(Message.created_at.desc())
            .limit(limit)
        )
        messages = list(reversed(result.scalars().all()))

        formatted = []
        for msg in messages:
            entry: dict[str, str] = {"role": msg.role, "content": msg.content}
            formatted.append(entry)
        return formatted

    async def clear_history(self, user_id: int) -> None:
        """Delete all messages for a user."""
        await self.session.execute(
            delete(Message).where(Message.user_id == user_id)
        )
        await self.session.flush()

    async def get_message_count(self, user_id: int) -> int:
        """Get total message count for a user."""
        result = await self.session.execute(
            select(Message).where(Message.user_id == user_id)
        )
        return len(result.scalars().all())