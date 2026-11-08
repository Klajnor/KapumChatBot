from __future__ import annotations

import datetime

from sqlalchemy import (
    BigInteger,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    false,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from bot.config import DEFAULT_SYSTEM_PROMPT_TEXT


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    telegram_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=False)
    username: Mapped[str | None] = mapped_column(String(128), nullable=True)
    first_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    last_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    is_admin: Mapped[bool] = mapped_column(default=False)
    guest_messages_used: Mapped[int] = mapped_column(Integer, default=0)
    # Access level set from the admin menu. ``is_admin`` above only mirrors
    # OWNER_IDS (owners); ``granted_admin`` is an admin promoted at runtime.
    granted_admin: Mapped[bool] = mapped_column(default=False, server_default=false())
    blocked: Mapped[bool] = mapped_column(default=False, server_default=false())
    # Personal reminders timezone: IANA name. NULL means "use the global
    # REMINDER_TIMEZONE". Set with /timezone.
    reminder_timezone: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # Relationships
    settings: Mapped[UserSettings | None] = relationship(
        "UserSettings", back_populates="user", uselist=False, cascade="all, delete-orphan"
    )
    messages: Mapped[list[Message]] = relationship(
        "Message", back_populates="user", cascade="all, delete-orphan"
    )


class UserSettings(Base):
    __tablename__ = "user_settings"

    user_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("users.telegram_id", ondelete="CASCADE"), primary_key=True
    )
    selected_model: Mapped[str] = mapped_column(
        String(256), default="qwen3.8-flash"
    )
    system_prompt: Mapped[str] = mapped_column(
        Text,
        default=DEFAULT_SYSTEM_PROMPT_TEXT,
    )
    temperature: Mapped[float] = mapped_column(default=0.7)
    max_tokens: Mapped[int] = mapped_column(default=2048)

    # Relationships
    user: Mapped[User] = relationship("User", back_populates="settings")


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("users.telegram_id", ondelete="CASCADE"), index=True
    )
    # Conversation the message belongs to (private chat or group). For private
    # chats this equals the user's Telegram ID.
    chat_id: Mapped[int | None] = mapped_column(BigInteger, index=True, nullable=True)
    role: Mapped[str] = mapped_column(String(16))  # "user" or "assistant"
    content: Mapped[str] = mapped_column(Text)
    image_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    model: Mapped[str | None] = mapped_column(String(256), nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationships
    user: Mapped[User] = relationship("User", back_populates="messages")


class AllowedUser(Base):
    """Users who are allowed to ask questions to the bot (managed by admin)."""

    __tablename__ = "allowed_users"

    telegram_id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=False
    )
    added_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    added_by: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    # Cached Telegram profile so /list_users can show a nickname even before
    # the user has written to the bot (and therefore has no ``users`` row).
    username: Mapped[str | None] = mapped_column(String(128), nullable=True)
    first_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    last_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    # Personal daily question limit; NULL means "use DAILY_LIMIT".
    daily_limit: Mapped[int | None] = mapped_column(Integer, nullable=True)


class DailyUsage(Base):
    """Daily usage tracking — how many queries a user made on a given day."""

    __tablename__ = "daily_usage"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    telegram_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("users.telegram_id", ondelete="CASCADE"), index=True
    )
    date: Mapped[datetime.date] = mapped_column(Date, nullable=False)
    query_count: Mapped[int] = mapped_column(Integer, default=0)

    __table_args__ = (
        UniqueConstraint("telegram_id", "date", name="uq_daily_usage_user_date"),
    )


class PhotoIntentUsage(Base):
    """Суточные попытки классификации фото; scope_id=0 — общий бюджет бота."""

    __tablename__ = "photo_intent_usage"

    date: Mapped[datetime.date] = mapped_column(Date, primary_key=True)
    scope_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    call_count: Mapped[int] = mapped_column(Integer, nullable=False)


class ReminderIntentUsage(Base):
    """Общий UTC-бюджет AI-разбора напоминаний и определения часового пояса."""

    __tablename__ = "reminder_intent_usage"

    date: Mapped[datetime.date] = mapped_column(Date, primary_key=True)
    scope_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    call_count: Mapped[int] = mapped_column(Integer, nullable=False)


class StatusIntentHourUsage(Base):
    """Часовые попытки AI-подтверждения запроса статуса (анти-спам)."""

    __tablename__ = "status_intent_hour_usage"

    telegram_id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=False
    )
    hour_start: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), primary_key=True
    )
    call_count: Mapped[int] = mapped_column(Integer, nullable=False)


class StatusIntentDayUsage(Base):
    """Суточные попытки AI-подтверждения запроса статуса (анти-спам)."""

    __tablename__ = "status_intent_day_usage"

    telegram_id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=False
    )
    date: Mapped[datetime.date] = mapped_column(Date, primary_key=True)
    call_count: Mapped[int] = mapped_column(Integer, nullable=False)


class StatusSearchCheck(Base):
    """Последняя живая проверка поиска в отчёте о статусе.

    Отдельная таблица, а не кэш в памяти: часовой лимит должен переживать
    перезапуск бота и не сбрасываться спамом «статус». Хранится только
    Telegram ID и время попытки — без запроса и результатов поиска.
    """

    __tablename__ = "status_search_checks"

    telegram_id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=False
    )
    last_checked_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


class Reminder(Base):
    """Scheduled one-time reminders and recurring messages / AI requests.

    ``text`` is the reminder body, or — when ``ai_mode`` is set — the prompt
    sent to the user's model at fire time (the answer is what gets delivered).
    ``fire_at`` is the next fire time in UTC; one-time reminders (``schedule``
    is NULL) are removed after firing, recurring ones are advanced according
    to ``schedule``: ``every:<minutes>``, ``daily:<HH>:<MM>`` or
    ``weekly:<dow>:<HH>:<MM>`` (dow: 0 = Monday).
    """

    __tablename__ = "reminders"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("users.telegram_id", ondelete="CASCADE"), index=True
    )
    # Chat the reminder fires in (the private chat or group where it was set).
    chat_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    ai_mode: Mapped[bool] = mapped_column(default=False)
    fire_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    schedule: Mapped[str | None] = mapped_column(String(64), nullable=True)
    enabled: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class AppSetting(Base):
    """Runtime settings edited from the admin menu (key → text value)."""

    __tablename__ = "app_settings"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[str] = mapped_column(Text, nullable=False)


class GeneratedSticker(Base):
    """Отправленный стикер: непрозрачный токен кнопки, владелец и Telegram file_id."""

    __tablename__ = "generated_stickers"

    token: Mapped[str] = mapped_column(String(32), primary_key=True)
    user_id: Mapped[int] = mapped_column(BigInteger, index=True)
    bot_id: Mapped[int] = mapped_column(BigInteger)
    file_id: Mapped[str] = mapped_column(Text)
    file_unique_id: Mapped[str] = mapped_column(String(256))


class StickerPack(Base):
    """Набор, созданный этим ботом для конкретного пользователя."""

    __tablename__ = "sticker_packs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(BigInteger, index=True)
    bot_id: Mapped[int] = mapped_column(BigInteger)
    name: Mapped[str] = mapped_column(String(64), unique=True)
    title: Mapped[str] = mapped_column(String(64))
    # Резервируем имя до Telegram-запроса: повтор после сетевого сбоя безопасен.
    created: Mapped[bool] = mapped_column(default=False)


class DigestChannel(Base):
    """A channel (or group) that gets a daily summary of its messages."""

    __tablename__ = "digest_channels"

    chat_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=False)
    title: Mapped[str] = mapped_column(String(256), default="")
    added_by: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    added_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    # End of the window covered by the previous summary (NULL = never).
    last_digest_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class ChannelMessage(Base):
    """A text message observed in a digest channel (input for the summary)."""

    __tablename__ = "channel_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    chat_id: Mapped[int] = mapped_column(BigInteger, index=True)
    message_id: Mapped[int] = mapped_column(BigInteger)
    author: Mapped[str] = mapped_column(String(128), default="")
    text: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), index=True
    )

    __table_args__ = (
        UniqueConstraint("chat_id", "message_id", name="uq_channel_message"),
    )
