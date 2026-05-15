from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    """Application configuration loaded from environment variables."""

    # Telegram
    bot_token: str = field(default_factory=lambda: os.getenv("BOT_TOKEN", ""))

    # OpenRouter
    openrouter_api_key: str = field(
        default_factory=lambda: os.getenv("OPENROUTER_API_KEY", "")
    )
    openrouter_base_url: str = field(
        default_factory=lambda: os.getenv(
            "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
        )
    )

    # Database
    database_url: str = field(
        default_factory=lambda: os.getenv(
            "DATABASE_URL",
            "postgresql+asyncpg://postgres:postgres@localhost:5432/chat_bot",
        )
    )

    # Defaults
    default_model: str = field(
        default_factory=lambda: os.getenv("DEFAULT_MODEL", "openai/gpt-4o")
    )
    default_system_prompt: str = field(
        default_factory=lambda: os.getenv(
            "DEFAULT_SYSTEM_PROMPT",
            "You are a helpful AI assistant. Answer questions concisely and accurately.",
        )
    )

    # Admin
    admin_ids: list[int] = field(default_factory=list)

    # Limits
    max_history_length: int = field(
        default_factory=lambda: int(os.getenv("MAX_HISTORY_LENGTH", "30"))
    )
    guest_message_limit: int = field(
        default_factory=lambda: int(os.getenv("GUEST_MESSAGE_LIMIT", "5"))
    )

    def __post_init__(self) -> None:
        """Parse admin IDs after initialization."""
        raw = os.getenv("ADMIN_IDS", "")
        if raw:
            self.admin_ids = [int(x.strip()) for x in raw.split(",") if x.strip()]

    def validate(self) -> None:
        """Validate that required config values are present."""
        missing = []
        if not self.bot_token:
            missing.append("BOT_TOKEN")
        if not self.openrouter_api_key:
            missing.append("OPENROUTER_API_KEY")
        if missing:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing)}. "
                f"Check .env file."
            )


config = Config()