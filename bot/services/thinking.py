"""Global thinking (reasoning) settings for user-visible answers.

The level and the optional token budget live in ``app_settings`` and are edited
in ``/admin → 🧠 Thinking`` (owner only). They are applied only to requests
whose answer is shown to a user; internal helpers (intent detection, OCR,
image-description prompts) keep the provider defaults.

The provider vocabulary is Qwen/DashScope:

* ``enable_thinking`` — turn the thinking phase on or off;
* ``reasoning_effort`` — low / medium / high / max (the provider maps the
  values it does not know to its own scale);
* ``thinking_budget`` — an exact cap in thinking tokens.

Qwen 3.8 rejects ``reasoning_effort`` and ``thinking_budget`` in the same
request, so a positive budget overrides the level and only one of the two is
sent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from bot.config import (
    DEFAULT_THINKING_LEVEL,
    MAX_THINKING_BUDGET,
    THINKING_LEVELS,
    config,
)
from bot.database.repository import AppSettingRepository
from bot.database.session import async_session_factory

logger = logging.getLogger(__name__)

SETTING_LEVEL = "thinking_level"
SETTING_BUDGET = "thinking_budget"


@dataclass(frozen=True, slots=True)
class ThinkingSettings:
    """Effective settings plus where each value came from."""

    level: str
    budget: int
    level_custom: bool = False
    budget_custom: bool = False


# Loaded from the database at startup and refreshed on every admin change.
# ``None`` means "not loaded yet": the .env defaults apply.
_current: ThinkingSettings | None = None


def parse_level(value: str | None) -> str | None:
    """Return a known level, or ``None`` for an empty/malformed value."""
    if value is None:
        return None
    level = value.strip().lower()
    return level if level in THINKING_LEVELS else None


def parse_budget(value: str | int | None) -> int | None:
    """Return ``0..MAX_THINKING_BUDGET``, or ``None`` when malformed."""
    if value is None:
        return None
    try:
        number = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if 0 <= number <= MAX_THINKING_BUDGET else None


def current() -> ThinkingSettings:
    """Effective settings; safe to call from synchronous payload builders."""
    if _current is not None:
        return _current
    return ThinkingSettings(config.thinking_level, config.thinking_budget)


def payload_fields() -> dict[str, Any]:
    """Thinking parameters for a chat-completions payload.

    ``off`` disables thinking; a positive budget caps it exactly; otherwise the
    level is sent as ``reasoning_effort``. The three are never combined in a way
    Qwen 3.8 rejects.
    """
    settings = current()
    if settings.level == "off":
        return {"enable_thinking": False}
    fields: dict[str, Any] = {"enable_thinking": True}
    if settings.budget > 0:
        fields["thinking_budget"] = settings.budget
    else:
        fields["reasoning_effort"] = settings.level
    return fields


async def load() -> ThinkingSettings:
    """Read the admin overrides from the database (startup / reload)."""
    global _current
    try:
        async with async_session_factory() as session:
            repo = AppSettingRepository(session)
            level_raw = await repo.get_setting(SETTING_LEVEL)
            budget_raw = await repo.get_setting(SETTING_BUDGET)
    except Exception:  # a broken settings row must not stop the bot
        logger.exception("Could not load thinking settings; using .env defaults")
        _current = None
        return current()

    level = parse_level(level_raw)
    budget = parse_budget(budget_raw)
    if level_raw is not None and level is None:
        logger.warning("Invalid stored %s=%r; using .env default", SETTING_LEVEL, level_raw)
    if budget_raw is not None and budget is None:
        logger.warning("Invalid stored %s=%r; using .env default", SETTING_BUDGET, budget_raw)

    _current = ThinkingSettings(
        level=level or config.thinking_level,
        budget=config.thinking_budget if budget is None else budget,
        level_custom=level is not None,
        budget_custom=budget is not None,
    )
    return _current


async def set_level(level: str) -> ThinkingSettings:
    """Store the level and apply it immediately."""
    global _current
    parsed = parse_level(level)
    if parsed is None:
        raise ValueError(f"unknown thinking level: {level!r}")
    await _store(SETTING_LEVEL, parsed)
    settings = current()
    _current = ThinkingSettings(
        level=parsed,
        budget=settings.budget,
        level_custom=True,
        budget_custom=settings.budget_custom,
    )
    return _current


async def set_budget(budget: int) -> ThinkingSettings:
    """Store the thinking-token cap (0 = use the level) and apply it."""
    global _current
    parsed = parse_budget(budget)
    if parsed is None:
        raise ValueError(f"thinking budget out of range: {budget!r}")
    await _store(SETTING_BUDGET, str(parsed))
    settings = current()
    _current = ThinkingSettings(
        level=settings.level,
        budget=parsed,
        level_custom=settings.level_custom,
        budget_custom=True,
    )
    return _current


async def reset() -> ThinkingSettings:
    """Remove both overrides so the .env defaults apply again."""
    global _current
    async with async_session_factory() as session:
        repo = AppSettingRepository(session)
        await repo.set_setting(SETTING_LEVEL, None)
        await repo.set_setting(SETTING_BUDGET, None)
        await session.commit()
    _current = None
    return current()


async def _store(key: str, value: str | None) -> None:
    async with async_session_factory() as session:
        await AppSettingRepository(session).set_setting(key, value)
        await session.commit()


def reset_cache() -> None:
    """Drop the in-memory copy (tests and process re-initialisation)."""
    global _current
    _current = None


# Re-exported for the admin panel and tests.
__all__ = [
    "DEFAULT_THINKING_LEVEL",
    "MAX_THINKING_BUDGET",
    "SETTING_BUDGET",
    "SETTING_LEVEL",
    "THINKING_LEVELS",
    "ThinkingSettings",
    "current",
    "load",
    "parse_budget",
    "parse_level",
    "payload_fields",
    "reset",
    "reset_cache",
    "set_budget",
    "set_level",
]
