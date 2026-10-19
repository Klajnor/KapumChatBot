from __future__ import annotations

import logging
from dataclasses import dataclass

from bot.config import config
from bot.services.themed_responses import get_random_themed_response
from bot.services.user_service import UserService

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AccessResult:
    """Outcome of an access check."""

    allowed: bool
    role: str  # "admin" | "allowed" | "guest" | "denied"
    message: str | None = None
    remaining: int | None = None


def _denied_message() -> str:
    return (
        "🌌 <b>Access restricted.</b>\n\n"
        "This bot is available to authorized users only.\n\n"
        + get_random_themed_response()
    )


async def check_access(
    service: UserService, telegram_id: int, *, consume: bool = True
) -> AccessResult:
    """Check whether a user may query the bot, consuming quota when allowed.

    The rules are, in order:
      * administrators   — unlimited;
      * whitelisted users — ``config.daily_limit`` questions per day;
      * everyone else    — ``config.guest_message_limit`` free-trial messages
        (when guest access is enabled), otherwise denied.
    """
    if config.is_admin(telegram_id):
        return AccessResult(allowed=True, role="admin")

    if await service.is_allowed_user(telegram_id):
        remaining = await service.get_daily_usage_remaining(telegram_id)
        if remaining <= 0:
            return AccessResult(
                allowed=False,
                role="allowed",
                message=(
                    f"⚠️ <b>Daily limit reached.</b>\n\n"
                    f"You have used all {config.daily_limit} questions for today. "
                    f"Come back tomorrow!"
                ),
            )
        if not consume:
            return AccessResult(allowed=True, role="allowed", remaining=remaining)
        used = await service.increment_daily_usage(telegram_id)
        return AccessResult(
            allowed=True,
            role="allowed",
            remaining=max(0, config.daily_limit - used),
        )

    if config.allow_guest_access:
        remaining = await service.get_guest_messages_remaining(telegram_id)
        if remaining <= 0:
            return AccessResult(
                allowed=False,
                role="guest",
                message=(
                    "⚠️ <b>Free trial messages used up.</b>\n\n"
                    f"You have used all {config.guest_message_limit} free "
                    "messages.\nContact an administrator to get full access."
                ),
            )
        if not consume:
            return AccessResult(allowed=True, role="guest", remaining=remaining)
        used = await service.increment_guest_messages(telegram_id)
        return AccessResult(
            allowed=True,
            role="guest",
            remaining=max(0, config.guest_message_limit - used),
        )

    return AccessResult(allowed=False, role="denied", message=_denied_message())


def guest_hint(result: AccessResult) -> str | None:
    """Return a short reminder about remaining free-trial messages, if relevant."""
    if result.role == "guest" and result.remaining is not None and result.remaining <= 2:
        return f"<i>Free trial messages left: {result.remaining}</i>"
    return None