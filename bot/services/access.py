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
    role: str  # "admin" | "allowed" | "guest" | "blocked" | "denied"
    message: str | None = None
    remaining: int | None = None
    consumed: bool = False  # a quota unit was taken and can be refunded


def _denied_message() -> str:
    return (
        "🌌 <b>Access restricted.</b>\n\n"
        "This bot is available to authorized users only.\n\n"
        + get_random_themed_response()
    )


def _blocked_result() -> AccessResult:
    return AccessResult(
        allowed=False,
        role="blocked",
        message=(
            "🚫 <b>Access blocked.</b>\n\n"
            "An administrator has disabled your access to this bot."
        ),
    )


def _daily_limit_result(limit: int) -> AccessResult:
    return AccessResult(
        allowed=False,
        role="allowed",
        message=(
            f"⚠️ <b>Daily limit reached.</b>\n\n"
            f"You have used all {limit} questions for today. "
            f"Come back tomorrow!"
        ),
    )


def _guest_limit_result() -> AccessResult:
    return AccessResult(
        allowed=False,
        role="guest",
        message=(
            "⚠️ <b>Free trial messages used up.</b>\n\n"
            f"You have used all {config.guest_message_limit} free "
            "messages.\nContact an administrator to get full access."
        ),
    )


async def refund_access(service: UserService, telegram_id: int, result: AccessResult) -> None:
    """Give the quota unit back when the request failed on our side."""
    if not result.consumed:
        return
    try:
        if result.role == "allowed":
            await service.release_daily_usage(telegram_id)
        elif result.role == "guest":
            await service.release_guest_message(telegram_id)
        result.consumed = False
    except Exception as exc:  # noqa: BLE001 - a refund must never break the reply
        logger.warning("Could not refund the quota: %s", exc)


async def check_access(
    service: UserService,
    telegram_id: int,
    *,
    consume: bool = True,
    require_quota: bool = True,
) -> AccessResult:
    """Check whether a user may query the bot, consuming quota when allowed.

    The rules are, in order:
      * administrators   — unlimited;
      * blocked users    — denied (even as guests);
      * whitelisted users — their personal limit, else ``config.daily_limit``,
        questions per day;
      * everyone else    — ``config.guest_message_limit`` free-trial messages
        (when guest access is enabled), otherwise denied.

    ``consume=False`` performs the check without taking a quota unit.

    ``require_quota=False`` (only meaningful with ``consume=False``) asks
    "is this user *authorized* at all?" and ignores the remaining budget. Free
    operations — creating or listing reminders — must use it, otherwise a user
    who has spent today's questions is locked out of features that cost nothing.
    """
    if config.is_admin(telegram_id):
        return AccessResult(allowed=True, role="admin")

    if await service.is_blocked(telegram_id):
        return _blocked_result()

    if await service.is_allowed_user(telegram_id):
        limit = await service.get_daily_limit(telegram_id)
        remaining = await service.get_daily_usage_remaining(telegram_id)
        if not consume:
            if require_quota and remaining <= 0:
                return _daily_limit_result(limit)
            return AccessResult(allowed=True, role="allowed", remaining=remaining)
        if remaining <= 0:
            return _daily_limit_result(limit)
        used = await service.consume_daily_usage(telegram_id)
        if used is None:  # a parallel request took the last unit
            return _daily_limit_result(limit)
        return AccessResult(
            allowed=True,
            role="allowed",
            remaining=max(0, limit - used),
            consumed=True,
        )

    if config.allow_guest_access:
        remaining = await service.get_guest_messages_remaining(telegram_id)
        if not consume:
            if require_quota and remaining <= 0:
                return _guest_limit_result()
            return AccessResult(allowed=True, role="guest", remaining=remaining)
        if remaining <= 0:
            return _guest_limit_result()
        used = await service.consume_guest_message(telegram_id)
        if used is None:  # a parallel request took the last unit
            return _guest_limit_result()
        return AccessResult(
            allowed=True,
            role="guest",
            remaining=max(0, config.guest_message_limit - used),
            consumed=True,
        )

    return AccessResult(allowed=False, role="denied", message=_denied_message())


def guest_hint(result: AccessResult) -> str | None:
    """Return a short reminder about remaining free-trial messages, if relevant."""
    if result.role == "guest" and result.remaining is not None and result.remaining <= 2:
        return f"<i>Free trial messages left: {result.remaining}</i>"
    return None