"""Свободный запрос статуса: «как дела», «статус» и похожие фразы.

Маршрут намеренно отделён от ``handle_text``: отчёт бесплатен для квоты, не
сохраняется в историю и не передаёт текст пользователя в обычный AI-ответ.
Короткий AI-вызов только подтверждает намерение после дешёвого префильтра.
"""

from __future__ import annotations

import logging

from aiogram.enums import ChatType
from aiogram.types import Message

from bot.config import config
from bot.services.access import check_access
from bot.services.progress import waiting
from bot.services.request_intent import (
    Intent,
    classify_request,
    looks_like_status_request,
)
from bot.services.status import render_status
from bot.services.telegram_utils import is_forwarded, safe_answer
from bot.services.user_service import user_service

logger = logging.getLogger(__name__)


async def maybe_status_request(message: Message, text: str) -> bool:
    """Answer a status request; ``True`` when it was handled.

    Returns ``False`` for ordinary messages and for blocked/unauthorized users:
    the normal access check in ``handle_text`` then shows the usual denial.
    """
    user = message.from_user
    if (
        user is None
        or not config.request_intent
        or is_forwarded(message)
        or not looks_like_status_request(text)
    ):
        return False

    async with user_service() as service:
        # Status is free for the question quota, but a blocked user must not
        # receive bot internals or spend the hourly search probe.
        access = await check_access(
            service, user.id, consume=False, require_quota=False
        )
        if not access.allowed:
            return False

        settings = await service.get_user_settings(user.id)
        try:
            intent = await classify_request(
                text, settings["model"], skip_reminder=True
            )
        except Exception:
            logger.warning("Status intent check failed", exc_info=True)
            return False
        if intent is not Intent.STATUS:
            return False

        try:
            async with waiting(message, text="⏳ Checking bot status…"):
                report = await render_status(
                    service,
                    user.id,
                    private=message.chat.type == ChatType.PRIVATE,
                )
        except Exception:
            logger.warning("Status report failed", exc_info=True)
            await safe_answer(
                message,
                "⚠️ Could not check the bot status right now. Please try again later.",
            )
            return True

    await safe_answer(message, report)
    return True
