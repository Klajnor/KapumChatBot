"""Кнопки «✅ Approve / 🚫 Block» под заявкой доступа из первого /start.

Кнопки идут всем админам, поэтому каждое нажатие сначала проверяет
актуальность запроса в БД: если другой админ уже одобрил или заблокировал
пользователя, состояние не меняется вовсе («ничего не делать»), а сообщение
помечается как обработанное. Гонка двух одновременных нажатий гасится
уникальным первичным ключом ``allowed_users.telegram_id`` — второй INSERT
получает IntegrityError и переводится в «уже одобрено».

Callback ``appr:a:<id>`` — одобрить, ``appr:b:<id>`` — заблокировать.
"""

from __future__ import annotations

import html
import logging

from aiogram import F, Router
from aiogram.types import CallbackQuery

from bot.config import config
from bot.services.telegram_utils import safe_answer_callback, safe_edit_text
from bot.services.user_service import user_service

logger = logging.getLogger(__name__)

router = Router()

# Итог → строка под заявкой (HTML) и ответ на нажатие (plain).
_OUTCOMES: dict[str, tuple[str, str]] = {
    "approved": ("✅ <b>Access granted</b>", "✅ Access granted"),
    "blocked": ("🚫 <b>Blocked</b>", "🚫 Blocked"),
    "already_allowed": (
        "ℹ️ <b>Already approved</b> — nothing changed",
        "ℹ️ Already approved by another admin",
    ),
    "already_blocked": (
        "ℹ️ <b>Already blocked</b> — nothing changed",
        "ℹ️ Already blocked by another admin",
    ),
}


def _admin_signature(callback: CallbackQuery) -> str:
    """«Имя (id)» нажавшего админа — для строки под заявкой."""
    actor = callback.from_user
    name = " ".join(
        part
        for part in (
            getattr(actor, "first_name", None),
            getattr(actor, "last_name", None),
        )
        if part
    ).strip() or (getattr(actor, "username", None) or "")
    if name:
        return f"{html.escape(name)} (<code>{actor.id}</code>)"
    return f"<code>{actor.id}</code>"


@router.callback_query(F.data.startswith("appr:"))
async def decide_access(callback: CallbackQuery) -> None:
    """Handle ✅ Approve / 🚫 Block under an access-request notice."""
    actor = callback.from_user.id if callback.from_user else 0
    if not config.is_admin(actor):
        await safe_answer_callback(callback, "Admins only", show_alert=True)
        return
    chat = getattr(callback.message, "chat", None)
    if getattr(chat, "type", None) != "private":
        await safe_answer_callback(
            callback, "Use the private chat with the bot.", show_alert=True
        )
        return

    parts = (callback.data or "").split(":")
    if len(parts) != 3 or parts[1] not in ("a", "b"):
        await safe_answer_callback(callback)
        return
    try:
        user_id = int(parts[2])
    except ValueError:
        await safe_answer_callback(callback)
        return

    approve = parts[1] == "a"
    async with user_service() as service:
        if approve:
            outcome = await service.approve_access_request(user_id, actor)
        else:
            outcome = await service.block_access_request(user_id)

    line, toast = _OUTCOMES.get(outcome, ("⚠️ Unknown outcome", "Unknown outcome"))
    status = f"{line} by {_admin_signature(callback)}."

    # Сообщение могло устареть (InaccessibleMessage) — тогда остаётся тост.
    original = getattr(callback.message, "html_text", None)
    if original:
        await safe_edit_text(
            callback.message,
            f"{original}\n\n{status}",
            parse_mode="HTML",
            reply_markup=None,
        )
    await safe_answer_callback(callback, toast, show_alert=outcome.startswith("already"))

    if outcome == "approved":
        # Новичку больше не нужно угадывать: доступ открыт — сообщаем в личку.
        try:
            await callback.bot.send_message(
                user_id,
                "✅ <b>An administrator has approved your access.</b>\n\n"
                "Send /start to get the full menu.",
            )
        except Exception as exc:  # noqa: BLE001 - уведомление не ломает решение
            logger.info("Could not notify approved user %s: %s", user_id, exc)
