"""Инлайн-клавиатура: только для уведомлений poller'а и напоминаний."""

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

# Варианты напоминаний: (подпись кнопки, часы). Только иконка часов + время —
# длинные подписи не влезают в инлайн-кнопку.
REMIND_OPTIONS = (("1 ч", 1), ("4 ч", 4), ("24 ч", 24))

REMIND_HOURS = {h for _, h in REMIND_OPTIONS}


def notify_keyboard() -> InlineKeyboardMarkup:
    """Напоминания + «Отвечено» (удаляет сообщение)."""
    row_remind = [
        InlineKeyboardButton(text=f"🕐 {label}", callback_data=f"samo:rm:{hours}")
        for label, hours in REMIND_OPTIONS
    ]
    row_done = [InlineKeyboardButton(text="✅ Отвечено", callback_data="samo:done")]
    return InlineKeyboardMarkup(inline_keyboard=[row_remind, row_done])
