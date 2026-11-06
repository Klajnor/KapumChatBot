"""Общие личные напоминания, независимые от рабочего плагина Redmine."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramAPIError

from .keyboards import notify_keyboard
from .storage import Storage

log = logging.getLogger("kapum.samo_helper.reminders")


class ReminderService:
    def __init__(self, bot: Any, owner_id: int, storage: Storage) -> None:
        self._bot = bot
        self._owner_id = owner_id
        self._db = storage
        self._stop = asyncio.Event()
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        if self._task is not None:
            return
        self._stop.clear()
        self._task = asyncio.create_task(self._run(), name="reminder-loop")

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            try:
                await self._task
            finally:
                self._task = None

    async def _run(self) -> None:
        while not self._stop.is_set():
            try:
                await self.tick()
            except Exception:
                log.exception("Ошибка в цикле напоминаний")
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=30)
            except TimeoutError:
                pass

    async def tick(self) -> None:
        for row in self._db.due_reminders(int(time.time())):
            try:
                await self._bot.send_message(
                    self._owner_id, row["text"], reply_markup=notify_keyboard(),
                    parse_mode=ParseMode.HTML if row["fmt"] == "html" else ParseMode.MARKDOWN_V2,
                )
            except TelegramAPIError:
                log.exception("Не удалось отправить напоминание #%s", row["id"])
            self._db.delete_reminder(row["id"])
