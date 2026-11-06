"""Подключение рабочего модуля к существующему aiogram Dispatcher хоста."""

from typing import Any

from aiogram import Bot, Dispatcher

from .config import SamoHelperConfig
from .knowledge import KnowledgeClient
from .plugin import SamoHelperPlugin
from .reminders import ReminderService
from .storage import Storage


class SamoHelperExtension:
    """Не владеет Bot, Dispatcher, SQLite или AI-клиентом хоста."""

    def __init__(
        self, bot: Bot, config: SamoHelperConfig, storage: Storage,
        llm: Any = None, knowledge: KnowledgeClient | None = None,
    ) -> None:
        self.plugin = SamoHelperPlugin(bot, config, storage, llm, knowledge=knowledge)
        self.reminders = ReminderService(bot, config.owner_id, storage)

    async def start(self) -> None:
        try:
            await self.reminders.start()
            await self.plugin.start()
        except BaseException as error:
            try:
                await self.stop()
            except Exception as cleanup_error:  # noqa: BLE001
                error.add_note(f"Ошибка остановки рабочего модуля: {type(cleanup_error).__name__}")
            raise

    async def stop(self) -> None:
        try:
            await self.plugin.stop()
        finally:
            await self.reminders.stop()


def attach_samo_helper(
    dispatcher: Dispatcher, bot: Bot, config: SamoHelperConfig, storage: Storage,
    *, llm: Any = None, knowledge: KnowledgeClient | None = None,
) -> SamoHelperExtension:
    """Вызывать до catch-all чата и обработчика неизвестных команд хоста.

    llm — только явно переданный внутренний адаптер с summarize(session, text).
    Отсутствие адаптера отключает резюме, AI из workflow-data не заимствуется.
    Хост останавливает extension в finally до закрытия storage и Bot-сессии.
    """
    extension = SamoHelperExtension(bot, config, storage, llm, knowledge)
    extension.plugin.register(dispatcher)
    dispatcher.startup.register(extension.start)
    dispatcher.shutdown.register(extension.stop)
    return extension
