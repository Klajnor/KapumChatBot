"""Локальные зависимости обработчиков: не изменяют workflow-data хоста."""

from typing import Any

from aiogram import BaseMiddleware
from aiogram.types import CallbackQuery, Message

from .access import private_owner_message, samo_callback
from .config import SamoHelperConfig


class SamoAccessMiddleware(BaseMiddleware):
    def __init__(self, config: SamoHelperConfig, legacy_commands: bool) -> None:
        self._config = config
        self._legacy_commands = legacy_commands

    async def __call__(self, handler, event, data):
        if isinstance(event, Message):
            allowed = private_owner_message(event, self._config)
        elif isinstance(event, CallbackQuery):
            allowed = samo_callback(
                event, self._config.owner_id, self._config.notification_chat_id,
                data["bot"], allow_legacy=self._legacy_commands,
            )
        else:
            allowed = False
        # Команда/кнопка уже сопоставлена нашему namespace. Молча поглощаем
        # запрещённый вызов, чтобы он не попал в catch-all ИИ приложения-хоста.
        if not allowed:
            return None
        return await handler(event, data)


class SamoContextMiddleware(BaseMiddleware):
    def __init__(self, **dependencies: Any) -> None:
        self._dependencies = dependencies

    async def __call__(self, handler, event, data):
        # Имена db/config/llm нужны существующим обработчикам, но видны только
        # внутри рабочего роутера. Исходный словарь других модулей не меняем.
        return await handler(event, {**data, **self._dependencies})
