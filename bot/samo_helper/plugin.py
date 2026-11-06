"""Сборка рабочего модуля без собственного Telegram polling и базы данных."""

from __future__ import annotations

import logging
from typing import Any

from aiogram import Bot, Dispatcher

from .config import SamoHelperConfig
from .context import SamoContextMiddleware
from .handlers import create_samo_router
from .knowledge import KnowledgeClient
from .llm import LLMClient, is_blocked_public_host
from .poller import RedminePoller
from .redmine import RedmineClient
from .storage import Storage
from .yougile import YouGileClient

log = logging.getLogger("kapum.samo_helper")


def create_yougile(config: SamoHelperConfig, storage: Storage) -> YouGileClient | None:
    if not config.yougile_enabled:
        log.info("YouGile не настроен — перенос сообщений выключен")
        return None
    client = YouGileClient(
        config.yougile_key, config.yougile_project, config.yougile_board,
        config.yougile_column, base_url=config.yougile_api_url,
    )
    known = storage.get_meta("yougile_column_id")
    if known:
        client.restore_column(known)
    client.on_column_resolved = lambda cid: storage.set_meta("yougile_column_id", cid)
    return client


class SamoHelperPlugin:
    id = "samo_helper"

    def __init__(
        self, bot: Bot, config: SamoHelperConfig, storage: Storage, llm: Any = None,
        *, legacy_commands: bool = False, knowledge: KnowledgeClient | None = None,
    ) -> None:
        if not isinstance(config, SamoHelperConfig):
            raise TypeError("Плагин принимает отдельный SamoHelperConfig")
        self._config = config
        self._storage = storage
        self._legacy_commands = legacy_commands
        self._registered = False
        self.redmine = RedmineClient(config.redmine_url, config.redmine_api_key)
        self.yougile = create_yougile(config, storage)
        if knowledge is None and config.knowledge_enabled:
            knowledge = KnowledgeClient(
                config.knowledge_url or "", config.knowledge_api_key,
                timeout=config.knowledge_timeout,
                allowed_hosts=config.knowledge_allowed_hosts,
            )
        self.knowledge = knowledge
        if llm is None and config.llm_enabled:
            if config.llm_internal_only and is_blocked_public_host(config.llm_api_url or ""):
                raise ValueError(
                    "SAMO_HELPER_LLM_API_URL указывает на публичный AI-хост, "
                    "а SAMO_HELPER_LLM_INTERNAL_ONLY=true"
                )
            llm = LLMClient(
                config.llm_api_key, config.llm_api_url or "", config.llm_model or "",
                on_usage=lambda ok, usage: storage.log_event(
                    "ai_request", tokens=int(usage.get("total_tokens") or 0),
                ),
            )
        self._llm = llm
        self.poller = RedminePoller(
            bot, config, storage, self.redmine, self._llm, self.yougile, self.knowledge,
        )

    def register(self, dispatcher: Dispatcher) -> None:
        if self._registered:
            raise RuntimeError("Рабочий плагин уже подключён к приложению")
        if any(router.name == self.id for router in dispatcher.sub_routers):
            raise RuntimeError("В приложении уже есть рабочий роутер")
        router = create_samo_router(self._config, legacy_commands=self._legacy_commands)
        context = SamoContextMiddleware(
            db=self._storage, config=self._config, llm=self._llm,
            redmine=self.redmine, yougile=self.yougile, poller=self.poller,
            knowledge=self.knowledge,
        )
        router.message.middleware(context)
        router.callback_query.middleware(context)
        dispatcher.include_router(router)
        self._registered = True

    async def start(self) -> None:
        await self.poller.start()

    async def stop(self) -> None:
        await self.poller.stop()
