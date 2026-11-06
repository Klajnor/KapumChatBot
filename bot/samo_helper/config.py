"""Настройки samo_helper внутри KapumChatBot.

Все переменные имеют префикс ``SAMO_HELPER_`` и не пересекаются с настройками
диалогового бота. Рабочая SQLite — отдельный абсолютный файл; БД Kapum не
используется и не заменяется.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from urllib.parse import urlparse


def _enabled(value: str, name: str) -> bool:
    value = value.strip().lower()
    if value not in {"true", "false", "1", "0"}:
        raise RuntimeError(f"{name}: допустимы true/false или 1/0")
    return value in {"true", "1"}


@dataclass(frozen=True)
class SamoHelperConfig:
    owner_id: int
    notification_chat_id: int
    redmine_url: str
    redmine_api_key: str = field(repr=False)
    db_path: str = ""
    redmine_poll_minutes: int = 5
    yougile_key: str | None = field(default=None, repr=False)
    yougile_project: str | None = None
    yougile_board: str | None = None
    yougile_column: str | None = None
    yougile_api_url: str = "https://ru.yougile.com"
    knowledge_url: str | None = field(default=None, repr=False)
    knowledge_api_key: str | None = field(default=None, repr=False)
    knowledge_timeout: float = 10
    knowledge_max_results: int = 5
    wiki_url: str | None = None
    knowledge_allowed_hosts: tuple[str, ...] = ()
    llm_api_url: str | None = None
    llm_api_key: str | None = field(default=None, repr=False)
    llm_model: str | None = None
    llm_internal_only: bool = True

    def __post_init__(self) -> None:
        if self.owner_id <= 0:
            raise ValueError("samo_helper требует положительный ID владельца")
        if self.notification_chat_id >= 0:
            raise ValueError(
                "SAMO_HELPER_NOTIFICATION_CHAT_ID должен быть отрицательным ID группы"
            )
        if not self.redmine_url.strip() or not self.redmine_api_key.strip():
            raise ValueError(
                "samo_helper требует SAMO_HELPER_REDMINE_URL и "
                "SAMO_HELPER_REDMINE_API_KEY"
            )
        if self.redmine_poll_minutes <= 0:
            raise ValueError("SAMO_HELPER_POLL_MINUTES должен быть положительным")
        if not self.db_path or not os.path.isabs(self.db_path):
            raise ValueError("SAMO_HELPER_DB_PATH должен быть абсолютным путём")
        if not 1 <= self.knowledge_timeout <= 60:
            raise ValueError("SAMO_HELPER_KNOWLEDGE_TIMEOUT должен быть от 1 до 60")
        if not 1 <= self.knowledge_max_results <= 20:
            raise ValueError("SAMO_HELPER_KNOWLEDGE_MAX_RESULTS должен быть от 1 до 20")
        for name, url in (
            ("SAMO_HELPER_KNOWLEDGE_URL", self.knowledge_url),
            ("SAMO_HELPER_WIKI_URL", self.wiki_url),
            ("SAMO_HELPER_LLM_API_URL", self.llm_api_url),
        ):
            if url:
                parsed = urlparse(url)
                if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                    raise ValueError(f"{name} должен быть http(s)-адресом")
        if bool(self.llm_api_url) != bool(self.llm_model):
            raise ValueError(
                "SAMO_HELPER_LLM_API_URL и SAMO_HELPER_LLM_MODEL задаются вместе"
            )

    @property
    def yougile_enabled(self) -> bool:
        return bool(
            self.yougile_key
            and self.yougile_project
            and self.yougile_board
            and self.yougile_column
        )

    @property
    def knowledge_enabled(self) -> bool:
        return bool(self.knowledge_url)

    @property
    def llm_enabled(self) -> bool:
        return bool(self.llm_api_url and self.llm_model)


def load_samo_config(
    owner_id: int,
    db_path: str,
    environ: Mapping[str, str] | None = None,
    *,
    default_enabled: bool = False,
) -> SamoHelperConfig | None:
    """Загрузить настройки; по умолчанию модуль в Kapum выключен."""
    env = os.environ if environ is None else environ
    raw_enabled = env.get("SAMO_HELPER_ENABLED")
    if raw_enabled is None:
        if not default_enabled:
            return None
        raw_enabled = "true"
    if not _enabled(raw_enabled, "SAMO_HELPER_ENABLED"):
        return None

    for name in ("SAMO_HELPER_REDMINE_URL", "SAMO_HELPER_REDMINE_API_KEY"):
        if not env.get(name, "").strip():
            raise RuntimeError(f"{name} не задан — заполните настройки samo_helper")
    try:
        chat_id = int(env.get("SAMO_HELPER_NOTIFICATION_CHAT_ID", ""))
    except ValueError:
        raise RuntimeError(
            "SAMO_HELPER_NOTIFICATION_CHAT_ID должен быть отрицательным ID группы"
        ) from None
    try:
        minutes = int(env.get("SAMO_HELPER_POLL_MINUTES", "5"))
    except ValueError:
        raise RuntimeError("SAMO_HELPER_POLL_MINUTES должен быть целым числом") from None
    try:
        knowledge_timeout = float(env.get("SAMO_HELPER_KNOWLEDGE_TIMEOUT", "10"))
    except ValueError:
        raise RuntimeError("SAMO_HELPER_KNOWLEDGE_TIMEOUT должен быть числом") from None
    try:
        knowledge_max_results = int(env.get("SAMO_HELPER_KNOWLEDGE_MAX_RESULTS", "5"))
    except ValueError:
        raise RuntimeError("SAMO_HELPER_KNOWLEDGE_MAX_RESULTS должен быть целым") from None

    redmine_url = env["SAMO_HELPER_REDMINE_URL"].strip()
    wiki_url = (env.get("SAMO_HELPER_WIKI_URL") or "").strip() or None
    knowledge_url = (env.get("SAMO_HELPER_KNOWLEDGE_URL") or "").strip() or None
    allowed = [
        value.strip().lower()
        for value in env.get("SAMO_HELPER_ALLOWED_HOSTS", "").split(",")
        if value.strip()
    ]
    for url in (redmine_url, wiki_url):
        host = urlparse(url).hostname
        if host and host.lower() not in allowed:
            allowed.append(host.lower())

    llm_url = (env.get("SAMO_HELPER_LLM_API_URL") or "").strip() or None
    llm_model = (env.get("SAMO_HELPER_LLM_MODEL") or "").strip() or None
    llm_internal_only = _enabled(
        env.get("SAMO_HELPER_LLM_INTERNAL_ONLY", "true"),
        "SAMO_HELPER_LLM_INTERNAL_ONLY",
    )
    try:
        return SamoHelperConfig(
            owner_id=owner_id,
            notification_chat_id=chat_id,
            redmine_url=redmine_url,
            redmine_api_key=env["SAMO_HELPER_REDMINE_API_KEY"].strip(),
            db_path=db_path,
            redmine_poll_minutes=minutes,
            yougile_key=(env.get("SAMO_HELPER_YOUGILE_KEY") or "").strip() or None,
            yougile_project=(env.get("SAMO_HELPER_YOUGILE_PROJECT") or "").strip() or None,
            yougile_board=(env.get("SAMO_HELPER_YOUGILE_BOARD") or "").strip() or None,
            yougile_column=(env.get("SAMO_HELPER_YOUGILE_COLUMN") or "").strip() or None,
            yougile_api_url=(
                env.get("SAMO_HELPER_YOUGILE_API_URL") or "https://ru.yougile.com"
            ).strip(),
            knowledge_url=knowledge_url,
            knowledge_api_key=(env.get("SAMO_HELPER_KNOWLEDGE_API_KEY") or "").strip() or None,
            knowledge_timeout=knowledge_timeout,
            knowledge_max_results=knowledge_max_results,
            wiki_url=wiki_url,
            knowledge_allowed_hosts=tuple(allowed),
            llm_api_url=llm_url,
            llm_api_key=(env.get("SAMO_HELPER_LLM_API_KEY") or "").strip() or None,
            llm_model=llm_model,
            llm_internal_only=llm_internal_only,
        )
    except ValueError as error:
        raise RuntimeError(str(error)) from None
