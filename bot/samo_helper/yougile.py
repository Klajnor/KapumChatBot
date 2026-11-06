"""Клиент YouGile REST API v2.

Документация: https://ru.yougile.com/api-v2 (OpenAPI-спек: https://ru.yougile.com/api-json).
Авторизация: `Authorization: Bearer <ключ>`; ключ создаётся POST /api-v2/auth/keys
(login + password + companyId, companyId в интерфейсе — Ctrl+Alt+Q). Шаги — в README.
Комментарии к задаче пишутся в её чат: chatId = идентификатор задачи.

Цепочка проект -> доска -> колонка находится по точным title и создаётся, если
чего-то нет. Результат (columnId) кэшируется в памяти и через колбэк
on_column_resolved — в SQLite-мете бота; полная перепроверка/пересоздание
происходят только при неизвестном columnId или ошибке создания задачи.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from typing import Any

import aiohttp

log = logging.getLogger("kapum.samo_helper.yougile")

DEFAULT_API_URL = "https://ru.yougile.com"

TASK_TITLE_MAX_LEN = 200
COLUMN_COLOR = 1  # системный цвет колонки при создании (1..16)

# Сетевой сбой при создании задачи: до 3 ретраев с нарастающей паузой.
# Сумма пауз (3+10+30=43 c) гарантированно не больше минуты.
RETRY_DELAYS_S = (3, 10, 30)


class YouGileError(RuntimeError):
    """Ошибка обращения к YouGile (HTTP, формат ответа)."""


def _is_transient(exc: BaseException) -> bool:
    """Сетевые/таймаут/5xx —上有 смысл повторить; 4xx — нет."""
    if isinstance(exc, (aiohttp.ClientError, asyncio.TimeoutError)):
        return True
    return isinstance(exc, YouGileError) and "HTTP 5" in str(exc)


class YouGileClient:
    def __init__(
        self,
        key: str,
        project_title: str,
        board_title: str,
        column_title: str,
        base_url: str = DEFAULT_API_URL,
    ) -> None:
        self._headers = {"Authorization": f"Bearer {key}"}
        # Допускаем базовый адрес и с суффиксом /api-v2 — пути ниже абсолютные от хоста.
        self._base = base_url.rstrip("/")
        self._base = self._base.removesuffix("/api-v2")
        self._project_title = project_title
        self._board_title = board_title
        self._column_title = column_title
        self._project_id: str | None = None
        self._board_id: str | None = None
        self._column_id: str | None = None
        # Колбэк сохранения column_id (meta БД), ставится рабочим плагином.
        self.on_column_resolved: Callable[[str], None] | None = None

    # --- низкоуровневые вызовы ---

    async def _request(
        self, session: aiohttp.ClientSession, method: str, path: str, **kwargs: Any
    ) -> dict[str, Any]:
        async with session.request(
            method, f"{self._base}{path}", headers=self._headers, **kwargs
        ) as resp:
            body = await resp.text()
            if resp.status not in (200, 201):
                raise YouGileError(f"{method} {path}: HTTP {resp.status}: {body[:200]}")
            return json.loads(body) if body else {}

    async def _list(
        self, session: aiohttp.ClientSession, path: str, **params: Any
    ) -> list[dict[str, Any]]:
        data = await self._request(session, "GET", path, params=params)
        return data.get("content", [])

    async def me(self, session: aiohttp.ClientSession) -> dict[str, Any]:
        """Текущий пользователь (проверка ключа)."""
        return await self._request(session, "GET", "/api-v2/users/me")

    # --- цепочка project -> board -> column ---

    def restore_column(self, column_id: str) -> None:
        """Восстановить кэшированный columnId из мета-БД при старте."""
        self._column_id = column_id
        log.info("YouGile: columnId из кэша БД: %s", column_id)

    def _invalidate(self) -> None:
        self._project_id = self._board_id = self._column_id = None

    async def _ensure_by_title(
        self,
        session: aiohttp.ClientSession,
        list_path: str,
        create_path: str,
        title: str,
        kind: str,
        list_params: dict[str, Any] | None = None,
        create_body: dict[str, Any] | None = None,
    ) -> str:
        items = await self._list(
            session, list_path, title=title, limit=100, **(list_params or {})
        )
        found = next((i for i in items if i.get("title") == title), None)
        if found is None:
            body = {"title": title, **(create_body or {})}
            created = await self._request(session, "POST", create_path, json=body)
            if not created.get("id"):
                raise YouGileError(f"{kind}: не удалось создать {title!r}: {str(created)[:200]}")
            log.info("YouGile: создан %s %r (id=%s)", kind, title, created["id"])
            return str(created["id"])
        return str(found["id"])

    async def ensure_column(self, session: aiohttp.ClientSession) -> str:
        """Колонка для новых задач: из кэша либо пересобрать всю цепочку."""
        if self._column_id is not None:
            return self._column_id
        if self._project_id is None:
            self._project_id = await self._ensure_by_title(
                session, "/api-v2/projects", "/api-v2/projects",
                self._project_title, "проект",
            )
        if self._board_id is None:
            self._board_id = await self._ensure_by_title(
                session, "/api-v2/boards", "/api-v2/boards",
                self._board_title, "доска",
                list_params={"projectId": self._project_id},
                create_body={"projectId": self._project_id},
            )
        self._column_id = await self._ensure_by_title(
            session, "/api-v2/columns", "/api-v2/columns",
            self._column_title, "колонка",
            list_params={"boardId": self._board_id},
            create_body={"boardId": self._board_id, "color": COLUMN_COLOR},
        )
        if self.on_column_resolved is not None:
            self.on_column_resolved(self._column_id)
        return self._column_id

    @property
    def column_id(self) -> str | None:
        return self._column_id

    def ids_for_report(self) -> dict[str, str | None]:
        return {
            "project": self._project_id,
            "board": self._board_id,
            "column": self._column_id,
        }

    # --- задачи ---

    async def _post_task(
        self, session: aiohttp.ClientSession, title: str, description: str, column_id: str
    ) -> dict[str, Any]:
        return await self._request(
            session,
            "POST",
            "/api-v2/tasks",
            json={
                "title": title[:TASK_TITLE_MAX_LEN],
                "description": description,
                "columnId": column_id,
            },
        )

    async def create_task(
        self, session: aiohttp.ClientSession, title: str, description: str = ""
    ) -> dict[str, Any]:
        """Создать задачу.

        Сетевой сбой/5xx — до len(RETRY_DELAYS_S) ретраев с нарастающей паузой
        (суммарно <60 с). 4xx (колонку могли удалить) — пересборка цепочки и
        один повтор.
        """
        column_id = await self.ensure_column(session)
        chain_retry = False
        attempt = 0
        while True:
            try:
                resp = await self._post_task(session, title, description, column_id)
                if not resp.get("id"):
                    raise YouGileError(f"в ответе создания задачи нет id: {str(resp)[:200]}")
                return resp
            except Exception as e:
                if _is_transient(e) and attempt < len(RETRY_DELAYS_S):
                    delay = RETRY_DELAYS_S[attempt]
                    attempt += 1
                    log.warning("YouGile create_task: %s — ретрай %d через %d с",
                                type(e).__name__ or e, attempt, delay)
                    await asyncio.sleep(delay)
                    continue
                if isinstance(e, YouGileError) and not chain_retry and not _is_transient(e):
                    log.warning("YouGile create_task: %s — пересобираю цепочку", e)
                    self._invalidate()
                    column_id = await self.ensure_column(session)
                    chain_retry = True
                    continue
                raise YouGileError(f"не удалось создать задачу: {e}") from e

    async def task_exists(self, session: aiohttp.ClientSession, task_id: str) -> bool:
        """Жива ли карточка (не удалена); false при 404/флаге deleted."""
        try:
            data = await self._request(session, "GET", f"/api-v2/tasks/{task_id}")
        except YouGileError as e:
            if "HTTP 404" in str(e):
                return False
            raise
        return not data.get("deleted", False)

    async def add_comment(
        self, session: aiohttp.ClientSession, task_id: str, text: str
    ) -> None:
        """Сообщение в чат задачи (chatId = id задачи)."""
        await self._request(
            session, "POST", f"/api-v2/chats/{task_id}/messages", json={"text": text}
        )
