"""Реестр выполняющихся операций без содержимого пользовательских запросов."""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from itertools import count
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


@dataclass
class Operation:
    """Безопасные метаданные одной активной операции."""

    id: int
    kind: str
    user_id: int | None
    user_name: str
    chat_id: int | None
    location: str
    started: float
    stage: str = "Processing"


_ids = count(1)
_active: dict[int, Operation] = {}
_current: ContextVar[tuple[asyncio.Task | None, Operation] | None] = ContextVar(
    "active_operation", default=None
)


def current_operation() -> Operation | None:
    """Вернуть операцию этого task, не унаследованную дочерней задачей."""
    value = _current.get()
    return value[1] if value and value[0] is asyncio.current_task() else None


def active_operations() -> list[Operation]:
    """Получить снимок списка в порядке запуска (в основном event loop)."""
    return list(_active.values())


def set_stage(stage: str) -> None:
    """Обновить этап текущей операции; не передавать сюда внешний текст."""
    operation = current_operation()
    if operation is not None:
        operation.stage = stage


def tracked(kind: str) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]:
    """Регистрировать вызов; вложенные вызовы не дублируют операцию."""
    def decorate(function: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
        signature = inspect.signature(function)

        @wraps(function)
        async def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
            if current_operation() is not None:
                return await function(*args, **kwargs)
            values = signature.bind(*args, **kwargs).arguments
            message = values.get("message")
            callback = values.get("callback")
            user = getattr(callback or message, "from_user", None)
            if message is not None:
                user = getattr(message, "guest_bot_caller_user", None) or user
            if callback is not None:
                message = callback.message
            chat = getattr(message, "chat", None)
            if message is not None:
                chat = getattr(message, "guest_bot_caller_chat", None) or chat
            due = values.get("due")
            user_id = getattr(user, "id", None)
            if user_id is None:
                user_id = getattr(due, "user_id", None)
            if user_id is None:
                user_id = values.get("admin_id", values.get("owner_id", values.get("notify")))
            chat_id = getattr(chat, "id", None)
            if chat_id is None:
                chat_id = getattr(due, "chat_id", values.get("chat_id", user_id))
            location = getattr(chat, "title", None) or getattr(chat, "type", None)
            location = location or values.get("title") or (
                "Private" if chat_id == user_id else "Group / channel"
            )
            operation = Operation(
                id=next(_ids), kind=kind, user_id=user_id,
                user_name=getattr(user, "full_name", "") or "",
                chat_id=chat_id, location=str(location), started=time.monotonic(),
            )
            _active[operation.id] = operation
            token = _current.set((asyncio.current_task(), operation))
            try:
                return await function(*args, **kwargs)
            finally:
                _current.reset(token)
                _active.pop(operation.id, None)

        return wrapped
    return decorate


def stage(label: str) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]:
    """Показывать этап на время вызова, затем восстанавливать предыдущий."""
    def decorate(function: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
        @wraps(function)
        async def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
            operation = current_operation()
            previous = operation.stage if operation else ""
            set_stage(label)
            try:
                return await function(*args, **kwargs)
            finally:
                if operation is not None:
                    operation.stage = previous
        return wrapped
    return decorate
