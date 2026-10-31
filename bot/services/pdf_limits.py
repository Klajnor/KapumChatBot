"""Ограниченная распаковка Flate в pdfminer без изменения глобального zlib."""

from __future__ import annotations

import threading
import zlib
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, NoReturn

MAX_FLATE_STREAM_BYTES = 8 * 1024 * 1024
MAX_FLATE_DOCUMENT_BYTES = 64 * 1024 * 1024


class PdfResourceLimit(ValueError):
    """PDF превысил безопасный бюджет обработки; повторный OCR запрещён."""


@dataclass
class _Budget:
    remaining: int
    stream_limit: int
    rejected: bool = False

    def reject(self) -> NoReturn:
        self.rejected = True
        raise PdfResourceLimit("PDF decompression limit exceeded")


_budget: ContextVar[_Budget | None] = ContextVar("pdf_flate_budget", default=None)
_install_lock = threading.Lock()


class _LimitedInflater:
    def __init__(self, inflater: Any, budget: _Budget) -> None:
        self._inflater = inflater
        self._budget = budget
        self._remaining = budget.stream_limit

    def decompress(self, data: bytes, max_length: int = 0) -> bytes:
        budget = self._budget
        if budget.rejected:
            budget.reject()
        available = min(self._remaining, budget.remaining)
        # max_length=0 у zlib означает НЕОГРАНИЧЕННУЮ распаковку.
        # Один дополнительный байт позволяет отличить точный предел от бомбы.
        limit = min(max_length, available + 1) if max_length else available + 1
        result = self._inflater.decompress(data, limit)
        if len(result) > available:
            budget.reject()
        self._remaining -= len(result)
        budget.remaining -= len(result)
        return result

    @property
    def eof(self) -> bool:
        return self._inflater.eof

    @property
    def unconsumed_tail(self) -> bytes:
        return self._inflater.unconsumed_tail


class _PdfZlib:
    """Адаптер только для pdfminer.pdftypes; бюджеты изолированы по контексту."""

    error = zlib.error

    def decompress(self, data: bytes, wbits: int = zlib.MAX_WBITS, bufsize: int = 16384) -> bytes:
        budget = _budget.get()
        if budget is None:
            return zlib.decompress(data, wbits, bufsize)
        inflater = self.decompressobj(wbits)
        result = inflater.decompress(data)
        # Не используем flush(): его length — размер буфера, а не жёсткий cap.
        if not inflater.eof:
            raise zlib.error("Incomplete PDF Flate stream")
        return result

    def decompressobj(self, *args: Any, **kwargs: Any) -> Any:
        inflater = zlib.decompressobj(*args, **kwargs)
        budget = _budget.get()
        return inflater if budget is None else _LimitedInflater(inflater, budget)


@contextmanager
def pdf_stream_limits() -> Iterator[None]:
    """Ограничить каждый Flate-этап и суммарную распаковку одного PDF.

    Адаптер установлен постоянно только в модуле pdfminer, а не в zlib.
    ContextVar не даёт двум потокам делить бюджет или снимать защиту друг друга.
    Повреждённые потоки используют тот же bounded decompressobj. Sticky-отказ
    сохраняется, даже если библиотека перехватила/обернула исключение.
    """
    from pdfminer import pdftypes

    with _install_lock:
        if not isinstance(pdftypes.zlib, _PdfZlib):
            pdftypes.zlib = _PdfZlib()
    budget = _Budget(MAX_FLATE_DOCUMENT_BYTES, MAX_FLATE_STREAM_BYTES)
    token = _budget.set(budget)
    try:
        yield
    finally:
        _budget.reset(token)
        if budget.rejected:
            raise PdfResourceLimit("PDF decompression limit exceeded")
