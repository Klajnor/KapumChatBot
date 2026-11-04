"""Ограниченное чтение HTTP-тела с учётом частичных сетевых чтений."""

from __future__ import annotations

from typing import Any


async def read_bounded(stream: Any, limit: int) -> bytes:
    """Прочитать до EOF или limit + 1 байт для обнаружения превышения."""
    data = bytearray()
    while len(data) <= limit:
        chunk = await stream.read(min(64 * 1024, limit + 1 - len(data)))
        if not chunk:
            break
        data.extend(chunk)
    return bytes(data)
