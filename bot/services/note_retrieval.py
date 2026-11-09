"""Строгий разбор карточек MCP для чтения найденных заметок без участия модели."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

_MAX_METADATA_CHARS = 256_000
_MAX_NOTES = 3
_UNSAFE_PATH_RE = re.compile(r"[\x00-\x1f\x7f:]")


@dataclass(frozen=True, slots=True)
class NoteReference:
    """Проверенный путь и идентификаторы фрагментов одной карточки выдачи."""

    path: str
    chunk_ids: tuple[int, ...] = ()


def _cards(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        if "path" in value:
            return [value]
        # FastMCP оборачивает типизированный список в structuredContent.result.
        if isinstance(value.get("result"), list):
            return value["result"]
    return []


def note_references(text: str) -> list[NoteReference]:
    """Прочитать только JSON-карточки search_notes/list_notes, не тело заметки.

    Поддерживаются массив, оболочка FastMCP и последовательные JSON content
    blocks. Проза, Markdown, поля summary/related и JSON внутри них никогда
    не становятся аргументами чтения. Путь должен быть относительным .md без
    обхода каталогов; сам сервер дополнительно проверяет границы vault.
    """
    if not text or len(text) > _MAX_METADATA_CHARS:
        return []
    decoder = json.JSONDecoder()
    values: list[Any] = []
    offset = 0
    try:
        while offset < len(text):
            while offset < len(text) and text[offset].isspace():
                offset += 1
            if offset == len(text):
                break
            value, offset = decoder.raw_decode(text, offset)
            values.extend(_cards(value))
    except (ValueError, RecursionError):
        # Не восстанавливаем JSON из произвольной строки/инструкции.
        return []
    references: list[NoteReference] = []
    seen: set[str] = set()
    for card in values:
        if not isinstance(card, dict):
            continue
        path = card.get("path")
        if not isinstance(path, str) or not path or len(path) > 1024:
            continue
        path = path.replace("\\", "/")
        parts = path.split("/")
        if (
            _UNSAFE_PATH_RE.search(path)
            or any(part in {"", ".", ".."} for part in parts)
            or PurePosixPath(path).suffix.lower() != ".md"
            or path in seen
        ):
            continue
        chunks = card.get("chunks")
        chunk_ids = tuple(dict.fromkeys(
            chunk["chunk_id"] for chunk in (chunks if isinstance(chunks, list) else [])
            if isinstance(chunk, dict) and type(chunk.get("chunk_id")) is int
            and chunk["chunk_id"] > 0
        ))[:2]
        references.append(NoteReference(path, chunk_ids))
        seen.add(path)
        if len(references) == _MAX_NOTES:
            break
    return references


def read_failed(text: str) -> bool:
    """Распознать ошибку чтения MCP, в том числе строковый ответ ObsidianBrain."""
    stripped = text.strip()
    if not stripped or stripped == "(empty result)":
        return True
    if stripped.casefold().startswith(("ошибка", "error:")):
        return True
    try:
        value = json.loads(stripped)
    except (ValueError, RecursionError):
        return False
    return isinstance(value, dict) and bool(value.get("error"))
