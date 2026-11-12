"""Текст цитат Telegram, включая TextQuote.rich_texts и rich-сообщения."""

from __future__ import annotations

from typing import Any

MAX_QUOTE_CHARS = 4000
# Rich-сообщение — это сам запрос пользователя, а не короткая цитата: не
# обрезаем его на 4000 символах, как контекст reply.
MAX_RICH_MESSAGE_CHARS = 32000
DEFAULT_QUOTE_QUESTION = "Поясни процитированный фрагмент."


def _field(value: Any, name: str) -> Any:
    return value.get(name) if isinstance(value, dict) else getattr(value, name, None)


def _rich_plain(value: Any, limit: int) -> str:
    """Ограниченно прочитать только текстовые поля rich-дерева, не repr объекта."""
    parts: list[str] = []
    remaining = limit + 1
    nodes = 2000

    def walk(item: Any, depth: int = 0, joiner: str = "") -> None:
        nonlocal remaining, nodes
        if remaining <= 0 or nodes <= 0 or depth > 20:
            return
        nodes -= 1
        if isinstance(item, str):
            piece = item[:remaining]
            parts.append(piece)
            remaining -= len(piece)
        elif isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                if remaining <= 0 or nodes <= 0:
                    break
                if index and joiner:
                    walk(joiner, depth + 1)
                walk(child, depth + 1)
        elif item is not None:
            # В историю попадает видимый текст, не скрытые thinking-блоки,
            # file_id, URL медиаресурсов или произвольные дополнительные поля.
            kind = _field(item, "type")
            if getattr(kind, "value", kind) == "thinking":
                return
            for name in (
                "title",
                "text",
                "expression",
                "alternative_text",
                "summary",
                "credit",
                "caption",
                "label",
                "value",
                "button",
            ):
                child = _field(item, name)
                if child is not None:
                    walk(child, depth + 1)
            for name in ("blocks", "items", "rows", "cells", "header", "buttons"):
                children = _field(item, name)
                if isinstance(children, (list, tuple)):
                    # Ячейки одной строки таблицы разделяем, а не склеиваем
                    # вроде «AB12»; строки и блоки идут с новой строки.
                    inner_joiner = " | " if name in {"cells", "rows", "header"} else ""
                    for child in children:
                        if remaining <= 0 or nodes <= 0:
                            break
                        if parts and not parts[-1].endswith("\n"):
                            walk("\n", depth + 1)
                        walk(child, depth + 1, inner_joiner)

    walk(value)
    text = "".join(parts)
    return text[:limit] + "…" if len(text) > limit else text


def has_quote(message: Any) -> bool:
    """Пользователь выбрал фрагмент, даже если SDK не умеет его rich-формат."""
    return _field(message, "quote") is not None


def selected_quote_text(message: Any, *, limit: int = MAX_QUOTE_CHARS) -> str:
    """Взять переданный Telegram фрагмент, не нарезая исходник по UTF-16 position.

    В новых Bot API обычный text у rich-цитаты может быть пустым. Aiogram
    сохраняет пока не объявленные rich_texts в extra-полях TelegramObject.
    """
    quote = _field(message, "quote")
    text = _field(quote, "text")
    if isinstance(text, str) and text.strip():
        return text[:limit] + "…" if len(text) > limit else text
    return _rich_plain(_field(quote, "rich_texts"), limit)


def reply_message_text(message: Any, *, limit: int = MAX_QUOTE_CHARS) -> str:
    """Обычный текст/подпись либо видимые текстовые блоки rich-сообщения."""
    rich = _field(message, "rich_message")
    if rich is not None:
        text = _rich_plain(rich, limit)
        if text.strip():
            return text
    for name in ("text", "caption"):
        text = _field(message, name)
        if isinstance(text, str) and text.strip():
            return text[:limit] + "…" if len(text) > limit else text
    return ""


def message_text(
    message: Any, *, limit: int = MAX_RICH_MESSAGE_CHARS
) -> str:
    """Видимый текст самого сообщения: text/caption или rich_message.

    В отличие от ``reply_message_text`` (контекст цитаты/reply), здесь
    допустим полный rich-запрос до 32 000 символов: это не чужой фрагмент,
    а сообщение, которое пользователь попросил разобрать.
    """
    text = reply_message_text(message, limit=limit)
    # Rich-дерево из пустых блоков/thinking может дать только переводы строк.
    return text if text.strip() else ""
