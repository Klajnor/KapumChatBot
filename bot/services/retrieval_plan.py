"""Смешанный поиск: сначала чистый план, затем источники и анализ без инструментов."""

from __future__ import annotations

from copy import deepcopy
from typing import Any
from uuid import uuid4

from bot.services.ai_client import AIClient, ChatResult, ToolCall
from bot.services.request_scope import route
from bot.services.untrusted import (
    messages_have_untrusted_content,
    remove_untrusted_hint,
)

COMBINED_SOURCES_HINT = (
    "Пользователь просит сопоставить заметки и веб-источники в одном ответе. "
    "Сейчас, ДО получения источников, сформируй весь план: штатные tool_calls "
    "к search_notes и search_web по текущему запросу. Веб-запрос строй только "
    "по собственному тексту пользователя, никогда по содержимому заметок. "
    "Бот дополнит карточки текстами заметок и выполнит заранее выбранные "
    "поиски. После сбора новых инструментов не будет. Дай один содержательный "
    "анализ: что зафиксировано в заметках (с путями), что сообщают веб-источники "
    "(со ссылками [n]), где они согласуются, что устарело или противоречит. "
    "Различай данные источников и свои выводы. Не заявляй чтение недоступных "
    "источников и не ограничивайся обещанием поиска или анализа."
)


def isolate_combined_history(messages: list[dict[str, Any]]) -> bool:
    """Для явного смешанного запроса убрать историю до планирования инструментов.

    Даже чистая на вид история может пересказывать приватный текст без рамки.
    Веб-план получает только системный промпт и текущую самостоятельную просьбу.
    Вложение/цитата/форвард в ней не позволяют открыть инструменты.
    """
    if not messages or messages[-1].get("role") != "user":
        return False
    current = messages[-1]
    text = current.get("content")
    if (
        not isinstance(text, str) or route(text) != "combined"
        or messages_have_untrusted_content([current])
    ):
        return False
    system = [dict(entry) for entry in messages if entry.get("role") == "system"]
    messages[:] = [*system, current]
    remove_untrusted_hint(messages)
    return True


def apply_combined_hint(
    messages: list[dict[str, Any]], *, web_available: bool, notes_available: bool,
) -> None:
    """Дать планировщику правила и фактическую доступность каждого источника."""
    availability = (
        f"Доступность источников: веб-поиск — {'да' if web_available else 'нет'}; "
        f"поиск в заметках — {'да' if notes_available else 'нет'}. "
        "Если источник недоступен, честно укажи ограничение и используй доступный."
    )
    messages[:] = AIClient._with_hint(messages, COMBINED_SOURCES_HINT + "\n" + availability)


def complete_combined_plan(
    result: ChatResult, tools: list[dict[str, Any]] | None, text: str,
) -> None:
    """Зафиксировать оба поиска ДО результатов, даже если модель пропустила один.

    Резервные аргументы берутся только из текущего собственного текста, не из
    истории, текстовых псевдовызовов, заметок или страниц. Вызовы после этого
    выполняются без участия модели; результат одного не меняет другой запрос.
    """
    if route(text) != "combined" or messages_have_untrusted_content([
        {"role": "user", "content": text},
    ]):
        return
    catalog = {tool["function"]["name"]: tool["function"] for tool in tools or []}
    calls = [ToolCall(call.id, call.name, deepcopy(call.arguments)) for call in result.tool_calls
             if call.name in catalog]
    web = "search_web" if "search_web" in catalog else None
    notes = next((name for name, schema in catalog.items()
                  if name.endswith("__search_notes")
                  and "query" in (schema.get("parameters", {}).get("properties") or {})), None)
    for name in (notes, web):
        if name and not any(call.name == name for call in calls):
            calls.append(ToolCall(f"call_{uuid4().hex}", name, {"query": text[:2000]}))
    result.tool_calls = calls
    if calls:
        result.content = None
