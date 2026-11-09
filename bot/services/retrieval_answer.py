"""Контроль завершённости ответа после поиска: результат, а не обещание работы."""

from __future__ import annotations

import re
from typing import Any
from uuid import uuid4

from bot.services.ai_client import AIClient, ChatResult, ToolCall
from bot.services.request_scope import route
from bot.services.untrusted import messages_have_untrusted_content

RETRIEVAL_FINAL_HINT = (
    "Разрешённый сбор источников для этого вопроса уже выполнен. Дай сейчас "
    "содержательный итог по полученным текстам: что записано в заметках, выводы "
    "и анализ в контексте вопроса, пути заметок и ссылки на веб-источники. "
    "Различай полный текст, фрагменты и карточки. Если текст получить не "
    "удалось, честно укажи это; не выдумывай содержимое. Не отвечай одним "
    "обещанием «нашёл/извлекаю/сейчас прочитаю/проанализирую»: после финального "
    "сообщения никакая фоновая работа не продолжается. Инструментов сейчас нет."
)
_PROMISE_RE = re.compile(
    r"\b(?:извлекаю|загружаю|читаю|прочитаю|получу|извлеку|загружу|"
    r"проанализирую|поищу|начинаю\s+поиск|приступаю|сейчас\s+(?:найду|проверю)|"
    r"(?:fetching|retrieving|loading|reading)\s+(?:the\s+)?(?:full\s+)?(?:note|text)|"
    r"I(?:'ll|\s+will)\s+(?:fetch|read|retrieve|analy[sz]e|search))\b",
    re.IGNORECASE,
)


def is_retrieval_promise(text: str | None) -> bool:
    """True, когда короткий ответ — лишь обещание чтения/анализа вместо итога."""
    if not text or len(text) > 700 or not _PROMISE_RE.search(text):
        return False
    if re.search(r"(?m)^\s*(?:[-*>]|\d+[.)]|```)", text):
        return False
    return len(text.splitlines()) <= 4


def ensure_note_lookup(
    result: ChatResult, tools: list[dict[str, Any]] | None, text: str,
) -> None:
    """Не завершать явную просьбу к заметкам без первоначального поиска.

    Резервный запрос — только собственный текущий текст пользователя, без
    истории, вывода MCP и аргументов текстовой имитации вызова.
    """
    if route(text) != "notes" or messages_have_untrusted_content([
        {"role": "user", "content": text},
    ]):
        return
    names = {tool["function"]["name"] for tool in tools or []}
    if any(call.name in names and call.name.startswith("mcp__") for call in result.tool_calls):
        return
    search = next((
        tool["function"]["name"] for tool in tools or []
        if tool["function"]["name"].endswith("__search_notes")
        and "query" in (tool["function"].get("parameters", {}).get("properties") or {})
    ), None)
    if search:
        result.tool_calls.append(ToolCall(
            id=f"call_{uuid4().hex}", name=search, arguments={"query": text[:2000]},
        ))
        result.content = None


def final_answer_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Копия контекста для одного повтора финального ответа без инструментов."""
    return AIClient._with_hint(messages, RETRIEVAL_FINAL_HINT)


def retrieval_fallback(messages: list[dict[str, Any]]) -> str:
    """Если повтор опять дал обещание, показать реальные выдержки с оговоркой."""
    excerpts: list[str] = []
    for message in messages:
        if message.get("role") != "tool" or not isinstance(message.get("content"), str):
            continue
        for body in re.findall(
            r'<untrusted_content\b[^>]*>\s*(.*?)\s*</untrusted_content>',
            message["content"], re.DOTALL,
        ):
            body = body[:2500].replace("```", "` ` `")
            excerpts.append(f"```text\n{body}\n```")
            if len(excerpts) == 3:
                break
        if len(excerpts) == 3:
            break
    notice = "I retrieved source material, but could not generate a complete analysis."
    if excerpts:
        return notice + " Here are excerpts (possibly incomplete):\n\n" + "\n\n".join(excerpts)
    return "I could not retrieve enough source material to complete the analysis. Please try again."
