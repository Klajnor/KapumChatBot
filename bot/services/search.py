"""Web search for RAG-style answers.

The chat model receives a ``search_web`` tool (see :data:`SEARCH_TOOL`) and
calls it when a question needs up-to-date information. The handler executes
the call against the configured provider, feeds the formatted results back
as a ``tool`` message and lets the model compose a cited answer
(``bot/handlers/chat.py`` runs the multi-round loop).

Supported providers (``SEARCH_PROVIDER``):

* ``tavily`` — LLM-oriented search API (https://tavily.com, ``tvly-…`` keys);
* ``brave``  — Brave Search API (``BSA…`` subscription tokens).

Both speak plain HTTPS through aiohttp and need ``SEARCH_API_KEY``. Without a
key the tool is simply not offered to the model and the bot behaves exactly
as before. Первые результаты читаются через SSRF-защищённый загрузчик
``bot/services/documents.py`` перед передачей модели для общего саммари.
"""

from __future__ import annotations

import asyncio
import html
import json
import logging
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

import aiohttp

from bot.config import config
from bot.services.ai_client import ToolCall
from bot.services.documents import fetch_url_text
from bot.services.untrusted import wrap_untrusted

logger = logging.getLogger(__name__)

# Providers with a built-in implementation.
PROVIDERS = ("tavily", "brave")

# Snippets are trimmed so one search cannot flood the model's context.
_SNIPPET_CHARS = 700

# Ограничиваем время, параллелизм и объём контекста глубокого чтения.
_PAGE_CHARS = 6000
_PAGE_TIMEOUT = 20
_PAGE_CONCURRENCY = 4

# Brave wraps matched words in <strong>…</strong>; other providers escape
# entities. Both are noise (and tags read like markup) inside a tool message.
_TAGS = re.compile(r"<[^>]{1,64}>")


def _clean(text: Any) -> str:
    """Plain text from a provider's HTML-ish field."""
    value = html.unescape(str(text or ""))
    return " ".join(_TAGS.sub(" ", value).split())


# Tool exposed to the chat model for live web search.
SEARCH_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "search_web",
        "description": (
            "Search the web for up-to-date information. Use this when the "
            "user asks about recent events, fresh releases, prices, weather, "
            "sports results, documentation changes or anything else that may "
            "have changed after your training cutoff — or when the user "
            "explicitly asks to search / find / look something up online. "
            "Reads the first 10 result pages by default for a cited summary."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A concise, specific search query.",
                },
            },
            "required": ["query"],
        },
    },
}

# Appended to the system prompt while the search tool is offered.
SEARCH_SYSTEM_HINT = (
    "Тебе доступен инструмент search_web, который ищет информацию в интернете "
    "в реальном времени. Используй его, когда ответ может зависеть от "
    "актуальных данных, вместо того чтобы догадываться или отказываться. "
    "Строй ответ на полученных результатах, цитируй фактически использованные "
    "источники маркерами [n] по номерам результатов и отвечай на русском "
    "языке, если пользователь явно не попросил другой язык."
    " Поиск читает страницы выдачи: дай общее саммари по прочитанным источникам, "
    "выдели основные выводы и противоречия. Не выдавай сниппет за прочитанную "
    "страницу; сообщи о неполном охвате, если часть страниц недоступна."
)


@dataclass(slots=True)
class SearchResult:
    """One web search hit."""

    title: str
    url: str
    snippet: str
    page_text: str = ""


def search_configured() -> bool:
    """True when the ``search_web`` tool should be offered to the model."""
    if not config.enable_web_search or not config.search_api_key:
        return False
    return config.search_provider.strip().lower() in PROVIDERS


# --- HTTP helpers (keys are never logged) --------------------------------------


async def _request_json(
    url: str,
    headers: dict[str, str],
    *,
    method: str = "POST",
    json_payload: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
) -> Any | None:
    """One JSON request to a search API; None on any failure."""
    host = urlsplit(url).netloc
    timeout = aiohttp.ClientTimeout(total=config.search_timeout)
    try:
        async with (
            aiohttp.ClientSession(timeout=timeout) as session,
            session.request(
                method, url, json=json_payload, params=params, headers=headers
            ) as resp,
        ):
            body = await resp.text()
            if resp.status != 200:
                logger.warning(
                    "Search API %s returned HTTP %s: %s",
                    host, resp.status, body[:200],
                )
                return None
            return json.loads(body)
    except (TimeoutError, aiohttp.ClientError, OSError) as exc:
        logger.warning("Search API request to %s failed: %s", host, exc)
        return None
    except json.JSONDecodeError:
        logger.warning("Search API %s returned invalid JSON", host)
        return None


# --- Providers ------------------------------------------------------------------


async def _search_tavily(query: str, max_results: int) -> list[SearchResult]:
    """Tavily Search API (POST /search, Bearer auth)."""
    url = config.search_api_url or "https://api.tavily.com/search"
    data = await _request_json(
        url,
        {
            "Authorization": f"Bearer {config.search_api_key}",
            "Content-Type": "application/json",
        },
        json_payload={
            "query": query,
            "max_results": max_results,
            "search_depth": "basic",
            "include_answer": False,
        },
    )
    if not isinstance(data, dict):
        return []
    results = []
    for item in data.get("results") or []:
        if not isinstance(item, dict) or not item.get("url"):
            continue
        results.append(
            SearchResult(
                title=_clean(item.get("title")) or str(item["url"]),
                url=str(item["url"]),
                snippet=_clean(item.get("content"))[:_SNIPPET_CHARS],
            )
        )
        if len(results) >= max_results:
            break
    return results


async def _search_brave(query: str, max_results: int) -> list[SearchResult]:
    """Brave Search API (GET /res/v1/web/search, subscription token header)."""
    url = config.search_api_url or "https://api.search.brave.com/res/v1/web/search"
    data = await _request_json(
        url,
        {
            "X-Subscription-Token": config.search_api_key,
            "Accept": "application/json",
        },
        method="GET",
        params={"q": query, "count": max_results},
    )
    if not isinstance(data, dict):
        return []
    results = []
    web = data.get("web") or {}
    for item in web.get("results") or []:
        if not isinstance(item, dict) or not item.get("url"):
            continue
        results.append(
            SearchResult(
                title=_clean(item.get("title")) or str(item["url"]),
                url=str(item["url"]),
                snippet=_clean(item.get("description"))[:_SNIPPET_CHARS],
            )
        )
        if len(results) >= max_results:
            break
    return results


# --- Entry point ----------------------------------------------------------------


async def _read_result_pages(results: list[SearchResult]) -> None:
    """Прочитать страницы без обхода ссылок; сбой одной не теряет выдачу."""
    semaphore = asyncio.Semaphore(_PAGE_CONCURRENCY)

    async def read(item: SearchResult) -> None:
        async with semaphore:
            try:
                async with asyncio.timeout(_PAGE_TIMEOUT):
                    page = await fetch_url_text(item.url)
                if page and page.text.strip():
                    text = page.text.strip()
                    item.page_text = text[:_PAGE_CHARS]
                    if len(text) > _PAGE_CHARS:
                        item.page_text += "\n[Текст страницы сокращён]"
            except Exception:  # noqa: BLE001 — недоступный источник не ломает ответ
                logger.warning("Search result page could not be read")

    await asyncio.gather(*(read(item) for item in results))


async def search_web(query: str, max_results: int | None = None) -> list[SearchResult]:
    """Run one web search; returns [] on any failure (never raises)."""
    # The query comes from the model's tool call: cap it so a degenerate
    # answer cannot turn into an oversized request URL.
    query = (query or "").strip()[:512]
    if not query:
        return []
    limit = (
        max_results
        if isinstance(max_results, int) and not isinstance(max_results, bool)
        else config.search_max_results
    )
    limit = max(1, min(limit, 10))
    provider = config.search_provider.strip().lower()
    try:
        if provider == "tavily":
            results = await _search_tavily(query, limit)
        elif provider == "brave":
            results = await _search_brave(query, limit)
        else:
            logger.warning("Unknown SEARCH_PROVIDER %r; search skipped", provider)
            return []
    except Exception:  # search must never break an answer
        logger.exception("Web search failed (%s)", provider)
        return []
    results = results[:limit]
    await _read_result_pages(results)
    # The query itself is user/model content: keep it out of the logs.
    logger.info("Web search (%s) -> %d result(s)", provider, len(results))
    return results


def format_search_results(query: str, results: list[SearchResult]) -> str:
    """Tool-message body: numbered results the model cites as [n].

    The results contain third-party page text and snippets, so they are wrapped as
    untrusted content (``bot/services/untrusted.py``): a page that tries to
    steer the model stays data. The closing instruction stays outside the
    tags — it is ours, not the web's.
    """
    if not results:
        return (
            f"Живой поиск по запросу {query!r} не дал результатов. Ответь из "
            "собственных знаний и упомяни, что поиск в интернете ничего не "
            "нашёл."
        )
    blocks = [f"Результаты веб-поиска по запросу {query!r}:"]
    for index, item in enumerate(results, start=1):
        if item.page_text:
            content = f"Прочитанный текст страницы:\n{item.page_text}"
        else:
            content = (
                "Страницу прочитать не удалось; доступен только сниппет поиска:\n"
                f"{item.snippet}"
            )
        blocks.append(f"[{index}] {item.title}\n{content}\nURL: {item.url}")
    wrapped = wrap_untrusted("search", "\n\n".join(blocks))
    return (
        f"{wrapped}\n\n"
        "Ответь пользователю: составь общее саммари по прочитанным страницам "
        "в контексте вопроса, выдели выводы и расхождения; "
        "цитируй использованные источники маркерами [n] и укажи их URL. "
        "Если часть страниц не прочитана, явно сообщи об ограничении охвата. "
        "Не утверждай, что прочитал недоступные страницы."
    )


# --- Tool-exchange message builders (OpenAI chat format) -------------------------


def build_assistant_tool_message(
    content: str | None, calls: list[ToolCall]
) -> dict[str, Any]:
    """The assistant turn carrying ``tool_calls``, as the API expects back.

    Every tool_call of an assistant turn must be answered with a ``tool``
    message in the next request, or the API rejects the conversation.
    """
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(call.arguments, ensure_ascii=False),
                },
            }
            for call in calls
        ],
    }


def build_tool_response(call_id: str, content: str) -> dict[str, Any]:
    """One ``role: tool`` answer for a tool_call id."""
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def apply_search_hint(messages: list[dict[str, Any]]) -> None:
    """Append the search guidance to the system prompt, in place, once."""
    for entry in messages:
        if entry.get("role") == "system":
            content = entry.get("content")
            if isinstance(content, str) and SEARCH_SYSTEM_HINT not in content:
                entry["content"] = f"{content}\n\n{SEARCH_SYSTEM_HINT}"
            return
