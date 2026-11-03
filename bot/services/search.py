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
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

import aiohttp

from bot.config import config
from bot.services.ai_client import ToolCall
from bot.services.documents import fetch_url_text
from bot.services.operations import current_operation, stage
from bot.services.untrusted import wrap_untrusted

logger = logging.getLogger(__name__)

# Providers with a built-in implementation.
PROVIDERS = ("tavily", "brave")

# Snippets are trimmed so one search cannot flood the model's context.
# A search provider response is JSON with short snippets; anything larger is
# a broken endpoint (or a proxy page), not a result set.
_MAX_SEARCH_BODY = 2 * 1024 * 1024

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


class SearchResults(list[SearchResult]):
    """Search hits plus a short failure code when the provider did not answer.

    Behaves exactly like a plain list, so callers that only need the hits keep
    working (and ``== []`` still holds on failure). ``error`` is a log-safe
    category such as ``timeout`` or ``http_401`` — never the provider body or
    the user's query.
    """

    def __init__(
        self, results: Iterable[SearchResult] = (), *, error: str = ""
    ) -> None:
        super().__init__(results)
        self.error = error

    @property
    def failed(self) -> bool:
        """True when the search provider could not be queried."""
        return bool(self.error)


@dataclass(slots=True)
class _RequestResult:
    """One search API response: parsed JSON or a log-safe failure code."""

    data: Any = None
    error: str = ""


def _operation_context() -> str:
    """Identify the request in the log without any user content."""
    operation = current_operation()
    if operation is None:
        return "op=-"
    return (
        f"op={operation.id} kind={operation.kind!r} "
        f"chat={operation.chat_id} user={operation.user_id} "
        f"stage={operation.stage!r}"
    )


def _response_parts(value: Any) -> tuple[Any, str]:
    """Split a response object into data and a failure code.

    ``None`` is accepted as a failure for backward compatibility with tests and
    older callers that patched :func:`_request_json` directly.
    """
    if isinstance(value, _RequestResult):
        return value.data, value.error
    if value is None:
        return None, "request_failed"
    return value, ""


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
) -> _RequestResult:
    """One JSON request to a search API; never raises."""
    host = urlsplit(url).netloc
    timeout = aiohttp.ClientTimeout(total=config.search_timeout)
    started = time.monotonic()
    try:
        async with (
            aiohttp.ClientSession(timeout=timeout) as session,
            session.request(
                method, url, json=json_payload, params=params, headers=headers
            ) as resp,
        ):
            body_bytes = await resp.content.read(_MAX_SEARCH_BODY + 1)
            elapsed = time.monotonic() - started
            if len(body_bytes) > _MAX_SEARCH_BODY:
                logger.warning(
                    "Search API %s returned an oversized body after %.2fs [%s]",
                    host, elapsed, _operation_context(),
                )
                return _RequestResult(error="oversized_response")
            body = body_bytes.decode("utf-8", "replace")
            if resp.status != 200:
                logger.warning(
                    "Search API %s returned HTTP %s after %.2fs [%s]: %s",
                    host, resp.status, elapsed, _operation_context(), body[:200],
                )
                return _RequestResult(error=f"http_{resp.status}")
            try:
                return _RequestResult(data=json.loads(body))
            except json.JSONDecodeError:
                logger.warning(
                    "Search API %s returned invalid JSON after %.2fs [%s]",
                    host, elapsed, _operation_context(),
                )
                return _RequestResult(error="invalid_json")
    except (TimeoutError, aiohttp.ClientError, OSError) as exc:
        reason = "timeout" if isinstance(exc, TimeoutError) else "network"
        logger.warning(
            "Search API request to %s failed after %.2fs [%s]: %s: %s",
            host, time.monotonic() - started, _operation_context(),
            type(exc).__name__, exc,
        )
        return _RequestResult(error=reason)


# --- Providers ------------------------------------------------------------------


async def _search_tavily(query: str, max_results: int) -> SearchResults:
    """Tavily Search API (POST /search, Bearer auth)."""
    url = config.search_api_url or "https://api.tavily.com/search"
    data, error = _response_parts(await _request_json(
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
    ))
    if error:
        return SearchResults(error=error)
    if not isinstance(data, dict):
        logger.warning(
            "Search API tavily returned an unexpected payload [%s]",
            _operation_context(),
        )
        return SearchResults(error="unexpected_response")
    results: list[SearchResult] = []
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
    return SearchResults(results)


async def _search_brave(query: str, max_results: int) -> SearchResults:
    """Brave Search API (GET /res/v1/web/search, subscription token header)."""
    url = config.search_api_url or "https://api.search.brave.com/res/v1/web/search"
    data, error = _response_parts(await _request_json(
        url,
        {
            "X-Subscription-Token": config.search_api_key,
            "Accept": "application/json",
        },
        method="GET",
        params={"q": query, "count": max_results},
    ))
    if error:
        return SearchResults(error=error)
    if not isinstance(data, dict):
        logger.warning(
            "Search API brave returned an unexpected payload [%s]",
            _operation_context(),
        )
        return SearchResults(error="unexpected_response")
    results: list[SearchResult] = []
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
    return SearchResults(results)


# --- Entry point ----------------------------------------------------------------


async def _read_result_pages(results: list[SearchResult]) -> None:
    """Прочитать страницы без обхода ссылок; сбой одной не теряет выдачу."""
    semaphore = asyncio.Semaphore(_PAGE_CONCURRENCY)
    context = _operation_context()

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
            except Exception as exc:  # noqa: BLE001 — недоступный источник не ломает ответ
                logger.warning(
                    "Search result page %s could not be read [%s]: %s",
                    urlsplit(item.url).netloc or "?",
                    context,
                    type(exc).__name__,
                )

    await asyncio.gather(*(read(item) for item in results))


@stage("Searching web / reading results")
async def search_web(query: str, max_results: int | None = None) -> SearchResults:
    """Run one web search; never raises.

    Returns a :class:`SearchResults` list. On a provider error the list is
    empty and ``results.failed`` is true with a short ``results.error`` code,
    so the caller can tell a broken search from a search with zero hits.
    """
    # The query comes from the model's tool call: cap it so a degenerate
    # answer cannot turn into an oversized request URL.
    query = (query or "").strip()[:512]
    if not query:
        return SearchResults()
    limit = (
        max_results
        if isinstance(max_results, int) and not isinstance(max_results, bool)
        else config.search_max_results
    )
    limit = max(1, min(limit, 10))
    provider = config.search_provider.strip().lower()
    try:
        if provider == "tavily":
            raw = await _search_tavily(query, limit)
        elif provider == "brave":
            raw = await _search_brave(query, limit)
        else:
            logger.warning("Unknown SEARCH_PROVIDER %r; search skipped", provider)
            return SearchResults(error="unknown_provider")
    except Exception:  # search must never break an answer
        logger.exception(
            "Web search failed (%s) [%s]", provider, _operation_context()
        )
        return SearchResults(error="internal_error")
    # Provider functions may be patched with a plain list in tests.
    results = raw if isinstance(raw, SearchResults) else SearchResults(raw or ())
    results = SearchResults(results[:limit], error=results.error)
    await _read_result_pages(results)
    # The query itself is user/model content: keep it out of the logs.
    if results.failed:
        logger.warning(
            "Web search (%s) failed: %s [%s]",
            provider, results.error, _operation_context(),
        )
    else:
        logger.info("Web search (%s) -> %d result(s)", provider, len(results))
    return results


def format_search_results(
    query: str, results: list[SearchResult], *, failed: bool = False
) -> str:
    """Tool-message body: numbered results the model cites as [n].

    The results contain third-party page text and snippets, so they are wrapped as
    untrusted content (``bot/services/untrusted.py``): a page that tries to
    steer the model stays data. The closing instruction stays outside the
    tags — it is ours, not the web's.
    """
    if failed:
        return (
            f"Веб-поиск по запросу {query!r} не удался по технической причине "
            "(провайдер поиска недоступен). Это не означает, что результатов "
            "нет. Ответь из собственных знаний и явно скажи пользователю, что "
            "поиск в интернете сейчас недоступен."
        )
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


def remove_search_hint(messages: list[dict[str, Any]]) -> None:
    """Remove the search guidance after the tool failed, in place.

    The next request must not tell the model that a working search tool exists
    when the tool is no longer offered.
    """
    for entry in messages:
        if entry.get("role") == "system":
            content = entry.get("content")
            if isinstance(content, str) and SEARCH_SYSTEM_HINT in content:
                entry["content"] = content.replace(
                    f"\n\n{SEARCH_SYSTEM_HINT}", ""
                ).replace(SEARCH_SYSTEM_HINT, "")
            return
