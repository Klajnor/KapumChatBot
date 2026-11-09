"""Web search for RAG-style answers.

The chat model receives a ``search_web`` tool (see :data:`SEARCH_TOOL`) and
calls it when a question needs up-to-date information. The handler executes
the call against the configured provider, feeds the formatted results back
as a ``tool`` message and lets the model compose a cited answer
(``bot/handlers/chat.py`` runs the multi-round loop).

Supported providers (``SEARCH_PROVIDER``):

* ``duckduckgo`` (default) — DuckDuckGo through the optional ``ddgs`` package;
  no API key and no account, so search works right after installation;
* ``tavily`` — LLM-oriented search API (https://tavily.com, ``tvly-…`` keys);
* ``brave``  — Brave Search API (``BSA…`` subscription tokens);
* ``exa``    — Exa Search API (https://exa.ai, keys from the dashboard).

Tavily/Brave/Exa speak plain HTTPS through aiohttp and need
``SEARCH_API_KEY``; without a key they are not offered to the model and the
bot behaves exactly as before. DuckDuckGo is a scraping provider — ``ddgs``
(or a proxy) may be blocked by the site, so successes and failures are logged
with an operation id.
Первые результаты читаются через SSRF-защищённый загрузчик
``bot/services/documents.py`` перед передачей модели для общего саммари.
"""

from __future__ import annotations

import asyncio
import html
import importlib.util
import json
import logging
import os
import re
import time
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from typing import Any
from urllib.parse import parse_qs, urlsplit

import aiohttp

from bot.config import config
from bot.services.ai_client import ToolCall
from bot.services.documents import fetch_url_text
from bot.services.http_body import read_bounded
from bot.services.operations import current_operation, stage
from bot.services.request_scope import has_notes_target, is_fresh_request
from bot.services.untrusted import messages_have_untrusted_content, wrap_untrusted

logger = logging.getLogger(__name__)

# Providers with a built-in implementation.
PROVIDERS = ("duckduckgo", "tavily", "brave", "exa")

# Short spellings accepted in SEARCH_PROVIDER.
_PROVIDER_ALIASES = {"ddg": "duckduckgo", "duck": "duckduckgo"}

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


def _safe_host(url: str) -> str:
    """Host for logs, without URL userinfo or query parameters."""
    try:
        parts = urlsplit(url)
        host = parts.hostname or "?"
        port = parts.port
    except ValueError:
        return "?"
    return f"{host}:{port}" if port else host


_PROXY_ENV_NAMES = (
    "HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy",
    "ALL_PROXY", "all_proxy",
)


def _redact_secrets(text: str, headers: dict[str, str]) -> str:
    """Remove search credentials from a body/exception before logging.

    A broken custom proxy or a debugging endpoint can reflect request headers.
    The provider key and proxy password must stay out of logs even then.
    """
    redacted = text
    for name, value in headers.items():
        if name.lower() not in {
            "authorization", "x-api-key", "x-subscription-token"
        }:
            continue
        secret = value.strip()
        if secret.lower().startswith("bearer "):
            secret = secret[7:].strip()
        if secret:
            redacted = redacted.replace(secret, "***")
    for name in _PROXY_ENV_NAMES:
        proxy = os.environ.get(name, "").strip()
        if not proxy:
            continue
        redacted = redacted.replace(proxy, "***")
        try:
            password = urlsplit(proxy).password
        except ValueError:
            password = None
        if password:
            redacted = redacted.replace(password, "***")
    return redacted


_BLOCK_PAGE_MARKERS = (
    "attention required",
    "just a moment",
    "checking your browser",
    "enable javascript and cookies",
    "cloudflare ray id",
)


def _looks_like_block_page(body: str, content_type: str, server: str) -> bool:
    """Cloudflare WAF block/challenge page instead of provider JSON.

    Exa blocks API access from several regions (Russia, Ukraine and others)
    before the request reaches the API. The page is HTML, sometimes with
    status 200, so ``json.loads`` would misreport it as ``invalid_json``.
    Ordinary Cloudflare 5xx pages do not carry these challenge markers and
    keep their real ``http_<code>`` error.
    """
    head = body[:4096].lower()
    if "html" not in content_type.lower() and "<html" not in head:
        return False
    if "cloudflare" not in server.lower() and "cloudflare" not in head:
        return False
    return any(marker in head for marker in _BLOCK_PAGE_MARKERS)


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

_ISOLATED_SEARCH_HINT = (
    "Для этого нового запроса история диалога исключена из контекста ради "
    "приватности ранее загруженных материалов. Выполни только текущий запрос; "
    "если он требует сведений из прежнего диалога, попроси пользователя "
    "уточнить их, не выдумывай недостающий контекст."
)


def isolate_search_history(messages: list[dict[str, Any]]) -> bool:
    """Изолировать новый чистый запрос от недоверенной истории для веб-поиска.

    Убирается вся история, включая ответы-пересказы внешних материалов, но не
    из БД. Вызывающий код должен оставить только поиск, без MCP/генерации/файлов:
    удаление маркера не даёт права повторно открыть приватные инструменты.

    Триггер — любой новый самостоятельный запрос, а не только «найди в
    интернете»: иначе один документ в старой истории навсегда отключал бы поиск
    для последующих вопросов на новые темы. Уточнения, текущие вложения,
    цитаты и страницы защитный гейт не обходят; явные запросы к заметкам
    изолируются ``mcp.isolate_mcp_history``, а не здесь.
    """
    if not messages or messages[-1].get("role") != "user":
        return False
    current = messages[-1]
    text = current.get("content")
    if (
        not isinstance(text, str)
        or messages_have_untrusted_content([current])
        or not is_fresh_request(text)
        or has_notes_target(text)
        or not messages_have_untrusted_content(messages[:-1], ignore_sources={"search"})
    ):
        return False
    # Не оставляем даже старые assistant-ответы: они могут пересказывать файл
    # без исходной untrusted-рамки и утечь в аргументы нового поиска.
    system = [dict(entry) for entry in messages if entry.get("role") == "system"]
    for entry in system:
        entry["content"] = f"{entry.get('content') or ''}\n\n{_ISOLATED_SEARCH_HINT}".strip()
    messages[:] = [*system, current]
    logger.info("New request isolated from external untrusted history for web search")
    return True


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
    category such as ``timeout``, ``http_401`` or ``blocked`` — never the
    provider body or the user's query.
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


@lru_cache(maxsize=1)
def duckduckgo_available() -> bool:
    """True when the optional ``ddgs`` package is installed.

    DuckDuckGo has no keyless JSON API: the provider scrapes the HTML result
    page through ``ddgs``. A broken installation counts as "not installed" so
    the tool is simply not offered instead of failing on every question.
    """
    try:
        return importlib.util.find_spec("ddgs") is not None
    except (ImportError, ValueError):
        return False


def provider_name() -> str:
    """Normalised ``SEARCH_PROVIDER`` (lower-case, aliases resolved)."""
    value = (config.search_provider or "").strip().lower()
    return _PROVIDER_ALIASES.get(value, value)


def search_configured() -> bool:
    """True when the ``search_web`` tool should be offered to the model."""
    if not config.enable_web_search:
        return False
    provider = provider_name()
    if provider not in PROVIDERS:
        return False
    if provider == "duckduckgo":
        return duckduckgo_available()
    return bool(config.search_api_key)


# --- HTTP helpers (keys are never logged) --------------------------------------


async def _request_json(
    url: str,
    headers: dict[str, str],
    *,
    method: str = "POST",
    json_payload: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
) -> _RequestResult:
    """One JSON request to a search API; never raises.

    Redirects are not followed: a cross-origin 3xx would otherwise forward
    custom auth headers (``x-api-key`` / ``X-Subscription-Token``) to another
    host. Search APIs answer directly, so a 3xx is a configuration error.

    ``trust_env`` honours the standard ``HTTPS_PROXY`` / ``HTTP_PROXY``
    variables: Exa blocks API access from several regions, so the bot must be
    able to reach it through a proxy/VPN egress.
    """
    host = _safe_host(url)
    timeout = aiohttp.ClientTimeout(total=config.search_timeout)
    started = time.monotonic()
    try:
        async with (
            aiohttp.ClientSession(timeout=timeout, trust_env=True) as session,
            session.request(
                method,
                url,
                json=json_payload,
                params=params,
                headers=headers,
                allow_redirects=False,
            ) as resp,
        ):
            body_bytes = await read_bounded(resp.content, _MAX_SEARCH_BODY)
            elapsed = time.monotonic() - started
            if len(body_bytes) > _MAX_SEARCH_BODY:
                logger.warning(
                    "Search API %s returned an oversized body after %.2fs [%s]",
                    host, elapsed, _operation_context(),
                )
                return _RequestResult(error="oversized_response")
            body = body_bytes.decode("utf-8", "replace")
            content_type = resp.headers.get("Content-Type", "")
            server = resp.headers.get("Server", "")
            if _looks_like_block_page(body, content_type, server):
                logger.warning(
                    "Search API %s returned a Cloudflare block page (HTTP %s) "
                    "after %.2fs [%s]; the provider may be regionally blocked — "
                    "set HTTPS_PROXY to an allowed network",
                    host, resp.status, elapsed, _operation_context(),
                )
                return _RequestResult(error="blocked")
            if resp.status != 200:
                logger.warning(
                    "Search API %s returned HTTP %s after %.2fs [%s]: %s",
                    host, resp.status, elapsed, _operation_context(),
                    _redact_secrets(body[:200], headers),
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
            type(exc).__name__, _redact_secrets(str(exc), headers),
        )
        return _RequestResult(error=reason)


# --- Providers ------------------------------------------------------------------


def _direct_url(url: str) -> str:
    """Resolve DuckDuckGo's ``/l/?uddg=<real url>`` redirect wrapper."""
    cleaned = (url or "").strip()
    if "duckduckgo.com/l/" in cleaned:
        target = (parse_qs(urlsplit(cleaned).query).get("uddg") or [""])[0]
        if target:
            cleaned = target
    return cleaned


def _http_url(value: Any) -> str:
    """Only absolute HTTP(S) links are returned; provider junk is dropped.

    The URL is displayed to the model and later fetched by the SSRF-protected
    loader, so ``javascript:``/``file:``/relative values must not get through.
    Whitespace and control characters are rejected too: a raw newline inside
    a URL could forge a line in the model's numbered result block.
    """
    url = str(value or "").strip()
    if not url or any(
        ch.isspace() or ord(ch) < 0x20 or ord(ch) == 0x7F for ch in url
    ):
        return ""
    try:
        parts = urlsplit(url)
    except ValueError:
        return ""
    if parts.scheme.lower() not in {"http", "https"} or not parts.hostname:
        return ""
    return url


def _duckduckgo_sync(query: str, max_results: int) -> list[dict[str, Any]]:
    """Blocking ``ddgs`` query; runs in a worker thread.

    ``ddgs`` is synchronous and spawns its own threads, so the event loop must
    never call it directly. An empty result set is a normal answer from the
    library (``DDGSException('No results found.')``), not a provider failure:
    ``html.duckduckgo.com`` answers 202 (an anti-bot page) instead of an error
    when the IP is throttled, and the library reports that as "no results".
    """
    from ddgs import DDGS
    from ddgs.exceptions import DDGSException

    backend = (config.search_ddgs_backend or "duckduckgo").strip().lower()
    try:
        with DDGS(timeout=config.search_timeout) as ddgs:
            return ddgs.text(
                query,
                region="wt-wt",
                safesearch="moderate",
                max_results=max_results,
                backend=backend or "duckduckgo",
            ) or []
    except DDGSException as exc:
        if "no results" in str(exc).lower():
            return []
        raise


async def _search_duckduckgo(query: str, max_results: int) -> SearchResults:
    """DuckDuckGo via the optional ``ddgs`` package (no API key)."""
    if not duckduckgo_available():
        logger.warning(
            "SEARCH_PROVIDER=duckduckgo needs the ddgs package "
            "(pip install ddgs) [%s]",
            _operation_context(),
        )
        return SearchResults(error="missing_dependency")
    try:
        # ddgs bounds each HTTP request by SEARCH_TIMEOUT; this also bounds the
        # thread call in case the library stalls inside its own thread pool.
        async with asyncio.timeout(config.search_timeout + 5):
            rows = await asyncio.to_thread(_duckduckgo_sync, query, max_results)
    except TimeoutError:
        logger.warning(
            "DuckDuckGo search timed out after %ss [%s]",
            config.search_timeout, _operation_context(),
        )
        return SearchResults(error="timeout")
    except Exception as exc:  # noqa: BLE001 — поиск не должен ломать ответ
        # Class name only: library messages may echo the query.
        reason = (
            "ratelimit" if "ratelimit" in type(exc).__name__.lower()
            else "provider_error"
        )
        logger.warning(
            "DuckDuckGo search failed [%s]: %s",
            _operation_context(), type(exc).__name__,
        )
        return SearchResults(error=reason)
    results: list[SearchResult] = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        url = _http_url(_direct_url(str(item.get("href") or item.get("url") or "")))
        if not url:
            continue
        results.append(
            SearchResult(
                title=_clean(item.get("title")) or url,
                url=url,
                snippet=_clean(item.get("body"))[:_SNIPPET_CHARS],
            )
        )
        if len(results) >= max_results:
            break
    return SearchResults(results)


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
        if not isinstance(item, dict):
            continue
        item_url = _http_url(item.get("url"))
        if not item_url:
            continue
        results.append(
            SearchResult(
                title=_clean(item.get("title")) or item_url,
                url=item_url,
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
        if not isinstance(item, dict):
            continue
        item_url = _http_url(item.get("url"))
        if not item_url:
            continue
        results.append(
            SearchResult(
                title=_clean(item.get("title")) or item_url,
                url=item_url,
                snippet=_clean(item.get("description"))[:_SNIPPET_CHARS],
            )
        )
        if len(results) >= max_results:
            break
    return SearchResults(results)


async def _search_exa(query: str, max_results: int) -> SearchResults:
    """Exa Search API (POST /search, ``x-api-key`` header).

    Only highlights are requested: full page text would duplicate the
    SSRF-protected deep read that every provider goes through and would make
    the provider response (and its cost) much larger. Highlights are also the
    fallback snippet when a result page cannot be fetched.
    """
    url = config.search_api_url or "https://api.exa.ai/search"
    data, error = _response_parts(await _request_json(
        url,
        {
            "x-api-key": config.search_api_key,
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
        json_payload={
            "query": query,
            "type": "auto",
            "numResults": max_results,
            "contents": {"highlights": True},
        },
    ))
    if error:
        return SearchResults(error=error)
    if not isinstance(data, dict):
        logger.warning(
            "Search API exa returned an unexpected payload [%s]",
            _operation_context(),
        )
        return SearchResults(error="unexpected_response")
    results: list[SearchResult] = []
    for item in data.get("results") or []:
        if not isinstance(item, dict):
            continue
        item_url = _http_url(item.get("url"))
        if not item_url:
            continue
        highlights = item.get("highlights")
        if isinstance(highlights, list):
            parts = [_clean(part) for part in highlights if part]
            snippet = " … ".join(part for part in parts if part)
        elif isinstance(highlights, str):
            snippet = _clean(highlights)
        else:
            snippet = ""
        if not snippet:
            # Defensive fallback: Exa may include text only when asked, but a
            # provider-side change must not leave the model without a snippet.
            snippet = _clean(item.get("text"))
        results.append(
            SearchResult(
                title=_clean(item.get("title")) or item_url,
                url=item_url,
                snippet=snippet[:_SNIPPET_CHARS],
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
    provider = provider_name()
    try:
        if provider == "duckduckgo":
            raw = await _search_duckduckgo(query, limit)
        elif provider == "tavily":
            raw = await _search_tavily(query, limit)
        elif provider == "brave":
            raw = await _search_brave(query, limit)
        elif provider == "exa":
            raw = await _search_exa(query, limit)
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
