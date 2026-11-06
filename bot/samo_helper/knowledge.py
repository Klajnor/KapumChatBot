"""Клиент отдельного сервиса рабочих знаний (саммари Redmine/MediaWiki).

Сервис — единственный источник саммари и ключевых слов. Бот не ходит в
полные тексты задач/страниц и не придумывает ссылки: источники приходят в
ответе сервиса и проверяются по allowlist хостов.

Ожидаемый контракт (см. README):
  GET /search?q=<текст>&limit=<n>&source=redmine|wiki
  GET /summaries/<source>/<id>
Ответ:
  {"results": [{
      "id": "...", "title": "...", "summary": "...",
      "keywords": ["..."], "sources": [
          {"type": "redmine", "id": "123", "title": "...", "url": "https://..."}
      ],
      "updated_at": "...", "generated_at": "...", "stale": false
  }]}
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import aiohttp

MAX_SUMMARY_CHARS = 6000


class KnowledgeError(RuntimeError):
    """Базовая ошибка сервиса знаний."""


class KnowledgeUnavailable(KnowledgeError):
    """Сеть, таймаут или 5xx — сервис временно недоступен."""


class KnowledgeAuthError(KnowledgeError):
    """Сервис отклонил ключ доступа."""


class KnowledgeNotFound(KnowledgeError):
    """Саммари по точному источнику ещё нет."""


class KnowledgePending(KnowledgeError):
    """Саммари ещё обрабатывается."""


class KnowledgeInvalidResponse(KnowledgeError):
    """Ответ не соответствует контракту."""


@dataclass(frozen=True)
class KnowledgeSource:
    type: str
    id: str
    title: str
    url: str


@dataclass(frozen=True)
class SummaryHit:
    id: str
    title: str
    summary: str
    keywords: tuple[str, ...]
    sources: tuple[KnowledgeSource, ...]
    updated_at: str | None = None
    generated_at: str | None = None
    stale: bool = False

    @property
    def has_sources(self) -> bool:
        return bool(self.sources)


def _host_allowed(url: str, allowed_hosts: Iterable[str] | None) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return False
    if parsed.username or parsed.password:
        return False
    if not allowed_hosts:
        return True
    host = parsed.hostname.lower()
    return any(host == item or host.endswith("." + item)
               for item in (h.lower() for h in allowed_hosts))


class KnowledgeClient:
    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        *,
        timeout: float = 10,
        allowed_hosts: Iterable[str] | None = None,
    ) -> None:
        parsed = urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("KNOWLEDGE_API_URL должен быть http(s)-адресом")
        if not 1 <= timeout <= 60:
            raise ValueError("KNOWLEDGE_TIMEOUT должен быть от 1 до 60 секунд")
        self._base = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._allowed_hosts = tuple(allowed_hosts or ())

    @property
    def enabled(self) -> bool:
        return bool(self._base)

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}

    async def _get(self, session: aiohttp.ClientSession, path: str, **params: Any) -> dict[str, Any]:
        try:
            async with session.get(
                f"{self._base}{path}", headers=self._headers(),
                params=params, timeout=aiohttp.ClientTimeout(total=self._timeout),
            ) as resp:
                if resp.status in (401, 403):
                    raise KnowledgeAuthError("сервис знаний: доступ запрещён")
                if resp.status == 404:
                    raise KnowledgeNotFound("саммари не найдено")
                if resp.status in (202, 409):
                    raise KnowledgePending("саммари ещё обрабатывается")
                if resp.status >= 500:
                    raise KnowledgeUnavailable(f"сервис знаний: HTTP {resp.status}")
                if resp.status != 200:
                    raise KnowledgeError(f"сервис знаний: HTTP {resp.status}")
                try:
                    data = await resp.json()
                except (ValueError, aiohttp.ContentTypeError) as error:
                    raise KnowledgeInvalidResponse("сервис знаний: ответ не JSON") from error
        except aiohttp.ClientError as error:
            raise KnowledgeUnavailable(f"сервис знаний: сеть: {error}") from error
        except TimeoutError as error:
            raise KnowledgeUnavailable("сервис знаний: таймаут") from error
        if not isinstance(data, dict):
            raise KnowledgeInvalidResponse("сервис знаний: ожидался объект")
        return data

    def _parse_hit(self, raw: Any) -> SummaryHit:
        if not isinstance(raw, dict):
            raise KnowledgeInvalidResponse("элемент выдачи не объект")
        title = str(raw.get("title") or "").strip()
        summary = str(raw.get("summary") or "").strip()
        if not title or not summary:
            raise KnowledgeInvalidResponse("в саммари нет title/summary")
        summary = summary[:MAX_SUMMARY_CHARS]
        keywords = tuple(
            str(item).strip() for item in (raw.get("keywords") or [])
            if str(item).strip()
        )
        sources: list[KnowledgeSource] = []
        for item in raw.get("sources") or []:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "").strip()
            if not _host_allowed(url, self._allowed_hosts):
                continue
            sources.append(KnowledgeSource(
                type=str(item.get("type") or "unknown"),
                id=str(item.get("id") or ""),
                title=str(item.get("title") or "").strip(),
                url=url,
            ))
        return SummaryHit(
            id=str(raw.get("id") or ""), title=title, summary=summary,
            keywords=keywords, sources=tuple(sources),
            updated_at=raw.get("updated_at"),
            generated_at=raw.get("generated_at"),
            stale=bool(raw.get("stale", False)),
        )

    async def search(
        self, session: aiohttp.ClientSession, query: str, *,
        limit: int = 5, source: str | None = None,
    ) -> list[SummaryHit]:
        query = " ".join(query.split())
        if not query:
            return []
        params: dict[str, Any] = {"q": query, "limit": max(1, min(limit, 20))}
        if source:
            params["source"] = source
        data = await self._get(session, "/search", **params)
        results = data.get("results")
        if results is None and "items" in data:
            results = data["items"]
        if not isinstance(results, list):
            raise KnowledgeInvalidResponse("в ответе нет списка results")
        hits: list[SummaryHit] = []
        for item in results:
            try:
                hits.append(self._parse_hit(item))
            except KnowledgeInvalidResponse:
                # Один битый элемент не должен ломать всю выдачу.
                continue
        return hits

    async def get_summary(
        self, session: aiohttp.ClientSession, source: str, source_id: str | int,
    ) -> SummaryHit:
        data = await self._get(session, f"/summaries/{source}/{source_id}")
        raw = data.get("result") or data.get("summary") or data
        return self._parse_hit(raw)


def format_source_links(hit: SummaryHit) -> list[tuple[str, str]]:
    """(подпись, url) для детерминированного блока источников."""
    result: list[tuple[str, str]] = []
    for source in hit.sources:
        label = source.title
        if not label:
            label = f"{source.type} #{source.id}" if source.id else source.url
        result.append((label, source.url))
    return result


def dedupe_sources(hits: Iterable[SummaryHit]) -> list[KnowledgeSource]:
    seen: set[str] = set()
    result: list[KnowledgeSource] = []
    for hit in hits:
        for source in hit.sources:
            if source.url not in seen:
                seen.add(source.url)
                result.append(source)
    return result
