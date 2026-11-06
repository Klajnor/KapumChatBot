"""Получение саммари: сервис знаний — канонический источник, LLM — legacy-режим."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import aiohttp

from .knowledge import KnowledgeClient, KnowledgeError, KnowledgeSource


@dataclass(frozen=True)
class SummaryResult:
    text: str = ""
    sources: tuple[KnowledgeSource, ...] = ()
    error: str | None = None


async def get_issue_summary(
    session: aiohttp.ClientSession,
    *,
    knowledge: KnowledgeClient | None,
    llm: Any,
    redmine: Any,
    issue: dict[str, Any],
    note: str = "",
) -> SummaryResult:
    """Саммари Redmine-задачи.

    Если сервис знаний настроен, локальная LLM-суммаризация НЕ выполняется:
    при ошибке/отсутствии саммари возвращается пустой текст и причина.
    """
    issue_id = issue.get("id")
    if knowledge is not None:
        if issue_id is None:
            return SummaryResult(error="у задачи нет Redmine ID")
        try:
            hit = await knowledge.get_summary(session, "redmine", issue_id)
        except KnowledgeError as error:
            return SummaryResult(error=str(error))
        if not hit.has_sources:
            return SummaryResult(error="саммари без проверяемых источников")
        return SummaryResult(text=hit.summary, sources=hit.sources)

    if llm is None:
        return SummaryResult()
    source = redmine.summary_source(issue)
    if note:
        source = f"{source}\n\n{note}".strip() if source else note
    if not source:
        return SummaryResult()
    try:
        return SummaryResult(text=await llm.summarize(session, source))
    except Exception as error:  # noqa: BLE001 - деградируем без падения обработчика
        return SummaryResult(error=f"{type(error).__name__}: {error}")


def plain_sources(sources: tuple[KnowledgeSource, ...]) -> str:
    seen: set[str] = set()
    lines: list[str] = []
    for source in sources:
        if source.url in seen:
            continue
        seen.add(source.url)
        lines.append(source.url)
    return "\n".join(lines)
