"""Подтверждённые API-метаданные поиска и reasoning, без анализа текста ответа."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


def _counter(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not 0 <= value <= 2**63 - 1 or not math.isfinite(value) or int(value) != value:
        return None
    return int(value)


def reasoning_tokens(usage: Any) -> int | None:
    """Счётчик размышления OpenAI, DashScope или совместимого API (не текст мысли)."""
    if not isinstance(usage, dict):
        return None
    for key in ("completion_tokens_details", "output_tokens_details"):
        details = usage.get(key)
        if isinstance(details, dict):
            value = _counter(details.get("reasoning_tokens"))
            if value is not None:
                return value
    return _counter(usage.get("reasoning_tokens"))


def model_search_used(data: dict[str, Any], *, enabled: bool | None = None) -> bool | None:
    """Факт провайдерского поиска; enable_search — разрешение, а не доказательство.

    DashScope сообщает usage.plugins.search.count и/или output.search_info.
    Пустая выдача без счётчика не доказывает отсутствие поиска. Ничего не
    заключаем по словам модели, ссылкам в обычном тексте или forced_search.
    """
    usage = data.get("usage")
    plugins = usage.get("plugins") if isinstance(usage, dict) else None
    search = plugins.get("search") if isinstance(plugins, dict) else None
    count = _counter(search.get("count")) if isinstance(search, dict) else None
    if count is not None and count > 0:
        return True
    output = data.get("output")
    for container in (data, output):
        if not isinstance(container, dict):
            continue
        info = container.get("search_info")
        if isinstance(info, dict) and isinstance(info.get("search_results"), list) and info["search_results"]:
            return True
    if count == 0:
        return False
    return False if enabled is False else None


def merge_usage(previous: dict[str, Any] | None, current: dict[str, Any]) -> dict[str, Any]:
    """Обновить накопительный SSE usage, не суммируя повторные снимки токенов."""
    merged = dict(previous or {})
    for key, value in current.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_usage(merged[key], value)
        else:
            merged[key] = value
    return merged


@dataclass(frozen=True, slots=True)
class ResponseMetadata:
    """Только известные счётчики; None не заменяется нулём."""

    model_search_used: bool | None = None
    api_search_used: bool = False
    reasoning_tokens: int | None = None
    reasoning_partial: bool = False


def response_metadata(data: dict[str, Any], *, search_enabled: bool | None = None) -> ResponseMetadata:
    """Выделить безопасную числовую/булеву диагностику обычного API-ответа."""
    return ResponseMetadata(
        model_search_used=model_search_used(data, enabled=search_enabled),
        reasoning_tokens=reasoning_tokens(data.get("usage")),
    )


def metadata_for_result(result: Any) -> ResponseMetadata:
    """Получить метаданные ChatResult, сохраняя совместимость с test doubles."""
    metadata = getattr(result, "metadata", None)
    if isinstance(metadata, ResponseMetadata):
        return metadata
    return response_metadata({"usage": getattr(result, "usage", None)})


def combine_metadata(responses: list[ResponseMetadata]) -> ResponseMetadata:
    """Объединить разные API-вызовы/раунды, не теряя ранний поиск и reasoning."""
    if not responses:
        return ResponseMetadata()
    searches = [m.model_search_used for m in responses]
    reported = [m.reasoning_tokens for m in responses if m.reasoning_tokens is not None]
    return ResponseMetadata(
        model_search_used=True if True in searches else (None if None in searches else False),
        api_search_used=any(m.api_search_used for m in responses),
        reasoning_tokens=sum(reported) if reported else None,
        reasoning_partial=bool(reported) and any(
            m.reasoning_tokens is None or m.reasoning_partial for m in responses
        ),
    )


@dataclass(slots=True)
class AnswerMetadata:
    """Метрики одного пользовательского вопроса: модельные раунды и локальный API."""

    responses: list[ResponseMetadata] = field(default_factory=list)
    api_search_used: bool = False

    def add(self, result: Any) -> None:
        """Учесть очередной результат модели ровно один раз."""
        self.responses.append(metadata_for_result(result))

    def summary(self) -> ResponseMetadata:
        """Сводка для футера; API-поиск учитывается по реальному выполнению."""
        combined = combine_metadata(self.responses)
        return ResponseMetadata(
            model_search_used=combined.model_search_used,
            api_search_used=self.api_search_used or combined.api_search_used,
            reasoning_tokens=combined.reasoning_tokens,
            reasoning_partial=combined.reasoning_partial,
        )
