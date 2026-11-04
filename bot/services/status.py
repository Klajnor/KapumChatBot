"""Отчёт о состоянии бота для свободного запроса «статус» / «как дела».

Отчёт собирается из дешёвых локальных проверок и двух внешних источников:

* второй мозг — MCP-сервер отдаёт только числовые счётчики заметок, тегов,
  оценённых и ожидающих связей (никаких путей и текста заметок);
* поиск — один настоящий пробный запрос к провайдеру не чаще
  ``STATUS_SEARCH_INTERVAL`` секунд на пользователя. Лимит хранится в БД,
  поэтому перезапуск бота его не сбрасывает. Если лимит исчерпан, отчёт
  показывает время следующей проверки и не считает это ошибкой.

Сама команда не расходует квоту вопросов, не пишет историю и не вызывает
модель для ответа: модель нужна только чтобы отличить статус бота от обычной
болтовни (см. ``services/request_intent.py``). Ошибки поиска и MCP
превращаются в короткий статус «недоступно», а не роняют ответ.
"""

from __future__ import annotations

import asyncio
import datetime
import html
import logging
from typing import Any

from bot.config import config
from bot.services.health import uptime_seconds
from bot.services.mcp import mcp_available, mcp_manager
from bot.services.reminders import get_timezone
from bot.services.search import provider_name, search_configured, search_web
from bot.services.user_service import UserService

logger = logging.getLogger(__name__)

# Neutral query: the probe checks the provider, not the user's interests, and
# its text never appears in the report.
_STATUS_SEARCH_QUERY = "OpenAI"
_STATUS_SEARCH_RESULTS = 1

_SEARCH_ERRORS = {
    "timeout": "timed out",
    "network": "network error",
    "blocked": "blocked by the provider or region",
    "oversized_response": "provider returned an oversized response",
    "invalid_json": "provider returned invalid JSON",
    "unexpected_response": "provider returned an unexpected response",
    "missing_dependency": "search package is not installed",
    "unknown_provider": "search provider is not configured",
    "internal_error": "internal search error",
    "ratelimit": "provider rate limit",
    "provider_error": "provider error",
    "request_failed": "request failed",
}


def _human_duration(seconds: float) -> str:
    """Compact uptime/cooldown text: ``2 d 3 h``, ``5 h 12 min``, ``42 s``."""
    total = max(0, int(seconds))
    days, rest = divmod(total, 86_400)
    hours, rest = divmod(rest, 3_600)
    minutes, secs = divmod(rest, 60)
    if days:
        return f"{days} d {hours} h"
    if hours:
        return f"{hours} h {minutes} min"
    if minutes:
        return f"{minutes} min {secs} s"
    return f"{secs} s"


def _as_utc(value: datetime.datetime) -> datetime.datetime:
    """Treat a naive SQLite datetime as UTC and return an aware UTC value."""
    if value.tzinfo is None:
        return value.replace(tzinfo=datetime.UTC)
    return value.astimezone(datetime.UTC)


def _local_time(value: datetime.datetime, tz: datetime.tzinfo) -> str:
    """User-facing date/time in the requested zone."""
    local = _as_utc(value).astimezone(tz)
    zone = local.tzname() or "UTC"
    return f"{local:%Y-%m-%d %H:%M} {zone}"


def _count(value: Any) -> int | None:
    """A non-negative counter from the MCP payload, or ``None``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return max(0, int(value))


def _search_error(code: str) -> str:
    """Safe human-readable search failure; the provider body is never shown."""
    return _SEARCH_ERRORS.get(code, f"provider error ({code or 'unknown'})")


async def _search_status(
    service: UserService,
    user_id: int,
    *,
    private: bool,
    tz: datetime.tzinfo,
    now: datetime.datetime,
) -> list[str]:
    """The search block: configuration, cooldown and one real probe."""
    provider = provider_name()
    if not config.enable_web_search:
        return ["🔎 <b>Search:</b> disabled"]
    if not search_configured():
        return [
            (
                f"🔎 <b>Search:</b> ⚠️ not configured "
                f"(provider <code>{html.escape(provider or 'unknown')}</code>)"
            )
        ]
    if not private:
        # A live probe is a paid provider call and its cooldown is personal:
        # do not turn a group status request into one.
        return [
            (
                f"🔎 <b>Search:</b> configured "
                f"(provider <code>{html.escape(provider)}</code>); "
                "run this check in DM for a live probe"
            )
        ]

    previous = await service.try_reserve_status_search(user_id)
    if previous is not None:
        next_at = _as_utc(previous) + datetime.timedelta(
            seconds=config.status_search_interval
        )
        wait = max(0.0, (next_at - now).total_seconds())
        return [
            "🔎 <b>Search:</b> hourly live check already used",
            (
                f"   Next real check: <b>{_local_time(next_at, tz)}</b> "
                f"(in {_human_duration(wait)})"
            ),
        ]

    try:
        results = await search_web(_STATUS_SEARCH_QUERY, max_results=_STATUS_SEARCH_RESULTS)
    except Exception:
        logger.warning("Status search probe failed", exc_info=True)
        return ["🔎 <b>Search:</b> ❌ check failed"]

    if results.failed:
        return [f"🔎 <b>Search:</b> ❌ {html.escape(_search_error(results.error))}"]
    count = len(results)
    if count:
        outcome = f"✅ reachable ({count} result{'s' if count != 1 else ''})"
    else:
        outcome = "✅ reachable (no results)"
    return [f"🔎 <b>Search:</b> {outcome} · provider <code>{html.escape(provider)}</code>"]


async def _brain_status(user_id: int) -> list[str]:
    """The second-brain block: counts only, never note text or paths."""
    if not config.enable_mcp:
        return ["🧠 <b>Second brain:</b> disabled"]
    if not mcp_available(user_id):
        return ["🧠 <b>Second brain:</b> not connected"]

    stats = await mcp_manager.brain_status()
    if stats is None or stats.get("error"):
        return ["🧠 <b>Second brain:</b> ⚠️ unavailable"]

    notes = _count(stats.get("notes"))
    chunks = _count(stats.get("chunks"))
    links = _count(stats.get("links"))
    strong = _count(stats.get("strong_links"))
    pending = _count(stats.get("pending_links"))
    failed = _count(stats.get("failed_links"))

    lines = [
        "🧠 <b>Second brain:</b> ✅ connected",
        (
            f"   📝 Notes: {notes if notes is not None else '—'}"
            f" · 🧩 Chunks: {chunks if chunks is not None else '—'}"
        ),
    ]
    if links is not None:
        line = f"   🔗 Links: {links} indexed"
        if strong is not None:
            line += f" ({strong} strong)"
        if pending is not None:
            line += f" · ⏳ {pending} not indexed"
        if failed:
            line += f" · ⚠️ {failed} deferred"
        lines.append(line)
    if stats.get("links_enabled") is False:
        lines.append("   <i>Link building is disabled (BRAIN_LINKS_ENABLED=0).</i>")
    return lines


async def render_status(
    service: UserService,
    user_id: int,
    *,
    private: bool,
) -> str:
    """Build the HTML status report for one user.

    ``private`` controls personal sections: daily quotas and the live search
    probe are shown only in a private chat, so a group cannot learn another
    participant's limits or spend their hourly check.
    """
    now = datetime.datetime.now(datetime.UTC)
    settings = await service.get_user_settings(user_id)
    model = config.resolve_model(settings["model"])
    model_name = model.get("name") or settings["model"] or config.default_model
    tz = get_timezone(await service.get_reminder_timezone(user_id))

    lines = [
        "🤖 <b>Bot status</b>",
        "",
        f"🟢 <b>Online</b> · uptime {_human_duration(uptime_seconds())}",
        f"🧠 <b>Model:</b> <code>{html.escape(str(model_name))}</code>",
        "💾 <b>Database:</b> ✅ responsive",
    ]

    search_lines, brain_lines = await asyncio.gather(
        _search_status(service, user_id, private=private, tz=tz, now=now),
        _brain_status(user_id),
    )
    lines.extend(search_lines)
    lines.extend(brain_lines)

    if private:
        level = await service.get_level(user_id)
        lines.extend(["", "👤 <b>Your access</b>"])
        if level in {"owner", "admin"}:
            lines.append("♾️ Admin — unlimited questions")
        elif level == "blocked":
            lines.append("🚫 Access is blocked")
        elif level == "user":
            remaining = await service.get_daily_usage_remaining(user_id)
            limit = await service.get_daily_limit(user_id)
            today = datetime.date.today().isoformat()
            lines.append(
                f"📅 {today}: <b>{remaining}/{limit}</b> questions left"
            )
        elif config.allow_guest_access:
            remaining = await service.get_guest_messages_remaining(user_id)
            lines.append(
                f"🎁 Free trial: <b>{remaining}/{config.guest_message_limit}</b> "
                "messages left"
            )
        else:
            lines.append("🌌 Access is restricted")
        lines.append("<i>Status checks do not use your question limit.</i>")

    return "\n".join(lines)
