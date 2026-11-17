"""Статистика использования: подписи пользователей и rich-таблица.

Имя для отчёта берётся из профиля Telegram (``first_name`` + ``last_name``),
затем ``@username``, затем числовой ID. Rich-вариант — GFM-таблица для
``sendRichMessage``; при выключенных или отклонённых rich-сообщениях
обработчики показывают HTML-список.
"""

from __future__ import annotations

import datetime
import html
from typing import Any

# Сколько строк показывать без прокрутки: rich-таблица на весь экран уже
# малочитаема, а список из сотен строк — тем более.
MAX_REPORT_ROWS = 50
MAX_PANEL_ROWS = 10
MAX_NAME_LENGTH = 32


def _clip(name: str) -> str:
    name = " ".join(name.split())
    if len(name) <= MAX_NAME_LENGTH:
        return name
    return f"{name[: MAX_NAME_LENGTH - 1]}…"


def usage_name(row: dict[str, Any] | None, telegram_id: int) -> str:
    """Имя для таблицы: профиль Telegram → @ник → числовой ID."""
    if row:
        full_name = " ".join(
            str(part).strip()
            for part in (row.get("first_name"), row.get("last_name"))
            if part
        ).strip()
        if full_name:
            return _clip(full_name)
        username = str(row.get("username") or "").strip().lstrip("@")
        if username:
            return _clip(f"@{username}")
    return str(telegram_id)


def aggregate_usage(
    entries: list[dict[str, Any]], known_users: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Свернуть строки ``daily_usage`` в итоги по пользователям.

    ``known_users`` — результат ``UserService.list_known_users``: оттуда
    берутся имя и ник. Пользователь без профиля показывается по ID.
    """
    known = {int(row["telegram_id"]): row for row in known_users}
    today = datetime.date.today()
    users: dict[int, dict[str, Any]] = {}
    for entry in entries:
        count = int(entry["query_count"])
        if count <= 0:
            continue
        telegram_id = int(entry["telegram_id"])
        item = users.setdefault(
            telegram_id,
            {"telegram_id": telegram_id, "total": 0, "today": 0, "days": 0},
        )
        item["total"] += count
        item["days"] += 1
        date = entry["date"]
        if isinstance(date, datetime.datetime):
            date = date.date()
        if date == today:
            item["today"] += count

    rows = list(users.values())
    for item in rows:
        item["name"] = usage_name(
            known.get(item["telegram_id"]), item["telegram_id"]
        )
    rows.sort(key=lambda item: (-item["total"], item["name"].casefold()))
    return rows


def _cell(value: str) -> str:
    """Убрать из ячейки всё, что ломает GFM-таблицу."""
    text = str(value).replace("\\", "\\\\").replace("|", "\\|")
    return " ".join(text.split())


def table_markdown(
    rows: list[dict[str, Any]], *, limit: int = MAX_REPORT_ROWS
) -> str:
    """GFM-таблица для rich-сообщения; ``rows`` уже отсортированы."""
    lines = [
        "| # | User | Today | 14 days | Days |",
        "|---:|:---|---:|---:|---:|",
    ]
    for index, row in enumerate(rows[:limit], start=1):
        lines.append(
            f"| {index} | {_cell(row['name'])} | {row['today']} | "
            f"{row['total']} | {row['days']} |"
        )
    if len(rows) > limit:
        lines.append(f"\n_{len(rows) - limit} more users not shown._")
    return "\n".join(lines)


def table_html(
    rows: list[dict[str, Any]], *, limit: int = MAX_REPORT_ROWS
) -> str:
    """Запасной HTML-вариант для сообщений без rich-рендеринга."""
    lines = [
        f"{index}. <b>{html.escape(row['name'])}</b> — {row['total']} "
        f"(today {row['today']}, days {row['days']})"
        for index, row in enumerate(rows[:limit], start=1)
    ]
    if len(rows) > limit:
        lines.append(f"\n<i>... and {len(rows) - limit} more users.</i>")
    return "\n".join(lines)