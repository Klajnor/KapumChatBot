"""Актуальная дата для относительных дат в запросах модели."""

from __future__ import annotations

import datetime


def current_time_hint(
    tz: datetime.tzinfo = datetime.UTC, *, now: datetime.datetime | None = None,
) -> str:
    """Дать дату и явно вычисленное «вчера» в выбранном часовом поясе."""
    local = (now or datetime.datetime.now(datetime.UTC)).astimezone(tz)
    yesterday = local.date() - datetime.timedelta(days=1)
    return (
        f"Текущая дата и время: {local:%Y-%m-%d %H:%M} ({local.tzname()}). "
        f"Сегодня: {local:%Y-%m-%d}. Вчера: {yesterday:%Y-%m-%d}. "
        "Относительные даты («сегодня», «вчера») вычисляй по этой дате, "
        "а не по датам из истории диалога или обучающих данных."
    )
