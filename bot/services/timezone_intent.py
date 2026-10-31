"""AI mapping of a free-form location («я в Казани») to a valid IANA zone.

Used by the ``/timezone`` dialog: the user describes where they live, the
model proposes one IANA zone name, and the handler shows the user the
current time in that zone for confirmation before anything is saved. The
returned name is re-validated against :mod:`zoneinfo`, so a hallucinated
zone can never reach the database.

Определение зоны не расходует квоту вопросов, но обработчик перед AI-вызовом
перепроверяет доступ и резервирует persistent-бюджет, общий с free-form
напоминаниями. Текстовый вход удерживает per-user slot. TTL диалога и max_tokens
не заменяют бюджет попыток. Диалог игнорирует форварды и истекает через 10 минут.
"""

from __future__ import annotations

import datetime
import json
import logging
from zoneinfo import ZoneInfo

from bot.services.ai_client import ai_client

logger = logging.getLogger(__name__)

DETECT_SYSTEM_PROMPT = (
    "You map a user's description of where they live or work to an IANA time "
    "zone name. Answer with STRICT JSON only — no prose, no code fences:\n"
    '{"zone": "<IANA name>"} — the single best zone for the place;\n'
    '{"zone": null} — the message is not a place or you cannot tell.\n\n'
    "Rules:\n"
    "- One valid IANA zone name (Area/City) exactly as in the tz database, "
    "e.g. Europe/Moscow, Europe/Kaliningrad, Asia/Yekaterinburg.\n"
    "- For cities without their own zone use the zone of the nearest large "
    "city in the same country that has one (Kazan / Ryazan → Europe/Moscow).\n"
    "- Single-time-zone countries: use their capital's zone "
    "(Astana → Asia/Almaty, Minsk → Europe/Minsk, Kyiv → Europe/Kyiv).\n"
    "- «same as Moscow time» / «как в Москве» counts: pick Europe/Moscow.\n"
    "- Never invent zone names; if unsure answer null."
)


def is_valid_zone_name(name: str) -> bool:
    """True when ``name`` is an actual IANA zone (usable with zoneinfo)."""
    try:
        ZoneInfo(name)
    except Exception:  # noqa: BLE001 - unknown name or missing tzdata
        return False
    return True


def _extract_zone(answer: str) -> str | None:
    start, end = answer.find("{"), answer.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(answer[start : end + 1])
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    zone = data.get("zone")
    if not isinstance(zone, str):
        return None
    return zone.strip()


async def detect_timezone(
    text: str, model: str, now: datetime.datetime | None = None
) -> str | None:
    """Turn a location description into a *validated* IANA zone name.

    Returns the zone name, or ``None`` when the AI failed, the JSON is
    broken, the answer is not a place, or the proposed name does not exist
    in the tz database (the caller keeps asking).
    """
    if not text or not text.strip():
        return None
    now = now or datetime.datetime.now(datetime.UTC)
    try:
        result = await ai_client.chat_completion_result(
            messages=[
                {"role": "system", "content": DETECT_SYSTEM_PROMPT},
                {"role": "user", "content": f"Current UTC time: {now:%Y-%m-%d %H:%M}\n{text}"},
            ],
            model=model,
            temperature=0.0,
            max_tokens=80,
        )
    except Exception:
        logger.warning("Timezone detection call failed", exc_info=True)
        return None

    proposed = _extract_zone(result.content or "")
    if proposed is None:
        return None
    if not is_valid_zone_name(proposed):
        logger.warning("Model proposed unknown IANA zone %r; rejected", proposed)
        return None
    return proposed
