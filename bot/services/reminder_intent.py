"""Free-form reminder requests: «напомни завтра позвонить маме» without a command.

``/remind`` requires a strict argument format, but users also write natural
requests right in the chat. This module turns them into the same
:class:`~bot.services.reminders.ParsedReminder` in three layers:

1. ``looks_like_reminder`` — a cheap regex pre-filter, so ordinary chat
   messages never pay for detection at all;
2. ``extract_freeform`` — one small AI call (``temperature=0``, like
   ``services/intent.py``) decides whether the message really is a reminder
   request and rewrites the time into the exact ``/remind`` vocabulary
   (``in 30 minutes``, ``tomorrow at 9:00``, ``daily at 10:00``, …);
3. the model's slot is validated by the plain ``parse_reminder`` regexes, so a
   hallucination cannot create a reminder that fires more often than
   REMINDER_MIN_INTERVAL_MINUTES or amounts to an unrepresentable interval.

Any fallback path — AI failure, broken JSON, a message that is not a request —
returns ``None`` and the caller continues the normal chat flow.

Intended behaviour (not a gap): the detection call is a small, bounded
service call and does NOT consume the daily question quota, exactly like
creating a reminder (see ``/remind``). Its cost is bounded by the pre-filter,
the per-user concurrency cap, ``max_tokens`` and the reminder limits. Forwarded
messages never reach it. The *scheduled* AI run, by contrast, is charged.
"""

from __future__ import annotations

import datetime
import json
import logging
import re
from dataclasses import dataclass

from bot.config import config
from bot.services.ai_client import ai_client
from bot.services.reminders import (
    ParsedReminder,
    get_timezone,
    parse_reminder,
)

logger = logging.getLogger(__name__)

# Cheap pre-filter over the first REMINDER_PREFILTER_CHARS characters. A false
# positive merely costs one tiny AI call, a false negative loses the feature,
# so the net is cast wide (rus+eng). Two independent signals:
# 1. reminder words by stem — «поставь/установи/устнови напоминалку» match on
#    the noun, so a typo or any verb in front does not matter;
_TRIGGER_WORDS = re.compile(
    r"напом|будильник|разбуд|не\s+(?:дай|давай)\s+забы|таймер|remind|alarm",
    re.IGNORECASE,
)
# 2. a schedule phrase with no reminder word at all: «завтра в 9 утра покажи
#    погоду в москве», «каждый день в 8 присылай новости». The AI call decides.
_SCHEDULE_PHRASE = re.compile(
    r"\bзавтра\b|\bпослезавтра\b"
    r"|\bчерез\s+(?:\d+|час|полчаса|минуту|пару|несколько)\b"
    r"|\bв\s+\d{1,2}(?:[:.]\d{2})?\b"
    r"|\bкажд(?:ый|ое|ую|ые)\b|\bежедневно\b|\bеженедельно\b"
    r"|\bпо\s+(?:будням|выходным|понедельникам|вторникам|средам|четвергам"
    r"|пятницам|субботам|воскресеньям)\b"
    r"|\btomorrow\b|\bin\s+\d+\b|\bevery\b|\bdaily\b|\bat\s+\d",
    re.IGNORECASE,
)

INTENT_SYSTEM_PROMPT = (
    "You convert natural-language requests into a strict reminder format for "
    "a Telegram bot. The user message may ask to SET a reminder (one-time or "
    "recurring messages like drinking water every hour). Answer with STRICT "
    "JSON only — no prose, no code fences:\n"
    '{"reminder": false} — the message is anything else: a question, chat, '
    "an opinion, asking to remember a fact for the conversation, asking "
    "about the bot's reminders, deciding between options.\n"
    '{"reminder": true, "spec": "<SLOT>", "text": "<TEXT>"} — it is a '
    "reminder request.\n\n"
    "<SLOT> MUST be one of, exactly:\n"
    "- in N minutes | in N hours | in N days | in N weeks  (N is a number)\n"
    "- at HH:MM  (24-hour; if the time already passed today, it is tomorrow)\n"
    "- tomorrow at HH:MM\n"
    "- every N minutes | every N hours | every N days  (recurring)\n"
    "- every hour  (recurring)\n"
    "- daily at HH:MM  (recurring)\n"
    "- weekly mon|tue|wed|thu|fri|sat|sun at HH:MM  (recurring)\n\n"
    "Rules:\n"
    "- Local time, date and weekday of the user are at the start of the user "
    "message; interpret all relative times against it.\n"
    "- Map vague times: morning≈09:00, midday/noon=12:00, "
    "afternoon≈15:00, evening≈20:00, night≈21:00.\n"
    "- Requested phrases like «every weekday» cannot be expressed exactly; "
    "use the closest slot only when it keeps the user's intent (weekday "
    "work requests ≈ daily), otherwise answer {\"reminder\": false}.\n"
    "- Decide PLAIN vs AI. PLAIN: the user must do something themselves "
    "(call mom, drink water, buy bread) — TEXT is just that note. AI: at fire "
    "time the bot itself must fetch or produce something (weather, news, "
    "exchange rates, a summary, ideas, «покажи», «пришли», «расскажи») — "
    "prefix TEXT with 'ai: ' and make the rest a self-contained request. The "
    "bot can search the web when the AI request runs.\n"
    "- A message with a time phrase but no word «remind» (e.g. «завтра в 9 "
    "утра покажи погоду в москве») is still a reminder request when it asks "
    "the bot to do something at that time.\n"
    "- TEXT is the reminder itself without any time phrase.\n\n"
    "Examples:\n"
    '«напомни завтра в 5 вечера позвонить маме» → '
    '{"reminder": true, "spec": "tomorrow at 17:00", "text": "позвонить маме"}\n'
    '«remind me in half an hour to check the oven» → '
    '{"reminder": true, "spec": "in 30 minutes", "text": "check the oven"}\n'
    '«every morning say hi» → '
    '{"reminder": true, "spec": "daily at 09:00", "text": "say hi"}\n'
    '«завтра в 9 утра покажи погоду в москве» → '
    '{"reminder": true, "spec": "tomorrow at 09:00", '
    '"text": "ai: покажи погоду в москве"}\n'
    '«каждый день в 8 присылай новости по ИИ» → '
    '{"reminder": true, "spec": "daily at 08:00", '
    '"text": "ai: пришли свежие новости по ИИ"}\n'
    '«поставь напоминалку на 18:00 купить хлеб» → '
    '{"reminder": true, "spec": "at 18:00", "text": "купить хлеб"}\n'
    '«в 5 классе мы учили алгебру» → {"reminder": false}\n'
    '«what can you remind me about?» → {"reminder": false}'
)


@dataclass(slots=True)
class _ModelAnswer:
    """The parsed JSON of the intent call."""

    reminder: bool
    spec: str
    text: str


def looks_like_reminder(text: str, limit: int | None = None) -> bool:
    """Cheap check: reminder words or a schedule phrase at the message start.

    Only the first ``limit`` characters are inspected (default
    ``REMINDER_PREFILTER_CHARS``); a trigger further in is not a request.
    """
    if not text:
        return False
    head = text[: limit or config.reminder_prefilter_chars]
    return bool(_TRIGGER_WORDS.search(head) or _SCHEDULE_PHRASE.search(head))


def _extract_json(answer: str) -> _ModelAnswer | None:
    """Parse the model's JSON response (tolerating code fences around it)."""
    start, end = answer.find("{"), answer.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(answer[start : end + 1])
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    return _ModelAnswer(
        reminder=bool(data.get("reminder")),
        spec=str(data.get("spec") or ""),
        text=str(data.get("text") or ""),
    )


def _could_not_parse_time() -> str:
    """Reply when the model flagged a reminder but its slot failed validation."""
    return (
        "⚠️ <b>I understood that you want a reminder</b>, but could not read "
        "a usable time from your request.\n\n"
        "Examples: <code>remind me in 30 minutes to check the oven</code>, "
        "<code>remind me tomorrow at 9:00 to call mom</code> — "
        "or use <code>/remind</code>."
    )


async def extract_freeform(
    text: str,
    model: str,
    now: datetime.datetime | None = None,
    tz: datetime.tzinfo | None = None,
) -> ParsedReminder | str | None:
    """Turn a free-form message into a reminder, or decide it is not one.

    Returns a :class:`ParsedReminder` when a valid reminder was recognised, a
    user-facing error string when the user clearly asked for a reminder but
    the recognised time failed validation, and ``None`` when the message is
    not a reminder request (including all AI and parsing failures — the
    caller should continue the chat flow).

    ``tz`` is the reminder owner's zone (personal ``/timezone`` override or
    the global ``REMINDER_TIMEZONE``): relative phrases like «завтра» are
    interpreted against it, and the created reminder follows it.
    """
    now = now or datetime.datetime.now(datetime.UTC)
    tz = tz or get_timezone()
    local_now = now.astimezone(tz)
    context = (
        f"Current local time in the {tz} timezone: "
        f"{local_now:%A, %Y-%m-%d %H:%M}\n"
    )
    try:
        result = await ai_client.chat_completion_result(
            messages=[
                {"role": "system", "content": INTENT_SYSTEM_PROMPT},
                {"role": "user", "content": context + text},
            ],
            model=model,
            temperature=0.0,
            max_tokens=300,
        )
    except Exception:
        logger.warning("Reminder intent call failed; not a reminder", exc_info=True)
        return None

    answer = _extract_json(result.content or "")
    if answer is None or not answer.reminder:
        return None

    spec = answer.spec.strip()
    if not spec:
        return _could_not_parse_time()
    # The reminder text is anything the model put after the slot: the plain
    # reminders parser then applies every rule it applies for /remind —
    # including the recurring intervals floor, the ~10-year ceiling and the
    # "ai:" prefix handling.
    parsed = parse_reminder(
        f"{spec} {answer.text.strip()}".strip(), now=now, tz=tz
    )
    if isinstance(parsed, str):
        # The model invented a time we cannot represent; rephrase politely.
        return _could_not_parse_time()
    return parsed
