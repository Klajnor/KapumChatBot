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
"""

from __future__ import annotations

import datetime
import json
import logging
import re
from dataclasses import dataclass

from bot.services.ai_client import ai_client
from bot.services.reminders import (
    ParsedReminder,
    get_timezone,
    parse_reminder,
)

logger = logging.getLogger(__name__)

# Cheap pre-filter. Only words that are unlikely in ordinary questions;
# a false positive merely costs one tiny AI call, a false negative loses the
# feature, so the net is cast wide (rus+eng).
_PREFILTER = re.compile(
    r"напом|будильник|разбуди|remind|reminder",
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
    "- If at fire time the user wants the bot to ask its AI model (news "
    "briefing, daily summary, birthday ideas...), prefix TEXT with 'ai: '.\n"
    "- TEXT is the reminder itself without any time phrase.\n\n"
    "Examples:\n"
    '«напомни завтра в 5 вечера позвонить маме» → '
    '{"reminder": true, "spec": "tomorrow at 17:00", "text": "позвонить маме"}\n'
    '«remind me in half an hour to check the oven» → '
    '{"reminder": true, "spec": "in 30 minutes", "text": "check the oven"}\n'
    '«every morning say hi» → '
    '{"reminder": true, "spec": "daily at 09:00", "text": "say hi"}\n'
    '«what can you remind me about?» → {"reminder": false}'
)


@dataclass(slots=True)
class _ModelAnswer:
    """The parsed JSON of the intent call."""

    reminder: bool
    spec: str
    text: str


def looks_like_reminder(text: str) -> bool:
    """Cheap check: does the message even contain reminder-like words?"""
    return bool(text and _PREFILTER.search(text))


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
