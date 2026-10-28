"""Daily channel digest: collect posts, summarise them, report token usage.

Channels are picked in ``/admin → Digest``. While a channel is on the list the
bot stores the text of every post it sees there (the bot must be an admin of
the channel; for a group it needs access to the messages). Once a day, at
``DIGEST_TIME`` in ``REMINDER_TIMEZONE``, every channel is processed:

1. the window is everything since the previous summary (first run: 24 hours);
2. when the window holds fewer than ``DIGEST_MIN_MESSAGES`` messages, up to
   ``DIGEST_CONTEXT_MESSAGES`` older ones are added as background;
3. the model chosen in the admin menu summarises them with the admin-defined
   prompt, and the summary is posted to the channel itself;
4. finally each admin gets the input/output token counts per channel.

The model and the prompt live in the ``app_settings`` table, so they change
without a restart. Channel text is third-party data: it is framed with
``<untrusted_content>`` like every other external source.
"""

from __future__ import annotations

import asyncio
import datetime
import html
import logging
from dataclasses import dataclass, field

from aiogram.exceptions import TelegramAPIError

from bot.bot import bot
from bot.config import config
from bot.database.repository import DigestRepository
from bot.database.session import async_session_factory
from bot.services.ai_client import ChatResult, ai_client
from bot.services.reminders import _send_html, _send_markdown, get_timezone
from bot.services.rich import send_rich_to
from bot.services.untrusted import apply_untrusted_hint, wrap_untrusted

logger = logging.getLogger(__name__)

SETTING_MODEL = "digest_model"
SETTING_PROMPT = "digest_prompt"
SETTING_MODIFIER = "digest_modifier"
SETTING_TEMPERATURE = "digest_temperature"
SETTING_VARY = "digest_vary"  # "1" / "0"; overrides DIGEST_VARY_PROMPT
SETTING_TIME = "digest_time"  # "HH:MM"; overrides DIGEST_TIME
SETTING_LAST_RUN = "digest_last_run"  # local date (ISO) of the last scheduled run

DEFAULT_DIGEST_PROMPT = (
    "Ты — редактор-аналитик. Тебе дают сообщения из Telegram-канала за сутки. "
    "Составь краткое саммари на русском языке: главные темы и события, важные "
    "решения, ссылки и цифры, если они есть. Структурируй ответ короткими "
    "пунктами. Не выдумывай того, чего нет в сообщениях. Блок «Контекст» — это "
    "более ранние сообщения: используй его только для понимания, не пересказывай."
)

DEFAULT_MODIFIER_PROMPT = (
    "Перепиши приведённый ниже промпт другими словами, сохранив суть и общие правила: "
    "язык ответа, формат, ограничения и порядок работы. Не добавляй новых требований "
    "и не убирай существующие. Каждый раз выбирай другие формулировки, порядок "
    "предложений и тон. Верни только текст нового промпта — без пояснений, заголовков "
    "и кавычек."
)
# A reworded prompt outside these bounds (characters) is discarded as broken.
_MIN_REWORDED = 40
_MAX_REWORDED = 4000

# A message stored for the summary is clipped, so one huge post cannot eat the budget.
_MAX_MESSAGE_CHARS = 1500
# Rows older than this are pruned (the newest ``_KEEP_LATEST`` always stay).
_RETENTION_DAYS = 7
_KEEP_LATEST = 100
_DEFAULT_WINDOW = datetime.timedelta(hours=24)

NO_MESSAGES = "no messages"  # ChannelReport.error: nothing stored for the channel

_run_lock = asyncio.Lock()
# chat ids currently on the digest list; ``None`` = not loaded yet.
_channel_ids: set[int] | None = None


@dataclass(slots=True)
class ChannelReport:
    """Outcome of one channel's run (what the admin statistics are built from)."""

    chat_id: int
    title: str
    messages: int = 0
    context: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None
    model: str = ""
    posted: bool = False
    error: str = ""
    recent_only: bool = False  # quiet period: summarised the latest stored messages
    test: bool = False  # a hand-made test request, not a channel
    first_at: datetime.datetime | None = None  # time of the oldest analysed message
    last_at: datetime.datetime | None = None  # time of the newest analysed message


@dataclass(slots=True)
class DigestResult:
    reports: list[ChannelReport] = field(default_factory=list)
    skipped: bool = False  # another run was in progress


# --- Settings -----------------------------------------------------------------


async def get_model() -> str:
    """The summary model (internal id); falls back to ``DEFAULT_MODEL``."""
    async with async_session_factory() as session:
        stored = await DigestRepository(session).get_setting(SETTING_MODEL)
    return config.resolve_model(stored)["id"] if stored else config.default_model


async def get_prompt() -> str:
    async with async_session_factory() as session:
        stored = await DigestRepository(session).get_setting(SETTING_PROMPT)
    return stored or DEFAULT_DIGEST_PROMPT


async def get_modifier() -> str:
    """The prompt that rewords the digest prompt before each run."""
    async with async_session_factory() as session:
        stored = await DigestRepository(session).get_setting(SETTING_MODIFIER)
    return stored or DEFAULT_MODIFIER_PROMPT


async def get_temperature() -> float:
    """Temperature of both digest requests: admin menu setting, else the env default."""
    async with async_session_factory() as session:
        stored = await DigestRepository(session).get_setting(SETTING_TEMPERATURE)
    parsed = parse_temperature(stored)
    return _clamp_temperature(config.digest_temperature) if parsed is None else parsed


async def get_vary() -> bool:
    """Whether the prompt is reworded before each run."""
    async with async_session_factory() as session:
        stored = await DigestRepository(session).get_setting(SETTING_VARY)
    return config.digest_vary_prompt if stored is None else stored == "1"


def _clamp_temperature(value: float) -> float:
    return max(0.0, min(2.0, value))


def parse_temperature(value: str | None) -> float | None:
    """``0``..``2`` (comma or dot) → temperature; ``None`` if malformed or out of range."""
    try:
        number = float((value or "").strip().replace(",", "."))
    except ValueError:
        return None
    return number if 0.0 <= number <= 2.0 else None


def parse_time(value: str | None) -> datetime.time | None:
    """``HH:MM`` (also ``H:MM``) → time of day; ``None`` when malformed."""
    try:
        hour, minute = (value or "").strip().split(":")
        return datetime.time(int(hour), int(minute))
    except ValueError:
        return None


async def get_time() -> datetime.time:
    """Local time of the daily run: admin menu setting, else ``DIGEST_TIME``."""
    async with async_session_factory() as session:
        stored = await DigestRepository(session).get_setting(SETTING_TIME)
    parsed = parse_time(stored) or parse_time(config.digest_time)
    if parsed is None:
        logger.warning("Invalid DIGEST_TIME %r; using 21:00", config.digest_time)
        return datetime.time(21, 0)
    return parsed


async def get_setting(key: str) -> str | None:
    async with async_session_factory() as session:
        return await DigestRepository(session).get_setting(key)


async def set_setting(key: str, value: str | None) -> None:
    async with async_session_factory() as session:
        await DigestRepository(session).set_setting(key, value)
        await session.commit()


# --- Channel list & message intake ----------------------------------------------


async def list_channels() -> list[tuple[int, str]]:
    async with async_session_factory() as session:
        rows = await DigestRepository(session).list_channels()
    return [(row.chat_id, row.title) for row in rows]


async def add_channel(chat_id: int, title: str, added_by: int | None) -> None:
    async with async_session_factory() as session:
        await DigestRepository(session).add_channel(chat_id, title, added_by)
        await session.commit()
    if _channel_ids is not None:
        _channel_ids.add(chat_id)


async def remove_channel(chat_id: int) -> bool:
    async with async_session_factory() as session:
        removed = await DigestRepository(session).remove_channel(chat_id)
        await session.commit()
    if _channel_ids is not None:
        _channel_ids.discard(chat_id)
    return removed


async def _tracked(chat_id: int) -> bool:
    global _channel_ids
    if _channel_ids is None:
        _channel_ids = {cid for cid, _ in await list_channels()}
    return chat_id in _channel_ids


def _utc(value: datetime.datetime) -> datetime.datetime:
    """SQLite returns naive datetimes (stored as UTC); make them aware."""
    return value if value.tzinfo is not None else value.replace(tzinfo=datetime.UTC)


async def record_message(
    chat_id: int,
    message_id: int,
    author: str,
    text: str | None,
    date: datetime.datetime,
    *,
    edited: bool = False,
) -> bool:
    """Store a post of a tracked channel. Returns True when it was stored."""
    text = (text or "").strip()
    if not text or not config.enable_digest or not await _tracked(chat_id):
        return False
    async with async_session_factory() as session:
        repo = DigestRepository(session)
        if edited:
            await repo.update_channel_message(chat_id, message_id, text)
            stored = False
        else:
            stored = await repo.add_channel_message(
                chat_id, message_id, author, text, _utc(date)
            )
        await session.commit()
    return stored


# --- Building the model input -----------------------------------------------------


def _format_line(author: str, created_at: datetime.datetime, text: str) -> str:
    local = _utc(created_at).astimezone(get_timezone())
    clipped = text if len(text) <= _MAX_MESSAGE_CHARS else text[:_MAX_MESSAGE_CHARS] + "…"
    who = f"{author}: " if author else ""
    return f"[{local:%d.%m %H:%M}] {who}{clipped}"


def build_input(
    messages: list, context: list, max_chars: int, *, recent_only: bool = False
) -> str:
    """Compose the user message; the oldest lines go first when over budget.

    ``recent_only`` marks a quiet period: ``messages`` are then the latest ones
    from before the window, and the model is told so.
    """
    main = [_format_line(m.author, m.created_at, m.text) for m in messages]
    background = [_format_line(m.author, m.created_at, m.text) for m in context]

    def size(lines: list[str]) -> int:
        return sum(len(line) + 1 for line in lines)

    # Keep the newest lines of the window; the context is the first to shrink.
    while background and size(main) + size(background) > max_chars:
        background.pop(0)
    while len(main) > 1 and size(main) > max_chars:
        main.pop(0)

    parts = []
    if background:
        parts.append("Контекст (более ранние сообщения):\n" + "\n".join(background))
    title = (
        "За период новых сообщений не было. Последние сообщения канала:"
        if recent_only
        else "Сообщения за период:"
    )
    parts.append(f"{title}\n" + "\n".join(main))
    return wrap_untrusted("channel", "\n\n".join(parts))


@dataclass(slots=True)
class Answer:
    """Summary text with the tokens of *all* requests that produced it."""

    content: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    model: str = ""
    prompt: str = ""  # the system prompt the summary was written with
    varied: bool = False  # the prompt was reworded by the modifier


def _add(a: int | None, b: int | None) -> int | None:
    return None if a is None and b is None else (a or 0) + (b or 0)


async def _vary_prompt(
    prompt: str, model: str, temperature: float
) -> tuple[str, ChatResult | None]:
    """Reword ``prompt`` with the modifier prompt; the original if that fails.

    Returns the prompt to use and the modifier's result (for its token usage).
    """
    modifier = await get_modifier()
    request = [
        {"role": "system", "content": modifier},
        {"role": "user", "content": prompt},
    ]
    try:
        result = await ai_client.chat_completion_result(
            request,
            model=model,
            temperature=temperature,
            max_tokens=config.digest_max_tokens,
        )
    except Exception:
        logger.exception("Prompt modifier request failed; using the prompt as is")
        return prompt, None
    reworded = (result.content or "").strip()
    if not _MIN_REWORDED <= len(reworded) <= max(_MAX_REWORDED, 2 * len(prompt)):
        logger.warning(
            "Prompt modifier returned %d characters; using the prompt as is", len(reworded)
        )
        return prompt, result
    logger.debug("Digest prompt reworded: %s", reworded)
    return reworded, result


async def _ask(text: str, model: str, prompt: str) -> Answer:
    """Summarise ``text``: reword the prompt first (when enabled), then ask.

    Two requests in that order — the modifier, then the reworded prompt with
    the messages — both with the digest temperature.
    """
    temperature = await get_temperature()
    used, modifier_result = prompt, None
    varied = await get_vary()
    if varied:
        used, modifier_result = await _vary_prompt(prompt, model, temperature)
        varied = used != prompt

    request = [
        {"role": "system", "content": used},
        {"role": "user", "content": text},
    ]
    apply_untrusted_hint(request)
    result = await ai_client.chat_completion_result(
        request,
        model=model,
        temperature=temperature,
        max_tokens=config.digest_max_tokens,
    )
    return Answer(
        content=result.content,
        input_tokens=_add(result.input_tokens, modifier_result and modifier_result.input_tokens),
        output_tokens=_add(
            result.output_tokens, modifier_result and modifier_result.output_tokens
        ),
        model=result.model or "",
        prompt=used,
        varied=varied,
    )


async def run_test(samples: list, admin_id: int) -> ChannelReport:
    """Summarise hand-picked messages (admin menu test) and send the result.

    ``samples`` are objects with ``author``, ``created_at`` and ``text``. The
    summary and the token statistics go to ``admin_id`` only; nothing is
    stored, posted to a channel, or marked as digested.
    """
    model, prompt = await get_model(), await get_prompt()
    report = ChannelReport(
        chat_id=0, title="Test", messages=len(samples), model=model, test=True
    )
    report.first_at, report.last_at = _span(samples)
    text = build_input(samples, [], config.digest_max_input_chars)
    result = await _ask(text, model, prompt)
    report.input_tokens, report.output_tokens = result.input_tokens, result.output_tokens
    if result.model:
        report.model = result.model
    if not result.content:
        report.error = "the model returned no answer"
        return report
    report.posted = await _deliver(admin_id, f"🧪 **Тестовое саммари**\n\n{result.content}")
    if result.varied:
        await show_prompt(admin_id, "🎲 Prompt used (reworded)", result.prompt)
    if not report.posted:
        report.error = "could not send the summary"
    await notify_admins(format_stats([report]), only=admin_id)
    return report


async def _deliver(chat_id: int, markdown: str) -> bool:
    """Post the summary as a rich message; HTML, then plain text, as fallbacks."""
    if await send_rich_to(bot, chat_id, markdown):
        return True
    return await _send_markdown(chat_id, markdown)


async def show_prompt(chat_id: int, title: str, prompt: str) -> bool:
    """Show a prompt in full, through the same rich → HTML → plain chain.

    The prompt can be longer than one HTML message (up to 4096 characters
    plus escaping), so it is not squeezed into the admin panel's screens.
    """
    return await _deliver(chat_id, f"**{title}**\n\n{prompt}")


# --- One channel -------------------------------------------------------------------


async def _digest_channel(
    chat_id: int, title: str, until: datetime.datetime, model: str, prompt: str
) -> ChannelReport:
    report = ChannelReport(chat_id=chat_id, title=title, model=model)
    async with async_session_factory() as session:
        repo = DigestRepository(session)
        channel = await repo.get_channel(chat_id)
        if channel is None:
            report.error = "channel was removed"
            return report
        since = _utc(channel.last_digest_at) if channel.last_digest_at else until - _DEFAULT_WINDOW
        messages = await repo.messages_between(chat_id, since, until)
        context = []
        if 0 < len(messages) < config.digest_min_messages and config.digest_context_messages > 0:
            context = await repo.messages_up_to(
                chat_id, since, config.digest_context_messages
            )
        if not messages and config.digest_context_messages > 0:
            # A quiet period still gets a summary: of the latest stored messages.
            messages = await repo.messages_up_to(
                chat_id, until, config.digest_context_messages
            )
            report.recent_only = bool(messages)
        if not messages:
            # Nothing stored at all: move the window so it does not grow forever.
            await repo.mark_digest(chat_id, until)
            await session.commit()
            report.error = NO_MESSAGES
            return report
        # Detach the rows before the (slow) model call.
        report.messages, report.context = len(messages), len(context)
        report.first_at, report.last_at = _span(context, messages)
        text = build_input(
            messages, context, config.digest_max_input_chars, recent_only=report.recent_only
        )

    result = await _ask(text, model, prompt)
    report.input_tokens, report.output_tokens = result.input_tokens, result.output_tokens
    if result.model:
        report.model = result.model
    if not result.content:
        report.error = "the model returned no answer"
        return report

    day = until.astimezone(get_timezone()).strftime("%d.%m.%Y")
    summary = f"📰 **Саммари за сутки · {day}**\n\n{result.content}"
    try:
        report.posted = await _deliver(chat_id, summary)
        if not report.posted:
            report.error = "could not post the summary"
    except TelegramAPIError as exc:
        report.error = f"could not post: {exc}"
    except Exception as exc:
        logger.exception("Digest delivery to %s failed", chat_id)
        report.error = f"could not post: {exc}"

    if report.posted:
        async with async_session_factory() as session:
            repo = DigestRepository(session)
            await repo.mark_digest(chat_id, until)
            cutoff = until - datetime.timedelta(days=_RETENTION_DAYS)
            await repo.prune_messages(chat_id, cutoff, _KEEP_LATEST)
            await session.commit()
    return report


# --- Statistics -----------------------------------------------------------------------


def format_tokens(value: int | None) -> str:
    return "n/a" if value is None else f"{value:,}".replace(",", " ")


def _span(*groups: list) -> tuple[datetime.datetime | None, datetime.datetime | None]:
    """Oldest and newest ``created_at`` over message lists (UTC), if any."""
    times = [_utc(m.created_at) for group in groups for m in group]
    return (min(times), max(times)) if times else (None, None)


def _stamp(value: datetime.datetime | None) -> str:
    if value is None:
        return "?"
    return _utc(value).astimezone(get_timezone()).strftime("%d.%m.%Y %H:%M")


NL = "\n"


def format_stats(reports: list[ChannelReport]) -> str:
    """Admin message: token usage per channel plus the total."""
    lines = ["📰 <b>Digest — token statistics</b>\n"]
    total_in = total_out = 0
    for r in reports:
        name = (
            "🧪 <b>Test request</b> (not a channel)"
            if r.test
            else f"📡 <b>{html.escape(r.title or str(r.chat_id))}</b>"
        )
        if r.error and not r.input_tokens and not r.output_tokens:
            lines.append(f"{name}{NL}   ⚠️ {html.escape(r.error)}")
            continue
        total_in += r.input_tokens or 0
        total_out += r.output_tokens or 0
        mark = "" if r.posted else " ⚠️ " + html.escape(r.error)
        context = f" (+{r.context} context)" if r.context else ""
        if r.recent_only:
            context = " (no new, latest)"
        period = (
            f"{NL}   🕒 {_stamp(r.first_at)} → {_stamp(r.last_at)}" if r.first_at else ""
        )
        lines.append(
            f"{name}{NL}"
            f"   📨 {r.messages} msg{context}{period}{NL}"
            f"   ↑ {format_tokens(r.input_tokens)} ↓ {format_tokens(r.output_tokens)} tok · "
            f"<code>{html.escape(r.model)}</code>{mark}"
        )
    if len([r for r in reports if r.input_tokens or r.output_tokens]) > 1:
        lines.append(f"\nΣ ↑ {format_tokens(total_in)} ↓ {format_tokens(total_out)} tok")
    return "\n".join(lines)


def admin_ids() -> list[int]:
    return sorted({*config.owner_ids, *config.extra_admin_ids})


async def notify_admins(text: str, only: int | None = None) -> None:
    targets = [only] if only is not None else admin_ids()
    for admin_id in targets:
        try:
            await _send_html(admin_id, text)
        except Exception as exc:  # noqa: BLE001 - an admin may never have started the bot
            logger.warning("Could not send digest statistics to %s: %s", admin_id, exc)


# --- Run ----------------------------------------------------------------------------------


async def run_digest(notify: int | None = None) -> DigestResult:
    """Process every channel once.

    ``notify`` limits the statistics to one admin (manual run from the menu);
    the scheduled run reports to all admins. Channels with no stored messages
    at all are left out of the statistics.
    """
    if _run_lock.locked():
        return DigestResult(skipped=True)
    async with _run_lock:
        until = datetime.datetime.now(datetime.UTC)
        model, prompt = await get_model(), await get_prompt()
        result = DigestResult()
        for chat_id, title in await list_channels():
            try:
                report = await _digest_channel(chat_id, title, until, model, prompt)
            except Exception as exc:
                logger.exception("Digest for channel %s failed", chat_id)
                report = ChannelReport(chat_id=chat_id, title=title, error=str(exc)[:200])
            result.reports.append(report)

        shown = [r for r in result.reports if r.error != NO_MESSAGES]
        if shown:
            await notify_admins(format_stats(shown), only=notify)
        return result


# --- Scheduler ----------------------------------------------------------------------------------


async def _scheduled_run_due() -> bool:
    """Claim today's scheduled run (stored before it starts, like reminders)."""
    now = datetime.datetime.now(datetime.UTC).astimezone(get_timezone())
    if now.time() < await get_time():
        return False
    today = now.date().isoformat()
    async with async_session_factory() as session:
        repo = DigestRepository(session)
        if await repo.get_setting(SETTING_LAST_RUN) == today:
            return False
        await repo.set_setting(SETTING_LAST_RUN, today)
        await session.commit()
    return True


async def digest_scheduler_loop() -> None:
    """Background task: run the digest once per local day at ``DIGEST_TIME``."""
    logger.info(
        "Digest scheduler started (daily at %s unless changed in the admin menu, "
        "timezone %s)",
        config.digest_time,
        config.reminder_timezone or "UTC",
    )
    while True:
        try:
            if await _scheduled_run_due():
                await run_digest()
        except asyncio.CancelledError:
            raise
        except Exception:  # the loop must survive any single failure
            logger.exception("Digest scheduler tick failed")
        await asyncio.sleep(60)
