"""Команды samo_helper: Redmine → группа, YouGile и поиск по саммари.

Ответы — MarkdownV2 и всегда с явным parse_mode: Kapum использует HTML по
умолчанию, поэтому нельзя полагаться на глобальную настройку хоста.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

import aiohttp
from aiogram import F, Router
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramAPIError
from aiogram.filters import Command
from aiogram.types import CallbackQuery, Message

from .config import SamoHelperConfig
from .context import SamoAccessMiddleware
from .keyboards import REMIND_HOURS
from .knowledge import KnowledgeClient, KnowledgeError
from .md import bold, md_escape, sources_block, split_message, with_part_headers
from .redmine import RedmineClient
from .storage import Storage
from .summaries import get_issue_summary, plain_sources
from .yougile import YouGileClient

log = logging.getLogger("kapum.samo_helper.handlers")

SAMO_HELP_TEXT = "\n".join([
    bold("samo_helper — Redmine ↔ YouGile"),
    md_escape("/samo_help — эта помощь"),
    md_escape("/samo_add <текст> — перенос в YouGile"),
    md_escape("/samo_find <запрос> — поиск по саммари и ключевым словам"),
    md_escape("/samo_check — проверка Redmine и рабочей БД"),
    md_escape("/samo_status — последние опросы"),
    md_escape("/samo_stat — статистика рабочих событий"),
    md_escape("/samo_force — принудительный опрос Redmine"),
    md_escape("/samo_ygtest — тестовая карточка YouGile"),
    md_escape("Уведомления приходят в настроенную группу, напоминания — в личку."),
])


async def answer_chunks(message: Message, text: str) -> None:
    for part in with_part_headers(split_message(text)):
        await message.answer(part, parse_mode=ParseMode.MARKDOWN_V2)


async def cmd_samo_help(message: Message) -> None:
    await answer_chunks(message, SAMO_HELP_TEXT)


async def cb_remind(callback: CallbackQuery, db: Storage) -> None:
    """Поставить/сдвинуть личное напоминание по кнопке уведомления."""
    try:
        data = (callback.data or "").removeprefix("samo:").removeprefix("work:")
        prefix, raw_hours = data.split(":")
        hours = int(raw_hours) if prefix == "rm" else -1
    except (ValueError, IndexError):
        hours = -1
    if hours not in REMIND_HOURS:
        await callback.answer("Некорректный вариант напоминания", show_alert=True)
        return
    source = (callback.message.html_text if callback.message else "") or ""
    if not source.strip():
        await callback.answer("Нечего напоминать", show_alert=True)
        return
    is_new = db.upsert_reminder(
        callback.message.chat.id,
        callback.message.message_id,
        source,
        int(time.time()) + hours * 3600,
        fmt="html",
    )
    verb = "напомню" if is_new else "сдвинул напоминание"
    await callback.answer(f"Ок, {verb} через {hours} ч", show_alert=True)


async def cb_done(callback: CallbackQuery) -> None:
    deleted = False
    if callback.message is not None:
        try:
            deleted = bool(await callback.message.delete())
        except TelegramAPIError:
            deleted = False
    await callback.answer(
        "Удалено" if deleted else "Не удалось удалить сообщение",
        show_alert=not deleted,
    )


async def cmd_check(
    message: Message, db: Storage, redmine: RedmineClient, config: SamoHelperConfig,
) -> None:
    lines = ["🔎 " + bold("Проверка Redmine")]
    timeout = aiohttp.ClientTimeout(total=15)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        started = time.perf_counter()
        try:
            me_id = await redmine.current_user_id(session)
            ms = round((time.perf_counter() - started) * 1000)
            lines.append(md_escape(
                f"Связь сейчас: успешно ({ms} мс), мой ID в Redmine: {me_id}"
            ))
        except Exception as error:  # noqa: BLE001 - диагностика не должна падать
            lines.append(md_escape(
                f"Связь сейчас: неуспешно — {type(error).__name__}: {error}"
            ))
    try:
        if db.check_writable():
            lines.append(md_escape("Рабочая БД (SQLite): запись OK"))
        else:
            lines.append(md_escape(
                f"Рабочая БД: НЕТ ЗАПИСИ в {config.db_path} — проверь права!"
            ))
        last_at = db.get_meta("last_poll_at")
        last_status = db.get_meta("last_poll_status")
        if last_at is None:
            lines.append(md_escape("Последний запрос: ещё не выполнялся"))
        else:
            lines.append(md_escape(
                f"Последний запрос: {last_at.replace('T', ' ')} — {last_status or '?'}"
            ))
    except Exception as error:  # noqa: BLE001
        lines.append(md_escape(
            f"Рабочая БД: ошибка обращения — {type(error).__name__}: {error}"
        ))
    lines.append(md_escape(f"Интервал опроса: {config.redmine_poll_minutes} мин"))
    await answer_chunks(message, "\n".join(lines))


# Ссылка на Redmine: URL с /issues/<id> или упоминание #<id>.
_ISSUE_RE = re.compile(r"(?:issues/|#)(\d{2,})")


def _find_issue_id(text: str) -> int | None:
    match = _ISSUE_RE.search(text)
    return int(match.group(1)) if match else None


def _extract_issue_id(message: Message) -> int | None:
    """ID задачи из текста ИЛИ из URL сущностей (скрытые text_link)."""
    entities = list(message.entities or [])
    entities += list(message.caption_entities or [])
    for entity in entities:
        url = getattr(entity, "url", None)
        if url:
            found = _find_issue_id(url)
            if found is not None:
                return found
    for source in (message.text or "", message.caption or ""):
        found = _find_issue_id(source)
        if found is not None:
            return found
    return None


def _summary_tail(summary: str, sources=()) -> str:
    if not summary:
        return ""
    tail = "\n\n🤖 " + bold("Кратко") + ": " + md_escape(summary)
    return tail + sources_block([(source.title, source.url) for source in sources])


async def cmd_ygtest(message: Message, yougile: YouGileClient | None = None) -> None:
    if yougile is None:
        await answer_chunks(message, md_escape(
            "YouGile не настроен: заполните SAMO_HELPER_YOUGILE_*."
        ))
        return
    timeout = aiohttp.ClientTimeout(total=30)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            user = await yougile.me(session)
            had_cache = yougile.column_id is not None
            await yougile.ensure_column(session)
            ids = yougile.ids_for_report()
            task = await yougile.create_task(
                session, "Тест samo_helper — можно удалить",
                "Создано командой /samo_ygtest",
            )
            task_id = str(task["id"])
            await yougile.add_comment(session, task_id, "Тестовый комментарий /samo_ygtest")
        await answer_chunks(
            message,
            "🧪 YouGile OK: " + md_escape(f"вошёл как {user.get('realName', '?')}")
            + "\n" + md_escape(
                f"проект {ids['project']} · доска {ids['board']} · колонка {ids['column']}"
            )
            + "\n" + md_escape(
                f"тестовая карточка #{task_id} — можно удалить вручную"
                + (" (columnId из кэша)" if had_cache else " (цепочка собрана заново)")
            ),
        )
    except Exception as error:  # noqa: BLE001 - диагностическая команда
        await answer_chunks(
            message, "🧪 YouGile НЕOK: " + md_escape(f"{type(error).__name__}: {error}")
        )


async def cmd_status(message: Message, db: Storage) -> None:
    rows = db.last_polls(10)
    lines = [bold("📊 Последние опросы Redmine")]
    if not rows:
        lines.append(md_escape("Опросов ещё не было"))
    for row in rows:
        mark = "✅" if row["status"] == "успешно" else "❌"
        lines.append(md_escape(
            f"{mark} {row['poll_at'].replace('T', ' ')} — {row['status']} "
            f"· новых: {row['new_count']} · обновлённых: {row['updated_count']}"
        ))
    await answer_chunks(message, "\n".join(lines))


async def cmd_force(message: Message, poller: Any = None) -> None:
    if poller is None:
        await answer_chunks(message, md_escape("Поллер недоступен"))
        return
    await message.answer("🔄 Опрашиваю Redmine…", parse_mode=ParseMode.MARKDOWN_V2)
    result = await poller.force_poll()
    if result["ok"]:
        await answer_chunks(
            message,
            "✅ Принудительный опрос: "
            + md_escape(f"новых: {result['new']} · обновлённых: {result['updated']}"),
        )
    else:
        await answer_chunks(
            message, "❌ " + md_escape(f"Опрос не удался: {result['error']}")
        )


STAT_WINDOWS = (
    ("последний час", 3600),
    ("последние 12 часов", 12 * 3600),
    ("последние сутки", 24 * 3600),
    ("последние 7 суток", 7 * 86400),
)


async def cmd_stat(message: Message, db: Storage) -> None:
    now = int(time.time())
    lines = [bold("📊 Статистика samo_helper")]
    for label, seconds in STAT_WINDOWS:
        stats = db.event_stats(now - seconds)
        new_n, _ = stats.get("redmine_new", (0, 0))
        upd_n, _ = stats.get("redmine_upd", (0, 0))
        created, _ = stats.get("yg_created", (0, 0))
        updated, _ = stats.get("yg_updated", (0, 0))
        ai_n, ai_tokens = stats.get("ai_request", (0, 0))
        lines.append(bold(f"За {label}:"))
        lines.append(md_escape(
            f"Redmine: обработано {new_n + upd_n} (новых {new_n}, обновлено {upd_n})"
        ))
        lines.append(md_escape(f"YouGile: создано {created}, изменено {updated}"))
        lines.append(md_escape(f"AI: запросов {ai_n}, токенов {ai_tokens}"))
    await answer_chunks(message, "\n".join(lines))


async def cmd_task(
    message: Message, db: Storage, config: SamoHelperConfig,
    yougile: YouGileClient | None = None,
    redmine: RedmineClient | None = None, llm=None,
    knowledge: KnowledgeClient | None = None,
) -> None:
    source = message.text or message.caption or ""
    parts = source.split(maxsplit=1)
    if len(parts) < 2 or not parts[1].strip():
        await answer_chunks(
            message, md_escape("Используй /samo_add <текст или ссылка на Redmine>")
        )
        return
    await owner_message(
        message, db, config, yougile, redmine, llm,
        task_text=parts[1], knowledge=knowledge,
    )


async def cmd_find(
    message: Message, config: SamoHelperConfig,
    knowledge: KnowledgeClient | None = None,
) -> None:
    """Поиск только по саммари/ключевым словам; ссылки — из ответа сервиса."""
    source = message.text or message.caption or ""
    parts = source.split(maxsplit=1)
    query = parts[1].strip() if len(parts) > 1 else ""
    if not query:
        await answer_chunks(message, md_escape("Используй /samo_find <запрос>"))
        return
    if knowledge is None:
        await answer_chunks(message, md_escape(
            "Сервис знаний не настроен: заполните SAMO_HELPER_KNOWLEDGE_URL."
        ))
        return
    try:
        async with aiohttp.ClientSession() as session:
            hits = await knowledge.search(
                session, query, limit=config.knowledge_max_results,
            )
    except KnowledgeError as error:
        await answer_chunks(message, "🔎 " + md_escape(f"Поиск недоступен: {error}"))
        return
    if not hits:
        await answer_chunks(message, md_escape(
            "По саммари и ключевым словам ничего не найдено."
        ))
        return
    lines = [bold(f"🔎 Саммари по запросу: {query}")]
    for number, hit in enumerate(hits, 1):
        title = f"{number}. {hit.title}"
        if hit.stale:
            title += " (устарело)"
        lines.append(bold(title))
        lines.append(md_escape(hit.summary))
        if hit.keywords:
            lines.append(md_escape("Ключевые слова: " + ", ".join(hit.keywords)))
        if hit.sources:
            lines.append(sources_block([(source.title, source.url) for source in hit.sources]))
        else:
            lines.append(md_escape(
                "Источники не указаны — не использовать как подтверждение."
            ))
        lines.append("")
    await answer_chunks(message, "\n".join(lines).rstrip())


async def owner_message(
    message: Message,
    db: Storage,
    config: SamoHelperConfig,
    yougile: YouGileClient | None = None,
    redmine: RedmineClient | None = None,
    llm=None,
    knowledge: KnowledgeClient | None = None,
    *,
    task_text: str | None = None,
) -> None:
    text = task_text if task_text is not None else (message.text or message.caption or "")
    if yougile is None:
        await answer_chunks(
            message,
            "Принято: " + md_escape(text) + "\n"
            + md_escape("(перенос в YouGile не настроен — см. SAMO_HELPER_YOUGILE_*)"),
        )
        return

    issue_id = _extract_issue_id(message)
    timeout = aiohttp.ClientTimeout(total=60)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            summary = ""
            summary_sources = ()
            customer = ""
            title = (
                text.strip().splitlines()[0][:200]
                if text.strip() else "Сообщение из Telegram"
            )
            redmine_url = ""
            if issue_id is not None and redmine is not None:
                issue = await redmine.issue_with_journals(session, issue_id)
                redmine_url = redmine.issue_url(issue_id)
                customer = redmine.customer_name(issue)
                title = f"{redmine_url} {issue.get('subject') or ''}".strip() or title
                note = f"Важное примечание от разработчика:\n{text}".strip()
                result = await get_issue_summary(
                    session, knowledge=knowledge, llm=llm,
                    redmine=redmine, issue=issue, note=note,
                )
                if result.error:
                    log.warning("Саммари для переноса #%s: %s", issue_id, result.error)
                summary = result.text
                summary_sources = result.sources

            linked = db.get_yougile_task(issue_id) if issue_id is not None else None
            if linked is not None and await yougile.task_exists(session, linked["task_id"]):
                comment = text if not summary else text + "\n\nРезюме задачи: " + summary
                await yougile.add_comment(session, linked["task_id"], comment)
                db.log_event("yg_updated")
                await answer_chunks(
                    message,
                    "✏️ " + md_escape(
                        f"Обновил карточку в YouGile (по Redmine-задаче #{issue_id})"
                    ) + _summary_tail(summary, summary_sources),
                )
                return

            parts = []
            if summary:
                parts.append(f"🤖 Резюме: {summary}")
            source_text = plain_sources(summary_sources)
            if source_text:
                parts.append(f"Источники саммари:\n{source_text}")
            if customer:
                parts.append(f"Заказчик: {customer}")
            if redmine_url:
                parts.append(f"Redmine: {redmine_url}")
            parts.append(f"Сообщение из Telegram:\n{text}")
            task = await yougile.create_task(session, title, "\n\n".join(parts))
            task_id = str(task["id"])
            db.log_event("yg_created")
            if issue_id is not None:
                db.link_yougile_task(issue_id, task_id)

            if linked is not None:
                reply = md_escape(
                    f"✳️ Воскресил карточку в YouGile (по Redmine-задаче #{issue_id}) — "
                    "старая была удалена"
                )
            else:
                reply = md_escape("🆕 Новая карточка в YouGile")
                if issue_id is not None:
                    reply += "\n" + md_escape(
                        f"Связал с Redmine #{issue_id}: ответы пойдут в эту карточку"
                    )
            await answer_chunks(message, reply + _summary_tail(summary, summary_sources))
    except Exception as error:  # noqa: BLE001 - ошибка не должна ронять бота
        await answer_chunks(
            message,
            "Не удалось перенести в YouGile: "
            + md_escape(f"{type(error).__name__}: {error}"),
        )


def _is_samo_command(message: Message) -> bool:
    parts = (message.text or message.caption or "").split(maxsplit=1)
    return bool(parts and parts[0].split("@", 1)[0].startswith("/samo_"))


def create_samo_router(config: SamoHelperConfig, *, legacy_commands: bool = False) -> Router:
    router = Router(name="samo_helper")
    access = SamoAccessMiddleware(config, legacy_commands)
    router.message.middleware(access)
    router.callback_query.middleware(access)
    for command, legacy, handler in (
        ("samo_help", None, cmd_samo_help),
        ("samo_check", "check", cmd_check),
        ("samo_status", "status", cmd_status),
        ("samo_force", "force", cmd_force),
        ("samo_stat", "stat", cmd_stat),
        ("samo_ygtest", "ygtest", cmd_ygtest),
        ("samo_add", "task", cmd_task),
        ("samo_find", None, cmd_find),
    ):
        commands = [command, legacy] if legacy_commands and legacy else [command]
        router.message.register(handler, Command(*commands))
    # Опечатки в namespace не должны уходить в catch-all ИИ Kapum.
    router.message.register(cmd_samo_help, _is_samo_command)

    reminder_filter = F.data.startswith("samo:rm:")
    done_filter = F.data == "samo:done"
    if legacy_commands:
        reminder_filter = reminder_filter | F.data.startswith(("rm:", "work:rm:"))
        done_filter = done_filter | F.data.in_({"done", "work:done"})
    router.callback_query.register(cb_remind, reminder_filter)
    router.callback_query.register(cb_done, done_filter)
    return router
