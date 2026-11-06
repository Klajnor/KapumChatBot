"""Фоновый опрос Redmine: новые задачи и ответы по ним → отдельная группа.

Первый запуск (база только создана) — фиксируем текущее состояние как baseline
и ничего не рассылаем: бот «начинает с сейчас».
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any

import aiohttp
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramAPIError

from .config import SamoHelperConfig
from .keyboards import notify_keyboard
from .knowledge import KnowledgeClient, KnowledgeSource
from .md import bold, link, md_escape, sources_block, split_message, with_part_headers
from .redmine import RedmineClient
from .storage import Storage
from .summaries import get_issue_summary, plain_sources

log = logging.getLogger("kapum.samo_helper.poller")


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(value)


class RedminePoller:
    def __init__(
        self,
        bot: Any,
        config: SamoHelperConfig,
        storage: Storage,
        redmine: RedmineClient,
        llm: Any | None = None,
        yougile: Any | None = None,
        knowledge: KnowledgeClient | None = None,
    ) -> None:
        self._bot = bot
        self._cfg = config
        self._db = storage
        self._redmine = redmine
        self._llm = llm  # LLMClient | None — legacy-резюме без сервиса знаний
        self._yougile = yougile  # YouGileClient | None — ответы по задаче → в карточку
        self._knowledge = knowledge  # канонический источник саммари
        self._stop = asyncio.Event()
        self._lock = asyncio.Lock()  # /force и таймер не должны опрашивать параллельно
        self._task: asyncio.Task | None = None

    # --- жизненный цикл ---

    async def start(self) -> None:
        if self._task is not None:
            return
        self._stop.clear()
        self._task = asyncio.create_task(self._run(), name="redmine-poller")

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            try:
                await self._task
            finally:
                self._task = None

    async def _run(self) -> None:
        timeout = aiohttp.ClientTimeout(total=30)
        interval = self._cfg.redmine_poll_minutes * 60
        async with aiohttp.ClientSession(timeout=timeout) as session:
            while not self._stop.is_set():
                try:
                    await self._attempt(session)
                except Exception:
                    # падает даже запись истории (например, БД только на чтение)
                    log.exception("Опрос Redmine: ошибка службы журнала")
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=interval)
                except TimeoutError:
                    pass

    async def force_poll(self) -> dict[str, Any]:
        """Принудительный опрос Redmine (команда /force); ждёт освобождения блокировки."""
        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            return await self._attempt(session)

    async def _attempt(self, session: aiohttp.ClientSession) -> dict[str, Any]:
        """Один цикл опроса; время, итог и счётчики пишутся в БД (для /check и /status)."""
        now = datetime.now().isoformat(timespec="seconds")
        async with self._lock:
            try:
                stats = await self.poll_once(session)
                self._db.set_meta("last_poll_at", now)
                self._db.set_meta("last_poll_status", "успешно")
                self._db.log_poll(now, "успешно", stats["new"], stats["updated"])
                return {"ok": True, **stats}
            except Exception as e:
                log.exception("Ошибка при опросе Redmine")
                status = f"неуспешно: {type(e).__name__}: {e}"[:200]
                self._db.set_meta("last_poll_at", now)
                self._db.set_meta("last_poll_status", status)
                self._db.log_poll(now, status, 0, 0)
                return {"ok": False, "error": f"{type(e).__name__}: {e}",
                        "new": 0, "updated": 0}

    # --- логика опроса ---

    async def poll_once(self, session: aiohttp.ClientSession) -> dict[str, int]:
        """Опрос; возвращает счётчики: новых уведомлений и обновлённых задач."""
        first_run = self._db.is_first_run()
        issues = await self._redmine.assigned_issues(session)
        me_id = await self._redmine.current_user_id(session)
        stats = {"new": 0, "updated": 0}

        if first_run:
            log.info(
                "Первый запуск: baseline из %d задач(ий), уведомления не рассылаются",
                len(issues),
            )

        for raw in issues:
            stored = self._db.get_issue(raw["id"])
            if stored is None:
                # Один запрос: journals для baseline + текст для резюме.
                issue_full = await self._redmine.issue_with_journals(session, raw["id"])
                baseline_jid = max(
                    (j["id"] for j in issue_full.get("journals") or []), default=0
                )
                self._db.save_issue(raw["id"], raw["updated_on"], baseline_jid)
                if not first_run:
                    stats["new"] += 1
                    self._db.log_event("redmine_new")
                    await self._notify_new_issue(raw, session, issue_full)
            elif _dt(raw["updated_on"]) > _dt(stored["updated_on"]):
                stats["updated"] += 1
                self._db.log_event("redmine_upd")
                await self._check_replies(session, raw, stored, me_id)

        if first_run:
            self._db.mark_initialized()
        return stats

    async def _check_replies(
        self,
        session: aiohttp.ClientSession,
        raw: dict[str, Any],
        stored: Any,
        me_id: int,
    ) -> None:
        journals = await self._redmine.issue_journals(session, raw["id"])
        last_id = max([stored["last_journal_id"]] + [j["id"] for j in journals])
        pending = [
            j for j in journals
            if j["id"] > stored["last_journal_id"]
            and j.get("user", {}).get("id") != me_id
            and (j.get("notes") or "").strip()
        ]
        # Watermark пишем ДО рассылки: одно событие -> максимум одно уведомление,
        # при краше/ошибке отправки повтор не придёт.
        self._db.save_issue(raw["id"], raw["updated_on"], last_id)
        if not pending:
            return  # свои/безтекстовые правки — молча двигаем watermark
        await self._notify(self._reply_text(raw, pending))
        await self._sync_yougile(session, raw, pending)

    async def _ensure_yougile_card(
        self, session: aiohttp.ClientSession, raw: dict[str, Any], summary: str = "",
        summary_sources: tuple[KnowledgeSource, ...] = (),
    ) -> str | None:
        """Карточка для задачи: существующая живая, иначе создание/восстановление."""
        if self._yougile is None:
            return None
        url = self._redmine.issue_url(raw["id"])
        link = self._db.get_yougile_task(raw["id"])
        if link is not None and await self._yougile.task_exists(session, link["task_id"]):
            return link["task_id"]
        parts = []
        if summary:
            parts.append(f"🤖 Резюме: {summary}")
        source_text = plain_sources(summary_sources)
        if source_text:
            parts.append(f"Источники саммари:\n{source_text}")
        customer = self._redmine.customer_name(raw)
        if customer:
            parts.append(f"Заказчик: {customer}")
        parts.append(f"Redmine: {url}")
        task = await self._yougile.create_task(
            session, f"{url} {raw.get('subject', '')}".strip(), "\n\n".join(parts)
        )
        task_id = str(task["id"])
        self._db.link_yougile_task(raw["id"], task_id)
        self._db.log_event("yg_created")
        log.info(
            "YouGile: %s карточка задачи #%s (%s)",
            "восстановил" if link is not None else "создал", raw["id"], task_id,
        )
        return task_id

    async def _sync_yougile(
        self, session: aiohttp.ClientSession, raw: dict[str, Any],
        pending: list[dict[str, Any]],
    ) -> None:
        """Ответы из Redmine -> один комментарий в карточку (создав её при нужде)."""
        if self._yougile is None:
            return
        comment = "\n\n".join(
            f"{j['user']['name']}: {(j['notes'] or '').strip()}" for j in pending
        )
        try:
            task_id = await self._ensure_yougile_card(session, raw)
            if task_id is None:
                return
            await self._yougile.add_comment(session, task_id, f"Ответы в Redmine:\n{comment}")
            self._db.log_event("yg_updated")
        except Exception as e:  # noqa: BLE001 - сбой карточки не должен ронять опрос
            log.warning("Комментарий в карточку YouGile по задаче #%s не добавлен: %s",
                        raw["id"], e)

    # --- тексты уведомлений ---

    async def _notify_new_issue(
        self,
        raw: dict[str, Any],
        session: aiohttp.ClientSession,
        issue_full: dict[str, Any] | None = None,
    ) -> None:
        text = self._new_issue_text(raw)
        full = {**raw, **(issue_full or {})}
        result = await get_issue_summary(
            session, knowledge=self._knowledge, llm=self._llm,
            redmine=self._redmine, issue=full,
        )
        if result.error:
            log.warning("Саммари задачи #%s недоступно: %s", raw["id"], result.error)
        summary = result.text
        issue_url = self._redmine.issue_url(raw["id"])
        extra_sources = [source for source in result.sources if source.url != issue_url]
        if summary:
            text += "\n\n🤖 " + bold("Кратко") + ": " + md_escape(summary)
            text += sources_block(
                [(source.title, source.url) for source in extra_sources],
                heading="Источники саммари",
            )
        await self._notify(text)
        # По уведомлению о новой задаче тоже заводим карточку YouGile.
        try:
            await self._ensure_yougile_card(
                session, full, summary, result.sources,
            )
        except Exception as e:  # noqa: BLE001 - карточка не должна блокировать уведомление
            log.warning("Карточка YouGile для новой задачи #%s не создана: %s",
                        raw["id"], e)

    def _new_issue_text(self, raw: dict[str, Any]) -> str:
        tracker = raw.get("tracker", {}).get("name", "?")
        status = raw.get("status", {}).get("name", "?")
        priority = raw.get("priority", {}).get("name", "?")
        project = raw.get("project", {}).get("name", "?")
        url = self._redmine.issue_url(raw["id"])
        return "\n".join(
            [
                "🆕 " + bold(f"Новая задача #{raw['id']} [{tracker}]"),
                bold(raw.get("subject", "")),
                md_escape(
                    f"Проект: {project} · Статус: {status} · Приоритет: {priority}"
                ),
                # Ссылка = сам видимый URL (кликабельный и на телефонном экране).
                link(url, url),
            ]
        )

    def _reply_text(self, raw: dict[str, Any], pending: list[dict[str, Any]]) -> str:
        # Одно уведомление на задачу за цикл: все новые ответы — в одном сообщении.
        # Ссылка (видимый URL) — сразу после заголовка: при разбиении на части
        # она попадёт в первое сообщение.
        url = self._redmine.issue_url(raw["id"])
        lines = [
            "💬 " + bold(f"Ответы по задаче #{raw['id']}"),
            bold(raw.get("subject", "")),
            link(url, url),
            "",
        ]
        for j in pending:
            lines.append(f"{md_escape(j['user']['name'])}:\n{md_escape((j['notes'] or '').strip())}")
            lines.append("")
        return "\n".join(lines).rstrip()

    # --- отправка ---

    async def _notify(self, text: str) -> None:
        kb = notify_keyboard()
        for part in with_part_headers(split_message(text)):
            try:
                await self._bot.send_message(
                    self._cfg.notification_chat_id, part, reply_markup=kb,
                    parse_mode=ParseMode.MARKDOWN_V2,
                )
            except TelegramAPIError:
                log.exception("Не удалось отправить часть уведомления")
