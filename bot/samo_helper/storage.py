"""SQLite-состояние бота: известные задачи Redmine, метки времени, мета."""

from __future__ import annotations

import sqlite3
import time
from datetime import UTC, datetime

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS issues (
    redmine_id      INTEGER PRIMARY KEY,
    updated_on      TEXT NOT NULL,
    last_journal_id INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS reminders (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id   INTEGER,
    message_id INTEGER,
    text      TEXT NOT NULL,
    remind_at INTEGER NOT NULL,
    fmt       TEXT NOT NULL DEFAULT 'md2'
);
CREATE TABLE IF NOT EXISTS yougile_tasks (
    redmine_id INTEGER PRIMARY KEY,
    task_id    TEXT NOT NULL,
    task_url   TEXT
);
CREATE TABLE IF NOT EXISTS poll_log (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    poll_at       TEXT NOT NULL,
    status        TEXT NOT NULL,
    new_count     INTEGER NOT NULL DEFAULT 0,
    updated_count INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS events (
    id     INTEGER PRIMARY KEY AUTOINCREMENT,
    kind   TEXT NOT NULL,
    ts     INTEGER NOT NULL,
    tokens INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS chat_history (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL,
    role    TEXT NOT NULL,
    content TEXT NOT NULL,
    ts      INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_chat_history_owner
    ON chat_history(chat_id, user_id, id);
"""

# Хранить события ~8 суток (окно /stat до 7 суток + запас)
_EVENTS_RETENTION_S = 8 * 86400


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


class Storage:
    def __init__(self, path: str) -> None:
        self._db = sqlite3.connect(path)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(_SCHEMA)
        # миграция старых баз: колонки fmt/chat_id/message_id появились позже
        for col, ddl in (
            ("fmt", "TEXT NOT NULL DEFAULT 'md2'"),
            ("chat_id", "INTEGER"),
            ("message_id", "INTEGER"),
        ):
            try:
                self._db.execute(f"ALTER TABLE reminders ADD COLUMN {col} {ddl}")
            except sqlite3.OperationalError:
                pass  # колонка уже есть
        # индекс — после миграции колонок (в legacy базе их сначала надо добавить)
        self._db.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS ux_reminders_msg "
            "ON reminders(chat_id, message_id) WHERE chat_id IS NOT NULL"
        )

    def close(self) -> None:
        self._db.close()

    # --- мета ---

    def get_meta(self, key: str) -> str | None:
        row = self._db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        self._db.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        self._db.commit()

    def is_first_run(self) -> bool:
        return self.get_meta("initialized") is None

    def mark_initialized(self) -> None:
        self.set_meta("initialized", _now_iso())

    # --- задачи Redmine ---

    def get_issue(self, redmine_id: int) -> sqlite3.Row | None:
        return self._db.execute(
            "SELECT updated_on, last_journal_id FROM issues WHERE redmine_id = ?",
            (redmine_id,),
        ).fetchone()

    def save_issue(self, redmine_id: int, updated_on: str, last_journal_id: int) -> None:
        self._db.execute(
            "INSERT INTO issues(redmine_id, updated_on, last_journal_id) VALUES(?, ?, ?) "
            "ON CONFLICT(redmine_id) DO UPDATE SET "
            "updated_on = excluded.updated_on, last_journal_id = excluded.last_journal_id",
            (redmine_id, updated_on, last_journal_id),
        )
        self._db.commit()

    def check_writable(self) -> bool:
        """Проба записи в БД (для /check): создаём служебную таблицу и пишем в неё."""
        try:
            self._db.execute("CREATE TABLE IF NOT EXISTS _write_probe (x INTEGER)")
            self._db.execute("INSERT INTO _write_probe VALUES (1)")
            self._db.execute("DELETE FROM _write_probe")
            self._db.commit()
            return True
        except sqlite3.Error:
            return False

    # --- напоминания ---

    def upsert_reminder(
        self, chat_id: int, message_id: int, text: str, remind_at_ts: int, fmt: str = "html"
    ) -> bool:
        """Новое напоминание для сообщения; повторное нажатие — СДВИГАЕТ время.

        Возвращает True, если создано, False — если сдвинуто существующее.
        """
        existing = self._db.execute(
            "SELECT id FROM reminders WHERE chat_id = ? AND message_id = ?",
            (chat_id, message_id),
        ).fetchone()
        if existing is not None:
            self._db.execute(
                "UPDATE reminders SET remind_at = ?, text = ?, fmt = ? WHERE id = ?",
                (remind_at_ts, text, fmt, existing["id"]),
            )
            self._db.commit()
            return False
        self._db.execute(
            "INSERT INTO reminders(chat_id, message_id, text, remind_at, fmt) "
            "VALUES(?, ?, ?, ?, ?)",
            (chat_id, message_id, text, remind_at_ts, fmt),
        )
        self._db.commit()
        return True

    def due_reminders(self, now_ts: int) -> list[sqlite3.Row]:
        return self._db.execute(
            "SELECT id, text, fmt FROM reminders WHERE remind_at <= ? ORDER BY remind_at",
            (now_ts,),
        ).fetchall()

    def delete_reminder(self, reminder_id: int) -> None:
        self._db.execute("DELETE FROM reminders WHERE id = ?", (reminder_id,))
        self._db.commit()

    # --- связка задача Redmine -> карточка YouGile ---

    def get_yougile_task(self, redmine_id: int) -> sqlite3.Row | None:
        return self._db.execute(
            "SELECT task_id, task_url FROM yougile_tasks WHERE redmine_id = ?",
            (redmine_id,),
        ).fetchone()

    def link_yougile_task(
        self, redmine_id: int, task_id: str, task_url: str | None = None
    ) -> None:
        self._db.execute(
            "INSERT INTO yougile_tasks(redmine_id, task_id, task_url) VALUES(?, ?, ?) "
            "ON CONFLICT(redmine_id) DO UPDATE SET task_id = excluded.task_id, "
            "task_url = excluded.task_url",
            (redmine_id, task_id, task_url),
        )
        self._db.commit()

    # --- история опросов Redmine (для /status) ---

    HISTORY_KEEP = 50

    def log_poll(
        self, poll_at: str, status: str, new_count: int, updated_count: int
    ) -> None:
        self._db.execute(
            "INSERT INTO poll_log(poll_at, status, new_count, updated_count) "
            "VALUES(?, ?, ?, ?)",
            (poll_at, status, new_count, updated_count),
        )
        self._db.execute(
            "DELETE FROM poll_log WHERE id NOT IN ("
            "SELECT id FROM poll_log ORDER BY id DESC LIMIT ?)",
            (self.HISTORY_KEEP,),
        )
        self._db.commit()

    def last_polls(self, n: int = 10) -> list[sqlite3.Row]:
        """Последние n опросов, свежие первыми (сортировка по дате)."""
        return self._db.execute(
            "SELECT poll_at, status, new_count, updated_count FROM poll_log "
            "ORDER BY poll_at DESC, id DESC LIMIT ?",
            (n,),
        ).fetchall()

    # --- события для статистики (/stat) ---

    def log_event(self, kind: str, tokens: int = 0, ts: int | None = None) -> None:
        now = int(time.time()) if ts is None else ts
        self._db.execute(
            "INSERT INTO events(kind, ts, tokens) VALUES(?, ?, ?)", (kind, now, tokens)
        )
        self._db.execute("DELETE FROM events WHERE ts < ?", (now - _EVENTS_RETENTION_S,))
        self._db.commit()

    def event_stats(self, since_ts: int) -> dict[str, tuple[int, int]]:
        """{kind: (кол-во, суммарные токены)} по событиям с since_ts."""
        rows = self._db.execute(
            "SELECT kind, COUNT(*) AS c, COALESCE(SUM(tokens), 0) AS t "
            "FROM events WHERE ts >= ? GROUP BY kind",
            (since_ts,),
        ).fetchall()
        return {r["kind"]: (r["c"], r["t"]) for r in rows}

    # --- история ИИ-диалога в личке ---

    CHAT_RETENTION_S = 30 * 86400

    def add_chat_message(
        self, chat_id: int, user_id: int, role: str, content: str, ts: int | None = None,
    ) -> None:
        if role not in {"user", "assistant"}:
            raise ValueError("role должен быть user или assistant")
        now = int(time.time()) if ts is None else ts
        self._db.execute(
            "INSERT INTO chat_history(chat_id, user_id, role, content, ts) "
            "VALUES(?, ?, ?, ?, ?)",
            (chat_id, user_id, role, content, now),
        )
        self._db.execute(
            "DELETE FROM chat_history WHERE ts < ?", (now - self.CHAT_RETENTION_S,),
        )
        self._db.commit()

    def chat_history(
        self, chat_id: int, user_id: int, limit: int = 20,
    ) -> list[sqlite3.Row]:
        """Последние limit сообщений в хронологическом порядке."""
        rows = self._db.execute(
            "SELECT role, content FROM chat_history "
            "WHERE chat_id = ? AND user_id = ? ORDER BY id DESC LIMIT ?",
            (chat_id, user_id, limit),
        ).fetchall()
        return list(reversed(rows))

    def clear_chat_history(self, chat_id: int, user_id: int) -> int:
        cursor = self._db.execute(
            "DELETE FROM chat_history WHERE chat_id = ? AND user_id = ?",
            (chat_id, user_id),
        )
        self._db.commit()
        return cursor.rowcount
