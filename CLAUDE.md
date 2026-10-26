# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Что это

Telegram-бот на **aiogram 3.x (≥ 3.29, Python 3.12+)** поверх OpenAI-совместимого API Qwen (Alibaba Model Studio) и других провайдеров. БД — SQLite по умолчанию (`sqlite+aiosqlite`), опционально PostgreSQL. Документация, README, CHANGELOG и комментарии — на русском; пользовательские сообщения бота — на английском; промпты модели по умолчанию — на русском.

## Команды

```bash
pip install -r requirements.txt
pip install pytest pytest-asyncio ruff   # dev-зависимости (в requirements.txt их нет)
cp .env.example .env                     # минимум: BOT_TOKEN, AI_API_KEY, ADMIN_IDS

python -m bot.main                       # запуск бота (так же в chat_bot.service)

ruff check bot/ tests/                   # линт (ruff.toml: target py312)
pytest tests/ -v                         # все тесты
pytest tests/test_regressions.py -v      # один файл
pytest tests/test_regressions.py::RegressionTests::test_name -v   # один тест
```

- Тесты — в основном `unittest.IsolatedAsyncioTestCase` (запускаются через pytest); БД-тесты используют реальную in-memory SQLite, без моков. Telegram-объекты подменяются `SimpleNamespace` / `unittest.mock`.
- `ffmpeg` нужен для голосовых (не pip-пакет); `ENABLE_VOICE=false` отключает.
- `scripts/check_search.py --query "..." [--ask]` — диагностика веб-поиска; `scripts/clean_pycache.*` — очистка кэша; `restart.sh` — деплой на сервер (`/opt/KapumChatBot`, systemd).

## Конвенции

- `from __future__ import annotations` в каждом модуле, type hints и docstrings на публичном API.
- `# noqa: BLE001` помечает намеренные широкие `except` вокруг вызовов Telegram, которые не должны ломать ответ.
- Новый параметр конфигурации → добавить в `bot/config.py`, `.env.example` и таблицу в README.
- Коммит-префиксы по области: `chat:`, `config:`, `guest:`, `formulas:`, `db:`, `tests:`, `docs:`. Каждую фичу коммитить с подробным описанием в теле; добавлять запись в `CHANGELOG.md` (сверху новые, по датам).

## Архитектура

Точка входа — `bot/main.py`: регистрирует роутеры, глобальный обработчик ошибок, запускает фоновые задачи (`_album_cleanup`, `reminder_scheduler_loop`, `digest_scheduler_loop`, health-сервер, MCP). **Порядок `dp.include_router` важен**: командные роутеры идут раньше catch-all `chat.router`; `guest.router` подключается только при `ENABLE_GUEST_MODE`.

Слои:
- `bot/handlers/` — Telegram-обработчики. `chat.py` (~3000 строк) — ядро: текст/голос/фото/документы/альбомы, ссылки, tool-loop (поиск, картинки, стикеры, MCP), стриминг (`_StreamPrinter`), доставка ответа (`deliver_answer`), callback-и «спросить другую модель» и выбора способа улучшения фото. `guest.py` — guest chat mode (inline-запрос → ровно один `answerGuestQuery`, поэтому там нет индикатора, поиска и многоходовых циклов).
- `bot/services/` — бизнес-логика. `ai_client.py` — единый `AIClient` (чат, SSE-стриминг, vision, ASR по WebSocket, генерация картинок через нативный DashScope API, замер времени/токенов, fallback-цепочка моделей). `user_service.py` + `access.py` + `limits.py` — уровни доступа (guest → user → admin → owner из `ADMIN_IDS`), дневные и персональные лимиты, атомарное списание квоты с возвратом при отказе. `documents.py` — извлечение текста (PDF/DOCX/XLSX/PPTX), загрузка ссылок с SSRF-защитой, обход ссылок. `mcp.py` — stdio MCP-клиент («второй мозг»). `rich.py`/`markdown.py`/`telegraph.py`/`formulas.py`/`science.py` — цепочка форматирования ответа.
- `bot/database/` — SQLAlchemy 2.0 async: `models.py`, `repository.py`, `session.py`. Миграций через Alembic нет: `init_db()` создаёт таблицы и применяет идемпотентные `ALTER TABLE` (отдельные ветки для SQLite и PostgreSQL) — новые колонки добавлять туда же.
- `bot/config.py` — единый `config` (dataclass из env). Модели задаются слотами `MODEL_1..6_*` + `FALLBACK_MODEL_*` (у каждой свой URL/ключ/`VISION`); `ID` — внутренний id бота, провайдеру уходит `MODEL`. При запуске недоступные у провайдера модели скрываются из `/models`.

Сквозные механизмы, которые видны только при чтении нескольких файлов:
- **Fallback моделей**: при ошибке/пустом ответе/превышении квоты запрос идёт следующей моделью по слотам, затем в `FALLBACK_MODEL_*` (скрыта из меню); для vision перебираются только модели с `VISION=true`.
- **Цепочка доставки ответа**: rich-сообщение (Bot API 10.1, `RICH_MESSAGES`) → MarkdownV2 → HTML → plain; таблицы/LaTeX при отказе rich → Telegraph; легаси-путь рендерит формулы в PNG / Unicode. Ошибки отправки не должны терять ответ; троттлинг ~1 сообщение/сек на чат и ретраи на 429 — в `telegram_utils.py`.
- **Недоверенный контент** (`services/untrusted.py`): документы, страницы, цитаты/форварды и результаты поиска оборачиваются в `<untrusted_content>`; маркер сохраняется в истории, и если он есть в запросе или истории — MCP-инструменты на этот запрос не предлагаются. Новые источники внешнего текста нужно оборачивать так же.
- **MCP**: по умолчанию только для админов; процесс сервера не наследует секреты бота (allowlist окружения `_ENV_ALLOWLIST`); `command` — абсолютный путь; внешний `.mcp.json` требует пин `MCP_CONFIG_SHA256`.
- **Квота**: запросы, отклонённые до обращения к модели (слишком большой файл, сбой распознавания и т.п.), квоту не расходуют; списанную квоту возвращают при сбое.
- История диалога хранится **по чату**, а не по пользователю; подпись `⏱️ …` в историю не попадает.
- `SECURITY_AUDIT.md` — результаты аудита; учитывать при правках SSRF, MCP, поиска и документов.
