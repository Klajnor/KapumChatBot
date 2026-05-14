# Telegram Bot с OpenRouter AI — План реализации

## 1. Общая архитектура

### Стек технологий
- **Язык**: Python 3.10+
- **Telegram API**: aiogram 3.x (асинхронный фреймворк)
- **База данных**: PostgreSQL + SQLAlchemy (async) + asyncpg
- **AI API**: OpenRouter (единый API для GPT, Claude, DeepSeek и др.)
- **Конфигурация**: python-dotenv (.env файл)

### Структура проекта

```
d:/-=AI=-/ChatBot/
├── .env                    # Секреты: токены, ключи API
├── .env.example            # Шаблон .env
├── requirements.txt        # Зависимости
├── README.md               # Инструкция по установке
├── bot/
│   ├── __init__.py
│   ├── main.py             # Точка входа, запуск бота
│   ├── config.py           # Загрузка конфигурации из .env
│   ├── bot.py              # Инициализация aiogram Bot и Dispatcher
│   ├── handlers/
│   │   ├── __init__.py
│   │   ├── start.py        # /start — приветствие
│   │   ├── help.py         # /help — список команд
│   │   ├── models.py       # /models — выбор модели AI
│   │   ├── prompt.py       # /prompt — управление системным промптом
│   │   ├── clear.py        # /clear — очистка истории диалога
│   │   └── chat.py         # Обработка текстовых и графических сообщений
│   ├── services/
│   │   ├── __init__.py
│   │   ├── openrouter.py   # Клиент для OpenRouter API
│   │   └── user_service.py # Бизнес-логика работы с пользователями
│   ├── database/
│   │   ├── __init__.py
│   │   ├── models.py       # SQLAlchemy модели (User, UserSettings, Message)
│   │   ├── session.py      # Менеджер сессий БД (async engine)
│   │   └── repository.py   # Репозиторий для работы с БД
│   └── keyboards/
│       ├── __init__.py
│       └── inline.py       # Inline-клавиатуры (выбор модели и т.д.)
```

---

## 2. Компоненты системы

### 2.1 База данных (PostgreSQL + SQLAlchemy)

**Таблицы:**

| Таблица | Назначение | Ключевые поля |
|---------|-----------|---------------|
| `users` | Пользователи Telegram | `telegram_id (PK)`, `username`, `first_name`, `last_name`, `created_at` |
| `user_settings` | Настройки пользователя | `user_id (FK)`, `selected_model`, `system_prompt`, `temperature`, `max_tokens` |
| `messages` | История сообщений | `id (PK)`, `user_id (FK)`, `role` (user/assistant), `content`, `image_url`, `created_at` |

**Связи:** User 1:1 UserSettings, User 1:N Messages

### 2.2 OpenRouter API Client

- Поддержка множества моделей: `openai/gpt-4o`, `anthropic/claude-3.5-sonnet`, `deepseek/deepseek-chat`, `google/gemini-pro` и др.
- Vision API: отправка изображений в модели, поддерживающие vision
- Асинхронные запросы через `aiohttp`
- Конфигурируемые параметры: `temperature`, `max_tokens`, `top_p`

**Формат запроса к OpenRouter:**
```python
POST https://openrouter.ai/api/v1/chat/completions
Headers: {
    "Authorization": "Bearer <ключ>",
    "Content-Type": "application/json"
}
Body: {
    "model": "<выбранная модель>",
    "messages": [
        {"role": "system", "content": "<системный промпт>"},
        {"role": "user", "content": [{"type": "text", "text": "..."}, {"type": "image_url", ...}]},
        ...
    ],
    "temperature": 0.7,
    "max_tokens": 2048
}
```

### 2.3 Telegram Handlers

| Команда | Описание |
|---------|----------|
| `/start` | Приветствие, регистрация пользователя в БД, кнопка "Начать диалог" |
| `/help` | Список всех команд и их описание |
| `/models` | Выбор AI-модели через inline-клавиатуру |
| `/prompt` | Просмотр и установка системного промпта |
| `/model` | Показать текущую модель |
| `/clear` | Очистить историю диалога с подтверждением |
| `/stats` | Статистика использования (кол-во сообщений, токенов) |
| `/settings` | Текущие настройки пользователя |

**Обработка сообщений:**
- **Текст**: отправляется в OpenRouter с историей диалога (последние N сообщений)
- **Изображения**: скачиваются с Telegram, конвертируются в base64, отправляются в vision-модель
- **Стриминг**: ответ от AI может выводиться порциями (по желанию)

### 2.4 Диаграмма потока сообщений

```mermaid
flowchart TD
    User([Пользователь Telegram]) -->|Текст/Изображение| Bot[aiogram Bot]
    Bot --> Handler{Обработчик}
    Handler -->|Команда| CmdHandler[Обработчик команд]
    Handler -->|Сообщение| ChatHandler[Обработчик чата]
    ChatHandler --> DB[(PostgreSQL<br/>История + Настройки)]
    ChatHandler --> OR[OpenRouter API]
    OR -->|Ответ AI| ChatHandler
    ChatHandler --> Bot
    Bot -->|Ответ| User
    CmdHandler --> DB
    CmdHandler --> Bot
```

---

## 3. Процесс установки и настройки

### Шаги для развёртывания:

1. **Установить зависимости:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Создать базу данных PostgreSQL:**
   ```sql
   CREATE DATABASE chat_bot;
   ```

3. **Настроить .env файл:**
   ```
   BOT_TOKEN=ваш_токен_бота
   OPENROUTER_API_KEY=ваш_ключ_openrouter
   OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
   DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/chat_bot
   DEFAULT_MODEL=openai/gpt-4o
   DEFAULT_SYSTEM_PROMPT=Вы полезный AI-ассистент.
   ADMIN_IDS=123456789,987654321
   MAX_HISTORY_LENGTH=30
   ```

4. **Запустить бота:**
   ```bash
   python -m bot.main
   ```

---

## 4. Пошаговый план реализации

### Шаг 1: Настройка проекта и зависимости
- Создать структуру директорий
- Написать `requirements.txt` (aiogram, sqlalchemy[asyncio], asyncpg, aiohttp, python-dotenv)
- Создать `.env.example`
- Создать `README.md`

### Шаг 2: Конфигурация и точка входа
- `bot/config.py` — загрузка переменных окружения через pydantic-settings или dotenv
- `bot/bot.py` — инициализация Bot, Dispatcher, регистрация роутеров
- `bot/main.py` — запуск: создание таблиц БД, старт polling

### Шаг 3: База данных
- `bot/database/models.py` — SQLAlchemy модели (User, UserSettings, Message)
- `bot/database/session.py` — async engine, session factory
- `bot/database/repository.py` — CRUD операции

### Шаг 4: Сервис OpenRouter
- `bot/services/openrouter.py` — асинхронный HTTP-клиент для OpenRouter
- Поддержка: список моделей, chat completion, vision (изображения)
- Обработка ошибок и таймаутов

### Шаг 5: Сервис пользователей
- `bot/services/user_service.py` — регистрация, получение/обновление настроек
- История сообщений: сохранение, загрузка последних N сообщений
- Очистка истории

### Шаг 6: Inline-клавиатуры
- `bot/keyboards/inline.py` — клавиатуры для выбора модели, подтверждения очистки

### Шаг 7: Handlers — команды
- `start.py` — `/start`
- `help.py` — `/help`
- `models.py` — `/models`, выбор модели
- `prompt.py` — `/prompt`, `/prompt <текст>`, просмотр/установка промпта
- `clear.py` — `/clear` с подтверждением

### Шаг 8: Handlers — чат
- `chat.py` — обработка текстовых сообщений
- `chat.py` — обработка изображений (скачивание, base64, vision API)
- Формирование контекста из истории + системный промпт
- Отправка ответа пользователю

### Шаг 9: Финальные доработки
- Обработка ошибок (защита от краша)
- Логирование
- Middleware (антифлуд, логирование)
- Команда `/stats` — статистика использования
- Команда `/settings` — просмотр настроек
