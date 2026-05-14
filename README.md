# 🤖 Telegram AI Bot — OpenRouter + PostgreSQL

Telegram-бот с интеграцией OpenRouter API. Поддерживает текстовые диалоги, анализ изображений (Vision API), выбор AI-модели и настройку системного промпта.

## ✨ Возможности

- 💬 **Чат с AI** — общайтесь с различными языковыми моделями через OpenRouter
- 🖼️ **Работа с изображениями** — отправляйте фото для анализа Vision-моделям
- 🤖 **Выбор модели** — переключайтесь между GPT-4o, Claude, DeepSeek и другими
- ⚙️ **Системный промпт** — настройте поведение AI под свои задачи
- 📚 **История диалога** — контекст сохраняется в PostgreSQL
- 🧹 **Очистка истории** — сброс диалога командой `/clear`

## 🛠️ Стек технологий

- **Python 3.10+**
- **aiogram 3.x** — асинхронный фреймворк для Telegram Bot API
- **PostgreSQL** — база данных
- **SQLAlchemy 2.0 (async)** — ORM
- **asyncpg** — драйвер PostgreSQL
- **OpenRouter API** — единый API для множества LLM

## 🚀 Установка и запуск

### 1. Клонировать репозиторий

```bash
git clone <repo-url>
cd ChatBot
```

### 2. Установить зависимости

```bash
pip install -r requirements.txt
```

### 3. Настроить базу данных PostgreSQL

```sql
CREATE DATABASE chat_bot;
```

### 4. Настроить переменные окружения

Скопируйте `.env.example` в `.env` и заполните:

```bash
cp .env.example .env
```

| Переменная | Описание |
|-----------|----------|
| `BOT_TOKEN` | Токен бота от [@BotFather](https://t.me/BotFather) |
| `OPENROUTER_API_KEY` | API-ключ от [OpenRouter](https://openrouter.ai/keys) |
| `DATABASE_URL` | Строка подключения к PostgreSQL |
| `DEFAULT_MODEL` | Модель AI по умолчанию (например, `openai/gpt-4o`) |
| `DEFAULT_SYSTEM_PROMPT` | Системный промпт по умолчанию |
| `ADMIN_IDS` | ID администраторов (через запятую) |
| `MAX_HISTORY_LENGTH` | Максимум сообщений в контексте (по умолчанию 30) |

### 5. Запустить бота

```bash
python -m bot.main
```

## 📋 Команды

| Команда | Описание |
|---------|----------|
| `/start` | Начать работу с ботом |
| `/help` | Список команд |
| `/models` | Выбрать AI-модель |
| `/model` | Показать текущую модель |
| `/prompt` | Просмотреть системный промпт |
| `/prompt <текст>` | Установить новый системный промпт |
| `/clear` | Очистить историю диалога |
| `/stats` | Статистика использования |
| `/settings` | Текущие настройки |

## 📁 Структура проекта

```
bot/
├── main.py              # Точка входа
├── config.py            # Конфигурация
├── bot.py               # Инициализация бота
├── handlers/            # Обработчики команд и сообщений
│   ├── start.py
│   ├── help.py
│   ├── models.py
│   ├── prompt.py
│   ├── clear.py
│   └── chat.py
├── services/            # Бизнес-логика
│   ├── openrouter.py
│   └── user_service.py
├── database/            # Работа с БД
│   ├── models.py
│   ├── session.py
│   └── repository.py
└── keyboards/           # Inline-клавиатуры
    └── inline.py