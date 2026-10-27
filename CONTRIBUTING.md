# CONTRIBUTING

## Quick start

```bash
# 1. Clone and set up a virtual environment
python -m venv .venv
.venv\Scripts\activate     # Windows
source .venv/bin/activate  # Linux/macOS

# 2. Install dependencies (including dev extras)
pip install -r requirements.txt
pip install pytest pytest-asyncio ruff

# 3. Copy and fill the configuration
cp .env.example .env
#   → set BOT_TOKEN, AI_API_KEY, OWNER_IDS at minimum

# 4. Run the tests
pytest tests/ -v
```

## Project structure

```
bot/
├── main.py          – entry point, router registration, global error handler
├── config.py        – Config dataclass, env parsing, model catalog
├── bot.py           – Bot/Dispatcher init, identity helpers
├── handlers/        – Telegram message/callback handlers
├── services/        – Business logic (AI client, markdown, formulas, …)
├── database/        – SQLAlchemy models, session, repository
└── keyboards/       – Inline keyboard builders
tests/               – Pytest test suite
scripts/             – Utility scripts (cache clean, migration, diagnostics)
```

## Code style

- **Type hints everywhere** — functions, methods, dataclasses.
- **`from __future__ import annotations`** in every module.
- **Docstrings** on every public function and class.
- **`black`-compatible** line length (88 chars recommended, up to ~120 accepted).
- **Lint with ruff** before committing: `ruff check bot/ tests/`
- **`# noqa: BLE001`** marks intentional broad exception handlers (Telegram calls
  that must never break a reply).

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run only regression tests
pytest tests/test_regressions.py -v

# Run only formula tests
pytest tests/test_formulas.py -v

# Run only database integration tests (in-memory SQLite)
pytest tests/test_database.py -v
```

Tests cover:
- Config parsing and validation
- API log sanitization
- Guest mode replies and timeouts
- Image generation edge cases
- Formula rendering (PNG, bounds, fallbacks)
- User settings defaults and limits
- Database CRUD and atomic quota operations (real SQLite, no mocks)

When adding new features:
1. Add tests that cover both the happy path and edge cases.
2. Use real SQLite (in-memory) for database-related tests.
3. Use `SimpleNamespace` / `unittest.mock` for Telegram API objects.

## Commit messages

Keep them short and descriptive. Prefix with the area changed:
- `chat:` — message handling, streaming, delivery
- `config:` — configuration, env vars
- `guest:` — guest mode
- `formulas:` — LaTeX → Unicode, PNG rendering
- `db:` — database schema, migrations, repository
- `tests:` — test suite
- `docs:` — README, docstrings

## Before submitting a PR

- [ ] Code has type hints and docstrings
- [ ] `ruff check` passes with no errors
- [ ] `pytest tests/` passes
- [ ] README updated if needed
- [ ] `.env.example` updated if new config keys were added