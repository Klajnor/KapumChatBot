# Changelog

## [Unreleased]

### Added
- **MCP tools / Obsidian «second brain» (admin-only)** — the chat model can
  use tools exposed by external MCP servers over stdio. Servers are read from a
  Claude-style `.mcp.json` (`MCP_CONFIG_FILE`, file or directory) or inline
  `MCP_SERVERS` JSON; inline entries override the file. The main use case is the
  `brain` server from the ObsidianBrain project (`search_notes`, `get_chunk`,
  `get_note`, `list_notes`): the model searches and reads personal notes when a
  question calls for them. Tools are namespaced as `mcp__<server>__<tool>`,
  executed through `tools/call` in the normal multi-round loop (up to
  `MCP_MAX_ROUNDS`), and one result is capped at `MCP_MAX_RESULT_CHARS`.
  **`MCP_ADMIN_ONLY=true` by default** — only `ADMIN_IDS` are offered the tools;
  everyone else chats exactly as before. `/mcp` (admin) prints server status and
  tool lists. A server that fails to start never blocks an answer: it is logged,
  retried after a minute and simply contributes no tools. `${VAR}` /
  `${VAR:-default}` expansion is supported in command/args/cwd/env. Requires the
  server's Python to have `mcp<2` (FastMCP was renamed in mcp 2.x).
- **Conversation export (MD / PDF)** — `/export [md|pdf]` saves the
  conversation (the requesting user's own thread, last `EXPORT_MAX_MESSAGES`
  messages) as a file, and 📄 PDF / 📝 MD buttons under every answer in
  private chats export that single Q&A pair. Answer buttons carry the stored
  message id and reload it **with an owner check** — a guessed id can never
  export somebody else's answer. PDF is rendered locally with fpdf2 (new
  optional dependency) using the DejaVu Unicode fonts shipped with
  matplotlib, so Cyrillic works out of the box; code blocks get a mono face,
  inline Markdown is flattened and emoji are stripped (no glyphs in DejaVu).
  When fpdf2 or the fonts are missing, PDF requests silently fall back to
  Markdown. Rendering runs in a worker thread; exports make no AI calls and
  never consume the daily quota. `ENABLE_EXPORT=false` disables everything.
- **Web search (RAG)** — the chat model gets a `search_web` tool and queries
  the live internet when a question needs current information; numbered
  results are fed back as `tool` messages and the answer cites them as `[n]`.
  Providers: Tavily (default) and Brave, selected via `SEARCH_PROVIDER` and
  authenticated with `SEARCH_API_KEY` (alias `TAVILY_API_KEY`). Without a key
  the tool is not offered — fully backward compatible. Search runs as a
  multi-round loop (up to `MAX_SEARCH_ROUNDS`, default 2); when the budget is
  spent a final tools-less request forces an answer from the gathered results.
  Image/sticker tool calls take priority over search in a mixed turn. Keys are
  never logged, snippets are capped at 700 chars, failures degrade to an empty
  result set, and the timing footer sums all rounds. Not wired into guest mode
  (a guest reply is a single `answerGuestQuery`).
- **Sticker generation (`/sticker` + `generate_sticker` tool)** — the model can
  answer with a real Telegram sticker when the user asks for one, and
  `/sticker <prompt>` does it explicitly. The generated image gets a die-cut
  sticker-style prompt suffix, is center-cropped to a square, resized to
  512×512 and re-encoded as WebP (≤512 KB, quality stepped down to fit), then
  sent via `answer_sticker` so it can be saved or added to a sticker set.
  Gated by `ENABLE_STICKER_GENERATION` (default on). `/image` and `/sticker`
  now share one generation command body.
- **AI personalities (`/persona`)** — a menu of ready-made system-prompt
  presets (Assistant, Programmer, Writer, Scientist, Teacher, Translator,
  Critic, Friend) applied with one tap via inline buttons; the active persona
  is marked ✓ and also shown in `/settings`. The catalog is extended/overridden
  with the `PERSONAS` env JSON (`key -> "prompt"` or `key -> {"name","prompt"}`);
  built-ins always remain. The active persona is derived by matching the stored
  prompt against the catalog (no schema change), so `/prompt` still sets a fully
  custom one. Persona prompts obey the same `MAX_SYSTEM_PROMPT_LENGTH`.
- **Reminders & scheduled messages** — `/remind <when> <text>` sets one-time
  (`in 30 minutes`, `at 15:30`, `tomorrow at 9:00`) or recurring
  (`every 2 hours`, `daily at 10:00`, `weekly mon at 10:00`) reminders. With
  the `ai:` prefix the stored prompt is sent to the user's model at fire time
  and the ANSWER is delivered. `/reminders` lists them, `/delreminder N|all`
  deletes. A background scheduler (`bot/services/reminders.py`) claims due
  reminders in the DB before firing (no double-fire, no missed-fire pile-up),
  each fires as an isolated task, and reminders auto-disable when the bot is
  blocked. Per-user caps: `REMINDER_LIMIT` (10) one-time +
  `REMINDER_RECURRING_LIMIT` (10) recurring, at least
  `REMINDER_MIN_INTERVAL_MINUTES` (30) apart; times use `REMINDER_TIMEZONE`
  (stored in UTC). Only admins and whitelisted users can create them.
- **Document analysis** — upload a PDF, DOCX or plain-text/code file and the
  bot extracts its text (pdfplumber / python-docx, both optional deps) and
  analyzes it with the model. Extracted text is clipped to
  `MAX_EXTRACTED_CHARS`, stored in history for follow-ups, and files over
  `MAX_DOCUMENT_BYTES` are rejected without consuming quota. Scanned PDFs
  (no text layer) are declined with a hint to send a photo instead.
- **Link analysis** — a message that is essentially bare links (≤
  `_LINK_REMAINDER_LIMIT` of non-URL text) has up to 2 pages fetched and their
  main content (trafilatura, with a stdlib fallback parser) injected into the
  request. Fetching is SSRF-guarded: only public http(s) hosts, redirects
  validated hop by hop, 5 MB cap, and only after the access check passes.
- **Reply / forward context** — replying to a text message quotes it for the
  model; replying to a photo analyzes the quoted image with the question;
  forwarded messages carry a source label. The bot's own answers are not
  re-quoted (they are already in history).
- **Automatic model fallback** — if a model does not answer (API error, empty
  reply) or reports an exhausted quota/rate limit, the request is retried with
  the next model from `AI_MODELS`; when the last one fails, the emergency
  `AI_FALLBACK_MODELS` list is used. Fallback models are hidden from `/models`
  and cannot be selected by users. Vision requests only try vision-capable
  models, and the timing footer now names the model that actually answered.
- **Token usage in the timing footer** — chat, vision and album answers now
  show prompt/completion token counts (`↑{input} ↓{output} tok`). The new
  `TIMING_FORMAT_TOKENS` template is used whenever the API reports usage
  (placeholders: `{seconds}`, `{model}`, `{tokens}`, `{input}`, `{output}`);
  image generation keeps the plain `TIMING_FORMAT`. Album analysis, which
  previously showed no token count at all, now reports usage too.
- **System prompt length limit** — enforced at 2048 characters by default,
  configurable via `MAX_SYSTEM_PROMPT_LENGTH` in `.env`.
- **Per-photo size limit** — configurable via `MAX_PHOTO_BYTES` (default 20 MB).
  Oversized images are rejected with an explanation.
- **Album total size limit** — configurable via `MAX_ALBUM_BYTES` (default
  50 MB). Albums exceeding this receive a descriptive rejection.
- **Guest text length limit** — `MAX_GUEST_TEXT_LENGTH` (default 4000).
- **Pagination for `/list_users`** — admin command now shows 15 users per page;
  `/list_users 2` jumps to a specific page.
- **Album memory cleanup** — a background task periodically discards stale
  media-group entries that were never completed.
- **Edited-message handling** — editing a question now produces a short notice
  (no quota consumed) and refreshes the remembered question for the
  "ask another model" button.
- **`/broadcast` admin command** — sends an announcement to every registered
  user after an explicit confirmation step.
- **Optional health endpoint** — `HEALTH_PORT`/`HEALTH_HOST` expose
  `GET /health` (JSON status + uptime) for systemd/Docker/uptime monitors.
  Disabled by default.
- **Database integration tests** — 12 tests covering atomic quota operations,
  parallel races, user CRUD, message history and usage statistics.
- **`user_service()` async context manager** — replaces the repeated
  `async_session_factory() + UserService(session)` boilerplate in handlers.

### Changed
- **Allowed users show nicknames, not just IDs** — `/list_users` now prints the
  name and `@username` next to each ID, `/add_user` and `/remove_user` echo the
  nickname. `/add_user`/`/remove_user` accept `@username` (or a `t.me/...`
  link) as well as a numeric ID. The profile is cached on the `allowed_users`
  row: it is captured when the admin adds the user (via `getChat`, when
  available), refreshed whenever that user writes to the bot, and shown even
  if the person has never messaged the bot. Existing SQLite and PostgreSQL
  databases are migrated in place (three nullable columns).
- **Default prompts are now Russian** — the default system prompt, all eight
  persona presets (with Russian display names) and the fallback question
  prompts (vision / voice / document / link) are translated to Russian and
  explicitly instruct the model to answer in Russian unless the user asks for
  another language. The web-search system hint and search-result tool messages
  are Russian too. A startup data migration replaces the *exact* legacy
  English default in `user_settings` with the new one (SQLite and PostgreSQL);
  custom prompts and chosen personas are never touched. Image-generation and
  intent-classifier prompts stay English on purpose (image models and the
  one-word classifier work best in English).
- **Python 3.12+ is now the minimum** — the codebase is modernized to 3.12
  idioms (`datetime.UTC` instead of `datetime.timezone.utc`, builtin
  `TimeoutError` instead of `asyncio.TimeoutError`), and a `ruff.toml` pins
  `target-version = "py312"` so lint rules match the documented floor.
  New optional dependencies: `pdfplumber`, `python-docx`, `trafilatura`
  (document/link analysis) and `tzdata` (reminder timezones on Windows).
- **All user-facing messages are now in English** — themed denial responses
  (Star Wars / Hitchhiker's Guide), guest-mode timeout text, image link
  captions, voice-recognition timing footer, and default timing format
  (`⏱️ {seconds} s` instead of `⏱️ {seconds} с`).
- Handlers use the shared `user_service()` helper instead of hand-rolled
  session management (stats, models, prompt, clear, help, start, image, admin).

### Fixed
- **Album race condition** — `_album_images` dictionary is now protected by
  an `asyncio.Lock`, so concurrent album photos from the same chat cannot
  interfere with each other.
- **Album size accounting** — the running total is no longer left inflated
  when an oversized image is rejected.
- **Album orphan fix** — an oversized last photo no longer prevents the other
  valid photos in the album from being processed.
- **Deduplicated "model does not support images" message** — a single
  constant `MSG_NO_VISION` replaces three identical copies in `chat.py`.
- **Guest quota is no longer wasted on rejected input** — too-long text and
  oversized/over-long voice notes are validated *before* a free-trial message
  is consumed, and quota is refunded when transcription, vision analysis or
  image generation fails.
- **Two pre-existing test failures** — `test_default_prompt_requests_display_math`
  now disables rich messages; `test_failed_guest_answer_is_not_retried` accepts
  the rich → markdown → plain fallback chain.

### Security & robustness audit (fixed)
- **Systemd unit ran the bot as root (HIGH)** — the unit had no `User=`/
  `Group=`, kept everything in `/root`, and left secret/database permissions as
  a manual TODO. It now runs as `brain`, keeps the code `root:root` in `/opt`
  (read-only for the service), stores mutable data in
  `StateDirectory=/var/lib/kapum-chatbot`, sets `UMask=0077`, and adds
  `ProtectHome`, `PrivateDevices`, `ProtectKernel*`, `RestrictAddressFamilies`,
  `RestrictNamespaces`, `LockPersonality`, `CapabilityBoundingSet=` and
  `RestartSec=5s`. `.pyc` are disabled (`PYTHONDONTWRITEBYTECODE=1`), so the
  pre-start `clean_pycache.sh` step is gone. Deployment steps and the required
  `DATABASE_URL` / `TELEGRAPH_ACCOUNT_FILE` values are in the README.
- **Personal settings printed in groups (MEDIUM)** — `/stats`, `/prompt`,
  `/settings` and `/model` echoed the user's system prompt, selected model and
  usage numbers into any group where the bot is a member. They now answer in
  private chats only: a group invocation gets a short hint instead, before any
  database lookup. `/export` invoked in a group keeps working, but only a
  generic acknowledgement is posted there — the file goes to the user's DM.
- **`/export` was a free local DoS (MEDIUM)** — the command and its PDF/MD
  callbacks rendered up to `EXPORT_MAX_MESSAGES` messages for anyone, with no
  access check and no frequency limit. Export now requires an authorized
  (non-guest) user, has a per-user 30 s cooldown and runs inside the per-user
  concurrency slot; the file (and single-answer exports) is always delivered to
  the requesting user's private chat, never to the group.
- **MCP servers inherited every bot secret (HIGH)** — `MCPClient.start()` passed
  `os.environ.copy()` to each stdio server, so third-party MCP code received
  `BOT_TOKEN`, `AI_API_KEY`, `SEARCH_API_KEY`, `TELEGRAPH_ACCESS_TOKEN` and
  `DATABASE_URL` (full bot takeover / billing abuse). Child processes now get a
  minimal allowlist (`PATH`, `HOME`/`USERPROFILE`, locale, `TEMP`/`TMP`, …) plus
  the server's own explicit `env` block; `_child_environment()` is covered by
  unit and end-to-end tests. `${BOT_TOKEN}` / `${AI_API_KEY}` references in the
  MCP config are refused as well, so a tampered `.mcp.json` cannot pull the bot
  credentials into a child process or its command line.
- **Paid AI call without an access check (HIGH)** — `_route_single_photo` ran
  the `detect_intent` classifier for *any* user, so an unauthorized account
  could farm the paid API by sending captioned photos in a loop (bounded only
  by the per-user concurrency cap, with no daily limit). Gated behind
  `check_access(..., consume=False)`; the album and guest paths were already
  correct.
- **Scheduled AI reminders outlived access revocation (HIGH)** — `_fire_ai`
  never re-checked permissions, so a user removed from the allow-list (or an
  admin dropped from `ADMIN_IDS`) kept triggering paid model calls forever.
  Access is now re-verified at fire time and the reminder is disabled on
  revocation.
- **Unhandled `OverflowError` in reminder parsing (HIGH)** — the interval
  regexes accept unbounded digits and `timedelta` overflows far below
  `sys.maxsize` (verified at ~10¹² minutes), escaping as a generic error.
  Intervals are now capped at ~10 years with a clear message.
- **Reminders lost on Telegram 429 (MEDIUM)** — a scheduler tick fires up to 50
  reminders across many chats while `throttle` only serialises *per chat*, and
  reminders are claimed (deleted/rescheduled) *before* delivery. Delivery now
  honours `retry_after` with bounded retries.
- **Fire-and-forget reminder tasks (MEDIUM)** — `create_task` results were
  discarded (the loop holds only weak references) and never cancelled at
  shutdown, so `engine.dispose()` could run under an in-flight DB call. Tasks
  are now tracked and cancelled before the engine is disposed.
- **DOCX zip bomb (MEDIUM)** — a ≤20 MB archive can declare gigabytes of XML,
  which `python-docx` builds in memory. Archives are now rejected on their
  declared uncompressed size (>100 MB) before parsing.
- **Unbounded PDF extraction (MEDIUM)** — every page was extracted before the
  text was clipped. Now capped at 200 pages and `MAX_EXTRACTED_CHARS`.
- **Image decompression bomb (MEDIUM)** — `ImageOps.exif_transpose` forced a
  full pixel decode *before* the 60 Mpx guard, and Pillow's own guard only
  raises above ~178 Mpx (~1 GB+ of RAM). The pixel count is now checked on the
  header, before any decode.
- **Album memory amplification (MEDIUM)** — album photos were downloaded before
  their size was validated (up to 10 × 20 MB in memory). Sizes are now checked
  against the declared `file_size` *before* downloading, and the cumulative
  album budget stops further downloads.
- **Broadcast confirmations were not bound to their author (MEDIUM)** — any
  admin could confirm or cancel another admin's draft via forged callback data.
  The owner id is now part of the callback payload and verified against the
  pressing user.
- **Broadcast had no length guard (MEDIUM)** — text over Telegram's 4096-char
  limit failed for *every* recipient. Now rejected up front.
- **Broadcast blocked its callback handler (MEDIUM)** — delivery ran inline
  (minutes for a large user base). It now runs as a tracked background task
  with 429 retries, and blocked users are skipped without a retry.
- **Invalid `HEALTH_PORT` aborted startup (MEDIUM)** — an out-of-range port
  makes asyncio raise `OverflowError`, which the `except OSError` handler did
  not catch. The port is now validated in config (invalid → disabled) and the
  bind-failure handler catches broadly.
- **Health endpoint leaked configuration** — it reported feature flags; the
  payload is now limited to liveness and uptime.
- **`MAX_GUEST_TEXT_LENGTH=0` rejected every guest query** — the value is now
  clamped to `[1, 4096]`.
- **`MAX_PHOTO_BYTES` could exceed Telegram's limit** — the clamp allowed
  100 MB while the Bot API refuses to hand a bot a file over 20 MB; the cap is
  now 20 MB.
- **Edited-message notices could be flooded** — repeated edits produced up to
  one bot message per second per chat. Notices are now limited to one per 30 s
  per chat, with a bounded registry.

### Critical fixes (second audit pass)
- **Reminders were permanently disabled by the daily question quota (CRITICAL,
  regression from the previous audit round)** — the fire-time access re-check
  used `check_access(..., consume=False)`, which still returns "daily limit
  reached" when the budget is spent. A whitelisted user who asked their
  `DAILY_LIMIT` questions today lost *every* scheduled AI reminder, even though
  reminders are free and consume no quota. The gate now uses
  `require_quota=False` and only reacts to a genuine loss of authorization
  (or a guest role, which may not hold reminders at all).
- **Free operations were blocked by an exhausted quota** — `check_access`
  returned the limit result *before* honouring `consume=False`, contradicting
  its own contract and the `/remind` docstring ("creation is free"). Added an
  explicit `require_quota` flag: `False` asks "is this user authorized at all?"
  without inspecting the remaining budget. Existing callers keep the old
  quota-aware behaviour (the flag defaults to `True`).
- **Reminder cap race** — the count check and the insert are two statements, so
  two concurrent `/remind` commands could both pass and overshoot
  `REMINDER_LIMIT`. Creation is now serialised by a per-user `asyncio.Lock`
  (bounded registry that never evicts a held lock).
- **`EXPORT_MAX_MESSAGES` was not validated** — the renderer keeps every row and
  the whole PDF in memory, so a typo like `100000000` would OOM the worker
  thread. Now clamped to `[10, 5000]`.
- **An invalid API key burned the whole model fallback chain on every message** —
  `401`/`403`/`413` are not model-specific, yet the chain tried every catalog
  and emergency model (each with its own retries) before giving up. The chain
  now aborts on those statuses in all four loops (chat, tools, vision,
  streaming); `AIStreamError` carries the HTTP status so streaming can tell a
  fatal error from a model-specific one. Transient `5xx`/`429` still walk the
  full chain.

### Known limitations (audit findings, not yet fixed)
- **No backoff on a permanently failing scheduled AI request** — the error
  notice is re-sent every interval instead of auto-disabling after N failures.
- **Disabled reminders are never purged** — they accumulate until the user runs
  `/delreminder all`.
- **DNS-rebinding TOCTOU in link fetching** — the SSRF guard resolves the host
  once and aiohttp resolves it again; documented as accepted in `documents.py`.
- **`handlers/reminders.py` interpolates `describe_schedule()` unescaped** in
  one place (escaped elsewhere). Currently unexploitable — schedule strings are
  digit-only — but an inconsistency worth removing.
- **`SEARCH_SYSTEM_HINT` / `format_search_results` are in Russian** while every
  other user-facing string was moved to English; the hint also instructs the
  model to answer in Russian by default. Intentional language policy or an
  oversight from the web-search feature — worth a decision.
- **A non-model-specific `400` still walks the whole chain** — e.g. a
  context-too-long rejection fails identically on every model. Bounded, but it
  multiplies latency and cost; consider aborting on `400` once the tools-shaped
  retry has been attempted.
- **`chat.py` is ~2500 lines / 85 KB** — a god-module; splitting delivery,
  albums, documents and enhance flows into separate modules would help.
- **`test_ws_asr.py` sits at the repo root and is tracked** — pytest collects
  any `test_*.py` from the rootdir, so this manual ASR probe is imported on
  every test run. Move it to `scripts/`.

### Documentation
- Added `CONTRIBUTING.md` with setup, style, testing, and PR instructions.
- Added `CHANGELOG.md` (this file).
- README updated: new limits, broadcast, health endpoint, edited messages.

---

## Format

This project follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and [Semantic Versioning](https://semver.org/).

Types of changes:
- `Added` — new features
- `Changed` — changes in existing functionality
- `Deprecated` — soon-to-be removed features
- `Removed` — now removed features
- `Fixed` — bug fixes
- `Security` — vulnerability fixes