"""MCP (Model Context Protocol) client: external tools for the chat model.

The bot can connect to one or more MCP servers and offer their tools to the
model. The primary use case is an Obsidian "second brain" (a ``brain`` server
that indexes a vault and exposes ``search_notes`` / ``get_chunk`` /
``get_note`` / ``list_notes``), but nothing here is Obsidian-specific: any
stdio MCP server from a ``.mcp.json`` works.

Design notes:

* Transport is the MCP **stdio** transport — newline-delimited JSON-RPC 2.0
  over the server process's stdin/stdout. No SDK dependency is needed on the
  bot side, so the bot keeps working even when the server itself uses another
  Python environment.
* Servers are started lazily, on the first message that needs a tool, and
  kept alive for later requests. A server that fails to start is retried
  after a cooldown instead of on every message.
* A server process receives a **minimal environment** (see
  :func:`_child_environment`): the bot's own secrets — ``BOT_TOKEN``,
  ``AI_API_KEY``, ``SEARCH_API_KEY`` (and its ``TAVILY_API_KEY`` /
  ``EXA_API_KEY`` aliases), ``TELEGRAPH_ACCESS_TOKEN``, ``DATABASE_URL`` —
  are never inherited by third-party MCP code, and
  ``${VAR}`` references to them in the config are refused. Anything a server
  needs beyond the allowlist must be listed explicitly in its ``env``.
* Чтение открыто пользователям диалога. Запись — отдельный канал владельца,
  никогда не инструмент модели; приватное хранилище подключать нельзя.

See README → «🧠 MCP-провайдер „Второй мозг“ (Obsidian)».
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

from bot.config import config
from bot.services.note_metadata import generate_note_metadata
from bot.services.note_retrieval import note_references, read_failed
from bot.services.request_scope import route
from bot.services.tool_budget import ToolBudget, clip_tool_text
from bot.services.untrusted import messages_have_untrusted_content, wrap_untrusted
from bot.services.user_service import user_service

logger = logging.getLogger(__name__)

# Protocol revision advertised by the client. Every FastMCP release in use
# today negotiates this (2024-11-05) or answers with its own revision.
PROTOCOL_VERSION = "2024-11-05"
CLIENT_INFO = {"name": "kapum-chatbot", "version": "1.0"}

# How long a failed server is left alone before another start attempt.
RETRY_COOLDOWN = 60.0

# Max line size for a JSON-RPC message from a server (one note can be large).
_MAX_LINE_BYTES = 8 * 1024 * 1024

# OpenAI function names allow only letters, digits, "_" and "-".
_NAME_RE = re.compile(r"[^A-Za-z0-9_-]+")
_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")

# Appended to the system prompt while read-only MCP tools are offered.
MCP_SYSTEM_HINT = (
    "Тебе доступен общий «второй мозг» бота — база заметок Obsidian "
    "(инструменты с префиксом mcp__). Это реальный доступ к заметкам через MCP, "
    "прямой доступ к локальным файлам для него не нужен. На явную просьбу "
    "поискать в заметках вызови инструмент, а не отвечай, что доступа нет. "
    "Если вопрос может касаться сохранённых "
    "заметок, проектов, прошлых решений, документации или сохранённых знаний, "
    "сначала найди подходящие заметки (search_notes). Бот сам прочитает "
    "найденные заметки через get_note, а при ошибке — доступный get_chunk, "
    "в пределах бюджета. В ответе инструмента различаются карточка, полный "
    "текст и частичный фрагмент. Отвечай содержательно по полученным текстам, "
    "а не обещанием «нашёл, сейчас извлекаю»: после этого ответа продолжения "
    "без нового сообщения пользователя не будет. Не "
    "выдумывай содержимое заметок: если поиск ничего не дал, честно скажи об "
    "этом. Указывай путь заметки, на которую опираешься."
)


class MCPError(RuntimeError):
    """An MCP server could not be started or returned an error."""


@dataclass(slots=True)
class MCPServerConfig:
    """One stdio MCP server from ``.mcp.json``."""

    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None
    # None — use MCP_ALLOWED_TOOLS; otherwise this server's own list.
    allowed_tools: list[str] | None = None

    def permits(self, tool_name: str) -> bool:
        """True when the model may see and call ``tool_name`` of this server.

        Default-deny: a tool the server adds later (a write or delete tool, say)
        stays invisible until it is listed in ``MCP_ALLOWED_TOOLS`` (or the
        server's ``allowed_tools``). ``"*"`` allows everything. The current
        list is the read-only set of the second-brain server and may change.
        """
        allowed = (
            self.allowed_tools
            if self.allowed_tools is not None
            else [part.strip() for part in config.mcp_allowed_tools.split(",")]
        )
        return "*" in allowed or tool_name in allowed


@dataclass(slots=True)
class MCPTool:
    """A tool advertised by a server, plus its namespaced OpenAI name."""

    server: str
    name: str
    description: str
    input_schema: dict[str, Any]
    exposed_name: str = ""

    def openai_schema(self) -> dict[str, Any]:
        """The tool in OpenAI ``function`` format, with a namespaced name."""
        return {
            "type": "function",
            "function": {
                "name": self.exposed_name or self.name,
                "description": f"[MCP:{self.server}] {self.description}".strip(),
                "parameters": _normalize_schema(self.input_schema),
            },
        }


# Bot credentials that must never reach an MCP server, explicitly or by
# inheritance (see ``_expand_env`` and ``_child_environment``). The suffix rule
# also covers helper modules (``SAMO_HELPER_*_API_KEY``) and future token names.
_SECRET_ENV_NAMES = frozenset({
    "BOT_TOKEN", "AI_API_KEY", "QWEN_API_KEY", "DASHSCOPE_API_KEY",
    "SEARCH_API_KEY", "TAVILY_API_KEY", "EXA_API_KEY",
    "TELEGRAPH_ACCESS_TOKEN",
    "DATABASE_URL",
})
_SECRET_ENV_SUFFIX_RE = re.compile(r"(?:TOKEN|SECRET|PASSWORD|API_KEY|DATABASE_URL)$")


def _is_secret_env_name(name: str) -> bool:
    """True when this environment variable must never be expanded for MCP."""
    upper = name.upper()
    return (
        upper in _SECRET_ENV_NAMES
        or bool(_SECRET_ENV_SUFFIX_RE.search(upper))
        or bool(re.fullmatch(r"(?:MODEL_\d+|FALLBACK_MODEL)_API_KEY", upper))
    )


def _expand_env(value: str) -> str:
    """Expand ``${VAR}`` / ``${VAR:-default}`` in a config string.

    References to the bot's own credentials are refused (with a warning): the
    ``.mcp.json`` can live in a synced directory, so a tampered entry must not
    be able to pull ``BOT_TOKEN``, ``SAMO_HELPER_*`` or any other API key into
    a server's environment or command line. Server-specific secrets belong in
    the server's ``env`` block.
    """

    def replace(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        if _is_secret_env_name(name):
            logger.warning(
                "MCP config references the bot secret %r; expansion refused", name
            )
            return default if default is not None else ""
        return os.getenv(name, default if default is not None else "")

    return _ENV_RE.sub(replace, value)


# --- Subprocess environment --------------------------------------------------

# Only these variables are inherited from the bot's own environment. An MCP
# server is third-party code: handing it ``os.environ`` would give it the bot
# token, every AI/search/Telegraph key and the database URL, i.e. full control
# over the bot and its billing. Server-specific variables (``VAULT_PATH`` …)
# are passed explicitly through the server's ``env`` block in ``.mcp.json`` —
# visible, reviewable and scoped to that one server.
_ENV_ALLOWLIST = frozenset({
    # POSIX basics and locale/terminal state.
    "PATH", "HOME", "USER", "LOGNAME", "SHELL", "TERM",
    "LANG", "LANGUAGE", "LC_ALL", "LC_CTYPE", "LC_MESSAGES", "TZ", "TMPDIR",
    "XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_DATA_HOME", "XDG_RUNTIME_DIR",
})

# Windows processes need these to start at all (system DLLs, temp dirs,
# per-user app data); none of them carry bot credentials.
_WINDOWS_ENV_ALLOWLIST = frozenset({
    "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT", "OS",
    "TEMP", "TMP", "USERPROFILE", "HOMEDRIVE", "HOMEPATH",
    "APPDATA", "LOCALAPPDATA", "PROGRAMDATA", "PROGRAMFILES",
    "PROGRAMFILES(X86)", "COMMONPROGRAMFILES", "NUMBER_OF_PROCESSORS",
    "PROCESSOR_ARCHITECTURE", "PSMODULEPATH",
})


def _child_environment(overrides: dict[str, str] | None = None) -> dict[str, str]:
    """Environment for an MCP subprocess: allowlisted base + explicit overrides.

    The bot's secrets are deliberately absent: a server that needs another
    variable declares it in its own ``env`` block (``${VAR}`` expansion is
    supported there), which keeps every credential hand-off explicit. Overrides
    are applied last and win over inherited values.
    """
    allowed = _ENV_ALLOWLIST
    if os.name == "nt":
        allowed = allowed | _WINDOWS_ENV_ALLOWLIST

    environment = {
        key: value for key, value in os.environ.items() if key.upper() in allowed
    }
    environment.update(
        {
            str(key): _expand_env(str(value))
            for key, value in (overrides or {}).items()
        }
    )
    return environment


def _sanitize(part: str) -> str:
    """Make one component safe for an OpenAI function name."""
    cleaned = _NAME_RE.sub("_", part).strip("_")
    return cleaned or "tool"


def _exposed_name(server: str, tool: str) -> str:
    """Namespaced tool name: ``mcp__<server>__<tool>`` (max 64 chars)."""
    name = f"mcp__{_sanitize(server)}__{_sanitize(tool)}"
    if len(name) <= 64:
        return name
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]
    return f"{name[:55]}_{digest}"


def _normalize_schema(schema: Any) -> dict[str, Any]:
    """Coerce an MCP ``inputSchema`` into a valid JSON Schema object."""
    if not isinstance(schema, dict) or not schema:
        return {"type": "object", "properties": {}}
    result = dict(schema)
    result.setdefault("type", "object")
    if result.get("type") == "object" and not isinstance(
        result.get("properties"), dict
    ):
        result["properties"] = {}
    return result


def _format_tool_result(result: Any) -> str:
    """Flatten an MCP ``tools/call`` result into text for the model."""
    if not isinstance(result, dict):
        return str(result)

    parts: list[str] = []
    for item in result.get("content") or []:
        if not isinstance(item, dict):
            continue
        kind = item.get("type")
        if kind == "text":
            text = item.get("text")
            if text:
                parts.append(str(text))
        elif kind == "resource":
            resource = item.get("resource") or {}
            value = resource.get("text") or resource.get("uri")
            if value:
                parts.append(str(value))
        elif kind in {"image", "audio"}:
            parts.append(f"[{kind}]")
        else:
            parts.append(json.dumps(item, ensure_ascii=False))

    if not parts and result.get("structuredContent") is not None:
        parts.append(
            json.dumps(result["structuredContent"], ensure_ascii=False, indent=2)
        )

    text = "\n".join(part for part in parts if part).strip() or "(empty result)"
    if result.get("isError"):
        text = f"Ошибка инструмента MCP: {text}"
    return text


def _json_object(text: str) -> dict[str, Any] | None:
    """Extract one JSON object from a tool result (FastMCP may wrap it in text)."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(text[start : end + 1])
    except (TypeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _note_error_reason(result: Any) -> str:
    """Диагностика записи без вывода недоверенного текста, путей и секретов."""
    if not isinstance(result, dict):
        return "MCP-сервер вернул некорректный ответ"
    text = _format_tool_result(result).casefold()
    reasons = (
        (("stage=file",),
         "не удалось создать файл заметки; проверьте права на хранилище и systemd ProtectHome/ReadWritePaths"),
        (("stage=index",),
         "сбой индексации SQLite; проверьте data/brain.db, права, свободное место и повторите"),
        (("stage=git",),
         "сбой Git при публикации; проверьте права на .git, ветку/upstream и доступ к remote"),
        (("unknown tool", "tool not found", "запись выключена"),
         "create_note недоступен: проверьте обновление ObsidianBrain и BRAIN_WRITE_ENABLED=true в env MCP-сервера"),
        (("dubious ownership", "небезопасный владелец git"),
         "Git отклонил владельца хранилища (dubious ownership); проверьте владельца каталога и safe.directory"),
        (("user.name", "user.email", "author identity unknown", "unable to auto-detect email"),
         "не настроены Git user.name/user.email у пользователя MCP-сервера"),
        (("аутентификация git", "authentication failed", "permission denied (publickey)", "could not read username",
          "could not read from remote", "repository not found", "terminal prompts disabled",
          "host key verification", "no such identity", "ssh-ключ недоступен"),
         "Git-аутентификация не удалась; проверьте SSH-ключ/credential helper пользователя MCP-сервера"),
        (("dns git", "could not resolve host", "could not resolve hostname"),
         "Git не может разрешить имя remote: проверьте DNS/сеть сервера"),
        (("сетевое подключение к git", "connection timed out", "connection refused"),
         "Git не может подключиться к remote: проверьте сеть сервера"),
        (("read-only file system", "readonly database", "read-only database"),
         "хранилище или SQLite доступны только для чтения; проверьте права и systemd ProtectSystem/ReadWritePaths"),
        (("permission denied", "errno 13", "no such file or directory", "operation not permitted"),
         "нет прав/доступа к файлу, SQLite или .git у пользователя MCP-сервера; проверьте права и systemd ProtectHome/ReadWritePaths"),
        (("database is locked", "database schema has changed"),
         "SQLite занят или схема изменилась: повторите сохранение позже"),
        (("unique constraint failed", "integrityerror"),
         "заметка уже проиндексирована параллельно (sync.py); повторите сохранение — дубликата не будет"),
        (("no such table", "no such column"),
         "схема SQLite не создана или устарела; проверьте data/brain.db и запустите sync.py"),
        (("database or disk is full", "no space left on device"),
         "на диске нет места для SQLite или Git; освободите место"),
        (("посторонние локальные коммиты",),
         "в хранилище есть посторонние локальные коммиты; автоматический push заблокирован"),
        (("уже staged",), "созданная заметка уже staged; автоматическое изменение индекса заблокировано"),
        (("нужен upstream", "ошибка git (config)"),
         "не настроен upstream текущей ветки Git-хранилища"),
        (("корнем git-репозитория", "not a git repository"),
         "первое хранилище не является корнем Git-клона"),
        (("ошибка git (merge-base)",),
         "ветка хранилища отстаёт от remote или разошлась с ним; синхронизируйте клон вручную"),
        (("ошибка git (commit-tree)",),
         "сбой создания Git-коммита; проверьте user.name/user.email и права на .git"),
        (("ошибка git (fetch)",),
         "сбой git fetch; проверьте сеть и Git-доступ пользователя MCP-сервера"),
        (("ошибка git (push)",),
         "сбой git push; проверьте write-доступ к remote и защиту ветки; заметка может остаться локально"),
        (("returned non-zero exit status", "calledprocesserror"),
         "сервер не смог выполнить Git-команду; обновите ObsidianBrain и проверьте его журнал"),
        (("unable to open database file", "disk i/o error"),
         "не удаётся открыть SQLite-индекс; проверьте путь, права и свободное место"),
        (("cannot lock ref", "unable to create", "index.lock"),
         "Git занят другим процессом или не может создать lock-файл; повторите позже"),
        (("git недоступен",), "Git недоступен или превышен таймаут; заметка может остаться локально"),
        (("таймаут",), "MCP-сервер не ответил за отведённое время; проверьте его журнал и повторите"),
        (("не удалось запустить", "не запущен"),
         "MCP-сервер не запущен; проверьте команду в .mcp.json и журнал сервера"),
        (("stage=sync",),
         "не удалось синхронизировать хранилище с origin; проверьте сеть, upstream и Git-доступ; заметка не создана"),
        (("конфликт request_id", "перезапись запрещена"),
         "конфликт с существующей заметкой; файл не перезаписан"),
    )
    for patterns, reason in reasons:
        if any(pattern in text for pattern in patterns):
            return reason
    stage = re.search(r"stage=(file|index|git|sync|validate|internal)(?::\s*([a-z_][a-z0-9_]*))?", text)
    if stage:
        name, exc_type = stage.group(1), stage.group(2) or "unknown"
        if name == "validate":
            return "некорректные данные записи или конфликт request_id; заметка не изменена"
        if name == "internal":
            return f"внутренняя ошибка MCP-сервера ({exc_type}); проверьте его журнал"
        # Имя стадии — фиксированный код из capture/git_publish, не текст ошибки.
        return (
            f"сбой на стадии {name} ({exc_type}); проверьте права на файл, индекс, "
            ".git и данные SQLite, затем повторите"
        )
    git_stage = re.search(r"ошибка git \(([a-z0-9-]+)\)", text)
    if git_stage:
        # Старый формат git_publish до появления stage=.
        return (
            f"сбой Git на стадии {git_stage.group(1)}; проверьте, что VAULT_PATH "
            "указывает на корень Git-клона, ветка не detached и есть права на .git"
        )
    return "неизвестная ошибка MCP-сервера; проверьте его журнал (исходный ответ скрыт во избежание утечки данных)"


def mcp_available(telegram_id: int | None) -> bool:
    """Чтение доступно всем пользователям обычного диалога, при включённом MCP."""
    if not config.enable_mcp or not config.mcp_servers:
        return False
    return telegram_id is not None


def mcp_write_available(telegram_id: int | None) -> bool:
    """Запись доступна только владельцам, независимо от настроек чтения."""
    return mcp_available(telegram_id) and config.is_owner(telegram_id)


READ_TOOLS = frozenset({"search_notes", "get_chunk", "get_note", "list_notes"})

# Внутренний инструмент отчёта о статусе. Он не входит в READ_TOOLS и никогда
# не предлагается модели: бот вызывает его сам, без аргументов, и получает
# только числовые счётчики (без путей и текста заметок).
STATUS_TOOL = "brain_status"


_ISOLATED_MCP_HINT = (
    "Для этого нового запроса ко второму мозгу вся история диалога исключена "
    "из контекста ради безопасности ранее загруженных материалов. Выполни "
    "только текущий запрос; если нужны сведения из прежнего диалога, попроси "
    "пользователя уточнить их. Не выдумывай недостающий контекст."
)


def isolate_mcp_history(messages: list[dict[str, Any]]) -> bool:
    """Изолировать явный чистый запрос к заметкам от недоверенной истории.

    Вызывается только перед первым раундом и при доступном MCP. Убирает всю
    прежнюю переписку, включая пересказы документов, но не меняет историю в
    БД. Вызывающий код обязан оставить только read-only MCP, без веб-поиска,
    генерации и отправки файлов. Внешний контент в текущем запросе и обычные
    уточнения не обходят защиту; результаты MCP снова включают общий гейт.
    Достаточно свежего запроса и явного указания на заметки: отдельный глагол
    действия больше не требуется («что там по проекту X в заметках?»).
    Маршрут считает общий ``request_scope.route``: явный веб-запрос, даже
    упоминающий Obsidian, остаётся веб-поиском и не перехватывается здесь.
    """
    if not messages or messages[-1].get("role") != "user":
        return False
    current = messages[-1]
    text = current.get("content")
    if (
        not isinstance(text, str)
        or messages_have_untrusted_content([current])
        or route(text) != "notes"
        or not messages_have_untrusted_content(messages[:-1])
    ):
        return False
    system = [dict(entry) for entry in messages if entry.get("role") == "system"]
    for entry in system:
        entry["content"] = f"{entry.get('content') or ''}\n\n{_ISOLATED_MCP_HINT}".strip()
    messages[:] = [*system, current]
    logger.info("Explicit second-brain request isolated from untrusted history")
    return True


def apply_mcp_hint(messages: list[dict[str, Any]]) -> None:
    """Append the "second brain" guidance to the system prompt, once."""
    for entry in messages:
        if entry.get("role") != "system":
            continue
        content = entry.get("content")
        if isinstance(content, str) and MCP_SYSTEM_HINT not in content:
            entry["content"] = f"{content}\n\n{MCP_SYSTEM_HINT}"
        return


def remove_mcp_hint(messages: list[dict[str, Any]]) -> None:
    """Drop the "second brain" guidance when its tools are withheld.

    Used when untrusted content disables MCP mid-conversation: the model must
    not be told about tools it cannot call in this request.
    """
    for entry in messages:
        if entry.get("role") != "system":
            continue
        content = entry.get("content")
        if isinstance(content, str) and MCP_SYSTEM_HINT in content:
            entry["content"] = content.replace(f"\n\n{MCP_SYSTEM_HINT}", "")
        return


class MCPClient:
    """A single stdio MCP connection (own subprocess and JSON-RPC channel)."""

    def __init__(self, server: MCPServerConfig, timeout: float) -> None:
        self.server = server
        self.timeout = timeout
        self._proc: asyncio.subprocess.Process | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._pending: dict[int, asyncio.Future[Any]] = {}
        self._next_id = 0
        self._write_lock = asyncio.Lock()
        self._start_lock = asyncio.Lock()
        self._tools: list[MCPTool] | None = None
        self.hidden_tools: list[str] = []  # offered by the server, not allowed
        self._closed = False

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    async def start(self) -> None:
        """Spawn the server (if needed) and complete the MCP handshake."""
        if self.running:
            return
        async with self._start_lock:
            if self.running or self._closed:
                return
            await self._stop()

            environment = _child_environment(self.server.env)
            command = _expand_env(self.server.command)
            if not os.path.isabs(command):
                if not config.mcp_allow_path_command:
                    raise MCPError(
                        f"MCP-сервер {self.server.name!r}: команда {command!r} "
                        "не является абсолютным путём. Укажите полный путь к "
                        "интерпретатору в .mcp.json (или задайте "
                        "MCP_ALLOW_PATH_COMMAND=true, если PATH доверенный)."
                    )
                logger.warning(
                    "MCP server %r resolves command %r through PATH; make sure "
                    "no untrusted directory precedes it",
                    self.server.name,
                    command,
                )
            args = [_expand_env(arg) for arg in self.server.args]
            cwd = _expand_env(self.server.cwd) if self.server.cwd else None
            try:
                self._proc = await asyncio.create_subprocess_exec(
                    command,
                    *args,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=environment,
                    cwd=cwd,
                    limit=_MAX_LINE_BYTES,
                )
            except (OSError, ValueError) as exc:
                raise MCPError(
                    f"не удалось запустить MCP-сервер {self.server.name!r}: {exc}"
                ) from exc

            self._reader_task = asyncio.create_task(
                self._read_stdout(), name=f"mcp-{self.server.name}-stdout"
            )
            self._stderr_task = asyncio.create_task(
                self._read_stderr(), name=f"mcp-{self.server.name}-stderr"
            )

            try:
                await self._request(
                    "initialize",
                    {
                        "protocolVersion": PROTOCOL_VERSION,
                        "capabilities": {},
                        "clientInfo": CLIENT_INFO,
                    },
                    # A server that cannot start quickly must not hold up a
                    # reply for the full tool timeout (60 s by default).
                    timeout=min(self.timeout, 20.0),
                )
                await self._notify("notifications/initialized", {})
            except Exception:
                # Do not leave a half-initialized process behind.
                await self._stop()
                raise
            logger.info("MCP server %r started", self.server.name)

    async def list_tools(self) -> list[MCPTool]:
        """Return the server's tools (all pages of ``tools/list``)."""
        if self._tools is not None and self.running:
            return self._tools
        await self.start()
        tools: list[MCPTool] = []
        hidden: list[str] = []
        cursor: str | None = None
        for _ in range(20):  # pagination guard
            params: dict[str, Any] = {}
            if cursor:
                params["cursor"] = cursor
            result = await self._request(
                "tools/list", params, timeout=min(self.timeout, 20.0)
            )
            if not isinstance(result, dict):
                break
            for item in result.get("tools") or []:
                if not isinstance(item, dict) or not item.get("name"):
                    continue
                name = str(item["name"])
                if not self.server.permits(name):
                    hidden.append(name)
                    continue
                tools.append(
                    MCPTool(
                        server=self.server.name,
                        name=name,
                        description=str(item.get("description") or ""),
                        input_schema=item.get("inputSchema") or {},
                        exposed_name=_exposed_name(self.server.name, name),
                    )
                )
            cursor = result.get("nextCursor")
            if not cursor:
                break
        self._tools = tools
        self.hidden_tools = hidden
        if hidden:
            logger.info(
                "MCP %r: %d tool(s) not in the allowlist and hidden: %s",
                self.server.name,
                len(hidden),
                ", ".join(hidden[:20]),
            )
        return tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Invoke one tool and flatten the result into text."""
        if not self.server.permits(name):  # defence in depth: never reach the server
            raise MCPError(f"MCP tool {name!r} is not allowed")
        await self.start()
        result = await self._request(
            "tools/call",
            {"name": name, "arguments": arguments},
        )
        return _format_tool_result(result)

    async def call_status_tool(self) -> str:
        """Read the second-brain counters for the status report.

        This intentionally bypasses ``MCP_ALLOWED_TOOLS``: the name is fixed
        (no user/model input reaches it), the call has no arguments and the
        server-side tool returns only counts. It is never added to the model's
        tool catalog, so this does not widen what the chat model may call.
        """
        await self.start()
        result = await self._request(
            "tools/call",
            {"name": STATUS_TOOL, "arguments": {}},
            timeout=min(self.timeout, 30.0),
        )
        if not isinstance(result, dict) or result.get("isError"):
            raise MCPError("brain status tool failed")
        # The real tool returns a tiny JSON object; cap a misbehaving server
        # before it can allocate a large string in the status path.
        return _format_tool_result(result)[:64_000]

    async def create_note(
        self, content: str, request_id: str, *, title: str = "", tags: list[str] | None = None,
    ) -> str:
        """Отдельный канал создания заметок, не доступный модели чата."""
        arguments: dict[str, Any] = {"content": content, "request_id": request_id}
        if title or tags is not None:
            arguments.update(title=title, tags=tags or [])
        try:
            await self.start()
            result = await self._request(
                "tools/call",
                {"name": "create_note", "arguments": arguments},
                # Создание/индекс + fetch/commit/push занимают дольше обычного чтения.
                timeout=max(self.timeout, 180.0),
            )
        except MCPError as exc:
            # Текст JSON-RPC-ошибки не выводим: он может содержать пути/секреты.
            raise MCPError(
                f"Не удалось создать заметку: {_note_error_reason({'content': [{'type': 'text', 'text': str(exc)}]})}"
            ) from None
        if not isinstance(result, dict) or result.get("isError"):
            raise MCPError(f"Не удалось создать заметку: {_note_error_reason(result)}")
        text = _format_tool_result(result)
        confirmation = result.get("structuredContent")
        if not isinstance(confirmation, dict):
            try:
                confirmation = json.loads(text)
            except (ValueError, TypeError):
                confirmation = None
        if not isinstance(confirmation, dict) or confirmation.get("git_pushed") is not True:
            error = confirmation.get("error") if isinstance(confirmation, dict) else None
            if isinstance(error, str) and error:
                raise MCPError(
                    f"Не удалось создать заметку: "
                    f"{_note_error_reason({'content': [{'type': 'text', 'text': error}]})}"
                )
            raise MCPError(
                "MCP-сервер не подтвердил публикацию заметки в Git; обновите ObsidianBrain"
            )
        return text

    async def close(self) -> None:
        """Stop the server process and fail any in-flight request."""
        self._closed = True
        await self._stop()

    # --- JSON-RPC plumbing -------------------------------------------------

    async def _request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> Any:
        if not self.running:
            raise MCPError(f"MCP-сервер {self.server.name!r} не запущен")
        self._next_id += 1
        request_id = self._next_id
        future: asyncio.Future[Any] = (
            asyncio.get_running_loop().create_future()
        )
        self._pending[request_id] = future
        try:
            await self._send(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "method": method,
                    "params": params or {},
                }
            )
            return await asyncio.wait_for(future, timeout=timeout or self.timeout)
        except TimeoutError as exc:
            raise MCPError(
                f"MCP-сервер {self.server.name!r}: таймаут {method}"
            ) from exc
        finally:
            self._pending.pop(request_id, None)

    async def _notify(self, method: str, params: dict[str, Any]) -> None:
        await self._send(
            {"jsonrpc": "2.0", "method": method, "params": params}
        )

    async def _send(self, payload: dict[str, Any]) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None or proc.returncode is not None:
            raise MCPError(f"MCP-сервер {self.server.name!r} не запущен")
        line = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        async with self._write_lock:
            try:
                proc.stdin.write((line + "\n").encode("utf-8"))
                await proc.stdin.drain()
            except (BrokenPipeError, ConnectionResetError) as exc:
                raise MCPError(
                    f"MCP-сервер {self.server.name!r} закрыл соединение"
                ) from exc

    async def _read_stdout(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        try:
            while True:
                raw = await proc.stdout.readline()
                if not raw:
                    break
                line = raw.decode("utf-8", "replace").strip()
                if not line:
                    continue
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    logger.debug(
                        "MCP %r: non-JSON output: %s",
                        self.server.name,
                        line[:200],
                    )
                    continue
                if not isinstance(message, dict):
                    continue
                if "method" in message:
                    # Server-initiated request/notification (sampling, roots…):
                    # this client only offers tools, so decline politely.
                    if "id" in message:
                        await self._send(
                            {
                                "jsonrpc": "2.0",
                                "id": message["id"],
                                "error": {
                                    "code": -32601,
                                    "message": "method not supported",
                                },
                            }
                        )
                    continue
                request_id = message.get("id")
                if request_id is None:
                    continue
                try:
                    pending_id = int(request_id)
                except (TypeError, ValueError):
                    continue
                future = self._pending.get(pending_id)
                if future is None or future.done():
                    continue
                if "error" in message:
                    error = message.get("error") or {}
                    future.set_exception(
                        MCPError(
                            f"MCP-сервер {self.server.name!r}: "
                            f"{error.get('message') or error}"
                        )
                    )
                else:
                    future.set_result(message.get("result"))
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - transport must not crash the bot
            logger.warning(
                "MCP %r: stdout reader stopped: %s", self.server.name, exc
            )
            self._fail_pending(f"MCP-сервер {self.server.name!r} отключился")
        finally:
            self._fail_pending(f"MCP-сервер {self.server.name!r} отключился")

    async def _read_stderr(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        try:
            while True:
                raw = await proc.stderr.readline()
                if not raw:
                    break
                logger.debug(
                    "MCP %r stderr: %s",
                    self.server.name,
                    raw.decode("utf-8", "replace").rstrip()[:500],
                )
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - stderr is diagnostics only
            return

    def _fail_pending(self, message: str) -> None:
        for future in self._pending.values():
            if not future.done():
                future.set_exception(MCPError(message))
        self._pending.clear()

    async def _stop(self) -> None:
        proc = self._proc
        tasks = [task for task in (self._reader_task, self._stderr_task) if task]
        self._reader_task = self._stderr_task = None
        self._proc = None
        self._tools = None

        if proc is not None and proc.returncode is None:
            if proc.stdin is not None and not proc.stdin.is_closing():
                with suppress(Exception):
                    proc.stdin.close()
            with suppress(ProcessLookupError):
                proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=3)
            except (TimeoutError, ProcessLookupError):
                with suppress(ProcessLookupError):
                    proc.kill()
                with suppress(Exception):
                    await proc.wait()

        for task in tasks:
            task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await task
        self._fail_pending("MCP-сервер остановлен")


class MCPManager:
    """All configured MCP servers behind one tool catalog."""

    def __init__(self) -> None:
        self._clients: dict[str, MCPClient] = {}
        self._tools: dict[str, tuple[MCPTool, MCPClient]] = {}
        self._loaded = False
        self._retry_at = 0.0
        self._lock = asyncio.Lock()
        # Counts change slowly; caching also stops repeated status requests
        # from rebuilding the candidate graph on every message.
        self._brain_status_cache: tuple[float, dict[str, Any]] | None = None

    def _client(self, name: str) -> MCPClient:
        client = self._clients.get(name)
        if client is None:
            raw = config.mcp_servers[name]
            server = MCPServerConfig(
                name=name,
                command=str(raw.get("command") or ""),
                args=[str(arg) for arg in raw.get("args") or []],
                env={
                    str(key): str(value)
                    for key, value in (raw.get("env") or {}).items()
                },
                cwd=raw.get("cwd"),
                allowed_tools=(
                    [str(tool) for tool in raw["allowed_tools"]]
                    if isinstance(raw.get("allowed_tools"), list)
                    else None
                ),
            )
            client = MCPClient(server, timeout=float(config.mcp_timeout))
            self._clients[name] = client
        return client

    async def _load(self, *, force: bool = False) -> None:
        if not force and (
            self._loaded
            or time.monotonic() < self._retry_at
        ):
            return
        async with self._lock:
            if not force and (
                self._loaded or time.monotonic() < self._retry_at
            ):
                return
            catalog: dict[str, tuple[MCPTool, MCPClient]] = {}
            for name in config.mcp_servers:
                client = self._client(name)
                try:
                    tools = await client.list_tools()
                except Exception as exc:  # noqa: BLE001 - one server must not break the bot
                    logger.warning("MCP server %r unavailable: %s", name, exc)
                    continue
                for tool in tools:
                    exposed = tool.exposed_name
                    if exposed in catalog:
                        digest = hashlib.sha1(
                            f"{name}:{tool.name}".encode()
                        ).hexdigest()[:6]
                        tool.exposed_name = f"{exposed[:57]}_{digest}"
                        exposed = tool.exposed_name
                    catalog[exposed] = (tool, client)
                logger.info("MCP server %r: %d tool(s)", name, len(tools))
            self._tools = catalog
            self._loaded = bool(catalog)
            # Do not respawn a broken server for every single message.
            self._retry_at = 0.0 if catalog else time.monotonic() + RETRY_COOLDOWN

    async def openai_tools(self) -> list[dict[str, Any]]:
        """The tool catalog in OpenAI function format (empty on failure)."""
        await self._load()
        return [
            tool.openai_schema() for tool, _ in self._tools.values()
            if tool.name in READ_TOOLS
        ]

    def is_mcp_tool(self, name: str) -> bool:
        """True when ``name`` belongs to a known MCP tool."""
        return name in self._tools

    def tool_info(self, name: str) -> MCPTool | None:
        """Metadata for an exposed MCP tool name."""
        entry = self._tools.get(name)
        return entry[0] if entry else None

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        max_chars: int | None = None,
        read_budget: ToolBudget | None = None,
    ) -> str:
        """Execute a tool call; always returns framed text, never raises.

        ``read_budget`` разрешает детерминированно дополнить карточки поиска
        полными текстами. Исходный вызов уже списан вызывающим кодом; каждое
        дополнительное чтение тоже списывается из общего бюджета. Модель не
        выбирает новые пути после получения внешнего текста.
        """
        if not self._loaded:
            await self._load()
        entry = self._tools.get(name)
        if entry is None:
            return wrap_untrusted(
                "mcp", f"Ошибка: неизвестный MCP-инструмент {name!r}."
            )
        tool, client = entry
        if tool.name not in READ_TOOLS:
            return wrap_untrusted("mcp", "Ошибка: в диалоге разрешено только чтение.")
        logger.info("MCP call: %s.%s", tool.server, tool.name)
        try:
            text = await client.call_tool(tool.name, arguments or {})
        except Exception as exc:  # noqa: BLE001 - tool failure must not kill a reply
            logger.warning(
                "MCP call %s.%s failed: %s", tool.server, tool.name, exc
            )
            return wrap_untrusted(
                "mcp", f"Ошибка MCP-инструмента {tool.name}: {exc}"
            )
        if read_budget is not None and tool.name in {"search_notes", "list_notes"}:
            return await self._read_found_notes(
                tool, text, read_budget,
                max_chars=min(
                    read_budget.context_chars if max_chars is None else max_chars,
                    read_budget.context_chars,
                ),
            )
        limit = config.mcp_max_result_chars
        if max_chars is not None:
            limit = max(1, min(limit, max_chars))
        if len(text) > limit:
            text = text[:limit] + f"\n…[обрезано {len(text) - limit} символов]"
        # Заметки и ошибки MCP — внешние данные. Тег не даёт injection закрыть
        # рамку, а следующий раунд снимет MCP-инструменты по общему taint-gate.
        return wrap_untrusted("mcp", text)

    async def _read_found_notes(
        self, search_tool: MCPTool, cards: str, budget: ToolBudget, *, max_chars: int,
    ) -> str:
        """Дополнить выдачу чтением только её верхнеуровневых JSON-карточек."""
        references = note_references(cards)
        readers = {
            tool.name: client for tool, client in self._tools.values()
            if tool.server == search_tool.server and tool.name in {"get_note", "get_chunk"}
        }
        limit = max(0, max_chars)
        card_limit = min(config.mcp_max_result_chars, max(200, limit // 4))
        blocks = ["Карточки найденных заметок (это ещё не полный текст):\n"
                  + clip_tool_text(cards, card_limit)]
        targets = references[:min(len(references), budget.mcp_calls)]
        per_note = min(
            config.mcp_max_result_chars,
            max(0, (limit - len(blocks[0]) - 300) // max(1, len(targets)) - 150),
        )
        if not targets or not readers or per_note < 200:
            blocks.append("Полные тексты не получены: нет проверенных карточек, "
                          "разрешённого чтения или бюджета. Не обещай дальнейшую загрузку.")
        for reference in targets if per_note >= 200 else []:
            body = ""
            kind = "Полный текст заметки"
            reader = readers.get("get_note")
            if reader is not None and budget.take_mcp():
                try:
                    body = await reader.call_tool("get_note", {"path": reference.path})
                except Exception:  # noqa: BLE001 - отказ чтения не теряет карточки
                    logger.warning("MCP automatic get_note failed (server=%s)", search_tool.server)
            if read_failed(body):
                body = ""
                reader = readers.get("get_chunk")
                if reader is not None and reference.chunk_ids and budget.take_mcp():
                    try:
                        body = await reader.call_tool("get_chunk", {"chunk_id": reference.chunk_ids[0]})
                    except Exception:  # noqa: BLE001 - финальный ответ объяснит неполный охват
                        logger.warning("MCP automatic get_chunk failed (server=%s)", search_tool.server)
                kind = "Фрагмент заметки (не полный текст)"
            if read_failed(body):
                blocks.append(f"{reference.path}: текст прочитать не удалось; доступна только карточка.")
                continue
            if len(body) > per_note:
                kind = "Частичный текст заметки (обрезан по бюджету)"
                body = clip_tool_text(body, per_note)
            blocks.append(f"{kind}: {reference.path}\n{body}")
        # Все карточки и тексты по-прежнему данные. На следующем раунде модель
        # получает их для анализа без права запускать инструменты из заметок.
        return clip_tool_text(wrap_untrusted("mcp", "\n\n".join(blocks)), limit)

    async def save_note(self, telegram_id: int, content: str, request_id: str) -> str:
        """Создать заметку по явному действию владельца; ошибки не маскируются."""
        if not mcp_write_available(telegram_id):
            raise MCPError("Запись разрешена только владельцу")
        if not content.strip() or len(content) > 100_000:
            raise MCPError("Недопустимый размер заметки")
        if not request_id or len(request_id) > 200:
            raise MCPError("Недопустимый request_id заметки")
        if config.mcp_write_server not in config.mcp_servers:
            raise MCPError("Сервер записи не настроен")
        async with user_service() as service:
            settings = await service.get_user_settings(telegram_id)
        metadata = await generate_note_metadata(content, settings["model"])
        return await self._client(config.mcp_write_server).create_note(
            content, request_id, title=metadata.title, tags=metadata.tags,
        )

    async def brain_status(self) -> dict[str, Any] | None:
        """Counts from the second-brain server for the status report.

        Returns ``None`` when no write/second-brain server is configured and a
        dict with ``error`` when it is unavailable. Only known numeric fields
        are passed through: note paths, summaries and raw MCP errors never
        reach the user-facing report.
        """
        name = config.mcp_write_server
        if name not in config.mcp_servers:
            return None
        cached = self._brain_status_cache
        if cached is not None:
            age = time.monotonic() - cached[0]
            ttl = 30.0 if cached[1].get("error") else 60.0
            if age < ttl:
                return cached[1]
        try:
            text = await self._client(name).call_status_tool()
        except Exception as exc:  # noqa: BLE001 - status must never break a reply
            logger.warning("MCP status %r failed: %s", name, type(exc).__name__)
            result: dict[str, Any] = {"error": "unavailable"}
        else:
            data = _json_object(text)
            if data is None:
                result = {"error": "invalid_response"}
            else:
                allowed = {
                    "notes",
                    "chunks",
                    "tags",
                    "links",
                    "strong_links",
                    "pending_links",
                    "failed_links",
                    "links_enabled",
                    "links_min_score",
                }
                result = {key: data[key] for key in allowed if key in data}
        self._brain_status_cache = (time.monotonic(), result)
        return result

    async def status(self) -> list[dict[str, Any]]:
        """Per-server diagnostic for the admin ``/mcp`` command."""
        report: list[dict[str, Any]] = []
        for name in config.mcp_servers:
            client = self._client(name)
            tools: list[str] = []
            hidden: list[str] = []
            error: str | None = None
            try:
                listed = await client.list_tools()
                tools = [tool.name for tool in listed if tool.name in READ_TOOLS]
                hidden = [tool.name for tool in listed if tool.name not in READ_TOOLS]
            except Exception as exc:  # noqa: BLE001 - diagnostics only
                error = str(exc)
            report.append(
                {
                    "name": name,
                    "running": client.running,
                    "tools": tools,
                    "hidden": list(client.hidden_tools) + hidden,
                    "error": error,
                }
            )
        return report

    async def close(self) -> None:
        """Stop every server (called on bot shutdown)."""
        for client in list(self._clients.values()):
            with suppress(Exception):
                await client.close()
        self._clients.clear()
        self._tools.clear()
        self._loaded = False
        self._brain_status_cache = None


mcp_manager = MCPManager()
