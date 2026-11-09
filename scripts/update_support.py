"""Служебные проверки автообновления, без импорта и запуска самого бота."""

from __future__ import annotations

import argparse
import ast
import asyncio
import json
import os
import re
import stat
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath


class UpdateCheckError(ValueError):
    """Контролируемая диагностика, которую можно показать владельцу без секретов."""


def _ownership_error(path: Path, info: os.stat_result) -> UpdateCheckError:
    return UpdateCheckError(
        f"Untrusted path {str(path)!r}: uid={info.st_uid}, mode={stat.S_IMODE(info.st_mode):04o}; "
        "expected root ownership and no group/other write access."
    )


def trusted_tree(root: Path, name: str) -> None:
    """Запретить root-деплой поверх кода, изменяемого сервисным пользователем."""
    if not root.is_dir() or root.is_symlink():
        raise UpdateCheckError(f"Expected a real code directory, not a symlink: {str(root)!r}.")
    for parent in (root, *root.parents):
        info = parent.stat()
        if info.st_uid != 0 or info.st_mode & 0o022:
            raise _ownership_error(parent, info)
    for directory, dirs, files in os.walk(root, followlinks=False):
        # Только data/ мозга принадлежит brain; его содержимое не исполняем.
        if Path(directory) == root and name == "brain":
            dirs[:] = [item for item in dirs if item != "data"]
        for item in (Path(directory), *(Path(directory) / item for item in dirs + files)):
            info = item.lstat()
            if info.st_uid != 0 or (not stat.S_ISLNK(info.st_mode) and info.st_mode & 0o022):
                raise _ownership_error(item, info)
            if stat.S_ISLNK(info.st_mode):
                target = item.resolve(strict=True)
                for ancestor in (target, *target.parents):
                    destination = ancestor.stat()
                    if destination.st_uid != 0 or destination.st_mode & 0o022:
                        raise _ownership_error(ancestor, destination)


def _tree_paths(root: Path, commit: str) -> set[str]:
    result = subprocess.run(
        ["git", "-c", "core.hooksPath=/dev/null", "-C", str(root), "ls-tree", "-rz", "--name-only", commit],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    return {os.fsdecode(item) for item in result.stdout.split(b"\0") if item}


def check_target(root: Path, old: str, new: str) -> None:
    """Не разрешать коммиту затереть конфиги, данные или untracked-файлы."""
    before, after = _tree_paths(root, old), _tree_paths(root, new)
    for name in after:
        path = PurePosixPath(name)
        first = path.parts[0]
        if (
            first in {".venv", "data", ".mcp.json", "telegraph_account.json"}
            or (first.startswith(".env") and first != ".env.example")
            or path.name.endswith((".db", ".db-wal", ".db-shm"))
            or path.name.startswith("errors.log")
        ):
            raise UpdateCheckError(f"Target commit tracks protected runtime path {name!r}.")
        if name not in before and os.path.lexists(root / name):
            raise UpdateCheckError(f"Target commit would overwrite local file {name!r}.")
        for parent in path.parents:
            if parent == PurePosixPath("."):
                break
            local = root / str(parent)
            if local.is_symlink() or (local.exists() and not local.is_dir()):
                raise UpdateCheckError(f"Target path crosses a symlink or local file: {str(parent)!r}.")


def check_syntax(paths: list[Path]) -> None:
    """Проверить Python без импорта кода и создания __pycache__."""
    for root in paths:
        if not root.is_dir():
            raise UpdateCheckError(f"Code directory is missing: {str(root)!r}.")
        for directory, dirs, files in os.walk(root, followlinks=False):
            dirs[:] = [name for name in dirs if name not in {".venv", ".git", "data", "__pycache__"}]
            for name in files:
                if name.endswith(".py"):
                    file = Path(directory) / name
                    try:
                        ast.parse(file.read_bytes(), filename=str(file))
                    except SyntaxError as exc:
                        raise UpdateCheckError(
                            f"SyntaxError in {str(file)!r}, line {exc.lineno or 0}; source omitted."
                        ) from None


def read_config(root: Path) -> dict[str, str]:
    """Повторить выбор одного .env и приоритет окружения из bot/config.py."""
    from dotenv import dotenv_values

    selected = root / ".env.linux"
    if not selected.is_file():
        selected = root / ".env"
    values = dotenv_values(selected) if selected.is_file() else {}
    return {**{key: value for key, value in values.items() if value is not None}, **os.environ}


def health_url(config: dict[str, str]) -> str:
    """Построить URL только для loopback; не превращать проверку в SSRF."""
    port = int(config.get("HEALTH_PORT", "0") or "0")
    if not 0 <= port <= 65535:
        raise UpdateCheckError("HEALTH_PORT must be between 0 and 65535.")
    if not port:
        return ""
    host = config.get("HEALTH_HOST", "127.0.0.1").strip()
    if host in {"0.0.0.0", "127.0.0.1", "localhost"}:
        host = "127.0.0.1"
    elif host in {"::", "::1"}:
        host = "[::1]"
    else:
        raise UpdateCheckError("HEALTH_HOST must be loopback or wildcard for auto-update.")
    return f"http://{host}:{port}/health"


def _no_proxy_opener() -> urllib.request.OpenerDirector:
    """Opener that ignores proxy settings from the environment."""
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def http_check(url: str) -> None:
    """Проверить JSON health-ответа без прокси из окружения."""
    opener = _no_proxy_opener()
    with opener.open(url, timeout=3) as response:
        if json.loads(response.read(8192)).get("status") != "ok":
            raise UpdateCheckError("Health JSON does not contain status=ok.")


def notify(config: dict[str, str], text: str) -> bool:
    """Отправить уведомление владельцам, не показывая токен даже при ошибке."""
    token = config.get("BOT_TOKEN", "").strip()
    raw_ids = config.get("OWNER_IDS", "").strip() or config.get("ADMIN_IDS", "")
    ids = {int(item.strip()) for item in raw_ids.split(",") if item.strip().isdigit() and int(item.strip()) > 0}
    if not token or not ids:
        return False
    success = True
    # BOT_TOKEN встроен в URL запроса: прокси из окружения не должен видеть его.
    opener = _no_proxy_opener()
    for owner in sorted(ids):
        data = urllib.parse.urlencode({"chat_id": owner, "text": text[:4000]}).encode()
        request = urllib.request.Request(f"https://api.telegram.org/bot{token}/sendMessage", data=data)
        try:
            with opener.open(request, timeout=10) as response:
                success = bool(json.loads(response.read(65536)).get("ok")) and success
        except Exception as exc:  # noqa: BLE001 — токен/URL и тело ошибки нельзя печатать
            # Прямая доставка без прокси может быть недоступна в сети сервера.
            # Фиксируем сам факт, не печатая токен, URL и текст исключения.
            print(
                f"Telegram notify failed for chat {owner}: {type(exc).__name__}",
                file=sys.stderr,
            )
            success = False
    return success


def mcp_error_summary(stderr: bytes) -> str:
    """Извлечь только безопасную категорию ошибки, не возвращая сырой stderr."""
    missing = re.search(rb"No module named ['\"]([A-Za-z0-9_.]{1,120})['\"]", stderr)
    if missing:
        return f"Missing Python module: {missing.group(1).decode('ascii')}. Check brain .venv dependencies."
    if b"PermissionError" in stderr:
        return "PermissionError: check brain access to code, .venv, .env and data/."
    if b"VAULT_PATH" in stderr:
        return "Check VAULT_PATH in brain .env; vault paths were omitted."
    if b"SyntaxError" in stderr:
        return "SyntaxError during server startup; source omitted."
    return "Server stderr omitted; check brain configuration and dependencies."


async def check_mcp(python: Path, root: Path) -> None:
    """Выполнить MCP initialize/tools/list без LLM, записи заметок и чтения БД."""
    allowed = {"PATH", "HOME", "LANG", "LANGUAGE", "TZ"}
    env = {key: value for key, value in os.environ.items() if key in allowed or key.startswith("LC_")}
    env.update(PYTHONDONTWRITEBYTECODE="1", BRAIN_WRITE_ENABLED="false", LOG_FILE="off")
    proc = await asyncio.create_subprocess_exec(
        str(python), str(root / "mcp_server.py"), cwd=root, env=env,
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    stderr = bytearray()

    async def drain_stderr() -> None:
        assert proc.stderr is not None
        while chunk := await proc.stderr.read(4096):
            # Дренируем весь поток, но сохраняем только ограниченный префикс.
            stderr.extend(chunk[:max(0, 16384 - len(stderr))])

    stderr_task = asyncio.create_task(drain_stderr())

    async def send(message: dict) -> None:
        assert proc.stdin is not None
        proc.stdin.write(json.dumps({"jsonrpc": "2.0", **message}).encode() + b"\n")
        await proc.stdin.drain()

    async def response(expected: int) -> dict:
        assert proc.stdout is not None
        while line := await proc.stdout.readline():
            message = json.loads(line)
            if message.get("id") == expected:
                if "error" in message or not isinstance(message.get("result"), dict):
                    raise UpdateCheckError("Invalid MCP handshake response; response body omitted.")
                return message["result"]
        raise UpdateCheckError("MCP server exited before replying.")

    failure: Exception | None = None
    phase = "initialize"
    try:
        async with asyncio.timeout(30):
            await send({"id": 1, "method": "initialize", "params": {
                "protocolVersion": "2024-11-05", "capabilities": {},
                "clientInfo": {"name": "kapum-update", "version": "1"},
            }})
            await response(1)
            await send({"method": "notifications/initialized"})
            phase = "tools/list"
            await send({"id": 2, "method": "tools/list", "params": {}})
            result = await response(2)
            names = {tool.get("name") for tool in result.get("tools", [])}
            if not {"search_notes", "get_chunk", "get_note", "list_notes"} <= names:
                raise UpdateCheckError("Required MCP tools are missing (search_notes/get_chunk/get_note/list_notes).")
    except Exception as exc:  # noqa: BLE001 — публикуем только тип и безопасную категорию
        failure = exc
    finally:
        if proc.stdin:
            proc.stdin.close()
        try:
            await asyncio.wait_for(proc.wait(), timeout=2)
        except TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            await proc.wait()
        try:
            await asyncio.wait_for(stderr_task, timeout=2)
        except TimeoutError:
            stderr_task.cancel()
    if failure is not None:
        reason = str(failure) if isinstance(failure, UpdateCheckError) else type(failure).__name__
        raise UpdateCheckError(
            f"MCP {phase} failed ({reason}; server exit={proc.returncode}). {mcp_error_summary(bytes(stderr))}"
        ) from None


def check_error_summary(exc: Exception) -> str:
    """Сформировать диагностику без str() произвольного исключения с секретами."""
    if isinstance(exc, UpdateCheckError):
        return str(exc)[:700]
    if isinstance(exc, OSError) and isinstance(exc.filename, str):
        return f"{type(exc).__name__} for path {exc.filename!r} (errno={exc.errno})."
    return f"{type(exc).__name__}; exception details omitted."


def main() -> int:
    """Запустить одну служебную команду без вывода секретов/содержимого файлов."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("trusted", "target", "syntax", "health-url", "http-check", "notify", "mcp-check"))
    parser.add_argument("args", nargs="+")
    parsed = parser.parse_args()
    args = parsed.args
    try:
        if parsed.command == "trusted":
            trusted_tree(Path(args[0]), args[1])
        elif parsed.command == "target":
            check_target(Path(args[0]), args[1], args[2])
        elif parsed.command == "syntax":
            check_syntax([Path(item) for item in args])
        elif parsed.command == "health-url":
            print(health_url(read_config(Path(args[0]))))
        elif parsed.command == "http-check":
            http_check(args[0])
        elif parsed.command == "notify":
            return 0 if notify(read_config(Path(args[0])), args[1]) else 1
        elif parsed.command == "mcp-check":
            asyncio.run(check_mcp(Path(args[0]), Path(args[1])))
    except Exception as exc:  # noqa: BLE001 — не печатаем секреты, stderr MCP или исходный код
        print(f"Update check failed: {parsed.command}: {check_error_summary(exc)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
