"""Служебные проверки автообновления, без импорта и запуска самого бота."""

from __future__ import annotations

import argparse
import ast
import asyncio
import json
import os
import stat
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath


def trusted_tree(root: Path, name: str) -> None:
    """Запретить root-деплой поверх кода, изменяемого сервисным пользователем."""
    if not root.is_dir() or root.is_symlink():
        raise ValueError("Нет обычного каталога кода")
    for parent in (root, *root.parents):
        info = parent.stat()
        if info.st_uid != 0 or info.st_mode & 0o022:
            raise ValueError("Небезопасные права родительского каталога")
    for directory, dirs, files in os.walk(root, followlinks=False):
        # Только data/ мозга принадлежит brain; его содержимое не исполняем.
        if Path(directory) == root and name == "brain":
            dirs[:] = [item for item in dirs if item != "data"]
        for item in (Path(directory), *(Path(directory) / item for item in dirs + files)):
            info = item.lstat()
            if info.st_uid != 0 or (not stat.S_ISLNK(info.st_mode) and info.st_mode & 0o022):
                raise ValueError("Код, конфиги, .git и .venv должны быть root-owned без записи для группы")
            if stat.S_ISLNK(info.st_mode):
                target = item.resolve(strict=True)
                for ancestor in (target, *target.parents):
                    destination = ancestor.stat()
                    if destination.st_uid != 0 or destination.st_mode & 0o022:
                        raise ValueError("Ссылка из кода ведёт в изменяемый сервисом путь")


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
            raise ValueError("Коммит отслеживает защищённый путь")
        if name not in before and os.path.lexists(root / name):
            raise ValueError("Коммит затрёт локальный файл")
        for parent in path.parents:
            if parent == PurePosixPath("."):
                break
            local = root / str(parent)
            if local.is_symlink() or (local.exists() and not local.is_dir()):
                raise ValueError("Коммит проходит через ссылку или локальный файл")


def check_syntax(paths: list[Path]) -> None:
    """Проверить Python без импорта кода и создания __pycache__."""
    for root in paths:
        if not root.is_dir():
            raise ValueError("Нет каталога кода")
        for directory, dirs, files in os.walk(root, followlinks=False):
            dirs[:] = [name for name in dirs if name not in {".venv", ".git", "data", "__pycache__"}]
            for name in files:
                if name.endswith(".py"):
                    file = Path(directory) / name
                    ast.parse(file.read_bytes(), filename=str(file))


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
        raise ValueError("Некорректный HEALTH_PORT")
    if not port:
        return ""
    host = config.get("HEALTH_HOST", "127.0.0.1").strip()
    if host in {"0.0.0.0", "127.0.0.1", "localhost"}:
        host = "127.0.0.1"
    elif host in {"::", "::1"}:
        host = "[::1]"
    else:
        raise ValueError("Для автообновления HEALTH_HOST должен быть loopback или wildcard")
    return f"http://{host}:{port}/health"


def http_check(url: str) -> None:
    """Проверить JSON health-ответа без прокси из окружения."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=3) as response:
        if json.loads(response.read(8192)).get("status") != "ok":
            raise ValueError("Health не готов")


def notify(config: dict[str, str], text: str) -> bool:
    """Отправить уведомление владельцам, не показывая токен даже при ошибке."""
    token = config.get("BOT_TOKEN", "").strip()
    raw_ids = config.get("OWNER_IDS", "").strip() or config.get("ADMIN_IDS", "")
    ids = {int(item.strip()) for item in raw_ids.split(",") if item.strip().isdigit() and int(item.strip()) > 0}
    if not token or not ids:
        return False
    success = True
    for owner in sorted(ids):
        data = urllib.parse.urlencode({"chat_id": owner, "text": text[:4000]}).encode()
        request = urllib.request.Request(f"https://api.telegram.org/bot{token}/sendMessage", data=data)
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                success = bool(json.loads(response.read(65536)).get("ok")) and success
        except Exception:  # noqa: BLE001 — токен/URL и тело ошибки нельзя печатать
            success = False
    return success


async def check_mcp(python: Path, root: Path) -> None:
    """Выполнить MCP initialize/tools/list без LLM, записи заметок и чтения БД."""
    allowed = {"PATH", "HOME", "LANG", "LANGUAGE", "TZ"}
    env = {key: value for key, value in os.environ.items() if key in allowed or key.startswith("LC_")}
    env.update(PYTHONDONTWRITEBYTECODE="1", BRAIN_WRITE_ENABLED="false", LOG_FILE="off")
    proc = await asyncio.create_subprocess_exec(
        str(python), str(root / "mcp_server.py"), cwd=root, env=env,
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
    )

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
                    raise ValueError("Ошибка MCP handshake")
                return message["result"]
        raise ValueError("MCP завершился до ответа")

    try:
        async with asyncio.timeout(30):
            await send({"id": 1, "method": "initialize", "params": {
                "protocolVersion": "2024-11-05", "capabilities": {},
                "clientInfo": {"name": "kapum-update", "version": "1"},
            }})
            await response(1)
            await send({"method": "notifications/initialized"})
            await send({"id": 2, "method": "tools/list", "params": {}})
            result = await response(2)
            names = {tool.get("name") for tool in result.get("tools", [])}
            if not {"search_notes", "get_chunk", "get_note", "list_notes"} <= names:
                raise ValueError("Нет обязательных инструментов MCP")
    finally:
        if proc.stdin:
            proc.stdin.close()
        try:
            await asyncio.wait_for(proc.wait(), timeout=2)
        except TimeoutError:
            proc.kill()
            await proc.wait()


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
        print(f"Проверка {parsed.command} не пройдена ({type(exc).__name__}).", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
