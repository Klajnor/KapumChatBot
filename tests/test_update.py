"""Автообновление: локальные Git-репозитории и моки вместо сервера/Telegram."""

from __future__ import annotations

import asyncio
import io
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.update_support import (
    check_mcp,
    check_syntax,
    check_target,
    health_url,
    notify,
    read_config,
    trusted_tree,
)

ROOT = Path(__file__).resolve().parents[1]
BASH = (
    str(Path("C:/Program Files/Git/bin/bash.exe"))
    if os.name == "nt" and Path("C:/Program Files/Git/bin/bash.exe").is_file()
    else shutil.which("bash")
)


def git(root: Path, *args: str) -> str:
    """Выполнить Git только в явно переданном временном клоне."""
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True, encoding="utf-8", stderr=subprocess.DEVNULL,
    ).strip()


def make_repo(root: Path) -> str:
    """Создать временный репозиторий с изолированными именем и email."""
    root.mkdir()
    git(root, "init", "-b", "master")
    git(root, "config", "user.name", "Update test")
    git(root, "config", "user.email", "test@example.invalid")
    git(root, "config", "core.autocrlf", "false")
    (root / ".gitignore").write_text(".env\n.venv/\ndata/\n", encoding="utf-8")
    (root / "module.py").write_text("value = 1\n", encoding="utf-8")
    (root / "requirements.txt").write_text("old-package\n", encoding="utf-8")
    git(root, "add", ".")
    git(root, "commit", "-m", "old")
    return git(root, "rev-parse", "HEAD")


def candidate(root: Path, files: dict[str, str]) -> str:
    """Подготовить следующий коммит, оставив master на старом."""
    git(root, "switch", "-c", "candidate")
    for name, text in files.items():
        file = root / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(text, encoding="utf-8")
        git(root, "add", "-f", "--", name)
    git(root, "commit", "-m", "new")
    result = git(root, "rev-parse", "HEAD")
    git(root, "switch", "master")
    return result


def bash_path(path: Path | str) -> str:
    """Представить путь для Git Bash, не подменяя пути в рабочих файлах."""
    value = Path(path).as_posix()
    if os.name == "nt" and len(value) > 1 and value[1] == ":":
        value = f"/{value[0].lower()}{value[2:]}"
    return shlex.quote(value)


class UpdateSupportTests(unittest.TestCase):
    def test_service_owned_code_is_rejected_before_git_or_pip(self):
        root = Path("/fake-code")
        root_info = SimpleNamespace(st_uid=0, st_mode=0o40755)
        service_info = SimpleNamespace(st_uid=1000, st_mode=0o100644)

        def info(path):
            return service_info if path.name == "module.py" else root_info

        with (
            patch.object(Path, "stat", return_value=root_info),
            patch.object(Path, "lstat", info),
            patch("os.walk", return_value=[(str(root), ["data"], ["module.py"])]),
            self.assertRaises(ValueError),
        ):
            trusted_tree(root, "brain")

    def test_config_selects_one_platform_file_and_environment_wins(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            root = Path(tmp)
            (root / ".env").write_text("OWNER_IDS=999\nBOT_TOKEN=wrong\n", encoding="utf-8")
            (root / ".env.linux").write_text('export BOT_TOKEN = "right" # comment\nHEALTH_PORT=8080\n', encoding="utf-8")
            config = read_config(root)
            self.assertEqual(config["BOT_TOKEN"], "right")
            self.assertNotIn("OWNER_IDS", config)
            with patch.dict(os.environ, {"BOT_TOKEN": "environment"}):
                self.assertEqual(read_config(root)["BOT_TOKEN"], "environment")

    def test_health_url_only_uses_loopback(self):
        self.assertEqual(health_url({}), "")
        self.assertEqual(health_url({"HEALTH_PORT": "8080", "HEALTH_HOST": "0.0.0.0"}), "http://127.0.0.1:8080/health")
        self.assertEqual(health_url({"HEALTH_PORT": "8080", "HEALTH_HOST": "::"}), "http://[::1]:8080/health")
        for config in ({"HEALTH_PORT": "65536"}, {"HEALTH_PORT": "8", "HEALTH_HOST": "example.org"}):
            with self.assertRaises(ValueError):
                health_url(config)

    def test_notify_validates_private_owners_and_does_not_print_token(self):
        config = {"BOT_TOKEN": "SECRET", "OWNER_IDS": "-100,invalid,123,123,456"}
        response = unittest.mock.MagicMock()
        response.__enter__.return_value.read.return_value = b'{"ok":true}'
        with patch("urllib.request.urlopen", return_value=response) as send:
            self.assertTrue(notify(config, "updated"))
            self.assertEqual(send.call_count, 2)
            self.assertIn(b"chat_id=123", send.call_args_list[0].args[0].data)
        output = io.StringIO()
        with patch("urllib.request.urlopen", side_effect=OSError("SECRET")), patch("sys.stderr", output):
            self.assertFalse(notify(config, "failure"))
        self.assertNotIn("SECRET", output.getvalue())

    def test_syntax_gate_does_not_import_modules_or_write_bytecode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "module.py").write_text('raise RuntimeError("must not execute")\n', encoding="utf-8")
            (root / ".venv").mkdir()
            (root / ".venv/broken.py").write_text("broken = (", encoding="utf-8")
            check_syntax([root])
            self.assertFalse((root / "__pycache__").exists())
            (root / "module.py").write_text("broken = (", encoding="utf-8")
            with self.assertRaises(SyntaxError):
                check_syntax([root])

    def test_target_preserves_ignored_files_and_protected_paths(self):
        for name in (".env", "data/brain.db", ".venv/new.py", ".mcp.json", "local.py"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp) / "repo"
                old = make_repo(root)
                new = candidate(root, {name: "remote"})
                if name == "local.py":
                    (root / name).write_text("local", encoding="utf-8")
                with self.assertRaises(ValueError):
                    check_target(root, old, new)
                self.assertEqual(git(root, "rev-parse", "HEAD"), old)

    def test_regular_new_files_are_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            old = make_repo(root)
            new = candidate(root, {"module.py": "value = 2\n", "new.py": "pass\n"})
            check_target(root, old, new)

    def test_real_stdio_handshake_drops_secrets_and_disables_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "mcp_server.py").write_text(
                "import json, os, sys\n"
                "assert 'BOT_TOKEN' not in os.environ and 'LLM_API_KEY' not in os.environ\n"
                "assert os.environ['BRAIN_WRITE_ENABLED'] == 'false'\n"
                "for line in sys.stdin:\n"
                "    req = json.loads(line)\n"
                "    if 'id' not in req: continue\n"
                "    result = {} if req['method'] == 'initialize' else {'tools': [{'name': name} for name in "
                "['search_notes', 'get_note', 'get_chunk', 'list_notes']]}\n"
                "    print(json.dumps({'jsonrpc':'2.0','id':req['id'],'result':result}), flush=True)\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"BOT_TOKEN": "SECRET", "LLM_API_KEY": "SECRET", "BRAIN_WRITE_ENABLED": "true"}):
                asyncio.run(check_mcp(Path(sys.executable), root))
            (root / "mcp_server.py").write_text("raise ImportError('missing dependency')\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                asyncio.run(check_mcp(Path(sys.executable), root))


@unittest.skipUnless(BASH and shutil.which("git"), "Нужны Bash и Git")
class UpdateShellTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bot = self.root / "bot"
        self.brain = self.root / "brain"
        self.bot_old = make_repo(self.bot)
        self.brain_old = make_repo(self.brain)
        for repo in (self.bot, self.brain):
            (repo / ".venv").mkdir()
            (repo / ".venv/version").write_text("original", encoding="utf-8")
        (self.bot / "bot").mkdir()
        (self.bot / "scripts").mkdir()
        (self.root / "state").mkdir()
        (self.root / "run").mkdir()
        (self.root / "key").write_text("fake-key", encoding="utf-8")
        source = (ROOT / "scripts/update.sh").read_text(encoding="utf-8")
        # Загружаем те же определения в память, но без root-only _main.
        self.source = source[source.index("set -euo pipefail"):source.rindex('\n_main "$@"')]
        self.preamble = f"""
APP_DIR={bash_path(self.bot)}
BRAIN_DIR={bash_path(self.brain)}
STATE_DIR={bash_path(self.root / 'state')}
RUN_DIR={bash_path(self.root / 'run')}
SUPPORT={bash_path(ROOT / 'scripts/update_support.py')}
BOT_PY={bash_path(sys.executable)}
BRAIN_PY="$BOT_PY"
KEY={bash_path(self.root / 'key')}
CALLS={bash_path(self.root / 'calls')}
_system_support() {{
    if [ "$1" = trusted ]; then return 0; fi
    "$BOT_PY" -I "$SUPPORT" "$@"
}}
_notify() {{ printf '%s\\n' "$1" >> "$CALLS"; }}
_fetch() {{ return 0; }}
systemctl() {{ printf '%s\\n' "$*" >> "$CALLS"; }}
"""

    def run_bash(self, code: str, expected: int = 0) -> subprocess.CompletedProcess:
        result = subprocess.run(
            [BASH, "--noprofile", "--norc", "-s"], input=self.source + self.preamble + code,
            text=True, encoding="utf-8", errors="replace", capture_output=True, check=False, cwd=self.root, timeout=30,
        )
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        return result

    def prepare_remote(self, repo: Path, new: str):
        name = "KapumChatBot" if repo == self.bot else "ObsidianBrain"
        git(repo, "remote", "add", "origin", f"git@github.com:Klajnor/{name}.git")
        git(repo, "update-ref", "refs/remotes/origin/master", new)

    def test_no_changes_does_not_stop_service(self):
        self.prepare_remote(self.bot, self.bot_old)
        self.run_bash('''
_check_repo bot "$APP_DIR" "$KEY" "$BOT_PY" Klajnor/KapumChatBot.git
test "$REPO_UPD" = 0
test "$(_state bot)" = ok
test ! -f "$CALLS"
''')

    def test_dirty_repo_is_skipped_but_other_repo_can_update(self):
        bot_new = candidate(self.bot, {"module.py": "value = 2\n"})
        brain_new = candidate(self.brain, {"module.py": "value = 2\n"})
        self.prepare_remote(self.bot, bot_new)
        self.prepare_remote(self.brain, brain_new)
        (self.bot / "module.py").write_text("local user work\n", encoding="utf-8")
        self.run_bash('''
_check_repo bot "$APP_DIR" "$KEY" "$BOT_PY" Klajnor/KapumChatBot.git
test "$REPO_UPD" = 0
_check_repo brain "$BRAIN_DIR" "$KEY" "$BRAIN_PY" Klajnor/ObsidianBrain.git
test "$REPO_UPD" = 1
test "$BLOCKED" = 1
''')
        self.assertEqual((self.bot / "module.py").read_text(encoding="utf-8"), "local user work\n")

    def test_local_commits_are_never_discarded(self):
        new = candidate(self.bot, {"module.py": "value = 2\n"})
        self.prepare_remote(self.bot, new)
        (self.bot / "local.py").write_text("pass\n", encoding="utf-8")
        git(self.bot, "add", "local.py")
        git(self.bot, "commit", "-m", "local")
        head = git(self.bot, "rev-parse", "HEAD")
        self.run_bash('''
_check_repo bot "$APP_DIR" "$KEY" "$BOT_PY" Klajnor/KapumChatBot.git
test "$REPO_UPD" = 0
test "$BLOCKED" = 1
''')
        self.assertEqual(git(self.bot, "rev-parse", "HEAD"), head)

    def test_failed_commit_is_not_retried(self):
        new = candidate(self.bot, {"module.py": "value = 2\n"})
        self.prepare_remote(self.bot, new)
        (self.root / "state/bot.state").write_text(f"failed:{new}\n", encoding="utf-8")
        self.run_bash('''
_check_repo bot "$APP_DIR" "$KEY" "$BOT_PY" Klajnor/KapumChatBot.git
test "$REPO_UPD" = 0
test ! -f "$CALLS"
''')

    def test_fetch_failure_is_non_destructive_and_notifies_once(self):
        self.prepare_remote(self.bot, self.bot_old)
        self.run_bash('''
_fetch() { return 1; }
_check_repo bot "$APP_DIR" "$KEY" "$BOT_PY" Klajnor/KapumChatBot.git
_check_repo bot "$APP_DIR" "$KEY" "$BOT_PY" Klajnor/KapumChatBot.git
test "$REPO_UPD" = 0
test "$BLOCKED" = 1
''')
        self.assertEqual(git(self.bot, "rev-parse", "HEAD"), self.bot_old)
        calls = (self.root / "calls").read_text(encoding="utf-8")
        self.assertEqual(calls.count("git fetch failed"), 1)

    def test_syntax_failure_restores_brain_without_touching_data(self):
        new = candidate(self.brain, {"module.py": "value = (\n"})
        (self.brain / "data").mkdir()
        (self.brain / "data/brain.db").write_bytes(b"unchanged database")
        self.run_bash(f'''
UPD_BRAIN=1 BRAIN_OLD={self.brain_old} BRAIN_NEW={new}
_restart_services() {{ echo restart >> "$CALLS"; }}
if _apply_update; then exit 9; fi
_failed_update
''')
        self.assertEqual(git(self.brain, "rev-parse", "HEAD"), self.brain_old)
        self.assertEqual((self.brain / "data/brain.db").read_bytes(), b"unchanged database")

    def test_concurrent_local_commit_is_not_reset_by_rollback(self):
        new = candidate(self.brain, {"module.py": "value = 2\n"})
        (self.brain / "local.py").write_text("pass\n", encoding="utf-8")
        git(self.brain, "add", "local.py")
        git(self.brain, "commit", "-m", "concurrent administrator work")
        local_head = git(self.brain, "rev-parse", "HEAD")
        self.run_bash(f'''
UPD_BRAIN=1 BRAIN_OLD={self.brain_old} BRAIN_NEW={new}
_restart_services() {{ return 0; }}
if _apply_update; then exit 9; fi
test "$BRAIN_TOUCHED" = 0
_failed_update
''')
        self.assertEqual(git(self.brain, "rev-parse", "HEAD"), local_head)

    def test_both_repos_rollback_with_exact_venv_after_pip_failure(self):
        bot_new = candidate(self.bot, {"module.py": "value = 2\n", "requirements.txt": "new-package\n"})
        brain_new = candidate(self.brain, {"module.py": "value = 2\n", "requirements.txt": "new-package\n"})
        self.run_bash(f'''
UPD_BOT=1 UPD_BRAIN=1 BOT_REQ=1 BRAIN_REQ=1
BOT_OLD={self.bot_old} BOT_NEW={bot_new}
BRAIN_OLD={self.brain_old} BRAIN_NEW={brain_new}
_install_deps() {{
    echo modified > "$APP_DIR/.venv/version"
    echo added > "$APP_DIR/.venv/new-package"
    return 1
}}
_restart_services() {{ echo restart >> "$CALLS"; }}
if _apply_update; then exit 9; fi
_failed_update
test "$TRANSACTION" = 0
test "$(_state bot)" = failed:{bot_new}
test "$(_state brain)" = failed:{brain_new}
''')
        for repo, old in ((self.bot, self.bot_old), (self.brain, self.brain_old)):
            self.assertEqual(git(repo, "rev-parse", "HEAD"), old)
            self.assertEqual((repo / ".venv/version").read_text(encoding="utf-8"), "original")
            self.assertFalse((repo / ".venv/new-package").exists())
        calls = (self.root / "calls").read_text(encoding="utf-8")
        self.assertIn("Previous code and environments restored", calls)

    def test_brain_only_change_stops_bot_before_updating_and_keeps_bot_head(self):
        new = candidate(self.brain, {"module.py": "value = 2\n"})
        self.run_bash(f'''
UPD_BRAIN=1 BRAIN_OLD={self.brain_old} BRAIN_NEW={new}
_restart_services() {{ echo restarted >> "$CALLS"; }}
_apply_update
test "$BRAIN_TOUCHED" = 1
test "$BOT_TOUCHED" = 0
''')
        self.assertEqual(git(self.bot, "rev-parse", "HEAD"), self.bot_old)
        self.assertEqual(git(self.brain, "rev-parse", "HEAD"), new)
        self.assertTrue((self.root / "calls").read_text(encoding="utf-8").startswith("stop chat_bot.service\n"))

    def test_mcp_startup_failure_rolls_back_even_when_bot_is_healthy(self):
        new = candidate(self.brain, {"module.py": "value = 2\n"})
        (self.bot / "restart.sh").write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
        self.run_bash(f'''
UPD_BRAIN=1 BRAIN_OLD={self.brain_old} BRAIN_NEW={new}
_wait_healthy() {{ return 0; }}
timeout() {{ [ "$(_git "$BRAIN_DIR" rev-parse HEAD)" = "$BRAIN_OLD" ]; }}
if _apply_update; then exit 9; fi
_failed_update
test "$KEEP_BACKUP" = 0
''')
        self.assertEqual(git(self.brain, "rev-parse", "HEAD"), self.brain_old)
        self.assertIn("Previous code and environments restored", (self.root / "calls").read_text(encoding="utf-8"))

    def test_git_update_never_runs_repository_hooks(self):
        new = candidate(self.brain, {"module.py": "value = 2\n"})
        hook = self.brain / ".git/hooks/post-merge"
        sentinel = self.root / "hook-ran"
        hook.write_text(f"#!/bin/sh\ntouch {bash_path(sentinel)}\n", encoding="utf-8")
        hook.chmod(0o755)
        self.run_bash(f'_git "$BRAIN_DIR" merge --ff-only --no-edit {new}\n')
        self.assertFalse(sentinel.exists())

    def test_health_wait_is_relative_even_after_long_fetch(self):
        self.run_bash('''
SECONDS=100
_active() { return 0; }
_unit_fingerprint() { echo steady; }
_support() { return 0; }
sleep() { SECONDS=$((SECONDS + $1)); }
before=$SECONDS
_wait_healthy
test "$((SECONDS - before))" -ge 20
''')

    def test_crash_loop_never_counts_as_healthy(self):
        self.run_bash('''
SECONDS=100
_active() { return 0; }
_unit_fingerprint() { if [ "$SECONDS" -gt 100 ]; then echo restarted; else echo initial; fi; }
_support() { return 0; }
sleep() { SECONDS=$((SECONDS + $1)); }
if _wait_healthy; then exit 9; fi
''')

    def test_running_pull_is_deferred_without_killing_indexer(self):
        self.run_bash('''
UPD_BRAIN=1
_active() { [ "$1" = "$PULL_TIMER" ]; }
systemctl() {
    if [ "$1" = show ]; then echo activating; else echo "$*" >> "$CALLS"; fi
}
_pause_indexer
test "$UPD_BRAIN" = 0
_resume_timer
''')
        self.assertEqual((self.root / "calls").read_text(encoding="utf-8"), "stop brain-pull.timer\nstart brain-pull.timer\n")

    def test_incomplete_rollback_is_reported_and_backups_retained(self):
        self.run_bash('''
UPD_BRAIN=1 BRAIN_TOUCHED=1 BRAIN_NEW=new BRAIN_OLD=old
_git() { return 1; }
_restore_venv() { return 0; }
_restart_services() { return 0; }
_failed_update
test "$KEEP_BACKUP" = 1
''')
        self.assertIn("rollback is incomplete", (self.root / "calls").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
