"""Загрузка Git-ключа: ошибки и изоляция сессий, без настоящих ключей/агентов."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASH = (
    str(Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Git/bin/bash.exe")
    if os.name == "nt"
    else shutil.which("bash")
)

HARNESS = r"""
set -euo pipefail
trap ':' INT TERM
ssh-add() {
    printf 'ssh-add %s\n' "$*" >> calls.log
    if [[ "$1" == -l ]]; then
        return "$AGENT_STATUS"
    fi
    return "$ADD_STATUS"
}
ssh-agent() {
    printf 'ssh-agent %s\n' "$*" >> calls.log
    if [[ "$1" == -s ]]; then
        printf '%s\n' "$AGENT_OUTPUT"
        return "$START_STATUS"
    fi
    return "$STOP_STATUS"
}
command() {
    if [[ "$1" == -v && "$2" == "$MISSING_COMMAND" ]]; then
        return 1
    fi
    builtin command "$@"
}
export -f ssh-add ssh-agent command
before_options=$-
before_traps=$(trap -p INT TERM)
if [[ "$DIRECT" == 1 ]]; then
    if "$BASH" ./load_git_key.sh; then result=0; else result=$?; fi
else
    if source ./load_git_key.sh; then result=0; else result=$?; fi
fi
[[ "$before_options" == "$-" ]] || exit 90
[[ "$before_traps" == "$(trap -p INT TERM)" ]] || exit 91
if declare -F _load_git_key_main >/dev/null; then exit 92; fi
printf 'SOCK=%s\nPID=%s\n' "${SSH_AUTH_SOCK-unset}" "${SSH_AGENT_PID-unset}"
exit "$result"
"""


@unittest.skipUnless(BASH and Path(BASH).is_file(), "Для проверки нужен Bash")
class LoadGitKeyTests(unittest.TestCase):
    """Запускаем копию скрипта с временным путём и подменёнными SSH-командами."""

    def _run(
        self, *, key: str = "valid", direct: bool = False, **overrides: str,
    ) -> tuple[subprocess.CompletedProcess[str], list[str]]:
        env = {
            **os.environ,
            "AGENT_STATUS": "2",
            "ADD_STATUS": "0",
            "START_STATUS": "0",
            "STOP_STATUS": "0",
            "AGENT_OUTPUT": (
                "SSH_AUTH_SOCK=/new-agent; export SSH_AUTH_SOCK;\n"
                "SSH_AGENT_PID=456; export SSH_AGENT_PID;"
            ),
            "SSH_AUTH_SOCK": "/old-agent",
            "SSH_AGENT_PID": "123",
            "MISSING_COMMAND": "",
            "DIRECT": str(int(direct)),
            **overrides,
        }
        env.pop("BASH_ENV", None)
        env.pop("ENV", None)
        script = (ROOT / "load_git_key.sh").read_text(encoding="utf-8")
        self.assertIn('local key_file="/root/id_ed25519_github"', script)
        script = script.replace(
            'local key_file="/root/id_ed25519_github"',
            'local key_file="$PWD/test key"',
        )
        with tempfile.TemporaryDirectory(prefix="kapum-ssh-") as temp:
            root = Path(temp)
            (root / "load_git_key.sh").write_text(script, encoding="utf-8", newline="\n")
            if key == "directory":
                (root / "test key").mkdir()
            elif key != "missing":
                (root / "test key").write_text(
                    "" if key == "empty" else "dummy key, never read by ssh-add", encoding="utf-8",
                )
            result = subprocess.run(
                [BASH, "--noprofile", "--norc", "-c", HARNESS],
                cwd=root, env=env, capture_output=True, encoding="utf-8", timeout=15, check=False,
            )
            log = root / "calls.log"
            calls = log.read_text(encoding="utf-8").splitlines() if log.exists() else []
        return result, calls

    def test_requires_source(self) -> None:
        result, calls = self._run(direct=True)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("source ./load_git_key.sh", result.stderr)
        self.assertEqual(calls, [])

    def test_missing_dependencies(self) -> None:
        for tool in ("ssh-agent", "ssh-add"):
            with self.subTest(tool=tool):
                result, calls = self._run(MISSING_COMMAND=tool)
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn(f"команда {tool} не найдена", result.stderr)
                self.assertEqual(calls, [])

    def test_missing_empty_or_directory_key_does_not_start_agent(self) -> None:
        for key, error in (("missing", "не найден"), ("empty", "пуст"), ("directory", "не найден")):
            with self.subTest(key=key):
                result, calls = self._run(key=key)
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn(error, result.stderr)
                self.assertIn("SOCK=/old-agent\nPID=123", result.stdout)
                self.assertEqual(calls, [])

    def test_reuses_existing_agent_even_when_empty(self) -> None:
        for status in ("0", "1"):
            with self.subTest(status=status):
                result, calls = self._run(AGENT_STATUS=status)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("SOCK=/old-agent\nPID=123", result.stdout)
                self.assertEqual(calls[0], "ssh-add -l")
                self.assertTrue(calls[1].endswith("/test key"))
                self.assertEqual(len(calls), 2)

    def test_starts_new_agent_and_exports_environment(self) -> None:
        result, calls = self._run()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("SOCK=/new-agent\nPID=456", result.stdout)
        self.assertIn("загружен", result.stdout)
        self.assertEqual(calls[:2], ["ssh-add -l", "ssh-agent -s"])
        self.assertTrue(calls[2].endswith("/test key"))
        self.assertEqual(len(calls), 3)

    def test_unexpected_probe_error_is_not_treated_as_missing_agent(self) -> None:
        result, calls = self._run(AGENT_STATUS="3")
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("кодом 3", result.stderr)
        self.assertEqual(calls, ["ssh-add -l"])

    def test_start_failure_does_not_evaluate_output_or_load_key(self) -> None:
        result, calls = self._run(START_STATUS="1")
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("не удалось запустить", result.stderr)
        self.assertIn("SOCK=/old-agent\nPID=123", result.stdout)
        self.assertEqual(calls, ["ssh-add -l", "ssh-agent -s"])

    def test_empty_agent_output_is_rejected(self) -> None:
        result, calls = self._run(AGENT_OUTPUT="")
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("не вернул окружение", result.stderr)
        self.assertEqual(calls, ["ssh-add -l", "ssh-agent -s"])

    def test_invalid_agent_output_is_rejected(self) -> None:
        result, calls = self._run(AGENT_OUTPUT="false")
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("не удалось применить окружение", result.stderr)
        self.assertIn("SOCK=/old-agent\nPID=123", result.stdout)
        self.assertEqual(calls, ["ssh-add -l", "ssh-agent -s"])

    def test_load_failure_stops_only_new_agent_and_preserves_environment(self) -> None:
        result, calls = self._run(ADD_STATUS="1")
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("не удалось загрузить ключ", result.stderr)
        self.assertNotIn("загружен", result.stdout)
        self.assertIn("SOCK=/old-agent\nPID=123", result.stdout)
        self.assertEqual(calls[-1], "ssh-agent -k")

    def test_load_failure_does_not_stop_existing_agent(self) -> None:
        result, calls = self._run(AGENT_STATUS="0", ADD_STATUS="1")
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("SOCK=/old-agent\nPID=123", result.stdout)
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(call.startswith("ssh-add ") for call in calls))

    def test_failed_cleanup_is_reported(self) -> None:
        result, calls = self._run(ADD_STATUS="1", STOP_STATUS="1")
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("не удалось остановить", result.stderr)
        self.assertEqual(calls[-1], "ssh-agent -k")
