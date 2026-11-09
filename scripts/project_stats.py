"""Count tracked project lines, words and characters.

The output is written to ``PROJECT_STATS.md`` and refreshed by the tracked
pre-commit hook, so every commit carries current numbers. Only text files that
are tracked by Git are counted: ``.venv``, caches, databases and local secrets
are never measured.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "PROJECT_STATS.md"
SKIP_NAMES = {"PROJECT_STATS.md"}

_TEXT_SUFFIXES = (
    ".py", ".md", ".txt", ".toml", ".cfg", ".ini", ".sh", ".ps1",
    ".service", ".example", ".yml", ".yaml", ".json", ".gitignore",
    ".gitattributes",
)

# Display order; a path not matching an earlier prefix falls into "other".
_GROUPS = (
    ("bot/", "Код бота"),
    ("tests/", "Тесты"),
    ("scripts/", "Скрипты"),
)


@dataclass(frozen=True)
class Counts:
    files: int = 0
    lines: int = 0
    words: int = 0
    chars: int = 0

    def __add__(self, other: Counts) -> Counts:
        return Counts(
            self.files + other.files,
            self.lines + other.lines,
            self.words + other.words,
            self.chars + other.chars,
        )


def _tracked_files() -> list[Path]:
    """Paths of tracked files relative to the repository root."""
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    paths = []
    for raw in result.stdout.split(b"\0"):
        if not raw:
            continue
        path = raw.decode("utf-8", "surrogateescape")
        if Path(path).name in SKIP_NAMES:
            continue
        if Path(path).suffix.lower() in _TEXT_SUFFIXES or Path(path).name in {
            ".gitignore",
            ".gitattributes",
        }:
            paths.append(Path(path))
    return paths


def _read(path: Path) -> str | None:
    try:
        data = (ROOT / path).read_bytes()
    except OSError:
        return None
    if b"\0" in data:
        return None
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return None
    # Один и тот же коммит обязан давать одинаковую статистику на Linux (LF)
    # и Windows (CRLF): переводы строк не должны попадать в счётчик символов.
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _count(paths: list[Path]) -> Counts:
    total = Counts()
    for path in paths:
        text = _read(path)
        if text is None:
            continue
        total += Counts(
            files=1,
            lines=len(text.splitlines()),
            words=len(text.split()),
            chars=len(text),
        )
    return total


def collect() -> dict[str, Counts]:
    """Counts per display group plus the whole project."""
    files = _tracked_files()
    groups: dict[str, list[Path]] = {title: [] for _, title in _GROUPS}
    groups["Документация и конфиги"] = []
    for path in files:
        key = path.as_posix()
        for prefix, title in _GROUPS:
            if key.startswith(prefix):
                groups[title].append(path)
                break
        else:
            groups["Документация и конфиги"].append(path)
    result = {title: _count(paths) for title, paths in groups.items()}
    result["Весь проект"] = _count(files)
    return result


def render(counts: dict[str, Counts]) -> str:
    order = [title for _, title in _GROUPS] + [
        "Документация и конфиги",
        "Весь проект",
    ]
    lines = [
        "# Статистика проекта",
        "",
        (
            "Файл пересчитывается автоматически перед каждым коммитом "
            "(`.githooks/pre-commit`)."
        ),
        (
            "Считаются только текстовые файлы, отслеживаемые Git; `.venv`, кэши, "
            "базы данных и локальные секреты не входят."
        ),
        "",
        "| Область | Файлов | Строк | Слов | Символов |",
        "|---|---:|---:|---:|---:|",
    ]
    for title in order:
        value = counts[title]
        lines.append(
            f"| {title} | {value.files} | {value.lines} | "
            f"{value.words} | {value.chars} |"
        )
    total = counts["Весь проект"]
    lines.extend(
        [
            "",
            (
                f"Итого: **{total.files}** файлов, **{total.lines}** строк, "
                f"**{total.words}** слов, **{total.chars}** символов."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help="write PROJECT_STATS.md instead of printing it",
    )
    args = parser.parse_args()
    text = render(collect())
    if args.write:
        # Явно пишем LF: иначе на Windows write_text подставит CRLF и файл
        # будет отличаться от вывода в stdout (проверка свежести упадёт).
        with OUTPUT.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
    else:
        # Deterministic UTF-8 on every platform (Windows consoles default to
        # a legacy code page, which breaks piping into the freshness test).
        sys.stdout.buffer.write(text.encode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
