from __future__ import annotations

import csv
import io
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from bot.config import config

logger = logging.getLogger(__name__)

# Только текстовые форматы, которые не исполняются при открытии. Скрипты
# (py/js/sh/bat/ps1/vbs...), html/svg (JS в браузере), архивы и бинарники
# намеренно отсутствуют: модель может создавать лишь эти расширения.
SAFE_EXTENSIONS: frozenset[str] = frozenset(
    {
        "txt", "md", "markdown", "rst", "log",
        "json", "jsonl", "ndjson", "xml", "yaml", "yml", "toml", "ini",
        "csv", "tsv",
        "sql", "tex", "srt", "vtt", "ics",
    }
)

# Не даём модели подсунуть «report.exe.txt»: точки внутри имени заменяются,
# остаётся ровно одно (последнее) расширение.
_BAD_CHARS = re.compile(r'[\\/:*?"<>|\x00-\x1f\x7f]')
_MAX_NAME_LENGTH = 80
_DEFAULT_STEM = "file"

# Тип контента -> подсказка модели; также используется в описании tool.
FILES_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "send_files",
        "description": (
            "Send one or more text files to the user as Telegram document "
            "attachments. Use this ONLY when the user explicitly asks for a "
            "file/document (for example 'give me this as a json file', "
            "'пришли файлом', 'сохрани в txt', 'сделай md-файл'). Allowed "
            "formats: " + ", ".join(sorted(SAFE_EXTENSIONS)) + ". Executable "
            "files, scripts, HTML and archives are NOT allowed. Put the full "
            "file text in `content`. At most {limit} files per reply."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "files": {
                    "type": "array",
                    "description": "Files to send.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "filename": {
                                "type": "string",
                                "description": (
                                    "File name with an allowed extension, "
                                    "e.g. 'notes.md' or 'data.json'."
                                ),
                            },
                            "content": {
                                "type": "string",
                                "description": "The complete text of the file.",
                            },
                        },
                        "required": ["filename", "content"],
                    },
                }
            },
            "required": ["files"],
        },
    },
}


def files_tool() -> dict[str, Any]:
    """Tool definition with the configured per-reply limit substituted."""
    tool = {**FILES_TOOL, "function": dict(FILES_TOOL["function"])}
    function = tool["function"]
    function["description"] = function["description"].replace(
        "{limit}", str(config.reply_files_max)
    )
    return tool


@dataclass(slots=True)
class PreparedFiles:
    """Validated outgoing files plus what was dropped and why."""

    files: list[tuple[str, bytes]] = field(default_factory=list)
    # (имя как запросила модель, причина) — для лога и сообщения пользователю.
    rejected: list[tuple[str, str]] = field(default_factory=list)
    # Сколько файлов сверх лимита было запрошено и не отправлено.
    over_limit: int = 0


def safe_filename(raw: str) -> str | None:
    """Return a sanitized name with an allowed extension, or None if unsafe.

    Strips directories, control and bidi-override characters (spoofed
    extensions like ``txt.exe`` rendered backwards), and collapses extra dots.
    """
    name = unicodedata.normalize("NFKC", str(raw or "")).strip()
    name = name.replace("\\", "/").rsplit("/", 1)[-1]
    # Символы категории Cf (в т.ч. U+202E RLO) позволяют подделать расширение.
    name = "".join(ch for ch in name if unicodedata.category(ch) != "Cf")
    name = _BAD_CHARS.sub("_", name).strip(" .")
    if "." not in name:
        return None
    stem, _, ext = name.rpartition(".")
    ext = ext.lower()
    if ext not in SAFE_EXTENSIONS:
        return None
    stem = stem.replace(".", "_").strip(" _") or _DEFAULT_STEM
    return f"{stem[:_MAX_NAME_LENGTH]}.{ext}"


# Ячейка, начинающаяся с этих символов, исполняется Excel/LibreOffice как
# формула (=HYPERLINK(...), DDE через =cmd|...). Это CSV injection.
_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")
_NUMBER_RE = re.compile(r"^[+-]?(\d+([.,]\d+)?|[.,]\d+)([eE][+-]?\d+)?%?$")


def _safe_cell(cell: str) -> str:
    """Prefix a formula-looking cell with an apostrophe; plain numbers stay."""
    if cell and cell[0] in _FORMULA_PREFIXES and not _NUMBER_RE.match(cell.strip()):
        return "'" + cell
    return cell


def _guess_delimiter(text: str, ext: str) -> str:
    if ext == "tsv":
        return "\t"
    head = "\n".join(text.splitlines()[:5])
    counts = {delimiter: head.count(delimiter) for delimiter in (",", ";", "\t", "|")}
    best = max(counts, key=lambda delimiter: counts[delimiter])
    return best if counts[best] else ","


def neutralize_spreadsheet_formulas(text: str, ext: str) -> str:
    """Make a CSV/TSV safe to open in a spreadsheet application.

    Cells that would be interpreted as formulas (``=``, ``+``, ``-``, ``@``,
    leading tab/CR) get a leading apostrophe; ordinary numbers like ``-5`` are
    left alone. Other formats are returned unchanged.
    """
    if ext not in ("csv", "tsv"):
        return text
    delimiter = _guess_delimiter(text, ext)
    try:
        rows = list(csv.reader(io.StringIO(text, newline=""), delimiter=delimiter))
    except csv.Error:
        rows = [line.split(delimiter) for line in text.splitlines()]
    out = io.StringIO()
    writer = csv.writer(out, delimiter=delimiter, lineterminator="\n")
    writer.writerows([[_safe_cell(cell) for cell in row] for row in rows])
    return out.getvalue()


def prepare_files(arguments: dict[str, Any]) -> PreparedFiles:
    """Validate the ``send_files`` tool arguments.

    Keeps at most ``config.reply_files_max`` valid files; the surplus is only
    counted (``over_limit``) so the caller can tell the user. Invalid entries
    (unsafe extension, empty or oversized content, wrong types) are skipped.
    """
    result = PreparedFiles()
    raw_files = arguments.get("files")
    if not isinstance(raw_files, list):
        return result

    limit = max(1, config.reply_files_max)
    max_bytes = config.reply_file_max_bytes
    used: set[str] = set()

    for item in raw_files:
        if not isinstance(item, dict):
            continue
        requested = str(item.get("filename") or "")
        content = item.get("content")
        name = safe_filename(requested)
        if name is None:
            result.rejected.append((requested, "unsupported or unsafe file type"))
            continue
        if not isinstance(content, str) or not content.strip():
            result.rejected.append((requested, "empty content"))
            continue
        if "\x00" in content:
            result.rejected.append((requested, "binary content"))
            continue
        content = neutralize_spreadsheet_formulas(content, name.rpartition(".")[2])
        data = content.encode("utf-8")
        if len(data) > max_bytes:
            result.rejected.append((requested, "file too large"))
            continue
        if len(result.files) >= limit:
            result.over_limit += 1
            continue

        base, _, ext = name.rpartition(".")
        candidate, counter = name, 1
        while candidate.lower() in used:
            counter += 1
            candidate = f"{base}_{counter}.{ext}"
        used.add(candidate.lower())
        result.files.append((candidate, data))

    if result.rejected:
        logger.info(
            "send_files: rejected %s",
            [(name, reason) for name, reason in result.rejected],
        )
    return result
