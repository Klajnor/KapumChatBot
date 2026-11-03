"""Rotating file log for post-mortem diagnostics.

By default (``LOG_FILE=auto``) warnings and errors go to ``errors.log`` next to
the SQLite database — a directory the service already has write access to
(``StateDirectory`` under systemd). ``LOG_FILE=off`` keeps logs only in
stdout/stderr (journald); an absolute path overrides the location.

Only records at ``LOG_FILE_LEVEL`` and above are written, so the file stays a
compact error journal rather than a second copy of the INFO log. Prompts and
tool results are never added here explicitly; keep ``LOG_PROMPTS=false`` when
sharing the file.
"""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

from sqlalchemy.engine import make_url

from bot.config import config

logger = logging.getLogger(__name__)

# A compact journal: enough history for a few days of failures.
_MAX_BYTES = 5 * 1024 * 1024
_BACKUPS = 3

# Values that mean "write no file" besides the documented ``off``.
_DISABLED = {"", "off", "none", "0", "false"}


def _database_log_path() -> Path | None:
    """``errors.log`` next to an absolute SQLite database, if there is one."""
    if not config.database_url.startswith("sqlite"):
        return None
    try:
        database = make_url(config.database_url).database
    except Exception:  # noqa: BLE001 - diagnostics must not break startup
        return None
    if not database or database == ":memory:":
        return None
    path = Path(os.path.expandvars(os.path.expanduser(database)))
    if not path.is_absolute():
        # A relative DB path is rejected by config.validate(); writing next to
        # the read-only working directory would fail anyway.
        return None
    return path.parent / "errors.log"


def resolve_log_file() -> Path | None:
    """Return the error-log path, or None when file logging is disabled."""
    raw = (config.log_file or "").strip()
    if raw.lower() in _DISABLED:
        return None
    if raw.lower() != "auto":
        return Path(os.path.expandvars(os.path.expanduser(raw)))
    return _database_log_path()


def _log_level() -> int:
    """Parse ``LOG_FILE_LEVEL``; fall back to WARNING with a warning."""
    name = (config.log_file_level or "").strip().upper()
    value = logging.getLevelNamesMapping().get(name)
    if isinstance(value, int):
        return value
    logger.warning(
        "Unknown LOG_FILE_LEVEL %r; file log uses WARNING", name
    )
    return logging.WARNING


def setup_file_logging() -> Path | None:
    """Attach the rotating file handler to the root logger; idempotent."""
    path = resolve_log_file()
    if path is None:
        return None

    root = logging.getLogger()
    for handler in root.handlers:
        if getattr(handler, "_kapum_log_file", None) == path:
            return path

    level = _log_level()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            path,
            maxBytes=_MAX_BYTES,
            backupCount=_BACKUPS,
            encoding="utf-8",
            delay=True,
        )
    except (OSError, ValueError) as exc:
        logger.warning("Could not open LOG_FILE %s: %s", path, exc)
        return None

    handler.setLevel(level)
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    handler._kapum_log_file = path  # type: ignore[attr-defined]
    root.addHandler(handler)
    logger.info(
        "File log: %s (level %s, rotation %s MiB x %s)",
        path,
        logging.getLevelName(level),
        _MAX_BYTES // (1024 * 1024),
        _BACKUPS,
    )
    return path
