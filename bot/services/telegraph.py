"""Publish answers Telegram cannot render as Telegraph pages.

Telegram messages have no Markdown tables and no LaTeX renderer. When an AI
answer still contains such constructs after all built-in normalization
(Unicode rewriting, PNG formula rendering), the whole answer is published as
a Telegraph page and the user receives a link instead of broken text.

Everything is best effort: if Telegraph is unreachable (regional blocks are
common), rejected the page format or the network fails, ``publish_answer``
returns ``None`` and the answer is delivered to Telegram exactly as before.
Node vocabulary differs between Telegraph API revisions, so publishing walks a
degrade chain: rich nodes → plain nodes → raw Markdown in the ``text`` field.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import aiohttp

from bot.config import config
from bot.services import markdown as _md
from bot.services.science import normalize_formulas

logger = logging.getLogger(__name__)

# Fenced and inline code never counts as a table or a formula. Double-backtick
# spans (``x``) must be matched before single-backtick ones.
_FENCE_RE = re.compile(r"```[\s\S]*?(?:```|\Z)")
_INLINE_CODE_RE = re.compile(r"``[^`\n]+``|`[^`\n]+`")

# ``$$...$$`` / ``\[...\]`` with at least one character between the delimiters.
_DISPLAY_MATH_RE = re.compile(r"\$\$([^$]*?)\$\$|\\\[([\s\S]*?)\\\]")
# Inline math (``$E=mc^2$``) — requires a mathy character inside, so prices
# like "$5 and $10" never match.
_INLINE_MATH_RE = re.compile(
    r"\$(?![\s$])([^\n$]*?(?:[\\^_{}|]|\s=\s)[^\n$]*?)\$"
)
# LaTeX commands Unicode normalization could not rewrite. Deliberately a
# curated list: the generic ``\\[a-zA-Z]+`` would also match Windows paths.
_RESIDUAL_LATEX_RE = re.compile(
    r"\\(?:frac\d?|dfrac|tfrac|sqrt|sum|prod|int|iint|oint|lim|limsup|liminf|"
    r"begin|end|matrix|pmatrix|bmatrix|vmatrix|cases|substack|binom|dbinom|"
    r"cdot|times|div|ast|star|circ|bullet|oplus|otimes|odg|pdv|odv|"
    r"leq|geq|neq|approx|equiv|sim|propto|ll|gg|infty|partial|nabla|"
    r"forall|exists|in|notin|subset|subseteq|supset|cup|cap|emptyset|"
    r"alpha|beta|gamma|delta|epsilon|varepsilon|zeta|eta|theta|iota|kappa|"
    r"lambda|mu|nu|xi|pi|rho|sigma|tau|upsilon|phi|varphi|chi|psi|omega|"
    r"Gamma|Delta|Theta|Lambda|Xi|Pi|Sigma|Phi|Psi|Omega|"
    r"hat|widehat|bar|overline|tilde|vec|dot|ddot|"
    r"mathbb|mathbf|mathit|mathrm|mathcal|operatorname|text|textrm|ce)\b"
)
_TABLE_ROW_RE = re.compile(r"^\s{0,3}\|.*\|\s*$")

MAX_TITLE_LENGTH = 70

# --- Detection --------------------------------------------------------------


def _cells(line: str) -> list[str]:
    """Split a Markdown table row into stripped cells."""
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _is_separator_row(line: str) -> bool:
    """Match the ``|---|---|`` line under a table header."""
    stripped = line.strip()
    if not stripped or "|" not in stripped or "-" not in stripped:
        return False
    if any(ch not in "|-: " for ch in stripped):
        return False
    cells = [cell for cell in _cells(stripped) if set(cell) and set(cell) - set("-: ")]
    return len(_cells(stripped)) >= 2 and not cells


def has_table(text: str) -> bool:
    """True when text contains a Markdown table outside code fences."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    in_fence = False
    for index, line in enumerate(lines):
        if line.strip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if (
            _TABLE_ROW_RE.match(line)
            and index + 1 < len(lines)
            and _is_separator_row(lines[index + 1])
        ):
            return True
    return False


def has_math(text: str) -> bool:
    """True when text contains LaTeX Telegram cannot display.

    With ``RENDER_FORMULA_IMAGES`` display math is rendered as PNG, so only
    inline math and un-rewritable commands count; otherwise ``$$$$`` and
    ``\\[..\\]`` reach the user raw and must go to Telegraph too.
    """
    stripped = _INLINE_CODE_RE.sub("", _FENCE_RE.sub(" ", text))
    if config.render_formula_images:
        stripped = _DISPLAY_MATH_RE.sub(" ", stripped)
    elif "$$" in stripped or "\\[" in stripped:
        return True
    return bool(_INLINE_MATH_RE.search(stripped) or _RESIDUAL_LATEX_RE.search(stripped))


def needs_telegraph(text: str | None, *, tables_as_images: bool = False) -> bool:
    """True when the answer contains constructs Telegram cannot display.

    With ``tables_as_images`` tables are not a reason: the sender renders them
    as pictures (and falls back to a list when that fails).
    """
    if not text or not text.strip():
        return False
    if tables_as_images:
        return False  # the sender handles tables, formulas and Unicode itself
    try:
        return has_table(text) or has_math(text)
    except Exception as exc:  # noqa: BLE001 - detection must never break a reply
        logger.warning("Telegraph content detection failed: %s", exc)
        return False


# --- Node conversion --------------------------------------------------------


def make_title(question: str | None) -> str:
    """Derive a Telegraph page title from the user's question."""
    first = ""
    for line in (question or "").splitlines():
        if line.strip():
            first = line.strip()
            break
    first = re.sub(r"[#*_`>|]", "", first).strip()
    return first[:MAX_TITLE_LENGTH] or "AI answer"


def _blocks_with_tables(text: str) -> list[Any]:
    """Markdown blocks, with table blocks recognized before paragraphs."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    blocks: list[Any] = []
    buffer: list[str] = []
    in_fence = False
    index = 0

    def flush() -> None:
        if buffer:
            blocks.extend(_md._parse_blocks("\n".join(buffer)))
            buffer.clear()

    while index < len(lines):
        line = lines[index]
        if line.strip().startswith("```"):
            in_fence = not in_fence
            buffer.append(line)
            index += 1
            continue
        if (
            not in_fence
            and _TABLE_ROW_RE.match(line)
            and index + 1 < len(lines)
            and _is_separator_row(lines[index + 1])
        ):
            flush()
            header = _cells(line)
            raw = [line, lines[index + 1]]
            index += 2
            rows: list[list[str]] = []
            while index < len(lines) and _TABLE_ROW_RE.match(lines[index]):
                rows.append(_cells(lines[index]))
                raw.append(lines[index])
                index += 1
            blocks.append(("table", header, rows, "\n".join(raw)))
            continue
        buffer.append(line)
        index += 1

    flush()
    return blocks


def _convert_inline(nodes: list[Any], *, rich: bool) -> list[Any]:
    """Convert parsed inline Markdown nodes to Telegraph children."""
    out: list[Any] = []
    for node in nodes:
        kind = node[0]
        if kind == "text":
            if node[1]:
                out.append(node[1])
        elif kind == "code":
            out.append(
                {"tag": "code", "children": [node[1]]} if rich else f"`{node[1]}`"
            )
        elif kind in ("b", "i", "s", "u"):
            children = _convert_inline(node[1], rich=rich)
            if not rich and kind in ("s", "u"):
                out.extend(children)
                continue
            tag = {"i": "em"}.get(kind, kind)
            out.append({"tag": tag, "children": children})
        elif kind == "spoiler":
            out.extend(_convert_inline(node[1], rich=rich))
        elif kind == "link":
            out.append(
                {
                    "tag": "a",
                    "attrs": {"href": node[1]},
                    "children": _convert_inline(node[2], rich=rich),
                }
            )
    return out


def _paragraph_children(raw: str, *, rich: bool) -> list[Any]:
    """Inline children with display math split out as formula elements."""
    if not rich:
        return _convert_inline(_md._parse_inline(raw), rich=False)
    children: list[Any] = []
    position = 0
    for match in _DISPLAY_MATH_RE.finditer(raw):
        before = raw[position : match.start()]
        if before.strip():
            children.extend(_convert_inline(_md._parse_inline(before), rich=True))
        tex = (match.group(1) or match.group(2) or "").strip()
        if tex:
            children.append({"tag": "code", "children": [tex]})  # no formula tag in Telegraph
        position = match.end()
    tail = raw[position:]
    if tail.strip():
        children.extend(_convert_inline(_md._parse_inline(tail), rich=True))
    return children


def _table_items(header: list[str], rows: list[list[str]], *, rich: bool) -> list[Any]:
    """Telegraph has no table tags: each row becomes a list item.

    The first cell is bold, the remaining ones follow as ``header: value``
    pairs (just the value for a two-column table), so nothing depends on
    monospace alignment or on node types the API may drop.
    """
    items: list[list[Any]] = []
    for row in rows:
        cells = [*row, *[""] * (len(header) - len(row))]
        children: list[Any] = []
        lead = _convert_inline(_md._parse_inline(cells[0]), rich=True)
        if lead:
            children.append({"tag": "b", "children": lead})
        parts: list[list[Any]] = []
        for title, value in zip(header[1:], cells[1:]):
            value_nodes = _convert_inline(_md._parse_inline(value), rich=True)
            if not value_nodes:
                continue
            if len(header) > 2 and title.strip():
                value_nodes = [f"{title.strip()}: ", *value_nodes]
            parts.append(value_nodes)
        for index, part in enumerate(parts):
            children.append(" — " if index == 0 else "; ")
            children.extend(part)
        items.append(children or [""])
    if rich:
        return [{"tag": "ul", "children": [{"tag": "li", "children": c} for c in items]}]
    return [{"tag": "p", "children": ["• ", *c]} for c in items]


def to_nodes(text: str, *, rich: bool = True) -> list[Any]:
    """Convert Markdown to Telegraph nodes.

    ``rich`` uses the full Telegraph vocabulary (formula tags, real lists and
    tables); the plain variant degrades them to text paragraphs so a page can
    still be published when the API rejects a node type.
    """
    nodes: list[Any] = []
    for block in _blocks_with_tables(text):
        kind = block[0]
        if kind == "paragraph":
            children = _paragraph_children(block[1], rich=rich)
            if children:
                nodes.append({"tag": "p", "children": children})
        elif kind == "heading":
            children = _convert_inline(_md._parse_inline(block[1]), rich=rich)
            if children:
                nodes.append(
                    {"tag": "h3", "children": children}
                    if rich
                    else {"tag": "p", "children": [{"tag": "b", "children": children}]}
                )
        elif kind == "pre":
            nodes.append({"tag": "pre", "children": [block[2]]})
        elif kind == "quote":
            inner = _convert_inline(_md._parse_inline(block[1]), rich=rich)
            if inner:
                nodes.append(
                    {"tag": "blockquote", "children": [{"tag": "p", "children": inner}]}
                    if rich
                    else {"tag": "blockquote", "children": inner}
                )
        elif kind == "list":
            ordered, items = block[1], block[2]
            if rich:
                nodes.append(
                    {
                        "tag": "ol" if ordered else "ul",
                        "children": [
                            {
                                "tag": "li",
                                "children": _convert_inline(
                                    _md._parse_inline(item), rich=True
                                )
                                or [""],
                            }
                            for item in items
                        ],
                    }
                )
            else:
                for number, item in enumerate(items, start=1):
                    marker = f"{number}. " if ordered else "• "
                    children = _convert_inline(_md._parse_inline(item), rich=False)
                    nodes.append({"tag": "p", "children": [marker, *children]})
        elif kind == "hr":
            nodes.append({"tag": "hr"})
        elif kind == "table":
            _, header, rows, _raw = block
            nodes.extend(_table_items(header, rows, rich=rich))
    return nodes


# --- Publishing -------------------------------------------------------------

_session: aiohttp.ClientSession | None = None


async def _get_session() -> aiohttp.ClientSession:
    global _session
    if _session is None or _session.closed:
        _session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=config.telegraph_timeout),
            # telegra.ph is regionally blocked on some networks: honour
            # HTTPS_PROXY from the environment when one is configured.
            trust_env=False,
        )
    return _session


async def close() -> None:
    """Close the shared HTTP session (called on shutdown)."""
    global _session
    if _session is not None and not _session.closed:
        await _session.close()
    _session = None


async def _post(endpoint: str, data: dict[str, str]) -> dict[str, Any]:
    """POST to the Telegraph API; returns the decoded JSON or an error dict.

    The live API wraps successful payloads in a ``result`` object
    (``{"ok": true, "result": {"access_token": …}}``), while the docs show flat
    fields — unwrap it so every consumer sees one flat dictionary.
    """
    url = f"{config.telegraph_api_url}/{endpoint}"
    try:
        session = await _get_session()
        async with session.post(url, data=data) as resp:
            body = await resp.text()
            if resp.status != 200:
                return {"ok": False, "error": f"HTTP {resp.status}: {body[:200]}"}
            parsed = json.loads(body)
    except (TimeoutError, aiohttp.ClientError) as exc:
        return {"ok": False, "error": f"network: {exc}"}
    except json.JSONDecodeError as exc:
        return {"ok": False, "error": f"bad JSON: {exc}"}
    if not isinstance(parsed, dict):
        return {"ok": False, "error": "bad JSON"}
    if parsed.get("ok") and isinstance(parsed.get("result"), dict):
        flat = {key: value for key, value in parsed.items() if key != "result"}
        flat.update(parsed["result"])
        return flat
    return parsed


# --- Account ----------------------------------------------------------------
# createPage now requires an account token (anonymous calls answer
# ACCESS_TOKEN_INVALID). The pair (short_name, access_token) comes from .env,
# else from a JSON file the bot wrote after its own createAccount call.

_account: tuple[str, str] | None = None
_account_lock = asyncio.Lock()
# After a createAccount failure, do not hammer the API again for a while:
# publishing must fall back to Telegram quickly.
_account_fail_until = 0.0
_ACCOUNT_RETRY_DELAY = 300.0

_ACCOUNT_ERROR_RE = re.compile(r"ACCESS_TOKEN|AUTHORIZ", re.IGNORECASE)


def _describe(result: dict[str, Any]) -> str:
    """Compact, secret-safe rendering of an API response for the logs.

    Tokens are masked; a response that the API should never send (no ``ok``
    and no ``error`` — the sign of a middlebox or a proxy answer) is labelled,
    so the log shows what actually came back instead of ``None``.
    """
    safe: dict[str, Any] = {}
    for key, value in result.items():
        if "token" in str(key).lower():
            value = "***"
        elif isinstance(value, dict):
            value = {
                k: ("***" if "token" in str(k).lower() else v)
                for k, v in value.items()
            }
        safe[key] = value
    if not safe.get("ok") and not safe.get("error"):
        safe["note"] = "response without ok/error flags (proxy or block page?)"
    elif safe.get("ok") and not (result.get("access_token") or result.get("auth_token")):
        safe["note"] = "ok=true but no access_token in the response"
    try:
        return json.dumps(safe, ensure_ascii=False, default=str)[:300]
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return str(safe)[:300]


def _reset_account_cache() -> None:
    global _account
    _account = None


def _account_file() -> Path | None:
    name = (config.telegraph_account_file or "").strip()
    return Path(name) if name else None


def _write_account_file(path: Path, short_name: str, token: str) -> None:
    """Atomically persist the Telegraph account with 0600 permissions.

    A plain ``write_text`` exposes a partially written file and only chmods
    it afterwards; the token must never be world-readable, even briefly.
    """
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump({"short_name": short_name, "access_token": token}, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


async def ensure_account(*, force_new: bool = False) -> tuple[str, str] | None:
    """Return ``(short_name, access_token)`` for createPage, or ``None``.

    Priority: explicit .env pair → in-memory cache → the account file → a
    freshly created anonymous account (``createAccount``), which is then
    persisted so restarts reuse it instead of spawning new accounts.
    """
    global _account, _account_fail_until
    if not force_new:
        if config.telegraph_short_name and config.telegraph_access_token:
            return (config.telegraph_short_name, config.telegraph_access_token)
        if _account is not None:
            return _account

    async with _account_lock:
        if not force_new and _account is not None:
            return _account

        path = _account_file()
        if path is not None and not force_new and path.is_file():
            try:
                stored = json.loads(path.read_text(encoding="utf-8"))
                short = str(stored.get("short_name") or "")
                token = str(stored.get("access_token") or "")
                if short and token:
                    _account = (short, token)
                    return _account
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("Could not read the Telegraph account file: %s", exc)

        if time.monotonic() < _account_fail_until and not force_new:
            return None
        short = f"chatbot{uuid4().hex}"[:32]
        result = await _post(
            "createAccount",
            {"short_name": short, "author_name": config.telegraph_author},
        )
        token = ""
        if result.get("ok"):
            # access_token per the API docs; auth_token accepted as an alias.
            token = str(result.get("access_token") or result.get("auth_token") or "")
        if not token:
            _account_fail_until = time.monotonic() + _ACCOUNT_RETRY_DELAY
            logger.warning(
                "Telegraph createAccount failed: %s", _describe(result)
            )
            if (
                config.telegraph_short_name
                and config.telegraph_access_token
                and force_new
            ):
                return (config.telegraph_short_name, config.telegraph_access_token)
            return None
        short = str(result.get("short_name") or short)
        _account = (short, token)
        _account_fail_until = 0.0
        if path is not None:
            try:
                _write_account_file(path, short, token)
            except OSError as exc:
                logger.warning("Could not persist the Telegraph account: %s", exc)
        logger.info("Telegraph account ready (short_name=%s)", short)
        return _account


def _page_url(result: dict[str, Any]) -> str:
    url = str(result.get("url") or "")
    if not url and result.get("path"):
        url = f"https://telegra.ph{result['path']}"
    return url


async def create_page(title: str, markdown_text: str) -> str | None:
    """Publish an article, walking the degrade chain on rejected payloads.

    The account is ensured once per pass: if the API rejects the stored token
    (revoked/expired), the whole chain is retried once with a freshly created
    account. Returns the page URL, or ``None`` when every attempt failed.
    """
    if not markdown_text.strip():
        return None
    # Telegraph cannot show LaTeX either: rewrite formulas as Unicode first
    # (code blocks are left alone).
    markdown_text = normalize_formulas(markdown_text) or markdown_text

    attempts: list[tuple[str, dict[str, str]]] = []
    try:
        attempts.append(
            (
                "rich nodes",
                {"content": json.dumps(to_nodes(markdown_text, rich=True), ensure_ascii=False)},
            )
        )
        attempts.append(
            (
                "plain nodes",
                {"content": json.dumps(to_nodes(markdown_text, rich=False), ensure_ascii=False)},
            )
        )
    except Exception as exc:  # noqa: BLE001 - conversion may hit an odd answer
        logger.warning("Telegraph node conversion failed: %s", exc)
    # Last resort: raw Markdown as plain paragraphs (createPage needs ``content``).
    raw_nodes = [
        {"tag": "p", "children": [part.strip()]}
        for part in re.split(r"\n\s*\n", markdown_text)
        if part.strip()
    ]
    attempts.append(
        ("markdown text", {"content": json.dumps(raw_nodes, ensure_ascii=False)})
    )

    for fresh_account in (False, True):
        account = await ensure_account(force_new=fresh_account)
        saw_account_error = False
        for name, extra in attempts:
            data = {
                "author_name": config.telegraph_author,
                "title": title,
                **extra,
            }
            if account:
                data["short_name"], data["access_token"] = account
            result = await _post("createPage", data)
            if result.get("ok"):
                page = _page_url(result)
                if page:
                    logger.info(
                        "Telegraph page published via %s: %s", name, page
                    )
                    return page
                logger.warning("Telegraph %s accepted but returned no URL", name)
                continue
            error = str(result.get("error") or "")
            if account and _ACCOUNT_ERROR_RE.search(error):
                # A stale token poisons every attempt: mark it and rebuild
                # the account for the second pass.
                saw_account_error = True
                _reset_account_cache()
                logger.warning(
                    "Telegraph rejected the access token (%s): %s",
                    name,
                    _describe(result),
                )
                continue
            logger.warning("Telegraph %s attempt rejected: %s", name, _describe(result))
        if not saw_account_error:
            break  # formatting/network issues: a new account cannot help
    return None


async def publish_answer(question: str | None, answer: str) -> str | None:
    """Publish ``answer`` as a Telegraph page; ``None`` on any failure."""
    try:
        # Pages are public: the user's question stays out of the title unless
        # TELEGRAPH_TITLE_FROM_QUESTION is enabled.
        title = make_title(question) if config.telegraph_title_from_question else "AI answer"
        return await create_page(title, answer)
    except Exception as exc:  # noqa: BLE001 - publishing must never break a reply
        logger.warning("Telegraph publishing failed: %s", exc)
        return None


def notice_text(url: str) -> str:
    """Render the TELEGRAPH_NOTICE template with the page URL."""
    # ``str.replace`` (not .format) so user templates with stray braces are safe.
    return config.telegraph_notice.replace("{url}", url)
