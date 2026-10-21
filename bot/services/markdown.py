from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Characters that must be escaped in Telegram MarkdownV2 outside entities.
# The backslash itself must be escaped too (it is the escape character).
_MDV2_SPECIAL = "_*[]()~`>#+-=|{}.!\\"
_MDV2_SPECIAL_RE = re.compile("([" + re.escape(_MDV2_SPECIAL) + "])")

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_HR_RE = re.compile(r"^\s{0,3}([-*_])(\s*\1){2,}\s*$")
_LIST_RE = re.compile(r"^(\s*)([-*+]|\d{1,9}[.)])\s+(.*)$")
_LINK_RE = re.compile(r"!?\[([^\]]*)\]\(\s*([^)\s]+)(?:\s+\"[^\"]*\")?\s*\)")


def _escape_html(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


# Only these link schemes are kept; anything else (javascript:, data:, file:)
# is rendered as plain link text.
_SAFE_URL_RE = re.compile(r"(?i)^(https?://|tg://|mailto:)")


def _escape_mdv2(text: str) -> str:
    return _MDV2_SPECIAL_RE.sub(r"\\\1", text)


# --- Inline parsing --------------------------------------------------------


def _parse_inline(text: str) -> list[Any]:
    """Parse inline Markdown into a list of nodes.

    Node shapes: ("text", s), ("code", s), ("b", children), ("i", children),
    ("u", children), ("s", children), ("spoiler", children),
    ("link", url, children).
    """
    nodes: list[Any] = []
    buffer: list[str] = []
    i = 0
    n = len(text)

    def flush() -> None:
        if buffer:
            nodes.append(("text", "".join(buffer)))
            buffer.clear()

    while i < n:
        ch = text[i]

        # Inline code: `...`, ``...``
        if ch == "`":
            j = i
            while j < n and text[j] == "`":
                j += 1
            fence = text[i:j]
            end = text.find(fence, j)
            if end != -1:
                flush()
                nodes.append(("code", text[j:end]))
                i = end + len(fence)
                continue
            buffer.append(ch)
            i += 1
            continue

        # Links / images: [text](url)
        if ch == "[" or (ch == "!" and text.startswith("![", i)):
            match = _LINK_RE.match(text, i)
            if match:
                flush()
                if _SAFE_URL_RE.match(match.group(2)):
                    nodes.append(
                        ("link", match.group(2), _parse_inline(match.group(1)))
                    )
                else:
                    nodes.extend(_parse_inline(match.group(1)))
                i = match.end()
                continue

        # Bold / italic / underline: **x**, __x__, *x*, _x_
        if ch in "*_":
            pair = text[i : i + 2]
            if pair in ("**", "__"):
                end = text.find(pair, i + 2)
                if end > i + 2:
                    inner = text[i + 2 : end]
                    if inner.strip():
                        flush()
                        nodes.append(("b", _parse_inline(inner)))
                        i = end + 2
                        continue
            # Single-character emphasis. Skip word-internal underscores
            # (snake_case) to avoid mangling identifiers.
            prev_char = text[i - 1] if i > 0 else ""
            next_char = text[i + 1] if i + 1 < n else ""
            if ch == "_" and (prev_char.isalnum() or next_char == " " or not next_char):
                buffer.append(ch)
                i += 1
                continue
            end = text.find(ch, i + 1)
            if end > i + 1:
                inner = text[i + 1 : end]
                after = text[end + 1] if end + 1 < n else ""
                if (
                    "\n" not in inner
                    and ch not in inner
                    and inner.strip()
                    and not after.isalnum()
                ):
                    flush()
                    nodes.append(("i", _parse_inline(inner)))
                    i = end + 1
                    continue

        # Strikethrough: ~~x~~
        if text.startswith("~~", i):
            end = text.find("~~", i + 2)
            if end != -1:
                flush()
                nodes.append(("s", _parse_inline(text[i + 2 : end])))
                i = end + 2
                continue

        # Spoiler: ||x||
        if text.startswith("||", i):
            end = text.find("||", i + 2)
            if end != -1:
                flush()
                nodes.append(("spoiler", _parse_inline(text[i + 2 : end])))
                i = end + 2
                continue

        buffer.append(ch)
        i += 1

    flush()
    return nodes


# --- Block parsing ---------------------------------------------------------


def _parse_blocks(text: str) -> list[Any]:
    """Parse Markdown into block-level nodes."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    blocks: list[Any] = []
    i = 0
    total = len(lines)

    while i < total:
        line = lines[i]
        stripped = line.strip()

        # Fenced code block
        if stripped.startswith("```"):
            language = stripped[3:].strip()
            i += 1
            code: list[str] = []
            while i < total and not lines[i].strip().startswith("```"):
                code.append(lines[i])
                i += 1
            i += 1  # skip closing fence
            blocks.append(("pre", language, "\n".join(code)))
            continue

        if not stripped:
            i += 1
            continue

        heading = _HEADING_RE.match(stripped)
        if heading:
            blocks.append(("heading", heading.group(2).strip()))
            i += 1
            continue

        if _HR_RE.match(line):
            blocks.append(("hr",))
            i += 1
            continue

        if stripped.startswith(">"):
            quote: list[str] = []
            while i < total and lines[i].strip().startswith(">"):
                quote.append(lines[i].strip().lstrip(">").strip())
                i += 1
            blocks.append(("quote", "\n".join(quote)))
            continue

        list_match = _LIST_RE.match(line)
        if list_match:
            ordered = bool(re.match(r"\d{1,9}[.)]", list_match.group(2)))
            items: list[str] = []
            while i < total:
                item = _LIST_RE.match(lines[i])
                if not item:
                    break
                items.append(item.group(3))
                i += 1
            blocks.append(("list", ordered, items))
            continue

        # Paragraph: consume until a blank line or another block start.
        paragraph: list[str] = []
        while i < total:
            current = lines[i]
            current_stripped = current.strip()
            if not current_stripped:
                break
            if paragraph and (
                current_stripped.startswith("```")
                or _HEADING_RE.match(current_stripped)
                or _HR_RE.match(current)
                or current_stripped.startswith(">")
                or _LIST_RE.match(current)
            ):
                break
            paragraph.append(current_stripped)
            i += 1
        blocks.append(("paragraph", " ".join(paragraph)))

    return blocks


# --- Renderers -------------------------------------------------------------


def _render_inline_html(nodes: list[Any]) -> str:
    out: list[str] = []
    for node in nodes:
        kind = node[0]
        if kind == "text":
            out.append(_escape_html(node[1]))
        elif kind == "code":
            out.append(f"<code>{_escape_html(node[1])}</code>")
        elif kind == "b":
            out.append(f"<b>{_render_inline_html(node[1])}</b>")
        elif kind == "i":
            out.append(f"<i>{_render_inline_html(node[1])}</i>")
        elif kind == "s":
            out.append(f"<s>{_render_inline_html(node[1])}</s>")
        elif kind == "spoiler":
            out.append(f"<tg-spoiler>{_render_inline_html(node[1])}</tg-spoiler>")
        elif kind == "link":
            url = _escape_html(node[1])
            out.append(f'<a href="{url}">{_render_inline_html(node[2])}</a>')
    return "".join(out)


def render_html(text: str) -> str:
    """Render Markdown as Telegram-safe HTML."""
    parts: list[str] = []
    for block in _parse_blocks(text):
        kind = block[0]
        if kind == "paragraph":
            parts.append(_render_inline_html(_parse_inline(block[1])))
        elif kind == "heading":
            parts.append(f"<b>{_render_inline_html(_parse_inline(block[1]))}</b>")
        elif kind == "pre":
            language, code = block[1], block[2]
            if language:
                parts.append(
                    f'<pre><code class="language-{_escape_html(language)}">'
                    f"{_escape_html(code)}</code></pre>"
                )
            else:
                parts.append(f"<pre>{_escape_html(code)}</pre>")
        elif kind == "quote":
            body = _render_inline_html(_parse_inline(block[1]))
            parts.append(f"<blockquote>{body}</blockquote>")
        elif kind == "list":
            ordered, items = block[1], block[2]
            tag = "ol" if ordered else "ul"
            rendered = "".join(
                f"<li>{_render_inline_html(_parse_inline(item))}</li>"
                for item in items
            )
            parts.append(f"<{tag}>{rendered}</{tag}>")
        elif kind == "hr":
            parts.append("──────────")
    return "\n\n".join(p for p in parts if p)


def _render_inline_mdv2(nodes: list[Any]) -> str:
    out: list[str] = []
    for node in nodes:
        kind = node[0]
        if kind == "text":
            out.append(_escape_mdv2(node[1]))
        elif kind == "code":
            body = node[1].replace("\\", "\\\\").replace("`", "\\`")
            out.append(f"`{body}`")
        elif kind == "b":
            out.append(f"*{_render_inline_mdv2(node[1])}*")
        elif kind == "i":
            out.append(f"_{_render_inline_mdv2(node[1])}_")
        elif kind == "s":
            out.append(f"~{_render_inline_mdv2(node[1])}~")
        elif kind == "spoiler":
            out.append(f"||{_render_inline_mdv2(node[1])}||")
        elif kind == "link":
            url = node[1].replace("\\", "\\\\").replace(")", "\\)")
            out.append(f"[{_render_inline_mdv2(node[2])}]({url})")
    return "".join(out)


def render_markdown_v2(text: str) -> str:
    """Render Markdown as escaped Telegram MarkdownV2."""
    parts: list[str] = []
    for block in _parse_blocks(text):
        kind = block[0]
        if kind == "paragraph":
            parts.append(_render_inline_mdv2(_parse_inline(block[1])))
        elif kind == "heading":
            parts.append(f"*{_render_inline_mdv2(_parse_inline(block[1]))}*")
        elif kind == "pre":
            language, code = block[1], block[2]
            body = code.replace("\\", "\\\\").replace("`", "\\`")
            parts.append(f"```{language}\n{body}\n```")
        elif kind == "quote":
            body = _render_inline_mdv2(_parse_inline(block[1]))
            parts.append("\n".join(f">{line}" for line in body.split("\n")))
        elif kind == "list":
            ordered, items = block[1], block[2]
            lines = []
            for index, item in enumerate(items, start=1):
                marker = f"{index}\\." if ordered else "•"
                lines.append(f"{marker} {_render_inline_mdv2(_parse_inline(item))}")
            parts.append("\n".join(lines))
        elif kind == "hr":
            parts.append("──────────")
    return "\n\n".join(p for p in parts if p)


def render_plain(text: str) -> str:
    """Return the text without any Telegram markup (safe fallback)."""
    return text


def render(text: str, mode: str) -> str:
    """Render ``text`` for the requested parse mode, never raising."""
    try:
        if mode == "markdown":
            return render_markdown_v2(text)
        if mode == "html":
            return render_html(text)
    except Exception as exc:  # noqa: BLE001 - formatting must never break a reply
        logger.warning("Markdown rendering failed (%s); using plain text", exc)
    return render_plain(text)


# --- Chunking --------------------------------------------------------------


def _raw_paragraphs(text: str) -> list[str]:
    """Split raw Markdown into chunks at blank lines, keeping code fences."""
    chunks: list[str] = []
    current: list[str] = []
    in_fence = False
    for line in text.replace("\r\n", "\n").split("\n"):
        if line.strip().startswith("```"):
            in_fence = not in_fence
        if not line.strip() and not in_fence:
            if current:
                chunks.append("\n".join(current))
                current = []
            continue
        current.append(line)
    if current:
        chunks.append("\n".join(current))
    return chunks


def split_chunks(text: str, mode: str, limit: int = 4000) -> list[str]:
    """Split raw Markdown into parts whose *rendered* form fits ``limit``.

    Returns raw Markdown chunks; render each one with :func:`render` when
    sending so a failing parse mode can be retried with another one.
    """
    if not text or not text.strip():
        return []

    target = mode if mode in {"markdown", "html"} else "plain"
    if target == "plain":
        return [text[i : i + limit] for i in range(0, len(text), limit)]

    def fits(candidate: str) -> bool:
        return len(render(candidate, target)) <= limit

    chunks: list[str] = []
    current = ""
    for piece in _raw_paragraphs(text):
        candidate = f"{current}\n\n{piece}" if current else piece
        if fits(candidate):
            current = candidate
            continue
        if current:
            chunks.append(current)
            current = ""
        if fits(piece):
            current = piece
        else:
            chunks.extend(_hard_split(piece, target, limit))

    if current:
        chunks.append(current)
    return chunks or [text[:limit]]


def _hard_split(text: str, target: str, limit: int) -> list[str]:
    """Split a single oversized block by lines, then by characters."""
    parts: list[str] = []
    current = ""
    for line in text.split("\n"):
        candidate = f"{current}\n{line}" if current else line
        if len(render(candidate, target)) <= limit:
            current = candidate
            continue
        if current:
            parts.append(current)
        while len(render(line, target)) > limit:
            size = max(1, limit // 2)
            parts.append(line[:size])
            line = line[size:]
        current = line
    if current:
        parts.append(current)
    return parts