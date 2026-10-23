"""Conversation export: Markdown documents and PDF files.

Two scopes share the same renderer:

* a single answer (the 📄/📝 buttons under each assistant message) — the
  question plus the answer;
* the whole conversation (``/export``) — every message of the user's thread.

Markdown is produced directly (the stored content is already Markdown, so it
nests cleanly). PDF is rendered with ``fpdf2`` using the DejaVu Unicode fonts
that ship with matplotlib (already a dependency) — so Cyrillic and most
symbols work without bundling font files. ``fpdf2`` and the fonts are optional
at runtime: when either is missing, :func:`render_pdf` returns ``None`` and the
caller falls back to Markdown.

Emoji are stripped from PDF text: DejaVu has no colour-emoji glyphs and a
missing glyph can abort the render. Markdown keeps everything.
"""

from __future__ import annotations

import datetime
import logging
import os
import re
from dataclasses import dataclass
from typing import Any

from bot.bot import get_bot_username
from bot.config import config

try:  # optional dependency: without fpdf2 only Markdown export is available.
    from fpdf.enums import XPos, YPos
except ImportError:  # pragma: no cover - exercised on servers without fpdf2
    XPos = YPos = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# --- Entry model --------------------------------------------------------------


@dataclass(slots=True)
class ExportEntry:
    """One exported message, decoupled from the ORM row."""

    role: str                              # "user" | "assistant"
    content: str
    created_at: datetime.datetime | None = None
    model: str | None = None


def entries_from_messages(messages: list[Any]) -> list[ExportEntry]:
    """Build export entries from ORM ``Message`` rows (oldest first)."""
    entries: list[ExportEntry] = []
    for row in messages:
        entries.append(
            ExportEntry(
                role=getattr(row, "role", "user"),
                content=getattr(row, "content", "") or "",
                created_at=getattr(row, "created_at", None),
                model=getattr(row, "model", None),
            )
        )
    return entries


def _format_ts(created_at: datetime.datetime | None) -> str:
    if created_at is None:
        return ""
    if created_at.tzinfo is None:
        # SQLite hands back naive datetimes; stored times are UTC.
        created_at = created_at.replace(tzinfo=datetime.UTC)
    return created_at.strftime("%Y-%m-%d %H:%M:%S UTC")


def _role_label(role: str) -> str:
    return "🤖 Assistant" if role == "assistant" else "👤 User"


# --- Markdown -----------------------------------------------------------------


def render_markdown(
    entries: list[ExportEntry], *, title: str, meta: list[str]
) -> str:
    """Render the export as a Markdown document."""
    lines = [f"# {title}", ""]
    lines.extend(f"- {item}" for item in meta)
    lines.append("")
    lines.append("---")
    lines.append("")
    for entry in entries:
        header = f"## {_role_label(entry.role)}"
        stamp = _format_ts(entry.created_at)
        if stamp:
            header += f" — {stamp}"
        if entry.role == "assistant" and entry.model:
            header += f" · `{entry.model}`"
        lines.append(header)
        lines.append("")
        lines.append(entry.content.strip())
        lines.append("")
        lines.append("---")
        lines.append("")
    footer = "Exported from Kapum ChatBot"
    username = get_bot_username()
    if username:
        footer += f" (@{username})"
    lines.append(f"_{footer}_")
    return "\n".join(lines).rstrip() + "\n"


# --- PDF -----------------------------------------------------------------------

# Emoji and pictographic ranges DejaVu lacks; stripped before PDF rendering.
# Arrows (2190–21FF), math (2200+), Greek and Cyrillic are kept — DejaVu has them.
_EMOJI_RE = re.compile(
    "["
    "\U0001F000-\U0001FAFF"
    "\U00002600-\U000026FF"
    "\U00002700-\U000027BF"
    "\U0001F1E6-\U0001F1FF"
    "\U0000FE00-\U0000FE0F"
    "\u200d\u20e3"
    "]+",
    flags=re.UNICODE,
)

# Inline Markdown markers reduced to their text for a readable PDF.
_MD_BOLD = re.compile(r"\*\*(.+?)\*\*|__(.+?)__", re.DOTALL)
_MD_ITALIC = re.compile(r"(?<!\*)\*([^*\n]+)\*(?!\*)|(?<!_)_([^_\n]+)_(?!_)")
_MD_CODE = re.compile(r"`([^`]+)`")
_MD_STRIKE = re.compile(r"~~(.+?)~~", re.DOTALL)
_MD_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


def _strip_inline(text: str) -> str:
    text = _MD_LINK.sub(r"\1 (\2)", text)
    text = _MD_BOLD.sub(lambda m: m.group(1) or m.group(2), text)
    text = _MD_STRIKE.sub(r"\1", text)
    text = _MD_ITALIC.sub(lambda m: m.group(1) or m.group(2) or "", text)
    text = _MD_CODE.sub(r"\1", text)
    return text


def _pdf_safe(text: str) -> str:
    return _EMOJI_RE.sub("", _strip_inline(text))


def _font_paths() -> dict[str, str] | None:
    """DejaVu TTFs bundled with matplotlib (regular / bold / mono)."""
    try:
        from matplotlib import font_manager as fm
    except Exception:  # noqa: BLE001 - matplotlib missing -> no PDF
        return None
    try:
        regular = fm.findfont(fm.FontProperties(family="DejaVu Sans"))
        bold = fm.findfont(fm.FontProperties(family="DejaVu Sans", weight="bold"))
        mono = fm.findfont(fm.FontProperties(family="DejaVu Sans Mono"))
    except Exception:  # noqa: BLE001 - font resolution failed
        return None
    if not regular or not os.path.exists(regular):
        return None
    return {
        "regular": regular,
        "bold": bold if bold and os.path.exists(bold) else regular,
        "mono": mono if mono and os.path.exists(mono) else regular,
    }


def pdf_available() -> bool:
    """True when both fpdf2 and a Unicode font are usable."""
    try:
        import fpdf  # noqa: F401
    except Exception:  # noqa: BLE001 - fpdf2 not installed
        return False
    return _font_paths() is not None


def _mc(pdf: Any, height: float, text: str, **kwargs: Any) -> None:
    """``multi_cell`` that returns to the left margin on the next line.

    fpdf2 ≥2.7 defaults ``new_x`` to RIGHT, which would leave zero width for
    the following full-width cell; pin the cursor back explicitly.
    """
    pdf.multi_cell(0, height, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT, **kwargs)


def _render_entry_pdf(pdf: Any, entry: ExportEntry) -> None:
    """Render one entry (role heading + Markdown-aware content) into the PDF."""
    heading = "Assistant" if entry.role == "assistant" else "User"
    stamp = _format_ts(entry.created_at)
    if stamp:
        heading += f" - {stamp}"
    if entry.role == "assistant" and entry.model:
        heading += f"  [{entry.model}]"

    pdf.set_font("DejaVu", "B", 12)
    pdf.set_text_color(30, 80, 160)
    _mc(pdf, 6, heading)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(1)

    in_code = False
    for raw_line in entry.content.split("\n"):
        line = raw_line.rstrip()
        if line.lstrip().startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            pdf.set_font("DejaVuMono", "", 9)
            pdf.set_fill_color(244, 244, 244)
            _mc(pdf, 4.6, _EMOJI_RE.sub("", line) or " ", fill=True)
            continue
        if not line.strip():
            pdf.ln(2.5)
            continue
        pdf.set_font("DejaVu", "", 10.5)
        _mc(pdf, 5.4, _pdf_safe(line))
    pdf.ln(2)
    pdf.set_draw_color(200, 200, 200)
    y = pdf.get_y()
    pdf.line(pdf.l_margin, y, pdf.w - pdf.r_margin, y)
    pdf.ln(4)


def render_pdf(
    entries: list[ExportEntry], *, title: str, meta: list[str]
) -> bytes | None:
    """Render the export as a PDF; None when fpdf2/fonts are unavailable."""
    try:
        from fpdf import FPDF
    except Exception:  # noqa: BLE001 - fpdf2 not installed
        logger.warning("fpdf2 is not installed; PDF export unavailable")
        return None
    fonts = _font_paths()
    if fonts is None:
        logger.warning("No Unicode font for PDF export; falling back to Markdown")
        return None

    try:
        pdf = FPDF(format="A4")
        pdf.set_margins(15, 15, 15)
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_font("DejaVu", "", fonts["regular"])
        pdf.add_font("DejaVu", "B", fonts["bold"])
        pdf.add_font("DejaVuMono", "", fonts["mono"])
        pdf.add_page()

        pdf.set_font("DejaVu", "B", 17)
        _mc(pdf, 9, _EMOJI_RE.sub("", title))
        pdf.ln(1)
        pdf.set_font("DejaVu", "", 9.5)
        pdf.set_text_color(90, 90, 90)
        for item in meta:
            _mc(pdf, 5, _EMOJI_RE.sub("", item))
        pdf.set_text_color(0, 0, 0)
        pdf.ln(3)

        for entry in entries:
            _render_entry_pdf(pdf, entry)

        pdf.set_font("DejaVu", "", 8.5)
        pdf.set_text_color(120, 120, 120)
        footer = "Exported from Kapum ChatBot"
        username = get_bot_username()
        if username:
            footer += f" (@{username})"
        _mc(pdf, 5, footer)
        return bytes(pdf.output())
    except Exception:
        logger.exception("PDF rendering failed")
        return None


# --- High-level builders -------------------------------------------------------


def render_file(
    fmt: str,
    entries: list[ExportEntry],
    *,
    title: str,
    meta: list[str],
) -> tuple[bytes, str, str] | None:
    """Build the export file for ``fmt`` ("md" or "pdf").

    Returns ``(bytes, filename, mime)`` or None when nothing could be produced
    (e.g. PDF requested but unavailable and Markdown also failed).
    """
    fmt = (fmt or "md").lower()
    if fmt == "pdf":
        data = render_pdf(entries, title=title, meta=meta)
        if data:
            return data, _filename("pdf"), "application/pdf"
        # PDF unavailable -> fall back to Markdown so the user still gets a file.
        fmt = "md"
    markdown = render_markdown(entries, title=title, meta=meta)
    if not markdown:
        return None
    return markdown.encode("utf-8"), _filename("md"), "text/markdown"


def _filename(ext: str) -> str:
    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d_%H%M%S")
    return f"chatbot_export_{stamp}.{ext}"


def build_meta(chat_id: int, count: int, *, scope: str) -> list[str]:
    """The small header block (chat, timestamp, message count)."""
    now = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d %H:%M UTC")
    return [
        f"Scope: {scope}",
        f"Chat ID: {chat_id}",
        f"Exported: {now}",
        f"Messages: {count}",
        f"Max messages per export: {config.export_max_messages}",
    ]
