"""Render display math and Markdown tables locally as PNG (no TeX needed)."""
from __future__ import annotations

import asyncio
import io
import logging
import re
import textwrap
import threading
from functools import lru_cache

logger = logging.getLogger(__name__)

# Code takes precedence, including an unfinished fenced code block.
_CODE = r"```[\s\S]*?(?:```|\Z)|`[^`\n]*`"
_PARTS = re.compile(
    rf"(?P<code>{_CODE})"
    r"|(?P<math>\\\[[\s\S]*?\\\]|\$\$[\s\S]*?\$\$)"
)
_RENDER_LOCK = threading.Lock()
MAX_FORMULA_CHARS = 2000
MAX_FORMULA_IMAGES = 6
MAX_TABLE_IMAGES = 4
TABLE_ROWS_PER_IMAGE = 25
MAX_TABLE_WIDTH_PX = 4000

# Inline math with constructs that read badly as Unicode: shown as an image.
_COMPLEX_TEX = re.compile(
    r"\\(?:d?frac|tfrac|sqrt|sum|prod|int|iint|oint|lim|binom|begin|overline|"
    r"widehat|underbrace|overbrace|vec|hat|bar|partial|nabla|mathbb|mathcal)\b"
)
_INLINE_SPLIT = re.compile(rf"({_CODE})")
_INLINE_DOLLAR = re.compile(r"(?<![\$\\])\$(?!\$)([^\n$]+?)(?<!\\)\$(?!\$)")
_INLINE_PAREN = re.compile(r"\\\(([\s\S]+?)\\\)")

_LINE_ENVS = re.compile(
    r"\\begin\{(aligned|align\*?|gathered|split|eqnarray\*?|equation\*?)\}"
    r"([\s\S]*?)\\end\{\1\}"
)


def promote_complex_inline(text: str) -> str:
    """Turn inline ``$…$`` / ``\\(…\\)`` with complex TeX into display blocks.

    An image cannot sit inside a sentence, so such a formula becomes its own
    ``\\[…\\]`` block (rendered as a picture) between two paragraphs.
    """

    def promote(match: re.Match[str]) -> str:
        body = match.group(1).strip()
        if body and _COMPLEX_TEX.search(body):
            return f"\n\n\\[{body}\\]\n\n"
        return match.group(0)

    parts = _INLINE_SPLIT.split(text)
    for index in range(0, len(parts), 2):  # odd items are code
        segment = _INLINE_DOLLAR.sub(promote, parts[index])
        parts[index] = _INLINE_PAREN.sub(promote, segment)
    return "".join(parts)


# --- Tables and display math ------------------------------------------------

_TABLE_ROW = re.compile(r"^\s{0,3}\|.*\|\s*$")


def _cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _is_separator(line: str) -> bool:
    stripped = line.strip()
    if "|" not in stripped or "-" not in stripped:
        return False
    return all(ch in "|-: " for ch in stripped) and len(_cells(stripped)) >= 2


def split_tables(text: str) -> list[tuple[str, str]]:
    """Split text into ``("text", …)`` and ``("table", …)`` parts.

    Tables inside fenced code blocks are left alone.
    """
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    parts: list[tuple[str, str]] = []
    buffer: list[str] = []
    in_fence = False
    index = 0

    def flush() -> None:
        if buffer:
            parts.append(("text", "\n".join(buffer)))
            buffer.clear()

    while index < len(lines):
        line = lines[index]
        if line.strip().startswith("```"):
            in_fence = not in_fence
        if (
            not in_fence
            and _TABLE_ROW.match(line)
            and index + 1 < len(lines)
            and _is_separator(lines[index + 1])
        ):
            flush()
            rows = [line, lines[index + 1]]
            index += 2
            while index < len(lines) and _TABLE_ROW.match(lines[index]):
                rows.append(lines[index])
                index += 1
            parts.append(("table", "\n".join(rows)))
            continue
        buffer.append(line)
        index += 1
    flush()
    return parts


def has_table(text: str) -> bool:
    """True when the text contains a Markdown table outside code fences."""
    return any(kind == "table" for kind, _ in split_tables(text))


def split_display_math(text: str) -> list[tuple[str, str]]:
    """Return ordered text/math parts, preserving code and math delimiters."""
    parts: list[tuple[str, str]] = []
    start = 0
    for match in _PARTS.finditer(text):
        if not match.group("math"):
            continue
        if match.start() > start:
            parts.append(("text", text[start:match.start()]))
        parts.append(("math", match.group()))
        start = match.end()
    if start < len(text):
        parts.append(("text", text[start:]))
    return parts


def split_blocks(text: str) -> list[tuple[str, str]]:
    """Ordered ``text`` / ``math`` / ``table`` parts of an answer."""
    blocks: list[tuple[str, str]] = []
    for kind, body in split_tables(text):
        if kind == "table":
            blocks.append((kind, body))
        else:
            blocks.extend(split_display_math(body))
    return blocks


# --- Formula images ---------------------------------------------------------


def _prepare_tex(expression: str) -> str:
    """Map common LaTeX that mathtext lacks onto what it understands."""
    expression = re.sub(r"\\[dt]frac\b", r"\\frac", expression)
    expression = re.sub(r"\\(?:displaystyle|textstyle|limits|nolimits)\b", "", expression)
    expression = re.sub(r"\\operatorname\*?\{([^{}]*)\}", r"\\mathrm{\1}", expression)
    expression = re.sub(r"\\(?:quad|qquad)\b", r"\\ \\ ", expression)
    expression = re.sub(r"\\[;:!]", r"\\,", expression)

    def chem(match: re.Match[str]) -> str:
        body = re.sub(r"(?<=[A-Za-z\)\]])(\d+)", r"_{\1}", match.group(1))
        body = body.replace("->", r"\rightarrow ").replace("<=>", r"\rightleftharpoons ")
        return r"\mathrm{" + body.replace(" ", r"\ ") + "}"

    return re.sub(r"\\ce\{([^{}]*)\}", chem, expression)


def _expression_lines(block: str) -> list[str]:
    """Split a display block into lines (``\\\\`` and aligned/gathered envs)."""
    expression = block[2:-2].strip()
    env = _LINE_ENVS.fullmatch(expression)
    if env:
        expression = env.group(2)
    if "\\begin" in expression:
        return []  # matrices and other environments are unsupported
    lines = re.split(r"\\\\(?:\[[^\]]*\])?", expression)
    cleaned = [line.replace("&", " ").replace("\\nonumber", "").strip() for line in lines]
    return [line for line in cleaned if line]


def _mathtext_png(expression: str) -> bytes | None:
    """One line of TeX as a PNG on a transparent-free white background."""
    from matplotlib import rc_context
    from matplotlib.font_manager import FontProperties
    from matplotlib.mathtext import MathTextParser, math_to_image
    from PIL import Image

    expression = "$" + _prepare_tex(expression) + "$"
    prop = FontProperties(size=20)
    with rc_context({"text.usetex": False, "mathtext.fontset": "dejavusans",
                     "savefig.facecolor": "white", "savefig.transparent": False}):
        layout = MathTextParser("path").parse(expression, dpi=160, prop=prop)
        if layout.width > 3600 or layout.height > 1600:
            return None
        output = io.BytesIO()
        math_to_image(expression, output, prop=prop, dpi=160, format="png", color="black")
    output.seek(0)
    with Image.open(output) as image:
        rgba = image.convert("RGBA")
        canvas = Image.new("RGB", rgba.size, "white")
        canvas.paste(rgba, mask=rgba.getchannel("A"))
        result = io.BytesIO()
        canvas.save(result, format="PNG")
        return result.getvalue()


@lru_cache(maxsize=64)
def _render_png(block: str) -> bytes | None:
    if len(block) > MAX_FORMULA_CHARS:
        return None
    lines = _expression_lines(block)
    if not lines or any("$" in line for line in lines) or len(lines) > 12:
        return None
    try:
        # Matplotlib's state and caches are shared: serialize all rendering.
        with _RENDER_LOCK:
            from PIL import Image, ImageOps

            rows = []
            for line in lines:
                png = _mathtext_png(line)
                if png is None:
                    return None
                rows.append(Image.open(io.BytesIO(png)).convert("RGB"))
            width = max(row.width for row in rows)
            height = sum(row.height for row in rows)
            canvas = Image.new("RGB", (width, height), "white")
            top = 0
            for row in rows:
                canvas.paste(row, ((width - row.width) // 2, top))
                top += row.height
            padded = ImageOps.expand(canvas, border=24, fill="white")
            result = io.BytesIO()
            padded.save(result, format="PNG")
            return result.getvalue()
    except Exception as exc:  # malformed or unsupported math must keep its text
        logger.debug("Formula rendering fell back to text: %s", exc)
        return None


async def render_formula(block: str) -> bytes | None:
    """Keep CPU work off the bot's event loop."""
    if len(block) > MAX_FORMULA_CHARS:
        return None
    return await asyncio.to_thread(_render_png, block)


# --- Table images -----------------------------------------------------------

_EMPHASIS = re.compile(r"\*\*|__|~~|(?<!\w)[*_](?=\S)|(?<=\S)[*_](?!\w)|`")
_MATH_SPAN = re.compile(r"\$([^$\n]+)\$")


def _cell_text(cell: str) -> tuple[str, bool]:
    """Plain text for a cell and whether it is a mathtext string.

    Complex TeX stays in ``$…$`` for matplotlib's mathtext; everything else is
    rewritten as Unicode (H₂SO₄, α, x²) and any stray ``$`` is escaped.
    """
    from bot.services.science import normalize_formulas

    spans = _MATH_SPAN.findall(cell)
    if spans and any(_COMPLEX_TEX.search(span) for span in spans):
        text = _MATH_SPAN.sub(lambda m: "$" + _prepare_tex(m.group(1)) + "$", cell)
        return _EMPHASIS.sub("", text), True
    text = normalize_formulas(cell) or cell
    text = _EMPHASIS.sub("", text).replace("$", r"\$")
    return text, False


def _fit(value: str, width: int = 30) -> str:
    return "\n".join(
        textwrap.fill(line, width=width, break_long_words=True) or " "
        for line in value.split("\n")
    )


def _table_chunk_png(header: list[str], rows: list[list[str]]) -> bytes | None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle

    dpi = 150
    columns = len(header)
    figure = Figure(dpi=dpi)
    FigureCanvasAgg(figure)
    renderer = figure.canvas.get_renderer()

    grid = [header, *[[*row, *[""] * (columns - len(row))][:columns] for row in rows]]
    artists: list[list] = []
    sizes: list[list[tuple[float, float]]] = []
    for row_index, row in enumerate(grid):
        artist_row, size_row = [], []
        for cell in row:
            text, is_math = _cell_text(cell)
            if not is_math:
                text = _fit(text)
            weight = "bold" if row_index == 0 else "normal"
            artist = figure.text(0, 0, text, fontsize=11, fontweight=weight, va="center")
            try:
                box = artist.get_window_extent(renderer)
            except Exception:  # mathtext rejected the cell: fall back to plain text
                artist.remove()
                plain = _fit(_EMPHASIS.sub("", re.sub(r"[$\\{}]", "", cell)))
                artist = figure.text(0, 0, plain, fontsize=11, fontweight=weight, va="center")
                box = artist.get_window_extent(renderer)
            artist_row.append(artist)
            size_row.append((box.width, box.height))
        artists.append(artist_row)
        sizes.append(size_row)

    pad_x, pad_y = 16, 12
    col_widths = [
        max(sizes[r][c][0] for r in range(len(grid))) + 2 * pad_x for c in range(columns)
    ]
    row_heights = [max(h for _, h in row) + 2 * pad_y for row in sizes]
    width, height = sum(col_widths), sum(row_heights)
    if width > MAX_TABLE_WIDTH_PX or height > 6000:
        return None

    figure.set_size_inches(width / dpi, height / dpi)
    y = height
    for r, artist_row in enumerate(artists):
        y -= row_heights[r]
        x = 0.0
        shade = "#e3ebf6" if r == 0 else ("#f6f7f9" if r % 2 == 0 else "#ffffff")
        for c, artist in enumerate(artist_row):
            figure.add_artist(
                Rectangle(
                    (x / width, y / height), col_widths[c] / width, row_heights[r] / height,
                    transform=figure.transFigure, facecolor=shade,
                    edgecolor="#b9bec7", linewidth=0.8,
                )
            )
            artist.set_position(((x + pad_x) / width, (y + row_heights[r] / 2) / height))
            artist.set_zorder(5)
            x += col_widths[c]

    output = io.BytesIO()
    figure.savefig(output, format="png", facecolor="white")
    return output.getvalue()


def _render_table(block: str) -> list[bytes] | None:
    lines = block.split("\n")
    header = _cells(lines[0])
    rows = [_cells(line) for line in lines[2:] if line.strip()]
    if len(header) < 2 or not rows:
        return None
    images: list[bytes] = []
    try:
        with _RENDER_LOCK:
            for start in range(0, len(rows), TABLE_ROWS_PER_IMAGE):
                png = _table_chunk_png(header, rows[start:start + TABLE_ROWS_PER_IMAGE])
                if png is None:
                    return None
                images.append(png)
    except Exception as exc:  # the caller keeps the text version
        logger.warning("Table rendering fell back to text: %s", exc)
        return None
    return images


async def render_table(block: str) -> list[bytes] | None:
    """Render a Markdown table as one or more PNGs (long tables are split)."""
    return await asyncio.to_thread(_render_table, block)
