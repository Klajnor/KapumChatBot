"""Красивый PDF-рендер экспорта: Markdown-блоки, таблицы, формулы, код.

Вместо «стрип разметки» ответ разбирается на блоки (``formulas.split_blocks``
уже умеет отделять таблицы и display-формулы от текста):

* заголовки ``#``…``####``, списки (вложенные, нумерованные, чекбоксы),
  цитаты, горизонтальные линии, блоки кода — родными средствами fpdf2;
* инлайн-разметка (**жирный**, *курсив*, ~~зачёркнутый~~, `код`, ссылки)
  пишется кусками с переключением шрифта DejaVu;
* таблицы — ``pdf.table`` с шапкой, зеброй и выравниванием колонок;
* display-формулы — PNG из mathtext (``formulas._render_png``), простые
  инлайн-формулы — Unicode (``science.normalize_formulas``), сложные
  инлайн-формулы выносятся в отдельные строки (``promote_complex_inline``).

Модуль импортируется лениво из ``export.render_pdf``: fpdf2 необязателен.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import Any

from bot.bot import get_bot_username
from bot.services import formulas
from bot.services.export import _EMOJI_RE, ExportEntry, _format_ts
from bot.services.science import normalize_formulas

# Цвета оформления (RGB).
_ACCENT = (37, 84, 166)
_TEXT = (28, 31, 36)
_MUTED = (110, 116, 126)
_RULE = (210, 214, 220)
_CODE_BG = (244, 245, 247)
_CODE_FG = (176, 52, 78)
_LINK = (20, 90, 200)
_USER_BAND = (232, 238, 248)
_BOT_BAND = (232, 245, 238)
_BOT_ACCENT = (24, 120, 74)
_TABLE_HEAD = (227, 235, 246)
_TABLE_ZEBRA = (247, 248, 250)

_BODY = 10.5
_LINE = 5.6
_MAX_FORMULAS = 80          # потолок картинок-формул на один PDF
_FORMULA_SCALE = 0.7        # mathtext 20pt -> ~14pt: формула читается наравне с текстом

_HEADING_SIZES = {1: 16.0, 2: 13.5, 3: 12.0, 4: 11.0}
_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$")
_BULLET_RE = re.compile(r"^(\s*)[-*+•]\s+(.*)$")
_ORDERED_RE = re.compile(r"^(\s*)(\d{1,3})[.)]\s+(.*)$")
_QUOTE_RE = re.compile(r"^\s{0,3}>\s?(.*)$")
_RULE_RE = re.compile(r"^\s{0,3}([-*_])(?:\s*\1){2,}\s*$")
_TASK_RE = re.compile(r"^\[([ xX])\]\s+(.*)$")
_ALIGN_CELL = re.compile(r"^\s*(:?)-{1,}(:?)\s*$")

# Инлайн: код, ссылка, ***bi***, **b**/__b__, *i*/_i_, ~~s~~ (в порядке приоритета).
_INLINE_RE = re.compile(
    r"`(?P<code>[^`\n]+)`"
    r"|\[(?P<ltext>[^\]\n]+)\]\((?P<lurl>[^)\s]+)\)"
    r"|\*\*\*(?P<bi>.+?)\*\*\*"
    r"|\*\*(?P<b>.+?)\*\*(?!\*)|__(?P<b2>.+?)__"
    r"|(?<![\w*])\*(?P<i>[^*\n]+?)\*(?![\w*])|(?<![\w_])_(?P<i2>[^_\n]+?)_(?![\w_])"
    r"|~~(?P<s>.+?)~~",
    re.DOTALL,
)


@dataclass(slots=True, frozen=True)
class _Style:
    bold: bool = False
    italic: bool = False
    strike: bool = False
    code: bool = False
    link: str | None = None


_PLAIN = _Style()


def clean(text: str) -> str:
    """Убирает эмодзи (в DejaVu их нет), оставляя ✓ ✔ ✗ ✘."""
    return _EMOJI_RE.sub("", text)


def _spans(text: str, style: _Style = _PLAIN) -> list[tuple[str, _Style]]:
    """Разбор инлайн-Markdown в (текст, стиль); вложенность поддерживается."""
    out: list[tuple[str, _Style]] = []
    pos = 0
    for match in _INLINE_RE.finditer(text):
        if match.start() > pos:
            out.append((text[pos:match.start()], style))
        pos = match.end()
        kind = match.lastgroup
        if kind == "code":
            out.append((match.group("code"), _Style(code=True, link=style.link)))
        elif kind == "lurl":
            out.extend(_spans(match.group("ltext"), _Style(
                style.bold, style.italic, style.strike, False, match.group("lurl"))))
        elif kind == "bi":
            out.extend(_spans(match.group("bi"), _Style(True, True, style.strike, False, style.link)))
        elif kind in ("b", "b2"):
            out.extend(_spans(match.group(kind), _Style(True, style.italic, style.strike, False, style.link)))
        elif kind in ("i", "i2"):
            out.extend(_spans(match.group(kind), _Style(style.bold, True, style.strike, False, style.link)))
        elif kind == "s":
            out.extend(_spans(match.group("s"), _Style(style.bold, style.italic, True, False, style.link)))
    if pos < len(text):
        out.append((text[pos:], style))
    return out


def _fpdf_markdown(text: str) -> str:
    """Инлайн-Markdown -> разметка fpdf2 (**жирный**, __курсив__) для ячеек."""
    parts: list[str] = []
    for chunk, style in _spans(text):
        chunk = chunk.replace("**", "").replace("__", "")
        if chunk and style.bold:
            chunk = f"**{chunk}**"
        if chunk and style.italic:
            chunk = f"__{chunk}__"
        parts.append(chunk)
    return "".join(parts)


class _Renderer:
    """Состояние одного PDF: шрифты, счётчик формул, блочные примитивы."""

    def __init__(self, pdf: Any, content_width: float) -> None:
        self.pdf = pdf
        self.width = content_width
        self.formulas_left = _MAX_FORMULAS

    # --- шрифты ---------------------------------------------------------

    def font(self, size: float = _BODY, style: _Style = _PLAIN) -> None:
        pdf = self.pdf
        if style.code:
            pdf.set_font("DejaVuMono", "", size * 0.9)
            pdf.set_text_color(*(_LINK if style.link else _CODE_FG))
            return
        flags = ("B" if style.bold else "") + ("I" if style.italic else "")
        if style.strike:
            flags += "S"
        if style.link:
            flags += "U"
        pdf.set_font("DejaVu", flags, size)
        pdf.set_text_color(*(_LINK if style.link else _TEXT))

    # --- инлайн ---------------------------------------------------------

    def write_inline(
        self, text: str, *, size: float = _BODY, line: float = _LINE,
        base: _Style = _PLAIN, color: tuple[int, int, int] | None = None,
    ) -> None:
        """Пишет абзац с инлайн-форматированием, перенос — по левому полю."""
        pdf = self.pdf
        for chunk, style in _spans(text, base):
            if not chunk:
                continue
            self.font(size, style)
            if color is not None and not style.link and not style.code:
                pdf.set_text_color(*color)
            pdf.write(line, chunk, link=style.link or "")
        pdf.ln(line)
        pdf.set_text_color(*_TEXT)

    # --- блоки ----------------------------------------------------------

    def ensure_space(self, height: float) -> None:
        if self.pdf.will_page_break(height):
            self.pdf.add_page()

    def heading(self, level: int, text: str) -> None:
        pdf = self.pdf
        size = _HEADING_SIZES.get(level, 11.0)
        self.ensure_space(size * 0.9 + 14)
        pdf.ln(2.5 if level > 1 else 3.5)
        self.write_inline(text, size=size, line=size * 0.5 + 1.5,
                          base=_Style(bold=True), color=_ACCENT if level <= 2 else _TEXT)
        if level <= 2:
            y = pdf.get_y()
            pdf.set_draw_color(*_RULE)
            pdf.set_line_width(0.25)
            pdf.line(pdf.l_margin, y, pdf.w - pdf.r_margin, y)
            pdf.ln(1.5)
        else:
            pdf.ln(0.5)

    def list_item(self, indent: int, marker: str, text: str) -> None:
        pdf = self.pdf
        left = pdf.l_margin
        shift = 5.0 + indent * 5.5
        marker_w = 5.0 if len(marker) == 1 else 2.2 + 2.0 * len(marker)
        self.ensure_space(_LINE + 1)
        self.font()
        pdf.set_x(left + shift)
        pdf.set_text_color(*_ACCENT)
        pdf.cell(marker_w, _LINE, marker)
        pdf.set_left_margin(left + shift + marker_w)
        pdf.set_x(left + shift + marker_w)
        self.write_inline(text)
        pdf.set_left_margin(left)
        pdf.set_x(left)

    def quote(self, lines: list[str]) -> None:
        pdf = self.pdf
        left = pdf.l_margin
        pdf.ln(1)
        self.ensure_space(_LINE + 2)
        page, top = pdf.page_no(), pdf.get_y()
        pdf.set_left_margin(left + 5)
        pdf.set_x(left + 5)
        for line in lines:
            if not line.strip():
                pdf.ln(2)
                continue
            self.write_inline(line, base=_Style(italic=True), color=_MUTED)
            pdf.set_x(left + 5)
        pdf.set_left_margin(left)
        pdf.set_x(left)
        if pdf.page_no() == page:
            pdf.set_draw_color(*_ACCENT)
            pdf.set_line_width(0.7)
            pdf.line(left + 1.2, top, left + 1.2, pdf.get_y())
        pdf.ln(1.5)

    def code(self, lines: list[str]) -> None:
        pdf = self.pdf
        pdf.ln(1)
        pdf.set_font("DejaVuMono", "", 8.8)
        pdf.set_text_color(*_TEXT)
        pdf.set_fill_color(*_CODE_BG)
        pdf.set_draw_color(*_RULE)
        for line in lines or [""]:
            pdf.multi_cell(
                0, 4.7, "  " + (line.replace("\t", "    ") or " "), fill=True,
                new_x="LMARGIN", new_y="NEXT", max_line_height=4.7,
            )
        pdf.ln(2.5)

    def rule(self) -> None:
        pdf = self.pdf
        pdf.ln(2)
        y = pdf.get_y()
        pdf.set_draw_color(*_RULE)
        pdf.set_line_width(0.3)
        pdf.line(pdf.l_margin, y, pdf.w - pdf.r_margin, y)
        pdf.ln(3)

    def paragraph(self, text: str) -> None:
        self.write_inline(text)

    # --- формулы ----------------------------------------------------------

    def math(self, block: str) -> None:
        """Display-формула картинкой; при отказе — читаемый Unicode-текст."""
        pdf = self.pdf
        png = None
        if self.formulas_left > 0:
            self.formulas_left -= 1
            png = formulas._render_png(block)
        if png:
            from PIL import Image

            with Image.open(io.BytesIO(png)) as image:
                px_w, px_h = image.size
            mm_per_px = 25.4 / 160 * _FORMULA_SCALE
            width, height = px_w * mm_per_px, px_h * mm_per_px
            if width > self.width:
                height *= self.width / width
                width = self.width
            self.ensure_space(height + 2)
            pdf.image(io.BytesIO(png), x=pdf.l_margin + (self.width - width) / 2,
                      y=pdf.get_y() + 0.5, w=width, h=height)
            pdf.set_y(pdf.get_y() + height + 1.5)
            return
        body = normalize_formulas(block) or block
        pdf.set_font("DejaVu", "I", _BODY)
        pdf.set_text_color(*_TEXT)
        pdf.multi_cell(0, _LINE + 0.6, clean(body).strip(), align="C",
                       new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)

    # --- таблицы ----------------------------------------------------------

    def table(self, block: str) -> None:
        from fpdf.fonts import FontFace

        lines = block.split("\n")
        header = formulas._cells(lines[0])
        aligns = []
        for cell in formulas._cells(lines[1]):
            m = _ALIGN_CELL.match(cell)
            aligns.append("C" if m and m.group(1) and m.group(2)
                          else "R" if m and m.group(2) else "L")
        rows = [formulas._cells(line) for line in lines[2:] if line.strip()]
        columns = len(header)

        def fit(row: list[str]) -> list[str]:
            row = [*row, *[""] * (columns - len(row))][:columns]
            return [_fpdf_markdown(normalize_formulas(clean(c)) or "") for c in row]

        grid = [fit(header), *[fit(r) for r in rows]]
        weights = [
            min(max(max(len(row[c].replace("*", "").replace("_", "")) for row in grid), 4), 42)
            for c in range(columns)
        ]
        pdf = self.pdf
        pdf.ln(1.5)
        self.ensure_space(20)
        pdf.set_font("DejaVu", "", 9.5)
        pdf.set_text_color(*_TEXT)
        pdf.set_draw_color(*_RULE)
        pdf.set_fill_color(*_TABLE_ZEBRA)
        pdf.set_line_width(0.2)
        with pdf.table(
            col_widths=tuple(weights),
            width=self.width,
            text_align=tuple(aligns),
            line_height=5,
            padding=(1.4, 2),
            markdown=True,
            borders_layout="HORIZONTAL_LINES",
            cell_fill_color=_TABLE_ZEBRA,
            cell_fill_mode="EVEN_ROWS",
            headings_style=FontFace(emphasis="BOLD", color=_ACCENT, fill_color=_TABLE_HEAD),
        ) as table:
            for row in grid:
                table_row = table.row()
                for cell in row:
                    table_row.cell(cell)
        pdf.set_x(pdf.l_margin)
        pdf.ln(2.5)

    # --- текст ------------------------------------------------------------

    def text(self, body: str) -> None:
        """Построчный разбор не-табличного, не-формульного текста."""
        lines = normalize_formulas(body).split("\n") if body.strip() else []
        index, count = 0, len(lines)
        while index < count:
            raw = lines[index].rstrip()
            if raw.lstrip().startswith("```"):
                fence: list[str] = []
                index += 1
                while index < count and not lines[index].lstrip().startswith("```"):
                    fence.append(lines[index].rstrip())
                    index += 1
                index += 1
                self.code(fence)
                continue
            if not raw.strip():
                self.pdf.ln(2.2)
            elif (m := _HEADING_RE.match(raw)):
                self.heading(len(m.group(1)), m.group(2))
            elif _RULE_RE.match(raw):
                self.rule()
            elif _QUOTE_RE.match(raw):
                quoted: list[str] = []
                while index < count and (q := _QUOTE_RE.match(lines[index])):
                    quoted.append(q.group(1))
                    index += 1
                self.quote(quoted)
                continue
            elif (m := _BULLET_RE.match(raw)):
                task = _TASK_RE.match(m.group(2))
                marker, item = ("•", m.group(2)) if not task else (
                    "☑" if task.group(1) in "xX" else "☐", task.group(2))
                self.list_item(len(m.group(1).replace("\t", "    ")) // 2, marker, item)
            elif (m := _ORDERED_RE.match(raw)):
                self.list_item(len(m.group(1).replace("\t", "    ")) // 2,
                               f"{m.group(2)}.", m.group(3))
            else:
                self.paragraph(raw.strip())
            index += 1


def _entry(r: _Renderer, entry: ExportEntry) -> None:
    pdf = r.pdf
    assistant = entry.role == "assistant"
    band, accent = (_BOT_BAND, _BOT_ACCENT) if assistant else (_USER_BAND, _ACCENT)
    r.ensure_space(34)
    y, height = pdf.get_y(), 8.0
    pdf.set_fill_color(*band)
    pdf.rect(pdf.l_margin, y, r.width, height, style="F", round_corners=True, corner_radius=1.6)
    pdf.set_fill_color(*accent)
    pdf.rect(pdf.l_margin, y, 1.6, height, style="F")
    label = "Assistant" if assistant else "User"
    if assistant and entry.model:
        label += f"  ·  {entry.model}"
    pdf.set_xy(pdf.l_margin + 4, y)
    pdf.set_font("DejaVu", "B", 10.5)
    pdf.set_text_color(*accent)
    pdf.cell(r.width * 0.62, height, clean(label), new_x="RIGHT", new_y="TOP")
    pdf.set_font("DejaVu", "", 8.5)
    pdf.set_text_color(*_MUTED)
    pdf.cell(r.width - r.width * 0.62 - 4, height, _format_ts(entry.created_at),
             align="R", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_text_color(*_TEXT)

    content = clean(formulas.promote_complex_inline(entry.content.strip()))
    for kind, body in formulas.split_blocks(content):
        if kind == "table":
            r.table(body)
        elif kind == "math":
            r.math(body)
        else:
            r.text(body)
    pdf.set_x(pdf.l_margin)
    pdf.ln(4)


def build_pdf(
    entries: list[ExportEntry], *, title: str, meta: list[str], fonts: dict[str, str]
) -> bytes:
    """Собирает PDF. Исключения пробрасываются — ловит ``export.render_pdf``."""
    from fpdf import FPDF

    class _Doc(FPDF):
        def footer(self) -> None:
            self.set_y(-11)
            self.set_font("DejaVu", "", 8)
            self.set_text_color(*_MUTED)
            self.cell(0, 5, f"{self.page_no()} / {{nb}}", align="C")

    pdf = _Doc(format="A4")
    pdf.set_margins(16, 16, 16)
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.alias_nb_pages()
    pdf.set_title(clean(title))
    pdf.set_creator("Kapum ChatBot")
    pdf.add_font("DejaVu", "", fonts["regular"])
    pdf.add_font("DejaVu", "B", fonts["bold"])
    pdf.add_font("DejaVu", "I", fonts["italic"])
    pdf.add_font("DejaVu", "BI", fonts["bolditalic"])
    pdf.add_font("DejaVuMono", "", fonts["mono"])
    pdf.add_page()
    r = _Renderer(pdf, pdf.w - pdf.l_margin - pdf.r_margin)

    # Шапка документа.
    pdf.set_font("DejaVu", "B", 19)
    pdf.set_text_color(*_ACCENT)
    pdf.multi_cell(0, 9.5, clean(title), new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(*_ACCENT)
    pdf.set_line_width(0.8)
    y = pdf.get_y() + 1.5
    pdf.line(pdf.l_margin, y, pdf.l_margin + 28, y)
    pdf.ln(3.5)
    for item in meta:
        key, sep, value = clean(item).partition(": ")
        pdf.set_font("DejaVu", "B" if sep else "", 8.8)
        pdf.set_text_color(*_MUTED)
        if sep:
            pdf.cell(pdf.get_string_width(key + ":") + 1.5, 4.8, key + ":")
            pdf.set_font("DejaVu", "", 8.8)
            pdf.cell(0, 4.8, value, new_x="LMARGIN", new_y="NEXT")
        else:
            pdf.cell(0, 4.8, key, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    for entry in entries:
        _entry(r, entry)

    pdf.set_font("DejaVu", "I", 8.5)
    pdf.set_text_color(*_MUTED)
    footer = "Exported from Kapum ChatBot"
    username = get_bot_username()
    if username:
        footer += f" (@{username})"
    pdf.multi_cell(0, 5, footer, new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
