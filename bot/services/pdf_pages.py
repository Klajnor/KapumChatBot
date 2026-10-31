"""Ограниченный обход PDF-страниц без полного pdfplumber.PDF.pages."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from io import BytesIO
from typing import Any

from bot.services.pdf_limits import PdfResourceLimit

MAX_PAGE_TREE_NODES = 4096
MAX_PAGE_TREE_DEPTH = 128


@dataclass
class PdfPages:
    """Выбранные страницы; total_pages неизвестно, если остался хвост."""

    pages: Iterator[Any]
    total_pages: int | None = None


def _page_nodes(
    document: Any, check: Callable[[], None]
) -> Iterator[tuple[Any, dict[str, Any]]]:
    """Обойти дерево итеративно, не доверяя Count и не создавая PDFPage."""
    from pdfminer.pdfpage import PDFPage
    from pdfminer.pdftypes import PDFObjRef
    from pdfminer.psparser import LIT

    def resolve(value: Any) -> Any:
        # resolve1 не ограничивает цепочки косвенных ссылок.
        for _ in range(MAX_PAGE_TREE_DEPTH):
            check()
            if not isinstance(value, PDFObjRef):
                return value
            value = value.resolve()
        raise PdfResourceLimit("PDF page reference depth exceeded")

    root = document.catalog.get("Pages")
    if root is None:
        raise ValueError("PDF has no page tree")
    stack = [(iter([root]), {})]
    visited: set[Any] = set()
    nodes = 0
    while stack:
        check()
        children, parent = stack[-1]
        try:
            ref = next(children)
        except StopIteration:
            stack.pop()
            continue
        nodes += 1
        if nodes > MAX_PAGE_TREE_NODES:
            raise PdfResourceLimit("PDF page tree node limit exceeded")
        key = ("ref", ref.objid) if isinstance(ref, PDFObjRef) else ("inline", id(ref))
        if key in visited:
            continue
        visited.add(key)
        raw = resolve(ref)
        if not isinstance(raw, dict):
            raise TypeError("Invalid PDF page node")
        attrs = {**parent, **raw}
        kind = attrs.get("Type", attrs.get("type"))
        if kind is LIT("Page"):
            yield key[1], attrs
        elif kind is LIT("Pages"):
            kids = resolve(attrs.get("Kids", []))
            if not isinstance(kids, list):
                raise ValueError("Invalid PDF page children")
            if len(stack) >= MAX_PAGE_TREE_DEPTH:
                raise PdfResourceLimit("PDF page tree depth exceeded")
            inherited = {k: attrs[k] for k in PDFPage.INHERITABLE_ATTRS if k in attrs}
            stack.append((iter(kids), inherited))
        else:
            raise ValueError("Invalid PDF page node type")


@contextmanager
def open_pdf_pages(
    data: bytes, limit: int, check: Callable[[], None]
) -> Iterator[PdfPages]:
    """Открыть не более limit страниц, проверяя дедлайн между узлами.

    Один дополнительный лист проверяется без создания Page, чтобы отличить
    точное число от обрезанного документа. Count и labels не перечисляются.
    Очистка намеренно НЕ вызывает PDF.close(): тот заново строит PDF.pages.
    """
    import pdfplumber
    from pdfminer.pdfpage import PDFPage
    from pdfplumber.page import Page

    if limit < 1:
        raise ValueError("PDF page limit must be positive")
    check()
    with BytesIO(data) as source:
        pdf = pdfplumber.open(source)
        created: list[Any] = []

        def pages() -> Iterator[Any]:
            doctop = 0
            count = 0
            for pageid, attrs in _page_nodes(pdf.doc, check):
                if count == limit:
                    return  # Точное общее число неизвестно: есть ещё страницы.
                check()
                raw_page = PDFPage(pdf.doc, pageid, attrs, label=None)
                page = Page(pdf, raw_page, page_number=count + 1, initial_doctop=doctop)
                created.append(page)
                doctop += page.height
                count += 1
                yield page
            batch.total_pages = count

        iterator = pages()
        batch = PdfPages(iterator)
        try:
            yield batch
        finally:
            iterator.close()
            for page in created:
                page.close()
            pdf.flush_cache()
