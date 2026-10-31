"""H-2: ограниченный обход страниц, включая очистку и враждебное дерево."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import PropertyMock, patch

from pdfminer.pdfpage import PDFPage
from pdfminer.psparser import LIT

from bot.services import documents, pdf_ocr, pdf_pages
from bot.services.pdf_limits import PdfResourceLimit
from bot.services.pdf_pages import _page_nodes, open_pdf_pages


def _pdf(pages: int, *, count: int | None = None) -> bytes:
    kids = " ".join(f"{number + 3} 0 R" for number in range(pages))
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        (
            f"<< /Type /Pages /Count {pages if count is None else count} "
            f"/Kids [{kids}] /MediaBox [0 0 100 200] /Rotate 90 >>"
        ).encode(),
    ]
    objects.extend(b"<< /Type /Page /Parent 2 0 R >>" for _ in range(pages))
    data = b"%PDF-1.4\n"
    offsets = []
    for number, obj in enumerate(objects, 1):
        offsets.append(len(data))
        data += f"{number} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref = len(data)
    data += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    data += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets)
    return data + (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref}\n%%EOF\n"
    ).encode()


class PageIterationTests(unittest.TestCase):
    def test_thousand_pages_constructs_only_selected_pages_and_never_uses_full_list(self):
        import pdfplumber

        constructed = []
        original = PDFPage.__init__

        def init(page, *args, **kwargs):
            constructed.append(args[1])
            return original(page, *args, **kwargs)

        with (
            patch.object(PDFPage, "__init__", init),
            patch.object(pdfplumber.PDF, "pages", new_callable=PropertyMock) as full_list,
        ):
            full_list.side_effect = AssertionError("Unbounded PDF.pages was accessed")
            with open_pdf_pages(_pdf(1000), 3, lambda: None) as batch:
                pages = list(batch.pages)
                self.assertEqual([page.page_number for page in pages], [1, 2, 3])
                self.assertIsNone(batch.total_pages)
            full_list.assert_not_called()
        self.assertEqual(len(constructed), 3)

    def test_exact_total_only_for_complete_tree_and_count_is_not_trusted(self):
        for claimed in (0, 1, 999999):
            with self.subTest(claimed=claimed):
                with open_pdf_pages(_pdf(3, count=claimed), 3, lambda: None) as batch:
                    self.assertEqual(len(list(batch.pages)), 3)
                    self.assertEqual(batch.total_pages, 3)
                with open_pdf_pages(_pdf(4, count=claimed), 3, lambda: None) as batch:
                    self.assertEqual(len(list(batch.pages)), 3)
                    self.assertIsNone(batch.total_pages)

    def test_inherited_page_geometry_and_rotation_are_preserved(self):
        with open_pdf_pages(_pdf(1), 1, lambda: None) as batch:
            page = next(batch.pages)
            self.assertEqual(page.rotation, 90)
            self.assertEqual((page.width, page.height), (200, 100))
            self.assertEqual(page.page_obj.mediabox, (0, 0, 100, 200))

    def test_early_exit_cleanup_never_enumerates_remaining_pages(self):
        import pdfplumber

        with patch.object(pdfplumber.PDF, "pages", new_callable=PropertyMock) as full_list:
            full_list.side_effect = AssertionError("Cleanup enumerated all pages")
            with open_pdf_pages(_pdf(1000), 200, lambda: None) as batch:
                next(batch.pages)
            full_list.assert_not_called()

    def test_exception_cleanup_never_enumerates_remaining_pages(self):
        import pdfplumber

        with patch.object(pdfplumber.PDF, "pages", new_callable=PropertyMock) as full_list:
            full_list.side_effect = AssertionError("Cleanup enumerated all pages")
            with (
                self.assertRaisesRegex(RuntimeError, "consumer failed"),
                open_pdf_pages(_pdf(1000), 200, lambda: None) as batch,
            ):
                next(batch.pages)
                raise RuntimeError("consumer failed")
            full_list.assert_not_called()

    def test_deadline_checked_on_empty_internal_nodes(self):
        nodes = [{"Type": LIT("Pages"), "Kids": []} for _ in range(100)]
        root = {"Type": LIT("Pages"), "Kids": nodes}
        checked = 0

        def check():
            nonlocal checked
            checked += 1
            if checked >= 10:
                raise documents.DocumentParseTimeout

        with self.assertRaises(documents.DocumentParseTimeout):
            list(_page_nodes(SimpleNamespace(catalog={"Pages": root}), check))
        self.assertEqual(checked, 10)

    def test_node_budget_rejects_large_empty_tree(self):
        root = {"Type": LIT("Pages"), "Kids": [
            {"Type": LIT("Pages"), "Kids": []} for _ in range(20)
        ]}
        with (
            patch.object(pdf_pages, "MAX_PAGE_TREE_NODES", 5),
            self.assertRaises(PdfResourceLimit),
        ):
            list(_page_nodes(SimpleNamespace(catalog={"Pages": root}), lambda: None))

    def test_deep_tree_rejected_without_recursion(self):
        root = {"Type": LIT("Page")}
        for _ in range(20):
            root = {"Type": LIT("Pages"), "Kids": [root]}
        with (
            patch.object(pdf_pages, "MAX_PAGE_TREE_DEPTH", 5),
            self.assertRaises(PdfResourceLimit),
        ):
            list(_page_nodes(SimpleNamespace(catalog={"Pages": root}), lambda: None))

    def test_cycle_does_not_repeat_pages(self):
        root = {"Type": LIT("Pages"), "Kids": []}
        root["Kids"].extend([root, {"Type": LIT("Page")}])
        self.assertEqual(len(list(_page_nodes(SimpleNamespace(catalog={"Pages": root}), lambda: None))), 1)

    def test_cyclic_indirect_reference_is_bounded(self):
        from pdfminer.pdftypes import PDFObjRef

        document = SimpleNamespace()
        ref = PDFObjRef(document, 1)
        document.catalog = {"Pages": ref}
        document.getobj = lambda _id: ref
        with self.assertRaises(PdfResourceLimit):
            list(_page_nodes(document, lambda: None))


class PagePipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_text_character_cap_stops_page_construction_early(self):
        from pdfplumber.page import Page

        constructed = []
        original = PDFPage.__init__

        def init(page, *args, **kwargs):
            constructed.append(args[1])
            return original(page, *args, **kwargs)

        with (
            patch.object(PDFPage, "__init__", init),
            patch.object(Page, "extract_text", return_value="hello"),
            patch.object(documents.config, "max_extracted_chars", 5),
        ):
            self.assertEqual(await documents.extract_document_text("pdf", _pdf(1000)), "hello")
        self.assertEqual(len(constructed), 1)

    async def test_page_tree_resource_limit_propagates_from_both_pipelines(self):
        with patch.object(pdf_pages, "MAX_PAGE_TREE_NODES", 1):
            with self.assertRaises(PdfResourceLimit):
                await documents.extract_document_text("pdf", _pdf(2))
            with self.assertRaises(PdfResourceLimit):
                await pdf_ocr.ocr_pdf(_pdf(2))

    async def test_text_pipeline_and_close_never_load_all_pages(self):
        import pdfplumber

        with (
            patch.object(documents, "_MAX_PDF_PAGES", 2),
            patch.object(pdfplumber.PDF, "pages", new_callable=PropertyMock) as full_list,
        ):
            full_list.side_effect = AssertionError("Unbounded text pipeline")
            self.assertIsNone(await documents.extract_document_text("pdf", _pdf(1000)))
            full_list.assert_not_called()

    async def test_ocr_only_renders_selected_pages_and_reports_unknown_total(self):
        import pdfplumber
        from PIL import Image

        rendered = []

        def render(page, **kwargs):
            rendered.append(page.page_number)
            return SimpleNamespace(original=Image.new("RGB", (20, 20), "white"))

        with (
            patch.object(pdfplumber.PDF, "pages", new_callable=PropertyMock) as full_list,
            patch.object(pdfplumber.page.Page, "to_image", render),
        ):
            full_list.side_effect = AssertionError("Unbounded OCR pipeline")
            images, total = await documents._run_parser(pdf_ocr._render_pages, _pdf(1000), 2)
            full_list.assert_not_called()
        self.assertEqual(len(images), 2)
        self.assertEqual(rendered, [1, 2])
        self.assertIsNone(total)
        self.assertTrue(pdf_ocr.OcrResult("text", 2, None).partial)

    async def test_small_ocr_pdf_has_exact_count(self):
        from pdfplumber.page import Page
        from PIL import Image

        with patch.object(Page, "to_image", return_value=SimpleNamespace(
            original=Image.new("RGB", (20, 20), "white"),
        )):
            images, total = await documents._run_parser(pdf_ocr._render_pages, _pdf(2), 3)
        self.assertEqual((len(images), total), (2, 2))
        self.assertFalse(pdf_ocr.OcrResult("text", 2, 2).partial)
