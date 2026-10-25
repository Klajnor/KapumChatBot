"""XLSX / PPTX analysis: type detection and text extraction from real files."""

from __future__ import annotations

import unittest
from io import BytesIO
from unittest.mock import patch

from bot.services import documents
from bot.services.documents import detect_file_type, extract_document_text


def _xlsx() -> bytes:
    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Sales"
    ws.append(["Item", "Qty", "Price"])
    ws.append(["Apple", 3, 1.5])
    ws.append([None, None, None])
    ws.append(["Pear", 2.0, None])
    wb.create_sheet("Empty")
    wb.create_sheet("Notes").append(["hello\nworld"])
    buffer = BytesIO()
    wb.save(buffer)
    return buffer.getvalue()


def _pptx() -> bytes:
    import pptx
    from pptx.util import Inches

    prs = pptx.Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = "Roadmap"
    slide.placeholders[1].text = "Ship the bot"
    slide.notes_slide.notes_text_frame.text = "Say hi"
    table = prs.slides.add_slide(prs.slide_layouts[6]).shapes.add_table(
        2, 2, Inches(1), Inches(1), Inches(4), Inches(1)
    ).table
    table.cell(0, 0).text = "Q1"
    table.cell(0, 1).text = "Q2"
    table.cell(1, 0).text = "10"
    table.cell(1, 1).text = "20"
    buffer = BytesIO()
    prs.save(buffer)
    return buffer.getvalue()


class OfficeDetectionTests(unittest.TestCase):
    def test_by_extension_and_mime(self):
        self.assertEqual(detect_file_type("Report.XLSX"), "xlsx")
        self.assertEqual(detect_file_type("deck.pptx"), "pptx")
        self.assertEqual(
            detect_file_type(
                None,
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ),
            "xlsx",
        )
        self.assertEqual(
            detect_file_type(
                None,
                "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            ),
            "pptx",
        )
        self.assertIsNone(detect_file_type("old.xls"))
        self.assertIsNone(detect_file_type("old.ppt"))


class OfficeExtractionTests(unittest.IsolatedAsyncioTestCase):
    async def test_xlsx_rows_and_sheets(self):
        text = await extract_document_text("xlsx", _xlsx())
        self.assertIn("## Sheet: Sales", text)
        self.assertIn("Item | Qty | Price", text)
        self.assertIn("Apple | 3 | 1.5", text)
        self.assertIn("Pear | 2", text)  # 2.0 renders as 2, trailing blanks dropped
        self.assertNotIn("Pear | 2 |", text)
        self.assertNotIn("Sheet: Empty", text)  # empty sheets are skipped
        self.assertIn("hello world", text)  # newlines inside a cell are flattened

    async def test_pptx_slides_tables_and_notes(self):
        text = await extract_document_text("pptx", _pptx())
        self.assertIn("## Slide 1", text)
        self.assertIn("Roadmap", text)
        self.assertIn("Ship the bot", text)
        self.assertIn("[Notes] Say hi", text)
        self.assertIn("## Slide 2", text)
        self.assertIn("Q1 | Q2", text)
        self.assertIn("10 | 20", text)

    async def test_garbage_returns_none(self):
        self.assertIsNone(await extract_document_text("xlsx", b"not a zip"))
        self.assertIsNone(await extract_document_text("pptx", b"not a zip"))

    async def test_expansion_guard(self):
        with patch.object(documents, "_MAX_EXPANDED_BYTES", 10):
            self.assertIsNone(await extract_document_text("xlsx", _xlsx()))
            self.assertIsNone(await extract_document_text("pptx", _pptx()))

    async def test_row_cap(self):
        with patch.object(documents, "_MAX_SHEET_ROWS", 2):
            text = await extract_document_text("xlsx", _xlsx())
        self.assertIn("Item | Qty | Price", text)
        self.assertNotIn("Pear", text)
