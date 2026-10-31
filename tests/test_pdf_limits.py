"""Регрессии H-1: ограничение Flate до материализации огромных PDF-потоков."""

from __future__ import annotations

import threading
import unittest
import zlib
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from pdfminer.pdftypes import PDFStream
from pdfminer.psparser import LIT

from bot.handlers import chat, guest
from bot.services import documents, pdf_limits, pdf_ocr
from bot.services.access import AccessResult
from bot.services.pdf_limits import PdfResourceLimit, pdf_stream_limits


def _stream(body: bytes, filters=None) -> PDFStream:
    return PDFStream({"Filter": filters or LIT("FlateDecode")}, body)


def _pdf(body: bytes) -> bytes:
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
            b"/Resources << /Font << /F1 << /Type /Font /Subtype /Type1 "
            b"/BaseFont /Helvetica >> >> >> >>"
        ),
        f"<< /Length {len(body)} /Filter /FlateDecode >>\nstream\n".encode()
        + body + b"\nendstream",
    ]
    data = b"%PDF-1.4\n"
    offsets = []
    for index, obj in enumerate(objects, 1):
        offsets.append(len(data))
        data += f"{index} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref = len(data)
    data += b"xref\n0 5\n0000000000 65535 f \n"
    data += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets)
    return data + f"trailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()


class FlateLimitsTests(unittest.TestCase):
    def setUp(self):
        # Маленькие лимиты делают регрессии безопасными даже для старого кода.
        self.stream_patch = patch.object(pdf_limits, "MAX_FLATE_STREAM_BYTES", 1024)
        self.total_patch = patch.object(pdf_limits, "MAX_FLATE_DOCUMENT_BYTES", 2048)
        self.stream_patch.start()
        self.total_patch.start()
        self.addCleanup(self.stream_patch.stop)
        self.addCleanup(self.total_patch.stop)

    def test_small_stream_and_exact_limit(self):
        for size in (0, 100, 1024):
            with pdf_stream_limits():
                self.assertEqual(_stream(zlib.compress(b"x" * size)).get_data(), b"x" * size)

    def test_stream_bomb_is_rejected(self):
        with self.assertRaises(PdfResourceLimit), pdf_stream_limits():
            _stream(zlib.compress(b"x" * 4096)).get_data()

    def test_total_budget_across_streams(self):
        with self.assertRaises(PdfResourceLimit), pdf_stream_limits():
            for _ in range(3):
                _stream(zlib.compress(b"x" * 900)).get_data()

    def test_chained_filters_cannot_hide_a_bomb(self):
        packed = zlib.compress(zlib.compress(b"x" * 4096))
        with self.assertRaises(PdfResourceLimit), pdf_stream_limits():
            _stream(packed, [LIT("Fl"), LIT("FlateDecode")]).get_data()

    def test_corrupt_checksum_fallback_is_also_bounded(self):
        from pdfminer.pdftypes import decompress_corrupted

        packed = zlib.compress(b"x" * 4096)[:-1] + b"\x00"
        with self.assertRaises(PdfResourceLimit), pdf_stream_limits():
            decompress_corrupted(packed)

    def test_corrupt_stream_pdfminer_path_is_bounded(self):
        packed = zlib.compress(b"x" * 4096)[:-1] + b"\x00"
        with self.assertRaises(PdfResourceLimit), pdf_stream_limits():
            _stream(packed).get_data()

    def test_guard_never_uses_unbounded_decompress(self):
        packed = zlib.compress(b"ok")
        with (
            pdf_stream_limits(),
            patch.object(zlib, "decompress", side_effect=AssertionError("unbounded")),
        ):
            self.assertEqual(_stream(packed).get_data(), b"ok")

    def test_incremental_decoder_cannot_reset_per_stream_budget(self):
        from pdfminer import pdftypes

        with self.assertRaises(PdfResourceLimit), pdf_stream_limits():
            inflater = pdftypes.zlib.decompressobj()
            packed = zlib.compress(b"x" * 4096)
            inflater.decompress(packed, 600)
            inflater.decompress(inflater.unconsumed_tail, 600)

    def test_library_cannot_swallow_rejection(self):
        with self.assertRaises(PdfResourceLimit), pdf_stream_limits():
            try:
                _stream(zlib.compress(b"x" * 4096)).get_data()
            except ValueError:
                pass

    def test_parallel_documents_have_independent_budgets(self):
        barrier = threading.Barrier(2)

        def parse():
            with pdf_stream_limits():
                barrier.wait(timeout=5)
                for _ in range(2):
                    self.assertEqual(len(_stream(zlib.compress(b"x" * 900)).get_data()), 900)

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(parse) for _ in range(2)]
            for future in futures:
                future.result(timeout=10)

    def test_stdlib_zlib_and_context_cleanup_are_unchanged(self):
        original = zlib.decompress
        with self.assertRaises(PdfResourceLimit), pdf_stream_limits():
            _stream(zlib.compress(b"x" * 4096)).get_data()
        self.assertIs(zlib.decompress, original)
        self.assertIsNone(pdf_limits._budget.get())
        self.assertEqual(len(zlib.decompress(zlib.compress(b"x" * 4096))), 4096)
        with pdf_stream_limits():
            self.assertEqual(_stream(zlib.compress(b"ok")).get_data(), b"ok")


class PdfIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_pdf_rejected_without_large_allocations(self):
        with (
            patch.object(pdf_limits, "MAX_FLATE_STREAM_BYTES", 1024),
            self.assertRaises(PdfResourceLimit),
        ):
            await documents.extract_document_text("pdf", _pdf(zlib.compress(b" " * 4096)))

    async def test_normal_pdf_still_parses(self):
        body = b"BT /F1 12 Tf 100 700 Td (Hello) Tj ET"
        self.assertEqual(await documents.extract_document_text("pdf", _pdf(zlib.compress(body))), "Hello")

    async def test_ocr_pdfminer_metadata_rejection_is_not_swallowed(self):
        from pdfminer import pdftypes

        def fail_on_open(*args, **kwargs):
            pdftypes.zlib.decompress(zlib.compress(b"x" * 4096))

        with (
            patch.object(pdf_limits, "MAX_FLATE_STREAM_BYTES", 1024),
            patch("pdfplumber.open", side_effect=fail_on_open),
            self.assertRaises(PdfResourceLimit),
        ):
            await pdf_ocr.ocr_pdf(b"pdf")

    async def test_handler_refunds_and_never_falls_back_to_ocr(self):
        message = SimpleNamespace(
            from_user=SimpleNamespace(id=5),
            chat=SimpleNamespace(id=5, type="private"),
            document=SimpleNamespace(
                file_name="bomb.pdf", mime_type="application/pdf", file_size=1000, file_id="f",
            ),
            caption=None,
            bot=SimpleNamespace(
                send_chat_action=AsyncMock(),
                get_file=AsyncMock(return_value=SimpleNamespace(file_path="p")),
                download_file=AsyncMock(return_value=SimpleNamespace(read=lambda: b"pdf")),
            ),
        )

        @asynccontextmanager
        async def waiting(_message):
            yield

        service = SimpleNamespace(session=SimpleNamespace(close=AsyncMock()))
        access = AccessResult(allowed=True, role="user", consumed=True)
        with (
            patch.object(chat, "_is_addressed", return_value=True),
            patch.object(chat, "_open_registered", AsyncMock(return_value=service)),
            patch.object(chat, "_consume_access", AsyncMock(return_value=access)),
            patch.object(chat, "extract_document_text", AsyncMock(side_effect=PdfResourceLimit)),
            patch.object(chat, "refund_access", AsyncMock()) as refund,
            patch.object(chat, "safe_answer", AsyncMock()) as answer,
            patch.object(chat, "ocr_pdf", AsyncMock()) as ocr,
            patch.object(chat, "waiting", waiting),
            patch.object(chat.config, "enable_document_analysis", True),
            patch.object(chat.config, "enable_pdf_ocr", True),
        ):
            await chat._handle_document(message)
        refund.assert_awaited_once_with(service, 5, access)
        ocr.assert_not_awaited()
        self.assertIn("safe decompression limits", answer.call_args.args[1])

    async def test_guest_returns_quota_on_resource_limit(self):
        caller = SimpleNamespace(id=5, username=None, first_name=None, last_name=None)
        service = SimpleNamespace(register_user=AsyncMock())
        document = SimpleNamespace(
            file_name="bomb.pdf", mime_type="application/pdf", file_size=1000, file_id="f",
        )
        access = AccessResult(allowed=True, role="user", consumed=True)
        with (
            patch.object(guest, "_extract_text", return_value=""),
            patch.object(guest, "_voice_of", return_value=None),
            patch.object(guest, "_document_of", return_value=document),
            patch.object(guest, "_download", AsyncMock(return_value=b"pdf")),
            patch.object(guest, "check_access", AsyncMock(return_value=access)),
            patch.object(guest, "extract_document_text", AsyncMock(side_effect=PdfResourceLimit)),
            patch.object(guest, "refund_access", AsyncMock()) as refund,
            patch.object(guest.config, "enable_document_analysis", True),
        ):
            reply = await guest._build_reply(SimpleNamespace(), service, caller, 5)
        refund.assert_awaited_once_with(service, 5, access)
        self.assertIn("safe decompression limits", reply.text)
