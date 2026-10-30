"""Регрессии жизненного цикла парсеров и процессов конвертации аудио."""

from __future__ import annotations

import asyncio
import sys
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from bot.services import ai_client, documents, pdf_ocr


class ParserLifetimeTests(unittest.IsolatedAsyncioTestCase):
    async def _check_busy_after_abandonment(self, *, cancel: bool) -> None:
        release = threading.Event()
        started = [threading.Event(), threading.Event()]
        slots = threading.BoundedSemaphore(2)
        tasks = []

        def blocked(index, deadline=None):
            started[index].set()
            release.wait(10)
            raise ValueError("late parser failure")

        with (
            ThreadPoolExecutor(max_workers=2) as pool,
            patch.object(documents, "_parse_pool", pool),
            patch.object(documents, "_parse_slots", slots),
            patch.object(documents.config, "document_parse_timeout", 0.01),
        ):
            try:
                tasks = [asyncio.create_task(documents._run_parser(blocked, i)) for i in range(2)]
                for event in started:
                    self.assertTrue(await asyncio.to_thread(event.wait, 2))
                if cancel:
                    for task in tasks:
                        task.cancel()
                results = await asyncio.gather(*tasks, return_exceptions=True)
                expected = asyncio.CancelledError if cancel else documents.DocumentParseTimeout
                self.assertTrue(all(isinstance(result, expected) for result in results))
                probe = Mock(return_value="queued")
                for _ in range(10):
                    with self.assertRaises(documents.DocumentParseTimeout):
                        await documents._run_parser(probe, b"another document")
                probe.assert_not_called()
            finally:
                release.set()
                await asyncio.gather(*tasks, return_exceptions=True)
        # Оба реальных потока завершились: слоты должны стать доступны снова.
        self.assertTrue(slots.acquire(blocking=False))
        self.assertTrue(slots.acquire(blocking=False))
        self.assertFalse(slots.acquire(blocking=False))

    async def test_timeout_does_not_allow_more_documents_into_busy_pool(self):
        await self._check_busy_after_abandonment(cancel=False)

    async def test_cancellation_does_not_release_running_parser_slots(self):
        await self._check_busy_after_abandonment(cancel=True)


class AudioProcessTests(unittest.IsolatedAsyncioTestCase):
    async def _run_process(self, script: str, *, cancel: bool = False, timeout: float = 5):
        """Реальный дочерний Python имитирует ffmpeg, платных API и сети нет."""
        spawn = asyncio.create_subprocess_exec
        started = asyncio.Event()
        processes = []
        commands = []

        async def replacement(*args, **kwargs):
            commands.append(args)
            process = await spawn(sys.executable, "-c", script, **kwargs)
            processes.append(process)
            started.set()
            return process

        with (
            patch.object(ai_client.asyncio, "create_subprocess_exec", replacement),
            patch.object(ai_client, "_TRANSCODE_TIMEOUT", timeout),
        ):
            task = asyncio.create_task(ai_client._transcode_to_pcm16(b"input"))
            await asyncio.wait_for(started.wait(), 5)
            if cancel:
                # Дать корутине начать обмен через pipe после создания процесса.
                await asyncio.sleep(0)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(task, 5)
                result = None
            else:
                result = await asyncio.wait_for(task, 10)
        self.assertIsNotNone(processes[0].returncode, "child process was leaked")
        return result, commands[0]

    async def test_normal_conversion_and_network_file_restrictions(self):
        result, command = await self._run_process(
            "import sys; sys.stdin.buffer.read(); sys.stdout.buffer.write(b'pcm')"
        )
        self.assertEqual(result, b"pcm")
        self.assertEqual(command[command.index("-protocol_whitelist") + 1], "pipe")
        self.assertLess(command.index("-protocol_whitelist"), command.index("-i"))
        formats = command[command.index("-format_whitelist") + 1].split(",")
        self.assertNotIn("hls", formats)
        self.assertNotIn("concat", formats)
        self.assertEqual(command[command.index("-t") + 1], "300")

    async def test_timeout_kills_and_reaps_process(self):
        result, _ = await self._run_process("import time; time.sleep(30)", timeout=0.05)
        self.assertIsNone(result)

    async def test_cancel_kills_and_reaps_process(self):
        await self._run_process("import time; time.sleep(30)", cancel=True)

    async def test_oversized_stdout_and_stderr_kill_process(self):
        for stream in ("stdout", "stderr"):
            with (
                patch.object(ai_client, "_MAX_TRANSCODE_BYTES", 1024),
                patch.object(ai_client, "_MAX_TRANSCODE_ERRORS", 1024),
            ):
                result, _ = await self._run_process(
                    f"import sys,time; sys.{stream}.buffer.write(b'x'*131072); "
                    f"sys.{stream}.flush(); time.sleep(30)"
                )
            self.assertIsNone(result)


class OcrResourceTests(unittest.IsolatedAsyncioTestCase):
    async def test_giant_pdf_page_is_bounded_before_rasterization(self):
        import pdfplumber
        from PIL import Image

        # Валидный пустой PDF 2000 x 2000 дюймов: при 150 DPI это
        # 90 миллиардов пикселей. Предохранитель не даст старому коду OOM.
        objects = [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            (
                b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 144000 144000] "
                b"/Resources << >> /Contents 4 0 R >>"
            ),
            b"<< /Length 0 >>\nstream\n\nendstream",
        ]
        data = b"%PDF-1.4\n"
        offsets = []
        for index, obj in enumerate(objects, 1):
            offsets.append(len(data))
            data += f"{index} 0 obj\n".encode() + obj + b"\nendobj\n"
        xref = len(data)
        data += b"xref\n0 5\n0000000000 65535 f \n"
        data += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets)
        data += f"trailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
        render = pdfplumber.page.Page.to_image
        dimensions = []

        def guarded_render(page, *, resolution):
            side = max(page.width, page.height) * resolution / 72
            dimensions.append(side)
            if side > 1601:
                raise AssertionError("Oversized bitmap requested before thumbnail")
            return render(page, resolution=resolution)

        with patch.object(pdfplumber.page.Page, "to_image", guarded_render):
            images, total = await documents._run_parser(pdf_ocr._render_pages, data, 1)
        self.assertEqual(total, 1)
        self.assertEqual(len(images), 1)
        self.assertLessEqual(max(dimensions), 1601)
        with Image.open(BytesIO(images[0])) as image:
            self.assertLessEqual(max(image.size), 1600)

    async def test_ocr_does_not_call_paid_api_when_rendering_times_out(self):
        client = SimpleNamespace(chat_with_image_result=AsyncMock())
        with (
            patch.object(pdf_ocr, "_run_parser", AsyncMock(side_effect=documents.DocumentParseTimeout)),
            patch.object(pdf_ocr, "ai_client", client),
            self.assertRaises(documents.DocumentParseTimeout),
        ):
            await pdf_ocr.ocr_pdf(b"pdf")
        client.chat_with_image_result.assert_not_awaited()
