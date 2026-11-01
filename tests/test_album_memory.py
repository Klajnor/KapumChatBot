"""H-3: резерв RAM альбомов до загрузки и до конца обработки."""

from __future__ import annotations

import asyncio
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from bot.handlers import chat


class AlbumMemoryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        for name in ("_album_images", "_album_total_bytes", "_album_download_bytes"):
            stack.enter_context(patch.object(chat, name, {}))
        stack.enter_context(patch.object(chat, "_album_lock", asyncio.Lock()))
        stack.enter_context(patch.object(chat, "_ALBUM_COLLECT_SECONDS", 0))
        stack.enter_context(patch.object(chat, "_ALBUM_MAX_TOTAL_BYTES", 100))
        stack.enter_context(patch.object(chat.config, "max_photo_bytes", 10))
        stack.enter_context(patch.object(chat.config, "max_album_bytes", 30))
        self.answer = stack.enter_context(patch.object(chat, "safe_answer", AsyncMock()))
        stack.enter_context(
            patch.object(
                chat, "_album_access_allowed", AsyncMock(return_value=(True, None))
            )
        )
        self.guarded = stack.enter_context(patch.object(chat, "_guarded", AsyncMock()))
        self.bot = SimpleNamespace(
            get_file=AsyncMock(return_value=SimpleNamespace(file_path="photo")),
            download_file=AsyncMock(side_effect=self._download),
        )

    async def _download(self, path, *, destination):
        destination.write(b"data")
        return destination

    def message(self, *, size=4, group="g", user=1, file_id="f"):
        return SimpleNamespace(
            media_group_id=group, chat=SimpleNamespace(id=user),
            from_user=SimpleNamespace(id=user), caption=None,
            photo=[SimpleNamespace(file_id=file_id, file_size=size)], bot=self.bot,
        )

    def assert_released(self):
        self.assertEqual(chat._album_bytes_in_flight(), 0)
        self.assertEqual(chat._album_download_bytes, {})
        self.assertEqual(chat._album_images, {})

    async def test_parallel_downloads_reserve_global_budget_before_network(self):
        release = asyncio.Event()
        started = asyncio.Event()
        calls = 0

        async def download(path, *, destination):
            nonlocal calls
            calls += 1
            if calls == 2:
                started.set()
            await release.wait()
            destination.write(b"data")

        self.bot.download_file.side_effect = download
        with patch.object(chat, "_ALBUM_MAX_TOTAL_BYTES", 20):
            tasks = [asyncio.create_task(chat._handle_album_photo(
                self.message(group=str(i), user=i + 1)
            )) for i in range(8)]
            try:
                await asyncio.wait_for(started.wait(), 2)
                await asyncio.sleep(0)
                self.assertEqual(calls, 2)
                self.assertEqual(chat._album_bytes_in_flight(), 16)
            finally:
                release.set()
                await asyncio.gather(*tasks)
        self.assert_released()

    async def test_parallel_downloads_reserve_per_album_budget(self):
        release = asyncio.Event()
        started = asyncio.Event()

        async def download(path, *, destination):
            started.set()
            await release.wait()
            destination.write(b"data")

        self.bot.download_file.side_effect = download
        with patch.object(chat.config, "max_album_bytes", 6):
            first = asyncio.create_task(chat._handle_album_photo(self.message()))
            try:
                await asyncio.wait_for(started.wait(), 2)
                await chat._handle_album_photo(self.message(file_id="second"))
                self.assertEqual(self.bot.download_file.await_count, 1)
                self.assertEqual(chat._album_bytes_in_flight(), 8)
            finally:
                release.set()
                await first
        self.assert_released()

    async def test_global_budget_also_applies_to_existing_album(self):
        release = asyncio.Event()
        started = asyncio.Event()

        async def download(path, *, destination):
            if self.bot.download_file.await_count == 2:
                started.set()
            await release.wait()
            destination.write(b"data")

        self.bot.download_file.side_effect = download
        with patch.object(chat, "_ALBUM_MAX_TOTAL_BYTES", 20):
            tasks = [asyncio.create_task(chat._handle_album_photo(self.message())) for _ in range(2)]
            try:
                await asyncio.wait_for(started.wait(), 2)
                await chat._handle_album_photo(self.message(file_id="third"))
                self.assertEqual(self.bot.download_file.await_count, 2)
                self.assertEqual(chat._album_bytes_in_flight(), 16)
            finally:
                release.set()
                await asyncio.gather(*tasks)
        self.assert_released()

    async def test_photo_count_limit_includes_pending_downloads(self):
        release = asyncio.Event()
        started = asyncio.Event()

        async def download(path, *, destination):
            started.set()
            await release.wait()
            destination.write(b"data")

        self.bot.download_file.side_effect = download
        with patch.object(chat, "_ALBUM_MAX_PHOTOS", 1):
            first = asyncio.create_task(chat._handle_album_photo(self.message()))
            try:
                await asyncio.wait_for(started.wait(), 2)
                await chat._handle_album_photo(self.message(file_id="second"))
                self.assertEqual(self.bot.download_file.await_count, 1)
            finally:
                release.set()
                await first
        self.assert_released()

    async def test_unknown_size_reserves_maximum_then_shrinks_to_actual(self):
        observed = []

        async def download(path, *, destination):
            observed.append(chat._album_bytes_in_flight())
            destination.write(b"data")

        async def process(*args):
            observed.append(chat._album_bytes_in_flight())

        self.bot.download_file.side_effect = download
        self.guarded.side_effect = process
        await chat._handle_album_photo(self.message(size=None))
        self.assertEqual(observed, [20, 4])
        self.assert_released()

    async def test_understated_size_is_rejected_during_write(self):
        buffers = []

        async def download(path, *, destination):
            buffers.append(destination)
            destination.write(b"ab")
            destination.write(b"cde")

        self.bot.download_file.side_effect = download
        await chat._handle_album_photo(self.message(size=4))
        self.assertTrue(buffers[0].closed)
        self.guarded.assert_not_awaited()
        self.assertIn("reserved limit", self.answer.call_args.args[1])
        self.assert_released()

    async def test_buffer_never_accepts_bytes_over_reserved_limit(self):
        with chat._AlbumBuffer(4) as buffer:
            buffer.write(b"ab")
            with self.assertRaises(chat._AlbumSizeExceeded):
                buffer.write(b"cde")
            self.assertEqual(buffer.getvalue(), b"ab")

    async def test_download_failure_releases_reservation(self):
        self.bot.download_file.side_effect = OSError("download failed")
        await chat._handle_album_photo(self.message())
        self.guarded.assert_not_awaited()
        self.assert_released()
        self.bot.download_file.side_effect = self._download
        await chat._handle_album_photo(self.message())
        self.guarded.assert_awaited_once()
        self.assert_released()

    async def test_get_file_failure_releases_reservation(self):
        self.bot.get_file.side_effect = TimeoutError
        await chat._handle_album_photo(self.message())
        self.bot.download_file.assert_not_awaited()
        self.assert_released()

    async def test_cancellation_closes_buffer_before_releasing_reservation(self):
        started = asyncio.Event()
        buffers = []

        async def download(path, *, destination):
            buffers.append(destination)
            destination.write(b"ab")
            started.set()
            await asyncio.Event().wait()

        self.bot.download_file.side_effect = download
        task = asyncio.create_task(chat._handle_album_photo(self.message()))
        await asyncio.wait_for(started.wait(), 2)
        self.assertEqual(chat._album_bytes_in_flight(), 8)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(buffers[0].closed)
        self.assert_released()

    async def test_album_not_processed_while_another_download_is_pending(self):
        release = asyncio.Event()
        started = asyncio.Event()
        seen_sizes = []

        async def download(path, *, destination):
            if self.bot.download_file.await_count == 1:
                started.set()
                await release.wait()
            destination.write(b"data")

        async def process(message, inner):
            seen_sizes.append(chat._album_bytes_in_flight())

        self.bot.download_file.side_effect = download
        self.guarded.side_effect = process
        first = asyncio.create_task(chat._handle_album_photo(self.message()))
        try:
            await asyncio.wait_for(started.wait(), 2)
            await chat._handle_album_photo(self.message(file_id="second"))
            self.guarded.assert_not_awaited()
            self.assertEqual(chat._album_bytes_in_flight(), 12)
        finally:
            release.set()
            await first
        self.assertEqual(seen_sizes, [8])
        self.assert_released()

    async def test_active_processing_stays_in_budget_and_rejects_late_photos(self):
        started = asyncio.Event()
        release = asyncio.Event()

        async def process(*args):
            started.set()
            await release.wait()

        self.guarded.side_effect = process
        with patch.object(chat, "_ALBUM_MAX_TOTAL_BYTES", 10):
            first = asyncio.create_task(chat._handle_album_photo(self.message()))
            try:
                await asyncio.wait_for(started.wait(), 2)
                self.assertEqual(chat._album_bytes_in_flight(), 4)
                await chat._handle_album_photo(self.message(file_id="late"))
                await chat._handle_album_photo(self.message(group="other", user=2))
                self.assertEqual(self.bot.download_file.await_count, 1)
            finally:
                release.set()
                await first
        self.assert_released()

    async def test_processing_error_releases_buffers(self):
        self.guarded.side_effect = RuntimeError("processing failed")
        with self.assertRaises(RuntimeError):
            await chat._handle_album_photo(self.message())
        self.assert_released()

    async def test_valid_images_still_reach_album_processor(self):
        received = []

        async def guarded(message, inner):
            await inner(message)

        async def process(message, images):
            received.extend(images)
            self.assertEqual(chat._album_bytes_in_flight(), 4)

        self.guarded.side_effect = guarded
        with patch.object(chat, "_process_album", AsyncMock(side_effect=process)):
            await chat._handle_album_photo(self.message())
        self.assertEqual(received, [("f", b"data")])
        self.assert_released()

    async def test_processing_cancellation_releases_buffers(self):
        started = asyncio.Event()

        async def process(*args):
            started.set()
            await asyncio.Event().wait()

        self.guarded.side_effect = process
        task = asyncio.create_task(chat._handle_album_photo(self.message()))
        await asyncio.wait_for(started.wait(), 2)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assert_released()

    async def test_cleanup_does_not_release_active_download_or_processing(self):
        chat._album_images.update({
            "download": {"last": 0, "pending": 1},
            "process": {"last": 0, "processing": True},
            "stale": {"last": 0},
        })
        chat._album_total_bytes.update(dict.fromkeys(chat._album_images, 4))
        chat._album_download_bytes["download"] = 4
        with (
            patch.object(chat.asyncio, "sleep", AsyncMock(side_effect=[None, asyncio.CancelledError])),
            self.assertRaises(asyncio.CancelledError),
        ):
            await chat._album_cleanup()
        self.assertEqual(set(chat._album_images), {"download", "process"})
        self.assertEqual(chat._album_bytes_in_flight(), 12)
