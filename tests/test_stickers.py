"""Прозрачность, ограничения обработки и наборы: без внешних API и загрузки моделей."""
from __future__ import annotations

import asyncio
import io
import os
import sys
import threading
import unittest
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

os.environ["BOT_TOKEN"] = "123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghi"
os.environ["AI_API_KEY"] = "test-key"

from aiogram.exceptions import TelegramBadRequest, TelegramRetryAfter
from PIL import Image, ImageDraw
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from bot.config import Config, config
from bot.database.models import Base, StickerPack
from bot.handlers import sticker_sets as handler
from bot.services import sticker_sets as sets
from bot.services import stickers


def png(image: Image.Image) -> bytes:
    """Закодировать тестовое изображение."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def subject() -> Image.Image:
    """Вытянутый объект с белой деталью и полупрозрачным краем."""
    image = Image.new("RGBA", (800, 400))
    draw = ImageDraw.Draw(image)
    draw.rectangle((30, 100, 770, 300), fill=(255, 0, 0, 255))
    draw.rectangle((40, 110, 80, 290), fill=(255, 255, 255, 255))
    draw.rectangle((30, 100, 770, 102), fill=(255, 0, 0, 100))
    return image


class StickerProcessingTests(unittest.IsolatedAsyncioTestCase):
    """Сегментация предшествует масштабированию; альфа-канал сохраняется."""

    def test_full_subject_and_alpha_survive_webp(self):
        data = stickers.to_sticker_webp(png(subject()))
        self.assertLessEqual(len(data), stickers.STICKER_MAX_BYTES)
        image = Image.open(io.BytesIO(data)).convert("RGBA")
        self.assertEqual(image.size, (512, 512))
        alpha = image.getchannel("A")
        left, top, right, bottom = alpha.getbbox()
        self.assertGreater(left, 0)
        self.assertGreater(top, 0)
        self.assertLess(right, 512)
        self.assertLess(bottom, 512)
        self.assertAlmostEqual((right - left) / (bottom - top), 741 / 201, delta=0.15)
        self.assertEqual(alpha.getextrema(), (0, 255))
        self.assertTrue(any(0 < a < 255 for a in alpha.tobytes()))
        # Белая деталь объекта остаётся белой и непрозрачной.
        r, g, b, a = image.getpixel((left + 18, 256))
        self.assertGreater(min(r, g, b), 220)
        self.assertEqual(a, 255)

    def test_empty_invalid_and_oversize_rejected(self):
        self.assertIsNone(stickers.to_sticker_webp(b"invalid"))
        self.assertIsNone(stickers.to_sticker_webp(png(Image.new("RGBA", (20, 20)))))
        with patch.object(stickers, "STICKER_MAX_BYTES", 1):
            self.assertIsNone(stickers.to_sticker_webp(png(subject())))

    def test_rembg_uses_original_resolution_and_reuses_session(self):
        opaque = png(Image.new("RGB", (800, 400), "white"))
        fake = SimpleNamespace(new_session=MagicMock(return_value=object()), remove=MagicMock(return_value=subject()))
        with patch.dict(sys.modules, {"rembg": fake}), patch.object(stickers, "_session", None):
            for _ in range(2):
                self.assertIsNotNone(stickers._prepare_sticker(opaque))
            fake.new_session.assert_called_once_with("u2net", providers=["CPUExecutionProvider"])
            self.assertEqual(fake.remove.call_args.args[0].size, (800, 400))

    def test_native_alpha_skips_rembg(self):
        with patch.dict(sys.modules, {"rembg": None}):
            self.assertIsNotNone(stickers._prepare_sticker(png(subject())))

    def test_failed_background_removal_never_returns_opaque_substitute(self):
        opaque = Image.new("RGBA", (40, 40), "white")
        for output in (opaque, Image.new("RGBA", (40, 40)), Image.new("RGBA", (40, 40), (255, 255, 255, 254))):
            fake = SimpleNamespace(new_session=MagicMock(), remove=MagicMock(return_value=output))
            with patch.dict(sys.modules, {"rembg": fake}), patch.object(stickers, "_session", None):
                self.assertIsNone(stickers._prepare_sticker(png(opaque)))
        with patch.dict(sys.modules, {"rembg": None}):
            self.assertIsNone(stickers._prepare_sticker(png(opaque)))

    async def test_timeout_keeps_worker_busy_without_queue(self):
        started, release = threading.Event(), threading.Event()

        def blocked(data):
            started.set()
            release.wait(5)
            return b"webp"

        with patch.object(stickers, "_pending", None), patch.object(stickers, "_prepare_sticker", blocked), patch.object(config, "sticker_processing_timeout", 0.02):
            try:
                self.assertIsNone(await stickers.prepare_sticker(b"first"))
                self.assertTrue(started.is_set())
                self.assertFalse(stickers._pending.done())
                self.assertIsNone(await stickers.prepare_sticker(b"second"))
            finally:
                release.set()
                await asyncio.shield(stickers._pending)
            self.assertEqual(await stickers.prepare_sticker(b"third"), b"webp")

    async def test_cancellation_does_not_release_running_worker(self):
        started, release = threading.Event(), threading.Event()

        def blocked(data):
            started.set()
            release.wait(5)
            return b"webp"

        with patch.object(stickers, "_pending", None), patch.object(stickers, "_prepare_sticker", blocked):
            task = asyncio.create_task(stickers.prepare_sticker(b"first"))
            try:
                await asyncio.to_thread(started.wait, 2)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                self.assertIsNone(await stickers.prepare_sticker(b"second"))
            finally:
                release.set()
                await asyncio.shield(stickers._pending)

    def test_timeout_config_clamped(self):
        for value, expected in (("1", 10), ("9999", 600), ("120", 120)):
            with patch.dict(os.environ, {"STICKER_PROCESSING_TIMEOUT": value}):
                self.assertEqual(Config().sticker_processing_timeout, expected)


class StickerSetTests(unittest.IsolatedAsyncioTestCase):
    """Настоящая SQLite + замоканный Telegram: права, перезапуск, идемпотентность."""

    async def asyncSetUp(self):
        self.engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        self.factory = async_sessionmaker(self.engine, expire_on_commit=False)
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            # Повторная инициализация схемы безопасна.
            await conn.run_sync(Base.metadata.create_all)
        self.patches = [
            patch.object(sets, "async_session_factory", self.factory),
            patch.object(handler, "get_bot_username", return_value="ExampleBot"),
            patch.object(config, "enable_sticker_generation", True),
        ]
        for p in self.patches:
            p.start()
        self.token = await sets.save_sticker(10, 20, "file", "unique")
        self.access = AsyncMock(return_value=SimpleNamespace(allowed=True))

        @asynccontextmanager
        async def service():
            yield SimpleNamespace()

        self.patches += [patch.object(handler, "user_service", service), patch.object(handler, "check_access", self.access)]
        for p in self.patches[-2:]:
            p.start()

    async def asyncTearDown(self):
        for p in reversed(self.patches):
            p.stop()
        await self.engine.dispose()

    def callback(self, action="menu", number="0", *, user_id=10, bot_id=20):
        """Подготовить Telegram callback без сети."""
        return SimpleNamespace(
            data=f"stk:{action}:{self.token}" + (f":{number}" if number is not None else ""),
            from_user=SimpleNamespace(id=user_id, first_name="Tester"),
            answer=AsyncMock(),
            message=SimpleNamespace(edit_text=AsyncMock(), answer=AsyncMock()),
            bot=SimpleNamespace(
                id=bot_id, get_sticker_set=AsyncMock(return_value=SimpleNamespace(stickers=[])),
                create_new_sticker_set=AsyncMock(), add_sticker_to_set=AsyncMock(),
            ),
        )

    async def pack(self, user_id=10, bot_id=20, name="owned_by_examplebot", created=True):
        async with self.factory() as session:
            pack = StickerPack(user_id=user_id, bot_id=bot_id, name=name, title="Test set", created=created)
            session.add(pack)
            await session.commit()
            return pack

    async def test_owner_and_bot_checked_for_draft(self):
        self.assertIsNotNone(await sets.get_sticker(self.token, 10, 20))
        self.assertIsNone(await sets.get_sticker(self.token, 11, 20))
        self.assertIsNone(await sets.get_sticker(self.token, 10, 21))
        for user_id, bot_id in ((11, 20), (10, 21)):
            cb = self.callback(user_id=user_id, bot_id=bot_id)
            await handler.sticker_set_action(cb)
            cb.message.edit_text.assert_not_awaited()
            cb.bot.create_new_sticker_set.assert_not_awaited()
            self.assertTrue(cb.answer.call_args.kwargs["show_alert"])

    async def test_menu_does_not_consume_or_require_quota(self):
        cb = self.callback()
        await handler.sticker_set_action(cb)
        self.access.assert_awaited_once_with(unittest.mock.ANY, 10, consume=False, require_quota=False)
        cb.message.edit_text.assert_awaited_once()
        keyboard = cb.message.edit_text.call_args.kwargs["reply_markup"]
        self.assertIn("Create new", keyboard.inline_keyboard[0][0].text)

    async def test_blocked_or_disabled_denied(self):
        self.access.return_value.allowed = False
        cb = self.callback("new", None)
        await handler.sticker_set_action(cb)
        cb.bot.create_new_sticker_set.assert_not_awaited()
        self.assertTrue(cb.answer.call_args.kwargs["show_alert"])
        with patch.object(config, "enable_sticker_generation", False):
            cb = self.callback()
            await handler.sticker_set_action(cb)
            cb.message.edit_text.assert_not_awaited()

    async def test_malformed_callbacks_rejected(self):
        for data in ("stk:new:bad", f"stk:new:{self.token}:1", f"stk:add:{self.token}", f"stk:menu:{self.token}:²", f"stk:menu:{self.token}:" + "9" * 100):
            cb = self.callback()
            cb.data = data
            await handler.sticker_set_action(cb)
            cb.message.edit_text.assert_not_awaited()
            cb.bot.create_new_sticker_set.assert_not_awaited()

    async def test_create_set_and_persist(self):
        cb = self.callback("new", None)
        cb.bot.get_sticker_set.side_effect = TelegramBadRequest(method=MagicMock(), message="STICKERSET_INVALID")
        await handler.sticker_set_action(cb)
        cb.bot.create_new_sticker_set.assert_awaited_once()
        args = cb.bot.create_new_sticker_set.call_args.kwargs
        self.assertEqual(args["user_id"], 10)
        self.assertTrue(args["name"].endswith("_by_examplebot"))
        self.assertLessEqual(len(args["name"]), 64)
        self.assertEqual(args["stickers"][0].sticker, "file")
        self.assertEqual(args["stickers"][0].emoji_list, ["🙂"])
        # Новая сессия читает сохранённый набор, как после перезапуска.
        packs = await sets.list_packs(10, 20, 0)
        self.assertEqual(len(packs), 1)
        self.assertEqual(packs[0].name, args["name"])
        self.assertIn(args["name"], cb.message.edit_text.call_args.kwargs["reply_markup"].inline_keyboard[0][0].url)

    async def test_repeat_create_recovers_reserved_name(self):
        cb = self.callback("new", None)
        cb.bot.get_sticker_set.return_value.stickers = [SimpleNamespace(file_unique_id="unique")]
        await handler.sticker_set_action(cb)
        await handler.sticker_set_action(cb)
        self.assertEqual(len(await sets.list_packs(10, 20, 0)), 1)
        cb.bot.create_new_sticker_set.assert_not_awaited()
        cb.bot.add_sticker_to_set.assert_not_awaited()

    async def test_add_and_duplicate_guard(self):
        pack = await self.pack()
        cb = self.callback("add", str(pack.id))
        await handler.sticker_set_action(cb)
        cb.bot.add_sticker_to_set.assert_awaited_once()
        cb.bot.get_sticker_set.return_value.stickers = [SimpleNamespace(file_unique_id="unique")]
        await handler.sticker_set_action(cb)
        cb.bot.add_sticker_to_set.assert_awaited_once()

    async def test_foreign_and_unconfirmed_sets_never_mutated(self):
        for user_id, bot_id, created in ((11, 20, True), (10, 21, True), (10, 20, False)):
            pack = await self.pack(user_id, bot_id, f"pack_{user_id}_{bot_id}_{created}", created)
            cb = self.callback("add", str(pack.id))
            await handler.sticker_set_action(cb)
            cb.bot.get_sticker_set.assert_not_awaited()
            cb.bot.add_sticker_to_set.assert_not_awaited()

    async def test_pagination_scoped_and_callback_size(self):
        for i in range(9):
            await self.pack(name=f"pack_{i}")
        await self.pack(user_id=11, name="foreign")
        first = await sets.list_packs(10, 20, 0)
        second = await sets.list_packs(10, 20, 1)
        self.assertEqual(len(first), 7)
        self.assertEqual(len(second), 3)
        for page, rows in ((0, first), (1, second)):
            keyboard = sets.packs_keyboard(self.token, rows, page)
            for row in keyboard.inline_keyboard:
                for button in row:
                    self.assertLessEqual(len(button.callback_data.encode()), 64)
            self.assertEqual(keyboard.inline_keyboard[-1][0].text, "▶" if page == 0 else "◀")

    async def test_rate_limit_and_full_set_reported(self):
        pack = await self.pack()
        for error in (TelegramRetryAfter(method=MagicMock(), message="wait", retry_after=30), TelegramBadRequest(method=MagicMock(), message="STICKERS_TOO_MUCH")):
            cb = self.callback("add", str(pack.id))
            cb.bot.add_sticker_to_set.side_effect = error
            await handler.sticker_set_action(cb)
            cb.message.answer.assert_awaited_once()

    async def test_busy_slot_rejects_second_operation(self):
        cb = self.callback("new", None)
        with handler._set_slot(10, 20):
            await handler.sticker_set_action(cb)
        cb.bot.create_new_sticker_set.assert_not_awaited()
        self.assertTrue(cb.answer.call_args.kwargs["show_alert"])


class StickerDeliveryTests(unittest.IsolatedAsyncioTestCase):
    """Доставка и квота: кнопка необязательна для успеха уже отправленного стикера."""

    async def test_sent_sticker_gets_persistent_button_without_timing(self):
        from bot.handlers import chat
        from bot.services.ai_client import ImageResult

        @asynccontextmanager
        async def no_wait(message):
            yield

        sent = SimpleNamespace(sticker=SimpleNamespace(file_id="file", file_unique_id="unique"))
        message = SimpleNamespace(
            chat=SimpleNamespace(id=-100), from_user=SimpleNamespace(id=10),
            bot=SimpleNamespace(id=20, send_chat_action=AsyncMock()),
            answer_sticker=AsyncMock(return_value=sent),
        )
        for failure in (False, True):
            with (
                patch.object(chat, "waiting", no_wait),
                patch.object(chat.throttle, "wait", AsyncMock()),
                patch.object(chat.ai_client, "generate_sticker", AsyncMock(return_value=ImageResult(data=b"webp"))),
                patch.object(chat, "format_timing", return_value=""),
                patch.object(chat, "save_sticker", AsyncMock(return_value="abcdef", side_effect=RuntimeError("db") if failure else None)) as save,
                patch.object(chat, "safe_answer", AsyncMock()) as answer,
            ):
                self.assertTrue(await chat.generate_and_send_sticker(message, "cat", user_id=10))
                save.assert_awaited_once_with(10, 20, "file", "unique")
                if not failure:
                    keyboard = answer.call_args.kwargs["reply_markup"]
                    self.assertEqual(keyboard.inline_keyboard[0][0].callback_data, "stk:menu:abcdef:0")

    async def test_failed_processing_refunds_generation_quota(self):
        from bot.handlers import image

        service = SimpleNamespace(register_user=AsyncMock(), get_user_settings=AsyncMock(return_value={"model": "m"}))

        @asynccontextmanager
        async def service_context():
            yield service

        message = SimpleNamespace(answer=AsyncMock())
        user = SimpleNamespace(id=10, username=None, first_name="Test", last_name=None)
        access = SimpleNamespace(allowed=True)
        with (
            patch.object(image, "user_service", service_context),
            patch.object(image, "check_access", AsyncMock(return_value=access)),
            patch.object(image, "refund_access", AsyncMock()) as refund,
            patch.object(image, "guest_hint", return_value=None),
        ):
            await image._generate(message, user, "cat", AsyncMock(return_value=False))
            refund.assert_awaited_once_with(service, 10, access)

    async def test_failed_sticker_tool_refunds_even_with_introductory_text(self):
        from bot.handlers import chat
        from bot.services.access import AccessResult
        from bot.services.ai_client import ChatResult, ToolCall

        @asynccontextmanager
        async def no_wait(message):
            yield SimpleNamespace()

        message = SimpleNamespace(
            chat=SimpleNamespace(id=10, type="private"),
            from_user=SimpleNamespace(id=10), answer=AsyncMock(),
        )
        service = SimpleNamespace(
            get_user_settings=AsyncMock(return_value={
                "model": "m", "system_prompt": "sys", "temperature": 0.5, "max_tokens": 100,
            }), get_history=AsyncMock(return_value=[]), add_message=AsyncMock(),
        )
        access = AccessResult(allowed=True, role="allowed", consumed=True)
        with (
            patch.object(config, "stream_responses", False),
            patch.object(config, "enable_sticker_generation", True),
            patch.object(chat, "search_configured", return_value=False),
            patch.object(chat, "mcp_available", return_value=False),
            patch.object(chat, "waiting", no_wait),
            patch.object(chat, "_single_shot", AsyncMock(return_value=ChatResult(
                content="Here is your sticker!", tool_calls=[ToolCall(id="c", name="generate_sticker", arguments={"prompt": "cat"})],
            ))),
            patch.object(chat, "deliver_answer", AsyncMock()),
            patch.object(chat, "_retry_markup", return_value=None),
            patch.object(chat, "generate_and_send_sticker", AsyncMock(return_value=False)),
            patch.object(chat, "refund_access", AsyncMock()) as refund,
        ):
            await chat._answer_with_ai(message, service, "make a sticker", access)
            refund.assert_awaited_once_with(service, 10, access)
