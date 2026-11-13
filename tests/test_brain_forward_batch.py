"""Пересланная владельцу пачка: один вопрос Analyze/Remember и один анализ."""

from __future__ import annotations

import asyncio
import time
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from aiogram.dispatcher.event.bases import SkipHandler

from bot.handlers import brain, chat


def forwarded(
    *,
    text: str | None = None,
    photo: str | None = None,
    document: str | None = None,
    message_id: int = 1,
    chat_id: int = 1,
    user_id: int = 1,
):
    return SimpleNamespace(
        text=text,
        caption=None,
        rich_message=None,
        quote=None,
        photo=[SimpleNamespace(file_id=photo, file_size=10)] if photo else None,
        document=SimpleNamespace(file_id=document) if document else None,
        voice=None,
        chat=SimpleNamespace(id=chat_id, type="private"),
        from_user=SimpleNamespace(id=user_id, first_name="Owner"),
        message_id=message_id,
        media_group_id=None,
        forward_origin=SimpleNamespace(
            type="user", sender_user=SimpleNamespace(id=2, first_name="Alice")
        ),
        forward_date=None,
        reply_to_message=None,
        external_reply=None,
        answer=AsyncMock(),
        bot=SimpleNamespace(send_chat_action=AsyncMock()),
    )


class _FakeMessage:
    """Подмена aiogram Message для isinstance в callback-обработчике."""


def callback(data: str, *, user_id: int = 1):
    message = _FakeMessage()
    message.chat = SimpleNamespace(id=user_id, type="private")
    message.message_id = 100
    message.answer = AsyncMock()
    return SimpleNamespace(
        data=data,
        from_user=SimpleNamespace(id=user_id),
        message=message,
    )


@contextmanager
def granted_slot(user_id: int):
    yield True


async def _wait_for_prompt(key: str) -> None:
    """Дождаться фонового вопроса владельцу, если он ещё не задан."""
    await asyncio.sleep(0)
    entry = brain._forward_batches.get(key)
    if entry is not None and entry.get("task") is not None:
        await entry["task"]


class OwnerForwardBatchTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        brain._pending.clear()
        brain._forward_batches.clear()
        for name, value in (
            ("_forward_batch_lock", asyncio.Lock()),
        ):
            item = patch.object(brain, name, value)
            item.start()
            self.addCleanup(item.stop)
        merge = patch.object(chat, "_FORWARD_TEXT_MERGE_SECONDS", 0)
        merge.start()
        self.addCleanup(merge.stop)
        owner = patch.object(brain, "_owner_dm", return_value=True)
        owner.start()
        self.addCleanup(owner.stop)

    async def asyncTearDown(self):
        tasks = [
            entry["task"]
            for entry in brain._forward_batches.values()
            if entry.get("task") is not None
        ]
        for task in tasks:
            task.cancel()
        for task in tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
        brain._forward_batches.clear()
        brain._pending.clear()

    async def test_batch_prompts_owner_once(self):
        parts = [
            forwarded(photo="f1", message_id=1),
            forwarded(text="Post body", message_id=2),
            forwarded(photo="f2", message_id=3),
        ]

        with patch.object(brain, "safe_answer", AsyncMock()) as answer:
            for part in parts:
                await brain.choose_forward(part)
            key = brain._forward_batch_key(parts[0])
            await _wait_for_prompt(key)

        answer.assert_awaited_once()
        self.assertIn("What would you like to do?", answer.await_args.args[1])
        self.assertEqual(len(brain._pending), 1)
        pending = next(iter(brain._pending.values()))
        self.assertEqual(pending.parts, parts)

    async def test_single_message_still_prompts(self):
        part = forwarded(text="One message")

        with patch.object(brain, "safe_answer", AsyncMock()) as answer:
            await brain.choose_forward(part)
            key = brain._forward_batch_key(part)
            await _wait_for_prompt(key)

        answer.assert_awaited_once()
        pending = next(iter(brain._pending.values()))
        self.assertEqual(pending.parts, [part])

    async def test_non_owner_forward_is_skipped(self):
        with patch.object(brain, "_owner_dm", return_value=False):
            with self.assertRaises(SkipHandler):
                await brain.choose_forward(forwarded(text="Not mine"))

    async def test_analyze_dispatches_every_part_once(self):
        first = forwarded(photo="f1")
        second = forwarded(text="Post body")
        third = forwarded(document="doc")
        token = "batch"
        brain._pending[token] = brain.PendingForward(
            [first, second, third], time.monotonic() + 60
        )

        with (
            patch.object(brain, "Message", _FakeMessage),
            patch.object(brain, "safe_answer_callback", AsyncMock()),
            patch.object(brain, "safe_answer", AsyncMock()),
            patch.object(brain.limits, "slot", granted_slot),
            patch.object(chat, "handle_text", AsyncMock()) as text_handler,
            patch.object(chat, "handle_photo", AsyncMock()) as photo_handler,
            patch.object(chat, "handle_document", AsyncMock()) as doc_handler,
            patch.object(chat, "handle_voice", AsyncMock()) as voice_handler,
        ):
            await brain.brain_callback(callback(f"brain:analyze:{token}"))

        photo_handler.assert_awaited_once_with(first)
        text_handler.assert_awaited_once_with(second)
        doc_handler.assert_awaited_once_with(third)
        voice_handler.assert_not_awaited()
        self.assertNotIn(token, brain._pending)

    async def test_remember_saves_batch_as_one_note(self):
        parts = [
            forwarded(text="First line", message_id=1),
            forwarded(text="Second line", message_id=2),
        ]
        token = "note"
        brain._pending[token] = brain.PendingForward(
            parts, time.monotonic() + 60
        )

        with (
            patch.object(brain, "Message", _FakeMessage),
            patch.object(brain, "safe_answer_callback", AsyncMock()),
            patch.object(brain, "safe_answer", AsyncMock()),
            patch.object(brain.limits, "slot", granted_slot),
            patch.object(brain, "_save", AsyncMock(return_value=True)) as save,
        ):
            await brain.brain_callback(callback(f"brain:forward:{token}"))

        save.assert_awaited_once()
        self.assertEqual(save.await_args.args[2], "First line\n\nSecond line")
        self.assertEqual(save.await_args.args[3], "tg:1:1")
        self.assertNotIn(token, brain._pending)

    async def test_unsupported_parts_do_not_duplicate_the_prompt(self):
        unsupported = forwarded(text=None)
        token = "mixed"
        brain._pending[token] = brain.PendingForward(
            [unsupported], time.monotonic() + 60
        )

        with (
            patch.object(brain, "Message", _FakeMessage),
            patch.object(brain, "safe_answer_callback", AsyncMock()),
            patch.object(brain.limits, "slot", granted_slot),
            patch.object(brain, "safe_answer", AsyncMock()) as answer,
            patch.object(chat, "handle_text", AsyncMock()),
            patch.object(chat, "handle_photo", AsyncMock()),
            patch.object(chat, "handle_document", AsyncMock()),
            patch.object(chat, "handle_voice", AsyncMock()),
        ):
            await brain.brain_callback(callback(f"brain:analyze:{token}"))

        answer.assert_awaited_once()
        self.assertIn("cannot be analyzed", answer.await_args.args[1])


if __name__ == "__main__":
    unittest.main()