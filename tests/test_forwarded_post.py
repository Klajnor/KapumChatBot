"""Пересланный пост канала: текст и все картинки уходят в один анализ."""

from __future__ import annotations

import asyncio
import unittest
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

from bot.handlers import chat
from bot.services.access import AccessResult
from bot.services.ai_client import ChatResult
from bot.services.untrusted import has_untrusted_content

ALLOWED = AccessResult(True, "user")


@asynccontextmanager
async def no_wait(*args, **kwargs):
    yield SimpleNamespace()


def channel_origin():
    return SimpleNamespace(
        type="channel",
        sender_chat=SimpleNamespace(id=42, title="Channel"),
    )


def album_message(*, caption: str = "", forwarded: bool = True):
    return SimpleNamespace(
        from_user=SimpleNamespace(id=1, first_name="Me"),
        chat=SimpleNamespace(id=1, type="private"),
        caption=caption,
        text=None,
        rich_message=None,
        quote=None,
        media_group_id="grp",
        forward_date=None,
        forward_origin=channel_origin() if forwarded else None,
        reply_to_message=None,
        external_reply=None,
        answer=AsyncMock(),
        bot=SimpleNamespace(send_chat_action=AsyncMock()),
    )


def post_text(text: str = "Long post body"):
    """Текст поста, пришедший отдельным сообщением (без медиагруппы)."""
    message = album_message()
    message.caption = None
    message.media_group_id = None
    message.text = text
    return message


def service():
    return SimpleNamespace(
        get_user_settings=AsyncMock(
            return_value={
                "model": "vision",
                "system_prompt": "sys",
                "temperature": 0.5,
                "max_tokens": 100,
            }
        ),
        get_history=AsyncMock(return_value=[]),
        add_message=AsyncMock(return_value=SimpleNamespace(id=1)),
        consume_photo_intent_budget=AsyncMock(return_value=True),
        session=SimpleNamespace(close=AsyncMock()),
    )


def collecting_album(message, *, channel_post: bool = True):
    return {
        "images": [("f1", b"image")],
        "message": message,
        "last": 0.0,
        "user_id": message.from_user.id,
        "pending": 0,
        "channel_post": channel_post,
    }


def personal_origin(author: str = "Alice"):
    return SimpleNamespace(
        type="user", sender_user=SimpleNamespace(id=2, first_name=author)
    )


def forwarded_text(text: str):
    message = album_message(forwarded=False)
    message.caption = None
    message.media_group_id = None
    message.text = text
    message.forward_origin = personal_origin()
    return message


def forwarded_photo(file_id: str = "f1", *, caption: str = ""):
    message = album_message(caption=caption)
    message.media_group_id = None
    message.photo = [SimpleNamespace(file_id=file_id, file_size=10)]
    message.forward_origin = personal_origin()
    return message


class ForwardedPostMergeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        chat._album_images.clear()
        chat._album_total_bytes.clear()
        chat._album_download_bytes.clear()
        lock = patch.object(chat, "_album_lock", asyncio.Lock())
        lock.start()
        self.addCleanup(lock.stop)
        merge = patch.object(chat, "_FORWARD_TEXT_MERGE_SECONDS", 0)
        merge.start()
        self.addCleanup(merge.stop)

    def tearDown(self):
        chat._album_images.clear()
        chat._album_total_bytes.clear()
        chat._album_download_bytes.clear()

    async def test_post_text_joins_collecting_channel_album(self):
        message = post_text()
        key = chat._album_key(message.chat.id, "grp")
        chat._album_images[key] = collecting_album(album_message())

        absorbed = await chat._absorb_forwarded_post_text(message, "Long post body")

        self.assertTrue(absorbed)
        self.assertEqual(chat._album_images[key]["extra_text"], ["Long post body"])

    async def test_own_album_does_not_absorb_post_text(self):
        message = post_text()
        key = chat._album_key(message.chat.id, "grp")
        chat._album_images[key] = collecting_album(
            album_message(forwarded=False), channel_post=False
        )

        absorbed = await chat._absorb_forwarded_post_text(message, "Long post body")

        self.assertFalse(absorbed)
        self.assertNotIn("extra_text", chat._album_images[key])

    async def test_forwarded_user_message_is_not_a_channel_post(self):
        message = post_text()
        message.forward_origin = SimpleNamespace(
            type="user", sender_user=SimpleNamespace(id=2, first_name="Alice")
        )

        self.assertFalse(chat._is_forwarded_post(message))

    async def test_handle_text_does_not_answer_forwarded_post_twice(self):
        message = post_text()
        key = chat._album_key(message.chat.id, "grp")
        chat._album_images[key] = collecting_album(album_message())

        with patch.object(chat, "_open_registered", AsyncMock()) as register:
            await chat.handle_text(message)

        register.assert_not_awaited()
        self.assertEqual(chat._album_images[key]["extra_text"], ["Long post body"])

    async def test_album_analysis_sends_text_and_images_in_one_request(self):
        message = album_message(caption="Channel caption")
        svc = service()
        album_call = AsyncMock(
            return_value=ChatResult(content="Album description", elapsed=0.1)
        )
        with (
            patch.object(chat, "_is_addressed", return_value=True),
            patch.object(chat, "is_vision_model", return_value=True),
            patch.object(chat, "_open_registered", AsyncMock(return_value=svc)),
            patch.object(chat, "_consume_access", AsyncMock(return_value=ALLOWED)),
            patch.object(chat, "waiting", no_wait),
            patch.object(chat, "deliver_answer", AsyncMock()),
            patch.object(chat, "guest_hint", return_value=None),
            patch.object(chat, "mcp_write_available", return_value=False),
            patch.object(chat, "safe_answer", AsyncMock()),
            patch.object(chat, "detect_intent", AsyncMock()) as detect,
            patch.object(chat.ai_client, "chat_completion_result", album_call),
        ):
            await chat._process_album(
                message,
                [("f1", b"image1"), ("f2", b"image2")],
                ["Long post body"],
            )

        detect.assert_not_awaited()
        album_call.assert_awaited_once()
        parts = album_call.await_args.args[0][-1]["content"]
        self.assertEqual(parts[0]["type"], "text")
        self.assertIn("Channel caption", parts[0]["text"])
        self.assertIn("Long post body", parts[0]["text"])
        self.assertEqual([part["type"] for part in parts].count("image_url"), 2)
        stored = svc.add_message.await_args_list[0].kwargs["content"]
        self.assertIn("Long post body", stored)
        self.assertTrue(has_untrusted_content(stored))


class ForwardedBurstTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        chat._album_images.clear()
        chat._album_total_bytes.clear()
        chat._album_download_bytes.clear()
        chat._forward_bursts.clear()
        for name, value in (
            ("_album_lock", asyncio.Lock()),
            ("_forward_burst_lock", asyncio.Lock()),
            ("_FORWARD_TEXT_MERGE_SECONDS", 0),
        ):
            item = patch.object(chat, name, value)
            item.start()
            self.addCleanup(item.stop)

    async def asyncTearDown(self):
        tasks = [
            entry["task"]
            for entry in chat._forward_bursts.values()
            if entry.get("task") is not None
        ]
        for task in tasks:
            task.cancel()
        for task in tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
        chat._forward_bursts.clear()
        chat._album_images.clear()
        chat._album_total_bytes.clear()
        chat._album_download_bytes.clear()

    async def test_batch_of_forwarded_texts_is_answered_once(self):
        first = forwarded_text("First message")
        second = forwarded_text("Second message")
        third = forwarded_text("Third message")

        with patch.object(chat, "handle_text", AsyncMock()) as answer:
            for message in (first, second, third):
                self.assertTrue(await chat._queue_forwarded_burst(message))
            entry = chat._forward_bursts[chat._forward_burst_key(first)]
            await entry["task"]

        answer.assert_awaited_once()
        self.assertIs(answer.await_args.args[0], first)
        self.assertEqual(
            answer.await_args.kwargs["merged_text"],
            "First message\n\nSecond message\n\nThird message",
        )

    async def test_batch_with_photos_uses_one_album_and_all_texts(self):
        text = forwarded_text("Caption text")
        first = forwarded_photo("f1", caption="Photo one")
        second = forwarded_photo("f2")
        seen: dict[str, Any] = {"groups": set(), "extra": None}

        async def fake_handler(proxy):
            key = chat._album_key(proxy.chat.id, proxy.media_group_id)
            seen["groups"].add(proxy.media_group_id)
            seen["extra"] = list(chat._album_images[key].get("extra_text", []))

        with (
            patch.object(
                chat, "_album_access_allowed", AsyncMock(return_value=(True, None))
            ),
            patch.object(
                chat, "_handle_album_photo", AsyncMock(side_effect=fake_handler)
            ) as handler,
        ):
            await chat._process_forwarded_burst([text, first, second])

        self.assertEqual(handler.await_count, 2)
        self.assertEqual(len(seen["groups"]), 1)
        self.assertEqual(seen["extra"], ["Caption text", "Photo one"])
        self.assertEqual(chat._album_images, {})
        self.assertEqual(chat._album_total_bytes, {})

    async def test_channel_forward_is_not_a_personal_batch(self):
        message = forwarded_text("Channel text")
        message.forward_origin = SimpleNamespace(
            type="channel", sender_chat=SimpleNamespace(id=9, title="Channel")
        )

        self.assertFalse(await chat._queue_forwarded_burst(message))
        self.assertEqual(chat._forward_bursts, {})

    async def test_own_message_is_not_a_personal_batch(self):
        message = forwarded_text("My own text")
        message.forward_origin = None

        self.assertFalse(await chat._queue_forwarded_burst(message))

    async def test_group_forward_is_not_a_personal_batch(self):
        message = forwarded_text("Group text")
        message.chat = SimpleNamespace(id=-100, type="supergroup")

        self.assertFalse(await chat._queue_forwarded_burst(message))

    async def test_reply_forwarded_message_is_not_batched(self):
        message = forwarded_text("Reply text")
        message.reply_to_message = SimpleNamespace(text="quoted")

        self.assertFalse(await chat._queue_forwarded_burst(message))

    async def test_handle_text_queues_personal_forward(self):
        message = forwarded_text("Only message")

        with patch.object(chat, "_open_registered", AsyncMock()) as register:
            await chat.handle_text(message)

        register.assert_not_awaited()
        self.assertIn(chat._forward_burst_key(message), chat._forward_bursts)


if __name__ == "__main__":
    unittest.main()