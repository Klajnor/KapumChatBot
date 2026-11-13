"""Пересланные rich-сообщения: разбор блоков и анализ в чате/guest."""

from __future__ import annotations

import time
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from aiogram import F
from aiogram.types import Message

from bot.config import config
from bot.handlers import brain, chat, guest
from bot.services import telegram_utils
from bot.services.access import AccessResult
from bot.services.ai_client import ChatResult
from bot.services.telegram_quotes import (
    MAX_QUOTE_CHARS,
    message_text,
    reply_message_text,
)
from bot.services.untrusted import (
    UNTRUSTED_SYSTEM_HINT,
    has_untrusted_content,
)


def rich_parsed(
    *,
    text: str = "Hello ",
    bold: str = "world",
    table: bool = False,
    chat_type: str = "private",
) -> Message:
    blocks = [
        {"type": "paragraph", "text": [text, {"type": "bold", "text": bold}]},
    ]
    if table:
        blocks.append(
            {
                "type": "table",
                "cells": [
                    [
                        {"text": "A", "align": "left", "valign": "middle"},
                        {"text": "B", "align": "left", "valign": "middle"},
                    ],
                    [
                        {"text": "1", "align": "left", "valign": "middle"},
                        {"text": "2", "align": "left", "valign": "middle"},
                    ],
                ],
            }
        )
    return Message.model_validate(
        {
            "message_id": 10,
            "date": 1700000000,
            "chat": {"id": 1, "type": chat_type, "first_name": "U"},
            "from": {"id": 1, "is_bot": False, "first_name": "Me"},
            "rich_message": {"blocks": blocks},
        }
    )


def forwarded_rich(*, text: str = "Hello ") -> Message:
    return Message.model_validate(
        {
            "message_id": 10,
            "date": 1700000000,
            "chat": {"id": 1, "type": "private", "first_name": "U"},
            "from": {"id": 1, "is_bot": False, "first_name": "Me"},
            "forward_origin": {
                "type": "user",
                "date": 1700000000,
                "sender_user": {"id": 2, "is_bot": False, "first_name": "Alice"},
            },
            "rich_message": {
                "blocks": [
                    {
                        "type": "paragraph",
                        "text": [text, {"type": "bold", "text": "world"}],
                    },
                ]
            },
        }
    )


def chat_message(rich: Message):
    return SimpleNamespace(
        text=rich.text,
        caption=rich.caption,
        rich_message=rich.rich_message,
        message_id=rich.message_id,
        chat=rich.chat,
        from_user=SimpleNamespace(
            id=1, first_name="Me", username="me", last_name=None
        ),
        forward_origin=rich.forward_origin,
        forward_date=rich.forward_date,
        reply_to_message=None,
        external_reply=None,
        quote=None,
        photo=[],
        voice=None,
        document=None,
        bot=SimpleNamespace(send_chat_action=AsyncMock()),
        answer=AsyncMock(),
    )


@contextmanager
def slot_granted(*args, **kwargs):
    yield True


class RichExtractionTests(unittest.TestCase):
    def test_table_cells_are_separated(self):
        parsed = rich_parsed(text="Hello ", table=True)
        text = message_text(parsed)
        self.assertIn("Hello world", text)
        self.assertIn("A | B", text)
        self.assertIn("1 | 2", text)

    def test_rich_request_is_not_limited_to_quote_size(self):
        parsed = rich_parsed(text="x" * 5000, bold="")
        self.assertEqual(len(message_text(parsed)), 5000)
        self.assertEqual(len(reply_message_text(parsed)), MAX_QUOTE_CHARS + 1)

    def test_rich_blocks_are_preferred_over_plain_text_fallback(self):
        message = {
            "text": "plain fallback",
            "rich_message": {"blocks": [{"type": "paragraph", "text": "rich body"}]},
        }
        self.assertEqual(message_text(message), "rich body")
        self.assertEqual(reply_message_text(message), "rich body")

    def test_rich_request_is_capped_at_32k(self):
        parsed = rich_parsed(text="x" * 33000, bold="")
        self.assertEqual(len(message_text(parsed)), 32001)

    def test_thinking_only_message_has_no_visible_text(self):
        hidden = {
            "rich_message": {
                "blocks": [
                    {"type": "paragraph", "text": ""},
                    {"type": "thinking", "text": "SECRET"},
                ]
            }
        }
        self.assertEqual(message_text(hidden), "")


class RichChatTests(unittest.IsolatedAsyncioTestCase):
    def _service(self):
        return SimpleNamespace(
            session=SimpleNamespace(close=AsyncMock()),
            add_message=AsyncMock(return_value=SimpleNamespace(id=1)),
        )

    @contextmanager
    def _text_env(self):
        service = self._service()
        with (
            patch.object(chat.limits, "slot", slot_granted),
            patch.object(chat, "_is_addressed", return_value=True),
            patch.object(chat, "_open_registered", AsyncMock(return_value=service)),
            patch.object(
                chat, "_consume_access", AsyncMock(return_value=AccessResult(True, "user"))
            ),
            patch.object(chat, "_answer_with_ai", AsyncMock()) as answer,
            patch.object(chat, "maybe_timezone_dialog", AsyncMock(return_value=False)),
            patch.object(chat, "maybe_freeform_reminder", AsyncMock(return_value=False)),
            patch.object(chat, "maybe_status_request", AsyncMock(return_value=False)),
            patch("bot.handlers.brain.maybe_remember", AsyncMock(return_value=False)),
            patch.object(config, "enable_link_analysis", False),
        ):
            yield answer

    async def test_forwarded_rich_message_reaches_ai_as_untrusted_forward(self):
        msg = chat_message(forwarded_rich(text="Important "))
        msg.rich_message = {
            "blocks": [
                {"type": "paragraph", "text": "Important world"},
                {"type": "thinking", "text": "SECRET"},
            ]
        }
        with self._text_env() as answer:
            await chat.handle_text(msg)

        prompt = answer.await_args.args[2]
        self.assertIn("Important world", prompt)
        self.assertIn("[Forwarded message from Alice]", prompt)
        self.assertIn('<untrusted_content source="forward">', prompt)
        self.assertNotIn("SECRET", prompt)
        self.assertIsNone(answer.await_args.kwargs["intent_text"])

    async def test_own_rich_message_is_analyzed_as_trusted_text(self):
        msg = chat_message(rich_parsed(text="My own "))
        with self._text_env() as answer:
            await chat.handle_text(msg)

        prompt = answer.await_args.args[2]
        self.assertIn("My own world", prompt)
        self.assertNotIn("[Forwarded message", prompt)
        self.assertEqual(answer.await_args.kwargs["intent_text"], "My own world")

    async def test_thinking_only_rich_message_is_ignored_before_quota(self):
        msg = chat_message(rich_parsed(text="", bold=""))
        msg.rich_message = {
            "blocks": [
                {"type": "paragraph", "text": ""},
                {"type": "thinking", "text": "SECRET"},
            ]
        }
        with (
            self._text_env() as answer,
            patch.object(chat, "_open_registered", AsyncMock()) as register,
        ):
            await chat.handle_text(msg)

        answer.assert_not_awaited()
        register.assert_not_awaited()

    def test_handler_filter_accepts_rich_message(self):
        parsed = forwarded_rich()
        self.assertTrue(F.rich_message.resolve(parsed))
        self.assertTrue(
            (F.rich_message | (F.text & ~F.text.startswith("/"))).resolve(parsed)
        )

    async def test_edited_rich_message_refreshes_remembered_question(self):
        msg = chat_message(rich_parsed(text="Edited "))
        with (
            patch.object(chat, "_is_addressed", return_value=True),
            patch.object(chat, "_edit_notice_allowed", return_value=True),
            patch.object(chat, "_remember_question") as remember,
            patch.object(chat, "safe_answer", AsyncMock()),
            patch.object(chat, "_retry_markup", return_value=None),
        ):
            await chat.handle_edited_text(msg)

        remember.assert_called_once()
        self.assertIn("Edited world", remember.call_args.args[1])

    def test_group_rich_message_is_addressed_by_mention_inside_content(self):
        parsed = rich_parsed(text="@bot explain ", chat_type="group")
        msg = chat_message(parsed)
        with patch.object(
            telegram_utils, "get_bot_username", return_value="bot"
        ):
            self.assertTrue(chat._is_addressed(msg))


class RichGuestTests(unittest.IsolatedAsyncioTestCase):
    def _service(self):
        return SimpleNamespace(
            register_user=AsyncMock(),
            get_user_settings=AsyncMock(
                return_value={
                    "model": "m",
                    "system_prompt": "sys",
                    "temperature": 0.5,
                    "max_tokens": 100,
                }
            ),
            get_history=AsyncMock(return_value=[]),
            add_message=AsyncMock(),
            consume_photo_intent_budget=AsyncMock(return_value=True),
        )

    async def test_guest_oversized_rich_text_is_rejected_before_access(self):
        rich = rich_parsed(text="x" * 5000, bold="")
        msg = SimpleNamespace(
            text=None,
            caption=None,
            rich_message=rich.rich_message,
            photo=[],
            voice=None,
            document=None,
            reply_to_message=None,
            forward_date=None,
            forward_origin=None,
            from_user=SimpleNamespace(
                id=1, username="u", first_name="U", last_name=None
            ),
        )
        svc = self._service()
        with (
            patch.object(config, "max_guest_text_length", 4000),
            patch.object(guest, "check_access", AsyncMock()) as access,
        ):
            reply = await guest._build_reply(msg, svc, msg.from_user, 42)

        self.assertIn("too long", reply.text)
        access.assert_not_awaited()

    async def test_forwarded_rich_guest_message_is_parsed_and_untrusted(self):
        rich = forwarded_rich(text="Guest ")
        msg = SimpleNamespace(
            text=None,
            caption=None,
            rich_message={
                "blocks": [
                    {"type": "paragraph", "text": "Guest world"},
                    {"type": "thinking", "text": "SECRET"},
                ]
            },
            photo=[],
            voice=None,
            document=None,
            reply_to_message=None,
            forward_date=rich.forward_date,
            forward_origin=rich.forward_origin,
            from_user=SimpleNamespace(
                id=1, username="u", first_name="U", last_name=None
            ),
        )
        svc = self._service()
        tools = AsyncMock(return_value=ChatResult(content="answer", elapsed=0.1))
        with (
            patch.object(
                guest,
                "check_access",
                AsyncMock(return_value=AccessResult(True, "guest", consumed=True)),
            ),
            patch.object(guest, "search_configured", return_value=True),
            patch.object(guest, "mcp_available", return_value=False),
            patch.object(config, "builtin_web_search", return_value=False),
            patch.object(config, "enable_image_generation", True),
            patch.object(config, "enable_link_analysis", False),
            patch.object(guest.ai_client, "chat_with_tools", tools),
        ):
            await guest._build_reply(msg, svc, msg.from_user, 42)

        messages = tools.await_args.kwargs["messages"]
        prompt = messages[-1]["content"]
        self.assertIn("Guest world", prompt)
        self.assertIn('<untrusted_content source="forward">', prompt)
        self.assertNotIn("SECRET", prompt)
        self.assertIn(UNTRUSTED_SYSTEM_HINT, messages[0]["content"])
        tool_names = {
            tool["function"]["name"] for tool in tools.await_args.kwargs["tools"]
        }
        self.assertNotIn("generate_image", tool_names)
        stored = svc.add_message.await_args_list[0].kwargs["content"]
        self.assertIn("Guest world", stored)
        self.assertTrue(has_untrusted_content(stored))


class BrainRichTests(unittest.IsolatedAsyncioTestCase):
    def _bot_message(self):
        return Message.model_validate(
            {
                "message_id": 11,
                "date": 1700000000,
                "chat": {"id": 1, "type": "private", "first_name": "Me"},
                "from": {"id": 999, "is_bot": True, "first_name": "Bot"},
                "text": "Choose an action",
            }
        )

    def _callback(self, action: str):
        return SimpleNamespace(
            data=f"brain:{action}:key",
            from_user=SimpleNamespace(id=1),
            message=self._bot_message(),
        )

    def _pending(self, origin: Message):
        brain._pending["key"] = brain.PendingForward(
            parts=[origin], expires=time.monotonic() + 3600
        )
        self.addCleanup(lambda: brain._pending.pop("key", None))

    async def test_analyze_button_routes_rich_forward_to_chat_handler(self):
        origin = forwarded_rich(text="Analyze ")
        self._pending(origin)
        with (
            patch.object(brain, "mcp_write_available", return_value=True),
            patch.object(brain.limits, "slot", slot_granted),
            patch.object(brain, "safe_answer_callback", AsyncMock()),
            patch.object(brain, "safe_answer", AsyncMock()),
            patch("bot.handlers.chat.handle_text", AsyncMock()) as handler,
        ):
            await brain.brain_callback(self._callback("analyze"))

        handler.assert_awaited_once_with(origin)
        self.assertNotIn("key", brain._pending)

    async def test_remember_button_saves_rich_visible_text(self):
        origin = forwarded_rich(text="Remember ")
        self._pending(origin)
        save = AsyncMock(return_value=True)
        with (
            patch.object(brain, "mcp_write_available", return_value=True),
            patch.object(brain.limits, "slot", slot_granted),
            patch.object(brain, "safe_answer_callback", AsyncMock()),
            patch.object(brain, "safe_answer", AsyncMock()),
            patch.object(brain, "_save", save),
        ):
            await brain.brain_callback(self._callback("forward"))

        save.assert_awaited_once()
        saved = save.await_args.args[2]
        self.assertIn("Remember world", saved)
        self.assertNotIn("key", brain._pending)
