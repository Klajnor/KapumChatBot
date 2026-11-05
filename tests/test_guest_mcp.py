"""Гостевой режим читает «второй мозг»: MCP-инструменты и их ограничения."""

from __future__ import annotations

import datetime
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from bot.config import config
from bot.handlers import guest
from bot.services.access import AccessResult
from bot.services.ai_client import ChatResult, ToolCall
from bot.services.mcp import MCP_SYSTEM_HINT
from bot.services.untrusted import (
    UNTRUSTED_SYSTEM_HINT,
    messages_have_untrusted_content,
    wrap_untrusted,
)

ACCESS = AccessResult(True, "guest", consumed=True)
NOTE_TOOL = "mcp__brain__get_note"
TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": NOTE_TOOL,
        "description": "Прочитать заметку",
        "parameters": {"type": "object", "properties": {}},
    },
}


def caller():
    return SimpleNamespace(id=1, username="u", first_name="U", last_name=None)


def message(*, text="@bot what do my notes say about rust?", **overrides):
    values = {
        "text": text,
        "caption": None,
        "photo": [],
        "voice": None,
        "document": None,
        "reply_to_message": None,
        "forward_date": None,
        "forward_origin": None,
        "from_user": caller(),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def service():
    return SimpleNamespace(
        register_user=AsyncMock(),
        get_user_settings=AsyncMock(return_value={
            "model": "m", "system_prompt": "sys",
            "temperature": 0.5, "max_tokens": 100,
        }),
        get_history=AsyncMock(return_value=[]),
        add_message=AsyncMock(),
        consume_photo_intent_budget=AsyncMock(return_value=True),
    )


def manager(note: str = "Rust notes: the borrow checker."):
    return SimpleNamespace(
        openai_tools=AsyncMock(return_value=[TOOL_SCHEMA]),
        is_mcp_tool=lambda name: name == NOTE_TOOL,
        call_tool=AsyncMock(return_value=wrap_untrusted("mcp", note)),
    )


class GuestSecondBrainTests(unittest.IsolatedAsyncioTestCase):
    async def test_guest_reads_a_note_and_answers_from_it(self):
        svc = service()
        msg = message()
        first = ChatResult(
            content=None,
            tool_calls=[
                ToolCall(id="c1", name=NOTE_TOOL, arguments={"path": "rust.md"})
            ],
        )
        final = ChatResult(
            content="Your notes mention the borrow checker.", elapsed=0.2
        )
        first_seen: dict[str, object] = {}

        async def record_tools(**kwargs):
            # The loop mutates ``messages`` in place: snapshot the first round.
            first_seen["messages"] = [dict(entry) for entry in kwargs["messages"]]
            first_seen["tools"] = kwargs["tools"]
            return first

        calls = AsyncMock(side_effect=record_tools)
        complete = AsyncMock(return_value=final)
        brain = manager()
        with (
            patch.object(guest, "check_access", AsyncMock(return_value=ACCESS)),
            patch.object(guest, "detect_intent", AsyncMock()),
            patch.object(config, "enable_image_generation", True),
            patch.object(config, "enable_link_analysis", False),
            patch.object(config, "mcp_max_rounds", 2),
            patch.object(guest, "mcp_available", return_value=True),
            patch.object(guest, "mcp_manager", brain),
            patch.object(guest.ai_client, "chat_with_tools", calls),
            patch.object(guest.ai_client, "chat_completion_result", complete),
        ):
            reply = await guest._build_reply(msg, svc, msg.from_user, 42)

        self.assertIn("borrow checker", reply.text)
        brain.openai_tools.assert_awaited_once()
        brain.call_tool.assert_awaited_once()
        self.assertEqual(brain.call_tool.await_args.args[0], NOTE_TOOL)
        self.assertEqual(calls.await_count, 1)
        self.assertEqual(complete.await_count, 1)

        # Первый запрос: заметки предложены, хинт второго мозга в промпте.
        first_messages = first_seen["messages"]
        self.assertIn(MCP_SYSTEM_HINT, first_messages[0]["content"])
        first_names = {
            tool["function"]["name"] for tool in first_seen["tools"]
        }
        self.assertIn(NOTE_TOOL, first_names)
        self.assertIn("generate_image", first_names)

        # Второй запрос: результат заметки в контексте, инструменты сняты,
        # текст заметки помечен как недоверенный.
        second_call = complete.await_args_list[0]
        tool_turns = [
            entry
            for entry in second_call.kwargs["messages"]
            if entry.get("role") == "tool"
        ]
        self.assertEqual(len(tool_turns), 1)
        self.assertIn("borrow checker", tool_turns[0]["content"])
        self.assertTrue(
            messages_have_untrusted_content(second_call.kwargs["messages"])
        )
        self.assertIn(
            UNTRUSTED_SYSTEM_HINT, second_call.kwargs["messages"][0]["content"]
        )

        # В историю попадают только вопрос и финальный ответ, не выдача MCP.
        saved = [c.kwargs.get("content") for c in svc.add_message.await_args_list]
        self.assertEqual(saved[0], msg.text)
        self.assertNotIn(tool_turns[0]["content"], saved)
        self.assertEqual(saved[-1], final.content)

    async def test_forwarded_summon_never_offers_the_notes_tools(self):
        svc = service()
        msg = message(
            text="@bot summarize this",
            forward_date=datetime.datetime(2026, 10, 31),
            forward_origin=SimpleNamespace(
                sender_user=SimpleNamespace(first_name="Alice")
            ),
        )
        brain = manager()
        complete = AsyncMock(return_value=ChatResult(content="ok", elapsed=0.1))
        tools_call = AsyncMock(return_value=ChatResult(content="unused"))
        with (
            patch.object(guest, "check_access", AsyncMock(return_value=ACCESS)),
            patch.object(config, "enable_image_generation", True),
            patch.object(config, "enable_link_analysis", False),
            patch.object(guest, "mcp_available", return_value=True),
            patch.object(guest, "mcp_manager", brain),
            patch.object(guest.ai_client, "chat_with_tools", tools_call),
            patch.object(guest.ai_client, "chat_completion_result", complete),
        ):
            await guest._build_reply(msg, svc, msg.from_user, 42)

        brain.openai_tools.assert_not_awaited()
        brain.call_tool.assert_not_awaited()
        tools_call.assert_not_awaited()
        complete.assert_awaited_once()

    async def test_unavailable_second_brain_keeps_the_plain_path(self):
        svc = service()
        msg = message(text="@bot hello")
        brain = manager()
        complete = AsyncMock(return_value=ChatResult(content="hi", elapsed=0.1))
        tools_call = AsyncMock(return_value=ChatResult(content="unused"))
        with (
            patch.object(guest, "check_access", AsyncMock(return_value=ACCESS)),
            patch.object(config, "enable_image_generation", False),
            patch.object(config, "enable_link_analysis", False),
            patch.object(guest, "mcp_available", return_value=False),
            patch.object(guest, "mcp_manager", brain),
            patch.object(guest.ai_client, "chat_with_tools", tools_call),
            patch.object(guest.ai_client, "chat_completion_result", complete),
        ):
            await guest._build_reply(msg, svc, msg.from_user, 42)

        brain.openai_tools.assert_not_awaited()
        tools_call.assert_not_awaited()
        complete.assert_awaited_once()
        messages = complete.await_args.kwargs["messages"]
        self.assertNotIn(MCP_SYSTEM_HINT, messages[0]["content"])
