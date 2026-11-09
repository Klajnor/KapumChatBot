"""Явный поиск по заметкам не должен блокироваться старым внешним контентом."""

from __future__ import annotations

import copy
import os
import unittest
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock, patch

# Тесты не используют локальные ключи и внешние службы.
os.environ["BOT_TOKEN"] = "123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghi"
os.environ["AI_API_KEY"] = "test-key"
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"

from bot.config import config
from bot.handlers import chat, guest
from bot.services.access import AccessResult
from bot.services.ai_client import ChatResult, ToolCall
from bot.services.mcp import MCP_SYSTEM_HINT, isolate_mcp_history
from bot.services.untrusted import UNTRUSTED_SYSTEM_HINT, wrap_untrusted

QUESTION = "Найди во втором мозге заметки про настройку поиска в интернете"
SEARCH_NAME = "mcp__brain__search_notes"
SEARCH_SCHEMA = {
    "type": "function",
    "function": {
        "name": SEARCH_NAME,
        "description": "Поиск по заметкам",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}


def service(history):
    return SimpleNamespace(
        register_user=AsyncMock(),
        get_user_settings=AsyncMock(return_value={
            "model": "m", "system_prompt": "sys", "temperature": 0.5, "max_tokens": 100,
        }),
        get_history=AsyncMock(return_value=history),
        add_message=AsyncMock(return_value=SimpleNamespace(id=1)),
    )


def manager():
    return SimpleNamespace(
        openai_tools=AsyncMock(return_value=[SEARCH_SCHEMA]),
        is_mcp_tool=lambda name: name == SEARCH_NAME,
        tool_info=lambda name: SimpleNamespace(name="search_notes"),
        call_tool=AsyncMock(return_value=wrap_untrusted("mcp", "Search.md: настройка поиска")),
    )


@asynccontextmanager
async def no_wait(*args, **kwargs):
    yield SimpleNamespace()


class MCPHistoryTests(unittest.IsolatedAsyncioTestCase):
    async def run_chat(self, text, history, *, streaming=False, available=True, catalog_ready=True):
        svc = service(history)
        brain = manager()
        if not catalog_ready:
            brain.openai_tools.return_value = []
        seen = []
        first = ChatResult(tool_calls=[ToolCall(
            id="notes", name=SEARCH_NAME, arguments={"query": "настройка поиска"},
        )])

        async def answer(messages, *args, tools=None, **kwargs):
            # _single_shot передаёт tools позиционно, stream — по имени.
            offered = args[2] if len(args) == 3 else tools
            seen.append({"messages": copy.deepcopy(messages), "tools": offered})
            names = {tool["function"]["name"] for tool in offered or []}
            return first if len(seen) == 1 and SEARCH_NAME in names else ChatResult(content="По Search.md: ответ")

        message = SimpleNamespace(
            chat=SimpleNamespace(id=1, type="private"),
            from_user=SimpleNamespace(id=1), answer=AsyncMock(),
        )
        printer = SimpleNamespace(on_delta=AsyncMock(), discard=AsyncMock())
        with (
            patch.multiple(config, stream_responses=streaming, request_intent=False,
                           enable_image_generation=True, enable_sticker_generation=True,
                           enable_file_replies=True),
            patch.object(chat, "search_configured", return_value=True),
            patch.object(chat, "mcp_available", return_value=available),
            patch.object(chat, "mcp_manager", brain),
            patch.object(chat, "_single_shot", side_effect=answer),
            patch.object(chat.ai_client, "stream_chat_completion", side_effect=answer),
            patch.object(chat, "_StreamPrinter", return_value=printer),
            patch.object(chat, "waiting", no_wait),
            patch.object(chat, "safe_answer", AsyncMock()),
            patch.object(chat, "deliver_answer", AsyncMock(return_value=True)) as deliver,
            patch.object(chat, "guest_hint", return_value=None),
            patch.object(chat, "_retry_markup", return_value=None),
        ):
            await chat._answer_with_ai(message, svc, text, AccessResult(True, "user"))
        return svc, brain, seen, deliver

    async def test_explicit_note_search_isolated_from_all_external_history(self):
        for streaming in (False, True):
            for source in ("document", "web", "forward", "quoted", "photo", "search", "mcp"):
                with self.subTest(streaming=streaming, source=source):
                    history = [
                        {"role": "user", "content": wrap_untrusted(source, "PRIVATE MATERIAL")},
                        {"role": "assistant", "content": "PRIVATE SUMMARY"},
                    ]
                    original = copy.deepcopy(history)
                    svc, brain, seen, deliver = await self.run_chat(QUESTION, history, streaming=streaming)
                    brain.call_tool.assert_awaited_once_with(
                        SEARCH_NAME, {"query": "настройка поиска"}, max_chars=config.tool_max_context_chars,
                        read_budget=ANY,
                    )
                    self.assertEqual(len(seen), 2)
                    first, final = seen
                    self.assertEqual([tool["function"]["name"] for tool in first["tools"]], [SEARCH_NAME])
                    self.assertIn(MCP_SYSTEM_HINT, first["messages"][0]["content"])
                    self.assertEqual(first["messages"][-1], {"role": "user", "content": QUESTION})
                    self.assertEqual(len(first["messages"]), 2)
                    self.assertIsNone(final["tools"])
                    self.assertIn(UNTRUSTED_SYSTEM_HINT, final["messages"][0]["content"])
                    self.assertNotIn(MCP_SYSTEM_HINT, final["messages"][0]["content"])
                    for payload in seen:
                        self.assertNotIn("PRIVATE", str(payload))
                    self.assertEqual(history, original)
                    self.assertEqual(
                        [call.kwargs["content"] for call in svc.add_message.await_args_list],
                        [QUESTION, "По Search.md: ответ"],
                    )
                    self.assertFalse(deliver.await_args.kwargs["allow_telegraph"])

    async def test_current_external_content_and_followups_keep_the_gate(self):
        for text in (
            "Объясни второй пункт",
            "Найди ещё по этой теме",
            QUESTION + "\n" + wrap_untrusted("quoted", "PRIVATE QUOTE"),
            wrap_untrusted("forward", QUESTION),
        ):
            with self.subTest(text=text):
                history = [{"role": "user", "content": wrap_untrusted("document", "PRIVATE MATERIAL")}]
                _, brain, seen, _ = await self.run_chat(text, history)
                brain.call_tool.assert_not_awaited()
                self.assertIsNone(seen[0]["tools"])
                self.assertIn("PRIVATE MATERIAL", str(seen[0]["messages"]))

    async def test_disabled_mcp_does_not_isolate_history_or_expose_tools(self):
        history = [{"role": "user", "content": wrap_untrusted("document", "PRIVATE MATERIAL")}]
        # Без упоминания интернета: это не запрос на изолированный веб-поиск.
        _, brain, seen, _ = await self.run_chat("Поищи во втором мозге про Python", history, available=False)
        brain.openai_tools.assert_not_awaited()
        self.assertIn("PRIVATE MATERIAL", str(seen[0]["messages"]))
        self.assertIsNone(seen[0]["tools"])

    async def test_clean_history_is_not_discarded(self):
        history = [{"role": "user", "content": "Ранее обсуждали Python"}]
        _, _, seen, _ = await self.run_chat("Поищи во втором мозге про Python", history)
        self.assertIn(history[0], seen[0]["messages"])
        # Явная просьба только к заметкам не должна случайно отправляться в веб.
        self.assertNotIn("search_web", {tool["function"]["name"] for tool in seen[0]["tools"]})

    async def test_empty_catalog_does_not_open_web_search_or_media_tools(self):
        history = [{"role": "user", "content": wrap_untrusted("document", "PRIVATE MATERIAL")}]
        _, brain, seen, _ = await self.run_chat(QUESTION, history, catalog_ready=False)
        brain.openai_tools.assert_awaited_once()
        brain.call_tool.assert_not_awaited()
        self.assertIsNone(seen[0]["tools"])
        self.assertNotIn(MCP_SYSTEM_HINT, seen[0]["messages"][0]["content"])
        # Пустой каталог не должен вырезать историю: изоляция теперь решается
        # только после успешной загрузки инструментов.
        self.assertIn("PRIVATE MATERIAL", str(seen[0]["messages"]))

    async def test_explicit_web_search_does_not_switch_to_notes(self):
        history = [{"role": "user", "content": wrap_untrusted("document", "PRIVATE MATERIAL")}]
        _, brain, seen, _ = await self.run_chat("Найди в интернете плагины для Obsidian", history)
        brain.openai_tools.assert_awaited_once()
        brain.call_tool.assert_not_awaited()
        self.assertEqual([tool["function"]["name"] for tool in seen[0]["tools"]], ["search_web"])

    async def test_web_search_mentioning_obsidian_stays_web(self):
        """«… в интернете … в Obsidian» не должен глушить search_web."""
        history = [{"role": "user", "content": wrap_untrusted("document", "PRIVATE MATERIAL")}]
        _, brain, seen, _ = await self.run_chat("Найди в интернете плагины в Obsidian", history)
        brain.call_tool.assert_not_awaited()
        self.assertEqual([tool["function"]["name"] for tool in seen[0]["tools"]], ["search_web"])
        self.assertNotIn("PRIVATE", str(seen[0]["messages"]))


class GuestMCPHistoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_guest_explicit_note_search_can_leave_external_history_behind(self):
        svc = service([
            {"role": "user", "content": wrap_untrusted("web", "PRIVATE MATERIAL")},
            {"role": "assistant", "content": "PRIVATE SUMMARY"},
        ])
        brain = manager()
        caller = SimpleNamespace(id=1, username="u", first_name="U", last_name=None)
        message = SimpleNamespace(
            text=QUESTION, caption=None, photo=[], voice=None, document=None,
            reply_to_message=None, forward_date=None, forward_origin=None, from_user=caller,
        )
        seen = []

        async def answer(**kwargs):
            seen.append(copy.deepcopy(kwargs))
            if len(seen) == 1 and kwargs.get("tools"):
                return ChatResult(tool_calls=[ToolCall(id="notes", name=SEARCH_NAME, arguments={"query": "поиск"})])
            return ChatResult(content="По Search.md: ответ")

        with (
            patch.multiple(config, enable_image_generation=True, enable_link_analysis=False),
            patch.object(guest, "check_access", AsyncMock(return_value=AccessResult(True, "guest"))),
            patch.object(guest, "mcp_available", return_value=True),
            patch.object(guest, "mcp_manager", brain),
            patch.object(guest.ai_client, "chat_with_tools", side_effect=answer),
            patch.object(guest.ai_client, "chat_completion_result", side_effect=answer),
            patch.object(guest, "_maybe_telegraph_reply", AsyncMock(return_value=None)) as telegraph,
        ):
            reply = await guest._build_reply(message, svc, caller, 42)
        brain.openai_tools.assert_awaited_once()
        brain.call_tool.assert_awaited_once()
        self.assertIn("Search.md", reply.text)
        self.assertEqual([tool["function"]["name"] for tool in seen[0]["tools"]], [SEARCH_NAME])
        self.assertEqual(len(seen[0]["messages"]), 2)
        self.assertIn(MCP_SYSTEM_HINT, seen[0]["messages"][0]["content"])
        # История вырезана: подсказка про недоверенные теги больше не нужна.
        self.assertNotIn(UNTRUSTED_SYSTEM_HINT, seen[0]["messages"][0]["content"])
        self.assertNotIn("tools", seen[1])
        for payload in seen:
            self.assertNotIn("PRIVATE", str(payload))
        self.assertFalse(telegraph.await_args.kwargs["allow"])

    async def test_guest_forward_with_explicit_note_request_cannot_bypass_gate(self):
        svc = service([{"role": "user", "content": wrap_untrusted("web", "PRIVATE MATERIAL")}])
        brain = manager()
        caller = SimpleNamespace(id=1, username="u", first_name="U", last_name=None)
        message = SimpleNamespace(
            text=QUESTION, caption=None, photo=[], voice=None, document=None,
            reply_to_message=None, forward_origin=SimpleNamespace(sender_user=caller), from_user=caller,
        )
        with (
            patch.multiple(config, enable_image_generation=True, enable_link_analysis=False),
            patch.object(guest, "check_access", AsyncMock(return_value=AccessResult(True, "guest"))),
            patch.object(guest, "mcp_available", return_value=True),
            patch.object(guest, "mcp_manager", brain),
            patch.object(guest.ai_client, "chat_with_tools", AsyncMock()) as tools,
            patch.object(guest.ai_client, "chat_completion_result", AsyncMock(return_value=ChatResult(content="ok"))) as plain,
        ):
            await guest._build_reply(message, svc, caller, 42)
        brain.openai_tools.assert_not_awaited()
        brain.call_tool.assert_not_awaited()
        tools.assert_not_awaited()
        plain.assert_awaited_once()
        self.assertIn("PRIVATE MATERIAL", str(plain.await_args.kwargs["messages"]))


class MCPHistoryIsolationTests(unittest.TestCase):
    def test_explicit_russian_and_english_requests_can_be_isolated(self):
        for question in (
            "Найди во втором мозге про Rust",
            "Поищи в моём втором мозге про Rust",
            "Посмотри в моих заметках про Rust",
            "Поиск по сохранённым заметкам про Rust",
            "Прочитай в Obsidian заметку Rust.md",
            "Search in my second brain for Rust",
            "Find in my notes information about Rust",
            "Search my notes for Rust",
        ):
            with self.subTest(question=question):
                history = [{"role": "user", "content": wrap_untrusted("document", "PRIVATE MATERIAL")}]
                system = {"role": "system", "content": "sys"}
                messages = [system, *history, {"role": "user", "content": question}]
                self.assertTrue(isolate_mcp_history(messages))
                self.assertEqual(len(messages), 2)
                self.assertNotIn("PRIVATE", str(messages))
                self.assertEqual(system, {"role": "system", "content": "sys"})
                self.assertEqual(len(history), 1)

    def test_non_user_and_multimodal_requests_are_not_isolated(self):
        history = [{"role": "user", "content": wrap_untrusted("document", "PRIVATE MATERIAL")}]
        for current in (
            {"role": "tool", "content": QUESTION},
            {"role": "assistant", "content": QUESTION},
            {"role": "user", "content": [{"type": "text", "text": QUESTION}]},
        ):
            with self.subTest(current=current):
                messages = [*history, current]
                self.assertFalse(isolate_mcp_history(messages))
                self.assertEqual(messages, [*history, current])
        self.assertFalse(isolate_mcp_history([]))
