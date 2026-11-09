"""Регрессия: «нашёл, извлекаю» не становится финальным ответом по заметкам."""

from __future__ import annotations

import copy
import json
import unittest
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from bot.config import config
from bot.handlers import chat, guest
from bot.services.access import AccessResult
from bot.services.ai_client import ChatResult, ToolCall
from bot.services.mcp import MCPManager, MCPTool
from bot.services.retrieval_answer import (
    RETRIEVAL_FINAL_HINT,
    is_retrieval_promise,
    retrieval_fallback,
)

PROMISE = "Нашёл точное совпадение. Извлекаю полный текст заметки."
QUESTION = "Найди в моих заметках план проекта и проанализируй его"
NOTE = "Полный план: сначала прототип, затем испытания. Риск: сроки интеграции."


@asynccontextmanager
async def no_wait(*args, **kwargs):
    yield SimpleNamespace()


def fixtures():
    async def backend_call(name, arguments):
        if name == "search_notes":
            return json.dumps([{"path": "Vault/Plan.md", "summary": "Короткая карточка"}])
        return NOTE

    backend = SimpleNamespace(call_tool=AsyncMock(side_effect=backend_call))
    manager = MCPManager()
    manager._loaded = True
    for name, parameter in (("search_notes", "query"), ("get_note", "path")):
        tool = MCPTool("brain", name, name, {
            "type": "object", "properties": {parameter: {"type": "string"}},
        }, f"mcp__brain__{name}")
        manager._tools[tool.exposed_name] = (tool, backend)
    svc = SimpleNamespace(
        register_user=AsyncMock(),
        get_user_settings=AsyncMock(return_value={
            "model": "m", "system_prompt": "sys", "temperature": 0.5, "max_tokens": 100,
        }),
        get_history=AsyncMock(return_value=[]),
        add_message=AsyncMock(return_value=SimpleNamespace(id=1)),
    )
    caller = SimpleNamespace(id=1, username="u", first_name="U", last_name=None)
    message = SimpleNamespace(
        text=QUESTION, caption=None, photo=[], voice=None, document=None,
        forward_date=None, forward_origin=None, reply_to_message=None,
        chat=SimpleNamespace(id=1, type="private"), from_user=caller, answer=AsyncMock(),
    )
    return manager, backend, svc, message


class RetrievalPromiseTests(unittest.TestCase):
    def test_promise_is_not_an_analysis(self):
        self.assertTrue(is_retrieval_promise(PROMISE))
        self.assertTrue(is_retrieval_promise("Found a match. I will retrieve the full note."))
        self.assertFalse(is_retrieval_promise("По Vault/Plan.md: сначала прототип. Риск — сроки."))
        self.assertFalse(is_retrieval_promise("Прочитаю текст.\n- Содержательный вывод: сроки сдвинулись."))

    def test_fallback_shows_source_but_does_not_pretend_to_have_analyzed_it(self):
        messages = [{"role": "tool", "content": f'<untrusted_content source="mcp">{NOTE}</untrusted_content>'}]
        result = retrieval_fallback(messages)
        self.assertIn(NOTE, result)
        self.assertIn("could not generate a complete analysis", result)


class NoteAnswerIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def run_chat(self, *, streaming=False, initial_promise=False, repeated_promise=False):
        manager, backend, service, message = fixtures()
        seen = []

        async def answer(messages, *args, tools=None, **kwargs):
            catalog = args[2] if len(args) == 3 else tools
            seen.append({"messages": copy.deepcopy(messages), "tools": catalog})
            if len(seen) == 1:
                return ChatResult(content=PROMISE, tool_calls=[] if initial_promise else [ToolCall(
                    id="s1", name="mcp__brain__search_notes", arguments={"query": "план"},
                )])
            if len(seen) == 2 or repeated_promise:
                return ChatResult(content=PROMISE)
            return ChatResult(content="По Vault/Plan.md: прототип перед испытаниями; риск — сроки интеграции.")

        with (
            patch.multiple(config, stream_responses=streaming, request_intent=False,
                           enable_image_generation=False, enable_sticker_generation=False,
                           enable_file_replies=False, mcp_max_calls=4),
            patch.object(chat, "mcp_available", return_value=True),
            patch.object(chat, "mcp_manager", manager),
            patch.object(chat, "search_configured", return_value=False),
            patch.object(chat, "_single_shot", side_effect=answer),
            patch.object(chat.ai_client, "stream_chat_completion", side_effect=answer),
            patch.object(chat, "_StreamPrinter", return_value=SimpleNamespace(on_delta=AsyncMock(), discard=AsyncMock())),
            patch.object(chat, "waiting", no_wait),
            patch.object(chat, "safe_answer", AsyncMock()),
            patch.object(chat, "deliver_answer", AsyncMock(return_value=True)) as deliver,
            patch.object(chat, "guest_hint", return_value=None),
            patch.object(chat, "_retry_markup", return_value=None),
        ):
            await chat._answer_with_ai(message, service, QUESTION, AccessResult(True, "user"))
        return backend, seen, deliver.await_args.args[2], service

    async def test_search_full_read_analysis_and_one_bounded_final_retry(self):
        for streaming in (False, True):
            with self.subTest(streaming=streaming):
                backend, seen, text, service = await self.run_chat(streaming=streaming)
                self.assertEqual([c.args[0] for c in backend.call_tool.await_args_list], ["search_notes", "get_note"])
                self.assertIn(NOTE, str(seen[1]["messages"]))
                self.assertIsNone(seen[1]["tools"])
                self.assertIn(RETRIEVAL_FINAL_HINT, seen[2]["messages"][0]["content"])
                self.assertNotEqual(text, PROMISE)
                self.assertIn("риск", text)
                self.assertEqual(len(seen), 3)
                saved = [c.kwargs["content"] for c in service.add_message.await_args_list]
                self.assertNotIn(PROMISE, saved)
                self.assertNotIn(NOTE, saved)

    async def test_initial_plain_promise_is_replaced_by_real_note_lookup(self):
        backend, _seen, text, _service = await self.run_chat(initial_promise=True)
        self.assertEqual(backend.call_tool.await_count, 2)
        self.assertIn("Vault/Plan.md", text)
        self.assertEqual(backend.call_tool.await_args_list[0].args[1], {"query": QUESTION})

    async def test_repeated_promise_ends_with_actual_excerpts_not_another_promise(self):
        _backend, seen, text, _service = await self.run_chat(repeated_promise=True)
        self.assertEqual(len(seen), 3)
        self.assertIn(NOTE, text)
        self.assertIn("could not generate", text)

    async def test_guest_uses_the_same_full_text_and_completion_guard(self):
        manager, backend, service, message = fixtures()
        seen = []

        async def answer(**kwargs):
            seen.append(copy.deepcopy(kwargs))
            if len(seen) == 1:
                return ChatResult(tool_calls=[ToolCall(
                    id="s1", name="mcp__brain__search_notes", arguments={"query": "план"},
                )])
            if len(seen) == 2:
                return ChatResult(content=PROMISE)
            return ChatResult(content="По Vault/Plan.md: прототип, испытания, риск сроков.")

        with (
            patch.multiple(config, enable_image_generation=False, enable_link_analysis=False),
            patch.object(guest, "check_access", AsyncMock(return_value=AccessResult(True, "guest"))),
            patch.object(guest, "mcp_available", return_value=True),
            patch.object(guest, "mcp_manager", manager),
            patch.object(guest.ai_client, "chat_with_tools", side_effect=answer),
            patch.object(guest.ai_client, "chat_completion_result", side_effect=answer),
            patch.object(guest, "_maybe_telegraph_reply", AsyncMock(return_value=None)) as telegraph,
        ):
            reply = await guest._build_reply(message, service, message.from_user, 42)
        self.assertEqual(backend.call_tool.await_count, 2)
        self.assertIn(NOTE, str(seen[1]))
        self.assertNotIn(PROMISE, reply.text)
        self.assertIn("риск", reply.text)
        # Даже без внешней истории MCP-текст нельзя публиковать в Telegraph.
        self.assertFalse(telegraph.await_args.kwargs["allow"])
