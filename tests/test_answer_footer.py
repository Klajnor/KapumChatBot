"""Итоговый футер не теряет поиск и reasoning первого tool-раунда."""

from __future__ import annotations

import unittest
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from bot.config import config
from bot.handlers import chat, guest
from bot.services import reminders
from bot.services.access import AccessResult
from bot.services.ai_client import ChatResult, ToolCall
from bot.services.response_metadata import AnswerMetadata, ResponseMetadata
from bot.services.search import SearchResults
from bot.services.tool_budget import ToolBudget


@asynccontextmanager
async def no_wait(*args, **kwargs):
    yield SimpleNamespace()


def setup():
    settings = {"model": "m", "system_prompt": "sys", "temperature": 0.5, "max_tokens": 100}
    service = SimpleNamespace(register_user=AsyncMock(), get_user_settings=AsyncMock(return_value=settings),
                              get_history=AsyncMock(return_value=[]),
                              add_message=AsyncMock(return_value=SimpleNamespace(id=1)))
    caller = SimpleNamespace(id=1, first_name="U", last_name=None, username="u")
    message = SimpleNamespace(chat=SimpleNamespace(id=1, type="private"), from_user=caller,
                              text="Find current news", caption=None, photo=[], voice=None, document=None,
                              reply_to_message=None, forward_origin=None, forward_date=None, answer=AsyncMock())
    responses = [
        ChatResult(tool_calls=[ToolCall("s1", "search_web", {"query": "news"})], elapsed=1,
                   metadata=ResponseMetadata(True, False, 2)),
        ChatResult(content="Final answer", elapsed=1, metadata=ResponseMetadata(False, False, 5)),
    ]
    return settings, service, message, responses


class ChatFooterTests(unittest.IsolatedAsyncioTestCase):
    async def test_chat_and_stream_report_both_searches_and_sum_reasoning(self):
        for streaming in (False, True):
            with self.subTest(streaming=streaming):
                _settings, service, message, responses = setup()
                with (
                    patch.multiple(config, stream_responses=streaming, request_intent=False, show_timing=True,
                                   enable_image_generation=False, enable_sticker_generation=False,
                                   enable_file_replies=False, timing_format_tokens="{model}"),
                    patch.object(config, "builtin_web_search", return_value=False),
                    patch.object(chat, "mcp_available", return_value=False),
                    patch.object(chat, "search_configured", return_value=True),
                    patch.object(chat, "_single_shot", AsyncMock(side_effect=responses)),
                    patch.object(chat.ai_client, "stream_chat_completion", AsyncMock(side_effect=responses)),
                    patch.object(chat, "_StreamPrinter", return_value=SimpleNamespace(on_delta=AsyncMock(), discard=AsyncMock())),
                    patch.object(chat, "search_web", AsyncMock(return_value=[])),
                    patch.object(chat, "waiting", no_wait),
                    patch.object(chat, "safe_answer", AsyncMock()),
                    patch.object(chat, "deliver_answer", AsyncMock(return_value=True)) as deliver,
                    patch.object(chat, "guest_hint", return_value=None),
                    patch.object(chat, "_retry_markup", return_value=None),
                ):
                    await chat._answer_with_ai(message, service, message.text, AccessResult(True, "user"))
                footer = deliver.await_args.args[3]
                self.assertIn("🔎 model + API", footer)
                self.assertIn("🧠 7 tok", footer)
                saved = [c.kwargs["content"] for c in service.add_message.await_args_list]
                self.assertEqual(saved, [message.text, "Final answer"])

    async def test_unexecuted_search_does_not_mark_api_as_used(self):
        _settings, _service, message, _responses = setup()
        for query, calls_left, expected in (("", 1, False), ("news", 0, False), ("news", 1, True)):
            with self.subTest(query=query, calls_left=calls_left):
                stats = AnswerMetadata()
                call = ToolCall("s1", "search_web", {"query": query})
                with (
                    patch.object(chat, "safe_answer", AsyncMock()),
                    patch.object(chat, "search_web", AsyncMock(return_value=SearchResults(error="timeout"))),
                ):
                    await chat._resolve_tool_round(
                        message, [], ChatResult(tool_calls=[call]), [call], search_ready=True,
                        search_allowed=True, mcp_allowed=False,
                        budget=ToolBudget(search_calls=calls_left), metadata=stats,
                    )
                self.assertEqual(stats.summary().api_search_used, expected)

    async def test_guest_footer_keeps_initial_round_metadata(self):
        _settings, service, message, responses = setup()
        with (
            patch.multiple(config, show_timing=True, timing_format_tokens="{model}",
                           enable_image_generation=False, enable_link_analysis=False),
            patch.object(config, "builtin_web_search", return_value=False),
            patch.object(guest, "check_access", AsyncMock(return_value=AccessResult(True, "guest"))),
            patch.object(guest, "mcp_available", return_value=False),
            patch.object(guest, "search_configured", return_value=True),
            patch.object(guest, "search_web", AsyncMock(return_value=[])),
            patch.object(guest.ai_client, "chat_with_tools", AsyncMock(side_effect=responses)),
            patch.object(guest, "_maybe_telegraph_reply", AsyncMock(return_value=None)),
        ):
            reply = await guest._build_reply(message, service, message.from_user, 42)
        self.assertIn("🔎 model + API", reply.text)
        self.assertIn("🧠 7 tok", reply.text)

    async def test_reminder_result_keeps_metrics_across_rounds(self):
        settings, _service, _message, responses = setup()
        with (
            patch.object(config, "builtin_web_search", return_value=False),
            patch.object(reminders, "search_configured", return_value=True),
            patch.object(reminders, "search_web", AsyncMock(return_value=[])),
            patch.object(reminders.ai_client, "chat_with_tools", AsyncMock(side_effect=responses)),
        ):
            result, _elapsed = await reminders._run_ai_prompt(
                reminders.DueReminder(id=1, user_id=1, chat_id=1, text="news", ai_mode=True, schedule=None),
                settings,
            )
        self.assertEqual(result.metadata, ResponseMetadata(True, True, 7))
