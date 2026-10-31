"""M-4: общий лимит tool calls и контекста на один запрос."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from bot.config import config
from bot.handlers import chat
from bot.services import reminders
from bot.services.access import AccessResult
from bot.services.ai_client import ChatResult, ToolCall
from bot.services.tool_budget import (
    CALL_LIMIT_NOTICE,
    CONTEXT_LIMIT_NOTICE,
    ToolBudget,
    clip_tool_text,
)
from bot.services.untrusted import has_untrusted_content, wrap_untrusted


class ClipToolTextTests(unittest.TestCase):
    def test_short_text_is_unchanged_and_budget_is_charged(self):
        budget = ToolBudget(search_calls=1, mcp_calls=1, context_chars=100)
        self.assertEqual(budget.clip("hello"), "hello")
        self.assertEqual(budget.context_chars, 95)

    def test_cut_inside_frame_keeps_one_closing_tag(self):
        wrapped = wrap_untrusted("search", "x" * 500)
        clipped = clip_tool_text(wrapped, 80)
        self.assertLessEqual(len(clipped), 80)
        self.assertTrue(has_untrusted_content(clipped))
        self.assertEqual(clipped.count("</untrusted_content>"), 1)
        self.assertIn("обрезан", clipped)

    def test_cut_after_frame_does_not_add_a_closing_tag(self):
        text = f"{wrap_untrusted('mcp', 'short')}\n\ntrusted suffix"
        clipped = clip_tool_text(text, 50)
        self.assertEqual(clipped.count("</untrusted_content>"), 1)

    def test_no_budget_returns_a_notice_instead_of_a_broken_frame(self):
        budget = ToolBudget(search_calls=1, mcp_calls=1, context_chars=5)
        self.assertEqual(budget.clip("x" * 100), CONTEXT_LIMIT_NOTICE)
        self.assertEqual(budget.context_chars, 0)
        self.assertFalse(budget.search_left)
        self.assertFalse(budget.mcp_left)

    def test_call_limits_are_independent(self):
        budget = ToolBudget(search_calls=2, mcp_calls=1, context_chars=1000)
        self.assertEqual([budget.take_search() for _ in range(3)], [True, True, False])
        self.assertEqual([budget.take_mcp() for _ in range(2)], [True, False])
        self.assertFalse(budget.search_left)
        self.assertFalse(budget.mcp_left)


class ResolveRoundBudgetTests(unittest.IsolatedAsyncioTestCase):
    def message(self):
        return SimpleNamespace(chat=SimpleNamespace(id=1))

    async def test_seven_search_calls_in_one_round_stop_at_the_budget(self):
        calls = [
            ToolCall(
                id=f"c{i}", name="search_web", arguments={"query": f"q{i}"}
            )
            for i in range(7)
        ]
        messages = []
        budget = ToolBudget(search_calls=2, mcp_calls=1, context_chars=10_000)
        with (
            patch.object(chat, "search_web", AsyncMock(return_value=[])) as web,
            patch.object(chat, "format_search_results", return_value="result"),
            patch.object(chat, "safe_answer", AsyncMock()),
        ):
            await chat._resolve_tool_round(
                self.message(),
                messages,
                ChatResult(content=None, tool_calls=calls),
                calls,
                search_ready=True,
                search_allowed=True,
                mcp_allowed=False,
                budget=budget,
            )
        self.assertEqual(web.await_count, 2)
        tool_messages = [m for m in messages if m["role"] == "tool"]
        self.assertEqual(len(tool_messages), 7)
        self.assertEqual(
            sum(CALL_LIMIT_NOTICE in m["content"] for m in tool_messages), 5
        )
        self.assertFalse(budget.search_left)

    async def test_search_context_budget_can_stop_a_second_call(self):
        calls = [
            ToolCall(id=f"c{i}", name="search_web", arguments={"query": "q"})
            for i in range(3)
        ]
        messages = []
        budget = ToolBudget(search_calls=3, mcp_calls=1, context_chars=100)
        with (
            patch.object(chat, "search_web", AsyncMock(return_value=[])) as web,
            patch.object(chat, "format_search_results", return_value="x" * 500),
            patch.object(chat, "safe_answer", AsyncMock()),
        ):
            await chat._resolve_tool_round(
                self.message(),
                messages,
                ChatResult(content=None, tool_calls=calls),
                calls,
                search_ready=True,
                search_allowed=True,
                mcp_allowed=False,
                budget=budget,
            )
        self.assertEqual(web.await_count, 1)
        self.assertEqual(budget.context_chars, 0)
        self.assertLessEqual(len(messages[-3]["content"]), 100)
        self.assertIn("обрезан", messages[-3]["content"])
        self.assertIn(CALL_LIMIT_NOTICE, messages[-1]["content"])

    async def test_mcp_calls_share_the_total_context_budget(self):
        calls = [
            ToolCall(id=f"m{i}", name="mcp__brain__get_note", arguments={})
            for i in range(5)
        ]
        messages = []
        budget = ToolBudget(search_calls=1, mcp_calls=2, context_chars=5_000)
        output = wrap_untrusted("mcp", "y" * 1_000)
        with (
            patch.object(chat.mcp_manager, "is_mcp_tool", return_value=True),
            patch.object(
                chat.mcp_manager, "call_tool", AsyncMock(return_value=output)
            ) as call,
            patch.object(chat, "safe_answer", AsyncMock()),
        ):
            await chat._resolve_tool_round(
                self.message(),
                messages,
                ChatResult(content=None, tool_calls=calls),
                calls,
                search_ready=False,
                search_allowed=False,
                mcp_allowed=True,
                budget=budget,
            )
        self.assertEqual(call.await_count, 2)
        self.assertTrue(all(c.kwargs["max_chars"] <= 5_000 for c in call.await_args_list))
        tool_messages = [m for m in messages if m["role"] == "tool"]
        self.assertEqual(len(tool_messages), 5)
        self.assertEqual(
            sum(CALL_LIMIT_NOTICE in m["content"] for m in tool_messages), 3
        )


class AnswerLoopBudgetTests(unittest.IsolatedAsyncioTestCase):
    async def test_answer_with_ai_limits_calls_across_the_round(self):
        message = SimpleNamespace(
            chat=SimpleNamespace(id=1, type="private"),
            from_user=SimpleNamespace(id=1),
            bot=SimpleNamespace(send_chat_action=AsyncMock()),
            answer=AsyncMock(),
        )
        service = SimpleNamespace(
            get_user_settings=AsyncMock(return_value={
                "model": "m", "system_prompt": "sys",
                "temperature": 0.5, "max_tokens": 100,
            }),
            get_history=AsyncMock(return_value=[]),
            add_message=AsyncMock(),
        )
        calls = [
            ToolCall(id=f"c{i}", name="search_web", arguments={"query": "q"})
            for i in range(7)
        ]
        first = ChatResult(content=None, tool_calls=calls, elapsed=1.0)
        final = ChatResult(content="answer", elapsed=1.0)
        with (
            patch.object(config, "stream_responses", False),
            patch.object(config, "enable_image_generation", False),
            patch.object(config, "enable_sticker_generation", False),
            patch.object(config, "enable_file_replies", False),
            patch.object(config, "enable_web_search", True),
            patch.object(config, "search_provider", "tavily"),
            patch.object(config, "search_api_key", "tvly-test"),
            patch.object(config, "max_search_rounds", 2),
            patch.object(config, "search_max_calls", 2),
            patch.object(chat, "waiting", _no_wait),
            patch.object(chat, "_single_shot", AsyncMock(side_effect=[first, final])),
            patch.object(chat, "search_web", AsyncMock(return_value=[])) as web,
            patch.object(chat, "deliver_answer", AsyncMock(return_value=True)),
            patch.object(chat, "safe_answer", AsyncMock()),
            patch.object(chat, "guest_hint", return_value=None),
            patch.object(chat, "_retry_markup", return_value=None),
        ):
            await chat._answer_with_ai(
                message, service, "seven searches", AccessResult(True, "admin")
            )
        self.assertEqual(web.await_count, 2)


class ReminderToolBudgetTests(unittest.IsolatedAsyncioTestCase):
    async def test_scheduled_run_also_limits_calls_in_one_round(self):
        due = reminders.DueReminder(
            id=1, user_id=1, chat_id=1, text="question",
            ai_mode=True, schedule="every:60",
        )
        settings = {
            "model": "m", "system_prompt": "sys",
            "temperature": 0.5, "max_tokens": 100,
        }
        calls = [
            ToolCall(id=f"c{i}", name="search_web", arguments={"query": "q"})
            for i in range(7)
        ]
        first = ChatResult(content=None, tool_calls=calls, elapsed=1.0)
        final = ChatResult(content="done", elapsed=1.0)
        with (
            patch.object(reminders, "search_configured", return_value=True),
            patch.object(config, "search_max_calls", 2),
            patch.object(
                reminders.ai_client, "chat_with_tools",
                AsyncMock(side_effect=[first, final]),
            ),
            patch.object(reminders, "search_web", AsyncMock(return_value=[])) as web,
            patch.object(reminders, "format_search_results", return_value="result"),
        ):
            result, _elapsed = await reminders._run_ai_prompt(due, settings)
        self.assertEqual(result.content, "done")
        self.assertEqual(web.await_count, 2)


class _NoWait:
    async def __aenter__(self):
        return SimpleNamespace()

    async def __aexit__(self, *exc):
        return False


def _no_wait(*args, **kwargs):
    return _NoWait()