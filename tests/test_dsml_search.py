"""Регрессии DSML-поиска DeepSeek в запланированных AI-запросах."""
from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from bot.config import config
from bot.services import reminders, thinking
from bot.services.ai_client import AIClient, ChatResult, ToolCall
from bot.services.search import SEARCH_TOOL


def dsml(query: str = "ИИ новости 1 ноября 2026") -> str:
    return (
        '<｜DSML｜calls><｜DSML｜invoke name="search_web">'
        f'<｜DSML｜parameter name="query" string="true">{query}'
        '</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜calls>'
    )


class DSMLSearchTests(unittest.IsolatedAsyncioTestCase):
    def test_recovers_parallel_searches_and_keeps_unicode_arguments(self):
        text = dsml().replace('</｜DSML｜calls>', '')
        text += dsml("AI news November 1 2026").replace('<｜DSML｜calls>', '')
        result = ChatResult(content=text)
        AIClient._recover_search_dsml(result, [SEARCH_TOOL])
        self.assertIsNone(result.content)
        self.assertEqual(len(result.tool_calls), 2)
        self.assertEqual(result.tool_calls[0].arguments["query"], "ИИ новости 1 ноября 2026")
        self.assertNotEqual(result.tool_calls[0].id, result.tool_calls[1].id)

    def test_native_tool_calls_are_not_duplicated(self):
        calls = [ToolCall(id="native", name="search_web", arguments={"query": "q"})]
        result = ChatResult(content=dsml(), tool_calls=calls)
        AIClient._recover_search_dsml(result, [SEARCH_TOOL])
        self.assertIs(result.tool_calls, calls)
        self.assertIsNone(result.content)

    def test_spaced_tags_and_typed_parameters(self):
        content = dsml("AI &amp; ML").replace(
            '</｜DSML｜invoke>',
            '<｜DSML｜parameter name="max_results" string="false">3'
            '</｜DSML｜parameter></｜DSML｜invoke>',
        ).replace('｜DSML｜', '｜DSML｜ ')
        result = ChatResult(content=content)
        AIClient._recover_search_dsml(result, [SEARCH_TOOL])
        self.assertEqual(result.tool_calls[0].arguments, {"query": "AI & ML", "max_results": 3})

    def test_does_not_execute_quotes_or_unknown_tools_or_disabled_search(self):
        for content, tools in [
            (dsml(), []),
            (dsml().replace('name="search_web"', 'name="send_files"'), [SEARCH_TOOL]),
            (dsml()[:-10], [SEARCH_TOOL]),
            (dsml().replace('string="true"', 'string="false"'), [SEARCH_TOOL]),
            (dsml("<!DOCTYPE x>"), [SEARCH_TOOL]),
        ]:
            with self.subTest(content=content):
                result = ChatResult(content=content)
                AIClient._recover_search_dsml(result, tools)
                self.assertFalse(result.tool_calls)
                self.assertIsNone(result.content)
        result = ChatResult(content="Пример синтаксиса: " + dsml())
        AIClient._recover_search_dsml(result, [SEARCH_TOOL])
        self.assertFalse(result.tool_calls)
        self.assertTrue(result.content)

    async def test_scheduled_run_executes_dsml_then_delivers_final_summary(self):
        for level in ("off", "medium"):
            with self.subTest(thinking=level):
                client = AIClient(api_key="test")
                response = {"choices": [{"message": {"content": dsml()}}]}
                final = {"choices": [{"message": {"content": "Итоговый обзор [1]"}}]}
                post = AsyncMock(side_effect=[(200, response, 0.1), (200, final, 0.1)])
                due = reminders.DueReminder(
                    id=1, user_id=1, chat_id=1, text="Проанализируй вчерашние статьи про ИИ",
                    ai_mode=True, schedule=None,
                )
                settings = {
                    "model": "m", "system_prompt": "sys", "temperature": 0.5,
                    "max_tokens": 100,
                }
                with (
                    patch.object(reminders, "ai_client", client),
                    patch.object(reminders, "search_configured", return_value=True),
                    patch.object(reminders, "search_web", AsyncMock(return_value=[])) as web,
                    patch.object(client, "model_chain", return_value=["m"]),
                    patch.object(client, "_post_chat", post),
                    patch.object(thinking, "_current", thinking.ThinkingSettings(level, 0)),
                ):
                    result, _ = await reminders._run_ai_prompt(due, settings)
                self.assertEqual(result.content, "Итоговый обзор [1]")
                web.assert_awaited_once_with("ИИ новости 1 ноября 2026", None)
                self.assertEqual(post.await_count, 2)
                messages = post.await_args_list[1].args[0]["messages"]
                self.assertEqual(messages[-1]["role"], "tool")
                self.assertEqual(messages[-2]["tool_calls"][0]["function"]["name"], "search_web")

    async def test_malformed_dsml_uses_fallback_instead_of_sending_markup(self):
        client = AIClient(api_key="test")
        bad = {"choices": [{"message": {"content": dsml()[:-10]}}]}
        good = {"choices": [{"message": {"content": "Ответ запасной модели"}}]}
        with (
            patch.object(client, "model_chain", return_value=["m1", "m2"]),
            patch.object(client, "_post_chat", AsyncMock(side_effect=[(200, bad, 0), (200, good, 0)])),
        ):
            result = await client.chat_with_tools([], tools=[SEARCH_TOOL])
        self.assertEqual(result.content, "Ответ запасной модели")
        self.assertEqual(result.model, "m2")

    async def test_recovered_calls_still_obey_reminder_search_budget(self):
        client = AIClient(api_key="test")
        content = dsml().replace('</｜DSML｜calls>', '')
        content += dsml("another query").replace('<｜DSML｜calls>', '')
        responses = [
            (200, {"choices": [{"message": {"content": content}}]}, 0),
            (200, {"choices": [{"message": {"content": "done"}}]}, 0),
        ]
        with (
            patch.object(reminders, "ai_client", client),
            patch.object(reminders, "search_configured", return_value=True),
            patch.object(reminders, "search_web", AsyncMock(return_value=[])) as web,
            patch.object(client, "model_chain", return_value=["m"]),
            patch.object(client, "_post_chat", AsyncMock(side_effect=responses)),
            patch.object(config, "search_max_calls", 1),
        ):
            result, _ = await reminders._run_ai_prompt(
                reminders.DueReminder(id=1, user_id=1, chat_id=1, text="news", ai_mode=True, schedule=None),
                {"model": "m", "system_prompt": "sys", "temperature": 0.5, "max_tokens": 100},
            )
        self.assertEqual(result.content, "done")
        self.assertEqual(web.await_count, 1)
