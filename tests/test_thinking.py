"""Global thinking settings: payload mapping, persistence and owner access."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from bot.config import config
from bot.database.repository import AppSettingRepository
from bot.handlers import admin_panel as panel
from bot.services import thinking
from bot.services.ai_client import AIClient
from tests.test_database import DBTest


class TestThinking(DBTest):
    def setUp(self) -> None:
        super().setUp()
        thinking.reset_cache()
        self.addCleanup(thinking.reset_cache)

    def _run(self, coro_fn):
        return self._loop.run_until_complete(coro_fn())

    def test_parse_level_and_budget(self):
        self.assertEqual(thinking.parse_level(" LOW "), "low")
        self.assertIsNone(thinking.parse_level("auto"))
        self.assertEqual(thinking.parse_budget("0"), 0)
        self.assertEqual(
            thinking.parse_budget(thinking.MAX_THINKING_BUDGET),
            thinking.MAX_THINKING_BUDGET,
        )
        self.assertIsNone(thinking.parse_budget("-1"))
        self.assertIsNone(thinking.parse_budget("many"))

    def test_payload_fields_off_level_and_budget(self):
        cases = [
            (thinking.ThinkingSettings("off", 0), {"enable_thinking": False}),
            (
                thinking.ThinkingSettings("off", 8192),
                {"enable_thinking": False},
            ),
            (
                thinking.ThinkingSettings("medium", 0),
                {"enable_thinking": True, "reasoning_effort": "medium"},
            ),
            (
                thinking.ThinkingSettings("low", 4096),
                {"enable_thinking": True, "thinking_budget": 4096},
            ),
        ]
        for settings, expected in cases:
            with patch.object(thinking, "_current", settings):
                self.assertEqual(thinking.payload_fields(), expected)

    def test_public_flag_only_adds_thinking_to_public_payloads(self):
        client = AIClient(api_key="test-key")
        settings = thinking.ThinkingSettings("medium", 0)
        with patch.object(thinking, "_current", settings):
            public = client._chat_payload([], None, 0.5, 100, public=True)
            internal = client._chat_payload([], None, 0.5, 100)
        self.assertEqual(public["reasoning_effort"], "medium")
        self.assertTrue(public["enable_thinking"])
        self.assertNotIn("enable_thinking", internal)
        self.assertNotIn("reasoning_effort", internal)

    def test_400_retries_without_thinking_and_remembers_the_model(self):
        client = AIClient(api_key="test-key")
        ok = {"choices": [{"message": {"content": "answer"}}]}
        settings = thinking.ThinkingSettings("medium", 0)
        with (
            patch.object(thinking, "_current", settings),
            patch.object(client, "model_chain", return_value=["m1"]),
            patch.object(
                client,
                "_post_chat",
                AsyncMock(side_effect=[(400, None, 0.1), (200, ok, 0.2)]),
            ) as request,
        ):
            result = self._run(
                lambda: client.chat_completion_result(
                    [{"role": "user", "content": "hi"}], public=True
                )
            )

        self.assertEqual(result.content, "answer")
        self.assertIn("enable_thinking", request.await_args_list[0].args[0])
        self.assertNotIn("enable_thinking", request.await_args_list[1].args[0])
        self.assertIn("m1", client._thinking_unsupported)

        # The next public request skips the rejected fields from the start.
        with (
            patch.object(thinking, "_current", settings),
            patch.object(client, "model_chain", return_value=["m1"]),
            patch.object(
                client, "_post_chat", AsyncMock(return_value=(200, ok, 0.1))
            ) as request,
        ):
            self._run(
                lambda: client.chat_completion_result(
                    [{"role": "user", "content": "again"}], public=True
                )
            )
        self.assertNotIn("enable_thinking", request.await_args.args[0])

    def test_load_set_reload_and_reset(self):
        async def go():
            await thinking.load()
            current = thinking.current()
            self.assertEqual((current.level, current.budget), ("medium", 0))
            self.assertFalse(current.level_custom or current.budget_custom)

            await thinking.set_level("low")
            await thinking.set_budget(4096)
            current = thinking.current()
            self.assertEqual((current.level, current.budget), ("low", 4096))
            self.assertTrue(current.level_custom and current.budget_custom)

            thinking.reset_cache()
            await thinking.load()
            current = thinking.current()
            self.assertEqual((current.level, current.budget), ("low", 4096))

            await thinking.reset()
            current = thinking.current()
            self.assertEqual((current.level, current.budget), ("medium", 0))
            self.assertFalse(current.level_custom or current.budget_custom)

        with (
            patch("bot.services.thinking.async_session_factory", self._factory),
            patch.object(config, "thinking_level", "medium"),
            patch.object(config, "thinking_budget", 0),
        ):
            self._run(go)

    def test_invalid_stored_values_fall_back_to_env(self):
        async def go():
            async with self._factory() as session:
                repo = AppSettingRepository(session)
                await repo.set_setting(thinking.SETTING_LEVEL, "bogus")
                await repo.set_setting(thinking.SETTING_BUDGET, "999999999")
                await session.commit()
            await thinking.load()
            current = thinking.current()
            self.assertEqual((current.level, current.budget), ("medium", 0))
            self.assertFalse(current.level_custom or current.budget_custom)

        with (
            patch("bot.services.thinking.async_session_factory", self._factory),
            patch.object(config, "thinking_level", "medium"),
            patch.object(config, "thinking_budget", 0),
        ):
            self._run(go)

    def test_budget_input_and_reset_from_the_admin_panel(self):
        async def go():
            await thinking.set_level("low")
            message = SimpleNamespace(answer=AsyncMock())
            await panel._thinking_input(message, 1, "4096")
            self.assertEqual(thinking.current().budget, 4096)
            self.assertEqual(message.answer.await_count, 1)

            await panel._confirm_reset(message, 1, "thr", None, "CONFIRM")
            self.assertEqual(thinking.current().level, "medium")
            self.assertEqual(thinking.current().budget, 0)
            self.assertEqual(message.answer.await_count, 2)

        with (
            patch("bot.services.thinking.async_session_factory", self._factory),
            patch.object(config, "owner_ids", [1]),
            patch.object(config, "extra_admin_ids", set()),
            patch.object(config, "thinking_level", "medium"),
            patch.object(config, "thinking_budget", 0),
        ):
            self._run(go)

    def test_home_button_is_owner_only(self):
        with (
            patch.object(config, "owner_ids", [1]),
            patch.object(config, "extra_admin_ids", {2}),
        ):
            owner = {
                button.callback_data
                for row in panel._home_markup(1).inline_keyboard
                for button in row
            }
            admin = {
                button.callback_data
                for row in panel._home_markup(2).inline_keyboard
                for button in row
            }
        self.assertIn("adm:th", owner)
        self.assertNotIn("adm:th", admin)

    def test_non_owner_cannot_open_the_thinking_screen(self):
        callback = SimpleNamespace(from_user=SimpleNamespace(id=2))
        with (
            patch.object(config, "owner_ids", [1]),
            patch.object(config, "extra_admin_ids", {2}),
            patch.object(panel, "safe_answer_callback", AsyncMock()) as answer,
        ):
            self._run(lambda: panel._thinking_action(callback, 2, "th", ["adm", "th"]))
        answer.assert_awaited_once()
        self.assertIn("Owners only", answer.await_args.args[1])

    def test_level_callback_is_dispatched_without_int_parsing(self):
        callback = SimpleNamespace(
            data="adm:ths:low",
            from_user=SimpleNamespace(id=1),
            message=SimpleNamespace(chat=SimpleNamespace(id=1, type="private")),
            answer=AsyncMock(),
        )
        with (
            patch("bot.services.thinking.async_session_factory", self._factory),
            patch.object(panel, "_is_admin_cb", return_value=True),
            patch.object(panel, "_show", AsyncMock()) as show,
            patch.object(panel, "safe_answer_callback", AsyncMock()),
            patch.object(config, "owner_ids", [1]),
            patch.object(config, "extra_admin_ids", set()),
        ):
            self._run(lambda: panel.cb_admin(callback))
        self.assertEqual(thinking.current().level, "low")
        show.assert_awaited()

    def test_screen_shows_custom_level_and_budget(self):
        settings = thinking.ThinkingSettings("low", 4096, True, True)
        with patch.object(thinking, "_current", settings):
            text, markup = self._run(panel._thinking_screen)
        self.assertIn("Low", text)
        self.assertIn("4 096", text)
        self.assertIn("custom", text)
        callbacks = {
            button.callback_data for row in markup.inline_keyboard for button in row
        }
        self.assertIn("adm:thr", callbacks)
