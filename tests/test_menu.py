from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from aiogram.enums import ChatType
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.base import StorageKey
from aiogram.fsm.storage.memory import MemoryStorage

from bot.config import config
from bot.handlers import menu
from bot.keyboards import reply


def _labels(markup) -> list[str]:
    return [b.text for row in markup.keyboard for b in row]


def _message(text: str, user_id: int = 1):
    msg = SimpleNamespace(
        text=text,
        from_user=SimpleNamespace(id=user_id),
        chat=SimpleNamespace(id=user_id, type=ChatType.PRIVATE),
        forward_origin=None,
        forward_date=None,
    )
    msg.model_copy = lambda update: SimpleNamespace(**{**msg.__dict__, **update})
    return msg


def _state(user_id: int = 1) -> FSMContext:
    return FSMContext(
        MemoryStorage(), StorageKey(bot_id=1, chat_id=user_id, user_id=user_id)
    )


class KeyboardTests(unittest.TestCase):
    def test_layout_and_admin_button(self):
        user = _labels(reply.get_main_menu_keyboard(False))
        admin = _labels(reply.get_main_menu_keyboard(True))
        self.assertNotIn(reply.BTN_ADMIN, user)
        self.assertIn(reply.BTN_ADMIN, admin)
        for btn in (reply.BTN_MODELS, reply.BTN_PERSONA, reply.BTN_CLEAR, reply.BTN_HELP):
            self.assertIn(btn, user)

    def test_disabled_features_are_hidden(self):
        with (
            patch.object(config, "enable_image_generation", False),
            patch.object(config, "enable_reminders", False),
            patch.object(config, "enable_export", False),
        ):
            labels = _labels(reply.get_main_menu_keyboard(False))
        for btn in (reply.BTN_IMAGE, reply.BTN_REMINDERS, reply.BTN_EXPORT):
            self.assertNotIn(btn, labels)

    def test_every_button_is_routed(self):
        self.assertEqual(
            set(_labels(reply.get_main_menu_keyboard(True))) - reply.ALL_BUTTONS, set()
        )


class MenuForUserTests(unittest.IsolatedAsyncioTestCase):
    async def test_switched_off(self):
        with patch.object(config, "enable_reply_menu", False):
            self.assertIsNone(await reply.menu_for_user(1))

    async def test_admin_gets_menu_without_db(self):
        with (
            patch.object(config, "enable_reply_menu", True),
            patch.object(config, "is_admin", return_value=True),
        ):
            markup = await reply.menu_for_user(1)
        self.assertIn(reply.BTN_ADMIN, _labels(markup))


class DialogTests(unittest.IsolatedAsyncioTestCase):
    async def test_image_button_opens_dialog_then_generates(self):
        state = _state()
        msg = _message(reply.BTN_IMAGE)
        with (
            patch.object(config, "enable_image_generation", True),
            patch.object(menu, "safe_answer", AsyncMock()) as answer,
            patch.object(menu, "generate_image_from_prompt", AsyncMock()) as gen,
        ):
            await menu.on_menu_button(msg, state)
            self.assertEqual(await state.get_state(), menu.MenuDialog.image.state)
            answer.assert_awaited_once()

            await menu.on_image_prompt(_message("a red fox"), state)
            gen.assert_awaited_once()
            self.assertEqual(gen.await_args.args[1], "a red fox")
            self.assertIsNone(await state.get_state())

    async def test_cancel_word_stops_dialog(self):
        state = _state()
        await menu._start_dialog(state, menu.MenuDialog.image)
        with (
            patch.object(menu, "safe_answer", AsyncMock()),
            patch.object(menu, "generate_image_from_prompt", AsyncMock()) as gen,
        ):
            await menu.on_image_prompt(_message("Cancel"), state)
        gen.assert_not_awaited()
        self.assertIsNone(await state.get_state())

    async def test_expired_dialog_falls_through_to_chat(self):
        from aiogram.dispatcher.event.bases import SkipHandler

        state = _state()
        await menu._start_dialog(state, menu.MenuDialog.image)
        await state.update_data(started=-10_000.0)
        with self.assertRaises(SkipHandler):
            await menu.on_image_prompt(_message("hello"), state)
        self.assertIsNone(await state.get_state())

    async def test_remind_dialog_stays_open_on_parse_error(self):
        state = _state()
        await menu._start_dialog(state, menu.MenuDialog.remind)
        with patch.object(
            menu, "create_reminder_from_text", AsyncMock(return_value=False)
        ):
            await menu.on_remind_text(_message("garbage"), state)
        self.assertEqual(await state.get_state(), menu.MenuDialog.remind.state)

        with patch.object(
            menu, "create_reminder_from_text", AsyncMock(return_value=True)
        ):
            await menu.on_remind_text(_message("in 5 minutes tea"), state)
        self.assertIsNone(await state.get_state())

    async def test_command_in_dialog_clears_state_and_passes_on(self):
        from aiogram.dispatcher.event.bases import SkipHandler

        state = _state()
        await menu._start_dialog(state, menu.MenuDialog.remind)
        with self.assertRaises(SkipHandler):
            await menu.on_command_in_dialog(_message("/models"), state)
        self.assertIsNone(await state.get_state())

    async def test_simple_button_calls_command_handler_and_resets_state(self):
        state = _state()
        await menu._start_dialog(state, menu.MenuDialog.image)
        with patch.object(menu, "cmd_stats", AsyncMock()) as stats:
            await menu.on_menu_button(_message(reply.BTN_STATS), state)
        stats.assert_awaited_once()
        self.assertIsNone(await state.get_state())

    async def test_export_button_passes_plain_export_command(self):
        with patch.object(menu, "cmd_export", AsyncMock()) as export:
            await menu.on_menu_button(_message(reply.BTN_EXPORT), _state())
        self.assertEqual(export.await_args.args[0].text, "/export")


if __name__ == "__main__":
    unittest.main()
