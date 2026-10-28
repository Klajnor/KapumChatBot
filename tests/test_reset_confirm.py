"""Reset-to-default buttons in /admin apply only after a typed CONFIRM."""

from __future__ import annotations

import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from aiogram.types import Chat, Message, User

from bot.config import config
from bot.handlers import admin_panel as panel
from bot.services import digest
from bot.services.user_service import user_service
from tests.test_database import DBTest

OWNER, USER = 1, 3
NOW = datetime.datetime(2026, 10, 28, 12, 0, tzinfo=datetime.UTC)


def _dm(text: str) -> Message:
    return Message(
        message_id=1,
        date=NOW,
        chat=Chat(id=OWNER, type="private"),
        from_user=User(id=OWNER, is_bot=False, first_name="Ann"),
        text=text,
    )


def _callback() -> SimpleNamespace:
    return SimpleNamespace(
        from_user=SimpleNamespace(id=OWNER),
        message=SimpleNamespace(chat=SimpleNamespace(id=OWNER, type="private")),
    )


class TestResetConfirm(DBTest):
    def _run(self, coro_fn):
        patches = [
            patch("bot.database.session.async_session_factory", self._factory),
            patch("bot.services.digest.async_session_factory", self._factory),
            patch.object(config, "owner_ids", [OWNER]),
            patch.object(config, "extra_admin_ids", set()),
            patch.object(config, "daily_limit", 3),
            patch.object(panel, "_show", AsyncMock()),
            patch.object(panel, "safe_answer_callback", AsyncMock()),
            patch.object(Message, "answer", AsyncMock()),
        ]
        for p in patches:
            p.start()
        panel._awaiting.clear()
        try:
            return self._loop.run_until_complete(coro_fn())
        finally:
            panel._awaiting.clear()
            for p in reversed(patches):
                p.stop()

    async def _press(self, data: str) -> None:
        parts = data.split(":")
        target = int(parts[2]) if parts[1] in ("lim", "rsp") else None
        if target is not None:
            await panel._user_action(_callback(), OWNER, parts[1], target, parts)
        else:
            await panel._digest_action(_callback(), OWNER, parts[1], parts)

    async def _type(self, text: str) -> None:
        message = _dm(text)
        self.assertTrue(panel._expecting_input(message))
        await panel.on_admin_input(message)

    def test_user_limit_and_prompt_need_exact_confirm(self):
        async def go() -> None:
            async with user_service() as s:
                await s.register_user(USER)
                await s.add_allowed_user(USER)
                await s.set_daily_limit(USER, 10)
                await s.set_system_prompt(USER, "custom")

            await self._press(f"adm:lim:{USER}:d")
            self.assertEqual(panel._awaiting[OWNER], ("reset:lim", USER))
            await self._type("confirm")  # wrong case
            self.assertNotIn(OWNER, panel._awaiting)
            async with user_service() as s:
                self.assertEqual(await s.get_daily_limit(USER), 10)

            await self._press(f"adm:lim:{USER}:d")
            await self._type("CONFIRM")
            async with user_service() as s:
                self.assertEqual(await s.get_daily_limit(USER), 3)

            await self._press(f"adm:rsp:{USER}")
            await self._type("yes")
            async with user_service() as s:
                self.assertEqual((await s.get_user_settings(USER))["system_prompt"], "custom")
            await self._press(f"adm:rsp:{USER}")
            await self._type("CONFIRM")
            async with user_service() as s:
                self.assertEqual(
                    (await s.get_user_settings(USER))["system_prompt"],
                    config.default_system_prompt,
                )

        self._run(go)

    def test_digest_settings_need_exact_confirm(self):
        async def go() -> None:
            for action, setting in (
                ("dgd", digest.SETTING_PROMPT),
                ("dgmd", digest.SETTING_MODIFIER),
                ("dgkd", digest.SETTING_TEMPERATURE),
                ("dgtd", digest.SETTING_TIME),
            ):
                await digest.set_setting(setting, "1")
                await self._press(f"adm:{action}")
                self.assertEqual(panel._awaiting[OWNER], (f"reset:{action}", None))
                await self._type("Confirm")
                self.assertEqual(await digest.get_setting(setting), "1")

                await self._press(f"adm:{action}")
                await self._type("CONFIRM")
                self.assertIsNone(await digest.get_setting(setting))

        self._run(go)

    def test_pressing_another_button_cancels_the_reset(self):
        async def go() -> None:
            await digest.set_setting(digest.SETTING_PROMPT, "custom")
            await self._press("adm:dgd")
            panel._awaiting.pop(OWNER)  # what cb_admin does on any other button
            self.assertFalse(panel._expecting_input(_dm("CONFIRM")))
            self.assertEqual(await digest.get_setting(digest.SETTING_PROMPT), "custom")

        self._run(go)
