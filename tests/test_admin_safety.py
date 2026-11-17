"""Security-critical /admin actions never happen from a single accidental tap.

Promoting admins, blocking accounts and granting or revoking access require the
typed ``CONFIRM`` word; reversible per-user settings use an explicit Apply step.
"""

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

OWNER, USER, CHANNEL = 1, 3, -1001234567890
NOW = datetime.datetime(2026, 10, 28, 12, 0, tzinfo=datetime.UTC)


def _dm(text: str) -> Message:
    return Message(
        message_id=1,
        date=NOW,
        chat=Chat(id=OWNER, type="private"),
        from_user=User(id=OWNER, is_bot=False, first_name="Ann"),
        text=text,
    )


def _callback(data: str) -> SimpleNamespace:
    return SimpleNamespace(
        data=data,
        from_user=SimpleNamespace(id=OWNER),
        message=SimpleNamespace(chat=SimpleNamespace(id=OWNER, type="private")),
    )


class TestAdminSafety(DBTest):
    def _run(self, coro_fn):
        self._show = AsyncMock()
        patches = [
            patch("bot.database.session.async_session_factory", self._factory),
            patch("bot.services.digest.async_session_factory", self._factory),
            patch.object(config, "owner_ids", [OWNER]),
            patch.object(config, "extra_admin_ids", set()),
            patch.object(config, "daily_limit", 3),
            patch.object(panel, "_show", self._show),
            patch.object(panel, "safe_answer_callback", AsyncMock()),
            patch.object(panel, "_fetch_profile", AsyncMock(return_value={})),
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
        await panel.cb_admin(_callback(data))

    async def _type(self, text: str) -> None:
        message = _dm(text)
        self.assertTrue(panel._expecting_input(message))
        await panel.on_admin_input(message)

    async def _register(self, *, allowed: bool = True) -> None:
        async with user_service() as service:
            await service.register_user(USER)
            if allowed:
                await service.add_allowed_user(USER)

    def test_admin_role_requires_the_exact_confirm_word(self):
        async def go() -> None:
            await self._register()

            await self._press(f"adm:adm:{USER}")
            self.assertEqual(panel._awaiting[OWNER], ("confirm:admin_on", USER))
            self.assertNotIn(USER, config.extra_admin_ids)
            await self._type("confirm")
            self.assertNotIn(USER, config.extra_admin_ids)
            async with user_service() as service:
                self.assertNotIn(USER, await service.user_repo.get_granted_admin_ids())

            await self._press(f"adm:adm:{USER}")
            await self._type("CONFIRM")
            self.assertIn(USER, config.extra_admin_ids)
            async with user_service() as service:
                self.assertIn(USER, await service.user_repo.get_granted_admin_ids())

            await self._press(f"adm:adm:{USER}")
            self.assertEqual(panel._awaiting[OWNER], ("confirm:admin_off", USER))
            await self._type("CONFIRM")
            self.assertNotIn(USER, config.extra_admin_ids)

        self._run(go)

    def test_access_and_blocking_require_the_exact_confirm_word(self):
        async def go() -> None:
            await self._register(allowed=False)

            await self._press(f"adm:grant:{USER}")
            self.assertEqual(panel._awaiting[OWNER], ("confirm:grant", USER))
            await self._type("yes")
            async with user_service() as service:
                self.assertFalse(await service.user_repo.is_allowed_user(USER))

            await self._press(f"adm:grant:{USER}")
            await self._type("CONFIRM")
            async with user_service() as service:
                self.assertTrue(await service.user_repo.is_allowed_user(USER))

            await self._press(f"adm:revoke:{USER}")
            self.assertEqual(panel._awaiting[OWNER], ("confirm:revoke", USER))
            await self._type("CONFIRM")
            async with user_service() as service:
                self.assertFalse(await service.user_repo.is_allowed_user(USER))

            await self._press(f"adm:blk:{USER}")
            self.assertEqual(panel._awaiting[OWNER], ("confirm:block_on", USER))
            await self._type("no")
            async with user_service() as service:
                self.assertFalse(await service.user_repo.is_blocked(USER))

            await self._press(f"adm:blk:{USER}")
            await self._type("CONFIRM")
            async with user_service() as service:
                self.assertTrue(await service.user_repo.is_blocked(USER))

            await self._press(f"adm:blk:{USER}")
            await self._type("CONFIRM")
            async with user_service() as service:
                self.assertFalse(await service.user_repo.is_blocked(USER))

        self._run(go)

    def test_settings_apply_only_after_the_apply_button(self):
        async def go() -> None:
            await self._register()
            async with user_service() as service:
                await service.set_daily_limit(USER, 10)
                await service.user_repo.try_consume_daily_usage(
                    USER, datetime.date.today(), 100
                )
                await service.session.commit()

            await self._press(f"adm:lim:{USER}:5")
            async with user_service() as service:
                self.assertEqual(await service.get_daily_limit(USER), 10)
            markup = self._show.await_args.args[2]
            apply_data = {
                button.callback_data
                for row in markup.inline_keyboard
                for button in row
            }
            self.assertIn(f"adm:ok:limv:{USER}:15", apply_data)

            await self._press(f"adm:ok:limv:{USER}:15")
            async with user_service() as service:
                self.assertEqual(await service.get_daily_limit(USER), 15)

            used = 1
            await self._press(f"adm:rst:{USER}")
            async with user_service() as service:
                self.assertEqual(await service.get_daily_usage_used(USER), used)

            await self._press(f"adm:ok:rst:{USER}")
            async with user_service() as service:
                self.assertEqual(await service.get_daily_usage_used(USER), 0)

            current = config.ai_models[0]["id"]
            other = config.ai_models[1]["id"]
            async with user_service() as service:
                await service.set_model(USER, current)

            await self._press(f"adm:mset:{USER}:1")
            async with user_service() as service:
                self.assertEqual((await service.get_user_settings(USER))["model"], current)

            await self._press(f"adm:ok:mset:{USER}:1")
            async with user_service() as service:
                self.assertEqual((await service.get_user_settings(USER))["model"], other)

        self._run(go)

    def test_another_button_cancels_a_pending_confirm(self):
        async def go() -> None:
            await self._register()

            await self._press(f"adm:adm:{USER}")
            self.assertIn(OWNER, panel._awaiting)

            # cb_admin clears the prompt on any other button, so a later
            # "CONFIRM" cannot apply a request the admin navigated away from.
            await self._press("adm:home")
            self.assertNotIn(OWNER, panel._awaiting)
            self.assertFalse(panel._expecting_input(_dm("CONFIRM")))
            self.assertNotIn(USER, config.extra_admin_ids)

        self._run(go)

    def test_admin_callbacks_are_refused_outside_a_private_chat(self):
        async def go() -> None:
            await self._register()
            callback = _callback(f"adm:nm:{USER}")
            callback.message.chat.type = "group"
            await panel.cb_admin(callback)
            self.assertNotIn(OWNER, panel._awaiting)

        self._run(go)

    def test_digest_channel_removal_requires_the_exact_confirm_word(self):
        async def go() -> None:
            await digest.add_channel(CHANNEL, "Test channel", OWNER)

            await self._press(f"adm:dgx:{CHANNEL}")
            self.assertEqual(panel._awaiting[OWNER], ("confirm:dgx", CHANNEL))
            self.assertEqual(dict(await digest.list_channels())[CHANNEL], "Test channel")

            await self._type("delete")
            self.assertIn(CHANNEL, dict(await digest.list_channels()))

            await self._press(f"adm:dgx:{CHANNEL}")
            await self._type("CONFIRM")
            self.assertNotIn(CHANNEL, dict(await digest.list_channels()))

        self._run(go)