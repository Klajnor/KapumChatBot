"""Заявка доступа из первого /start: уведомление админам и кнопки одобрить/блок.

Проверяются: отправка только на первом /start известного пользователя,
актуальность кнопок (повторное нажатие / чужое решение ничего не меняют),
гейт «только админ» и приватный чат.
"""

from __future__ import annotations

import datetime
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from bot.config import config
from bot.handlers import access_requests as requests_router
from bot.handlers import start as start_handler
from bot.services.user_service import user_service
from tests.test_database import DBTest

OWNER, ADMIN, USER = 1, 5, 42
NOW = datetime.datetime(2026, 11, 19, 12, 0, tzinfo=datetime.UTC)


def _dm(user_id: int) -> SimpleNamespace:
    """Message-заглушка для /start: answer и bot.send_message — моки."""
    from_user = SimpleNamespace(
        id=user_id, is_bot=False, first_name="Ann", last_name=None, username=None
    )
    return SimpleNamespace(
        from_user=from_user,
        chat=SimpleNamespace(id=user_id, type="private"),
        date=NOW,
        answer=AsyncMock(),
        bot=SimpleNamespace(send_message=AsyncMock()),
    )


def _callback(data: str, actor: int = OWNER) -> SimpleNamespace:
    return SimpleNamespace(
        data=data,
        from_user=SimpleNamespace(
            id=actor, is_bot=False, first_name="Ow", last_name=None, username=None
        ),
        message=SimpleNamespace(
            chat=SimpleNamespace(id=actor, type="private"),
            html_text="🆕 <b>New user started the bot</b>",
            edit_text=AsyncMock(),
        ),
        answer=AsyncMock(),
        bot=SimpleNamespace(send_message=AsyncMock()),
    )


class AccessRequestTests(DBTest):
    def _run(self, coro_fn):
        patches = [
            patch("bot.database.session.async_session_factory", self._factory),
            patch.object(config, "owner_ids", [OWNER]),
            patch.object(config, "extra_admin_ids", {ADMIN}),
            patch.object(config, "enable_reply_menu", False),
            patch.object(config, "enable_guest_mode", False),
        ]
        for p in patches:
            p.start()
        start_handler._recent_requests.clear()
        try:
            return self._loop.run_until_complete(coro_fn())
        finally:
            start_handler._recent_requests.clear()
            for p in reversed(patches):
                p.stop()

    # --- /start → заявка ------------------------------------------------------

    def test_first_start_notifies_admins_once(self):
        async def go() -> None:
            message = _dm(USER)
            await start_handler.cmd_start(message)
            sent = message.bot.send_message
            self.assertEqual(sent.await_count, 2)  # владелец + админ
            first_text = sent.await_args_list[0].args[1]
            self.assertIn("New user", first_text)
            self.assertIn(str(USER), first_text)
            first_markup = sent.await_args_list[0].kwargs["reply_markup"]
            rows = first_markup.inline_keyboard[0]
            self.assertEqual(
                [b.text for b in rows], ["✅ Approve", "🚫 Block"]
            )
            self.assertEqual(rows[0].callback_data, f"appr:a:{USER}")

            # Повторный /start известного пользователя — тишина.
            message2 = _dm(USER)
            message2.bot = SimpleNamespace(send_message=AsyncMock())
            await start_handler.cmd_start(message2)
            self.assertEqual(message2.bot.send_message.await_count, 0)

        self._run(go)

    def test_start_of_allowed_user_sends_no_request(self):
        async def go() -> None:
            async with user_service() as service:
                await service.register_user(USER)
                await service.add_allowed_user(USER)
            message = _dm(USER)
            message.bot = SimpleNamespace(send_message=AsyncMock())
            await start_handler.cmd_start(message)
            self.assertEqual(message.bot.send_message.await_count, 0)

        self._run(go)

    def test_admin_start_is_not_a_request(self):
        async def go() -> None:
            message = _dm(ADMIN)
            await start_handler.cmd_start(message)
            self.assertEqual(message.bot.send_message.await_count, 0)

        self._run(go)

    # --- кнопки ---------------------------------------------------------------

    async def _press(self, data: str, actor: int = OWNER) -> SimpleNamespace:
        callback = _callback(data, actor=actor)
        await requests_router.decide_access(callback)
        return callback

    def test_approve_adds_user_and_marks_message(self):
        async def go() -> None:
            callback = await self._press(f"appr:a:{USER}")
            async with user_service() as service:
                self.assertTrue(await service.is_allowed_user(USER))
                self.assertFalse(await service.is_blocked(USER))
            edited = callback.message.edit_text.await_args
            self.assertIn("Access granted", edited.args[0])
            self.assertEqual(edited.kwargs.get("reply_markup"), None)
            # Новичку сообщают, что доступ открыт.
            callback.bot.send_message.assert_awaited_once()
            self.assertIn("approved", callback.bot.send_message.await_args.args[1])

        self._run(go)

    def test_block_sets_blocked_and_no_user_message(self):
        async def go() -> None:
            callback = await self._press(f"appr:b:{USER}")
            async with user_service() as service:
                self.assertFalse(await service.is_allowed_user(USER))
                self.assertTrue(await service.is_blocked(USER))
            edited = callback.message.edit_text.await_args
            self.assertIn("Blocked", edited.args[0])
            callback.bot.send_message.assert_not_awaited()

        self._run(go)

    def test_stale_approve_after_block_changes_nothing(self):
        async def go() -> None:
            await self._press(f"appr:b:{USER}")
            callback = await self._press(f"appr:a:{USER}", actor=ADMIN)
            async with user_service() as service:
                self.assertFalse(await service.is_allowed_user(USER))
                self.assertTrue(await service.is_blocked(USER))
            edited = callback.message.edit_text.await_args
            self.assertIn("Already blocked", edited.args[0])

        self._run(go)

    def test_stale_block_after_approve_changes_nothing(self):
        async def go() -> None:
            await self._press(f"appr:a:{USER}")
            callback = await self._press(f"appr:b:{USER}", actor=ADMIN)
            async with user_service() as service:
                self.assertTrue(await service.is_allowed_user(USER))
                self.assertFalse(await service.is_blocked(USER))
            edited = callback.message.edit_text.await_args
            self.assertIn("Already approved", edited.args[0])

        self._run(go)

    def test_second_approve_is_a_noop(self):
        async def go() -> None:
            await self._press(f"appr:a:{USER}")
            callback = await self._press(f"appr:a:{USER}", actor=ADMIN)
            async with user_service() as service:
                self.assertTrue(await service.is_allowed_user(USER))
            edited = callback.message.edit_text.await_args
            self.assertIn("Already approved", edited.args[0])

        self._run(go)

    def test_non_admin_cannot_decide(self):
        async def go() -> None:
            callback = await self._press(f"appr:a:{USER}", actor=USER)
            answered = callback.answer
            answered.assert_awaited_once()
            self.assertEqual(answered.await_args.args[0], "Admins only")
            callback.message.edit_text.assert_not_awaited()
            async with user_service() as service:
                self.assertFalse(await service.is_allowed_user(USER))

        self._run(go)

    def test_garbage_callback_is_ignored(self):
        async def go() -> None:
            callback = await self._press("appr:x:oops")
            callback.answer.assert_awaited_once()
            callback.message.edit_text.assert_not_awaited()
            async with user_service() as service:
                self.assertFalse(await service.is_allowed_user(USER))
                self.assertFalse(await service.is_blocked(USER))

        self._run(go)


if __name__ == "__main__":
    unittest.main()
