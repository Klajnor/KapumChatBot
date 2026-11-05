"""Access levels: personal limits, blocking, runtime admins and menu rules."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from bot.config import config
from bot.handlers.admin_panel import _can_manage, _render_card, _render_list
from bot.handlers.help import cmd_help
from bot.handlers.start import cmd_start
from bot.services.access import check_access
from bot.services.user_service import user_service
from tests.test_database import DBTest

OWNER, ADMIN, USER, GUEST = 1, 2, 3, 4


def _message(user_id: int) -> SimpleNamespace:
    return SimpleNamespace(
        from_user=SimpleNamespace(
            id=user_id, username="bob", first_name="Bob", last_name=None,
        ),
        chat=SimpleNamespace(type="private"),
        answer=AsyncMock(),
    )


class TestAccessLevels(DBTest):
    def _run(self, coro_fn):
        patches = (
            patch("bot.database.session.async_session_factory", self._factory),
            patch.object(config, "owner_ids", [OWNER]),
            patch.object(config, "extra_admin_ids", set()),
            patch.object(config, "daily_limit", 3),
            patch.object(config, "allow_guest_access", True),
            patch.object(config, "guest_message_limit", 5),
        )
        for p in patches:
            p.start()
        try:
            self._loop.run_until_complete(coro_fn())
        finally:
            for p in reversed(patches):
                p.stop()

    def test_personal_limit_overrides_default(self):
        async def go() -> None:
            async with user_service() as s:
                await s.register_user(USER)
                await s.add_allowed_user(USER)
                self.assertEqual(await s.get_daily_limit(USER), 3)
                await s.set_daily_limit(USER, 1)
            async with user_service() as s:
                first = await check_access(s, USER)
                second = await check_access(s, USER)
            self.assertTrue(first.allowed)
            self.assertEqual(first.remaining, 0)
            self.assertFalse(second.allowed)
            self.assertIn("1 questions", second.message)
            async with user_service() as s:
                await s.set_daily_limit(USER, None)
                self.assertEqual(await s.get_daily_limit(USER), 3)

        self._run(go)

    def test_help_and_start_show_effective_daily_limit(self):
        async def go() -> None:
            async with user_service() as s:
                await s.register_user(USER)
                await s.add_allowed_user(USER)

            for override, expected in ((None, 3), (1, 1), (8, 8), (0, 0), (None, 3)):
                async with user_service() as s:
                    await s.set_daily_limit(USER, override)
                for command in (cmd_help, cmd_start):
                    with self.subTest(command=command.__name__, override=override):
                        message = _message(USER)
                        await command(message)
                        message.answer.assert_awaited_once()
                        self.assertIn(
                            f"<b>Daily limit:</b> {expected} questions per day",
                            message.answer.await_args.args[0],
                        )

            async with user_service() as s:
                self.assertEqual(await s.get_daily_usage_used(USER), 0)

        self._run(go)

    def test_help_and_start_keep_admin_and_guest_limits(self):
        async def go() -> None:
            async with user_service() as s:
                await s.register_user(ADMIN)
                await s.add_allowed_user(ADMIN)
                await s.set_daily_limit(ADMIN, 0)
                await s.set_admin(ADMIN, True)

            for user_id in (OWNER, ADMIN, GUEST):
                for command in (cmd_help, cmd_start):
                    with self.subTest(command=command.__name__, user_id=user_id):
                        message = _message(user_id)
                        await command(message)
                        message.answer.assert_awaited_once()
                        text = message.answer.await_args.args[0]
                        self.assertNotIn("Daily limit:", text)
                        if user_id == GUEST:
                            self.assertIn("up to 5 questions", text)
                        else:
                            self.assertIn("Admin", text)

        self._run(go)

    def test_blocked_user_denied_even_as_guest(self):
        async def go() -> None:
            async with user_service() as s:
                await s.register_user(GUEST)
                self.assertTrue((await check_access(s, GUEST, consume=False)).allowed)
                await s.set_blocked(GUEST, True)
                result = await check_access(s, GUEST)
                self.assertFalse(result.allowed)
                self.assertEqual(result.role, "blocked")
                self.assertEqual(await s.get_level(GUEST), "blocked")
                await s.set_blocked(GUEST, False)
                self.assertTrue((await check_access(s, GUEST, consume=False)).allowed)

        self._run(go)

    def test_promoted_admin_is_unlimited_and_revocable(self):
        async def go() -> None:
            async with user_service() as s:
                await s.register_user(ADMIN)
                await s.set_admin(ADMIN, True)
                self.assertTrue(config.is_admin(ADMIN))
                self.assertFalse(config.is_owner(ADMIN))
                self.assertEqual(await s.get_level(ADMIN), "admin")
                self.assertEqual((await check_access(s, ADMIN)).role, "admin")
                # survives a restart
                config.extra_admin_ids.clear()
                await s.load_granted_admins()
                self.assertTrue(config.is_admin(ADMIN))
                await s.set_admin(ADMIN, False)
                self.assertFalse(config.is_admin(ADMIN))

        self._run(go)

    def test_menu_permissions(self):
        self.assertFalse(_can_manage(OWNER, "owner"))
        self.assertFalse(_can_manage(ADMIN, "owner"))
        with patch.object(config, "owner_ids", [OWNER]):
            self.assertTrue(_can_manage(OWNER, "admin"))
            self.assertFalse(_can_manage(ADMIN, "admin"))
        self.assertTrue(_can_manage(ADMIN, "user"))
        self.assertTrue(_can_manage(ADMIN, "guest"))

    def test_menu_renders_list_and_card(self):
        async def go() -> None:
            async with user_service() as s:
                await s.register_user(USER, username="bob")
                await s.add_allowed_user(USER)
                await s.register_user(GUEST)
            _text, markup = await _render_list(OWNER, "u", 0)
            data = [b.callback_data for row in markup.inline_keyboard for b in row]
            self.assertIn(f"adm:u:{USER}", data)
            self.assertNotIn(f"adm:u:{GUEST}", data)

            card, markup = await _render_card(OWNER, USER)
            data = [b.callback_data for row in markup.inline_keyboard for b in row]
            self.assertIn(f"adm:lim:{USER}:5", data)
            self.assertIn(f"adm:adm:{USER}", data)  # owners can promote
            self.assertIn("0/3", card)

            _, markup = await _render_card(ADMIN, USER)  # an admin cannot
            with patch.object(config, "extra_admin_ids", {ADMIN}):
                _, markup = await _render_card(ADMIN, USER)
            data = [b.callback_data for row in markup.inline_keyboard for b in row]
            self.assertNotIn(f"adm:adm:{USER}", data)

        self._run(go)
