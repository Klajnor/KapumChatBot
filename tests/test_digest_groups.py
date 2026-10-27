"""Digest in groups: observation through the real dispatcher, /digest_on, edits."""

from __future__ import annotations

import datetime
from unittest.mock import AsyncMock, patch

from aiogram import Dispatcher, Router
from aiogram.types import Chat, Message, Update, User

from bot.config import config
from bot.database.repository import DigestRepository
from bot.handlers import digest as handler
from bot.services import digest
from tests.test_database import DBTest

GROUP, ADMIN, MEMBER = -1002, 7, 8
NOW = datetime.datetime(2026, 10, 25, 12, 0, tzinfo=datetime.UTC)


def _message(text: str, user_id: int, message_id: int, *, edited: bool = False) -> Update:
    message = Message(
        message_id=message_id,
        date=NOW,
        edit_date=int(NOW.timestamp()) if edited else None,
        chat=Chat(id=GROUP, type="supergroup", title="Team"),
        from_user=User(id=user_id, is_bot=False, first_name="Ann"),
        text=text,
    )
    return Update(
        update_id=message_id,
        **({"edited_message": message} if edited else {"message": message}),
    )


class TestGroupDigest(DBTest):
    def _run(self, coro_fn):
        patches = [
            patch("bot.services.digest.async_session_factory", self._factory),
            patch("bot.services.digest._channel_ids", None),
            patch.object(config, "owner_ids", [ADMIN]),
            patch.object(config, "extra_admin_ids", set()),
        ]
        for p in patches:
            p.start()
        try:
            return self._loop.run_until_complete(coro_fn())
        finally:
            for p in reversed(patches):
                p.stop()

    def _dispatcher(self) -> tuple[Dispatcher, list[int]]:
        reached: list[int] = []
        later = Router()  # stands in for the chat router that follows

        @later.message()
        async def _catch_all(message: Message) -> None:
            reached.append(message.message_id)

        dp = Dispatcher()
        handler.router._parent_router = None  # the module router is reused per test
        dp.include_router(handler.router)
        self.addCleanup(setattr, handler.router, "_parent_router", None)
        dp.include_router(later)
        return dp, reached

    async def _stored(self) -> list[str]:
        async with self._factory() as s:
            rows = await DigestRepository(s).messages_up_to(GROUP, NOW, 50)
        return [r.text for r in rows]

    def test_group_messages_are_stored_and_still_reach_other_handlers(self):
        async def go() -> None:
            await digest.add_channel(GROUP, "Team", ADMIN)
            dp, reached = self._dispatcher()
            bot = AsyncMock()
            await dp.feed_update(bot, _message("hello", MEMBER, 1))
            self.assertEqual(await self._stored(), ["hello"])
            self.assertEqual(reached, [1])
            await dp.feed_update(bot, _message("hello again", MEMBER, 1, edited=True))
            self.assertEqual(await self._stored(), ["hello again"])

        self._run(go)

    def test_own_messages_of_the_bot_are_ignored(self):
        async def go() -> None:
            await digest.add_channel(GROUP, "Team", ADMIN)
            dp, _ = self._dispatcher()
            with patch("bot.handlers.digest.get_bot_id", return_value=MEMBER):
                await dp.feed_update(AsyncMock(), _message("my own answer", MEMBER, 1))
                await dp.feed_update(AsyncMock(), _message("a human", ADMIN, 2))
            self.assertEqual(await self._stored(), ["a human"])

        self._run(go)

    def test_untracked_group_is_ignored(self):
        async def go() -> None:
            dp, _ = self._dispatcher()
            await dp.feed_update(AsyncMock(), _message("hello", MEMBER, 1))
            self.assertEqual(await self._stored(), [])

        self._run(go)

    def test_only_admins_toggle_a_group(self):
        async def go() -> None:
            dp, _ = self._dispatcher()
            with patch.object(Message, "answer", AsyncMock()) as answer:
                await dp.feed_update(AsyncMock(), _message("/digest_on", MEMBER, 1))
                self.assertEqual(await digest.list_channels(), [])
                answer.assert_not_awaited()

                await dp.feed_update(AsyncMock(), _message("/digest_on", ADMIN, 2))
                self.assertEqual(await digest.list_channels(), [(GROUP, "Team")])
                answer.assert_awaited_once()

                await dp.feed_update(AsyncMock(), _message("/digest_off", ADMIN, 3))
                self.assertEqual(await digest.list_channels(), [])

        self._run(go)

    def test_migration_to_supergroup_keeps_the_group(self):
        async def go() -> None:
            await digest.add_channel(GROUP, "Team", ADMIN)
            message = Message(
                message_id=5,
                date=NOW,
                chat=Chat(id=GROUP, type="group", title="Team"),
                migrate_to_chat_id=-1009,
            )
            await handler.on_group_migrated(message)
            self.assertEqual(await digest.list_channels(), [(-1009, "Team")])

        self._run(go)
