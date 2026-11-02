"""Транзиентные сбои Telegram не должны останавливать бота."""

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

# Keep local credentials out of the tests and never contact configured services.
os.environ["BOT_TOKEN"] = "123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghi"
os.environ["AI_API_KEY"] = "test-key"
os.environ["DATABASE_URL"] = "postgresql+asyncpg://test:test@localhost/test"

from aiogram.exceptions import TelegramNetworkError, TelegramUnauthorizedError
from aiogram.methods import GetMe, GetUpdates
from aiogram.types import User

from bot import bot as bot_module
from bot import main as main_module


class SetupBotRetryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._bot_id = bot_module.bot_id
        self._bot_username = bot_module.bot_username
        self.addCleanup(self._restore)

    def _restore(self) -> None:
        bot_module.bot_id = self._bot_id
        bot_module.bot_username = self._bot_username

    async def test_network_timeout_is_retried(self) -> None:
        me = User(id=42, is_bot=True, first_name="Test", username="test_bot")
        timeout = TelegramNetworkError(
            method=GetMe(), message="Request timeout error"
        )
        with (
            patch.object(
                bot_module.bot, "me", AsyncMock(side_effect=[timeout, me])
            ) as me_call,
            patch.object(bot_module.bot, "set_my_commands", AsyncMock()),
            patch.object(bot_module.asyncio, "sleep", AsyncMock()) as sleep,
        ):
            await bot_module.setup_bot()

        self.assertEqual(me_call.await_count, 2)
        self.assertEqual(bot_module.bot_id, 42)
        self.assertEqual(bot_module.bot_username, "test_bot")
        sleep.assert_awaited_once()
        self.assertEqual(sleep.await_args.args[0], bot_module.RETRY_INITIAL_DELAY)

    async def test_bad_token_is_not_retried(self) -> None:
        unauthorized = TelegramUnauthorizedError(
            method=GetMe(), message="Unauthorized"
        )
        with (
            patch.object(
                bot_module.bot, "me", AsyncMock(side_effect=unauthorized)
            ),
            patch.object(bot_module.asyncio, "sleep", AsyncMock()) as sleep,
            self.assertRaises(TelegramUnauthorizedError),
        ):
            await bot_module.setup_bot()

        sleep.assert_not_awaited()


class PollingRetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_polling_restarts_after_network_timeout(self) -> None:
        timeout = TelegramNetworkError(
            method=GetUpdates(), message="Request timeout error"
        )
        with (
            patch.object(
                main_module.dp,
                "start_polling",
                AsyncMock(side_effect=[timeout, None]),
            ) as polling,
            patch.object(main_module.asyncio, "sleep", AsyncMock()) as sleep,
        ):
            await main_module._poll_forever()

        self.assertEqual(polling.await_count, 2)
        sleep.assert_awaited_once()
        # The session stays open for the retry; main() closes it on shutdown.
        self.assertFalse(polling.call_args.kwargs["close_bot_session"])

    async def test_graceful_shutdown_stops_the_loop(self) -> None:
        with patch.object(
            main_module.dp, "start_polling", AsyncMock()
        ) as polling:
            await main_module._poll_forever()

        polling.assert_awaited_once()

    async def test_unexpected_errors_are_not_swallowed(self) -> None:
        with (
            patch.object(
                main_module.dp,
                "start_polling",
                AsyncMock(side_effect=RuntimeError("boom")),
            ),
            self.assertRaises(RuntimeError),
        ):
            await main_module._poll_forever()


class ErrorHandlerTests(unittest.IsolatedAsyncioTestCase):
    async def test_network_error_is_not_answered(self) -> None:
        message = SimpleNamespace(chat=SimpleNamespace(id=1), answer=AsyncMock())
        event = SimpleNamespace(
            update=SimpleNamespace(
                update_id=7,
                message=message,
                edited_message=None,
                callback_query=None,
            ),
            exception=TelegramNetworkError(
                method=GetUpdates(), message="Request timeout error"
            ),
        )
        with self.assertLogs("bot.main", level="WARNING"):
            await main_module.on_unhandled_error(event)

        message.answer.assert_not_awaited()
