"""Отчёт о статусе: лимиты, второй мозг и часовая проверка поиска."""

from __future__ import annotations

import datetime
import unittest
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from bot.config import config
from bot.services.access import AccessResult
from bot.services.mcp import mcp_manager
from bot.services.request_intent import Intent
from bot.services.search import SearchResult, SearchResults
from bot.services.status import render_status


class FakeService:
    """Minimal UserService surface used by :func:`render_status`."""

    def __init__(self, *, previous: datetime.datetime | None = None) -> None:
        self.previous = previous
        self.try_reserve_status_search = AsyncMock(return_value=previous)

    async def get_user_settings(self, telegram_id: int) -> dict:
        return {"model": "test-model"}

    async def get_reminder_timezone(self, telegram_id: int) -> str | None:
        return "UTC"

    async def get_level(self, telegram_id: int) -> str:
        return "user"

    async def get_daily_usage_remaining(self, telegram_id: int) -> int:
        return 7

    async def get_daily_limit(self, telegram_id: int) -> int:
        return 10

    async def get_guest_messages_remaining(self, telegram_id: int) -> int:
        return 2


def _result() -> SearchResults:
    return SearchResults(
        [SearchResult(title="Example", url="https://example.com", snippet="ok")]
    )


class StatusReportTests(unittest.IsolatedAsyncioTestCase):
    async def test_report_has_limits_and_brain_counts(self) -> None:
        service = FakeService()
        brain = {
            "notes": 120,
            "chunks": 340,
            "links": 456,
            "strong_links": 78,
            "pending_links": 12,
            "failed_links": 3,
        }
        with (
            patch("bot.services.status.search_configured", return_value=True),
            patch("bot.services.status.provider_name", return_value="exa"),
            patch("bot.services.status.search_web", AsyncMock(return_value=_result())) as search,
            patch("bot.services.status.mcp_available", return_value=True),
            patch.object(
                mcp_manager, "brain_status", AsyncMock(return_value=brain)
            ),
            patch.object(config, "enable_web_search", True),
            patch.object(config, "enable_mcp", True),
            patch.object(config, "status_search_interval", 3600),
            patch.object(config, "resolve_model", return_value={"name": "Test Model"}),
        ):
            report = await render_status(service, 42, private=True)

        self.assertIn("Test Model", report)
        self.assertIn("7/10", report)
        self.assertIn("Notes: 120", report)
        self.assertIn("456 indexed", report)
        self.assertIn("78 strong", report)
        self.assertIn("12 not indexed", report)
        self.assertIn("3 deferred", report)
        self.assertIn("reachable", report)
        self.assertNotIn("https://example.com", report)
        search.assert_awaited_once()

    async def test_hourly_limit_reports_next_time_without_search(self) -> None:
        previous = datetime.datetime.now(datetime.UTC) - datetime.timedelta(minutes=20)
        service = FakeService(previous=previous)
        with (
            patch("bot.services.status.search_configured", return_value=True),
            patch("bot.services.status.provider_name", return_value="exa"),
            patch("bot.services.status.search_web", AsyncMock()) as search,
            patch("bot.services.status.mcp_available", return_value=False),
            patch.object(config, "enable_web_search", True),
            patch.object(config, "enable_mcp", True),
            patch.object(config, "status_search_interval", 3600),
            patch.object(config, "resolve_model", return_value={"name": "Test Model"}),
        ):
            report = await render_status(service, 42, private=True)

        self.assertIn("hourly live check already used", report)
        self.assertIn("Next real check", report)
        self.assertRegex(report, r"in (39|40) min")
        search.assert_not_awaited()

    async def test_group_report_keeps_limits_and_probe_private(self) -> None:
        service = FakeService()
        with (
            patch("bot.services.status.search_configured", return_value=True),
            patch("bot.services.status.provider_name", return_value="exa"),
            patch("bot.services.status.search_web", AsyncMock()) as search,
            patch("bot.services.status.mcp_available", return_value=False),
            patch.object(config, "enable_web_search", True),
            patch.object(config, "enable_mcp", True),
            patch.object(config, "status_search_interval", 3600),
            patch.object(config, "resolve_model", return_value={"name": "Test Model"}),
        ):
            report = await render_status(service, 42, private=False)

        self.assertNotIn("Your access", report)
        self.assertNotIn("7/10", report)
        self.assertIn("run this check in DM", report)
        search.assert_not_awaited()
        service.try_reserve_status_search.assert_not_awaited()

    async def test_brain_failure_is_reported_without_details(self) -> None:
        service = FakeService()
        with (
            patch("bot.services.status.search_configured", return_value=False),
            patch("bot.services.status.provider_name", return_value="exa"),
            patch("bot.services.status.mcp_available", return_value=True),
            patch.object(
                mcp_manager,
                "brain_status",
                AsyncMock(return_value={"error": "/secret/path/traceback"}),
            ),
            patch.object(config, "enable_web_search", True),
            patch.object(config, "enable_mcp", True),
            patch.object(config, "resolve_model", return_value={"name": "Test Model"}),
        ):
            report = await render_status(service, 42, private=True)

        self.assertIn("Second brain:</b> ⚠️ unavailable", report)
        self.assertNotIn("/secret/path/traceback", report)

    async def test_search_failure_uses_a_safe_error_code(self) -> None:
        service = FakeService()
        with (
            patch("bot.services.status.search_configured", return_value=True),
            patch("bot.services.status.provider_name", return_value="exa"),
            patch(
                "bot.services.status.search_web",
                AsyncMock(return_value=SearchResults(error="http_401")),
            ),
            patch("bot.services.status.mcp_available", return_value=False),
            patch.object(config, "enable_web_search", True),
            patch.object(config, "enable_mcp", True),
            patch.object(config, "status_search_interval", 3600),
            patch.object(config, "resolve_model", return_value={"name": "Test Model"}),
        ):
            report = await render_status(service, 42, private=True)

        self.assertIn("❌", report)
        self.assertIn("http_401", report)


class StatusIntentLimitTests(unittest.IsolatedAsyncioTestCase):
    """AI-подтверждение «статуса»: guest закрыт, обычные пользователи лимитированы."""

    def _message(self, user_id: int = 7):
        return SimpleNamespace(
            from_user=SimpleNamespace(id=user_id),
            chat=SimpleNamespace(type="private"),
            forward_date=None,
        )

    async def test_guest_cannot_use_status_intent(self) -> None:
        from bot.handlers.status import maybe_status_request

        service = SimpleNamespace(
            consume_status_intent_budget=AsyncMock(return_value=True),
            get_user_settings=AsyncMock(return_value={"model": "m"}),
        )

        @asynccontextmanager
        async def fake_user_service():
            yield service

        with (
            patch("bot.handlers.status.user_service", fake_user_service),
            patch(
                "bot.handlers.status.check_access",
                AsyncMock(return_value=AccessResult(allowed=True, role="guest")),
            ),
            patch(
                "bot.handlers.status.classify_request",
                AsyncMock(return_value=Intent.STATUS),
            ),
        ):
            handled = await maybe_status_request(self._message(), "статус")

        self.assertFalse(handled)
        service.consume_status_intent_budget.assert_not_awaited()

    async def test_admin_bypasses_status_intent_budget(self) -> None:
        from bot.handlers.status import maybe_status_request

        service = SimpleNamespace(
            consume_status_intent_budget=AsyncMock(return_value=False),
            get_user_settings=AsyncMock(return_value={"model": "m"}),
        )

        @asynccontextmanager
        async def fake_user_service():
            yield service

        with (
            patch("bot.handlers.status.user_service", fake_user_service),
            patch(
                "bot.handlers.status.check_access",
                AsyncMock(return_value=AccessResult(allowed=True, role="admin")),
            ),
            patch.object(config, "is_admin", return_value=True),
            patch(
                "bot.handlers.status.classify_request",
                AsyncMock(return_value=Intent.CHAT),
            ),
        ):
            handled = await maybe_status_request(self._message(42), "статус")

        self.assertFalse(handled)
        service.consume_status_intent_budget.assert_not_awaited()

    async def test_regular_user_hits_status_intent_budget(self) -> None:
        from bot.handlers.status import maybe_status_request

        service = SimpleNamespace(
            consume_status_intent_budget=AsyncMock(return_value=False),
            get_user_settings=AsyncMock(return_value={"model": "m"}),
        )

        @asynccontextmanager
        async def fake_user_service():
            yield service

        with (
            patch("bot.handlers.status.user_service", fake_user_service),
            patch(
                "bot.handlers.status.check_access",
                AsyncMock(return_value=AccessResult(allowed=True, role="allowed")),
            ),
            patch.object(config, "is_admin", return_value=False),
            patch(
                "bot.handlers.status.classify_request",
                AsyncMock(return_value=Intent.STATUS),
            ) as classify,
            patch(
                "bot.handlers.status.safe_answer", AsyncMock()
            ) as answer,
        ):
            handled = await maybe_status_request(self._message(), "статус")

        self.assertTrue(handled)
        service.consume_status_intent_budget.assert_awaited_once()
        classify.assert_not_awaited()
        answer.assert_awaited_once()
        self.assertIn("limited", answer.call_args.args[1])

    async def test_regular_user_consumes_status_intent_budget(self) -> None:
        from bot.handlers.status import maybe_status_request

        service = SimpleNamespace(
            consume_status_intent_budget=AsyncMock(return_value=True),
            get_user_settings=AsyncMock(return_value={"model": "m"}),
        )

        @asynccontextmanager
        async def fake_user_service():
            yield service

        with (
            patch("bot.handlers.status.user_service", fake_user_service),
            patch(
                "bot.handlers.status.check_access",
                AsyncMock(return_value=AccessResult(allowed=True, role="allowed")),
            ),
            patch.object(config, "is_admin", return_value=False),
            patch(
                "bot.handlers.status.classify_request",
                AsyncMock(return_value=Intent.CHAT),
            ) as classify,
        ):
            handled = await maybe_status_request(self._message(), "статус")

        self.assertFalse(handled)
        service.consume_status_intent_budget.assert_awaited_once()
        classify.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
