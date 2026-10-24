"""Regression tests for the issues found in the code/security audit.

Every test here pins a specific hole that was closed:

* broadcast confirmations are bound to the admin who drafted the text;
* over-long announcements are rejected before any send is attempted;
* album photos are size-checked *before* they are downloaded (memory guard);
* the album's cumulative budget stops further downloads;
* "message edited" notices are rate-limited per chat;
* the paid intent classifier never runs for a user without access;
* an out-of-range HEALTH_PORT cannot abort startup.

Offline only: no Telegram, AI or database connections.
"""
from __future__ import annotations

import os
import unittest
from contextlib import asynccontextmanager
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

os.environ.setdefault("BOT_TOKEN", "123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghi")
os.environ.setdefault("AI_API_KEY", "test-key")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite://")

from bot.config import Config, config
from bot.handlers import admin, chat, models, prompt, stats
from bot.services.access import AccessResult
from bot.services.telegram_utils import require_private_chat


class BroadcastAuthTests(unittest.IsolatedAsyncioTestCase):
    """Only the drafting admin may confirm or cancel a broadcast."""

    def _callback(self, data: str, user_id: int) -> SimpleNamespace:
        return SimpleNamespace(data=data, from_user=SimpleNamespace(id=user_id))

    def test_owner_is_accepted(self):
        with patch.object(config, "admin_ids", [111]):
            self.assertEqual(
                admin._pending_owner(self._callback("broadcast:send:111", 111), "send"),
                111,
            )

    def test_another_admin_cannot_confirm(self):
        with patch.object(config, "admin_ids", [111, 222]):
            self.assertIsNone(
                admin._pending_owner(self._callback("broadcast:send:111", 222), "send")
            )

    def test_non_admin_cannot_confirm(self):
        with patch.object(config, "admin_ids", [111]):
            self.assertIsNone(
                admin._pending_owner(self._callback("broadcast:send:111", 999), "send")
            )

    def test_malformed_callback_data_is_rejected(self):
        with patch.object(config, "admin_ids", [111]):
            for data in ("broadcast:send:", "broadcast:send:abc", "broadcast:send", ""):
                self.assertIsNone(
                    admin._pending_owner(self._callback(data, 111), "send"), data
                )

    def test_wrong_action_is_rejected(self):
        with patch.object(config, "admin_ids", [111]):
            # A "cancel" payload must not authorize a "send".
            self.assertIsNone(
                admin._pending_owner(self._callback("broadcast:cancel:111", 111), "send")
            )

    async def test_forged_confirm_does_not_send(self):
        admin._pending_broadcasts[111] = "secret announcement"
        callback = SimpleNamespace(
            data="broadcast:send:111",
            from_user=SimpleNamespace(id=222),
            message=SimpleNamespace(edit_text=AsyncMock()),
            bot=SimpleNamespace(send_message=AsyncMock()),
            answer=AsyncMock(),
        )
        with (
            patch.object(config, "admin_ids", [111, 222]),
            patch.object(admin, "safe_answer_callback", AsyncMock()) as ack,
            patch.object(admin, "_deliver_broadcast", AsyncMock()) as deliver,
        ):
            await admin.cb_broadcast_send(callback)
        deliver.assert_not_awaited()
        ack.assert_awaited_once()
        # The draft survives: the rightful owner can still send it.
        self.assertIn(111, admin._pending_broadcasts)
        admin._pending_broadcasts.clear()

    async def test_overlong_announcement_is_rejected(self):
        message = SimpleNamespace(
            from_user=SimpleNamespace(id=1),
            text="/broadcast " + "x" * (admin._MAX_BROADCAST_LENGTH + 1),
            reply_to_message=None,
            answer=AsyncMock(),
        )
        with (
            patch.object(config, "admin_ids", [1]),
            patch.object(admin, "user_service") as us,
        ):
            await admin.cmd_broadcast(message)
        us.assert_not_called()
        message.answer.assert_awaited_once()
        self.assertIn("too long", message.answer.call_args.args[0])


class AlbumSizeGuardTests(unittest.IsolatedAsyncioTestCase):
    """Photos must be rejected on their declared size, before downloading."""

    def setUp(self) -> None:
        chat._album_images.clear()
        chat._album_total_bytes.clear()

    def tearDown(self) -> None:
        chat._album_images.clear()
        chat._album_total_bytes.clear()

    def _message(self, size: int, caption: str | None = None) -> SimpleNamespace:
        return SimpleNamespace(
            media_group_id="grp",
            chat=SimpleNamespace(id=42),
            caption=caption,
            photo=[SimpleNamespace(file_id="f1", file_size=size)],
            bot=SimpleNamespace(get_file=AsyncMock(), download_file=AsyncMock()),
            answer=AsyncMock(),
        )

    async def test_oversized_photo_is_never_downloaded(self):
        message = self._message(size=25 * 1024 * 1024)
        with (
            patch.object(config, "max_photo_bytes", 20 * 1024 * 1024),
            patch.object(config, "max_album_bytes", 50 * 1024 * 1024),
            patch.object(chat, "_ALBUM_COLLECT_SECONDS", 0),
            patch.object(chat, "safe_answer", AsyncMock()) as answer,
        ):
            await chat._handle_album_photo(message)
        message.bot.get_file.assert_not_awaited()
        answer.assert_awaited()
        self.assertIn("too large", answer.call_args.args[1])

    async def test_exhausted_album_budget_stops_downloads(self):
        key = chat._album_key(42, "grp")
        chat._album_images[key] = {"images": [], "message": self._message(1), "last": 0.0}
        chat._album_total_bytes[key] = 2 * 1024 * 1024
        message = self._message(size=1024 * 1024)  # 1 MB more -> over a 2 MB cap
        with (
            patch.object(config, "max_photo_bytes", 20 * 1024 * 1024),
            patch.object(config, "max_album_bytes", 2 * 1024 * 1024),
            patch.object(chat, "_ALBUM_COLLECT_SECONDS", 0),
            patch.object(chat, "safe_answer", AsyncMock()) as answer,
            patch.object(chat, "_guarded", AsyncMock()) as guarded,
        ):
            await chat._handle_album_photo(message)
        message.bot.get_file.assert_not_awaited()
        answer.assert_awaited()
        self.assertIn("Album size limit", answer.call_args.args[1])
        guarded.assert_not_awaited()


class EditedNoticeThrottleTests(unittest.IsolatedAsyncioTestCase):
    """Repeated edits must not produce a stream of bot messages."""

    def test_cooldown_blocks_second_notice(self):
        chat._edit_notice_last.clear()
        self.assertTrue(chat._edit_notice_allowed(7))
        self.assertFalse(chat._edit_notice_allowed(7))
        # A different chat is unaffected.
        self.assertTrue(chat._edit_notice_allowed(8))
        chat._edit_notice_last.clear()

    def test_registry_is_bounded(self):
        chat._edit_notice_last.clear()
        for chat_id in range(chat._EDIT_NOTICE_CAP + 50):
            chat._edit_notice_allowed(chat_id)
        self.assertLessEqual(len(chat._edit_notice_last), chat._EDIT_NOTICE_CAP)
        chat._edit_notice_last.clear()

    async def test_handler_stays_silent_within_cooldown(self):
        chat._edit_notice_last.clear()
        message = SimpleNamespace(
            from_user=SimpleNamespace(id=9),
            text="hello again",
            chat=SimpleNamespace(id=9, type="private"),
            reply_to_message=None,
            forward_date=None,
        )
        with (
            patch.object(chat, "_is_addressed", return_value=True),
            patch.object(chat, "safe_answer", AsyncMock()) as answer,
        ):
            await chat.handle_edited_text(message)
            await chat.handle_edited_text(message)
        self.assertEqual(answer.await_count, 1)
        chat._edit_notice_last.clear()


class IntentAccessGateTests(unittest.IsolatedAsyncioTestCase):
    """The paid intent classifier must not run for unauthorized users."""

    async def test_denied_user_never_triggers_paid_classification(self):
        message = SimpleNamespace(
            from_user=SimpleNamespace(id=5),
            caption="draw me a cat",
            chat=SimpleNamespace(id=5, type="private"),
            answer=AsyncMock(),
        )
        service = SimpleNamespace(
            get_user_settings=AsyncMock(
                return_value={"model": "m", "temperature": 0.7, "max_tokens": 100}
            ),
            session=SimpleNamespace(close=AsyncMock()),
        )
        denied = AccessResult(allowed=False, role="denied", message="⛔ No access")
        with (
            patch.object(chat, "_is_addressed", return_value=True),
            patch.object(chat, "_open_registered", AsyncMock(return_value=service)),
            patch.object(chat, "check_access", AsyncMock(return_value=denied)) as gate,
            patch.object(chat, "detect_intent", AsyncMock()) as detect,
            patch.object(chat, "safe_answer", AsyncMock()) as answer,
        ):
            await chat._route_single_photo(message)
        gate.assert_awaited_once()
        detect.assert_not_awaited()
        answer.assert_awaited_once()
        self.assertIn("No access", answer.call_args.args[1])

    async def test_allowed_user_still_gets_classification(self):
        message = SimpleNamespace(
            from_user=SimpleNamespace(id=6),
            caption="draw me a cat",
            chat=SimpleNamespace(id=6, type="private"),
            answer=AsyncMock(),
        )
        service = SimpleNamespace(
            get_user_settings=AsyncMock(
                return_value={"model": "m", "temperature": 0.7, "max_tokens": 100}
            ),
            session=SimpleNamespace(close=AsyncMock()),
        )
        allowed = AccessResult(allowed=True, role="admin")
        with (
            patch.object(config, "enable_image_generation", False),
            patch.object(chat, "_is_addressed", return_value=True),
            patch.object(chat, "_open_registered", AsyncMock(return_value=service)),
            patch.object(chat, "check_access", AsyncMock(return_value=allowed)),
            patch.object(chat, "detect_intent", AsyncMock(return_value=("generate", []))) as detect,
            patch.object(chat, "_handle_photo", AsyncMock()) as handle,
        ):
            await chat._route_single_photo(message)
        detect.assert_awaited_once()
        handle.assert_awaited_once()


class ConfigValidationTests(unittest.TestCase):
    """Bad configuration must degrade safely, never abort startup."""

    def test_out_of_range_health_port_is_disabled(self):
        for raw in ("99999", "-5", "not-a-number"):
            with patch.dict(os.environ, {"HEALTH_PORT": raw}):
                self.assertEqual(Config().health_port, 0, raw)

    def test_valid_health_port_is_kept(self):
        with patch.dict(os.environ, {"HEALTH_PORT": "8080"}):
            self.assertEqual(Config().health_port, 8080)

    def test_photo_cap_never_exceeds_telegram_download_limit(self):
        with patch.dict(os.environ, {"MAX_PHOTO_BYTES": str(500 * 1024 * 1024)}):
            self.assertEqual(Config().max_photo_bytes, 20 * 1024 * 1024)

    def test_guest_text_limit_cannot_be_zero(self):
        with patch.dict(os.environ, {"MAX_GUEST_TEXT_LENGTH": "0"}):
            self.assertGreaterEqual(Config().max_guest_text_length, 1)

    def test_album_budget_is_at_least_one_photo(self):
        with patch.dict(
            os.environ,
            {"MAX_PHOTO_BYTES": str(10 * 1024 * 1024), "MAX_ALBUM_BYTES": "1024"},
        ):
            cfg = Config()
            self.assertGreaterEqual(cfg.max_album_bytes, cfg.max_photo_bytes)


class ReminderIntervalTests(unittest.TestCase):
    """Huge intervals must be rejected, not crash with OverflowError."""

    def test_absurd_one_shot_interval_is_rejected(self):
        from bot.services import reminders

        result = reminders.parse_reminder("in 99999999999999 days buy milk")
        self.assertIsInstance(result, str)
        self.assertIn("too far in the future", result)

    def test_absurd_recurring_interval_is_rejected(self):
        from bot.services import reminders

        result = reminders.parse_reminder("every 999999999999 days stand up")
        self.assertIsInstance(result, str)
        self.assertIn("too far in the future", result)

    def test_normal_intervals_still_parse(self):
        from bot.services import reminders

        parsed = reminders.parse_reminder("in 30 minutes tea")
        self.assertNotIsInstance(parsed, str)
        self.assertEqual(parsed.text, "tea")


class ReminderAccessGateTests(unittest.IsolatedAsyncioTestCase):
    """A revoked user's scheduled AI reminder must stop calling the paid API."""

    async def test_revoked_user_disables_reminder_without_ai_call(self):
        from bot.services import reminders
        from bot.services.access import AccessResult as AR

        due = reminders.DueReminder(
            id=1, user_id=7, chat_id=7, text="summary", ai_mode=True, schedule="every:60"
        )
        session = SimpleNamespace(close=AsyncMock())
        service = SimpleNamespace(
            get_user_settings=AsyncMock(
                return_value={
                    "system_prompt": "s", "model": "m",
                    "temperature": 0.5, "max_tokens": 100,
                }
            )
        )
        denied = AR(allowed=False, role="denied", message="no")

        @asynccontextmanager
        async def fake_session():
            yield session

        with (
            patch.object(reminders, "async_session_factory", fake_session),
            patch.object(reminders, "UserService", return_value=service),
            patch.object(reminders, "check_access", AsyncMock(return_value=denied)),
            patch.object(reminders, "_disable_reminder", AsyncMock()) as disable,
            patch.object(
                reminders.ai_client, "chat_completion_result", AsyncMock()
            ) as ai,
        ):
            await reminders._fire_ai(due)
        ai.assert_not_awaited()
        disable.assert_awaited_once_with(1)

    async def test_exhausted_daily_quota_does_not_disable_reminder(self):
        """Regression: reminders are free, so an exhausted *question* budget
        must not switch them off — only losing authorization may."""
        from bot.services import reminders
        from bot.services.access import AccessResult as AR

        due = reminders.DueReminder(
            id=2, user_id=8, chat_id=8, text="summary", ai_mode=True, schedule="every:60"
        )
        session = SimpleNamespace(close=AsyncMock())
        service = SimpleNamespace(
            get_user_settings=AsyncMock(
                return_value={
                    "system_prompt": "s", "model": "m",
                    "temperature": 0.5, "max_tokens": 100,
                }
            )
        )
        exhausted = AR(allowed=True, role="allowed", remaining=0)
        answer = SimpleNamespace(
            content="ok", elapsed=1.0, total_tokens=5, input_tokens=2, output_tokens=3
        )

        @asynccontextmanager
        async def fake_session():
            yield session

        with (
            patch.object(reminders, "async_session_factory", fake_session),
            patch.object(reminders, "UserService", return_value=service),
            patch.object(reminders, "check_access", AsyncMock(return_value=exhausted)),
            patch.object(reminders, "_disable_reminder", AsyncMock()) as disable,
            patch.object(reminders, "_send_markdown", AsyncMock(return_value=True)),
            patch.object(
                reminders.ai_client,
                "chat_completion_result",
                AsyncMock(return_value=answer),
            ) as ai,
        ):
            await reminders._fire_ai(due)
        disable.assert_not_awaited()
        ai.assert_awaited_once()

    async def test_fire_time_probe_ignores_quota(self):
        """The gate must ask 'still authorized?', never 'any quota left?'."""
        from bot.services import reminders
        from bot.services.access import AccessResult as AR

        due = reminders.DueReminder(
            id=3, user_id=9, chat_id=9, text="s", ai_mode=True, schedule=None
        )
        session = SimpleNamespace(close=AsyncMock())
        service = SimpleNamespace(
            get_user_settings=AsyncMock(
                return_value={
                    "system_prompt": "s", "model": "m",
                    "temperature": 0.5, "max_tokens": 100,
                }
            )
        )
        captured: dict[str, object] = {}

        async def fake_check(_service, _telegram_id, **kwargs):
            captured.update(kwargs)
            return AR(allowed=True, role="allowed", remaining=0)

        @asynccontextmanager
        async def fake_session():
            yield session

        with (
            patch.object(reminders, "async_session_factory", fake_session),
            patch.object(reminders, "UserService", return_value=service),
            patch.object(reminders, "check_access", side_effect=fake_check),
            patch.object(reminders, "_disable_reminder", AsyncMock()),
            patch.object(reminders, "_send_markdown", AsyncMock(return_value=True)),
            patch.object(
                reminders.ai_client,
                "chat_completion_result",
                AsyncMock(
                    return_value=SimpleNamespace(
                        content="ok", elapsed=1.0, total_tokens=1,
                        input_tokens=1, output_tokens=1,
                    )
                ),
            ),
        ):
            await reminders._fire_ai(due)
        self.assertIs(captured.get("consume"), False)
        self.assertIs(captured.get("require_quota"), False)

    async def test_guest_role_reminder_is_disabled(self):
        from bot.services import reminders
        from bot.services.access import AccessResult as AR

        due = reminders.DueReminder(
            id=4, user_id=10, chat_id=10, text="s", ai_mode=True, schedule="every:60"
        )
        session = SimpleNamespace(close=AsyncMock())
        service = SimpleNamespace(
            get_user_settings=AsyncMock(
                return_value={
                    "system_prompt": "s", "model": "m",
                    "temperature": 0.5, "max_tokens": 100,
                }
            )
        )

        @asynccontextmanager
        async def fake_session():
            yield session

        with (
            patch.object(reminders, "async_session_factory", fake_session),
            patch.object(reminders, "UserService", return_value=service),
            patch.object(
                reminders, "check_access",
                AsyncMock(return_value=AR(allowed=True, role="guest", remaining=3)),
            ),
            patch.object(reminders, "_disable_reminder", AsyncMock()) as disable,
            patch.object(
                reminders.ai_client, "chat_completion_result", AsyncMock()
            ) as ai,
        ):
            await reminders._fire_ai(due)
        ai.assert_not_awaited()
        disable.assert_awaited_once_with(4)


class AccessProbeTests(unittest.IsolatedAsyncioTestCase):
    """check_access(consume=False) semantics for operations that cost nothing."""

    async def test_require_quota_false_allows_exhausted_whitelisted_user(self):
        from bot.services.access import check_access

        service = SimpleNamespace(
            is_allowed_user=AsyncMock(return_value=True),
            get_daily_usage_remaining=AsyncMock(return_value=0),
            consume_daily_usage=AsyncMock(),
        )
        with patch.object(config, "admin_ids", []):
            probe = await check_access(service, 1, consume=False, require_quota=False)
            strict = await check_access(service, 1, consume=False)
        self.assertTrue(probe.allowed)
        self.assertEqual(probe.role, "allowed")
        self.assertFalse(strict.allowed)  # the quota-aware probe still refuses
        service.consume_daily_usage.assert_not_awaited()

    async def test_require_quota_false_still_denies_strangers(self):
        from bot.services.access import check_access

        service = SimpleNamespace(
            is_allowed_user=AsyncMock(return_value=False),
            get_guest_messages_remaining=AsyncMock(return_value=0),
        )
        with (
            patch.object(config, "admin_ids", []),
            patch.object(config, "allow_guest_access", False),
        ):
            probe = await check_access(service, 2, consume=False, require_quota=False)
        self.assertFalse(probe.allowed)
        self.assertEqual(probe.role, "denied")


class DocumentExpansionGuardTests(unittest.TestCase):
    """Archive bombs must be rejected on declared size, before parsing."""

    def _zip_bytes(self, payload: bytes) -> bytes:
        import zipfile

        buffer = BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("word/document.xml", payload)
        return buffer.getvalue()

    def test_highly_compressible_docx_is_flagged(self):
        from bot.services import documents

        data = self._zip_bytes(b"<x/>" * 2_000_000)  # ~8 MB from a few KB
        expanded = documents._docx_expanded_size(data)
        self.assertIsNotNone(expanded)
        self.assertGreater(expanded, 1024 * 1024)
        with patch.object(documents, "_MAX_EXPANDED_BYTES", 1024):
            self.assertIsNone(documents._extract_docx(data))

    def test_non_zip_input_is_not_flagged(self):
        from bot.services import documents

        self.assertIsNone(documents._docx_expanded_size(b"not a zip at all"))


class ImageDecodeGuardTests(unittest.TestCase):
    """The pixel cap must be applied before any full decode."""

    def test_oversized_image_rejected_before_exif_transpose(self):
        from bot.services import image_tools

        huge = SimpleNamespace(width=20000, height=20000)  # 400 Mpx
        opened = MagicMock()
        opened.__enter__ = MagicMock(return_value=huge)
        opened.__exit__ = MagicMock(return_value=False)
        with (
            patch.object(image_tools, "MAX_INPUT_PIXELS", 1000),
            patch("PIL.Image.open", return_value=opened),
            patch("PIL.ImageOps.exif_transpose", side_effect=AssertionError("decoded!")) as transpose,
        ):
            result, ops = image_tools._apply_sync(b"fake", [])
        self.assertIsNone(result)
        self.assertEqual(ops, [])
        transpose.assert_not_called()


class ModelFallbackChainTests(unittest.IsolatedAsyncioTestCase):
    """A non-model-specific HTTP error must not burn the whole chain."""

    async def test_auth_error_aborts_the_chain(self):
        from bot.services.ai_client import AIClient

        client = AIClient(api_key="test-key")
        with (
            patch.object(client, "model_chain", return_value=["a", "b", "c"]),
            patch.object(
                client, "_post_chat", AsyncMock(return_value=(401, None, 0.0))
            ) as post,
        ):
            result = await client.chat_completion_result([{"role": "user", "content": "x"}])
        self.assertIsNone(result.content)
        post.assert_awaited_once()  # not three times

    async def test_forbidden_aborts_the_chain_with_tools(self):
        from bot.services.ai_client import AIClient

        client = AIClient(api_key="test-key")
        with (
            patch.object(client, "model_chain", return_value=["a", "b", "c"]),
            patch.object(
                client, "_post_chat", AsyncMock(return_value=(403, None, 0.0))
            ) as post,
        ):
            await client.chat_with_tools([{"role": "user", "content": "x"}], tools=[])
        post.assert_awaited_once()

    async def test_transient_error_still_tries_every_model(self):
        """500 is model-agnostic in a different sense: the chain must survive."""
        from bot.services.ai_client import AIClient

        client = AIClient(api_key="test-key")
        with (
            patch.object(client, "model_chain", return_value=["a", "b", "c"]),
            patch.object(
                client, "_post_chat", AsyncMock(return_value=(500, None, 0.0))
            ) as post,
        ):
            await client.chat_completion_result([{"role": "user", "content": "x"}])
        self.assertEqual(post.await_count, 3)

    async def test_vision_chain_aborts_on_auth_error(self):
        from bot.services.ai_client import AIClient

        client = AIClient(api_key="test-key")
        with (
            patch.object(client, "model_chain", return_value=["v1", "v2"]),
            patch.object(
                client, "_post_chat", AsyncMock(return_value=(401, None, 0.0))
            ) as post,
        ):
            await client.chat_with_image_result("q", b"img", [], model="v1")
        post.assert_awaited_once()

    def test_fatal_status_helper(self):
        from bot.services.ai_client import _chain_fatal

        for status in (401, 403, 413):
            self.assertTrue(_chain_fatal(status), status)
        for status in (200, 400, 404, 422, 429, 500, 503, None):
            self.assertFalse(_chain_fatal(status), status)


class ExportLimitTests(unittest.TestCase):
    """The export renderer holds every row in memory: the cap must be bounded."""

    def test_absurd_export_cap_is_clamped(self):
        with patch.dict(os.environ, {"EXPORT_MAX_MESSAGES": "100000000"}):
            self.assertLessEqual(Config().export_max_messages, 5000)

    def test_zero_export_cap_is_clamped_up(self):
        with patch.dict(os.environ, {"EXPORT_MAX_MESSAGES": "0"}):
            self.assertGreaterEqual(Config().export_max_messages, 1)

    def test_sane_export_cap_is_kept(self):
        with patch.dict(os.environ, {"EXPORT_MAX_MESSAGES": "250"}):
            self.assertEqual(Config().export_max_messages, 250)


class GroupPrivacyTests(unittest.IsolatedAsyncioTestCase):
    """Personal settings must never be printed into a group chat."""

    @staticmethod
    def _group_message() -> SimpleNamespace:
        return SimpleNamespace(
            chat=SimpleNamespace(id=-100, type="group"),
            from_user=SimpleNamespace(id=1),
            text="/stats",
            answer=AsyncMock(),
        )

    async def test_helper_blocks_group_and_explains(self):
        message = self._group_message()
        self.assertFalse(await require_private_chat(message, what="/stats"))
        message.answer.assert_awaited_once()
        text = message.answer.call_args.args[0]
        self.assertIn("/stats", text)
        self.assertIn("private chat", text)

    async def test_helper_allows_private_quietly(self):
        message = SimpleNamespace(
            chat=SimpleNamespace(id=1, type="private"), answer=AsyncMock()
        )
        self.assertTrue(await require_private_chat(message, what="/stats"))
        message.answer.assert_not_awaited()

    async def test_private_only_commands_never_touch_the_database_in_groups(self):
        cases = (
            (stats, "cmd_stats"),
            (prompt, "cmd_prompt"),
            (prompt, "cmd_settings"),
            (models, "cmd_model"),
        )
        for module, handler_name in cases:
            with self.subTest(handler=handler_name):
                message = self._group_message()
                with (
                    patch.object(
                        module,
                        "require_private_chat",
                        AsyncMock(return_value=False),
                    ) as guard,
                    patch.object(module, "user_service") as service,
                ):
                    await getattr(module, handler_name)(message)
                guard.assert_awaited_once()
                service.assert_not_called()

    async def test_model_selection_callback_is_private_only(self):
        callback = SimpleNamespace(
            data="model:qwen3.8-flash",
            from_user=SimpleNamespace(id=1),
            message=SimpleNamespace(chat=SimpleNamespace(type="group")),
        )
        with patch.object(models, "safe_answer_callback", AsyncMock()) as ack:
            await models.cb_model_selected(callback)
        ack.assert_awaited_once()
        self.assertTrue(ack.call_args.kwargs.get("show_alert"))


if __name__ == "__main__":
    unittest.main()