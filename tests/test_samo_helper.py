"""samo_helper в Kapum: изоляция, ACL, команды и настройки без сети."""

from __future__ import annotations

import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch

from aiogram import Bot, Dispatcher, Router
from aiogram.types import CallbackQuery, Chat, Message, Update, User

from bot.samo_helper.access import private_owner_message, samo_callback
from bot.samo_helper.config import SamoHelperConfig, load_samo_config
from bot.samo_helper.handlers import cmd_find
from bot.samo_helper.integration import attach_samo_helper
from bot.samo_helper.knowledge import KnowledgeSource, SummaryHit
from bot.samo_helper.storage import Storage

TOKEN = "123456789:AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"


def config(tmp_path: Path, **overrides) -> SamoHelperConfig:
    values = {
        "owner_id": 42,
        "notification_chat_id": -1001234567890,
        "redmine_url": "https://redmine.example",
        "redmine_api_key": "secret",
        "db_path": str(tmp_path / "samo.db"),
        "knowledge_url": "https://knowledge.internal",
        "knowledge_api_key": "knowledge-secret",
        "knowledge_allowed_hosts": ("redmine.example", "wiki.example"),
        "wiki_url": "https://wiki.example",
    }
    values.update(overrides)
    return SamoHelperConfig(**values)


def message(chat_id=42, chat_type="private", user_id=42, text="Привет", **kwargs):
    return Message(
        message_id=10,
        date=datetime.now(UTC),
        chat=Chat(id=chat_id, type=chat_type),
        from_user=User(id=user_id, is_bot=user_id == 123456789, first_name="Тест"),
        text=text,
        **kwargs,
    )


class ConfigTests(unittest.TestCase):
    def test_disabled_by_default_in_kapum(self):
        self.assertIsNone(load_samo_config(42, str(Path(tempfile.gettempdir()) / "samo.db"), environ={}))
        self.assertIsNone(load_samo_config(
            42, str(Path(tempfile.gettempdir()) / "samo.db"), environ={"SAMO_HELPER_ENABLED": "false"},
        ))

    def test_prefixed_settings_and_allowlist(self):
        cfg = load_samo_config(42, str(Path(tempfile.gettempdir()) / "samo.db"), environ={
            "SAMO_HELPER_ENABLED": "true",
            "SAMO_HELPER_REDMINE_URL": "https://redmine.example",
            "SAMO_HELPER_REDMINE_API_KEY": "secret",
            "SAMO_HELPER_NOTIFICATION_CHAT_ID": "-100123",
            "SAMO_HELPER_KNOWLEDGE_URL": "https://knowledge.internal",
            "SAMO_HELPER_WIKI_URL": "https://wiki.example",
            "SAMO_HELPER_ALLOWED_HOSTS": "extra.example",
            "SAMO_HELPER_LLM_API_URL": "https://llm.internal/v1",
            "SAMO_HELPER_LLM_MODEL": "internal",
        })
        self.assertIsNotNone(cfg)
        self.assertEqual(cfg.owner_id, 42)
        self.assertEqual(cfg.knowledge_url, "https://knowledge.internal")
        self.assertIn("redmine.example", cfg.knowledge_allowed_hosts)
        self.assertIn("wiki.example", cfg.knowledge_allowed_hosts)
        self.assertIn("extra.example", cfg.knowledge_allowed_hosts)
        self.assertTrue(cfg.llm_enabled)
        self.assertNotIn("secret", repr(cfg))

    def test_invalid_or_relative_paths(self):
        base = {
            "SAMO_HELPER_ENABLED": "true",
            "SAMO_HELPER_REDMINE_URL": "https://redmine.example",
            "SAMO_HELPER_REDMINE_API_KEY": "secret",
            "SAMO_HELPER_NOTIFICATION_CHAT_ID": "-100123",
        }
        with self.assertRaisesRegex(RuntimeError, "абсолютным"):
            load_samo_config(42, "relative.db", environ=base)
        with self.assertRaisesRegex(RuntimeError, "NOTIFICATION"):
            load_samo_config(42, str(Path(tempfile.gettempdir()) / "samo.db"), environ={
                **base, "SAMO_HELPER_NOTIFICATION_CHAT_ID": "42",
            })
        with self.assertRaisesRegex(RuntimeError, "вместе"):
            load_samo_config(42, str(Path(tempfile.gettempdir()) / "samo.db"), environ={
                **base, "SAMO_HELPER_LLM_API_URL": "https://llm.internal",
            })


class AccessTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.cfg = SamoHelperConfig(
            owner_id=42, notification_chat_id=-1001234567890,
            redmine_url="https://redmine.example", redmine_api_key="secret",
            db_path=str(Path(tempfile.gettempdir()) / "samo-test.db"),
        )
        self.bot = Bot(TOKEN)

    async def asyncTearDown(self):
        await self.bot.session.close()

    def test_private_owner_only(self):
        self.assertTrue(private_owner_message(message(), self.cfg))
        self.assertFalse(private_owner_message(message(user_id=99), self.cfg))
        self.assertFalse(private_owner_message(
            message(chat_id=-1001234567890, chat_type="supergroup"), self.cfg,
        ))

    def test_group_callback_owner_and_bot_only(self):
        def allowed(sender=42, author=123456789, chat_id=-1001234567890,
                    chat_type="supergroup", data="samo:rm:1"):
            callback = CallbackQuery(
                id="cb", chat_instance="test", data=data,
                from_user=User(id=sender, is_bot=False, first_name="Тест"),
                message=message(chat_id=chat_id, chat_type=chat_type, user_id=author),
            )
            return samo_callback(callback, self.cfg.owner_id,
                                 self.cfg.notification_chat_id, self.bot)

        self.assertTrue(allowed())
        self.assertTrue(allowed(data="samo:done"))
        self.assertFalse(allowed(sender=99))
        self.assertFalse(allowed(author=1))
        self.assertFalse(allowed(chat_id=-999))
        self.assertFalse(allowed(data="other"))


class IntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Storage(str(Path(self.tmp.name) / "samo.db"))
        self.bot = Bot(TOKEN)
        self.dp = Dispatcher()
        self.cfg = config(Path(self.tmp.name))
        self.extension = attach_samo_helper(self.dp, self.bot, self.cfg, self.db)
        self.host_calls = []
        host = Router(name="host_chat")

        @host.message()
        async def host_message(message: Message):
            self.host_calls.append(message.text)

        @host.callback_query()
        async def host_callback(callback: CallbackQuery):
            self.host_calls.append(callback.data)

        self.dp.include_router(host)

    async def asyncTearDown(self):
        await self.extension.stop()
        await self.dp.storage.close()
        self.db.close()
        await self.bot.session.close()
        self.tmp.cleanup()

    async def test_router_registered_before_host_catch_all(self):
        self.assertEqual(
            [router.name for router in self.dp.sub_routers],
            ["samo_helper", "host_chat"],
        )

    async def test_owner_private_command_reaches_work_handler(self):
        with patch("bot.samo_helper.handlers.owner_message", new_callable=AsyncMock) as transfer:
            await self.dp.feed_update(self.bot, Update(
                update_id=1, message=message(text="/samo_add Задача"),
            ))
        transfer.assert_awaited_once()
        self.assertEqual(transfer.call_args.kwargs["task_text"], "Задача")
        self.assertEqual(self.host_calls, [])

    async def test_group_and_foreign_commands_are_swallowed(self):
        with (
            patch("bot.samo_helper.handlers.owner_message", new_callable=AsyncMock) as transfer,
            patch("bot.samo_helper.handlers.answer_chunks", new_callable=AsyncMock),
        ):
            for update_id, msg in enumerate((
                message(chat_id=-1001234567890, chat_type="supergroup", text="/samo_add групповое"),
                message(user_id=99, text="/samo_add чужое"),
                message(text="/samo_typo рабочее"),
            ), 1):
                await self.dp.feed_update(self.bot, Update(update_id=update_id, message=msg))
        transfer.assert_not_awaited()
        self.assertEqual(self.host_calls, [])

    async def test_callback_reminder_and_done_are_namespaced(self):
        source = message(chat_id=-1001234567890, chat_type="supergroup", user_id=123456789)
        callback = CallbackQuery(
            id="cb", chat_instance="test", data="samo:rm:1", message=source,
            from_user=User(id=42, is_bot=False, first_name="Тест"),
        )
        with patch.object(CallbackQuery, "answer", new_callable=AsyncMock):
            await self.dp.feed_update(self.bot, Update(
                update_id=5, callback_query=callback,
            ))
        rows = self.db.due_reminders(2**40)
        self.assertEqual(len(rows), 1)
        self.assertEqual(self.host_calls, [])

    async def test_extension_lifecycle_is_idempotent(self):
        from bot.samo_helper.poller import RedminePoller
        from bot.samo_helper.reminders import ReminderService

        with (
            patch.object(RedminePoller, "_run", new_callable=AsyncMock),
            patch.object(ReminderService, "_run", new_callable=AsyncMock),
        ):
            await self.extension.start()
            first = self.extension.plugin.poller._task
            await self.extension.start()
            self.assertIs(self.extension.plugin.poller._task, first)
            await self.extension.stop()
            self.assertIsNone(self.extension.plugin.poller._task)
            self.assertIsNone(self.extension.reminders._task)
            await self.extension.start()
            self.assertIsNotNone(self.extension.plugin.poller._task)
            await self.extension.stop()

    async def test_legacy_callbacks_belong_to_host_by_default(self):
        source = message(chat_id=-1001234567890, chat_type="supergroup", user_id=123456789)
        callback = CallbackQuery(
            id="cb", chat_instance="test", data="rm:1", message=source,
            from_user=User(id=42, is_bot=False, first_name="Тест"),
        )
        await self.dp.feed_update(self.bot, Update(update_id=6, callback_query=callback))
        self.assertEqual(self.host_calls, ["rm:1"])


class HandlerTests(unittest.IsolatedAsyncioTestCase):
    async def test_find_uses_summary_and_sources(self):
        cfg = SamoHelperConfig(
            owner_id=42, notification_chat_id=-1001234567890,
            redmine_url="https://redmine.example", redmine_api_key="secret",
            db_path=str(Path(tempfile.gettempdir()) / "samo-test.db"),
        )
        hit = SummaryHit(
            "s1", "Задача", "Кратко о кэше", ("кэш",),
            (KnowledgeSource("redmine", "1", "Задача",
                             "https://redmine.example/issues/1"),),
        )
        knowledge = AsyncMock()
        knowledge.search.return_value = [hit]
        with patch("bot.samo_helper.handlers.answer_chunks", new_callable=AsyncMock) as answer:
            await cmd_find(message(text="/samo_find кэш"), cfg, knowledge)
        text = answer.call_args.args[1]
        self.assertIn("Кратко о кэше", text)
        self.assertIn("https://redmine.example/issues/1", text)


class PollerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Storage(str(Path(self.tmp.name) / "samo.db"))
        self.cfg = config(Path(self.tmp.name))
        self.bot = AsyncMock()
        self.redmine = AsyncMock()
        self.redmine.issue_url = lambda issue_id: f"https://redmine.example/issues/{issue_id}"
        self.redmine.customer_name.return_value = ""

    async def asyncTearDown(self):
        self.db.close()
        self.tmp.cleanup()

    async def test_notifications_only_to_group_with_explicit_markdown(self):
        from bot.samo_helper.poller import RedminePoller
        poller = RedminePoller(
            self.bot, self.cfg, self.db, self.redmine,
        )
        await poller._notify("Текст " * 1500)
        self.assertGreater(self.bot.send_message.await_count, 1)
        for call in self.bot.send_message.call_args_list:
            self.assertEqual(call.args[0], self.cfg.notification_chat_id)
            self.assertEqual(call.kwargs["parse_mode"].value, "MarkdownV2")

    async def test_knowledge_summary_and_sources_in_new_issue(self):
        from bot.samo_helper.poller import RedminePoller
        knowledge = AsyncMock()
        knowledge.get_summary.return_value = SummaryHit(
            "s1", "Задача", "Готовое саммари", ("кэш",),
            (KnowledgeSource("redmine", "1", "Задача",
                             "https://wiki.example/page/1"),),
        )
        poller = RedminePoller(
            self.bot, self.cfg, self.db, self.redmine, None, None, knowledge,
        )
        await poller._notify_new_issue({"id": 123, "subject": "Задача"}, None)
        text = self.bot.send_message.call_args.args[1]
        self.assertIn("Готовое саммари", text)
        self.assertIn("https://wiki.example/page/1", text)
        self.assertEqual(self.bot.send_message.call_args.args[0],
                         self.cfg.notification_chat_id)

    async def test_reminder_goes_to_owner_with_html_links(self):
        from bot.samo_helper.reminders import ReminderService
        self.db.upsert_reminder(
            self.cfg.notification_chat_id, 10,
            '<a href="https://redmine.example/issues/1">Задача</a>', 0,
        )
        await ReminderService(self.bot, self.cfg.owner_id, self.db).tick()
        call = self.bot.send_message.call_args
        self.assertEqual(call.args[0], self.cfg.owner_id)
        self.assertEqual(call.kwargs["parse_mode"].value, "HTML")
        self.assertEqual(self.db.due_reminders(1), [])


class PluginTests(unittest.IsolatedAsyncioTestCase):
    async def test_public_llm_is_rejected(self):
        from bot.samo_helper.plugin import SamoHelperPlugin
        tmp = tempfile.TemporaryDirectory()
        db = Storage(str(Path(tmp.name) / "samo.db"))
        bot = Bot(TOKEN)
        cfg = config(
            Path(tmp.name), llm_api_url="https://api.openai.com/v1",
            llm_model="public",
        )
        try:
            with self.assertRaisesRegex(ValueError, "публичный"):
                SamoHelperPlugin(bot, cfg, db)
        finally:
            db.close()
            await bot.session.close()
            tmp.cleanup()
