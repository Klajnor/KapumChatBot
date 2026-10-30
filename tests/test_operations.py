"""Монитор операций: жизненный цикл, изоляция, приватность и права доступа."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, patch

from bot.config import config
from bot.handlers import admin, admin_panel, image, tasks
from bot.services import digest, operations, reminders
from bot.services.operations import (
    active_operations,
    current_operation,
    set_stage,
    stage,
    tracked,
)


def message(user_id=7, chat_type="private"):
    return SimpleNamespace(
        from_user=SimpleNamespace(id=user_id, full_name="<Admin>"),
        chat=SimpleNamespace(id=user_id, type=chat_type, title=None),
        text="SECRET PROMPT", answer=AsyncMock(),
    )


class OperationTests(IsolatedAsyncioTestCase):
    async def asyncTearDown(self):
        self.assertEqual(active_operations(), [])

    async def test_nested_stage_and_cleanup(self):
        @stage("Searching")
        async def search():
            self.assertEqual(len(active_operations()), 1)
            self.assertEqual(current_operation().stage, "Searching")

        @tracked("Inner")
        async def inner():
            await search()

        @tracked("Chat")
        async def run(message):
            operation = current_operation()
            self.assertEqual(operation.user_id, 7)
            set_stage("Waiting")
            await inner()
            self.assertEqual(operation.stage, "Waiting")
            self.assertEqual(operation.kind, "Chat")
            self.assertNotIn("SECRET PROMPT", repr(operation))

        await run(message())

    async def test_exception_cleanup(self):
        @tracked("Broken")
        @stage("Model")
        async def run(message):
            raise ValueError("secret error")

        with self.assertRaises(ValueError):
            await run(message())
        self.assertIsNone(current_operation())

    async def test_cancellation_cleanup(self):
        entered = asyncio.Event()

        @tracked("Slow")
        async def run(message):
            entered.set()
            await asyncio.Event().wait()

        task = asyncio.create_task(run(message()))
        await entered.wait()
        self.assertEqual(len(active_operations()), 1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

    async def test_child_task_does_not_inherit_parent_operation(self):
        @tracked("Child")
        async def child(message):
            self.assertEqual(len(active_operations()), 2)
            self.assertEqual(current_operation().user_id, 8)

        @tracked("Parent")
        async def parent(message):
            original = current_operation()
            other_message = SimpleNamespace(
                from_user=SimpleNamespace(id=8), chat=SimpleNamespace(id=8, type="private")
            )
            await asyncio.create_task(child(other_message))
            self.assertIs(current_operation(), original)
            self.assertEqual(len(active_operations()), 1)

        await parent(message())

    async def test_untracked_child_cannot_change_parent_stage(self):
        async def child():
            self.assertIsNone(current_operation())
            set_stage("Wrong")

        @tracked("Parent")
        async def run():
            await asyncio.create_task(child())
            self.assertEqual(current_operation().stage, "Processing")

        await run()

    async def test_background_metadata(self):
        @tracked("Reminder")
        async def reminder(due):
            operation = current_operation()
            self.assertEqual((operation.user_id, operation.chat_id), (42, -100))
            self.assertNotIn("secret", repr(operation))

        @tracked("Digest")
        async def digest(chat_id, title, owner_id=None):
            operation = current_operation()
            self.assertEqual((operation.user_id, operation.chat_id), (owner_id, -100))
            self.assertEqual(operation.location, "News")

        await reminder(SimpleNamespace(user_id=42, chat_id=-100, text="secret"))
        await digest(-100, "News")
        await digest(-100, "News", owner_id=7)

    async def test_callback_actor_is_not_bot_message_author(self):
        @tracked("Retry")
        async def run(callback):
            self.assertEqual(current_operation().user_id, 7)

        await run(SimpleNamespace(from_user=message().from_user, message=message(999)))

    async def test_guest_caller_metadata(self):
        @tracked("Guest")
        async def run(message):
            self.assertEqual(current_operation().user_id, 42)
            self.assertEqual(current_operation().chat_id, -100)

        event = message(999)
        event.guest_bot_caller_user = message(42).from_user
        event.guest_bot_caller_chat = SimpleNamespace(id=-100, title="Guest group")
        await run(event)

    async def test_pagination_escape_and_page_clamping(self):
        sample = [operations.Operation(
            id=i, kind="Chat", user_id=7, user_name="<User>", chat_id=-100,
            location="<Group>" + "x" * 500, started=0,
        ) for i in range(12)]
        with patch.object(tasks, "active_operations", return_value=sample):
            text, markup = tasks.render_tasks(7)
            self.assertIn("Page 1/3", text)
            self.assertIn("Your task", text)
            self.assertIn("&lt;User&gt;", text)
            self.assertNotIn("<Group>", text)
            self.assertLess(len(text), 4096)
            self.assertEqual(markup.inline_keyboard[0][-1].callback_data, "tasks:1")
            text, _ = tasks.render_tasks(8, 999999)
            self.assertIn("Page 3/3", text)
            self.assertNotIn("Your task", text)


class TaskAccessTests(IsolatedAsyncioTestCase):
    async def test_admin_menu_opens_monitor_and_cancels_pending_input(self):
        callback = SimpleNamespace(
            from_user=message().from_user, message=message(), data="adm:tasks",
        )
        with (
            patch.object(config, "owner_ids", [7]),
            patch.object(tasks, "safe_answer_callback", AsyncMock()),
            patch.object(tasks, "safe_edit_text", AsyncMock()) as edit,
            patch.dict(admin_panel._awaiting, {7: ("dgprompt", None)}),
        ):
            self.assertTrue(any(
                button.callback_data == "adm:tasks"
                for row in admin_panel._home_markup().inline_keyboard for button in row
            ))
            await admin_panel.cb_admin(callback)
            self.assertNotIn(7, admin_panel._awaiting)
            edit.assert_awaited_once()
            self.assertIn("Active tasks", edit.call_args.args[1])

    async def test_command_owner_and_granted_admin(self):
        with patch.object(config, "owner_ids", [7]), patch.object(config, "extra_admin_ids", {8}):
            for user_id in (7, 8):
                event = message(user_id)
                await tasks.cmd_tasks(event)
                self.assertIn("No active tasks", event.answer.call_args.args[0])

    async def test_command_denied_or_group_never_reads_registry(self):
        with (
            patch.object(config, "owner_ids", [7]),
            patch.object(config, "extra_admin_ids", set()),
            patch.object(tasks, "render_tasks") as render,
        ):
            await tasks.cmd_tasks(message(9))
            await tasks.cmd_tasks(message(7, "group"))
            render.assert_not_called()

    async def test_callback_rechecks_revocation_and_location(self):
        with (
            patch.object(config, "owner_ids", [7]),
            patch.object(config, "extra_admin_ids", set()),
            patch.object(tasks, "safe_answer_callback", AsyncMock()) as ack,
            patch.object(tasks, "safe_edit_text", AsyncMock()) as edit,
        ):
            for user_id, event in ((9, message()), (7, message(7, "group")), (7, None)):
                callback = SimpleNamespace(
                    from_user=message(user_id).from_user, message=event, data="tasks:0",
                )
                await tasks.cb_tasks(callback)
            self.assertEqual(ack.await_count, 3)
            edit.assert_not_awaited()

    async def test_malformed_callbacks(self):
        with (
            patch.object(config, "owner_ids", [7]),
            patch.object(tasks, "safe_answer_callback", AsyncMock()),
            patch.object(tasks, "safe_edit_text", AsyncMock()) as edit,
        ):
            for raw in ("-1", "²", "9" * 5000, "abc", ""):
                await tasks.cb_tasks(SimpleNamespace(
                    from_user=message().from_user, message=message(), data=f"tasks:{raw}",
                ))
            edit.assert_not_awaited()

    async def test_valid_callback_refresh(self):
        with (
            patch.object(config, "owner_ids", [7]),
            patch.object(tasks, "safe_answer_callback", AsyncMock()) as ack,
            patch.object(tasks, "safe_edit_text", AsyncMock()) as edit,
        ):
            await tasks.cb_tasks(SimpleNamespace(
                from_user=message().from_user, message=message(), data="tasks:999",
            ))
            ack.assert_awaited_once()
            edit.assert_awaited_once()
            self.assertIn("No active tasks", edit.call_args.args[1])


class OperationIntegrationTests(IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def pause(self, *args, **kwargs):
        self.entered.set()
        await self.release.wait()

    async def check_running(self, coroutine, kind, user_id, chat_id, stage_label=None):
        task = asyncio.create_task(coroutine)
        try:
            await asyncio.wait_for(self.entered.wait(), timeout=3)
            active = active_operations()
            self.assertEqual(len(active), 1)
            operation = active[0]
            self.assertEqual(operation.kind, kind)
            self.assertEqual(operation.user_id, user_id)
            self.assertEqual(operation.chat_id, chat_id)
            if stage_label:
                self.assertIn(stage_label, operation.stage)
        finally:
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertEqual(active_operations(), [])

    async def test_real_image_entry(self):
        with patch.object(image, "_generate", side_effect=self.pause):
            await self.check_running(
                image._run_generation(message(), AsyncMock(), "SECRET"),
                "Image / sticker generation", 7, 7,
            )

    async def test_real_digest_test_entry_and_stage(self):
        with (
            patch.object(digest, "get_model", AsyncMock(return_value="model")),
            patch.object(digest, "get_prompt", AsyncMock(return_value="secret")),
            patch.object(digest.ai_client, "chat_completion_result", side_effect=self.pause),
            patch.object(digest, "get_vary", AsyncMock(return_value=False)),
            patch.object(digest, "get_temperature", AsyncMock(return_value=0.5)),
        ):
            await self.check_running(
                digest.run_test([], 7), "Digest test", 7, 7, "Generating digest",
            )

    async def test_real_reminder_entry(self):
        due = reminders.DueReminder(
            id=1, user_id=42, chat_id=-100, text="SECRET", ai_mode=True, schedule=None,
        )
        with patch.object(reminders, "_fire_ai", side_effect=self.pause):
            await self.check_running(
                reminders.fire_reminder(due), "Reminder", 42, -100,
            )

    async def test_real_broadcast_progress(self):
        bot = SimpleNamespace(send_message=AsyncMock(side_effect=self.pause))
        await self.check_running(
            admin._deliver_broadcast(bot, [1, 2], "SECRET", None, owner_id=7),
            "Broadcast", 7, 7, "0/2",
        )
