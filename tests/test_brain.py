"""Запись во второй мозг: права, намерение, источники и ошибки."""

from __future__ import annotations

import datetime
import os
import time
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

os.environ.setdefault("BOT_TOKEN", "123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghi")
os.environ.setdefault("AI_API_KEY", "test-key")

from aiogram.types import CallbackQuery, Chat, Message, User

from bot.config import config
from bot.handlers import brain
from bot.keyboards.inline import get_answer_keyboard
from bot.services.mcp import MCPClient, MCPError, MCPManager, MCPServerConfig, MCPTool


def message(text: str = "запомни факт", user_id: int = 42) -> Message:
    return Message(
        message_id=10, date=datetime.datetime.now(datetime.UTC), text=text,
        chat=Chat(id=user_id, type="private"),
        from_user=User(id=user_id, is_bot=False, first_name="Owner"),
    )


class BrainTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        brain._pending.clear()
        self.patches = [
            patch.object(config, "enable_mcp", True),
            patch.object(config, "mcp_servers", {"brain": {"command": "x"}}),
            patch.object(config, "mcp_write_server", "brain"),
            patch.object(config, "owner_ids", [42]),
        ]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    async def test_non_owner_never_calls_intent_or_write(self) -> None:
        with patch.object(brain.ai_client, "chat_completion_result", new_callable=AsyncMock) as ai:
            self.assertFalse(await brain.maybe_remember(message(user_id=43), "запомни факт"))
            ai.assert_not_awaited()
        manager = MCPManager()
        with patch.object(manager, "_client") as client:
            with self.assertRaises(MCPError):
                await manager.save_note(43, "fact", "id")
            client.assert_not_called()

    async def test_incomplete_chat_metadata_fails_closed(self) -> None:
        incomplete = SimpleNamespace(from_user=SimpleNamespace(id=42), chat=SimpleNamespace(id=42))
        with patch.object(brain.ai_client, "chat_completion_result", new_callable=AsyncMock) as ai:
            self.assertFalse(await brain.maybe_remember(incomplete, "запомни факт"))
            ai.assert_not_awaited()

    async def test_intent_saves_exact_text_without_public_thinking(self) -> None:
        service = SimpleNamespace(get_user_settings=AsyncMock(return_value={"model": "test"}))
        context = AsyncMock()
        context.__aenter__.return_value = service
        with (
            patch.object(brain, "user_service", return_value=context),
            patch.object(brain.ai_client, "chat_completion_result", new_callable=AsyncMock,
                         return_value=SimpleNamespace(content='{"save":true,"content":"факт"}')) as ai,
            patch.object(brain, "_save", new_callable=AsyncMock) as save,
        ):
            self.assertTrue(await brain.maybe_remember(message(), "запомни факт"))
            save.assert_awaited_once()
            self.assertEqual(save.call_args.args[2], "факт")
            self.assertNotIn("public", ai.call_args.kwargs)

    async def test_intent_failure_does_not_fall_through_to_chat(self) -> None:
        context = AsyncMock()
        context.__aenter__.return_value.get_user_settings.return_value = {"model": "test"}
        with (
            patch.object(brain, "user_service", return_value=context),
            patch.object(brain.ai_client, "chat_completion_result", new_callable=AsyncMock,
                         return_value=SimpleNamespace(content="broken")),
            patch.object(brain, "safe_answer", new_callable=AsyncMock),
            patch.object(brain, "_save", new_callable=AsyncMock) as save,
        ):
            self.assertTrue(await brain.maybe_remember(message(), "запомни факт"))
            save.assert_not_awaited()

    async def test_write_tool_never_exposed_even_with_wildcard(self) -> None:
        manager = MCPManager()
        client = AsyncMock()
        read = MCPTool("brain", "search_notes", "", {}, "mcp__brain__search_notes")
        write = MCPTool("brain", "create_note", "", {}, "mcp__brain__create_note")
        manager._tools = {read.exposed_name: (read, client), write.exposed_name: (write, client)}
        manager._loaded = True
        tools = await manager.openai_tools()
        self.assertEqual([t["function"]["name"] for t in tools], [read.exposed_name])
        result = await manager.call_tool(write.exposed_name, {"content": "bad"})
        self.assertIn("только чтение", result)
        client.call_tool.assert_not_awaited()

    async def test_create_note_error_is_not_success(self) -> None:
        client = MCPClient(MCPServerConfig("brain", "x"), timeout=1)
        with (
            patch.object(client, "start", new_callable=AsyncMock),
            patch.object(client, "_request", new_callable=AsyncMock,
                         return_value={"isError": True, "content": []}),
            self.assertRaises(MCPError),
        ):
            await client.create_note("fact", "id")

    async def test_write_error_has_safe_actionable_diagnostic(self) -> None:
        client = MCPClient(MCPServerConfig("brain", "x"), timeout=1)
        cases = [
            ("Unknown tool: create_note", "BRAIN_WRITE_ENABLED"),
            ("[Errno 13] Permission denied: /secret/vault", "нет прав"),
            ("Ошибка Git (fetch): сбой DNS Git; https://user:secret-token@host", "DNS"),
            ("Ошибка Git (commit-tree): не настроены user.name/user.email", "user.name"),
            ("Ошибка Git (merge-base); заметка сохранена локально", "отстаёт"),
            ("Ошибка Git (push); https://user:secret-token@host", "git push"),
            ("Ошибка Git (rev-parse): https://user:secret-token@host", "стадии rev-parse"),
            ("Command '['git']' returned non-zero exit status 128", "обновите ObsidianBrain"),
            ("sqlite3.OperationalError: unable to open database file", "SQLite-индекс"),
            ("sqlite3.IntegrityError: UNIQUE constraint failed: notes.path", "параллельно"),
            ("stage=index: OperationalError", "сбой индексации SQLite"),
            ("stage=file: PermissionError", "создать файл"),
            ("stage=git: RuntimeError", "сбой Git"),
            ("stage=sync: RuntimeError", "синхронизировать хранилище"),
            ("stage=sync: RuntimeError: Ошибка Git (fetch): аутентификация Git не удалась", "аутентификация"),
            ("stage=sync: RuntimeError: Ошибка Git (fetch): сбой DNS Git", "DNS"),
            ("stage=sync: RuntimeError: Ошибка Git (fetch): Host key verification failed", "аутентификация"),
            ("stage=sync: RuntimeError: Ошибка Git (fetch): причина скрыта", "git fetch"),
            ("sqlite3.OperationalError: no such table: notes", "схема SQLite"),
            ("sqlite3.OperationalError: database or disk is full", "нет места"),
            ("fatal: cannot lock ref 'refs/heads/master'", "Git занят"),
            ("Unknown problem secret-token /secret/vault", "неизвестная ошибка"),
        ]
        for error, expected in cases:
            with (
                self.subTest(error=expected),
                patch.object(client, "start", new_callable=AsyncMock),
                patch.object(client, "_request", new_callable=AsyncMock, return_value={
                    "isError": True, "content": [{"type": "text", "text": error}],
                }),
                self.assertRaises(MCPError) as caught,
            ):
                await client.create_note("private note content", "id")
            diagnostic = str(caught.exception)
            self.assertIn(expected, diagnostic)
            self.assertNotIn("secret-token", diagnostic)
            self.assertNotIn("/secret/vault", diagnostic)
            self.assertNotIn("private note content", diagnostic)

    async def test_save_shows_safe_mcp_reason(self) -> None:
        with (
            patch.object(brain.mcp_manager, "save_note", new_callable=AsyncMock,
                         side_effect=MCPError("Не удалось создать заметку: ветка отстаёт")),
            patch.object(brain, "safe_answer", new_callable=AsyncMock) as answer,
        ):
            self.assertFalse(await brain._save(message(), 42, "text", "id"))
        body = answer.call_args.args[1]
        self.assertIn("ветка отстаёт", body)
        self.assertNotIn("Traceback", body)

    async def test_transport_error_is_sanitized_for_create_note(self) -> None:
        client = MCPClient(MCPServerConfig("brain", "x"), timeout=1)
        with (
            patch.object(client, "start", new_callable=AsyncMock),
            patch.object(client, "_request", new_callable=AsyncMock,
                         side_effect=MCPError("Unknown tool: create_note https://user:secret-token@host")),
            self.assertRaises(MCPError) as caught,
        ):
            await client.create_note("private note", "id")
        diagnostic = str(caught.exception)
        self.assertIn("BRAIN_WRITE_ENABLED", diagnostic)
        self.assertNotIn("secret-token", diagnostic)

    async def test_start_error_is_sanitized_for_create_note(self) -> None:
        client = MCPClient(MCPServerConfig("brain", "x"), timeout=1)
        with (
            patch.object(client, "start", new_callable=AsyncMock,
                         side_effect=MCPError("не удалось запустить MCP-сервер 'brain': /secret/python")),
            self.assertRaises(MCPError) as caught,
        ):
            await client.create_note("private note", "id")
        diagnostic = str(caught.exception)
        self.assertIn("не запущен", diagnostic)
        self.assertNotIn("/secret/python", diagnostic)

    async def test_structured_server_error_is_classified(self) -> None:
        client = MCPClient(MCPServerConfig("brain", "x"), timeout=1)
        for error, expected in [
            ("stage=index: OperationalError", "сбой индексации SQLite"),
            ("stage=internal: PermissionError", "внутренняя ошибка MCP-сервера (permissionerror)"),
            ("stage=validate: ValueError", "некорректные данные"),
        ]:
            with (
                self.subTest(expected=expected),
                patch.object(client, "start", new_callable=AsyncMock),
                patch.object(client, "_request", new_callable=AsyncMock, return_value={
                    "content": [], "structuredContent": {"git_pushed": False, "error": error},
                }),
                self.assertRaises(MCPError) as caught,
            ):
                await client.create_note("fact", "id")
            self.assertIn(expected, str(caught.exception))

    async def test_create_note_requires_git_push_confirmation(self) -> None:
        client = MCPClient(MCPServerConfig("brain", "x"), timeout=1)
        for payload, success in [
            ({"content": [{"type": "text", "text": '{"path":"vault/Telegram/n.md"}'}]}, False),
            ({"content": [], "structuredContent": {"git_pushed": False}}, False),
            ({"content": [{"type": "text", "text": '{"git_pushed":true}'}]}, True),
            ({"content": [], "structuredContent": {"git_pushed": True}}, True),
        ]:
            with (
                patch.object(client, "start", new_callable=AsyncMock),
                patch.object(client, "_request", new_callable=AsyncMock, return_value=payload) as request,
            ):
                if success:
                    await client.create_note("fact", "id")
                else:
                    with self.assertRaises(MCPError):
                        await client.create_note("fact", "id")
                self.assertGreaterEqual(request.call_args.kwargs["timeout"], 180)

    async def test_forward_choice_bound_to_owner_and_expires(self) -> None:
        origin = message("forwarded text")
        brain._pending["token"] = brain.PendingForward(origin, 0)
        callback = CallbackQuery(id="c", from_user=origin.from_user,
                                 chat_instance="c", message=message(), data="brain:forward:token")
        with (
            patch.object(brain, "safe_answer_callback", new_callable=AsyncMock),
            patch.object(brain, "safe_answer", new_callable=AsyncMock),
            patch.object(brain, "_save", new_callable=AsyncMock) as save,
        ):
            await brain.brain_callback(callback)
            save.assert_not_awaited()

    async def test_answer_callback_reloads_owned_assistant_row(self) -> None:
        context = AsyncMock()
        context.__aenter__.return_value.get_message_for_export.return_value = None
        callback = CallbackQuery(id="c", from_user=message().from_user,
                                 chat_instance="c", message=message(), data="brain:answer:999")
        with (
            patch.object(brain, "user_service", return_value=context),
            patch.object(brain, "safe_answer_callback", new_callable=AsyncMock),
            patch.object(brain, "safe_answer", new_callable=AsyncMock),
            patch.object(brain, "_save", new_callable=AsyncMock) as save,
        ):
            await brain.brain_callback(callback)
            save.assert_not_awaited()
            context.__aenter__.return_value.get_message_for_export.assert_awaited_once_with(999, 42)

    async def test_forward_save_uses_original_text_not_button_message(self) -> None:
        origin = message("original forwarded text")
        brain._pending["token"] = brain.PendingForward(origin, time.monotonic() + 60)
        callback = CallbackQuery(id="c", from_user=origin.from_user, chat_instance="c",
                                 message=message("choice prompt"), data="brain:forward:token")
        with (
            patch.object(brain, "safe_answer_callback", new_callable=AsyncMock),
            patch.object(brain, "_save", new_callable=AsyncMock, return_value=True) as save,
        ):
            await brain.brain_callback(callback)
            self.assertEqual(save.call_args.args[2], "original forwarded text")
            self.assertNotIn("token", brain._pending)

    async def test_analyze_delegates_to_existing_handler_without_saving(self) -> None:
        from bot.handlers import chat

        origin = message("original")
        brain._pending["token"] = brain.PendingForward(origin, time.monotonic() + 60)
        callback = CallbackQuery(id="c", from_user=origin.from_user, chat_instance="c",
                                 message=message(), data="brain:analyze:token")
        with (
            patch.object(brain, "safe_answer_callback", new_callable=AsyncMock),
            patch.object(brain, "_save", new_callable=AsyncMock) as save,
            patch.object(chat, "handle_text", new_callable=AsyncMock) as analyze,
        ):
            await brain.brain_callback(callback)
            analyze.assert_awaited_once_with(origin)
            save.assert_not_awaited()

    async def test_non_owner_callback_cannot_save(self) -> None:
        callback = CallbackQuery(id="c", from_user=message(user_id=43).from_user,
                                 chat_instance="c", message=message(user_id=43), data="brain:answer:1")
        with (
            patch.object(brain, "safe_answer_callback", new_callable=AsyncMock),
            patch.object(brain, "_save", new_callable=AsyncMock) as save,
            patch.object(brain, "user_service") as service,
        ):
            await brain.brain_callback(callback)
            save.assert_not_awaited()
            service.assert_not_called()

    async def test_successful_answer_save_uses_stored_raw_answer(self) -> None:
        context = AsyncMock()
        context.__aenter__.return_value.get_message_for_export.return_value = SimpleNamespace(
            role="assistant", chat_id=42, content="**Stored answer**, without timing footer",
        )
        callback = CallbackQuery(id="c", from_user=message().from_user,
                                 chat_instance="c", message=message("visible fragment"), data="brain:answer:7")
        with (
            patch.object(brain, "user_service", return_value=context),
            patch.object(brain, "safe_answer_callback", new_callable=AsyncMock),
            patch.object(brain, "_save", new_callable=AsyncMock) as save,
        ):
            await brain.brain_callback(callback)
            self.assertEqual(save.call_args.args[2], "**Stored answer**, without timing footer")
            self.assertEqual(save.call_args.args[3], "answer:42:7")

    def test_remember_button_without_export(self) -> None:
        keyboard = get_answer_keyboard(remember_id=123)
        self.assertEqual(keyboard.inline_keyboard[0][0].callback_data, "brain:answer:123")
