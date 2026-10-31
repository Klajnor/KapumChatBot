"""Digest test button: collect messages in a private chat, summarise on demand."""

from __future__ import annotations

import datetime
import io
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from aiogram.types import Chat, Message, User

from bot.config import config
from bot.handlers import admin_panel as panel
from bot.services import digest
from bot.services.ai_client import ChatResult
from tests.test_database import DBTest

ADMIN = 7
NOW = datetime.datetime(2026, 10, 25, 12, 0, tzinfo=datetime.UTC)


def _dm(text: str, user_id: int = ADMIN, chat_type: str = "private") -> Message:
    return Message(
        message_id=1,
        date=NOW,
        chat=Chat(id=user_id, type=chat_type),
        from_user=User(id=user_id, is_bot=False, first_name="Ann"),
        text=text,
    )


def _callback(chat_type: str = "private") -> SimpleNamespace:
    status = SimpleNamespace(
        chat=SimpleNamespace(type=chat_type), edit_text=AsyncMock()
    )
    return SimpleNamespace(from_user=SimpleNamespace(id=ADMIN), message=status)


def _prompt_document(data: bytes, name: str = "prompt.txt") -> SimpleNamespace:
    return SimpleNamespace(
        from_user=SimpleNamespace(id=ADMIN),
        chat=SimpleNamespace(id=ADMIN, type="private"),
        document=SimpleNamespace(
            file_id="f",
            file_unique_id="u",
            file_name=name,
            mime_type="text/plain",
            file_size=len(data),
        ),
        answer=AsyncMock(),
        bot=SimpleNamespace(
            get_file=AsyncMock(return_value=SimpleNamespace(file_path="prompt.txt")),
            download_file=AsyncMock(return_value=io.BytesIO(data)),
        ),
    )


class TestDigestTestButton(DBTest):
    def _run(self, coro_fn):
        patches = [
            patch("bot.services.digest.async_session_factory", self._factory),
            patch.object(config, "owner_ids", [ADMIN]),
            patch.object(config, "extra_admin_ids", set()),
            patch.object(panel, "_show", AsyncMock()),
            patch.object(panel, "safe_answer_callback", AsyncMock()),
        ]
        for p in patches:
            p.start()
        panel._test_state.clear()
        try:
            return self._loop.run_until_complete(coro_fn())
        finally:
            panel._test_state.clear()
            panel._awaiting.pop(ADMIN, None)
            for p in reversed(patches):
                p.stop()

    def test_collects_messages_then_summarises_for_the_admin_only(self):
        async def go() -> None:
            callback = _callback()
            await panel._digest_test_action(callback, ADMIN, "dgq")
            self.assertTrue(panel._collecting_test(_dm("hi")))
            self.assertFalse(panel._collecting_test(_dm("/start")))
            self.assertFalse(panel._collecting_test(_dm("hi", chat_type="supergroup")))

            for text in ("first", "second"):
                await panel.on_test_message(_dm(text))
            samples = panel._test_state[ADMIN]["samples"]
            self.assertEqual([s.text for s in samples], ["first", "second"])
            self.assertEqual(samples[0].author, "Ann")

            seen: dict = {}

            async def fake_ask(messages, **kwargs):
                seen["messages"] = messages
                return ChatResult(
                    content="done", usage={"prompt_tokens": 50, "completion_tokens": 9}
                )

            with (
                patch.object(digest.ai_client, "chat_completion_result", fake_ask),
                patch.object(digest, "send_rich_to", AsyncMock(return_value=True)) as rich,
                patch.object(digest, "notify_admins", AsyncMock()) as stats,
                patch.object(panel, "_digest_show", AsyncMock()),
            ):
                await panel._digest_test_action(callback, ADMIN, "dgqr")

            self.assertIn("second", seen["messages"][1]["content"])
            self.assertEqual(rich.await_args.args[1], ADMIN)  # sent to the admin's DM
            self.assertEqual(stats.await_args.kwargs["only"], ADMIN)
            self.assertNotIn(ADMIN, panel._test_state)

        self._run(go)

    def test_refuses_outside_a_private_chat_and_when_empty(self):
        async def go() -> None:
            await panel._digest_test_action(_callback("supergroup"), ADMIN, "dgq")
            self.assertNotIn(ADMIN, panel._test_state)

            await panel._digest_test_action(_callback(), ADMIN, "dgq")
            with patch.object(digest, "run_test", AsyncMock()) as run:
                await panel._digest_test_action(_callback(), ADMIN, "dgqr")
            run.assert_not_awaited()  # nothing collected yet

        self._run(go)

    def test_digest_screen_renders_within_the_telegram_limit(self):
        async def go() -> None:
            await digest.set_setting(digest.SETTING_PROMPT, "x" * panel.MAX_PROMPT_CHARS)
            await digest.set_setting(digest.SETTING_MODIFIER, "y" * panel.MAX_PROMPT_CHARS)
            text, markup = await panel._digest_screen("notice")
            self.assertLess(len(text), 4096)
            self.assertIn("Prompt variation", text)
            buttons = [b.callback_data for row in markup.inline_keyboard for b in row]
            for expected in ("adm:dgv", "adm:dgk", "adm:dgmp", "adm:dgmd"):
                self.assertIn(expected, buttons)

        self._run(go)

    def test_edit_screens_send_the_whole_prompt_through_the_rich_chain(self):
        async def go() -> None:
            prompt = "p" * panel.MAX_PROMPT_CHARS
            await digest.set_setting(digest.SETTING_PROMPT, prompt)
            callback = _callback()
            callback.message.chat.id = ADMIN
            with patch.object(digest, "send_rich_to", AsyncMock(return_value=True)) as rich:
                await panel._digest_action(callback, ADMIN, "dgp", ["adm", "dgp"])
            chat_id, markdown = rich.await_args.args[1:]
            self.assertEqual(chat_id, ADMIN)
            self.assertTrue(markdown.endswith(prompt))  # not clipped
            self.assertEqual(panel._awaiting.pop(ADMIN), ("dgprompt", None))

        self._run(go)

    def test_prompt_can_be_uploaded_as_a_text_file(self):
        async def go() -> None:
            prompt = "П" * panel.MAX_PROMPT_CHARS
            message = _prompt_document(prompt.encode())
            self.assertFalse(panel._awaiting_prompt_file(message))  # not asked yet
            panel._awaiting[ADMIN] = ("dgprompt", None)
            self.assertTrue(panel._awaiting_prompt_file(message))
            await panel.on_prompt_file(message)
            self.assertEqual(await digest.get_prompt(), prompt)
            self.assertNotIn(ADMIN, panel._awaiting)
            message.answer.assert_awaited()

        self._run(go)

    def test_prompt_file_is_rejected_when_not_text_or_too_long(self):
        async def go() -> None:
            panel._awaiting[ADMIN] = ("dgprompt", None)
            await panel.on_prompt_file(_prompt_document(b"x", "prompt.pdf"))
            self.assertIn(ADMIN, panel._awaiting)  # still waiting for a prompt

            too_long = _prompt_document(b"x" * (panel.MAX_PROMPT_CHARS + 1))
            await panel.on_prompt_file(too_long)
            self.assertIn(ADMIN, panel._awaiting)
            self.assertEqual(await digest.get_prompt(), digest.DEFAULT_DIGEST_PROMPT)
            too_long.bot.download_file.assert_awaited()  # bytes fit, chars did not

        self._run(go)
