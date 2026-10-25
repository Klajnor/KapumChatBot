"""Feedback for slow AI requests.

If a request to the model takes longer than ``config.progress_delay`` seconds,
a "waiting" message is posted in the chat and refreshed every
``config.progress_interval`` seconds (spinner + elapsed time) until the answer
arrives. The notice is deleted right before the reply is sent, so the user sees
a live counter instead of a silent hang — and sees nothing at all when the
model answers quickly.
"""

from __future__ import annotations

import asyncio
import logging
import time
from types import TracebackType
from typing import Any, Self

from aiogram.enums import ChatAction, ChatType
from aiogram.exceptions import TelegramAPIError, TelegramBadRequest
from aiogram.types import Message, ReplyParameters

from bot.config import config
from bot.services.telegram_utils import safe_delete, throttle

logger = logging.getLogger(__name__)

# Frames cycled while the notice is on screen.
SPINNER_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")

# The "typing…" chat action expires after ~5 seconds: refresh it periodically.
_TYPING_EVERY_TICKS = 4

# Stop trying after this many consecutive Telegram failures.
_MAX_FAILURES = 3


def render_notice(template: str, seconds: float, frame: str) -> str:
    """Fill the configured template, tolerating unknown placeholders."""
    try:
        return template.format(seconds=int(seconds), spinner=frame)
    except (IndexError, KeyError, ValueError):
        return template


class ProgressNotifier:
    """Async context manager showing a live "waiting for the answer" notice.

    Usage::

        async with waiting(message):
            result = await ai_client.chat_with_tools(...)

    Everything Telegram-related is best effort: failures are logged and never
    propagate into the handler, and the wrapped call's exceptions are not
    suppressed.
    """

    def __init__(
        self,
        message: Message,
        *,
        delay: float | None = None,
        interval: float | None = None,
        text: str | None = None,
        reply: bool | None = None,
        keep_typing: bool = True,
    ) -> None:
        self.message = message
        self.delay = config.progress_delay if delay is None else max(0.0, delay)
        self.interval = (
            config.progress_interval if interval is None else max(0.05, interval)
        )
        self.template = text if text is not None else config.progress_text
        # Quote the user's message in groups so the notice is attributable.
        self.reply = (
            message.chat.type != ChatType.PRIVATE if reply is None else reply
        )
        self.keep_typing = keep_typing

        self._task: asyncio.Task[None] | None = None
        self._notice: Message | None = None
        self._started = time.monotonic()
        self._stopped = False
        # Exposed for tests / logging.
        self.sent = False
        self.edits = 0

    # --- Context manager ---------------------------------------------------

    async def __aenter__(self) -> Self:
        if config.enable_progress:
            self._started = time.monotonic()
            self._stopped = False
            self._task = asyncio.create_task(self._run(), name="progress-notice")
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.stop()
        # Never suppress the wrapped exception.

    async def stop(self) -> None:
        """Cancel the loop and delete the notice. Idempotent.

        Streaming handlers call this as soon as the first answer chunk
        arrives: the live answer replaces the "waiting" notice.
        """
        self._stopped = True
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception as error:  # noqa: BLE001 - UI helper must not raise
                logger.debug("Progress notice task failed: %s", error)
        await self._delete_notice()

    # --- Loop --------------------------------------------------------------

    async def _run(self) -> None:
        """Wait out the delay, then keep the notice fresh once per interval."""
        if self.delay:
            await asyncio.sleep(self.delay)

        failures = 0
        ticks = 0
        while not self._stopped:
            elapsed = time.monotonic() - self._started
            frame = SPINNER_FRAMES[ticks % len(SPINNER_FRAMES)]
            body = render_notice(self.template, elapsed, frame)

            if self._notice is None:
                notice = await self._send(body)
                if notice is None:
                    failures += 1
                    if failures >= _MAX_FAILURES:
                        logger.warning("Progress notice could not be sent")
                        return
                else:
                    self._notice = notice
                    self.sent = True
                    failures = 0
            elif await self._edit(body):
                self.edits += 1
                failures = 0
            else:
                failures += 1
                if failures >= _MAX_FAILURES:
                    logger.warning("Progress notice could not be updated")
                    return

            ticks += 1
            if self.keep_typing and ticks % _TYPING_EVERY_TICKS == 0:
                await self._typing()
            await asyncio.sleep(self.interval)

    # --- Telegram calls ----------------------------------------------------

    async def _send(self, text: str) -> Message | None:
        kwargs: dict[str, Any] = {"parse_mode": None}
        if self.reply:
            kwargs["reply_parameters"] = ReplyParameters(
                message_id=self.message.message_id
            )
        try:
            # Share the per-chat rate budget with the answer that follows.
            await throttle.wait(self.message.chat.id)
            return await self.message.answer(text, **kwargs)
        except TelegramAPIError as exc:
            logger.warning("Sending the progress notice failed: %s", exc)
            return None
        except Exception as exc:  # noqa: BLE001 - UI helper must not raise
            logger.warning("Sending the progress notice failed: %s", exc)
            return None

    async def _edit(self, text: str) -> bool:
        notice = self._notice
        if notice is None:
            return False
        try:
            await throttle.wait(self.message.chat.id)
            await notice.edit_text(text, parse_mode=None)
            return True
        except TelegramBadRequest as exc:
            # "message is not modified" happens when the text repeats itself.
            if "not modified" in str(exc).lower():
                return True
            logger.debug("Editing the progress notice failed: %s", exc)
            return False
        except TelegramAPIError as exc:
            logger.debug("Editing the progress notice failed: %s", exc)
            return False
        except Exception as exc:  # noqa: BLE001 - UI helper must not raise
            logger.debug("Editing the progress notice failed: %s", exc)
            return False

    async def _typing(self) -> None:
        try:
            await self.message.bot.send_chat_action(
                chat_id=self.message.chat.id, action=ChatAction.TYPING
            )
        except Exception as exc:  # noqa: BLE001 - cosmetic, ignore failures
            logger.debug("Chat action failed: %s", exc)

    async def _delete_notice(self) -> None:
        notice, self._notice = self._notice, None
        if notice is None:
            return
        await safe_delete(notice)


def waiting(message: Message, **kwargs: Any) -> ProgressNotifier:
    """Return a :class:`ProgressNotifier` for ``message``.

    ``kwargs`` overrides the configured delay / interval / text, which is handy
    for calls with a different nature (image generation, speech recognition).
    """
    return ProgressNotifier(message, **kwargs)
