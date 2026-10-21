"""Per-user cap on simultaneously running AI requests.

Updates are handled concurrently, so without a cap one user can keep dozens of
slow (and paid) requests in flight. Slots live in memory: a restart resets them,
which is fine because the requests themselves die with the process.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from collections.abc import Iterator

from bot.config import config

_inflight: defaultdict[int, int] = defaultdict(int)


@contextmanager
def slot(user_id: int) -> Iterator[bool]:
    """Yield True when the user may start another request, False when over the cap.

    The slot is released on exit only if it was granted.
    """
    limit = max(1, config.max_concurrent_per_user)
    if _inflight[user_id] >= limit:
        yield False
        return
    _inflight[user_id] += 1
    try:
        yield True
    finally:
        _inflight[user_id] -= 1
        if _inflight[user_id] <= 0:
            _inflight.pop(user_id, None)


BUSY_MESSAGE = (
    "⏳ <b>Please wait.</b> Your previous request is still being processed."
)
