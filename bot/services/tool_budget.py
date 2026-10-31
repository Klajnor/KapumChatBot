"""Per-request budgets for model tool calls (web search and MCP).

Round limits alone do not bound a single assistant turn: the model may ask for
many ``search_web`` / MCP calls at once, and one search can read several pages.
The budget caps the total number of calls and the total amount of tool text
added to the model context for one request, regardless of how the calls are
split across rounds.

Tool output is third-party content wrapped in ``<untrusted_content>``. Trimming
must not cut the closing tag, otherwise the model would see an unbalanced frame,
so :func:`clip_tool_text` re-closes it when the cut lands inside one.
"""
from __future__ import annotations

from bot.config import config

CALL_LIMIT_NOTICE = (
    "Лимит инструментов на этот вопрос исчерпан. Ответь из уже собранной "
    "информации."
)
CONTEXT_LIMIT_NOTICE = (
    "Результат инструмента не добавлен: исчерпан бюджет контекста на этот вопрос."
)

_CLOSE_TAG = "</untrusted_content>"
_CUT_NOTE = "\n…[результат обрезан]"


def clip_tool_text(text: str, limit: int) -> str:
    """Trim tool output to ``limit`` characters, keeping the untrusted frame.

    A tool result is normally one (or more) ``<untrusted_content>`` blocks. If
    the cut falls inside a block, the closing tag is appended; the truncated
    body stays data and the following model turn still sees a valid marker.
    """
    if limit <= 0:
        return CONTEXT_LIMIT_NOTICE
    if len(text) <= limit:
        return text
    if limit <= len(_CUT_NOTE) + len(_CLOSE_TAG):
        return CONTEXT_LIMIT_NOTICE
    cut = limit - len(_CUT_NOTE)
    head = text[:cut]
    if head.count("<untrusted_content") > head.count(_CLOSE_TAG):
        cut = max(0, limit - len(_CUT_NOTE) - len(_CLOSE_TAG) - 1)
        head = text[:cut]
        return f"{head}{_CUT_NOTE}\n{_CLOSE_TAG}"
    return f"{head}{_CUT_NOTE}"


class ToolBudget:
    """Mutable per-request caps shared by every tool round."""

    __slots__ = ("context_chars", "mcp_calls", "search_calls")

    def __init__(
        self,
        *,
        search_calls: int | None = None,
        mcp_calls: int | None = None,
        context_chars: int | None = None,
    ) -> None:
        self.search_calls = (
            config.search_max_calls if search_calls is None else search_calls
        )
        self.mcp_calls = config.mcp_max_calls if mcp_calls is None else mcp_calls
        self.context_chars = (
            config.tool_max_context_chars
            if context_chars is None
            else context_chars
        )

    @property
    def search_left(self) -> bool:
        """True while another search call fits both call and context budgets."""
        return self.search_calls > 0 and self.context_chars > 0

    @property
    def mcp_left(self) -> bool:
        """True while another MCP call fits both call and context budgets."""
        return self.mcp_calls > 0 and self.context_chars > 0

    def take_search(self) -> bool:
        """Reserve one search call; False when the budget is exhausted."""
        if not self.search_left:
            return False
        self.search_calls -= 1
        return True

    def take_mcp(self) -> bool:
        """Reserve one MCP call; False when the budget is exhausted."""
        if not self.mcp_left:
            return False
        self.mcp_calls -= 1
        return True

    def clip(self, text: str) -> str:
        """Fit ``text`` into the remaining context budget and charge it."""
        if self.context_chars <= 0:
            return CONTEXT_LIMIT_NOTICE
        clipped = clip_tool_text(text, self.context_chars)
        self.context_chars = max(0, self.context_chars - len(clipped))
        return clipped