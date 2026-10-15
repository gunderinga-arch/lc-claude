"""Fitting a request into a model's context window.

The bot sends each model as much as its window holds (~1M tokens for Opus 5, GPT-6 Sol and Fable 5.1,
measured 15.10.2026). When a request outgrows the window — a long dialog meets a fallback with a smaller
window, tool results pile up, a continuation carries a long answer — the earliest dialog is left out first
(messages marked `droppable`), and only then the biggest text is cut in the middle, with a visible mark.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from bot.domain import IMAGE_TOKENS, count_tokens
from bot.llm.base import ChatMessage, ProviderTooLong

# Share of a window left unplanned: tool schemas, the gateway's own additions, the estimate's error.
WINDOW_SLACK = 0.04
MESSAGE_OVERHEAD = 4  # role and separators of one message
CUT_MARK = "\n\n[… {chars} символов отсюда пропущено: не поместились в окно модели …]\n\n"


def message_tokens(message: ChatMessage, tokenizer: str = "claude") -> int:
    calls = sum(count_tokens(c.name + c.arguments, tokenizer) + 8 for c in message.tool_calls)
    return (
        count_tokens(message.content, tokenizer)
        + IMAGE_TOKENS * len(message.images)
        + calls
        + MESSAGE_OVERHEAD
    )


def prompt_tokens(messages: Sequence[ChatMessage], tokenizer: str = "claude") -> int:
    return sum(message_tokens(m, tokenizer) for m in messages)


def fit_window(messages: Sequence[ChatMessage], limit: int, tokenizer: str = "claude") -> list[ChatMessage]:
    """`messages` within `limit` tokens; unchanged when they already fit.

    1. Droppable messages (the earlier dialog) go oldest first; an answer left without its question goes too.
    2. Still too big: the biggest non-system text is cut in the middle (its start and end stay), repeatedly.
    """
    sizes = [message_tokens(m, tokenizer) for m in messages]
    total = sum(sizes)
    if total <= limit:
        return list(messages)
    kept = list(zip(messages, sizes, strict=True))
    i = 0
    while total > limit and i < len(kept):
        if kept[i][0].droppable:
            total -= kept.pop(i)[1]
        else:
            i += 1
    # The dialog must not open with an answer to a question that was dropped.
    first = 1 if kept and kept[0][0].role == "system" else 0
    while first < len(kept) and kept[first][0].droppable and kept[first][0].role != "user":
        total -= kept.pop(first)[1]
    for _ in range(8):  # each cut lands within a few percent; a handful of passes always converges
        if total <= limit:
            break
        candidates = [
            k for k, (m, _) in enumerate(kept) if m.role != "system" and m.content and m.truncatable
        ]
        if not candidates:
            break
        k = max(candidates, key=lambda n: kept[n][1])
        message, size = kept[k]
        text_tokens = max(1, count_tokens(message.content, tokenizer))
        excess = total - limit
        keep_chars = int(len(message.content) * max(0.0, 1 - (excess + 50) / text_tokens) * 0.97)
        cut = cut_middle(message.content, keep_chars)
        new_size = message_tokens(replace(message, content=cut), tokenizer)
        kept[k] = (replace(message, content=cut), new_size)
        total += new_size - size
    if total > limit:
        raise ProviderTooLong("Required message content does not fit the model window")
    return [m for m, _ in kept]


def cut_middle(text: str, keep_chars: int) -> str:
    """`text` cut to about `keep_chars`: two thirds from the start, one third from the end, a mark between."""
    if keep_chars >= len(text):
        return text
    keep_chars = max(0, keep_chars)
    head = keep_chars * 2 // 3
    tail = keep_chars - head
    end = text[len(text) - tail :] if tail else ""
    return text[:head] + CUT_MARK.format(chars=len(text) - keep_chars) + end
