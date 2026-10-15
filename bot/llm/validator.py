"""Classify raw gateway text: a real answer, or one of the gateway's known failure shapes."""

from __future__ import annotations

from bot.llm.base import Verdict

# Canned replies the gateway substitutes for filtered topics. Retrying the same context never helps.
STUB_SIGNATURES: tuple[str, ...] = ("Oh, THAT island",)
# Whole-answer refusals (compared to the entire answer, so a real answer quoting them stays OK).
# Seen 24.09.2026 on Opus 4.8, Sol and Luna once a political answer is in the history.
STUB_ANSWERS: frozenset[str] = frozenset({"я не хочу обсуждать политические темы"})

# First-token-only answers seen from gpt-6-astra: the gateway cut the stream right after them.
CUT_FRAGMENTS: frozenset[str] = frozenset({"Let", "##", "#", "Looking", "**", "```"})

# A short answer ending in a colon is a lead-in whose content never arrived: gpt-6-astra with tools declared
# says "Собираю лендинг:" and the gateway drops the page that follows (seen 27.09.2026, 5 of 6 designer runs).
PROMISE_MAX_CHARS = 600

# Text made only of these characters is the start of markdown with no content yet.
_MARKDOWN_PUNCTUATION: frozenset[str] = frozenset("#*-`>_~")


def _is_markdown_punctuation_only(text: str) -> bool:
    return all(char in _MARKDOWN_PUNCTUATION or char.isspace() for char in text)


def classify(text: str) -> Verdict:
    """Return the verdict for a completion's text. Short legit answers ("42", "No.") are OK."""
    stripped = text.strip()
    if not stripped:
        return Verdict.EMPTY
    if any(signature in stripped for signature in STUB_SIGNATURES):
        return Verdict.STUB
    if stripped.rstrip(".!").casefold() in STUB_ANSWERS:
        return Verdict.STUB
    if stripped in CUT_FRAGMENTS or _is_markdown_punctuation_only(stripped):
        return Verdict.CUT
    if stripped.endswith(":") and len(stripped) <= PROMISE_MAX_CHARS:
        return Verdict.CUT
    return Verdict.OK
