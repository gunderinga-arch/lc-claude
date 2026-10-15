"""Provider-agnostic LLM contracts.

Two levels:
- `LLMProvider` is one raw HTTP call to one model. It raises typed `ProviderError`s and never retries.
- `LLMClient` is what services use. It owns retries, fallbacks, response validation and the
  concurrency limit, and it never raises for provider problems: it returns a `ChatResult`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal, Protocol

ChatRole = Literal["system", "user", "assistant", "tool"]

# OpenAI function-tool definitions ({"type": "function", "function": {...}}), sent as is.
ToolSpec = Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON string, as the model wrote it


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: ChatRole
    content: str
    images: tuple[str, ...] = ()  # data: URIs (vision); only on the current user message, never stored
    tool_calls: tuple[ToolCall, ...] = ()  # assistant turn that asked for tools
    tool_call_id: str | None = None  # role "tool": which call this is the result of
    droppable: bool = (
        False  # earlier dialog: left out, oldest first, when the request outgrows a model's window
    )
    truncatable: bool = True  # a digest part must be read whole, even on a smaller fallback


@dataclass(frozen=True, slots=True)
class CompletionRequest:
    model: str
    messages: Sequence[ChatMessage]
    max_tokens: int
    timeout_s: float
    tools: Sequence[ToolSpec] = ()


@dataclass(frozen=True, slots=True)
class Completion:
    text: str  # "" when the gateway returned null/empty content
    finish_reason: str | None
    model: str
    prompt_tokens: int | None  # informational only: the gateway fabricates parts of `usage`
    completion_tokens: int | None
    tool_calls: tuple[ToolCall, ...] = ()
    reserve: bool = False  # answered by the reserve gateway; `model` is then the reserve's model


# ---- provider errors -------------------------------------------------------------------------


class ProviderError(Exception):
    """Base class. `str(exc)` must never contain the API key or message contents."""


class ProviderTimeout(ProviderError):
    """No complete response within the request timeout."""


class ProviderUnavailable(ProviderError):
    """Network failure, 5xx or 429: worth retrying after a backoff."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class ModelAccessDenied(ProviderError):
    """403: this key has no access to the model. Retrying is pointless."""


class ProviderAuthError(ProviderError):
    """401: bad API key. Nothing works until the config is fixed."""


class ProviderBadResponse(ProviderError):
    """Other 4xx, or a 2xx body that is not the expected JSON shape."""


class ProviderTooLong(ProviderBadResponse):
    """The prompt is over the model's context window: the same request must be cut, not repeated."""


class LLMProvider(Protocol):
    async def complete(self, request: CompletionRequest) -> Completion: ...

    async def list_models(self) -> list[str]: ...

    async def aclose(self) -> None: ...


# ---- client level ----------------------------------------------------------------------------


class Verdict(StrEnum):
    OK = "ok"
    EMPTY = "empty"  # no text at all: usually max_tokens spent on hidden reasoning
    CUT = "cut"  # gateway cut the answer at the very start ("Let", "##")
    STUB = "stub"  # gateway's canned reply for filtered political topics


class ChatStatus(StrEnum):
    OK = "ok"
    STUB = "stub"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ChatResult:
    status: ChatStatus
    text: str  # the answer for OK; "" otherwise
    model_used: str  # model that produced the answer, or the last model tried
    attempts: int  # total HTTP attempts across all models
    latency_ms: int  # wall time including queueing and retries
    fallback_used: bool
    error: str | None = None  # short machine-readable reason for FAILED, e.g. "timeout", "auth"
    tool_calls: tuple[ToolCall, ...] = ()  # OK with tool calls: the model wants tool results first
    served_by: str | None = None  # reserve gateway's model when the emergency switch was on
    input_tokens: int | None = None  # estimates across retries, continuations and glue calls
    output_tokens: int | None = None


class LLMClient(Protocol):
    async def chat(
        self, messages: Sequence[ChatMessage], model_id: str, tools: Sequence[ToolSpec] = ()
    ) -> ChatResult:
        """Run a chat completion with retries and fallbacks.

        Never raises for provider failures; `asyncio.CancelledError` propagates.
        """
        ...

    def queue_depth(self) -> int:
        """How many requests are waiting for a free gateway slot right now."""
        ...
