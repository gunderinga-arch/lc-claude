"""HTTP provider for the BuyTokens OpenAI-compatible gateway.

One `complete()` call is exactly one HTTP request: no retries here (see `LLMClient`).
Response parsing is deliberately tolerant because the gateway omits or fabricates fields.

Answers are streamed (SSE): the gateway keeps a stream alive with keepalives, reports failures inside it
(`data: {"error": ...}` after a 200), and a stream that breaks midway still leaves the text received so far,
which is returned as cut at the length limit, so the client continues it instead of losing minutes of work.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import httpx

from bot.llm.base import (
    ChatMessage,
    Completion,
    CompletionRequest,
    ModelAccessDenied,
    ProviderAuthError,
    ProviderBadResponse,
    ProviderError,
    ProviderTimeout,
    ProviderTooLong,
    ProviderUnavailable,
    ToolCall,
)

logger = logging.getLogger(__name__)

CONNECT_TIMEOUT_S = 15.0
LIST_MODELS_TIMEOUT_S = 30.0
BODY_SNIPPET_CHARS = 200
_REDACTED = "***"
# No byte for this long = a stalled stream. The gateway sends keepalives every ~23 s, and 15.10.2026 even a
# 950k-token prompt gave its first token after 78 s.
STREAM_IDLE_S = 240.0
# Until the answer's first text the request's own timeout applies (a model thinking that long goes to its
# fallback); once text flows the stream may run this long. Streams are not cut at the 600th second as plain
# requests are: 15.10.2026 one ran 1231 s.
STREAM_MAX_S = 1800.0
# In-stream error codes of a failure on the gateway's side ("Upstream unavailable", "Upstream stream
# interrupted"); any other code is a bad response. Both are retried after a pause.
_RETRYABLE_CODES = frozenset(
    {"upstream_unavailable", "upstream_error", "rate_limit_exceeded", "server_error", "overloaded", "timeout"}
)
# How gateways say a prompt is over the model's window (OpenAI, Anthropic and router wordings).
_TOO_LONG_HINTS = (
    "context_length",
    "context length",
    "context window",
    "maximum context",
    "prompt is too long",
    "too many tokens",
    "input is too long",
    "reduce the length",
)


class BuyTokensProvider:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        stream: bool = True,
    ) -> None:
        self._api_key = api_key
        self._stream = stream
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            transport=transport,
        )

    async def complete(self, request: CompletionRequest) -> Completion:
        payload: dict[str, Any] = {
            "model": request.model,
            "max_tokens": request.max_tokens,
            "messages": [_message(m) for m in request.messages],
        }
        if request.tools:
            payload["tools"] = list(request.tools)
        where = _describe("POST", "/chat/completions", request.model)
        if self._stream:
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}
            return await self._complete_streamed(request, payload, where)
        timeout = httpx.Timeout(request.timeout_s, connect=CONNECT_TIMEOUT_S)
        response = await self._send("POST", "/chat/completions", where, timeout, payload=payload)
        body = self._parse_json(response, where)
        return _parse_completion(body, request.model, where, self._snippet(response))

    async def _complete_streamed(
        self, request: CompletionRequest, payload: dict[str, Any], where: str
    ) -> Completion:
        """One streamed request. Until the first text `request.timeout_s` caps it, then STREAM_MAX_S does;
        STREAM_IDLE_S caps a silence in it.

        A stream that breaks after some text (timeout, dropped connection, an error chunk) returns that text
        as cut at the length limit; with nothing received yet it raises like a plain request would.
        """
        state = _Stream(request.model, api_key=self._api_key)
        timeout = httpx.Timeout(min(STREAM_IDLE_S, request.timeout_s), connect=CONNECT_TIMEOUT_S)
        started = time.monotonic()
        loop_started = asyncio.get_running_loop().time()
        try:
            async with asyncio.timeout(request.timeout_s) as deadline:
                async with self._client.stream(
                    "POST", "/chat/completions", json=payload, timeout=timeout
                ) as response:
                    if not 200 <= response.status_code < 300:
                        await response.aread()
                        self._raise_for_status(response, where)
                    if "text/event-stream" not in response.headers.get("content-type", ""):
                        # A gateway that ignores `stream` answers with one JSON body, as without it.
                        await response.aread()
                        body = self._parse_json(response, where)
                        return _parse_completion(body, request.model, where, self._snippet(response))
                    flowing = False
                    async for line in _stream_lines(response):
                        if state.feed(line, where):
                            state.done = True
                            break
                        if not flowing and state.visible:
                            flowing = True  # the answer is coming: let it run its course
                            deadline.reschedule(loop_started + max(request.timeout_s, STREAM_MAX_S))
        except (TimeoutError, httpx.TimeoutException) as exc:
            failure: ProviderError = ProviderTimeout(f"{where}: timed out ({type(exc).__name__})")
            return _salvage(state, where, started, failure)
        except httpx.TransportError as exc:
            failure = ProviderUnavailable(f"{where}: transport error ({type(exc).__name__})")
            return _salvage(state, where, started, failure)
        except httpx.RequestError as exc:  # decoding errors, redirect loops: not the network's fault
            raise ProviderBadResponse(f"{where}: request failed ({type(exc).__name__})") from exc
        except ProviderError as exc:
            if not state.started:  # an error status, or an error chunk before any answer
                raise
            return _salvage(state, where, started, exc)
        logger.debug("buytokens %s streamed %d chars in %d ms", where, state.chars, _ms(started))
        if state.finish is None and not state.done and state.calls:
            raise ProviderBadResponse(f"{where}: stream ended during tool calls")
        if state.finish is None and not state.done and state.started:
            # The stream closed with neither a finish reason nor [DONE]: the gateway dropped it mid-answer.
            logger.warning("%s: stream ended without a finish reason after %d chars", where, state.chars)
            return state.completion(cut=True)
        return state.completion()

    async def list_models(self) -> list[str]:
        timeout = httpx.Timeout(LIST_MODELS_TIMEOUT_S, connect=CONNECT_TIMEOUT_S)
        where = _describe("GET", "/models", None)
        response = await self._send("GET", "/models", where, timeout)
        body = self._parse_json(response, where)
        return _parse_model_ids(body, where, self._snippet(response))

    async def aclose(self) -> None:
        await self._client.aclose()

    # ---- internals ---------------------------------------------------------------------------

    async def _send(
        self,
        method: str,
        path: str,
        where: str,
        http_timeout: httpx.Timeout,
        *,
        payload: dict[str, Any] | None = None,
    ) -> httpx.Response:
        """Perform one request and map transport failures and non-2xx statuses to `ProviderError`s."""
        started = time.monotonic()
        try:
            response = await self._client.request(method, path, json=payload, timeout=http_timeout)
        except httpx.TimeoutException as exc:
            raise ProviderTimeout(f"{where}: timed out ({type(exc).__name__})") from exc
        except httpx.TransportError as exc:
            raise ProviderUnavailable(f"{where}: transport error ({type(exc).__name__})") from exc
        except httpx.RequestError as exc:  # decoding errors, redirect loops: not the network's fault
            raise ProviderBadResponse(f"{where}: request failed ({type(exc).__name__})") from exc
        latency_ms = int((time.monotonic() - started) * 1000)
        logger.debug(
            "buytokens %s status=%d latency_ms=%d",
            where,
            response.status_code,
            latency_ms,
        )
        self._raise_for_status(response, where)
        return response

    def _raise_for_status(self, response: httpx.Response, where: str) -> None:
        status = response.status_code
        if 200 <= status < 300:
            return
        snippet = self._snippet(response)
        message = f"{where}: HTTP {status}: {snippet}"
        if status == 429 or status >= 500:
            raise ProviderUnavailable(message, status=status)
        if status == 401:
            raise ProviderAuthError(message)
        if status == 403:
            raise ModelAccessDenied(message)
        if status == 413 or (status == 400 and _says_too_long(snippet)):
            raise ProviderTooLong(message)
        raise ProviderBadResponse(message)

    def _parse_json(self, response: httpx.Response, where: str) -> Any:
        try:
            return json.loads(response.content)
        except ValueError as exc:  # JSONDecodeError and UnicodeDecodeError
            raise ProviderBadResponse(
                f"{where}: HTTP {response.status_code}: body is not JSON: {self._snippet(response)}"
            ) from exc

    def _snippet(self, response: httpx.Response) -> str:
        """First chars of the body for error messages, with the API key redacted if echoed back."""
        text = response.content.decode("utf-8", errors="replace")
        if self._api_key:
            text = text.replace(self._api_key, _REDACTED)
        return text[:BODY_SNIPPET_CHARS]


def _ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


async def _stream_lines(response: httpx.Response) -> AsyncIterator[str]:
    """SSE uses CR/LF, not the Unicode separators that httpx.aiter_lines also splits on."""
    pending = ""
    async for piece in response.aiter_text():
        pending += piece
        # Keep a final CR until the next piece in case it is half of CRLF.
        end = len(pending) - int(pending.endswith("\r"))
        start = 0
        for match in re.finditer(r"\r\n|\r|\n", pending[:end]):
            yield pending[start : match.start()]
            start = match.end()
        pending = pending[start:]
    if pending:
        yield pending.rstrip("\r")


def _unicode_text(text: str) -> str:
    """Recombine escaped UTF-16 pairs split between deltas; replace an unpaired surrogate."""
    if any("\ud800" <= char <= "\udfff" for char in text):
        return text.encode("utf-16-le", "surrogatepass").decode("utf-16-le", "replace")
    return text


def _says_too_long(text: str) -> bool:
    """The gateway refused the prompt as over the model's window.

    BuyTokens itself says only "Invalid request: one or more parameters have invalid values." to a prompt
    over the window (15.10.2026, ~980k tokens to Fable 5.1), as to any other bad parameter: the client tells
    the two apart by the prompt's size.
    """
    lowered = text.lower()
    return any(hint in lowered for hint in _TOO_LONG_HINTS)


def _salvage(state: _Stream, where: str, started: float, failure: ProviderError) -> Completion:
    """The text a broken stream delivered, as cut at the length limit; nothing usable raises `failure`."""
    if state.finish is not None:
        return state.completion()
    if not state.salvageable:
        raise failure
    logger.warning(
        "%s: stream broke after %d chars and %d ms (%s): keeping the part",
        where,
        state.chars,
        _ms(started),
        failure,
    )
    return state.completion(cut=True)


@dataclass(slots=True)
class _Stream:
    """What a streamed answer delivered so far: text, tool calls, finish reason, usage."""

    requested: str
    api_key: str = field(default="", repr=False)
    parts: list[str] = field(default_factory=list)
    chars: int = 0
    visible: bool = False  # some non-blank text arrived (blank chunks are the gateway's keepalives)
    done: bool = False  # the stream closed properly with [DONE]
    calls: dict[int, dict[str, str]] = field(default_factory=dict)
    finish: str | None = None
    model: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)

    @property
    def started(self) -> bool:
        return self.chars > 0 or bool(self.calls)

    @property
    def salvageable(self) -> bool:
        """Some answer text and no half-received tool calls (their arguments would be broken JSON)."""
        return not self.calls and bool("".join(self.parts).strip())

    def feed(self, line: str, where: str) -> bool:
        """Take one SSE line; True at the end of the stream. An error chunk raises its `ProviderError`."""
        if not line.startswith("data:"):
            return False  # blank separators, ": keepalive" comments, event:/id: fields
        data = line[5:].strip()
        if data == "[DONE]":
            return True
        try:
            chunk = json.loads(data)
        except ValueError:
            logger.warning("%s: malformed SSE data line skipped", where)
            return False  # one broken line in a long stream is skipped, not fatal
        if not isinstance(chunk, dict):
            return False
        if chunk.get("error") is not None:
            raise _stream_error(chunk["error"], where, self.api_key)
        if isinstance(chunk.get("model"), str) and chunk["model"]:
            self.model = chunk["model"]
        if isinstance(chunk.get("usage"), dict):
            self.usage = chunk["usage"]
        for choice in chunk.get("choices") or ():
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta") or choice.get("message") or {}
            if not isinstance(delta, dict):
                continue
            content = delta.get("content")
            if isinstance(content, str) and content:
                self.parts.append(content)
                self.chars += len(content)
                self.visible = self.visible or not content.isspace()
            for item in delta.get("tool_calls") or ():
                self._add_call(item)
            if isinstance(choice.get("finish_reason"), str):
                self.finish = choice["finish_reason"]
        return False

    def _add_call(self, item: Any) -> None:
        """Tool calls arrive in pieces keyed by index: the id and name once, the arguments as fragments."""
        if not isinstance(item, dict):
            return
        index = item["index"] if isinstance(item.get("index"), int) else len(self.calls)
        slot = self.calls.setdefault(index, {"id": "", "name": "", "arguments": ""})
        if isinstance(item.get("id"), str) and item["id"]:
            slot["id"] = item["id"]
        function = item.get("function")
        if not isinstance(function, dict):
            return
        if isinstance(function.get("name"), str) and function["name"] and not slot["name"]:
            slot["name"] = function["name"]
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            slot["arguments"] += arguments
        elif isinstance(arguments, dict):  # a gateway that sends the arguments already parsed
            slot["arguments"] = json.dumps(arguments)

    def completion(self, *, cut: bool = False) -> Completion:
        """The answer so far; `cut` marks it as stopped at the length limit (the client continues it)."""
        calls = tuple(
            ToolCall(
                id=slot["id"] or f"call_{index}",
                name=slot["name"],
                arguments=_unicode_text(slot["arguments"] or "{}"),
            )
            for index, slot in sorted(self.calls.items())
            if slot["name"]
        )
        return Completion(
            text=_unicode_text("".join(self.parts)),
            finish_reason="length" if cut else self.finish,
            model=self.model or self.requested,
            prompt_tokens=_opt_int(self.usage.get("prompt_tokens")),
            completion_tokens=_opt_int(self.usage.get("completion_tokens")),
            tool_calls=() if cut else calls,
        )


def _stream_error(error: Any, where: str, api_key: str = "") -> ProviderError:
    """An error chunk inside a 200 stream, mapped like the HTTP status it stands for."""
    if isinstance(error, dict):
        code = str(error.get("code") or error.get("type") or "")
        text = str(error.get("message") or "")
    else:
        code, text = "", str(error)
    if api_key:
        code, text = code.replace(api_key, _REDACTED), text.replace(api_key, _REDACTED)
    message = f"{where}: error in the stream: {code}: {text[:BODY_SNIPPET_CHARS]}"
    if _says_too_long(f"{code} {text}"):
        return ProviderTooLong(message)
    if not code or code in _RETRYABLE_CODES:
        return ProviderUnavailable(message, status=502)
    # "jailbreak_detected" (the gateway's filter, 15.10.2026 on a plain list of numbers) and the like.
    return ProviderBadResponse(message)


def _message(message: ChatMessage) -> dict[str, Any]:
    """OpenAI wire format, including tool calls and tool results (verified on the gateway 24.09.2026)."""
    wire: dict[str, Any] = {"role": message.role, "content": _content(message)}
    if message.tool_calls:
        wire["content"] = message.content or None
        wire["tool_calls"] = [
            {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
            for c in message.tool_calls
        ]
    if message.tool_call_id is not None:
        wire["tool_call_id"] = message.tool_call_id
    return wire


def _parse_tool_calls(raw: Any) -> tuple[ToolCall, ...]:
    """Tolerant: malformed entries are dropped, a missing id gets a stable placeholder."""
    if not isinstance(raw, list):
        return ()
    calls: list[ToolCall] = []
    for index, item in enumerate(raw):
        function = item.get("function") if isinstance(item, dict) else None
        if not isinstance(function, dict) or not isinstance(function.get("name"), str):
            continue
        arguments = function.get("arguments")
        call_id = item.get("id")
        calls.append(
            ToolCall(
                id=call_id if isinstance(call_id, str) and call_id else f"call_{index}",
                name=function["name"],
                arguments=arguments if isinstance(arguments, str) else json.dumps(arguments or {}),
            )
        )
    return tuple(calls)


def _content(message: ChatMessage) -> str | list[dict[str, Any]]:
    """Plain string, or OpenAI content parts when the message carries images (verified on all models)."""
    if not message.images:
        return message.content
    parts: list[dict[str, Any]] = [{"type": "text", "text": message.content}]
    parts += [{"type": "image_url", "image_url": {"url": uri}} for uri in message.images]
    return parts


def _describe(method: str, path: str, model: str | None) -> str:
    return f"{method} {path}" + (f" model={model}" if model is not None else "")


def _parse_completion(body: Any, requested_model: str, where: str, snippet: str) -> Completion:
    def bad(reason: str) -> ProviderError:
        return ProviderBadResponse(f"{where}: {reason}: {snippet}")

    if not isinstance(body, dict):
        raise bad("body is not a JSON object")
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise bad("no choices")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise bad("choices[0] is not an object")
    message = choice.get("message")
    if not isinstance(message, dict):
        raise bad("choices[0].message is missing")

    content = message.get("content")
    finish_reason = choice.get("finish_reason")
    model = body.get("model")
    usage = body.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    return Completion(
        text=content if isinstance(content, str) else "",
        finish_reason=finish_reason if isinstance(finish_reason, str) else None,
        model=model if isinstance(model, str) and model else requested_model,
        prompt_tokens=_opt_int(usage.get("prompt_tokens")),
        completion_tokens=_opt_int(usage.get("completion_tokens")),
        tool_calls=_parse_tool_calls(message.get("tool_calls")),
    )


def _parse_model_ids(body: Any, where: str, snippet: str) -> list[str]:
    items = body.get("data") if isinstance(body, dict) else None
    if not isinstance(items, list):
        raise ProviderBadResponse(f"{where}: no data list: {snippet}")
    ids: list[str] = []
    for item in items:
        model_id = item.get("id") if isinstance(item, dict) else None
        if not isinstance(model_id, str):
            raise ProviderBadResponse(f"{where}: item without string id: {snippet}")
        ids.append(model_id)
    return ids


def _opt_int(value: Any) -> int | None:
    # bool is an int subclass; a fabricated `true` is not a token count
    return value if isinstance(value, int) and not isinstance(value, bool) else None
