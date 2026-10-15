from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx
import pytest

from bot.llm.base import (
    ChatMessage,
    CompletionRequest,
    ModelAccessDenied,
    ProviderAuthError,
    ProviderBadResponse,
    ProviderError,
    ProviderTimeout,
    ProviderUnavailable,
)
from bot.llm.buytokens import BuyTokensProvider

BASE_URL = "https://gateway.test/v1"
API_KEY = "sk-secret-test-key-123"
SECRET_TEXT = "my very private question"

Handler = Callable[[httpx.Request], httpx.Response]


def make_request(model: str = "gpt-6-astra") -> CompletionRequest:
    return CompletionRequest(
        model=model,
        messages=[ChatMessage("system", "be brief"), ChatMessage("user", SECRET_TEXT)],
        max_tokens=512,
        timeout_s=42.0,
    )


@pytest.fixture
def captured() -> list[httpx.Request]:
    return []


@pytest.fixture
async def make_provider(
    captured: list[httpx.Request],
) -> AsyncIterator[Callable[[Handler], BuyTokensProvider]]:
    providers: list[BuyTokensProvider] = []

    def factory(handler: Handler) -> BuyTokensProvider:
        def recording(req: httpx.Request) -> httpx.Response:
            captured.append(req)
            return handler(req)

        provider = BuyTokensProvider(BASE_URL, API_KEY, transport=httpx.MockTransport(recording))
        providers.append(provider)
        return provider

    yield factory
    for provider in providers:
        await provider.aclose()


def json_response(body: Any, status: int = 200) -> Handler:
    return lambda _req: httpx.Response(status, json=body)


def chat_body(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "choices": [{"finish_reason": "stop", "index": 0, "message": {"content": "Hi", "role": "assistant"}}],
        "model": "gpt-6-astra",
        "usage": {"completion_tokens": 13, "prompt_tokens": 9, "total_tokens": 22},
    }
    body.update(overrides)
    return body


# ---- success parsing ---------------------------------------------------------------------------


async def test_complete_parses_real_gateway_shape(make_provider) -> None:
    provider = make_provider(json_response(chat_body()))
    result = await provider.complete(make_request())
    assert result.text == "Hi"
    assert result.finish_reason == "stop"
    assert result.model == "gpt-6-astra"
    assert result.prompt_tokens == 9
    assert result.completion_tokens == 13


@pytest.mark.parametrize("content", [None, ""])
async def test_null_or_empty_content_gives_empty_text(make_provider, content: str | None) -> None:
    body = chat_body(
        choices=[{"finish_reason": "length", "message": {"role": "assistant", "content": content}}]
    )
    result = await make_provider(json_response(body)).complete(make_request())
    assert result.text == ""
    assert result.finish_reason == "length"


async def test_missing_optional_fields_are_tolerated(make_provider) -> None:
    body = {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
    result = await make_provider(json_response(body)).complete(make_request("requested-model"))
    assert result.text == "ok"
    assert result.finish_reason is None
    assert result.model == "requested-model"
    assert result.prompt_tokens is None
    assert result.completion_tokens is None


async def test_partial_usage_gives_none_for_missing_keys(make_provider) -> None:
    body = chat_body(usage={"completion_tokens": 5})
    result = await make_provider(json_response(body)).complete(make_request())
    assert result.completion_tokens == 5
    assert result.prompt_tokens is None


# ---- request shape -----------------------------------------------------------------------------


async def test_request_body_and_headers(make_provider, captured: list[httpx.Request]) -> None:
    await make_provider(json_response(chat_body())).complete(make_request())
    (sent,) = captured
    assert sent.method == "POST"
    assert str(sent.url) == "https://gateway.test/v1/chat/completions"
    assert sent.headers["Authorization"] == f"Bearer {API_KEY}"
    assert json.loads(sent.content) == {
        "model": "gpt-6-astra",
        "max_tokens": 512,
        "messages": [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": SECRET_TEXT},
        ],
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    timeout = sent.extensions["timeout"]
    assert timeout["read"] == 42.0
    assert timeout["connect"] == 15.0


# ---- streaming ---------------------------------------------------------------------------------


def sse(*chunks: Any, done: bool = True) -> Handler:
    """A 200 event stream of `chunks` (dicts as JSON, strings as raw lines)."""
    lines = [c if isinstance(c, str) else "data: " + json.dumps(c, ensure_ascii=False) for c in chunks]
    body = "\n\n".join([*lines, *(["data: [DONE]"] if done else [])]) + "\n\n"
    return lambda _req: httpx.Response(
        200, content=body.encode(), headers={"content-type": "text/event-stream"}
    )


def delta(content: str | None = None, finish: str | None = None, **extra: Any) -> dict[str, Any]:
    part: dict[str, Any] = {} if content is None else {"content": content}
    part.update(extra)
    return {"model": "claude-opus-5", "choices": [{"index": 0, "delta": part, "finish_reason": finish}]}


async def test_stream_is_assembled_with_keepalives_and_usage(make_provider) -> None:
    handler = sse(
        ": keepalive",
        delta(role="assistant"),
        delta("Мет"),
        delta("р — "),
        "data: {broken json",
        delta("единица.", "stop"),
        {"usage": {"prompt_tokens": 12, "completion_tokens": 5}, "choices": []},
    )
    result = await make_provider(handler).complete(make_request())
    assert result.text == "Метр — единица."
    assert result.finish_reason == "stop"
    assert result.model == "claude-opus-5"
    assert (result.prompt_tokens, result.completion_tokens) == (12, 5)


async def test_streamed_tool_calls_are_assembled_from_fragments(make_provider) -> None:
    first = {
        "index": 0,
        "id": "call_a",
        "type": "function",
        "function": {"name": "web_search", "arguments": ""},
    }
    handler = sse(
        delta(tool_calls=[first]),
        delta(tool_calls=[{"index": 0, "function": {"arguments": '{"query": '}}]),
        delta(tool_calls=[{"index": 0, "function": {"arguments": '"погода"}'}}]),
        delta(tool_calls=[{"index": 1, "id": "call_b", "function": {"name": "open_url", "arguments": "{}"}}]),
        delta(finish="tool_calls"),
    )
    result = await make_provider(handler).complete(make_request())
    assert [(c.id, c.name, json.loads(c.arguments)) for c in result.tool_calls] == [
        ("call_a", "web_search", {"query": "погода"}),
        ("call_b", "open_url", {}),
    ]
    assert result.finish_reason == "tool_calls"


async def test_error_chunk_before_any_text_raises_like_a_status(make_provider) -> None:
    error = {"error": {"message": "Upstream unavailable.", "code": "upstream_unavailable"}}
    with pytest.raises(ProviderUnavailable):
        await make_provider(sse(error)).complete(make_request())
    blocked = {"error": {"message": "Request blocked", "code": "jailbreak_detected"}}
    with pytest.raises(ProviderBadResponse):
        await make_provider(sse(blocked)).complete(make_request())


async def test_text_before_a_broken_stream_is_kept_as_cut(make_provider) -> None:
    error = {"error": {"message": "Upstream unavailable.", "code": "upstream_unavailable"}}
    result = await make_provider(sse(delta("Половина отве"), error)).complete(make_request())
    assert result.text == "Половина отве"
    assert result.finish_reason == "length"  # the client continues it
    no_finish = await make_provider(sse(delta("Оборвалось на"), done=False)).complete(make_request())
    assert no_finish.text == "Оборвалось на" and no_finish.finish_reason == "length"
    # Closed properly with [DONE] but no finish reason: a whole answer, not one to continue.
    closed = await make_provider(sse(delta("Готово."), done=True)).complete(make_request())
    assert closed.text == "Готово." and closed.finish_reason is None


async def test_a_dropped_connection_keeps_the_text_or_reports_unavailable(make_provider) -> None:
    class Broken(httpx.AsyncByteStream):
        def __init__(self, first: bytes) -> None:
            self.first = first

        async def __aiter__(self):
            if self.first:
                yield self.first
            raise httpx.RemoteProtocolError("peer closed connection")

    def handler(text: bytes) -> Handler:
        return lambda _req: httpx.Response(
            200, stream=Broken(text), headers={"content-type": "text/event-stream"}
        )

    chunk = b'data: {"choices": [{"delta": {"content": "Hello"}}]}\n\n'
    kept = await make_provider(handler(chunk)).complete(make_request())
    assert kept.text == "Hello" and kept.finish_reason == "length"
    with pytest.raises(ProviderUnavailable):
        await make_provider(handler(b"")).complete(make_request())


@pytest.mark.parametrize(
    "body",
    [
        {"error": {"message": "prompt is too long: 1200000 tokens > 1000000 maximum"}},
        {"error": {"message": "This model's maximum context length is 1048576 tokens"}},
    ],
)
async def test_an_over_window_prompt_is_reported_as_too_long(make_provider, body: dict[str, Any]) -> None:
    from bot.llm.base import ProviderTooLong

    with pytest.raises(ProviderTooLong):
        await make_provider(json_response(body, 400)).complete(make_request())
    with pytest.raises(ProviderTooLong):
        await make_provider(sse(body)).complete(make_request())


# ---- error mapping -----------------------------------------------------------------------------


def raising(exc_type: type[httpx.TransportError]) -> Handler:
    def handler(req: httpx.Request) -> httpx.Response:
        raise exc_type("boom", request=req)

    return handler


@pytest.mark.parametrize(
    ("handler", "expected", "status"),
    [
        (raising(httpx.ReadTimeout), ProviderTimeout, None),
        (raising(httpx.ConnectTimeout), ProviderTimeout, None),
        (raising(httpx.ConnectError), ProviderUnavailable, None),
        (json_response({"error": "slow down"}, 429), ProviderUnavailable, 429),
        (json_response({"error": "oops"}, 500), ProviderUnavailable, 500),
        (json_response({"error": "bad gateway"}, 502), ProviderUnavailable, 502),
        (json_response({"error": f"invalid key {API_KEY}"}, 401), ProviderAuthError, None),
        (json_response({"error": "no access"}, 403), ModelAccessDenied, None),
        (json_response({"error": "bad request"}, 400), ProviderBadResponse, None),
        (lambda _req: httpx.Response(200, content=b"<html>not json</html>"), ProviderBadResponse, None),
        (json_response(chat_body(choices=[])), ProviderBadResponse, None),
        (json_response({"model": "x"}), ProviderBadResponse, None),
        (json_response(chat_body(choices=[{"finish_reason": "stop"}])), ProviderBadResponse, None),
        (json_response([1, 2, 3]), ProviderBadResponse, None),
    ],
)
async def test_error_mapping(
    make_provider, handler: Handler, expected: type[ProviderError], status: int | None
) -> None:
    provider = make_provider(handler)
    with pytest.raises(expected) as info:
        await provider.complete(make_request())
    exc = info.value
    if isinstance(exc, ProviderUnavailable):
        assert exc.status == status
    message = str(exc)
    assert API_KEY not in message
    assert SECRET_TEXT not in message
    assert "gpt-6-astra" in message


async def test_error_message_includes_status_and_truncated_body(make_provider) -> None:
    body = "x" * 1000
    provider = make_provider(lambda _req: httpx.Response(400, text=body))
    with pytest.raises(ProviderBadResponse) as info:
        await provider.complete(make_request())
    message = str(info.value)
    assert "400" in message
    assert "x" * 200 in message
    assert "x" * 201 not in message


# ---- list_models -------------------------------------------------------------------------------


async def test_list_models(make_provider, captured: list[httpx.Request]) -> None:
    body = {"object": "list", "data": [{"id": "gpt-6-astra", "object": "model"}, {"id": "claude-x"}]}
    models = await make_provider(json_response(body)).list_models()
    assert models == ["gpt-6-astra", "claude-x"]
    (sent,) = captured
    assert sent.method == "GET"
    assert str(sent.url) == "https://gateway.test/v1/models"
    assert sent.headers["Authorization"] == f"Bearer {API_KEY}"
    assert sent.extensions["timeout"]["read"] == 30.0


@pytest.mark.parametrize(
    ("handler", "expected"),
    [
        (json_response({"object": "list"}), ProviderBadResponse),
        (json_response({"data": [{"object": "model"}]}), ProviderBadResponse),
        (json_response({"error": f"bad key {API_KEY}"}, 401), ProviderAuthError),
        (raising(httpx.ConnectError), ProviderUnavailable),
    ],
)
async def test_list_models_errors(make_provider, handler: Handler, expected: type[ProviderError]) -> None:
    with pytest.raises(expected) as info:
        await make_provider(handler).list_models()
    assert API_KEY not in str(info.value)


async def test_a_flowing_answer_outlives_the_first_text_deadline(make_provider, monkeypatch) -> None:
    import asyncio

    from bot.llm import buytokens

    monkeypatch.setattr(buytokens, "STREAM_MAX_S", 5.0)

    class Slow(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"choices": [{"delta": {"content": "first "}}]}\n\n'
            await asyncio.sleep(0.4)  # past the request's own 0.2 s: it already answers, so it may go on
            yield b'data: {"choices": [{"delta": {"content": "second"}, "finish_reason": "stop"}]}\n\n'

    provider = make_provider(
        lambda _req: httpx.Response(200, stream=Slow(), headers={"content-type": "text/event-stream"})
    )
    request = CompletionRequest("m", [ChatMessage("user", "q")], 100, timeout_s=0.2)
    result = await provider.complete(request)
    assert result.text == "first second" and result.finish_reason == "stop"


async def test_a_model_silent_past_its_timeout_times_out(make_provider) -> None:
    import asyncio

    class Silent(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b": keepalive\n\n"
            await asyncio.sleep(0.5)
            yield b'data: {"choices": [{"delta": {"content": "late"}}]}\n\n'

    provider = make_provider(
        lambda _req: httpx.Response(200, stream=Silent(), headers={"content-type": "text/event-stream"})
    )
    with pytest.raises(ProviderTimeout):
        await provider.complete(CompletionRequest("m", [ChatMessage("user", "q")], 100, timeout_s=0.2))


async def test_blank_chunks_do_not_count_as_an_answer_under_way(make_provider) -> None:
    import asyncio

    class Thinking(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"choices": [{"delta": {"content": "\\n\\n"}}]}\n\n'
            await asyncio.sleep(0.5)  # still thinking past the request's own 0.2 s
            yield b'data: {"choices": [{"delta": {"content": "late"}}]}\n\n'

    provider = make_provider(
        lambda _req: httpx.Response(200, stream=Thinking(), headers={"content-type": "text/event-stream"})
    )
    with pytest.raises(ProviderTimeout):
        await provider.complete(CompletionRequest("m", [ChatMessage("user", "q")], 100, timeout_s=0.2))


@pytest.mark.parametrize("separator", ["\u2028", "\u2029", "\u0085"])
async def test_unicode_separators_inside_sse_are_preserved(make_provider, separator):
    text = f"Первая{separator}вторая"
    result = await make_provider(sse(delta(text, "length"))).complete(make_request())
    assert result.text == text
    assert result.finish_reason == "length"


async def test_finished_answer_survives_an_error_after_finish(make_provider):
    error = {"error": {"code": "upstream_error", "message": "late usage failed"}}
    result = await make_provider(sse(delta("Полный ответ.", "stop"), error)).complete(make_request())
    assert result.text == "Полный ответ."
    assert result.finish_reason == "stop"


async def test_stream_errors_redact_keys_before_logging_or_raising(make_provider, caplog):
    error = {"error": {"code": "upstream_error", "message": f"upstream key {API_KEY} failed"}}
    with pytest.raises(ProviderUnavailable) as info:
        await make_provider(sse(error)).complete(make_request())
    assert API_KEY not in str(info.value)
    assert info.value.status == 502
    await make_provider(sse(delta("Часть ответа"), error)).complete(make_request())
    assert API_KEY not in caplog.text


async def test_surrogate_pair_split_between_deltas_is_valid_utf8(make_provider):
    chunks = [
        'data: {"choices":[{"delta":{"content":"\\ud83d"}}]}',
        'data: {"choices":[{"delta":{"content":"\\ude00"},"finish_reason":"stop"}]}',
    ]
    result = await make_provider(sse(*chunks)).complete(make_request())
    assert result.text == "😀"
    assert result.text.encode("utf-8")


async def test_unfinished_tool_call_is_not_executed_on_eof(make_provider):
    chunk = delta(tool_calls=[{"index": 0, "id": "c1", "function": {"name": "digest_file"}}])
    with pytest.raises(ProviderBadResponse, match="stream ended during tool calls"):
        await make_provider(sse(chunk, done=False)).complete(make_request())
