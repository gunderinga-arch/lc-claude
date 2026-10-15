from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import pytest

from bot.catalog import Catalog, ModelSpec
from bot.domain import Mode
from bot.llm.base import (
    ChatMessage,
    ChatStatus,
    Completion,
    CompletionRequest,
    ModelAccessDenied,
    ProviderAuthError,
    ProviderBadResponse,
    ProviderTimeout,
    ProviderUnavailable,
)
from bot.llm.resilience import ResilientLLMClient

MESSAGES = [ChatMessage("user", "hello")]


# ---- fakes -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Gated:
    """A step that blocks inside the provider call until `gate` is set."""

    gate: asyncio.Event
    completion: Completion


Step = Completion | BaseException | Gated


def ok(text: str = "answer", model: str = "m") -> Completion:
    return Completion(text, "stop", model, None, None)


def empty() -> Completion:
    return ok("")


class FakeProvider:
    """Plays a scripted list of steps, one per `complete` call, recording every request."""

    def __init__(self, steps: Iterable[Step]) -> None:
        self._steps: deque[Step] = deque(steps)
        self.requests: list[CompletionRequest] = []
        self.in_flight = 0
        self.max_in_flight = 0

    async def complete(self, request: CompletionRequest) -> Completion:
        self.requests.append(request)
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            if not self._steps:
                raise AssertionError(f"unexpected provider call: {request.model}")
            step = self._steps.popleft()
            if isinstance(step, BaseException):
                raise step
            if isinstance(step, Gated):
                await step.gate.wait()
                return step.completion
            return step
        finally:
            self.in_flight -= 1

    async def list_models(self) -> list[str]:
        return []

    async def aclose(self) -> None:
        return None

    @property
    def models(self) -> list[str]:
        return [r.model for r in self.requests]

    @property
    def max_tokens(self) -> list[int]:
        return [r.max_tokens for r in self.requests]


class RecordingSleep:
    def __init__(self) -> None:
        self.calls: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.calls.append(delay)


def spec(
    model_id: str, fallback: str | None = None, max_tokens: int = 1000, timeout_s: float = 60
) -> ModelSpec:
    return ModelSpec(model_id, model_id, max_tokens, timeout_s, fallback, enabled=True, slow=False)


def make_catalog(*specs: ModelSpec, max_tokens_cap: int = 3000, max_attempts: int = 3) -> Catalog:
    models = {s.id: s for s in specs}
    return Catalog(
        models,
        {mode: specs[0].id for mode in Mode},
        max_tokens_cap=max_tokens_cap,
        max_attempts=max_attempts,
    )


def make_client(
    provider: FakeProvider,
    catalog: Catalog,
    *,
    max_concurrency: int = 4,
    sleep: RecordingSleep | None = None,
    clock: Callable[[], float] | None = None,
) -> ResilientLLMClient:
    return ResilientLLMClient(
        provider,
        catalog,
        max_concurrency=max_concurrency,
        backoff_s=(5.0, 10.0),
        sleep=sleep or RecordingSleep(),
        clock=clock or (lambda: 0.0),
    )


def two_models(**kwargs: int) -> Catalog:
    return make_catalog(spec("a", fallback="b"), spec("b"), **kwargs)


async def wait_until(predicate: Callable[[], bool]) -> None:
    # Prompt fitting runs in a worker thread; yielding 100 event-loop ticks isn't a real deadline.
    async with asyncio.timeout(3):
        while not predicate():  # noqa: ASYNC110 - the production API exposes a predicate, not an event
            await asyncio.sleep(0.001)


# ---- happy path and validation-driven retries ------------------------------------------------


async def test_ok_first_try() -> None:
    provider = FakeProvider([ok("42")])
    result = await make_client(provider, two_models()).chat(MESSAGES, "a")

    assert result.status is ChatStatus.OK
    assert result.text == "42"
    assert (result.model_used, result.attempts, result.fallback_used, result.error) == ("a", 1, False, None)
    request = provider.requests[0]
    assert (request.model, request.max_tokens, request.timeout_s) == ("a", 1000, 60)
    assert list(request.messages) == MESSAGES


async def test_empty_doubles_max_tokens_up_to_cap() -> None:
    provider = FakeProvider([empty(), empty(), empty(), ok()])
    catalog = two_models(max_tokens_cap=3000, max_attempts=4)
    result = await make_client(provider, catalog).chat(MESSAGES, "a")

    assert result.status is ChatStatus.OK
    assert provider.max_tokens == [1000, 2000, 3000, 3000]
    assert provider.models == ["a"] * 4
    assert result.attempts == 4


async def test_cut_retries_same_model_same_max_tokens() -> None:
    sleep = RecordingSleep()
    provider = FakeProvider([ok("Let"), ok("##"), ok("full answer")])
    result = await make_client(provider, two_models(), sleep=sleep).chat(MESSAGES, "a")

    assert result.status is ChatStatus.OK
    assert result.text == "full answer"
    assert provider.models == ["a", "a", "a"]
    assert provider.max_tokens == [1000, 1000, 1000]
    assert sleep.calls == []


async def test_stub_returns_immediately_without_retries() -> None:
    provider = FakeProvider([ok("Oh, THAT island! Anyway...")])
    result = await make_client(provider, two_models()).chat(MESSAGES, "a")

    assert result.status is ChatStatus.STUB
    assert result.text == ""
    assert result.attempts == 1
    assert result.error is None


# ---- provider errors -------------------------------------------------------------------------


async def test_unavailable_backs_off_with_configured_delays_then_ok() -> None:
    sleep = RecordingSleep()
    provider = FakeProvider([ProviderUnavailable("503", status=503), ProviderBadResponse("bad"), ok()])
    result = await make_client(provider, two_models(), sleep=sleep).chat(MESSAGES, "a")

    assert result.status is ChatStatus.OK
    assert sleep.calls == [5.0, 10.0]
    assert provider.models == ["a", "a", "a"]
    assert result.fallback_used is False


async def test_backoff_reuses_last_delay_and_skips_sleep_after_final_attempt() -> None:
    sleep = RecordingSleep()
    provider = FakeProvider([ProviderUnavailable("503")] * 4 + [ok()])
    result = await make_client(provider, two_models(max_attempts=4), sleep=sleep).chat(MESSAGES, "a")

    assert result.status is ChatStatus.OK
    assert result.model_used == "b"
    assert sleep.calls == [5.0, 10.0, 10.0]


async def test_unexpected_exception_is_treated_as_bad_response(caplog: pytest.LogCaptureFixture) -> None:
    sleep = RecordingSleep()
    provider = FakeProvider([RuntimeError("boom"), ok()])
    result = await make_client(provider, two_models(), sleep=sleep).chat(MESSAGES, "a")

    assert result.status is ChatStatus.OK
    assert sleep.calls == [5.0]
    assert any(record.exc_info for record in caplog.records)


async def test_timeout_goes_straight_to_fallback() -> None:
    sleep = RecordingSleep()
    provider = FakeProvider([ProviderTimeout("slow"), ok("from b")])
    result = await make_client(provider, two_models(), sleep=sleep).chat(MESSAGES, "a")

    assert result.status is ChatStatus.OK
    assert result.text == "from b"
    assert provider.models == ["a", "b"]
    assert (result.model_used, result.fallback_used, result.attempts) == ("b", True, 2)
    assert sleep.calls == []


async def test_access_denied_marks_model_unavailable_and_falls_back() -> None:
    catalog = two_models()
    provider = FakeProvider([ModelAccessDenied("403"), ok()])
    result = await make_client(provider, catalog).chat(MESSAGES, "a")

    assert result.status is ChatStatus.OK
    assert result.fallback_used is True
    assert provider.models == ["a", "b"]
    assert not catalog.is_available("a")
    assert [s.id for s in catalog.fallback_chain("a")] == ["b"]


async def test_auth_error_fails_without_trying_fallbacks() -> None:
    provider = FakeProvider([ProviderAuthError("401")])
    result = await make_client(provider, two_models()).chat(MESSAGES, "a")

    assert result.status is ChatStatus.FAILED
    assert result.error == "auth"
    assert provider.models == ["a"]
    assert (result.attempts, result.fallback_used) == (1, False)


async def test_everything_failing_reports_last_error_kind() -> None:
    sleep = RecordingSleep()
    provider = FakeProvider([empty(), ok("Let"), ok("```")] + [ProviderUnavailable("503")] * 3)
    result = await make_client(provider, two_models(), sleep=sleep).chat(MESSAGES, "a")

    assert result.status is ChatStatus.FAILED
    assert result.error == "unavailable"
    assert result.text == ""
    assert (result.model_used, result.attempts, result.fallback_used) == ("b", 6, True)
    assert provider.models == ["a"] * 3 + ["b"] * 3


@pytest.mark.parametrize(
    ("steps", "error"),
    [
        ([empty()] * 3, "empty"),
        ([ok("**")] * 3, "cut"),
        ([ProviderBadResponse("x")] * 3, "bad_response"),
        ([ProviderTimeout("t")], "timeout"),
        ([ModelAccessDenied("403")], "access_denied"),
    ],
)
async def test_single_model_failure_kinds(steps: list[Step], error: str) -> None:
    provider = FakeProvider(steps)
    result = await make_client(provider, make_catalog(spec("solo"))).chat(MESSAGES, "solo")

    assert result.status is ChatStatus.FAILED
    assert result.error == error
    assert (result.model_used, result.attempts, result.fallback_used) == ("solo", len(steps), False)


async def test_unknown_model_reports_no_models() -> None:
    provider = FakeProvider([])
    result = await make_client(provider, two_models()).chat(MESSAGES, "nope")

    assert result.status is ChatStatus.FAILED
    assert (result.error, result.attempts, result.model_used) == ("no_models", 0, "nope")
    assert provider.requests == []


async def test_empty_chain_reports_no_models() -> None:
    catalog = two_models()
    catalog.mark_unavailable("a")
    catalog.mark_unavailable("b")
    result = await make_client(FakeProvider([]), catalog).chat(MESSAGES, "a")

    assert (result.status, result.error, result.attempts) == (ChatStatus.FAILED, "no_models", 0)


async def test_latency_measured_with_injected_clock() -> None:
    ticks = iter([100.0, 102.5])
    provider = FakeProvider([ok()])
    result = await make_client(provider, two_models(), clock=lambda: next(ticks)).chat(MESSAGES, "a")

    assert result.latency_ms == 2500


def test_rejects_invalid_configuration() -> None:
    catalog = two_models()
    with pytest.raises(ValueError):
        ResilientLLMClient(FakeProvider([]), catalog, max_concurrency=0)
    with pytest.raises(ValueError):
        ResilientLLMClient(FakeProvider([]), catalog, max_concurrency=1, backoff_s=())


# ---- concurrency and cancellation ------------------------------------------------------------


async def test_concurrency_limit_and_queue_depth() -> None:
    gate = asyncio.Event()
    provider = FakeProvider([Gated(gate, ok("first")), ok("second")])
    client = make_client(provider, two_models(), max_concurrency=1)

    first = asyncio.create_task(client.chat(MESSAGES, "a"))
    await wait_until(lambda: provider.in_flight == 1)
    second = asyncio.create_task(client.chat(MESSAGES, "a"))
    await wait_until(lambda: client.queue_depth() == 1)
    assert len(provider.requests) == 1

    gate.set()
    results = await asyncio.wait_for(asyncio.gather(first, second), timeout=5)

    assert [r.text for r in results] == ["first", "second"]
    assert provider.max_in_flight == 1
    assert client.queue_depth() == 0


async def test_cancel_mid_call_propagates_and_releases_slot() -> None:
    never = asyncio.Event()
    provider = FakeProvider([Gated(never, ok()), ok("after")])
    client = make_client(provider, two_models(), max_concurrency=1)

    task = asyncio.create_task(client.chat(MESSAGES, "a"))
    await wait_until(lambda: provider.in_flight == 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    result = await asyncio.wait_for(client.chat(MESSAGES, "a"), timeout=5)
    assert result.text == "after"
    assert client.queue_depth() == 0


async def test_cancel_while_queued_resets_queue_depth() -> None:
    gate = asyncio.Event()
    provider = FakeProvider([Gated(gate, ok("first"))])
    client = make_client(provider, two_models(), max_concurrency=1)

    first = asyncio.create_task(client.chat(MESSAGES, "a"))
    await wait_until(lambda: provider.in_flight == 1)
    queued = asyncio.create_task(client.chat(MESSAGES, "a"))
    await wait_until(lambda: client.queue_depth() == 1)
    queued.cancel()
    with pytest.raises(asyncio.CancelledError):
        await queued

    assert client.queue_depth() == 0
    gate.set()
    assert (await asyncio.wait_for(first, timeout=5)).text == "first"
    assert len(provider.requests) == 1


# ---- answers cut at the token limit, and the reserve as the last resort ----------------------


def cut(text: str) -> Completion:
    return Completion(text, "length", "m", None, None)


async def test_cut_answer_is_continued_by_the_same_model_and_the_overlap_dropped() -> None:
    from bot.llm.resilience import CONTINUE_PROMPT

    provider = FakeProvider(
        [cut("Метр определили в 1791 году как долю меридиана"), ok("в 1791 году как долю меридиана Земли.")]
    )
    result = await make_client(provider, two_models()).chat(MESSAGES, "a")
    assert result.status is ChatStatus.OK
    assert result.text == "Метр определили в 1791 году как долю меридиана Земли."
    assert provider.models == ["a", "a"]
    asked = provider.requests[1].messages
    assert asked[-2].role == "assistant"
    assert asked[-1].content == CONTINUE_PROMPT.format(tail="Метр определили в 1791 году как долю меридиана")


async def test_still_cut_after_continuations_goes_out_with_a_note() -> None:
    from bot.llm.resilience import CUT_NOTE, MAX_CONTINUATIONS

    words = [f"слово{n} " for n in range(1 + MAX_CONTINUATIONS)]
    provider = FakeProvider([cut(w) for w in words])
    result = await make_client(provider, two_models()).chat(MESSAGES, "a")
    assert result.status is ChatStatus.OK
    assert result.text == "".join(words) + CUT_NOTE
    assert len(provider.requests) == 1 + MAX_CONTINUATIONS


def test_a_cut_part_loses_its_unreliable_end_back_to_a_line_end() -> None:
    from bot.llm.resilience import CUT_TRIM_CHARS, trim_cut

    text = "".join(f"строка {n}\n" for n in range(200)) + "оборванная стро"
    trimmed = trim_cut(text)
    assert trimmed.endswith("\n") and text.startswith(trimmed)
    assert CUT_TRIM_CHARS <= len(text) - len(trimmed) < CUT_TRIM_CHARS + 20
    assert trim_cut("коротко") == "коротко"
    assert trim_cut("а " * 1000).endswith("а ")  # no line ends: back to a space


async def test_a_clean_continuation_and_a_lead_in_are_joined_without_the_glue_model() -> None:
    from bot.llm.resilience import trim_cut

    before = "".join(f"Абзац {n} первого куска.\n" for n in range(100)) + "конец первого кус"
    kept = trim_cut(before)
    more = "Продолжаю перевод:\nАбзац заново, а дальше второй кусок " + "Б" * 2000
    provider = FakeProvider([cut(before), ok(more)])
    client = make_client(provider, with_glue())
    client._glue = "g"
    result = await client.chat(MESSAGES, "a")
    assert provider.models == ["a", "a"]  # nothing ambiguous: Haiku is not asked
    assert result.text == kept + "Абзац заново, а дальше второй кусок " + "Б" * 2000
    assert kept[-200:] in provider.requests[1].messages[-1].content  # the tail it was asked to go on from


async def test_a_reopened_code_fence_is_dropped_at_the_seam() -> None:
    before = "Код:\n```python\ndef f():\n    return 1\n\ndef g():\n    ret"
    more = "```python\nurn 2\n```"
    provider = FakeProvider([cut(before), ok(more)])
    result = await make_client(provider, two_models()).chat(MESSAGES, "a")
    assert result.text == before + "urn 2\n```"


async def test_an_altered_repeat_goes_to_the_glue_model_with_the_seam_only() -> None:
    from bot.llm.resilience import SEAM_CHARS

    before = "А" * 400 + " и вот последние слова первого куска перед обры"
    more = "последние  слова первого\nкуска перед обрывом, а дальше второй кусок " + "Б" * 2000
    b = more[:SEAM_CHARS]
    glued = (
        before[-SEAM_CHARS : -len("последние слова первого куска перед обры")]
        + "последние слова первого куска перед обрывом, а дальше второй кусок "
        + b[len("последние  слова первого\nкуска перед обрывом, а дальше второй кусок ") :]
    )
    provider = FakeProvider([cut(before), ok(more), ok(glued)])
    client = make_client(provider, with_glue())
    client._glue = "g"
    result = await client.chat(MESSAGES, "a")
    assert provider.models == ["a", "a", "g"]
    assert result.text == before[: -len("последние слова первого куска перед обры")] + (
        "последние слова первого куска перед обрывом, а дальше второй кусок " + "Б" * 2000
    )
    seam_request = provider.requests[2].messages[0].content
    assert len(seam_request) < 2 * SEAM_CHARS + 1000  # the glue model never sees the whole answer


async def test_an_unsafe_glued_seam_is_ignored() -> None:
    before = "x" * 400 + " повтор с изменением в конце первого куска"
    more = "повтор  с изменением в конце первого куска, и дальше" + "y" * 400
    provider = FakeProvider([cut(before), ok(more), ok("что-то совсем другое")])
    client = make_client(provider, with_glue())
    client._glue = "g"
    result = await client.chat(MESSAGES, "a")
    assert provider.models == ["a", "a", "g"]
    assert result.text == before + more  # nothing lost: the parts as they came


async def test_a_continuation_with_junk_first_is_joined_after_the_anchor_without_the_glue_model() -> None:
    before = "Начало ответа. " + "текст " * 40 + "последние слова перед обрывом"
    more = "Вот продолжение:\n" + before[-200:] + " и продолжение."
    provider = FakeProvider([cut(before), ok(more)])
    client = make_client(provider, with_glue())
    client._glue = "g"
    result = await client.chat(MESSAGES, "a")
    assert result.text == before + " и продолжение."
    assert provider.models == ["a", "a"]


async def test_a_continuation_that_starts_the_answer_over_is_asked_again() -> None:
    from bot.llm.resilience import trim_cut

    answer = "Глава первая. " + " ".join(f"слово{n}" for n in range(3000))
    before = answer[:12_000]
    kept = trim_cut(before)
    redo = answer[:5000] + " но уже по-другому"
    provider = FakeProvider([cut(before), cut(redo), ok(answer[len(kept) :])])
    result = await make_client(provider, two_models()).chat(MESSAGES, "a")
    assert result.text == answer
    assert len(provider.requests) == 3


def with_glue() -> Catalog:
    glue = ModelSpec("g", "g", 1000, 60, None, enabled=True, slow=False, reserve=True)
    return make_catalog(spec("a", fallback="b"), spec("b"), glue)


async def test_a_failed_continuation_keeps_what_arrived() -> None:
    from bot.llm.resilience import CUT_NOTE

    provider = FakeProvider(
        [cut("начало"), ProviderUnavailable("down"), ProviderUnavailable("down"), ProviderUnavailable("down")]
    )
    result = await make_client(provider, two_models()).chat(MESSAGES, "a")
    assert result.status is ChatStatus.OK and result.text == "начало" + CUT_NOTE


def with_reserve() -> Catalog:
    reserve = ModelSpec("r", "r", 1000, 60, None, enabled=True, slow=False, reserve=True)
    return make_catalog(spec("a", fallback="b"), spec("b"), reserve)


async def test_the_reserve_answers_when_the_whole_chain_failed() -> None:
    provider = FakeProvider(
        [ProviderTimeout(), ProviderTimeout(), Completion("спас", "stop", "glm", None, None)]
    )
    client = make_client(provider, with_reserve())
    client._rescue = "r"
    result = await client.chat(MESSAGES, "a")
    assert provider.models == ["a", "b", "r"]
    assert result.status is ChatStatus.OK and result.text == "спас"
    assert result.model_used == "r" and result.fallback_used


async def test_no_rescue_without_a_reserve_or_for_reserve_chains() -> None:
    provider = FakeProvider([ProviderTimeout(), ProviderTimeout()])
    result = await make_client(provider, with_reserve()).chat(MESSAGES, "a")  # no rescue model configured
    assert result.status is ChatStatus.FAILED and provider.models == ["a", "b"]

    provider = FakeProvider([ProviderTimeout()])
    client = make_client(provider, with_reserve())
    client._rescue = "r"
    result = await client.chat(MESSAGES, "r")  # already on the reserve: nothing more to try
    assert result.status is ChatStatus.FAILED and provider.models == ["r"]


def test_join_drops_a_short_repeat_after_a_cut_in_the_middle_of_a_number() -> None:
    from bot.llm.resilience import join_continuation

    before = '{"font": "Inter Tight", "size": 10.'
    more = '"size": 10.5, "color": "#5A5A5A"}'
    assert join_continuation(before, more) == '{"font": "Inter Tight", "size": 10.5, "color": "#5A5A5A"}'
    assert join_continuation("он сказал ", "что придёт") == "он сказал что придёт"  # nothing repeated


# ---- the context window --------------------------------------------------------------------------


def windowed(window: int, max_tokens: int = 100) -> Catalog:
    from dataclasses import replace as dc_replace

    a = dc_replace(spec("a", fallback="b", max_tokens=max_tokens), context_tokens=window)
    b = dc_replace(spec("b", max_tokens=max_tokens), context_tokens=window)
    return make_catalog(a, b)


def dialog(turns: int, words: int) -> list[ChatMessage]:
    history = [
        ChatMessage("user" if n % 2 == 0 else "assistant", f"реплика {n} " + "слово " * words, droppable=True)
        for n in range(turns)
    ]
    return [ChatMessage("system", "правила"), *history, ChatMessage("user", "последний вопрос")]


async def test_a_dialog_over_the_window_loses_its_earliest_turns_only() -> None:
    from bot.llm.window import prompt_tokens

    messages = dialog(20, 300)
    provider = FakeProvider([ok()])
    await make_client(provider, windowed(4000)).chat(messages, "a")
    sent = provider.requests[0].messages
    assert sent[0].content == "правила" and sent[-1].content == "последний вопрос"
    assert sent[1].role == "user"  # never an answer to a dropped question first
    assert 2 < len(sent) < len(messages)
    assert sent[1:-1] == messages[len(messages) - len(sent) + 1 : -1]  # the latest turns, intact
    assert prompt_tokens(sent) <= 4000 * 0.96 - 100


async def test_a_message_bigger_than_the_window_is_cut_in_the_middle() -> None:
    from bot.llm.window import CUT_MARK

    huge = "начало " + "середина " * 5000 + " конец"
    provider = FakeProvider([ok()])
    await make_client(provider, windowed(3000)).chat(
        [ChatMessage("system", "s"), ChatMessage("user", huge)], "a"
    )
    sent = provider.requests[0].messages[-1].content
    assert sent.startswith("начало ") and sent.endswith(" конец")
    assert CUT_MARK.split("{")[0] in sent and len(sent) < len(huge)


async def test_a_prompt_refused_as_too_long_is_cut_deeper_and_retried_at_once() -> None:
    from bot.llm.base import ProviderTooLong
    from bot.llm.window import prompt_tokens

    sleep = RecordingSleep()
    provider = FakeProvider([ProviderTooLong("too long"), ok()])
    messages = dialog(20, 300)
    client = make_client(provider, windowed(100_000), sleep=sleep)
    result = await client.chat(messages, "a")
    assert result.status is ChatStatus.OK and provider.models == ["a", "a"]
    first, second = (prompt_tokens(r.messages) for r in provider.requests)
    assert second <= first * 0.85 + 1
    assert sleep.calls == []


async def test_a_bad_response_to_a_prompt_near_the_window_counts_as_too_long() -> None:
    from bot.llm.window import prompt_tokens

    provider = FakeProvider([ProviderBadResponse("POST /chat/completions: HTTP 400: Invalid request"), ok()])
    messages = dialog(40, 300)
    result = await make_client(provider, windowed(20_000)).chat(messages, "a")
    assert result.status is ChatStatus.OK
    first, second = (prompt_tokens(r.messages) for r in provider.requests)
    assert second < first


def test_only_a_lead_in_line_is_dropped_never_the_answers_own_first_line() -> None:
    from bot.llm.resilience import _without_lead

    for junk in (
        "Продолжаю:",
        "Продолжаю перевод с места обрыва:",
        "(продолжение)",
        "*Продолжение:*",
        "Continuing…",
    ):
        assert _without_lead("x", f"{junk}\nтекст") == "текст", junk
    for own in ("Продолжение истории", "Продолжая мысль, скажу так", "## Продолжение работы"):
        assert _without_lead("x", f"{own}\nтекст") == f"{own}\nтекст", own


@pytest.mark.parametrize(
    "line", ["**Продолжительность:**", "Continuous delivery:", "Продолжительность (дни)"]
)
def test_real_heading_survives_continuation_cleanup(line):
    from bot.llm.resilience import join_continuation

    assert join_continuation("Начало\n", line + "\nДальше") == "Начало\n" + line + "\nДальше"


def test_closing_fence_and_repeated_structural_lines_survive():
    from bot.llm.resilience import join_continuation

    assert join_continuation("```python\nprint(1)\n", "```\nГотово.") == "```python\nprint(1)\n```\nГотово."
    before, more = "<div><div>текст</div>\n", "</div>\n"
    assert join_continuation(before, more) == before + more
    boilerplate = "<div class='row'>Обычная повторяющаяся строка</div>\n" * 4
    before, more = "Первый блок\n" + boilerplate, "Второй блок\n" + boilerplate + "Конец"
    assert join_continuation(before, more) == before + more


async def test_short_broken_continuation_never_erases_reliable_text_or_loops():
    from bot.llm.resilience import CUT_NOTE, trim_cut

    before = "".join(f"Строка {n}. Надёжный текст.\n" for n in range(500))
    provider = FakeProvider([cut(before), cut("Новая короткая часть")])
    result = await make_client(provider, two_models()).chat(MESSAGES, "a")
    assert result.text == trim_cut(before) + CUT_NOTE
    assert len(provider.requests) == 2


async def test_restart_with_lead_in_is_detected():
    from bot.llm.resilience import trim_cut

    answer = "Глава первая. " + " ".join(f"слово{n}" for n in range(3000))
    before = answer[:12000]
    provider = FakeProvider(
        [cut(before), cut("Продолжаю:\n" + answer[:5000]), ok(answer[len(trim_cut(before)) :])]
    )
    result = await make_client(provider, two_models(), sleep=RecordingSleep()).chat(MESSAGES, "a")
    assert result.text == answer


async def test_in_stream_refusal_never_shrinks_the_document():
    provider = FakeProvider([ProviderBadResponse("error in the stream: blocked"), ok()])
    result = await make_client(provider, windowed(20_000), sleep=RecordingSleep()).chat(dialog(40, 300), "a")
    assert result.status is ChatStatus.OK
    assert provider.requests[0].messages == provider.requests[1].messages


async def test_usage_counts_retries_and_continuation_requests():
    from bot.llm.window import prompt_tokens

    provider = FakeProvider([ProviderUnavailable("busy"), cut("Начало"), ok(" и конец")])
    result = await make_client(provider, two_models(), sleep=RecordingSleep()).chat(MESSAGES, "a")
    assert result.input_tokens == sum(prompt_tokens(r.messages) for r in provider.requests)
    assert result.output_tokens > 0
    assert len(provider.requests) == 3
