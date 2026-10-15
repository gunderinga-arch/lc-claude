"""`LLMClient` over a flaky `LLMProvider`: concurrency limit, validation, retries and fallbacks."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum

from bot.catalog import Catalog, ModelSpec, UnknownModelError
from bot.domain import count_tokens
from bot.llm.base import (
    ChatMessage,
    ChatResult,
    ChatStatus,
    Completion,
    CompletionRequest,
    LLMProvider,
    ModelAccessDenied,
    ProviderAuthError,
    ProviderBadResponse,
    ProviderTimeout,
    ProviderTooLong,
    ProviderUnavailable,
    ToolCall,
    ToolSpec,
    Verdict,
)
from bot.llm.validator import classify
from bot.llm.window import WINDOW_SLACK, fit_window, prompt_tokens

logger = logging.getLogger(__name__)

Sleep = Callable[[float], Awaitable[None]]
Clock = Callable[[], float]


class _Outcome(StrEnum):
    """Result of one HTTP attempt. Failure values double as `ChatResult.error` codes."""

    OK = "ok"
    STUB = "stub"
    AUTH = "auth"
    EMPTY = "empty"
    CUT = "cut"
    UNAVAILABLE = "unavailable"
    BAD_RESPONSE = "bad_response"
    TIMEOUT = "timeout"
    ACCESS_DENIED = "access_denied"
    TOO_LONG = "too_long"  # the prompt is over the model's window: cut deeper and retry at once


_VERDICT_OUTCOMES: dict[Verdict, _Outcome] = {
    Verdict.OK: _Outcome.OK,
    Verdict.STUB: _Outcome.STUB,
    Verdict.EMPTY: _Outcome.EMPTY,
    Verdict.CUT: _Outcome.CUT,
}

# Outcomes that end the whole chat, whatever fallbacks remain.
_TERMINAL: frozenset[_Outcome] = frozenset({_Outcome.OK, _Outcome.STUB, _Outcome.AUTH})
# Failures after which this model is abandoned for the next one in the chain.
_ABANDON_MODEL: frozenset[_Outcome] = frozenset({_Outcome.TIMEOUT, _Outcome.ACCESS_DENIED})
_NEEDS_BACKOFF: frozenset[_Outcome] = frozenset({_Outcome.UNAVAILABLE, _Outcome.BAD_RESPONSE})


@dataclass(frozen=True, slots=True)
class _Attempt:
    outcome: _Outcome
    text: str = ""  # the answer when outcome is OK
    tool_calls: tuple[ToolCall, ...] = ()
    served_by: str | None = None  # Completion.model when the reserve gateway answered
    finish: str | None = None  # the gateway's finish_reason; "length" = cut at the token limit
    reason: str = ""  # why it failed, for the log (the provider's message: no key, no user text)


# An answer cut at the output limit (~50k tokens a part on the gateway, 15.10.2026) or by a broken stream is
# continued by the same model this many times — up to ~10 parts — then sent with a note.
MAX_CONTINUATIONS = 9
CONTINUE_TAIL_CHARS = 300  # the end of the answer quoted in the request, so the model knows where to pick up
CONTINUE_PROMPT = (
    "Твой ответ оборвался на лимите длины. Вот его последние символы:\n<<<\n{tail}\n>>>\n"
    "Продолжи ровно со следующего символа после них: без повторов, вступления и пояснений, тем же языком и "
    "форматом. Если обрыв пришёлся на середину слова, числа или строки кода — допиши их. Если оборвался блок "
    "файла — допиши его и закрой."
)
CUT_NOTE = "\n\n_(Ответ оборвался на лимите длины. Напишите «продолжи» — допишу.)_"
MAX_OVERLAP = (
    3000  # chars a continuation may repeat from the end of the previous part (the quoted tail and more)
)
# The shortest repeat that counts. 14.10.2026 a deck cut at `"size": 10.` went on with `"size": 10.5` — an
# 11-char repeat the old 16-char floor kept, and the JSON broke; 6 still ignores a repeated short word.
MIN_OVERLAP = 6
# A continuation that starts with junk ("Продолжаю:", a reopened ``` fence) or anew from an earlier point
# still holds the previous part's last chars: the join goes right after them (long enough not to match by
# chance).
ANCHOR_CHARS = 120
ANCHOR_SEARCH_CHARS = 4000  # only the start of the continuation is searched: a repeat sits there
# The rest is a seam for the glue model: this much of each side, never the whole answer.
SEAM_CHARS = 700
SEAM_CHECK_CHARS = 40  # the glued seam must keep both ends exactly: the glue model may only mend the middle
GLUE_PROMPT = (
    "Длинный ответ другой модели пришёл двумя кусками: первый оборвался на лимите длины, "
    "второй его продолжает. "
    "На стыке бывают повтор текста, оборванное слово или число, лишнее вступление («Продолжаю:»), лишняя "
    "или пропущенная ``` . Ниже конец первого куска и начало второго. Склей их в один непрерывный текст: "
    "убери повтор и мусор на стыке, допиши оборванное слово по второму куску, больше ничего не меняй — "
    "ни слов, ни пробелов, ни переносов, ни разметки. Верни только склеенный текст, без пояснений.\n\n"
    "<<<КОНЕЦ_ПЕРВОГО\n{a}\nКОНЕЦ_ПЕРВОГО>>>\n\n<<<НАЧАЛО_ВТОРОГО\n{b}\nНАЧАЛО_ВТОРОГО>>>"
)
# A refused prompt: cut by this share more on the next attempt. BuyTokens answers a prompt over the window
# with a bare "Invalid request" 400, so a bad response to a prompt this close to the window is "too long" too.
TOO_LONG_SHRINK = 0.15
BIG_PROMPT_SHARE = 0.6


# A lead-in a continuation may open with, though asked not to; and a code fence it reopens.
# Only a short line ending like a lead-in ("Продолжаю перевод:", "(продолжение)", "Continuing…"): a heading
# "Продолжение истории" or a paragraph "Продолжая мысль, …" is the answer itself.
_JUNK_LEAD = re.compile(
    r"\A[ \t]*[(*_]*(?:продолжаю|продолжение|continuing|continued|continuation)\b"
    r"[^\n]{0,80}?(?::|\)|…|\.\.\.)[*_]*[ \t]*\n+",
    re.I,
)
_FENCE_LEAD = re.compile(r"\A[ \t]*```[\w+-]+[ \t]*\n")
_FENCE_LINE = re.compile(r"^[ \t]*```", re.M)
_SPACES = re.compile(r"\s+")
NEAR_REPEAT_CHARS = (
    30  # this much of a continuation's start found in the answer's end = a repeat, maybe altered
)


def _without_lead(before: str, more: str) -> str:
    """`more` without a "Продолжаю…" line, and without a ``` if `before` stopped inside a code block."""
    out = _JUNK_LEAD.sub("", more, count=1)
    if len(_FENCE_LINE.findall(before)) % 2:
        out = _FENCE_LEAD.sub("", out, count=1)
    return out


def _flat(text: str) -> str:
    return _SPACES.sub(" ", text).strip()


def restarted(before: str, more: str) -> bool:
    """The continuation began the whole answer anew (seen 15.10.2026 on Fable with a prefilled answer)."""
    head = _flat(more[:400])[:80]
    return len(before) > 2 * ANCHOR_SEARCH_CHARS and len(head) >= 40 and _flat(before[:400]).startswith(head)


# At a cut the gateway may send its last chunks out of order (15.10.2026: "…чепуху, ⟨этот наивный вопрос …
# ответил кня⟩⟨но в сущности на⟩", the last piece belonging before the one ahead of it): a cut part loses its
# last ~CUT_TRIM_CHARS, back to a line end, and the continuation writes them anew.
CUT_TRIM_CHARS = 300


def trim_cut(text: str) -> str:
    """A part cut at the length limit without its unreliable end (back to a line end, else a space)."""
    if len(text) <= 2 * CUT_TRIM_CHARS:
        return text
    limit = len(text) - CUT_TRIM_CHARS
    line = text.rfind("\n", max(0, limit - 2000), limit)
    if line > 0:
        return text[: line + 1]
    space = text.rfind(" ", max(0, limit - 300), limit)
    return text[: space + 1] if space > 0 else text[:limit]


def overlap_join(before: str, more: str) -> str | None:
    """`before` + `more` without what the continuation repeats, or None when the seam is not certain.

    Tried on `more` as it came and without its lead-in (`_without_lead`):
    1. `more` starts with the end of `before` (6 to 3000 chars): drop the repeat.
    2. `more` holds the last ANCHOR_CHARS of `before` near its start (or anywhere, if it started the whole
       answer over): join right after them.
    """
    for candidate in dict.fromkeys((more, _without_lead(before, more))):
        for size in range(min(MAX_OVERLAP, len(before), len(candidate)), MIN_OVERLAP - 1, -1):
            if before.endswith(candidate[:size]):
                # Repeated structural lines (nested closing tags/braces) can be real content.
                if before.endswith("\n") and not re.search(
                    r"[\w]{3}", re.sub(r"<[^>]*>", "", candidate[:size])
                ):
                    continue
                return before + candidate[size:]
        if len(before) >= ANCHOR_CHARS:
            if not (restarted(before, candidate) or candidate.startswith(("Вот продолжение:", "Продолжаю:"))):
                continue
            reach = len(candidate) if restarted(before, candidate) else ANCHOR_SEARCH_CHARS + ANCHOR_CHARS
            at = candidate.find(before[-ANCHOR_CHARS:], 0, reach)
            if at >= 0:
                return before + candidate[at + ANCHOR_CHARS :]
    return None


def needs_glue(before: str, more: str) -> bool:
    """A repeat the exact checks missed: the continuation's start is in the answer's end, give or take spaces.

    Anything else that did not overlap is a clean continuation (it starts right after the cut), joined as is:
    the glue model is asked only about these seams.
    """
    head = _flat(_without_lead(before, more)[:400])[:NEAR_REPEAT_CHARS]
    return len(head) == NEAR_REPEAT_CHARS and head in _flat(before[-MAX_OVERLAP:])


def join_continuation(before: str, more: str) -> str:
    """`before` + `more` without the repeat and the lead-in; with no repeat found they are put together."""
    joined = overlap_join(before, more)
    return joined if joined is not None else before + _without_lead(before, more)


def _strip_wrapper(text: str) -> str:
    """A glue model's answer without a ``` fence it may wrap everything in."""
    lines = text.strip("\n").split("\n")
    if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1])
    return text.strip("\n")


@dataclass(slots=True)
class _Run:
    """Mutable bookkeeping for one `chat` call."""

    attempts: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


class ResilientLLMClient:
    """Implements `LLMClient`. Never raises for provider failures; cancellation propagates."""

    def __init__(
        self,
        provider: LLMProvider,
        catalog: Catalog,
        *,
        max_concurrency: int,
        backoff_s: Sequence[float] = (5.0, 10.0),
        sleep: Sleep = asyncio.sleep,
        clock: Clock = time.monotonic,
        rescue_model: str | None = None,
        glue_model: str | None = None,
    ) -> None:
        """`rescue_model` (a reserve-gateway model) gets one last try when the whole fallback chain failed.

        `glue_model` (a fast one) mends the seams of a long answer written in parts, where the plain overlap
        check is not sure; it sees only SEAM_CHARS on each side of a seam.
        """
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        if not backoff_s or any(delay < 0 for delay in backoff_s):
            raise ValueError("backoff_s must be a non-empty sequence of non-negative delays")
        self._provider = provider
        self._catalog = catalog
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._backoff_s = tuple(backoff_s)
        self._sleep = sleep
        self._clock = clock
        self._waiting = 0
        self._rescue = rescue_model
        self._glue = glue_model

    def queue_depth(self) -> int:
        """How many provider calls are waiting for a free gateway slot right now."""
        return self._waiting

    async def chat(
        self, messages: Sequence[ChatMessage], model_id: str, tools: Sequence[ToolSpec] = ()
    ) -> ChatResult:
        started = self._clock()
        try:
            chain = self._catalog.fallback_chain(model_id)
        except UnknownModelError:
            chain = []
        if not chain:
            result = self._result(started, ChatStatus.FAILED, model_id, 0, False, error="no_models")
            self._log_result(result)
            return result

        run = _Run()
        last: _Outcome = _Outcome.UNAVAILABLE  # overwritten: every model makes >= 1 attempt
        for spec in [*chain, *self._rescue_for(chain)]:
            attempt = await self._try_model(spec, messages, tools, run)
            if attempt.outcome in _TERMINAL:
                if attempt.outcome is _Outcome.OK and attempt.finish == "length" and not attempt.tool_calls:
                    attempt = await self._continue(spec, messages, attempt, run)
                result = self._terminal_result(started, attempt, spec.id, run.attempts, spec.id != model_id)
                result = replace(result, input_tokens=run.input_tokens, output_tokens=run.output_tokens)
                self._log_result(result)
                return result
            last = attempt.outcome

        result = self._result(
            started, ChatStatus.FAILED, chain[-1].id, run.attempts, chain[-1].id != model_id, error=last.value
        )
        result = replace(result, input_tokens=run.input_tokens, output_tokens=run.output_tokens)
        self._log_result(result)
        return result

    def _rescue_for(self, chain: list[ModelSpec]) -> list[ModelSpec]:
        """The reserve model as the chain's last resort; not for reserve chains (already on the reserve)."""
        if self._rescue is None or chain[0].reserve or any(s.id == self._rescue for s in chain):
            return []
        try:
            return [self._catalog.get(self._rescue)]
        except UnknownModelError:
            return []

    async def _continue(
        self, spec: ModelSpec, messages: Sequence[ChatMessage], attempt: _Attempt, run: _Run
    ) -> _Attempt:
        """An answer cut at the token limit: the same model goes on from the cut, MAX_CONTINUATIONS times.

        Still cut after that (or a continuation failed) -> what arrived, with a note for the user.
        """
        text, finish, parts, glued = trim_cut(attempt.text), attempt.finish, 1, 0
        for _ in range(MAX_CONTINUATIONS):
            tail = text[-CONTINUE_TAIL_CHARS:]
            asked = [
                *messages,
                ChatMessage("assistant", text),
                ChatMessage("user", CONTINUE_PROMPT.format(tail=tail)),
            ]
            more = await self._try_model(spec, asked, (), run)
            if more.outcome is not _Outcome.OK:
                break
            joined = overlap_join(text, more.text)
            if joined is None and restarted(text, _without_lead(text, more.text)):
                logger.warning("a continuation started the answer over and lost the place: model=%s", spec.id)
                await self._sleep(self._backoff_s[0])
                continue  # asked again: the loop's bound still holds
            if joined is None and needs_glue(text, more.text):
                joined = await self._glue_seam(text, more.text, run)
                glued += joined is not None
            previous = text
            text = joined if joined is not None else text + _without_lead(text, more.text)
            finish, parts = more.finish, parts + 1
            if finish != "length":
                break
            trimmed = trim_cut(text)
            text = trimmed if len(trimmed) > len(previous) else previous
            if text == previous:
                break  # a short broken part must neither erase reliable text nor hammer the gateway
        if finish == "length":
            logger.warning(
                "answer cut at the length limit and not completed: model=%s parts=%d", spec.id, parts
            )
            text += CUT_NOTE
        else:
            logger.info(
                "answer cut at the length limit, continued: model=%s parts=%d glued=%d chars=%d",
                spec.id,
                parts,
                glued,
                len(text),
            )
        return replace(attempt, text=text, finish=finish)

    async def _glue_seam(self, before: str, more: str, run: _Run) -> str | None:
        """The glue model mends one seam (SEAM_CHARS a side); None without one, or if its seam is unsafe."""
        if self._glue is None or not self._catalog.is_available(self._glue):
            return None
        a, b = before[-SEAM_CHARS:], more[:SEAM_CHARS]
        asked = [ChatMessage("user", GLUE_PROMPT.format(a=a, b=b))]
        attempt = await self._try_model(self._catalog.get(self._glue), asked, (), run)
        if attempt.outcome is not _Outcome.OK:
            return None
        seam = _strip_wrapper(attempt.text)
        check = min(SEAM_CHECK_CHARS, len(a) // 2, len(b) // 2)
        # Both ends kept exactly and no more text than came in: the glue model touched the seam only.
        if (
            check < 1
            or not seam.startswith(a[:check])
            or not seam.endswith(b[-check:])
            or len(seam) > len(a) + len(b)
        ):
            logger.warning(
                "glue model returned an unusable seam (%d chars), parts joined as they are", len(seam)
            )
            return None
        return before[: len(before) - len(a)] + seam + more[len(b) :]

    async def _try_model(
        self, spec: ModelSpec, messages: Sequence[ChatMessage], tools: Sequence[ToolSpec], run: _Run
    ) -> _Attempt:
        """Retry one model. Returns a terminal attempt, or the last failure for this model.

        Every attempt first fits the request into the model's window: the earliest dialog is left out, and if
        that is not enough the biggest text is cut in the middle (see `fit_window`).
        """
        max_tokens = spec.max_tokens
        limit = self._prompt_limit(spec, max_tokens)
        backoffs = 0
        attempt = _Attempt(_Outcome.UNAVAILABLE)
        for attempt_no in range(1, self._catalog.max_attempts + 1):
            run.attempts += 1
            try:
                fitted = await asyncio.to_thread(fit_window, messages, limit, spec.tokenizer)
            except ProviderTooLong as exc:
                return _Attempt(_Outcome.TOO_LONG, reason=str(exc))
            request = CompletionRequest(spec.id, fitted, max_tokens, spec.timeout_s, tools)
            size = await asyncio.to_thread(prompt_tokens, fitted, spec.tokenizer)
            run.input_tokens += size
            attempt = await self._attempt(request)
            run.output_tokens += count_tokens(attempt.text, spec.tokenizer) + sum(
                count_tokens(call.arguments, spec.tokenizer) for call in attempt.tool_calls
            )
            if attempt.outcome in _TERMINAL:
                return attempt
            if (
                attempt.outcome is _Outcome.BAD_RESPONSE
                and ": HTTP 400:" in attempt.reason
                and size > BIG_PROMPT_SHARE * spec.context_tokens
            ):
                attempt = replace(attempt, outcome=_Outcome.TOO_LONG)
            logger.info(
                "llm attempt failed: model=%s attempt=%d outcome=%s max_tokens=%d prompt~%d reason=%s",
                spec.id,
                attempt_no,
                attempt.outcome.value,
                max_tokens,
                size,
                attempt.reason[:300],
            )
            if attempt.outcome is _Outcome.ACCESS_DENIED:
                self._catalog.mark_unavailable(spec.id)
            if attempt.outcome in _ABANDON_MODEL or attempt_no == self._catalog.max_attempts:
                return attempt
            if attempt.outcome is _Outcome.TOO_LONG:
                limit = int(min(limit, size) * (1 - TOO_LONG_SHRINK))  # our estimate fell short of its count
            elif attempt.outcome is _Outcome.EMPTY:
                max_tokens = min(
                    max_tokens * 2,
                    self._catalog.max_tokens_cap,
                    max(spec.max_tokens, int(spec.context_tokens * (1 - WINDOW_SLACK)) - size),
                )
                limit = min(limit, self._prompt_limit(spec, max_tokens))
            elif attempt.outcome in _NEEDS_BACKOFF:
                await self._sleep(self._backoff_s[min(backoffs, len(self._backoff_s) - 1)])
                backoffs += 1
        return attempt

    @staticmethod
    def _prompt_limit(spec: ModelSpec, max_tokens: int) -> int:
        """Tokens the prompt may take: the window less the answer's room and WINDOW_SLACK."""
        return max(1, int(spec.context_tokens * (1 - WINDOW_SLACK)) - max_tokens)

    async def _attempt(self, request: CompletionRequest) -> _Attempt:
        """One provider call, mapped to an outcome. Only `BaseException`s like cancellation escape."""
        try:
            completion = await self._call(request)
        except ProviderTimeout as exc:
            return _Attempt(_Outcome.TIMEOUT, reason=str(exc))
        except ModelAccessDenied as exc:
            return _Attempt(_Outcome.ACCESS_DENIED, reason=str(exc))
        except ProviderAuthError:
            logger.error("llm auth failed: model=%s; check the API key", request.model)
            return _Attempt(_Outcome.AUTH)
        except ProviderUnavailable as exc:
            return _Attempt(_Outcome.UNAVAILABLE, reason=str(exc))
        except ProviderTooLong as exc:
            return _Attempt(_Outcome.TOO_LONG, reason=str(exc))
        except ProviderBadResponse as exc:
            return _Attempt(_Outcome.BAD_RESPONSE, reason=str(exc))
        except Exception:
            logger.exception("llm provider raised an unexpected error: model=%s", request.model)
            return _Attempt(_Outcome.BAD_RESPONSE)
        served_by = completion.model if completion.reserve else None
        if completion.tool_calls:  # a tool request carries no text: nothing to validate
            declared = {tool["function"]["name"] for tool in request.tools}
            if not any(call.name in declared for call in completion.tool_calls):
                # gpt-6-astra's file blocks come back parsed as calls to tools like '<<<FILE</arg_value>',
                # and the page inside is lost: retrying beats spending a tool round on an error reply.
                return _Attempt(_Outcome.CUT)
            return _Attempt(_Outcome.OK, completion.text, completion.tool_calls, served_by)
        outcome = _VERDICT_OUTCOMES[classify(completion.text)]
        text = completion.text if outcome is _Outcome.OK else ""
        return _Attempt(outcome, text, served_by=served_by, finish=completion.finish_reason)

    async def _call(self, request: CompletionRequest) -> Completion:
        """Call the provider holding a concurrency slot only for the duration of the call."""
        self._waiting += 1
        try:
            await self._semaphore.acquire()
        finally:
            self._waiting -= 1
        try:
            return await self._provider.complete(request)
        finally:
            self._semaphore.release()

    def _terminal_result(
        self, started: float, attempt: _Attempt, model_used: str, attempts: int, fallback_used: bool
    ) -> ChatResult:
        if attempt.outcome is _Outcome.OK:
            return self._result(
                started,
                ChatStatus.OK,
                model_used,
                attempts,
                fallback_used,
                text=attempt.text,
                tool_calls=attempt.tool_calls,
                served_by=attempt.served_by,
            )
        if attempt.outcome is _Outcome.STUB:
            return self._result(
                started, ChatStatus.STUB, model_used, attempts, fallback_used, served_by=attempt.served_by
            )
        return self._result(
            started, ChatStatus.FAILED, model_used, attempts, fallback_used, error=attempt.outcome.value
        )

    def _result(
        self,
        started: float,
        status: ChatStatus,
        model_used: str,
        attempts: int,
        fallback_used: bool,
        *,
        text: str = "",
        error: str | None = None,
        tool_calls: tuple[ToolCall, ...] = (),
        served_by: str | None = None,
    ) -> ChatResult:
        latency_ms = max(0, round((self._clock() - started) * 1000))
        return ChatResult(
            status, text, model_used, attempts, latency_ms, fallback_used, error, tool_calls, served_by
        )

    @staticmethod
    def _log_result(result: ChatResult) -> None:
        logger.info(
            "llm chat finished: status=%s model=%s served_by=%s attempts=%d latency_ms=%d "
            "fallback=%s error=%s",
            result.status.value,
            result.model_used,
            result.served_by,
            result.attempts,
            result.latency_ms,
            result.fallback_used,
            result.error,
        )
