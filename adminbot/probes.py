"""Model probe: a tiny test request to every enabled model (and the reserve) on a schedule.

The main bot's heartbeat only says the *process* is alive; a model can be dead while the bot
happily polls Telegram. So the admin bot asks every model one short question itself, directly
through the providers (no fallbacks: a fallback would hide exactly the model we are checking).

One failed request is not an outage: a model that fails is asked once more after a pause, and
only a second failure marks it down. Auth and access errors are not retried, they do not heal.
"""

from __future__ import annotations

import asyncio
import html
import logging
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from aiogram.utils.keyboard import InlineKeyboardBuilder

from adminbot.alerts import AlertService
from adminbot.callbacks import Reserve, ReserveSet
from bot.catalog import Catalog
from bot.llm.base import (
    ChatMessage,
    CompletionRequest,
    LLMProvider,
    ModelAccessDenied,
    ProviderAuthError,
    ProviderBadResponse,
    ProviderTimeout,
    ProviderUnavailable,
    Verdict,
)
from bot.llm.reserve import DOWN_FLAG, ReserveSwitch
from bot.llm.validator import classify
from bot.storage.repositories import FlagRepo

logger = logging.getLogger(__name__)

PROBE_PROMPT = "Проверка связи. Ответь одним словом: работаю."
# Fable thinks for minutes on real questions; a one-word ping is far quicker, and a gateway that
# holds even that for 3 minutes is useless to a user anyway.
PROBE_TIMEOUT_S = 180.0
RESERVE_MAX_TOKENS = 4_000  # free reasoning models spend some on hidden thinking before "работаю"
RETRY_DELAY_S = 20.0
# The reserve gateway is a single free-tier account for every member of the pool. Probing the whole
# pool at once is what made four models report 504s while a lone request to the same model worked.
RESERVE_PROBE_CONCURRENCY = 2

Sleep = Callable[[float], Awaitable[None]]
Clock = Callable[[], float]


@dataclass(frozen=True, slots=True)
class ProbeResult:
    model: str  # catalog id, or the reserve's model name (e.g. "free-coded")
    title: str
    ok: bool
    latency_ms: int  # of the last attempt
    error: str | None = None  # short Russian reason when not ok
    reserve: bool = False
    answered_by: str | None = None  # reserve only: the combo member that actually answered


@dataclass(frozen=True, slots=True)
class _Target:
    provider: LLMProvider
    model: str
    title: str
    max_tokens: int
    reserve: bool


class _Fatal(Exception):
    """A failure that a retry cannot fix (bad key, no access)."""


class _Retryable(Exception):
    """A failure worth one more try (timeout, 5xx, network, garbage)."""


class ModelProber:
    """Probes the catalog's enabled models over `primary` and, if given, `reserve_model` over `reserve`."""

    def __init__(
        self,
        primary: LLMProvider,
        catalog: Catalog,
        reserve: LLMProvider | None = None,
        reserve_model: str = "free-coded",
        *,
        timeout_s: float = PROBE_TIMEOUT_S,
        retry_delay_s: float = RETRY_DELAY_S,
        reserve_concurrency: int = RESERVE_PROBE_CONCURRENCY,
        sleep: Sleep = asyncio.sleep,
        clock: Clock = time.monotonic,
    ) -> None:
        self._targets = [
            _Target(primary, spec.id, spec.title, spec.max_tokens, False)
            for spec in catalog.models()
            if spec.enabled and not spec.reserve and spec.probe  # a pipeline is no gateway model
        ]
        if reserve is not None:
            reserve_pool = catalog.reserve_selectable()
            if reserve_pool:
                self._targets += [
                    _Target(reserve, spec.id, spec.title, spec.max_tokens, True)
                    for spec in reserve_pool
                    if spec.probe
                ]
            else:
                # Catalogs without a reserve_models: section still get one probe of the single
                # reserve_model the caller configured (keeps old deployments working unchanged).
                self._targets.append(_Target(reserve, reserve_model, "🛟 Резерв", RESERVE_MAX_TOKENS, True))
        self._timeout_s = timeout_s
        self._retry_delay_s = retry_delay_s
        self._reserve_concurrency = max(1, reserve_concurrency)
        self._sleep = sleep
        self._clock = clock
        self._lock = asyncio.Lock()  # the scheduled round and a "check now" press must not overlap
        self.last: tuple[ProbeResult, ...] = ()
        self.last_at: datetime | None = None

    @property
    def busy(self) -> bool:
        return self._lock.locked()

    async def probe_all(self, now: datetime) -> list[ProbeResult]:
        """Probe every target; the results are also kept in `last` / `last_at`.

        The primary gateway takes the whole batch at once, the reserve is probed through a
        semaphore: it is usually one free-tier account behind a limiter, and a dozen simultaneous
        pings make the slow members fail on the queue alone - the round would then report an
        outage that only its own burst caused.
        """
        async with self._lock:
            slots = asyncio.Semaphore(self._reserve_concurrency)

            async def run(target: _Target) -> ProbeResult:
                if not target.reserve:
                    return await self._probe(target)
                async with slots:
                    return await self._probe(target)

            results = await asyncio.gather(*(run(target) for target in self._targets))
            self.last, self.last_at = tuple(results), now
            return list(results)

    async def _probe(self, target: _Target) -> ProbeResult:
        error: str | None = None
        latency_ms = 0
        for attempt in (1, 2):
            if attempt == 2:
                await self._sleep(self._retry_delay_s)
            started = self._clock()
            try:
                answered_by = await self._ask(target)
            except _Fatal as exc:
                error = str(exc)
                latency_ms = self._elapsed(started)
                break
            except _Retryable as exc:
                error = str(exc)
                latency_ms = self._elapsed(started)
                continue
            return ProbeResult(
                target.model,
                target.title,
                True,
                self._elapsed(started),
                reserve=target.reserve,
                answered_by=answered_by,
            )
        return ProbeResult(target.model, target.title, False, latency_ms, error, target.reserve)

    async def _ask(self, target: _Target) -> str | None:
        """One request. Returns the answering model's name (for the reserve) or raises a reason."""
        request = CompletionRequest(
            model=target.model,
            messages=[ChatMessage(role="user", content=PROBE_PROMPT)],
            max_tokens=target.max_tokens,
            timeout_s=self._timeout_s,
        )
        try:
            completion = await target.provider.complete(request)
        except ProviderTimeout:
            raise _Retryable(f"нет ответа за {int(self._timeout_s)} с") from None
        except ProviderUnavailable as exc:
            reason = f"HTTP {exc.status}" if exc.status else "нет связи с шлюзом"
            raise _Retryable(reason) from None
        except ProviderAuthError:
            raise _Fatal("неверный API-ключ (401)") from None
        except ModelAccessDenied:
            raise _Fatal("нет доступа к модели (403)") from None
        except ProviderBadResponse:
            raise _Retryable("шлюз ответил не по формату") from None
        except Exception as exc:  # a probe must never kill the loop
            raise _Retryable(f"ошибка {type(exc).__name__}") from None
        verdict = classify(completion.text)
        if verdict is Verdict.EMPTY and not completion.tool_calls:
            raise _Retryable("пустой ответ")
        if verdict is Verdict.CUT or completion.finish_reason == "length":
            raise _Retryable("ответ оборван")
        # STUB is the gateway's canned text: the model is reachable, which is all a probe asks.
        return completion.model if target.reserve else None

    def _elapsed(self, started: float) -> int:
        return max(0, round((self._clock() - started) * 1000))


# ---- change detection and texts (pure) ---------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProbeChange:
    down: tuple[ProbeResult, ...]  # were up (or unknown after a restart), now fail
    up: tuple[ProbeResult, ...]  # were down, now answer again

    def __bool__(self) -> bool:
        return bool(self.down or self.up)


def diff(previous: Mapping[str, bool] | None, results: Sequence[ProbeResult]) -> ProbeChange:
    """What changed since the previous round. `previous` None (first round) reports failures only."""
    before = previous or {}
    down = tuple(r for r in results if not r.ok and before.get(r.model, True))
    up = tuple(r for r in results if r.ok and previous is not None and before.get(r.model) is False)
    return ProbeChange(down, up)


def states(results: Sequence[ProbeResult]) -> dict[str, bool]:
    return {r.model: r.ok for r in results}


def primary_all_ok(results: Sequence[ProbeResult]) -> bool:
    primary = [r for r in results if not r.reserve]
    return bool(primary) and all(r.ok for r in primary)


def primary_half_down(results: Sequence[ProbeResult]) -> bool:
    """At least half of the primary models failed: the gateway is too sick to keep users on it.

    An empty probe list is not an outage: we simply know nothing.
    """
    primary = [r for r in results if not r.reserve]
    return bool(primary) and 2 * sum(not r.ok for r in primary) >= len(primary)


def down_models(results: Sequence[ProbeResult]) -> list[str]:
    """Models the main bot hides from the choice until a round finds them answering again."""
    return [r.model for r in results if not r.ok]


def autoswitch_target(results: Sequence[ProbeResult], mode: str, configured: bool) -> bool | None:
    """What the automatic switch wants the flag to be, or None to leave it alone.

    Only `mode` matters for the "switch it back" half: a reserve the admin turned ON by hand must
    never be turned off behind their back, however healthy the primary gateway looks. `None` (the
    flag was never set) and `"off"` are both "not automatic", so a manual ON is respected.
    """
    if not configured:
        return None
    if mode == "auto":
        # Automatic only ever goes back off; a recovering primary gateway must not fight a human
        # who deliberately keeps the reserve on (that click sets "on", which we never clear).
        # Fewer than half down: the dead ones are hidden from the choice, the rest serve users.
        probed = any(not r.reserve for r in results)
        return False if probed and not primary_half_down(results) else None
    if mode == "on":
        return None  # a human's choice, left alone in both directions
    return True if primary_half_down(results) else None


def _name(result: ProbeResult) -> str:
    return f"<b>{html.escape(result.title)}</b> <code>{html.escape(result.model)}</code>"


def render_change(change: ProbeChange, reserve_on: bool, all_primary_ok: bool) -> str:
    """The alert text for one round that changed something."""
    lines: list[str] = []
    if change.down:
        lines.append("🔴 <b>Модели не отвечают</b>")
        lines += [f"• {_name(r)} — {html.escape(r.error or 'ошибка')}" for r in change.down]
    if change.up:
        if lines:
            lines.append("")
        lines.append("🟢 <b>Модели снова отвечают</b>")
        lines += [f"• {_name(r)} — {r.latency_ms / 1000:.1f} с" for r in change.up]
    lines.append("")
    if reserve_on and all_primary_ok:
        lines.append("<i>Все основные модели в строю: можно вернуть их кнопкой ниже.</i>")
    elif change.down and not reserve_on:
        lines.append("<i>Если основные модели лежат, включите резерв кнопкой ниже.</i>")
    else:
        lines.append("<i>Подробности — в «🛟 Резерв и модели».</i>")
    return "\n".join(lines)


def render_autoswitch(on: bool, results: Sequence[ProbeResult]) -> str:
    """The alert for a switch the bot threw itself, so the admin knows why answers changed."""
    if on:
        dead = [r for r in results if not r.reserve and not r.ok]
        lines = [
            "🆘 <b>Резерв включён автоматически</b>",
            "Не отвечает половина основных моделей или больше:",
            *[f"• {_name(r)} — {html.escape(r.error or 'ошибка')}" for r in dead],
            "",
            "<i>Запросы пользователей идут в резервный шлюз. Когда основные модели вернутся, "
            "резерв выключится сам; вернуть их можно и вручную — кнопкой ниже.</i>",
        ]
        return "\n".join(lines)
    return (
        "✅ <b>Основные модели вернулись</b>\n"
        "<i>Резерв, включённый автоматически, выключен: бот снова работает через BuyTokens. "
        "Модели, которые ещё не отвечают, скрыты из выбора.</i>"
    )


# ---- background loop -----------------------------------------------------------------------------


def _change_markup(change: ProbeChange, reserve_on: bool, configured: bool) -> InlineKeyboardBuilder:
    builder = InlineKeyboardBuilder()
    if configured:
        if reserve_on:
            back = ReserveSet(on=False, confirm=True)
            builder.button(text="✅ Вернуть основные модели", callback_data=back)
        elif change.down:
            builder.button(text="🆘 Включить резерв", callback_data=ReserveSet(on=True, confirm=False))
    builder.button(text="🛟 Резерв и модели", callback_data=Reserve())
    builder.adjust(1)
    return builder


async def probe_round(
    prober: ModelProber,
    alerts: AlertService,
    switch: ReserveSwitch,
    configured: bool,
    previous: dict[str, bool] | None,
    now: datetime,
    flags: FlagRepo | None = None,
) -> dict[str, bool]:
    """One scheduled round: probe everything, alert on any change, return states for the next round.

    With `flags`, the failed models are published for the main bot, which hides them from the choice.
    """
    results = await prober.probe_all(now)
    logger.info(
        "probe round: %s", ", ".join(f"{r.model}={'ok' if r.ok else r.error}" for r in results) or "no models"
    )
    if flags is not None:
        try:
            await flags.set(DOWN_FLAG, ",".join(down_models(results)))
        except Exception:
            logger.warning("down-models flag write failed", exc_info=True)

    # The automatic switch acts on the same results, before the change alert: the admin should hear
    # "reserve came on" rather than a bare list of dead models they would have to act on themselves.
    mode, _ = await switch.mode()
    target = autoswitch_target(results, mode, configured)
    if target is not None:
        await switch.set(target, auto=True)
        logger.warning("резервный режим %s автоматически", "включён" if target else "выключен")
        key = f"autoswitch:{now.astimezone(UTC):%Y-%m-%dT%H:%M}"
        await alerts.fire(key, render_autoswitch(target, results))

    change = diff(previous, results)
    if change:
        reserve_on = await switch.active()
        text = render_change(change, reserve_on, primary_all_ok(results))
        markup = _change_markup(change, reserve_on, configured).as_markup()
        key = f"models:{now.astimezone(UTC):%Y-%m-%dT%H:%M}"
        await alerts.fire(key, text, markup)
    return states(results)


async def run_probes(
    prober: ModelProber,
    alerts: AlertService,
    switch: ReserveSwitch,
    configured: bool,
    interval_s: float,
    first_delay_s: float = 60.0,
    flags: FlagRepo | None = None,
) -> None:
    """Probe every `interval_s` until cancelled; start it with `start_probes`.

    The first round waits `first_delay_s`, same reasoning as the alert loop: a probe at t=0 would
    fire before the admin bot's own dependencies (the reserve provider) are fully warmed up, and a
    round takes up to a few minutes anyway, so a short head start costs nothing. A failing round is
    logged and the previous states are kept, so a one-off exception does not manufacture a false
    "everything came back" on the next round.
    """
    logger.info("probe loop started, one round every %s s", interval_s)
    await asyncio.sleep(max(0.0, first_delay_s))
    previous: dict[str, bool] | None = None
    while True:
        try:
            previous = await probe_round(
                prober, alerts, switch, configured, previous, datetime.now(UTC), flags
            )
        except Exception:
            logger.exception("probe round failed")
        await asyncio.sleep(interval_s)


def start_probes(
    prober: ModelProber,
    alerts: AlertService,
    switch: ReserveSwitch,
    configured: bool,
    interval_s: float,
    first_delay_s: float = 60.0,
    flags: FlagRepo | None = None,
) -> asyncio.Task[None]:
    """Start the loop as a named background task; `task.cancel()` stops it."""
    return asyncio.create_task(
        run_probes(prober, alerts, switch, configured, interval_s, first_delay_s, flags),
        name="adminbot-probes",
    )
