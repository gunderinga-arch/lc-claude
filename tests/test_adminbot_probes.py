"""adminbot.probes: the model prober's retry/timeout logic, the pure diff/render helpers, and the
background round that turns a change into exactly one deduplicated alert.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

from adminbot.callbacks import Reserve, ReserveSet
from adminbot.probes import (
    RETRY_DELAY_S,
    ModelProber,
    ProbeResult,
    autoswitch_target,
    diff,
    primary_all_ok,
    probe_round,
    render_autoswitch,
    render_change,
)
from bot.catalog import Catalog, ModelSpec
from bot.domain import Mode
from bot.llm.base import Completion, CompletionRequest, ProviderAuthError, ProviderTimeout
from bot.llm.reserve import DOWN_FLAG, DOWN_STALE_AFTER, load_down

NOW = datetime(2026, 5, 20, 12, 0, tzinfo=UTC)


def make_catalog(**overrides: bool) -> Catalog:
    """Two models, "m1" enabled and "m2" disabled unless overridden."""
    enabled = {"m1": True, "m2": False, **overrides}
    specs = {
        name: ModelSpec(name, name.upper(), 1000, 30.0, None, enabled=on, slow=False)
        for name, on in enabled.items()
    }
    return Catalog(specs, {mode: "m1" for mode in Mode}, max_tokens_cap=2000, max_attempts=2)


def ok(text: str = "работаю", model: str = "m1") -> Completion:
    return Completion(text, "stop", model, None, None)


class FakeProvider:
    """Plays pre-set responses (a Completion or an exception instance) in order."""

    def __init__(self, responses: list[Completion | Exception]) -> None:
        self.responses = list(responses)
        self.calls = 0

    async def complete(self, request: CompletionRequest) -> Completion:
        self.calls += 1
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    async def list_models(self) -> list[str]:
        return []

    async def aclose(self) -> None:
        pass


def make_prober(provider: FakeProvider, catalog: Catalog, sleeps: list[float], **kwargs) -> ModelProber:
    async def sleep(seconds: float) -> None:
        sleeps.append(seconds)

    return ModelProber(provider, catalog, sleep=sleep, clock=lambda: 0.0, **kwargs)


# ---- ModelProber: retries and classification -------------------------------------------------


async def test_probe_ok_first_try() -> None:
    provider = FakeProvider([ok()])
    sleeps: list[float] = []
    prober = make_prober(provider, make_catalog(), sleeps)

    results = await prober.probe_all(NOW)

    assert len(results) == 1
    assert results[0].ok is True
    assert provider.calls == 1
    assert sleeps == []


async def test_probe_rejects_a_salvaged_partial_answer() -> None:
    partial = Completion("Раб", "length", "m1", None, None)
    provider = FakeProvider([partial, partial])
    prober = make_prober(provider, make_catalog(), [])
    results = await prober.probe_all(NOW)
    assert not results[0].ok
    assert results[0].error == "ответ оборван"


async def test_probe_timeout_then_ok() -> None:
    provider = FakeProvider([ProviderTimeout("slow"), ok()])
    sleeps: list[float] = []
    prober = make_prober(provider, make_catalog(), sleeps)

    results = await prober.probe_all(NOW)

    assert results[0].ok is True
    assert provider.calls == 2
    assert sleeps == [RETRY_DELAY_S]


async def test_probe_two_timeouts_is_down() -> None:
    provider = FakeProvider([ProviderTimeout("slow"), ProviderTimeout("slow")])
    sleeps: list[float] = []
    prober = make_prober(provider, make_catalog(), sleeps)

    results = await prober.probe_all(NOW)

    assert results[0].ok is False
    assert "нет ответа" in (results[0].error or "")
    assert provider.calls == 2


async def test_probe_auth_error_is_fatal_no_retry() -> None:
    provider = FakeProvider([ProviderAuthError("bad key")])
    sleeps: list[float] = []
    prober = make_prober(provider, make_catalog(), sleeps)

    results = await prober.probe_all(NOW)

    assert results[0].ok is False
    assert "401" in (results[0].error or "")
    assert provider.calls == 1
    assert sleeps == []


async def test_probe_empty_text_twice_is_down() -> None:
    provider = FakeProvider([ok(text=""), ok(text="")])
    sleeps: list[float] = []
    prober = make_prober(provider, make_catalog(), sleeps)

    results = await prober.probe_all(NOW)

    assert results[0].ok is False
    assert "пустой ответ" in (results[0].error or "")


async def test_probe_reserve_target_reports_answered_by() -> None:
    primary = FakeProvider([ok()])
    reserve = FakeProvider([ok(text="работаю", model="nvidia/x")])
    sleeps: list[float] = []
    prober = make_prober(primary, make_catalog(), sleeps, reserve=reserve, reserve_model="free-coded")

    results = await prober.probe_all(NOW)

    reserve_result = next(r for r in results if r.reserve)
    assert reserve_result.ok is True
    assert reserve_result.answered_by == "nvidia/x"


async def test_probe_skips_reserve_models_marked_probe_false() -> None:
    # smart-reserve spends the owner's weekly Antigravity quota: 48 pings a day would eat a fifth of it.
    specs = {
        "m1": ModelSpec("m1", "M1", 1000, 30.0, None, enabled=True, slow=False),
        "smart": ModelSpec(
            "smart", "S", 1000, 30.0, None, enabled=True, slow=False, reserve=True, probe=False
        ),
        "free": ModelSpec("free", "F", 1000, 30.0, None, enabled=True, slow=False, reserve=True),
    }
    catalog = Catalog(specs, {mode: "m1" for mode in Mode}, max_tokens_cap=2000, max_attempts=2)
    reserve = FakeProvider([ok(model="free")])
    prober = make_prober(FakeProvider([ok()]), catalog, [], reserve=reserve)

    results = await prober.probe_all(NOW)

    assert [r.model for r in results if r.reserve] == ["free"]
    assert reserve.calls == 1


async def test_probe_reserve_runs_within_its_own_concurrency_limit() -> None:
    """The reserve pool is one free-tier account: probing it all at once is what caused the 504s.

    The primary side still goes out in one batch, so the round does not get slower overall.
    """
    live = {"primary": 0, "reserve": 0}
    peak = {"primary": 0, "reserve": 0}

    class TrackingProvider:
        def __init__(self, name: str) -> None:
            self.name = name

        async def complete(self, request: CompletionRequest) -> Completion:
            live[self.name] += 1
            peak[self.name] = max(peak[self.name], live[self.name])
            await asyncio.sleep(0)  # yield so the other tasks can pile up
            live[self.name] -= 1
            return ok(text="работаю", model=request.model)

        async def list_models(self) -> list[str]:
            return []

        async def aclose(self) -> None:
            pass

    # A real reserve pool (not the single legacy target): without the semaphore all six go at once.
    specs = {
        **{
            f"p{i}": ModelSpec(f"p{i}", f"P{i}", 1000, 30.0, None, enabled=True, slow=False) for i in range(5)
        },
        **{
            f"r{i}": ModelSpec(f"r{i}", f"R{i}", 1000, 30.0, None, enabled=True, slow=False, reserve=True)
            for i in range(6)
        },
    }
    catalog = Catalog(specs, {mode: "p0" for mode in Mode}, max_tokens_cap=2000, max_attempts=2)

    async def no_sleep(_seconds: float) -> None:
        return None

    prober = ModelProber(
        TrackingProvider("primary"),
        catalog,
        TrackingProvider("reserve"),
        "free-coded",
        reserve_concurrency=2,
        sleep=no_sleep,
        clock=lambda: 0.0,
    )

    results = await prober.probe_all(NOW)

    assert len(results) == 11  # 5 primary + 6 reserve
    assert peak["reserve"] <= 2, f"reserve sent {peak['reserve']} at once"
    assert peak["primary"] >= 4, "the primary side should not be throttled"


async def test_probe_skips_disabled_models() -> None:
    provider = FakeProvider([ok()])
    sleeps: list[float] = []
    prober = make_prober(provider, make_catalog(m1=True, m2=False), sleeps)

    results = await prober.probe_all(NOW)

    assert [r.model for r in results] == ["m1"]


# ---- pure diff / render -------------------------------------------------------------------------


def _result(model: str, ok_: bool, **kw) -> ProbeResult:
    return ProbeResult(model, model.upper(), ok_, 100, **kw)


def test_diff_first_round_reports_only_failures() -> None:
    results = [_result("m1", True), _result("m2", False, error="down")]
    change = diff(None, results)
    assert change.down == (results[1],)
    assert change.up == ()


def test_diff_later_round_reports_transitions_only() -> None:
    previous = {"m1": True, "m2": False}
    results = [_result("m1", False, error="down now"), _result("m2", True)]
    change = diff(previous, results)
    assert change.down == (results[0],)
    assert change.up == (results[1],)


def test_diff_no_change_is_empty() -> None:
    previous = {"m1": True}
    results = [_result("m1", True)]
    change = diff(previous, results)
    assert not change


def test_render_change_shows_return_hint_when_reserve_on_and_primary_ok() -> None:
    change = diff(None, [_result("m1", False, error="down")])
    text = render_change(change, reserve_on=True, all_primary_ok=True)
    assert "можно вернуть их кнопкой ниже" in text


def test_primary_all_ok_ignores_reserve() -> None:
    results = [_result("m1", True), _result("m2", True, reserve=True, answered_by="x")]
    assert primary_all_ok(results) is True


# ---- probe_round: one deduplicated alert per change ----------------------------------------------


class FakeSwitch:
    def __init__(self, on: bool = False, mode: str | None = None) -> None:
        self._on = on
        # `mode` defaults to whatever `on` implies, so older tests keep reading naturally.
        self._mode = mode if mode is not None else ("on" if on else "off")

    async def active(self) -> bool:
        return self._on

    async def mode(self) -> tuple[str, None]:
        return self._mode, None

    async def set(self, on: bool, *, auto: bool = False) -> None:
        self._on = on
        self._mode = "auto" if auto and on else ("on" if on else "off")


class FakeAlerts:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, object]] = []

    async def fire(self, key: str, text: str, reply_markup=None):
        self.calls.append((key, text, reply_markup))
        return object()


async def test_probe_round_sends_one_alert_on_change() -> None:
    """One change, one alert, with the manual-switch buttons on it.

    The catalog has three enabled models and only the first one fails: with half of them down the
    round would also auto-enable the reserve and send its own notice (covered separately), and
    this test is about the change alert alone.
    """
    provider = FakeProvider([ProviderTimeout("x"), ProviderTimeout("x"), ok(), ok()])
    sleeps: list[float] = []
    prober = make_prober(provider, make_catalog(m2=True, m3=True), sleeps)
    alerts = FakeAlerts()
    switch = FakeSwitch(on=False)

    previous = await probe_round(prober, alerts, switch, True, None, NOW)

    assert len(alerts.calls) == 1
    assert previous["m1"] is False and previous["m2"] is True
    key, _text, markup = alerts.calls[0]
    assert key == f"models:{NOW.astimezone(UTC):%Y-%m-%dT%H:%M}"
    callbacks = [btn.callback_data for row in markup.inline_keyboard for btn in row]
    assert ReserveSet(on=True, confirm=False).pack() in callbacks
    assert Reserve().pack() in callbacks


async def test_probe_round_sends_nothing_when_unchanged() -> None:
    provider = FakeProvider([ok(), ok()])
    sleeps: list[float] = []
    prober = make_prober(provider, make_catalog(), sleeps)
    alerts = FakeAlerts()
    switch = FakeSwitch(on=False)

    previous = await probe_round(prober, alerts, switch, True, None, NOW)
    assert alerts.calls == []  # first round, nothing down: no alert

    provider.responses = [ok()]
    await probe_round(prober, alerts, switch, True, previous, NOW)
    assert alerts.calls == []


async def test_probe_round_reserve_on_shows_return_button() -> None:
    provider = FakeProvider([ProviderTimeout("x"), ProviderTimeout("x")])
    sleeps: list[float] = []
    prober = make_prober(provider, make_catalog(), sleeps)
    alerts = FakeAlerts()
    switch = FakeSwitch(on=True)

    await probe_round(prober, alerts, switch, True, None, NOW)

    _, _, markup = alerts.calls[0]
    callbacks = [btn.callback_data for row in markup.inline_keyboard for btn in row]
    assert ReserveSet(on=False, confirm=True).pack() in callbacks


# ---- the automatic switch -----------------------------------------------------------------------


def test_autoswitch_turns_on_when_every_primary_model_is_down() -> None:
    results = [_result("m1", False, error="down"), _result("m2", False, error="down")]
    assert autoswitch_target(results, "off", True) is True


def test_autoswitch_stays_put_while_most_models_still_answer() -> None:
    """Fewer than half down is not an outage: the dead ones are hidden, the rest serve users."""
    results = [_result("m1", False, error="down"), _result("m2", True), _result("m3", True)]
    assert autoswitch_target(results, "off", True) is None
    assert autoswitch_target(results, "auto", True) is False


def test_autoswitch_turns_on_when_half_the_primary_models_are_down() -> None:
    results = [_result(m, m in ("m1", "m2"), error=None) for m in ("m1", "m2", "m3", "m4")]
    assert autoswitch_target(results, "off", True) is True
    assert autoswitch_target(results, "auto", True) is None  # still sick: stay in reserve


async def test_probe_round_publishes_the_dead_models() -> None:
    provider = FakeProvider([ProviderTimeout("x"), ProviderTimeout("x"), ok()])
    prober = make_prober(provider, make_catalog(m2=True), [])
    flags = FakeFlags()

    await probe_round(prober, FakeAlerts(), FakeSwitch(), True, None, NOW, flags)

    assert flags.values == {DOWN_FLAG: "m1"}


async def test_main_bot_hides_the_dead_models_until_they_answer() -> None:
    catalog = make_catalog(m2=True)
    flags = FakeFlags({DOWN_FLAG: "m1"})
    catalog.set_down(await load_down(flags, NOW))
    assert [s.id for s in catalog.selectable()] == ["m2"]
    assert [s.id for s in catalog.fallback_chain("m1")] == ["m1"]  # all hidden: still try it

    flags.at = NOW - DOWN_STALE_AFTER - timedelta(minutes=1)  # the admin bot stopped probing
    catalog.set_down(await load_down(flags, NOW))
    assert [s.id for s in catalog.selectable()] == ["m1", "m2"]


class FakeFlags:
    def __init__(self, values: dict[str, str] | None = None) -> None:
        self.values = dict(values or {})
        self.at = NOW

    async def get(self, key: str) -> tuple[str, datetime] | None:
        return (self.values[key], self.at) if key in self.values else None

    async def set(self, key: str, value: str) -> None:
        self.values[key] = value


def test_autoswitch_never_fights_a_human_choice() -> None:
    """A reserve the admin turned on by hand is left alone, however healthy the primary side is."""
    all_down = [_result("m1", False, error="down")]
    all_up = [_result("m1", True)]
    assert autoswitch_target(all_up, "on", True) is None
    assert autoswitch_target(all_down, "on", True) is None


def test_autoswitch_returns_to_primary_when_it_was_automatic() -> None:
    all_up = [_result("m1", True)]
    all_down = [_result("m1", False, error="down")]
    assert autoswitch_target(all_up, "auto", True) is False
    assert autoswitch_target(all_down, "auto", True) is None  # still down: stay in reserve


def test_autoswitch_does_nothing_without_a_reserve_gateway() -> None:
    results = [_result("m1", False, error="down")]
    assert autoswitch_target(results, "off", False) is None


def test_autoswitch_ignores_an_empty_probe_list() -> None:
    """No probes at all (a broken catalog) is ignorance, not an outage."""
    assert autoswitch_target([], "off", True) is None


def test_render_autoswitch_on_names_the_dead_models() -> None:
    """The admin must see why answers changed, not just that they did."""
    results = [_result("m1", False, error="нет ответа"), _result("m2", True)]
    text = render_autoswitch(True, results)
    assert "включён автоматически" in text
    assert "M1" in text and "нет ответа" in text
    assert "M2" not in text  # a model that still answers is not part of the outage report


def test_render_autoswitch_off_says_the_primary_gateway_is_back() -> None:
    text = render_autoswitch(False, [_result("m1", True)])
    assert "вернулись" in text
    assert "выключен" in text


async def test_probe_round_turns_the_reserve_on_by_itself() -> None:
    """Every model down: the flag becomes "auto" and the admin is told why answers changed."""
    provider = FakeProvider([ProviderTimeout("x"), ProviderTimeout("x")])
    sleeps: list[float] = []
    prober = make_prober(provider, make_catalog(), sleeps)
    alerts = FakeAlerts()
    switch = FakeSwitch(on=False)

    await probe_round(prober, alerts, switch, True, None, NOW)

    assert await switch.active() is True
    assert await switch.mode() == ("auto", None)
    keys = [key for key, _text, _markup in alerts.calls]
    assert any(key.startswith("autoswitch:") for key in keys)


async def test_probe_round_returns_to_primary_after_an_automatic_outage() -> None:
    """The models answer again and the reserve was automatic: the flag goes back off by itself."""
    provider = FakeProvider([ok(), ok()])
    sleeps: list[float] = []
    prober = make_prober(provider, make_catalog(), sleeps)
    alerts = FakeAlerts()
    switch = FakeSwitch(on=True, mode="auto")

    await probe_round(prober, alerts, switch, True, {"m1": False}, NOW)

    assert await switch.active() is False
    keys = [key for key, _text, _markup in alerts.calls]
    assert any(key.startswith("autoswitch:") for key in keys)


async def test_probe_round_leaves_a_manual_reserve_alone() -> None:
    """Mode "on" is the admin's call: a healthy primary gateway does not switch it back."""
    provider = FakeProvider([ok(), ok()])
    sleeps: list[float] = []
    prober = make_prober(provider, make_catalog(), sleeps)
    alerts = FakeAlerts()
    switch = FakeSwitch(on=True, mode="on")

    await probe_round(prober, alerts, switch, True, {"m1": False}, NOW)

    assert await switch.active() is True
    keys = [key for key, _text, _markup in alerts.calls]
    assert not any(key.startswith("autoswitch:") for key in keys)
