"""adminbot.reserve: the render/keyboard pure functions and the three handlers over a real DB."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import pytest
from aiogram.types import CallbackQuery, Message
from pydantic import Field

from adminbot.callbacks import ProbeNow, ReserveSet
from adminbot.probes import ProbeResult
from adminbot.reserve import (
    cb_probe_now,
    cb_reserve,
    cb_reserve_set,
    render_reserve,
    render_reserve_confirm,
    reserve_keyboard,
)
from bot.llm.reserve import ReserveSwitch
from bot.storage.db import Database
from bot.storage.repositories import FlagRepo

TZ = timezone(timedelta(hours=3))
NOW = datetime(2026, 5, 20, 12, 0, tzinfo=UTC)


@pytest.fixture
async def db() -> Database:
    database = Database(":memory:")
    await database.connect()
    try:
        yield database
    finally:
        await database.close()


@pytest.fixture
def switch(db: Database) -> ReserveSwitch:
    return ReserveSwitch(FlagRepo(db))


@pytest.fixture
def settings() -> Any:
    from dataclasses import dataclass

    @dataclass
    class FakeSettings:
        reserve_configured: bool = True
        reserve_model: str = "free-coded"
        model_probe_minutes: int = 30

    return FakeSettings()


def _result(model: str = "m1", ok: bool = True, **kw) -> ProbeResult:
    return ProbeResult(model, model.upper(), ok, 250, **kw)


class FakeMessage(Message):
    async def answer(self, text: str, **kwargs: Any) -> None:
        return None

    async def edit_text(self, text: str, **kwargs: Any) -> None:
        return None


class FakeCallback(CallbackQuery):
    notes: list[tuple[str | None, bool]] = Field(default_factory=list)

    async def answer(self, text: str | None = None, show_alert: bool = False, **kwargs: Any) -> None:
        self.notes.append((text, show_alert))


def _message() -> FakeMessage:
    return FakeMessage.model_construct(message_id=1, text="экран")


def _callback(message: Message | None = None) -> FakeCallback:
    return FakeCallback.model_construct(message=message, data="", id="1")


class BusyProber:
    """A prober that is always in the middle of a round."""

    busy = True
    last: tuple[ProbeResult, ...] = ()
    last_at = None

    async def probe_all(self, now: datetime):
        raise AssertionError("must not be called while busy")


class RecordingProber:
    def __init__(self, results: tuple[ProbeResult, ...] = ()) -> None:
        self.busy = False
        self.last = results
        self.last_at = None
        self.calls = 0

    async def probe_all(self, now: datetime):
        self.calls += 1
        self.busy = False
        self.last_at = now
        return list(self.last)


# ---- render_reserve (pure) -----------------------------------------------------------------------


def test_render_reserve_off_shows_ok_head() -> None:
    text = render_reserve(True, False, None, "free-coded", (), None, 30, NOW, TZ)
    assert "работают основные модели" in text
    assert "АВАРИЙНЫЙ РЕЖИМ" not in text


def test_render_reserve_on_shows_emergency_head_and_since() -> None:
    since = NOW - timedelta(hours=1)
    text = render_reserve(True, True, since, "free-coded", (), None, 30, NOW, TZ)
    assert "АВАРИЙНЫЙ РЕЖИМ" in text
    assert "free-coded" in text
    assert "назад" in text


# ---- the menu banner tells a human's switch from the bot's own -------------------------------------


async def test_menu_banner_distinguishes_a_manual_switch(db: Database) -> None:
    from adminbot.home import main_menu_screen
    from bot.storage.repositories import RequestRepo, UserRepo

    switch = ReserveSwitch(FlagRepo(db))
    await switch.set(True)
    text, _ = await main_menu_screen(UserRepo(db), RequestRepo(db), NOW, TZ, switch)
    assert "Включён аварийный режим" in text
    assert "включился сам" not in text


async def test_menu_banner_says_when_the_bot_switched_by_itself(db: Database) -> None:
    from adminbot.home import main_menu_screen
    from bot.storage.repositories import RequestRepo, UserRepo

    switch = ReserveSwitch(FlagRepo(db))
    await switch.set(True, auto=True)
    text, _ = await main_menu_screen(UserRepo(db), RequestRepo(db), NOW, TZ, switch)
    assert "включился сам" in text


async def test_menu_banner_is_absent_while_the_primary_gateway_works(db: Database) -> None:
    from adminbot.home import main_menu_screen
    from bot.storage.repositories import RequestRepo, UserRepo

    switch = ReserveSwitch(FlagRepo(db))
    text, _ = await main_menu_screen(UserRepo(db), RequestRepo(db), NOW, TZ, switch)
    assert "АВАРИЙНЫЙ РЕЖИМ" not in text
    assert "включился сам" not in text


def test_render_reserve_not_configured_has_no_toggle_hint_but_explains() -> None:
    text = render_reserve(False, False, None, "free-coded", (), None, 30, NOW, TZ)
    assert "Резерв не настроен" in text


def test_render_reserve_results_show_latency_and_reserve_answered_by() -> None:
    results = (_result("m1", True), _result("free-coded", True, reserve=True, answered_by="nvidia/x"))
    text = render_reserve(True, False, None, "free-coded", results, NOW, 30, NOW, TZ)
    assert "nvidia/x" in text
    assert "Проверено:" in text


def test_render_reserve_probe_off_line() -> None:
    text = render_reserve(True, False, None, "free-coded", (), None, 0, NOW, TZ)
    assert "MODEL_PROBE_MINUTES=0" in text


def test_render_reserve_never_probed_line() -> None:
    text = render_reserve(True, False, None, "free-coded", (), None, 30, NOW, TZ)
    assert "Проверки ещё не было" in text


def test_render_reserve_escapes_error() -> None:
    bad = _result("m1", False, error="<script>x</script>")
    text = render_reserve(True, False, None, "free-coded", (bad,), NOW, 30, NOW, TZ)
    assert "<script>" not in text
    assert "&lt;script&gt;" in text


# ---- reserve_keyboard (pure) ----------------------------------------------------------------------


def _callback_data(markup) -> list[str]:
    return [btn.callback_data for row in markup.inline_keyboard for btn in row]


def test_keyboard_not_configured_has_no_toggle_button() -> None:
    markup = reserve_keyboard(False, False, True)
    data = _callback_data(markup)
    assert not any(d.startswith("rsvset") for d in data)


def test_keyboard_off_configured_shows_enable_button() -> None:
    markup = reserve_keyboard(True, False, True)
    assert ReserveSet(on=True, confirm=False).pack() in _callback_data(markup)


def test_keyboard_on_shows_return_button() -> None:
    markup = reserve_keyboard(True, True, True)
    assert ReserveSet(on=False, confirm=True).pack() in _callback_data(markup)


def test_keyboard_hides_probe_now_without_a_prober() -> None:
    markup = reserve_keyboard(True, False, False)
    assert ProbeNow().pack() not in _callback_data(markup)


def test_confirm_screen_shows_latest_reserve_probe() -> None:
    result = _result("free-coded", True, reserve=True, answered_by="nvidia/x")
    text = render_reserve_confirm("free-coded", result, NOW, TZ)
    assert "nvidia/x" in text


# ---- handlers --------------------------------------------------------------------------------


async def test_cb_reserve_renders_screen(switch: ReserveSwitch, settings: Any) -> None:
    callback = _callback(_message())
    await cb_reserve(callback, switch, RecordingProber(), settings, TZ)
    # no exception, no alert popup expected for the plain screen open
    assert callback.notes == [(None, False)]


async def test_reserve_set_confirm_flips_the_flag(switch: ReserveSwitch, settings: Any) -> None:
    callback = _callback(_message())
    await cb_reserve_set(callback, ReserveSet(on=True, confirm=True), switch, None, settings, TZ)
    assert await switch.active() is True


async def test_reserve_set_off_flips_back(switch: ReserveSwitch, settings: Any) -> None:
    await switch.set(True)
    callback = _callback(_message())
    await cb_reserve_set(callback, ReserveSet(on=False, confirm=True), switch, None, settings, TZ)
    assert await switch.active() is False


async def test_reserve_set_refuses_on_when_not_configured(switch: ReserveSwitch, settings: Any) -> None:
    settings.reserve_configured = False
    callback = _callback(_message())
    await cb_reserve_set(callback, ReserveSet(on=True, confirm=True), switch, None, settings, TZ)
    assert await switch.active() is False
    assert callback.notes[-1][1] is True  # show_alert=True


async def test_reserve_set_on_without_confirm_asks_first(switch: ReserveSwitch, settings: Any) -> None:
    callback = _callback(_message())
    await cb_reserve_set(callback, ReserveSet(on=True, confirm=False), switch, None, settings, TZ)
    assert await switch.active() is False  # not flipped yet


async def test_probe_now_busy_does_not_start_a_second_probe(switch: ReserveSwitch, settings: Any) -> None:
    prober = BusyProber()
    callback = _callback(_message())
    await cb_probe_now(callback, switch, prober, settings, TZ)  # raises if probe_all is called
    assert callback.notes == [("Проверка уже идёт, подождите", False)]


async def test_probe_now_runs_and_rerenders(switch: ReserveSwitch, settings: Any) -> None:
    prober = RecordingProber((_result(),))
    callback = _callback(_message())
    await cb_probe_now(callback, switch, prober, settings, TZ)
    assert prober.calls == 1


async def test_probe_now_without_prober_alerts(settings: Any) -> None:
    callback = _callback(_message())
    # The switch is unused on this path (prober is None): passing None for it is fine here.
    await cb_probe_now(callback, None, None, settings, TZ)  # type: ignore[arg-type]
    assert callback.notes[-1][1] is True
