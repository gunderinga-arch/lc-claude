"""adminbot.usage: pure report rendering, the cost lines, the activity chart, the period keyboard and
the DB-backed screens.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta, timezone
from typing import Any

import pytest
from aiogram.types import CallbackQuery, Message
from pydantic import Field

from adminbot.callbacks import Activity, Dashboard, Home, UserCard
from adminbot.common import PERIOD_LABELS, TELEGRAM_LIMIT, Period
from adminbot.usage import (
    ACTIVITY_DAYS,
    MAX_MODEL_ROWS,
    MODEL_NAME_CHARS,
    activity_screen,
    bar_cells,
    catalog_prices,
    cb_activity,
    clip_message,
    cost_lines,
    cost_section,
    dashboard_screen,
    model_label,
    period_keyboard,
    render_activity,
    render_report,
    user_stats_screen,
)
from bot.catalog import Catalog, ModelSpec
from bot.domain import (
    DayUsage,
    Mode,
    ModelUsage,
    RequestRecord,
    RequestReport,
    RequestStatus,
    Role,
    TokenUsage,
    UploadKind,
    UploadStat,
)
from bot.storage.db import Database
from bot.storage.repositories import RequestRepo, UploadRepo, UserRepo

UTC_TZ = UTC
TZ = timezone(timedelta(hours=3))
NOW = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)


def _report(**overrides: Any) -> RequestReport:
    base: dict[str, Any] = dict(
        requests=0,
        ok=0,
        stub=0,
        failed=0,
        avg_latency_ms=0,
        total_tokens=0,
        usage=TokenUsage(),
        web_calls=0,
        by_model=(),
        last_at=None,
    )
    base.update(overrides)
    return RequestReport(**base)


@dataclass(frozen=True)
class FakeSettings:
    """Only what the cost lines read off Settings."""

    token_price: float = 0.01
    price_currency: str = "$"


def _catalog(prices: Mapping[str, float] | None = None) -> Catalog:
    """A real Catalog over ModelSpecs: enough for catalog_prices, which only reads spec.price."""
    specs = {
        model: ModelSpec(
            id=model,
            title=model,
            max_tokens=1000,
            timeout_s=10.0,
            fallback=None,
            enabled=True,
            slow=False,
            price=price,
        )
        for model, price in (prices or {}).items()
    }
    if not specs:  # a Catalog needs every mode mapped, so an empty price map gets one free stub
        specs["free"] = ModelSpec(
            id="free",
            title="free",
            max_tokens=1000,
            timeout_s=10.0,
            fallback=None,
            enabled=True,
            slow=False,
            price=None,
        )
    first = next(iter(specs))
    return Catalog(specs, {mode: first for mode in Mode}, max_tokens_cap=10_000, max_attempts=1)


def test_render_report_empty_report_and_no_uploads() -> None:
    text = render_report("Test", Period.WEEK, _report(), {}, NOW)
    assert "Запросов за период нет." in text
    assert PERIOD_LABELS[Period.WEEK] in text


def test_render_report_full_breakdown() -> None:
    usage = TokenUsage(input=1000, output=500, images=200, service=100, context=400, tools=0)
    report = _report(
        requests=10,
        ok=8,
        stub=1,
        failed=1,
        avg_latency_ms=2500,
        total_tokens=usage.total + 300,  # 300 tokens from before the split existed
        usage=usage,
        web_calls=3,
        by_model=(ModelUsage(model="gpt<x>", requests=5, tokens=800),),
        last_at=NOW - timedelta(minutes=5),
    )
    uploads = {
        UploadKind.PDF: UploadStat(count=2, bytes=2048),
        UploadKind.GEN_IMAGE: UploadStat(count=1, bytes=100_000),
    }
    text = render_report("Юзер", Period.TODAY, report, uploads, NOW)

    assert "Запросы: 10 · ✅ 8 · 🙊 1 · ❌ 1" in text
    assert "2.5 с" in text
    assert "история диалога — 400" in text  # context nonzero -> shown
    assert "результаты инструментов" not in text  # tools == 0 -> hidden
    assert "без разбивки (старые записи) — 300" in text
    assert "Веб-поиск: 3 вызовов" in text
    assert "gpt&lt;x&gt;" in text  # model name escaped
    assert "<b>Загрузки пользователя</b>" in text
    assert "<b>Создано ботом</b>" in text
    assert "PDF" in text
    assert "🎨 картинки" in text


def test_render_report_hides_zero_sections() -> None:
    usage = TokenUsage(input=100, output=50, images=0, service=0, context=0, tools=0)
    report = _report(requests=1, ok=1, total_tokens=usage.total, usage=usage, last_at=NOW)
    text = render_report("T", Period.ALL, report, {}, NOW)
    assert "◦" not in text  # both context and tools are 0
    assert "без разбивки" not in text  # unsplit_tokens == 0
    assert "Веб-поиск" not in text
    assert "Модели" not in text
    assert "Загрузки" not in text


def test_render_report_many_models_stays_within_limit() -> None:
    by_model = tuple(ModelUsage(model=f"model-{i}", requests=1, tokens=100) for i in range(100))
    report = _report(requests=100, ok=100, total_tokens=10_000, by_model=by_model, last_at=NOW)
    text = render_report("Все пользователи", Period.ALL, report, {}, NOW)
    assert len(text) <= TELEGRAM_LIMIT
    assert text.count("model-") == 15  # capped to MAX_MODEL_ROWS


def test_period_keyboard_marks_current_and_back_is_last() -> None:
    markup = period_keyboard(lambda p: Dashboard(period=p), Period.WEEK, "« Назад", Home())
    rows = markup.inline_keyboard
    period_row = rows[0]
    assert len(period_row) == 4
    assert sum(btn.text.startswith("✅") for btn in period_row) == 1
    assert any(btn.text == "✅ 7 дней" for btn in period_row)
    assert rows[-1][0].text == "« Назад"
    assert rows[-1][0].callback_data == Home().pack()


def test_period_keyboard_extra_button_sits_on_its_own_row() -> None:
    markup = period_keyboard(
        lambda p: Dashboard(period=p),
        Period.TODAY,
        "« В меню",
        Home(),
        extra=("📈 Активность по дням", Activity(tg_id=None)),
    )
    rows = markup.inline_keyboard
    assert [len(row) for row in rows] == [4, 1, 1]
    assert rows[1][0].text == "📈 Активность по дням"
    assert rows[1][0].callback_data == Activity(tg_id=None).pack()
    assert rows[2][0].callback_data == Home().pack()


# ---- money (pure) ----------------------------------------------------------------------------------


def test_model_label_escapes_before_cutting() -> None:
    """Escaping first and cutting after is the house rule: a cut entity would break the message."""
    assert model_label("a<b>") == "a&lt;b&gt;"
    label = model_label("<" * 200)
    assert label.endswith("…")
    assert not label.endswith("&")  # never a half entity at the cut
    assert "&lt;" * 10 in label


def test_model_label_drops_a_partial_entity_at_the_cut() -> None:
    """A cut inside "&lt;" would make Telegram reject the whole message, so it is dropped."""
    label = model_label("a" * (MODEL_NAME_CHARS - 2) + "<")
    assert label == "a" * (MODEL_NAME_CHARS - 2) + "…"
    assert "&" not in label


def test_clip_message_keeps_short_text_and_cuts_long_text_on_a_line_boundary() -> None:
    assert clip_message("short") == "short"
    long = "\n".join(f"line {i}" for i in range(2000))
    clipped = clip_message(long)
    assert len(clipped) <= TELEGRAM_LIMIT
    assert clipped.endswith("[обрезано]")
    assert "line 1999" not in clipped


def test_catalog_prices_uses_the_injected_catalog_and_skips_models_without_a_price() -> None:
    catalog = _catalog({"cheap": 0.001, "pricey": 5.0})
    assert catalog_prices(catalog) == {"cheap": 0.001, "pricey": 5.0}
    assert catalog_prices(None) == {}  # no catalog wired: the global price is used instead


def test_cost_lines_per_model_override_and_fallback_to_the_global_price() -> None:
    report = _report(
        requests=2,
        ok=2,
        total_tokens=3_000_000,
        by_model=(
            ModelUsage(model="pricey", requests=1, tokens=1_000_000),
            ModelUsage(model="plain", requests=1, tokens=2_000_000),
        ),
        last_at=NOW,
    )
    lines = cost_lines(report, {"pricey": 5.0}, 0.01, "$")

    assert lines[0] == "<b>Стоимость</b> (оценка, цена за 1M токенов)"
    # 1M * 5.0 + 2M * 0.01 = 5.02, and the cheaper row keeps a third digit
    assert "всего за период — $5.02" in lines[1]
    assert any("pricey — $5.00" in line for line in lines)
    assert any("plain — $0.020" in line for line in lines)


def test_cost_lines_tokens_without_a_model_split_use_the_global_price() -> None:
    report = _report(requests=1, ok=1, total_tokens=1_000_000, by_model=(), last_at=NOW)
    lines = cost_lines(report, {}, 0.01, "$")
    assert "всего за период — $0.010" in lines[1]
    assert len(lines) == 2  # no per-model rows at all


def test_cost_lines_empty_period_is_empty() -> None:
    assert cost_lines(_report(), {}, 0.01, "$") == []


def test_cost_lines_currency_after_the_number_for_roubles() -> None:
    report = _report(requests=1, ok=1, total_tokens=1_000_000, last_at=NOW)
    lines = cost_lines(report, {}, 12.5, "руб")
    assert "12.50 руб" in lines[1]


def test_cost_lines_caps_model_rows_and_escapes_names() -> None:
    by_model = tuple(ModelUsage(model=f"m<i>-{i}", requests=1, tokens=1_000) for i in range(40))
    report = _report(requests=40, ok=40, total_tokens=40_000, by_model=by_model, last_at=NOW)
    lines = cost_lines(report, {}, 1.0, "$")

    assert sum("   ◦" in line for line in lines) == MAX_MODEL_ROWS
    assert all("m&lt;i&gt;" in line for line in lines[2:])


def test_cost_section_without_settings_shows_no_money() -> None:
    report = _report(requests=1, ok=1, total_tokens=1_000, last_at=NOW)
    assert cost_section(report, None, _catalog({})) == []
    assert cost_section(report, FakeSettings(), _catalog({})) != []


def test_render_report_includes_the_cost_block_and_stays_within_the_limit() -> None:
    report = _report(
        requests=1,
        ok=1,
        total_tokens=1_000_000,
        by_model=(ModelUsage(model="m", requests=1, tokens=1_000_000),),
        last_at=NOW,
    )
    cost = cost_lines(report, {"m": 5.0}, 0.01, "$")
    text = render_report("Юзер", Period.WEEK, report, {}, NOW, cost)

    assert "<b>Стоимость</b>" in text
    assert "всего за период — $5.00" in text
    assert len(text) <= TELEGRAM_LIMIT


def test_render_report_has_no_cost_block_by_default() -> None:
    report = _report(requests=1, ok=1, total_tokens=1_000, last_at=NOW)
    assert "Стоимость" not in render_report("T", Period.WEEK, report, {}, NOW)


# ---- activity (pure) -------------------------------------------------------------------------------


def test_bar_cells_scales_to_the_peak_and_pads_to_a_fixed_width() -> None:
    assert bar_cells(0, 10) == "░" * 12
    assert bar_cells(10, 10) == "█" * 12
    assert bar_cells(5, 10) == "█" * 6 + "░" * 6
    assert bar_cells(5, 0) == "░" * 12  # an empty window never divides by zero
    assert len(bar_cells(1, 3)) == 12


def _days(values: list[int]) -> list[DayUsage]:
    first = date(2026, 1, 2)
    return [
        DayUsage(day=first + timedelta(days=i), requests=value, tokens=value * 100)
        for i, value in enumerate(values)
    ]


def test_render_activity_keeps_every_day_including_empty_ones() -> None:
    days = _days([0, 3, 0] + [0] * 11)
    text = render_activity("Все пользователи", days, NOW, TZ)

    assert len(days) == ACTIVITY_DAYS
    assert "Последние 14 дней" in text
    for day in days:
        assert f"{day.day:%d.%m}" in text  # every label is there, zero days included
    assert text.count("нет запросов") == 13
    assert "3 запр. · 300 ток." in text
    assert "<pre>" in text and "</pre>" in text  # monospace block: the columns line up
    assert "Итого: 3 запр. · 300 ток." in text


def test_render_activity_marks_today_and_reports_an_empty_window() -> None:
    text = render_activity("Все пользователи", _days([0] * 14), NOW, TZ)
    assert f"За {ACTIVITY_DAYS} дней запросов нет — график пустой." in text
    assert "<pre>" not in text

    text = render_activity("@gleb", _days([1] * 14), NOW, TZ)
    assert "15.01·" in text  # today carries the dot marker next to its label


def test_render_activity_single_day_and_stays_within_the_limit() -> None:
    text = render_activity("X", _days([1_000_000] * 14), NOW, TZ)
    assert len(text) <= TELEGRAM_LIMIT
    assert "<pre>" in text


# ---- activity screen (DB-backed) -------------------------------------------------------------------


@pytest.fixture
async def db() -> Database:
    database = Database(":memory:")
    await database.connect()
    try:
        yield database
    finally:
        await database.close()


async def _user(users: UserRepo, tg_id: int, username: str | None = None) -> None:
    await users.upsert_seen(tg_id, username, default_role=Role.USER, default_mode=Mode.NORMAL)


async def _request(db: Database, requests: RequestRepo, user_id: int, tokens: int, moment: datetime) -> None:
    await requests.add(RequestRecord(user_id, None, "m", RequestStatus.OK, 1, 10, tokens))
    conn = db.conn
    await conn.execute(
        "UPDATE requests SET created_at = ? WHERE id = (SELECT MAX(id) FROM requests)",
        (moment.isoformat(),),
    )
    await conn.commit()


class FakeMessage(Message):
    """A real Message (so `isinstance` checks in `show()` pass) that records what replaced it."""

    edits: list[str] = Field(default_factory=list)

    async def edit_text(self, text: str, reply_markup: Any = None, **kwargs: Any) -> None:
        self.edits.append(text)

    async def answer(self, text: str, reply_markup: Any = None, **kwargs: Any) -> None:
        self.edits.append(text)


class FakeCallback(CallbackQuery):
    """A real CallbackQuery that records the toasts/alerts the handlers raise."""

    notes: list[tuple[str | None, bool]] = Field(default_factory=list)

    async def answer(self, text: str | None = None, show_alert: bool = False, **kwargs: Any) -> None:
        self.notes.append((text, show_alert))


def _fake_callback(message: FakeMessage | None) -> FakeCallback:
    return FakeCallback.model_construct(message=message, data="", id="1")


def _fake_message(text: str = "") -> FakeMessage:
    return FakeMessage.model_construct(message_id=1, text=text)


async def test_activity_screen_for_one_user_keeps_the_empty_days(db: Database) -> None:
    users, requests = UserRepo(db), RequestRepo(db)
    await _user(users, 1, "gleb")
    await _user(users, 2, "other")
    await _request(db, requests, 1, 500, datetime(2026, 1, 15, 9, 0, tzinfo=UTC))
    await _request(db, requests, 2, 900, datetime(2026, 1, 15, 9, 0, tzinfo=UTC))

    screen = await activity_screen(users, requests, 1, NOW, TZ)

    assert screen is not None
    text, markup = screen
    assert "@gleb" in text
    assert "1 запр. · 500 ток." in text
    assert "дн" in text
    assert text.count("нет запросов") == ACTIVITY_DAYS - 1  # one busy day, 13 empty ones
    labels = [b.text for row in markup.inline_keyboard for b in row]
    assert labels == ["📊 Статистика", "« К пользователю"]
    packed = [b.callback_data for row in markup.inline_keyboard for b in row]
    assert UserCard(tg_id=1).pack() in packed
    assert Activity(tg_id=1).pack() not in packed


async def test_activity_screen_for_everyone_counts_all_users(db: Database) -> None:
    users, requests = UserRepo(db), RequestRepo(db)
    await _user(users, 1, "a")
    await _user(users, 2, "b")
    await _request(db, requests, 1, 300, datetime(2026, 1, 15, 9, 0, tzinfo=UTC))
    await _request(db, requests, 2, 200, datetime(2026, 1, 15, 9, 0, tzinfo=UTC))

    screen = await activity_screen(users, requests, None, NOW, TZ)

    assert screen is not None
    text, markup = screen
    assert "Все пользователи" in text
    assert "2 запр. · 500 ток." in text
    labels = [b.text for row in markup.inline_keyboard for b in row]
    assert labels == ["📊 Сводка", "« В меню"]


async def test_activity_screen_unknown_user_is_none(db: Database) -> None:
    assert await activity_screen(UserRepo(db), RequestRepo(db), 999, NOW, TZ) is None


async def test_activity_screen_with_no_requests_at_all_stays_friendly(db: Database) -> None:
    users, requests = UserRepo(db), RequestRepo(db)
    await _user(users, 1, None)

    screen = await activity_screen(users, requests, None, NOW, TZ)

    assert screen is not None
    text, _ = screen
    assert "запросов нет" in text
    assert "<pre>" not in text


async def test_activity_screen_escapes_the_username(db: Database) -> None:
    users, requests = UserRepo(db), RequestRepo(db)
    await _user(users, 1, "a<b>")

    screen = await activity_screen(users, requests, 1, NOW, TZ)

    assert screen is not None
    text, _ = screen
    assert "a&lt;b&gt;" in text
    assert "<b>" not in text.replace("📈 <b>Активность</b>", "")  # only our own bold tag survives


async def test_cb_activity_draws_the_chart_and_alerts_on_an_unknown_user(db: Database) -> None:
    """The router handler path, not just the screen: the filter and the DI names must line up."""
    users, requests = UserRepo(db), RequestRepo(db)
    await _user(users, 1, "gleb")
    message = _fake_message()
    callback = _fake_callback(message)

    await cb_activity(callback, Activity(tg_id=1), users, requests, TZ)
    assert message.edits and "Активность" in message.edits[0]

    missing = _fake_callback(None)
    await cb_activity(missing, Activity(tg_id=999), users, requests, TZ)
    assert missing.notes[-1] == ("Пользователь не найден", True)


# ---- usage screens (DB-backed) ---------------------------------------------------------------------


async def test_user_stats_screen_unknown_user_returns_none(db: Database) -> None:
    result = await user_stats_screen(
        UserRepo(db), RequestRepo(db), UploadRepo(db), 999, Period.WEEK, NOW, UTC_TZ
    )
    assert result is None


async def test_user_stats_screen_known_user(db: Database) -> None:
    users = UserRepo(db)
    await users.upsert_seen(1, "gleb", default_role=Role.USER, default_mode=Mode.NORMAL)
    screen = await user_stats_screen(users, RequestRepo(db), UploadRepo(db), 1, Period.WEEK, NOW, UTC_TZ)
    assert screen is not None
    text, markup = screen
    assert "@gleb" in text
    buttons = [btn for row in markup.inline_keyboard for btn in row]
    assert any(btn.text == "« К пользователю" for btn in buttons)
    assert any(btn.callback_data == UserCard(tg_id=1).pack() for btn in buttons)


async def test_dashboard_screen_includes_role_counts(db: Database) -> None:
    users = UserRepo(db)
    await users.upsert_seen(1, "a", default_role=Role.ADMIN, default_mode=Mode.NORMAL)
    await users.upsert_seen(2, "b", default_role=Role.USER, default_mode=Mode.NORMAL)
    await users.upsert_seen(3, "c", default_role=Role.PENDING, default_mode=Mode.NORMAL)

    text, _ = await dashboard_screen(users, RequestRepo(db), UploadRepo(db), Period.ALL, NOW, UTC_TZ)
    assert "Все пользователи" in text
    assert "Пользователи: 3 · 👑 1 · ✅ 1 · ⏳ 1 · 🚫 0" in text
