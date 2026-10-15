"""adminbot.health: the status/quotas screen, the error journal, the model report and the backup.

Pure renderers are tested directly; the DB-backed screens run against a real in-memory `Database`.
Nothing here talks to Telegram: the screens use real `Message`/`CallbackQuery` subclasses (so the
`isinstance` checks in the handlers behave as in production) while a fake captures the document, and a
fake `Database` covers the snapshot failures a real `VACUUM INTO` cannot be made to hit on demand.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from aiogram.types import CallbackQuery, Message
from pydantic import Field

from adminbot.callbacks import Backup, Errors, Health, Home, Models
from adminbot.common import PERIOD_LABELS, TELEGRAM_LIMIT, Period, Quotas, bot_alive
from adminbot.health import (
    ERROR_REASON_CHARS,
    ERRORS_PAGE,
    MAX_MODEL_ROWS,
    backup_filename,
    backup_warning,
    cb_backup,
    drop,
    errors_screen,
    fit_rows,
    fmt_uptime,
    make_snapshot,
    model_cost,
    models_screen,
    prices_from_catalog,
    quota_mark,
    quota_percent,
    render_errors,
    render_models,
    render_status,
    router,
    send_backup,
    status_screen,
)
from bot.domain import (
    BOT_STALE_AFTER_S,
    BotStatus,
    ErrorEntry,
    Mode,
    ModelStats,
    RequestRecord,
    RequestStatus,
    Role,
    UploadKind,
)
from bot.storage.db import Database
from bot.storage.repositories import RequestRepo, StatusRepo, UploadRepo, UserRepo

UTC_TZ = UTC
NOW = datetime(2026, 5, 20, 12, 0, tzinfo=UTC)
_BAD_AMP = re.compile(r"&(?!amp;|lt;|gt;|quot;|#39;)")  # a lone '&' means an entity got cut


# ---- fakes, fixtures and builders -------------------------------------------------------------------


@dataclass
class FakeSettings:
    """What the health screens read from Settings: prices, free-plan limits and the catalog path."""

    token_price: float = 0.01
    price_currency: str = "$"
    cloudflare_daily_neurons: int = 10_000
    models_file: Path = Path("bot/models.yaml")


class FakeMessage(Message):
    """A real Message (so the handlers' isinstance checks pass) that records sent documents."""

    documents: list[Any] = Field(default_factory=list)

    async def answer_document(self, document: Any, **kwargs: Any) -> None:
        self.documents.append(document)

    async def answer(self, text: str, **kwargs: Any) -> None:
        return None

    async def edit_text(self, text: str, **kwargs: Any) -> None:
        return None


class RefusingMessage(FakeMessage):
    """An upload that blows up on the Telegram side."""

    async def answer_document(self, document: Any, **kwargs: Any) -> None:
        raise RuntimeError("telegram is down")


class FakeCallback(CallbackQuery):
    """A real CallbackQuery that records the toasts/alerts the handlers raise."""

    notes: list[tuple[str | None, bool]] = Field(default_factory=list)

    async def answer(self, text: str | None = None, show_alert: bool = False, **kwargs: Any) -> None:
        self.notes.append((text, show_alert))


class FakeStatuses:
    """Stands in for StatusRepo when a test needs a heartbeat the real clock cannot produce."""

    def __init__(self, status: BotStatus | None) -> None:
        self.status = status

    async def get(self) -> BotStatus | None:
        return self.status


class BrokenSnapshot:
    """A Database whose snapshot always fails (a locked DB, a revoked directory, a full disk)."""

    async def snapshot(self, path: Path) -> None:
        # to_thread: ruff rightly forbids blocking pathlib calls straight inside a coroutine, and the
        # partial file this leaves behind is exactly the case make_snapshot must clean up.
        await asyncio.to_thread(path.write_bytes, b"half a backup")
        raise RuntimeError("database is locked")


def _message() -> FakeMessage:
    return FakeMessage.model_construct(message_id=1, text="экран")


def _callback(message: Message | None = None) -> FakeCallback:
    return FakeCallback.model_construct(message=message, data="", id="1")


def _quotas(**overrides: Any) -> Quotas:
    base: dict[str, Any] = dict(images=5, neurons_limit=10_000)
    base.update(overrides)
    return Quotas(**base)


def _entry(**overrides: Any) -> ErrorEntry:
    base: dict[str, Any] = dict(
        id=1,
        user_id=7,
        username="gleb",
        model="gpt-5.6-sol",
        status=RequestStatus.FAILED,
        error="timeout",
        attempts=3,
        latency_ms=1_500,
        created_at=NOW,
    )
    base.update(overrides)
    return ErrorEntry(**base)


def _stats(**overrides: Any) -> ModelStats:
    base: dict[str, Any] = dict(
        model="m",
        requests=4,
        ok=3,
        stub=1,
        failed=0,
        avg_ok_latency_ms=2_000,
        max_ok_latency_ms=3_500,
        tokens=50_000,
    )
    base.update(overrides)
    return ModelStats(**base)


def _record(
    *,
    user_id: int = 7,
    status: RequestStatus = RequestStatus.OK,
    error: str | None = None,
    model: str = "gpt-5.6-sol",
    attempts: int = 1,
    latency_ms: int = 1_000,
    approx_tokens: int = 100,
    web_calls: int = 0,
) -> RequestRecord:
    return RequestRecord(
        user_id=user_id,
        conversation_id=None,
        model=model,
        status=status,
        attempts=attempts,
        latency_ms=latency_ms,
        approx_tokens=approx_tokens,
        error=error,
        web_calls=web_calls,
    )


@pytest.fixture
async def db() -> Database:
    database = Database(":memory:")
    await database.connect()
    try:
        yield database
    finally:
        await database.close()


async def _seed_user(users: UserRepo, tg_id: int = 7, username: str | None = "gleb") -> None:
    await users.upsert_seen(tg_id, username, default_role=Role.USER, default_mode=Mode.NORMAL)


async def _insert(
    requests: RequestRepo, db: Database, record: RequestRecord, created_at: datetime | None = None
) -> None:
    """Add one request; `created_at` re-dates just that row, because the repo stamps "now"."""
    await requests.add(record)
    if created_at is None:
        return
    conn = db.conn
    await conn.execute(
        "UPDATE requests SET created_at = ? WHERE id = (SELECT MAX(id) FROM requests)",
        (created_at.isoformat(),),
    )
    await conn.commit()


# ---- quota helpers (pure) --------------------------------------------------------------------------


def test_quota_percent_rounds_and_handles_a_missing_limit() -> None:
    assert quota_percent(85, 100) == 85
    assert quota_percent(1, 3) == 33
    assert quota_percent(50, 0) == 0


def test_quota_mark_warns_above_eighty_percent_only() -> None:
    assert quota_mark(80, 100) == " ⚠️"  # exactly at the documented threshold
    assert quota_mark(79, 100) == ""
    assert quota_mark(101, 100) == " ⛔"
    assert quota_mark(5, 0) == ""


def test_fmt_uptime_switches_units() -> None:
    assert fmt_uptime(NOW - timedelta(minutes=5), NOW) == "5 мин"
    assert fmt_uptime(NOW - timedelta(hours=3, minutes=20), NOW) == "3 ч 20 мин"
    assert fmt_uptime(NOW - timedelta(days=2, hours=5), NOW) == "2 д 5 ч"
    assert fmt_uptime(NOW + timedelta(minutes=5), NOW) == "0 мин"  # a clock skew shows nothing odd


# ---- 1. status screen ------------------------------------------------------------------------------


def test_render_status_never_reported_is_friendly() -> None:
    text = render_status(None, _quotas(), NOW, UTC_TZ)
    assert "ещё не отчитывался" in text
    assert "строки статуса нет" in text
    assert "Лимиты бесплатных планов" in text  # quotas stay useful without a heartbeat


def test_render_status_alive_shows_uptime_queue_and_jobs() -> None:
    status = BotStatus(
        started_at=NOW - timedelta(days=1, hours=2),
        heartbeat_at=NOW - timedelta(seconds=BOT_STALE_AFTER_S),
        queue_depth=3,
        active_jobs=2,
    )
    text = render_status(status, _quotas(), NOW, UTC_TZ)
    assert bot_alive(status, NOW) is True
    assert "бот работает ✅" in text
    assert "1 д 2 ч" in text
    assert "Очередь к шлюзу: 3 · активных ответов: 2" in text
    assert "молчит" not in text


def test_render_status_stale_is_flagged() -> None:
    status = BotStatus(
        started_at=NOW - timedelta(hours=1),
        heartbeat_at=NOW - timedelta(seconds=BOT_STALE_AFTER_S + 1),
        queue_depth=0,
        active_jobs=0,
    )
    text = render_status(status, _quotas(), NOW, UTC_TZ)
    assert bot_alive(status, NOW) is False
    assert "бот молчит ⚠️" in text
    assert "упал или завис" in text


def test_render_status_marks_the_unknown_states_readably() -> None:
    text = render_status(None, _quotas(), NOW, UTC_TZ)
    assert "⚪" in text  # never reported: neither green nor warning
    assert "<i>" in text  # the explanation is italic HTML, since the bot parses HTML, not Markdown
    assert "_Оценка" not in text  # a Markdown underscore would show up literally in the message


def test_render_status_shows_quotas_with_percentages_and_warnings() -> None:
    quotas = _quotas(images=120, neurons_limit=10_000)
    text = render_status(None, quotas, NOW, UTC_TZ)
    assert "120 / 10 000 картинок в день — 1%" in text
    # 120 images * 104 neurons = 12 480 > the 10 000 daily neurons: past the free plan
    assert "Нейроны за сегодня: ~12 480 / 10 000 — 125% ⛔" in text


def test_render_status_stays_within_the_telegram_limit() -> None:
    text = render_status(None, _quotas(images=999_999_999), NOW, UTC_TZ)
    assert len(text) <= TELEGRAM_LIMIT


async def test_status_screen_reads_the_real_heartbeat_row(db: Database) -> None:
    statuses = StatusRepo(db)
    await statuses.beat(NOW - timedelta(hours=2), queue_depth=4, active_jobs=1)

    text, markup = await status_screen(statuses, RequestRepo(db), UploadRepo(db), FakeSettings(), NOW, UTC_TZ)
    assert "бот работает ✅" in text  # beat() stamps "now", so the heartbeat is fresh
    assert "Очередь к шлюзу: 4 · активных ответов: 1" in text
    buttons = [btn.text for row in markup.inline_keyboard for btn in row]
    assert "🗂 Журнал ошибок" in buttons
    assert "💾 Бэкап базы" in buttons
    assert buttons[-1] == "« В меню"


async def test_status_screen_went_quiet(db: Database) -> None:
    stale = BotStatus(
        started_at=NOW - timedelta(days=3),
        heartbeat_at=NOW - timedelta(seconds=BOT_STALE_AFTER_S + 5),
        queue_depth=0,
        active_jobs=0,
    )
    text, _ = await status_screen(
        FakeStatuses(stale), RequestRepo(db), UploadRepo(db), FakeSettings(), NOW, UTC_TZ
    )
    assert "бот молчит ⚠️" in text
    assert bot_alive(stale, NOW) is False


async def test_status_screen_without_a_row_says_so(db: Database) -> None:
    text, _ = await status_screen(
        StatusRepo(db), RequestRepo(db), UploadRepo(db), FakeSettings(), NOW, UTC_TZ
    )
    assert "ещё не отчитывался" in text


async def test_status_screen_counts_real_quota_usage(db: Database) -> None:
    """The image counters come from load_quotas, so real rows must move them."""
    await _seed_user(UserRepo(db))
    requests, uploads = RequestRepo(db), UploadRepo(db)
    for _ in range(2):
        await uploads.add(7, UploadKind.GEN_IMAGE, "pic.jpg", 1_000)

    text, _ = await status_screen(FakeStatuses(None), requests, uploads, FakeSettings(), NOW, UTC_TZ)
    assert "2 / 10 000 картинок в день" in text
    assert "~208 / 10 000" in text  # two images * NEURONS_PER_IMAGE


# ---- 2. error journal ------------------------------------------------------------------------------


def test_render_errors_empty_journal_is_friendly() -> None:
    text = render_errors([], 0, 1, 0, NOW, UTC_TZ)
    assert "Журнал ошибок" in text
    assert "стр. 1/1" in text
    assert "Записей нет" in text


def test_render_errors_shows_user_model_reason_attempts_and_marks() -> None:
    failed = _entry(error="Ошибка шлюза: 502")
    stub = _entry(id=2, status=RequestStatus.STUB, error="политическая тема", attempts=1)
    text = render_errors([failed, stub], 0, 2, 2, NOW, UTC_TZ)
    assert "всего 2, стр. 1/2" in text
    assert "❌ <b>gleb</b> · gpt-5.6-sol" in text
    assert "🙊 <b>gleb</b> · gpt-5.6-sol" in text
    assert "попыток: 3 · 1.5 с" in text
    assert "Ошибка шлюза: 502" in text


def test_render_errors_escapes_untrusted_gateway_text_and_names() -> None:
    entry = _entry(username="<b>root</b>", error='<script>alert("x")</script> & more', model="a<b>")
    text = render_errors([entry], 0, 1, 1, NOW, UTC_TZ)
    assert "<script>" not in text
    assert "&lt;script&gt;" in text
    assert "&lt;b&gt;root&lt;/b&gt;" in text
    assert "a&lt;b&gt;" in text
    assert "&amp; more" in text
    assert not _BAD_AMP.search(text)


def test_render_errors_truncates_raw_text_before_escaping() -> None:
    entry = _entry(error="&" * 5_000)  # 5 000 raw '&' would explode to 25 000 escaped characters
    text = render_errors([entry], 0, 1, 1, NOW, UTC_TZ)
    assert len(text) <= TELEGRAM_LIMIT
    assert text.count("&amp;") == ERROR_REASON_CHARS  # cut raw, then escaped whole
    assert not _BAD_AMP.search(text)


def test_render_errors_missing_reason_omits_the_italic_line() -> None:
    text = render_errors([_entry(error=None)], 0, 1, 1, NOW, UTC_TZ)
    assert "<i>" not in text
    assert "попыток: 3" in text


def test_render_errors_collapses_a_multiline_traceback() -> None:
    text = render_errors([_entry(error="Traceback:\n  line 1\n  line 2")], 0, 1, 1, NOW, UTC_TZ)
    assert "Traceback: line 1 line 2" in text


def test_fit_rows_stops_at_the_limit_and_marks_the_cut() -> None:
    text = fit_rows("head", ["x" * 4_000, "y" * 4_000])
    assert len(text) <= TELEGRAM_LIMIT
    assert text.endswith("[обрезано]")
    assert "y" not in text


async def test_errors_screen_empty_journal_has_no_navigation(db: Database) -> None:
    text, markup = await errors_screen(RequestRepo(db), 0, NOW, UTC_TZ)
    assert "Записей нет" in text
    buttons = [btn.text for row in markup.inline_keyboard for btn in row]
    assert "◀ Раньше" not in buttons and "Позже ▶" not in buttons
    assert buttons[-1] == "« К состоянию"


async def test_errors_screen_paged_newest_first(db: Database) -> None:
    await _seed_user(UserRepo(db))
    requests = RequestRepo(db)
    for n in range(ERRORS_PAGE * 2 + 1):  # 11 rows -> 3 pages
        await _insert(
            requests,
            db,
            _record(status=RequestStatus.FAILED, error=f"fail {n}", attempts=n + 1),
        )

    text, markup = await errors_screen(requests, 0, NOW, UTC_TZ)
    assert "всего 11, стр. 1/3" in text
    buttons = [btn.text for row in markup.inline_keyboard for btn in row]
    assert "◀ Раньше" not in buttons and "Позже ▶" in buttons

    text, markup = await errors_screen(requests, 1, NOW, UTC_TZ)
    assert "стр. 2/3" in text
    buttons = [btn.text for row in markup.inline_keyboard for btn in row]
    assert "◀ Раньше" in buttons and "Позже ▶" in buttons

    text, markup = await errors_screen(requests, 2, NOW, UTC_TZ)
    assert "стр. 3/3" in text
    buttons = [btn.text for row in markup.inline_keyboard for btn in row]
    assert "◀ Раньше" in buttons and "Позже ▶" not in buttons


async def test_errors_screen_clamps_a_page_beyond_the_range(db: Database) -> None:
    await _seed_user(UserRepo(db))
    requests = RequestRepo(db)
    for n in range(ERRORS_PAGE + 1):  # 2 pages
        await _insert(requests, db, _record(status=RequestStatus.FAILED, error=f"fail {n}"))

    text, markup = await errors_screen(requests, 99, NOW, UTC_TZ)
    assert "стр. 2/2" in text  # clamped to the last page, not an empty screen
    buttons = [btn.text for row in markup.inline_keyboard for btn in row]
    assert "Позже ▶" not in buttons and "◀ Раньше" in buttons


async def test_errors_screen_clamps_a_negative_page(db: Database) -> None:
    await _seed_user(UserRepo(db))
    requests = RequestRepo(db)
    for n in range(ERRORS_PAGE + 1):
        await _insert(requests, db, _record(status=RequestStatus.FAILED, error=f"fail {n}"))

    text, markup = await errors_screen(requests, -3, NOW, UTC_TZ)
    assert "стр. 1/2" in text
    buttons = [btn.text for row in markup.inline_keyboard for btn in row]
    assert "◀ Раньше" not in buttons


async def test_errors_screen_skips_successful_requests(db: Database) -> None:
    await _seed_user(UserRepo(db))
    requests = RequestRepo(db)
    await _insert(requests, db, _record(status=RequestStatus.OK))

    text, _ = await errors_screen(requests, 0, NOW, UTC_TZ)
    assert "Записей нет" in text


async def test_errors_screen_failed_only_filter_hides_stubs(db: Database) -> None:
    await _seed_user(UserRepo(db))
    requests = RequestRepo(db)
    await _insert(requests, db, _record(status=RequestStatus.STUB, error="stub"))

    text, _ = await errors_screen(requests, 0, NOW, UTC_TZ, failed_only=True)
    assert "Записей нет" in text
    text, _ = await errors_screen(requests, 0, NOW, UTC_TZ)
    assert "Записей нет" not in text  # without the filter the stub is part of the journal


async def test_errors_screen_filters_by_user(db: Database) -> None:
    users = UserRepo(db)
    await _seed_user(users)
    await users.upsert_seen(9, "other", default_role=Role.USER, default_mode=Mode.NORMAL)
    requests = RequestRepo(db)
    await _insert(requests, db, _record(status=RequestStatus.FAILED, error="mine"))
    await _insert(requests, db, _record(user_id=9, status=RequestStatus.FAILED, error="theirs"))

    text, _ = await errors_screen(requests, 0, NOW, UTC_TZ, tg_id=9)
    assert "всего 1" in text
    assert "other" in text
    assert "theirs" in text
    assert "mine" not in text


async def test_errors_screen_unknown_user_shows_nothing_rather_than_crashing(db: Database) -> None:
    text, _ = await errors_screen(RequestRepo(db), 0, NOW, UTC_TZ, tg_id=424_242)
    assert "Записей нет" in text


async def test_errors_screen_shows_a_prefixed_id_when_the_username_is_gone(db: Database) -> None:
    await _seed_user(UserRepo(db), 7, None)  # a user row without a username
    await _insert(RequestRepo(db), db, _record(status=RequestStatus.FAILED, error="no username"))
    text, _ = await errors_screen(RequestRepo(db), 0, NOW, UTC_TZ)
    assert "ID 7" in text


# ---- 3. models -------------------------------------------------------------------------------------


def test_model_cost_uses_the_per_model_override() -> None:
    stats = [_stats(model="cheap", tokens=1_000_000), _stats(model="pricey", tokens=1_000_000)]
    cost = model_cost(stats, {"pricey": 0.5}, 0.01)
    assert cost == pytest.approx(0.51)  # 1M * 0.01 + 1M * 0.5, both per 1M tokens


def test_model_cost_without_any_stats_is_zero() -> None:
    assert model_cost([], {}, 0.01) == 0.0


def test_render_models_empty_period_is_friendly() -> None:
    text = render_models([], Period.TODAY, {}, 0.01, "$")
    assert "Запросов за период нет." in text
    assert PERIOD_LABELS[Period.TODAY] in text


def test_render_models_shows_split_latency_tokens_and_money() -> None:
    text = render_models([_stats(model="m1", tokens=1_000_000)], Period.WEEK, {"m1": 0.5}, 0.01, "$")
    assert PERIOD_LABELS[Period.WEEK] in text
    assert "<b>m1</b> — 4 запр. · своя цена 0.5" in text
    assert "✅ 3 · 🙊 1 · ❌ 0" in text
    assert "⚡ сред. 2.0 с · макс. 3.5 с" in text
    assert "1.0M ток." in text  # short_number, as in the usage report
    # 1M tokens at the model's own 0.5 per 1M, both in the total and in the row
    assert "стоимость ≈ $0.500" in text
    assert "$0.500" in text


def test_render_models_falls_back_to_the_default_price() -> None:
    text = render_models([_stats(model="m", tokens=1_000_000)], Period.MONTH, {}, 0.01, "₽")
    assert "своя цена" not in text
    assert "0.010 ₽" in text  # a non-symbol currency goes after the number (fmt_money)


def test_render_models_unanswered_model_shows_dashes() -> None:
    text = render_models(
        [_stats(model="m", requests=2, ok=0, stub=2, avg_ok_latency_ms=0, max_ok_latency_ms=0)],
        Period.ALL,
        {},
        0.01,
        "$",
    )
    assert "⚡ сред. — · макс. —" in text


def test_render_models_many_models_are_capped_like_the_usage_report() -> None:
    stats = [_stats(model=f"model-{i}", tokens=1_000) for i in range(100)]
    text = render_models(stats, Period.ALL, {}, 0.01, "$")
    assert len(text) <= TELEGRAM_LIMIT
    assert text.count("<b>model-") == MAX_MODEL_ROWS
    assert "Моделей: 100 · в отчёте 15" in text
    assert "… ещё 85 моделей не показаны" in text


def test_render_models_escapes_model_names() -> None:
    text = render_models([_stats(model="<img src=x>")], Period.WEEK, {}, 0.01, "$")
    assert "<img" not in text
    assert "&lt;img src=x&gt;" in text


def test_prices_from_catalog_reads_overrides_and_survives_a_broken_file(tmp_path: Path) -> None:
    good = tmp_path / "models.yaml"
    good.write_text(
        "max_tokens_cap: 1000\nmax_attempts: 2\n"
        "modes: {fast: a, normal: a, deep: a}\n"
        "models:\n"
        "  a: {title: A, max_tokens: 100, timeout_s: 5, price: 0.5}\n"
        "  b: {title: B, max_tokens: 100, timeout_s: 5}\n",
        encoding="utf-8",
    )
    assert prices_from_catalog(good) == {"a": 0.5}  # only the model with its own price

    broken = tmp_path / "broken.yaml"
    broken.write_text("modes: {}\nmodels: {}\n", encoding="utf-8")
    assert prices_from_catalog(broken) == {}
    assert prices_from_catalog(tmp_path / "missing.yaml") == {}


async def test_models_screen_filters_by_period(db: Database) -> None:
    await _seed_user(UserRepo(db))
    requests = RequestRepo(db)
    await _insert(requests, db, _record(model="old-model"), created_at=NOW - timedelta(days=20))
    await _insert(requests, db, _record(model="new-model"), created_at=NOW - timedelta(hours=2))

    today, _ = await models_screen(requests, Period.TODAY, FakeSettings(), NOW, UTC_TZ)
    assert "new-model" in today
    assert "old-model" not in today

    week, _ = await models_screen(requests, Period.WEEK, FakeSettings(), NOW, UTC_TZ)
    assert "new-model" in week
    assert "old-model" not in week

    month, _ = await models_screen(requests, Period.MONTH, FakeSettings(), NOW, UTC_TZ)
    assert "old-model" in month and "new-model" in month

    everything, _ = await models_screen(requests, Period.ALL, FakeSettings(), NOW, UTC_TZ)
    assert "old-model" in everything


async def test_models_screen_empty_and_keyboard(db: Database) -> None:
    text, markup = await models_screen(RequestRepo(db), Period.WEEK, FakeSettings(), NOW, UTC_TZ)
    assert "Запросов за период нет." in text
    rows = markup.inline_keyboard
    assert len(rows[0]) == 4  # the four periods
    assert any(btn.text == "✅ 7 дней" for btn in rows[0])
    assert rows[-1][0].text == "« К состоянию"
    assert rows[-1][0].callback_data == Health().pack()


# ---- 4. backup -------------------------------------------------------------------------------------


def test_backup_filename_carries_the_date() -> None:
    assert backup_filename(NOW) == "bot-backup-2026-05-20.sqlite3"


def test_backup_warning_names_the_risk_and_asks_to_confirm() -> None:
    text = backup_warning()
    assert "переписку каждого пользователя" in text
    assert "только владелец" in text
    assert "Продолжаем?" in text


async def test_backup_unconfirmed_shows_the_warning_and_sends_nothing(db: Database) -> None:
    message = _message()
    callback = _callback(message)
    await cb_backup(callback, Backup(confirm=False), db)
    assert message.documents == []  # nothing was uploaded...
    assert callback.notes == [(None, False)]  # ...and the screen was merely replaced


async def test_backup_warning_keyboard_offers_confirm_and_cancel() -> None:
    from adminbot.health import backup_keyboard

    buttons = [btn for row in backup_keyboard().inline_keyboard for btn in row]
    assert buttons[0].callback_data == Backup(confirm=True).pack()
    assert buttons[1].callback_data == Health().pack()  # cancel, not a second confirm


async def test_backup_confirmed_sends_the_snapshot_and_deletes_it(db: Database) -> None:
    await _seed_user(UserRepo(db))
    message = _message()
    callback = _callback(message)
    await cb_backup(callback, Backup(confirm=True), db)

    assert len(message.documents) == 1
    document = message.documents[0]
    assert document.filename == backup_filename(datetime.now(UTC))  # dated, as asked
    # the temp copy must not outlive the sending (to_thread: blocking pathlib inside a coroutine)
    assert not await asyncio.to_thread(Path(document.path).exists)
    assert callback.notes == [("Бэкап отправлен", False)]


async def test_backup_confirm_without_a_message_still_answers(db: Database) -> None:
    callback = _callback(None)
    await cb_backup(callback, Backup(confirm=True), db)
    assert callback.notes == [("Бэкап отправлен", False)]  # nothing to attach, but no crash


async def test_send_backup_survives_a_failing_database() -> None:
    callback = _callback(_message())
    await send_backup(callback, BrokenSnapshot(), NOW)  # type: ignore[arg-type]

    note, alert = callback.notes[0]
    assert alert is True
    assert "Не получилось сделать бэкап" in (note or "")


async def test_send_backup_survives_a_failing_upload(db: Database) -> None:
    message = RefusingMessage.model_construct(message_id=1, text="экран")
    callback = _callback(message)
    await send_backup(callback, db, NOW)

    note, alert = callback.notes[0]
    assert alert is True
    assert "подробности в логе" in (note or "")


async def test_make_snapshot_writes_a_usable_copy(tmp_path: Path) -> None:
    database = Database(tmp_path / "bot.db")
    await database.connect()
    try:
        await _seed_user(UserRepo(database), 5, "kept")
        snapshot = await make_snapshot(database, NOW)
        try:
            assert snapshot.path.exists()
            assert snapshot.path.name == backup_filename(NOW)  # the temp copy is dated too
            backup = Database(snapshot.path)  # a real, openable database with the data in it
            await backup.connect()
            try:
                user = await UserRepo(backup).get(5)
                assert user is not None and user.username == "kept"
            finally:
                await backup.close()
        finally:
            drop(snapshot)  # also removes the private folder it was made in
        assert not snapshot.path.exists()
        assert not snapshot.path.parent.exists()
    finally:
        await database.close()


async def test_make_snapshot_into_a_directory_uses_the_dated_name(tmp_path: Path) -> None:
    database = Database(tmp_path / "bot.db")
    await database.connect()
    try:
        snapshot = await make_snapshot(database, NOW, tmp_path)
        assert snapshot.path == tmp_path / backup_filename(NOW)
        assert snapshot.path.exists()
        assert snapshot.own_dir is False  # someone else's folder: never delete it
    finally:
        await database.close()


async def test_make_snapshot_refuses_an_existing_file(tmp_path: Path) -> None:
    database = Database(tmp_path / "bot.db")
    await database.connect()
    try:
        existing = tmp_path / backup_filename(NOW)
        existing.write_bytes(b"old backup")
        with pytest.raises(FileExistsError):
            await make_snapshot(database, NOW, tmp_path)
        assert existing.read_bytes() == b"old backup"  # an older backup is never overwritten
    finally:
        await database.close()


async def test_make_snapshot_removes_a_partial_file(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError):
        await make_snapshot(BrokenSnapshot(), NOW, tmp_path)  # type: ignore[arg-type]
    assert not (tmp_path / backup_filename(NOW)).exists()


# ---- router wiring ---------------------------------------------------------------------------------


def test_router_exposes_exactly_the_four_screens() -> None:
    names = {handler.callback.__name__ for handler in router.callback_query.handlers}
    assert names == {"cb_health", "cb_errors", "cb_models", "cb_backup"}


def test_health_callbacks_are_distinct_and_typed() -> None:
    assert Errors(page=0).pack() != Errors(page=1).pack()
    assert Models(period=Period.WEEK).period is Period.WEEK
    assert Backup(confirm=False).confirm is False
    assert Home().pack() != Health().pack()


async def test_the_main_menu_offers_the_health_screen(db: Database) -> None:
    """The one line WP-9 adds to home.py: the screen must be reachable from the menu."""
    from adminbot.home import main_menu_screen

    _, markup = await main_menu_screen(UserRepo(db), RequestRepo(db), NOW, UTC_TZ)
    buttons = [btn for row in markup.inline_keyboard for btn in row]
    health = [btn for btn in buttons if btn.callback_data == Health().pack()]
    assert len(health) == 1
    assert "Состояние" in health[0].text
