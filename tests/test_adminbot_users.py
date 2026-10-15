"""Screen builders and pure helpers of adminbot.home / adminbot.users, no network."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pytest
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.base import StorageKey
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import CallbackQuery, Message
from pydantic import Field

from adminbot.callbacks import (
    IDLE_CUSTOM,
    IDLE_DEFAULT,
    LIMIT_CUSTOM,
    LIMIT_DEFAULT,
    ContextIdle,
    DailyLimit,
    SetContextIdle,
    SetDailyLimit,
    SetRole,
)
from adminbot.home import main_menu_screen, top_screen
from adminbot.users import (
    DAILY_PRESETS,
    ContextIdleForm,
    DailyLimitForm,
    cb_context_idle,
    cb_daily_limit,
    cb_set_context_idle,
    cb_set_daily_limit,
    context_idle_screen,
    daily_limit_screen,
    limit_screen,
    on_context_idle_value,
    on_daily_limit_value,
    parse_count,
    parse_hours,
    parse_tokens,
    set_role_guard,
    user_card_screen,
    user_list_screen,
)
from bot.domain import DAILY_LIMIT_MAX, DAILY_LIMIT_MIN, Mode, RequestRecord, RequestStatus, Role
from bot.render.format import short_number
from bot.storage.db import Database
from bot.storage.repositories import ConversationRepo, RequestRepo, UsageRepo, UserRepo


@dataclass(frozen=True)
class FakeSettings:
    """Just the attributes the screens read off Settings."""

    admin_ids: frozenset[int] = frozenset()
    history_token_budget: int = 350_000
    daily_request_limit: int = 50
    admin_utc_offset: int = 3
    context_idle_hours: float = 12.0


@dataclass
class Repos:
    db: Database
    users: UserRepo
    requests: RequestRepo
    daily: UsageRepo
    dialogs: ConversationRepo


@pytest.fixture
async def repos():
    db = Database(":memory:")
    await db.connect()
    bundle = Repos(db, UserRepo(db), RequestRepo(db), UsageRepo(db), ConversationRepo(db))
    yield bundle
    await db.close()


@pytest.fixture
def state() -> FSMContext:
    return FSMContext(storage=MemoryStorage(), key=StorageKey(bot_id=1, chat_id=1, user_id=1))


class FakeMessage(Message):
    """A real Message (so `isinstance` checks in the handlers pass) that records its answers."""

    answers: list[tuple[str, Any]] = Field(default_factory=list)

    async def answer(self, text: str, reply_markup: Any = None, **kwargs: Any) -> None:
        self.answers.append((text, reply_markup))


class FakeCallback(CallbackQuery):
    """A real CallbackQuery that records the toasts/alerts the handlers raise."""

    notes: list[tuple[str | None, bool]] = Field(default_factory=list)

    async def answer(self, text: str | None = None, show_alert: bool = False, **kwargs: Any) -> None:
        self.notes.append((text, show_alert))


def _message(text: str | None = None) -> FakeMessage:
    return FakeMessage.model_construct(message_id=1, text=text)


def _callback(message: FakeMessage | None = None) -> FakeCallback:
    return FakeCallback.model_construct(message=message, data="", id="1")


def _labels(markup) -> list[str]:
    return [button.text for row in markup.inline_keyboard for button in row]


# ---- parse_tokens ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("64000", 64_000),
        ("64 000", 64_000),
        ("64k", 64_000),
        ("64К", 64_000),
        ("1.5m", 1_500_000),
        ("1,5M", 1_500_000),
        ("  2М ", 2_000_000),
    ],
)
def test_parse_tokens_accepts(text: str, expected: int) -> None:
    assert parse_tokens(text) == expected


@pytest.mark.parametrize("text", ["", "abc", "-5", "0", "1e3x"])
def test_parse_tokens_rejects(text: str) -> None:
    assert parse_tokens(text) is None


# ---- main menu --------------------------------------------------------------------------------


async def test_main_menu_counts(repos: Repos) -> None:
    await repos.users.upsert_seen(1, "a", default_role=Role.USER, default_mode=Mode.NORMAL)
    await repos.users.upsert_seen(2, "b", default_role=Role.PENDING, default_mode=Mode.NORMAL)
    await repos.users.upsert_seen(3, "c", default_role=Role.PENDING, default_mode=Mode.NORMAL)
    await repos.requests.add(RequestRecord(1, None, "m", RequestStatus.OK, 1, 10, 500))
    await repos.requests.add(RequestRecord(1, None, "m", RequestStatus.OK, 1, 10, 500))

    text, _ = await main_menu_screen(repos.users, repos.requests, datetime.now(UTC), UTC)
    assert "Пользователей: 3" in text
    assert "ждут одобрения: 2" in text
    assert "Сегодня: 2 запросов" in text
    assert "1.0k" in text  # short_number(1000)


# ---- user card ----------------------------------------------------------------------------------


async def test_user_card_personal_limit_escaping_and_block_button(repos: Repos) -> None:
    await repos.users.upsert_seen(10, "a<b>", default_role=Role.USER, default_mode=Mode.NORMAL)
    await repos.users.set_context_limit(10, 64_000)
    now = datetime.now(UTC)

    settings = FakeSettings()
    result = await user_card_screen(
        10, repos.users, repos.requests, repos.daily, repos.dialogs, settings, UTC, now
    )
    assert result is not None
    text, markup = result
    assert "a&lt;b&gt;" in text
    assert "64 000 ток. (личный)" in text
    labels = _labels(markup)
    assert "🚫 Заблокировать" in labels
    role_words = ("Одобрить", "Разблокировать", "Отклонить")
    assert not any(word in label for label in labels for word in role_words)


async def test_user_card_global_limit(repos: Repos) -> None:
    await repos.users.upsert_seen(11, "b", default_role=Role.USER, default_mode=Mode.NORMAL)
    now = datetime.now(UTC)

    result = await user_card_screen(
        11, repos.users, repos.requests, repos.daily, repos.dialogs, FakeSettings(), UTC, now
    )
    assert result is not None
    text, _ = result
    assert "350 000 ток. (общий)" in text


async def test_user_card_admin_has_no_limit_line_and_no_role_buttons(repos: Repos) -> None:
    await repos.users.upsert_seen(12, "adm", default_role=Role.ADMIN, default_mode=Mode.NORMAL)
    now = datetime.now(UTC)

    result = await user_card_screen(
        12, repos.users, repos.requests, repos.daily, repos.dialogs, FakeSettings(), UTC, now
    )
    assert result is not None
    text, markup = result
    assert "без лимита" in text
    role_button_texts = {"✅ Одобрить", "🚫 Отклонить", "🚫 Заблокировать", "🔓 Разблокировать"}
    assert not (set(_labels(markup)) & role_button_texts)


async def test_user_card_pending_shows_approve_and_reject(repos: Repos) -> None:
    await repos.users.upsert_seen(13, "p", default_role=Role.PENDING, default_mode=Mode.NORMAL)
    now = datetime.now(UTC)

    result = await user_card_screen(
        13, repos.users, repos.requests, repos.daily, repos.dialogs, FakeSettings(), UTC, now
    )
    assert result is not None
    _, markup = result
    labels = _labels(markup)
    assert "✅ Одобрить" in labels
    assert "🚫 Отклонить" in labels
    approve_data = next(
        b.callback_data for row in markup.inline_keyboard for b in row if b.text == "✅ Одобрить"
    )
    assert SetRole.unpack(approve_data) == SetRole(tg_id=13, role=Role.USER)


async def test_user_card_blocked_shows_unblock(repos: Repos) -> None:
    await repos.users.upsert_seen(14, "bl", default_role=Role.BLOCKED, default_mode=Mode.NORMAL)
    now = datetime.now(UTC)

    result = await user_card_screen(
        14, repos.users, repos.requests, repos.daily, repos.dialogs, FakeSettings(), UTC, now
    )
    assert result is not None
    _, markup = result
    assert "🔓 Разблокировать" in _labels(markup)


async def test_user_card_unknown_id_is_none(repos: Repos) -> None:
    result = await user_card_screen(
        999, repos.users, repos.requests, repos.daily, repos.dialogs, FakeSettings(), UTC, datetime.now(UTC)
    )
    assert result is None


# ---- user list ----------------------------------------------------------------------------------


async def test_user_list_pagination_and_clamping(repos: Repos) -> None:
    for i in range(25):
        await repos.users.upsert_seen(i, f"u{i}", default_role=Role.USER, default_mode=Mode.NORMAL)

    text, markup = await user_list_screen(None, 0, repos.users)
    assert "стр. 1/3" in text
    assert [t for t in _labels(markup) if t in ("◀", "▶")] == ["▶"]

    text, markup = await user_list_screen(None, 1, repos.users)
    assert "стр. 2/3" in text
    assert [t for t in _labels(markup) if t in ("◀", "▶")] == ["◀", "▶"]

    text, markup = await user_list_screen(None, 99, repos.users)  # clamps to the last page
    assert "стр. 3/3" in text
    assert [t for t in _labels(markup) if t in ("◀", "▶")] == ["◀"]


async def test_user_list_role_filter_and_context_limit_mark(repos: Repos) -> None:
    await repos.users.upsert_seen(1, "a", default_role=Role.PENDING, default_mode=Mode.NORMAL)
    await repos.users.upsert_seen(2, "b", default_role=Role.USER, default_mode=Mode.NORMAL)
    await repos.users.set_context_limit(2, 64_000)

    text, _ = await user_list_screen(Role.PENDING, 0, repos.users)
    assert "Заявки на доступ" in text and "(1)" in text

    _, markup = await user_list_screen(None, 0, repos.users)
    assert any(f"🧠 {short_number(64_000)}" in label for label in _labels(markup))


async def test_user_list_empty_messages(repos: Repos) -> None:
    text, _ = await user_list_screen(Role.PENDING, 0, repos.users)
    assert text == "Заявок нет."
    text, _ = await user_list_screen(None, 0, repos.users)
    assert text == "Никого нет."


# ---- context limit screen ------------------------------------------------------------------------


async def test_limit_screen_marks_personal_preset(repos: Repos) -> None:
    await repos.users.upsert_seen(1, "a", default_role=Role.USER, default_mode=Mode.NORMAL)
    await repos.users.set_context_limit(1, 64_000)

    result = await limit_screen(1, repos.users, FakeSettings(history_token_budget=350_000))
    assert result is not None
    text, markup = result
    assert "64 000 ток. (личный)" in text and "350 000" in text
    labels = _labels(markup)
    assert "✅ 64k" in labels
    assert not any(label.startswith("✅") and "Общий" in label for label in labels)


async def test_limit_screen_marks_global_when_no_personal_limit(repos: Repos) -> None:
    await repos.users.upsert_seen(2, "b", default_role=Role.USER, default_mode=Mode.NORMAL)

    result = await limit_screen(2, repos.users, FakeSettings(history_token_budget=350_000))
    assert result is not None
    _, markup = result
    labels = _labels(markup)
    assert any(label.startswith("✅") and "Общий" in label for label in labels)
    assert any(short_number(350_000) in label for label in labels)


async def test_limit_screen_unknown_user_is_none(repos: Repos) -> None:
    assert await limit_screen(999, repos.users, FakeSettings()) is None


# ---- parse_count -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [("100", 100), ("  100  ", 100), ("1 000", 1_000), ("1_000", 1_000), ("5 000", 5_000)],
)
def test_parse_count_accepts_plain_integers(text: str, expected: int) -> None:
    assert parse_count(text) == expected


@pytest.mark.parametrize("text", ["", "  ", "abc", "-5", "0", "50k", "50к", "1.5", "1,5", "1m", "1e3"])
def test_parse_count_rejects_suffixes_and_non_numbers(text: str) -> None:
    """A "50k" shortcut must not silently become a 50 000-request cap for the admin."""
    assert parse_count(text) is None


# ---- daily limit screen ----------------------------------------------------------------------------


async def test_daily_limit_screen_shows_usage_and_marks_personal_preset(repos: Repos) -> None:
    await repos.users.upsert_seen(1, "a", default_role=Role.USER, default_mode=Mode.NORMAL)
    await repos.users.set_daily_limit(1, 100)
    await repos.daily.increment(1, 10)

    result = await daily_limit_screen(1, repos.users, FakeSettings(), repos.daily)
    assert result is not None
    text, markup = result
    assert "100 запросов в день (личный)" in text
    assert "общий — 50" in text
    assert "Использовано сегодня: 1 из 100 · осталось 99" in text
    labels = _labels(markup)
    assert "✅ 100" in labels
    assert not any(label.startswith("✅") and "Общий" in label for label in labels)
    assert "✏️ Своё значение" in labels


async def test_daily_limit_screen_marks_global_without_a_personal_limit(repos: Repos) -> None:
    await repos.users.upsert_seen(2, "b", default_role=Role.USER, default_mode=Mode.NORMAL)

    result = await daily_limit_screen(2, repos.users, FakeSettings(daily_request_limit=50), repos.daily)
    assert result is not None
    text, markup = result
    assert "50 запросов в день (общий)" in text
    assert "Использовано сегодня: 0 из 50 · осталось 50" in text
    assert any(label.startswith("✅") and "Общий (50)" in label for label in _labels(markup))


async def test_daily_limit_screen_escapes_username_and_survives_overuse(repos: Repos) -> None:
    await repos.users.upsert_seen(3, "a<b>", default_role=Role.USER, default_mode=Mode.NORMAL)
    await repos.users.set_daily_limit(3, 10)
    for _ in range(11):
        await repos.daily.increment(3, 1)

    result = await daily_limit_screen(3, repos.users, FakeSettings(), repos.daily)
    assert result is not None
    text, _ = result
    assert "a&lt;b&gt;" in text
    assert "осталось 0" in text  # over the cap: never a negative remainder


async def test_daily_limit_screen_unknown_user_is_none(repos: Repos) -> None:
    assert await daily_limit_screen(999, repos.users, FakeSettings(), repos.daily) is None


async def test_daily_limit_screen_offers_every_preset(repos: Repos) -> None:
    await repos.users.upsert_seen(4, "d", default_role=Role.USER, default_mode=Mode.NORMAL)
    result = await daily_limit_screen(4, repos.users, FakeSettings(), repos.daily)
    assert result is not None
    _, markup = result
    labels = _labels(markup)
    for value in DAILY_PRESETS:
        assert str(value) in labels
    packed = {b.callback_data for row in markup.inline_keyboard for b in row}
    assert SetDailyLimit(tg_id=4, value=DAILY_PRESETS[0]).pack() in packed


# ---- daily limit handlers --------------------------------------------------------------------------


async def test_cb_daily_limit_opens_the_screen_and_alerts_on_unknown_user(repos: Repos) -> None:
    await repos.users.upsert_seen(5, "e", default_role=Role.USER, default_mode=Mode.NORMAL)
    message = _message()
    callback = _callback(message)

    await cb_daily_limit(callback, DailyLimit(tg_id=5), repos.users, repos.daily, FakeSettings())
    assert message.answers and "Дневной лимит" in message.answers[0][0]

    missing = _callback(_message())
    await cb_daily_limit(missing, DailyLimit(tg_id=999), repos.users, repos.daily, FakeSettings())
    assert missing.notes[-1] == ("Пользователь не найден", True)


async def test_cb_set_daily_limit_preset_persists_the_value(repos: Repos, state: FSMContext) -> None:
    await repos.users.upsert_seen(6, "f", default_role=Role.USER, default_mode=Mode.NORMAL)
    message = _message()
    callback = _callback(message)

    await cb_set_daily_limit(
        callback, SetDailyLimit(tg_id=6, value=250), repos.users, repos.daily, FakeSettings(), state
    )

    user = await repos.users.get(6)
    assert user is not None and user.daily_limit == 250
    assert "250 запросов в день (личный)" in message.answers[0][0]
    assert callback.notes[-1] == ("Дневной лимит: 250", False)


async def test_cb_set_daily_limit_default_clears_the_personal_value(repos: Repos, state: FSMContext) -> None:
    await repos.users.upsert_seen(7, "g", default_role=Role.USER, default_mode=Mode.NORMAL)
    await repos.users.set_daily_limit(7, 100)
    message = _message()

    await cb_set_daily_limit(
        _callback(message),
        SetDailyLimit(tg_id=7, value=LIMIT_DEFAULT),
        repos.users,
        repos.daily,
        FakeSettings(),
        state,
    )

    user = await repos.users.get(7)
    assert user is not None and user.daily_limit is None
    assert "50 запросов в день (общий)" in message.answers[0][0]


@pytest.mark.parametrize("bad", [-5, -2, DAILY_LIMIT_MAX + 1, 10**9])
async def test_cb_set_daily_limit_rejects_out_of_range_values(
    repos: Repos, state: FSMContext, bad: int
) -> None:
    """0 and -1 are the sentinels (global / custom), so out-of-range means everything else."""
    await repos.users.upsert_seen(8, "h", default_role=Role.USER, default_mode=Mode.NORMAL)
    callback = _callback(_message())

    await cb_set_daily_limit(
        callback, SetDailyLimit(tg_id=8, value=bad), repos.users, repos.daily, FakeSettings(), state
    )

    user = await repos.users.get(8)
    assert user is not None and user.daily_limit is None  # nothing was written
    assert callback.notes[-1] == ("Недопустимое значение лимита.", True)


async def test_cb_set_daily_limit_guards_admins(repos: Repos, state: FSMContext) -> None:
    await repos.users.upsert_seen(9, "adm", default_role=Role.ADMIN, default_mode=Mode.NORMAL)
    callback = _callback(_message())

    await cb_set_daily_limit(
        callback,
        SetDailyLimit(tg_id=9, value=100),
        repos.users,
        repos.daily,
        FakeSettings(admin_ids=frozenset({9})),
        state,
    )

    user = await repos.users.get(9)
    assert user is not None and user.daily_limit is None
    assert callback.notes[-1] is not None and callback.notes[-1][1] is True
    assert "ADMIN_IDS" in (callback.notes[-1][0] or "")


async def test_cb_set_daily_limit_custom_asks_for_a_number(repos: Repos, state: FSMContext) -> None:
    await repos.users.upsert_seen(10, "i", default_role=Role.USER, default_mode=Mode.NORMAL)
    message = _message()

    await cb_set_daily_limit(
        _callback(message),
        SetDailyLimit(tg_id=10, value=LIMIT_CUSTOM),
        repos.users,
        repos.daily,
        FakeSettings(),
        state,
    )

    assert await state.get_state() == DailyLimitForm.waiting_for_value.state
    assert await state.get_data() == {"tg_id": 10}
    assert "целое число" in message.answers[0][0]
    user = await repos.users.get(10)
    assert user is not None and user.daily_limit is None


async def test_on_daily_limit_value_clamps_to_min_and_max(repos: Repos, state: FSMContext) -> None:
    await repos.users.upsert_seen(11, "j", default_role=Role.USER, default_mode=Mode.NORMAL)
    await state.set_state(DailyLimitForm.waiting_for_value)
    await state.update_data(tg_id=11)

    low = _message(str(DAILY_LIMIT_MIN - 1))
    await on_daily_limit_value(low, state, repos.users, repos.daily, FakeSettings())
    assert low.answers and "от 1 до 100000" in low.answers[0][0]
    user = await repos.users.get(11)
    assert user is not None and user.daily_limit is None

    high = _message(str(DAILY_LIMIT_MAX + 1))
    await on_daily_limit_value(high, state, repos.users, repos.daily, FakeSettings())
    assert high.answers and "от 1 до 100000" in high.answers[0][0]
    assert await state.get_state() == DailyLimitForm.waiting_for_value.state  # still waiting

    ok = _message(str(DAILY_LIMIT_MAX))
    await on_daily_limit_value(ok, state, repos.users, repos.daily, FakeSettings())
    user = await repos.users.get(11)
    assert user is not None and user.daily_limit == DAILY_LIMIT_MAX
    assert "100000 запросов в день (личный)" in ok.answers[0][0]
    assert await state.get_state() is None


@pytest.mark.parametrize("bad", ["", "abc", "50k", "1.5", "-3"])
async def test_on_daily_limit_value_rejects_suffixes_without_crashing(
    repos: Repos, state: FSMContext, bad: str
) -> None:
    await repos.users.upsert_seen(12, "k", default_role=Role.USER, default_mode=Mode.NORMAL)
    await state.set_state(DailyLimitForm.waiting_for_value)
    await state.update_data(tg_id=12)
    message = _message(bad)

    await on_daily_limit_value(message, state, repos.users, repos.daily, FakeSettings())

    user = await repos.users.get(12)
    assert user is not None and user.daily_limit is None
    assert message.answers and "не работают" in message.answers[0][0]
    assert await state.get_state() == DailyLimitForm.waiting_for_value.state


# ---- parse_hours ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [("12", 12.0), (" 6 ", 6.0), ("0", 0.0), ("0.5", 0.5), ("6,5", 6.5), ("168", 168.0)],
)
def test_parse_hours_accepts_integers_and_fractions(text: str, expected: float) -> None:
    assert parse_hours(text) == expected


@pytest.mark.parametrize("text", ["", "abc", "-5", "12ч", "1e3x", "12k"])
def test_parse_hours_rejects_non_numbers(text: str) -> None:
    assert parse_hours(text) is None


# ---- context idle screen ---------------------------------------------------------------------------


async def test_context_idle_screen_marks_personal_preset(repos: Repos) -> None:
    await repos.users.upsert_seen(1, "a", default_role=Role.USER, default_mode=Mode.NORMAL)
    await repos.users.set_context_idle_hours(1, 6)

    result = await context_idle_screen(1, repos.users, FakeSettings())
    assert result is not None
    text, markup = result
    assert "6 ч (личный)" in text and "12 ч" in text
    labels = _labels(markup)
    assert "✅ 6 ч" in labels
    assert "0 — выкл" in labels
    assert not any(label.startswith("✅") and "Общий" in label for label in labels)
    assert "✏️ Своё значение" in labels


async def test_context_idle_screen_marks_global_without_personal_value(repos: Repos) -> None:
    await repos.users.upsert_seen(2, "b", default_role=Role.USER, default_mode=Mode.NORMAL)

    result = await context_idle_screen(2, repos.users, FakeSettings())
    assert result is not None
    text, markup = result
    assert "12 ч (общий)" in text
    labels = _labels(markup)
    assert any(label.startswith("✅") and "Общий (12 ч)" in label for label in labels)
    assert "0 — выкл" in labels


async def test_context_idle_screen_marks_personal_zero_as_off(repos: Repos) -> None:
    await repos.users.upsert_seen(3, "c", default_role=Role.USER, default_mode=Mode.NORMAL)
    await repos.users.set_context_idle_hours(3, 0)

    result = await context_idle_screen(3, repos.users, FakeSettings())
    assert result is not None
    text, markup = result
    assert "не сбрасывать (личный)" in text
    assert "✅ 0 — выкл" in _labels(markup)


async def test_context_idle_screen_unknown_user_is_none(repos: Repos) -> None:
    assert await context_idle_screen(999, repos.users, FakeSettings()) is None


async def test_user_card_shows_idle_line_and_button(repos: Repos) -> None:
    await repos.users.upsert_seen(4, "d", default_role=Role.USER, default_mode=Mode.NORMAL)
    now = datetime.now(UTC)

    result = await user_card_screen(
        4, repos.users, repos.requests, repos.daily, repos.dialogs, FakeSettings(), UTC, now
    )
    assert result is not None
    text, markup = result
    assert "🕒 Сброс контекста: 12 ч (общий)" in text
    assert "🕒 Сброс контекста" in _labels(markup)


# ---- context idle handlers -------------------------------------------------------------------------


async def test_cb_context_idle_opens_the_screen_and_alerts_on_unknown_user(repos: Repos) -> None:
    await repos.users.upsert_seen(5, "e", default_role=Role.USER, default_mode=Mode.NORMAL)
    message = _message()
    callback = _callback(message)

    await cb_context_idle(callback, ContextIdle(tg_id=5), repos.users, FakeSettings())
    assert message.answers and "Сброс контекста" in message.answers[0][0]

    missing = _callback(_message())
    await cb_context_idle(missing, ContextIdle(tg_id=999), repos.users, FakeSettings())
    assert missing.notes[-1] == ("Пользователь не найден", True)


async def test_cb_set_context_idle_preset_persists_the_value(repos: Repos, state: FSMContext) -> None:
    await repos.users.upsert_seen(6, "f", default_role=Role.USER, default_mode=Mode.NORMAL)
    message = _message()
    callback = _callback(message)

    await cb_set_context_idle(callback, SetContextIdle(tg_id=6, value=24), repos.users, FakeSettings(), state)

    user = await repos.users.get(6)
    assert user is not None and user.context_idle_hours == 24
    assert "24 ч (личный)" in message.answers[0][0]
    assert callback.notes[-1] == ("Сброс контекста: 24 ч", False)


async def test_cb_set_context_idle_zero_persists_off(repos: Repos, state: FSMContext) -> None:
    await repos.users.upsert_seen(7, "g", default_role=Role.USER, default_mode=Mode.NORMAL)
    message = _message()
    callback = _callback(message)

    await cb_set_context_idle(callback, SetContextIdle(tg_id=7, value=0), repos.users, FakeSettings(), state)

    user = await repos.users.get(7)
    assert user is not None and user.context_idle_hours == 0
    assert callback.notes[-1] == ("Сброс контекста: не сбрасывать", False)


async def test_cb_set_context_idle_default_clears_the_personal_value(repos: Repos, state: FSMContext) -> None:
    await repos.users.upsert_seen(8, "h", default_role=Role.USER, default_mode=Mode.NORMAL)
    await repos.users.set_context_idle_hours(8, 6)
    message = _message()
    callback = _callback(message)

    await cb_set_context_idle(
        callback, SetContextIdle(tg_id=8, value=IDLE_DEFAULT), repos.users, FakeSettings(), state
    )

    user = await repos.users.get(8)
    assert user is not None and user.context_idle_hours is None
    assert "12 ч (общий)" in message.answers[0][0]
    assert callback.notes[-1] == ("Сброс контекста: общий", False)


async def test_cb_set_context_idle_custom_asks_for_a_number(repos: Repos, state: FSMContext) -> None:
    await repos.users.upsert_seen(9, "i", default_role=Role.USER, default_mode=Mode.NORMAL)
    message = _message()

    await cb_set_context_idle(
        _callback(message), SetContextIdle(tg_id=9, value=IDLE_CUSTOM), repos.users, FakeSettings(), state
    )

    assert await state.get_state() == ContextIdleForm.waiting_for_value.state
    assert await state.get_data() == {"tg_id": 9}
    assert "число от 0 до 168" in message.answers[0][0]
    user = await repos.users.get(9)
    assert user is not None and user.context_idle_hours is None


async def test_on_context_idle_value_accepts_fractions_and_zero(repos: Repos, state: FSMContext) -> None:
    await repos.users.upsert_seen(10, "j", default_role=Role.USER, default_mode=Mode.NORMAL)
    await state.set_state(ContextIdleForm.waiting_for_value)
    await state.update_data(tg_id=10)

    ok = _message("0.5")
    await on_context_idle_value(ok, state, repos.users, FakeSettings())
    user = await repos.users.get(10)
    assert user is not None and user.context_idle_hours == 0.5
    assert "0.5 ч (личный)" in ok.answers[0][0]
    assert await state.get_state() is None


async def test_on_context_idle_value_rejects_garbage(repos: Repos, state: FSMContext) -> None:
    await repos.users.upsert_seen(11, "k", default_role=Role.USER, default_mode=Mode.NORMAL)
    await state.set_state(ContextIdleForm.waiting_for_value)
    await state.update_data(tg_id=11)

    for bad in ("", "abc", "-3", "50k", "200"):
        message = _message(bad)
        await on_context_idle_value(message, state, repos.users, FakeSettings())
        assert message.answers and "от 0 до 168" in message.answers[0][0]
    user = await repos.users.get(11)
    assert user is not None and user.context_idle_hours is None
    assert await state.get_state() == ContextIdleForm.waiting_for_value.state


# ---- SetRole guard --------------------------------------------------------------------------------


def test_set_role_guard_protects_admin_ids_and_rejects_other_roles() -> None:
    assert set_role_guard(1, Role.USER, frozenset({1})) is not None
    assert set_role_guard(2, Role.USER, frozenset({1})) is None
    assert set_role_guard(2, Role.PENDING, frozenset()) is not None


# ---- leaderboard ----------------------------------------------------------------------------------


async def test_top_screen_marks_current_order_and_falls_back(repos: Repos) -> None:
    await repos.users.upsert_seen(1, "a", default_role=Role.USER, default_mode=Mode.NORMAL)
    await repos.requests.add(RequestRecord(1, None, "m", RequestStatus.OK, 1, 10, 100))
    now = datetime.now(UTC)

    text, markup = await top_screen("tokens", repos.requests, now)
    assert "Токены" in text
    labels = _labels(markup)
    assert "✅ Токены" in labels
    assert "Запросы" in labels and "Активность" in labels

    text, _ = await top_screen("bogus-order", repos.requests, now)  # unknown -> requests
    assert "Запросы" in text


async def test_top_screen_empty() -> None:
    db = Database(":memory:")
    await db.connect()
    text, _ = await top_screen("requests", RequestRepo(db), datetime.now(UTC))
    assert text == "Пользователей пока нет."
    await db.close()
