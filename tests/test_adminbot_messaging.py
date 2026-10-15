"""adminbot.messaging: the escaping/clipping helpers, the rendered screens, and both send flows.

No network. The tests use real aiogram types where the handlers care about them — `Message` and
`CallbackQuery` subclasses (so `isinstance` checks in the handlers behave exactly as in production)
and a real `FSMContext` over `MemoryStorage` (the brief is that the text must survive in the FSM
between the prompt and the confirmation). Only the main bot is a fake: it records what it was asked
to send and can refuse a chat id, which is how a user who blocked the bot behaves.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any

import pytest
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.base import StorageKey
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import CallbackQuery, Message
from pydantic import Field

from adminbot.callbacks import Broadcast, BroadcastTo, Home, SendConfirm, UserCard, WriteTo
from adminbot.common import AUDIENCE_LABELS, Audience
from adminbot.messaging import (
    MAX_TEXT,
    PERSONAL_TITLE,
    RATE_PAUSE_S,
    DeliveryReport,
    Kind,
    MessageForm,
    audience_screen,
    broadcast,
    cancelled_screen,
    cb_broadcast,
    cb_broadcast_to,
    cb_send_confirm,
    cb_write_to,
    empty_audience_screen,
    escape_and_clip,
    on_message_text,
    render_broadcast_result,
    render_confirm,
    render_personal_result,
)
from bot.domain import Mode, Role
from bot.storage.db import Database
from bot.storage.repositories import UserRepo

_BAD_AMP = re.compile(r"&(?!amp;|lt;|gt;|quot;|#39;)")  # a lone '&' means an entity got cut


# ---- fakes ----------------------------------------------------------------------------------------


@dataclass
class FakeBot:
    """The part of the main bot we use: send_message, refusing every chat id in `blocked`."""

    blocked: set[int] = field(default_factory=set)
    sent: list[tuple[int, str]] = field(default_factory=list)

    async def send_message(self, chat_id: int, text: str) -> object:
        if chat_id in self.blocked:
            raise RuntimeError("bot was blocked by the user")
        self.sent.append((chat_id, text))
        return object()


class FakeMessage(Message):
    """A real Message (so `isinstance(callback.message, Message)` passes) that records its answers."""

    answers: list[tuple[str, Any]] = Field(default_factory=list)

    async def answer(self, text: str, reply_markup: Any = None, **kwargs: Any) -> None:
        self.answers.append((text, reply_markup))

    async def edit_text(self, text: str, reply_markup: Any = None, **kwargs: Any) -> None:
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


@pytest.fixture
async def users() -> UserRepo:
    db = Database(":memory:")
    await db.connect()
    try:
        yield UserRepo(db)
    finally:
        await db.close()


@pytest.fixture
def state() -> FSMContext:
    return FSMContext(storage=MemoryStorage(), key=StorageKey(bot_id=1, chat_id=1, user_id=1))


async def _seed(repo: UserRepo, tg_id: int, role: Role, username: str | None = None) -> None:
    await repo.upsert_seen(tg_id, username, default_role=role, default_mode=Mode.NORMAL)


# ---- escape_and_clip (pure) ------------------------------------------------------------------------


def test_escape_and_clip_escapes_html() -> None:
    body, clipped = escape_and_clip("<b>привет</b> & <script>")
    assert body == "&lt;b&gt;привет&lt;/b&gt; &amp; &lt;script&gt;"
    assert clipped is False


def test_escape_and_clip_strips_outer_whitespace_only() -> None:
    body, _ = escape_and_clip("  Привет!\nВторая строка.  ")
    assert body == "Привет!\nВторая строка."


def test_escape_and_clip_empty() -> None:
    body, clipped = escape_and_clip("   ")
    assert body == ""
    assert clipped is False


def test_escape_and_clip_short_text_untouched() -> None:
    body, clipped = escape_and_clip("x" * MAX_TEXT)
    assert body == "x" * MAX_TEXT
    assert clipped is False


def test_escape_and_clip_cuts_long_text_and_never_leaves_a_broken_entity() -> None:
    body, clipped = escape_and_clip("&" * 5000)
    assert clipped is True
    assert body.endswith("…")
    assert not _BAD_AMP.search(body)
    assert len(body) <= MAX_TEXT + 1


def test_escape_and_clip_escapes_before_cutting() -> None:
    """Escaping first is what keeps "<" from turning into markup; cutting may only shorten it."""
    body, clipped = escape_and_clip("<" * 1000 + "у" * 10)
    assert clipped is True
    assert body.count("&lt;") == MAX_TEXT // 4  # 1000 "<" escaped, cut at the 3500-character cap
    assert "<" not in body  # nothing left that Telegram would read as markup
    assert not _BAD_AMP.search(body)


def test_escape_and_clip_drops_a_partial_entity_at_the_cut() -> None:
    body, clipped = escape_and_clip("a" * (MAX_TEXT - 4) + "&")
    assert clipped is True
    assert body == "a" * (MAX_TEXT - 4) + "…"  # "&amp;" was started but could not fit whole


# ---- rendered screens (pure) -----------------------------------------------------------------------


def test_render_confirm_shows_subject_and_exact_text() -> None:
    text = render_confirm(PERSONAL_TITLE, "@gleb", "Привет &lt;b&gt;")
    assert "@gleb" in text
    assert "Привет &lt;b&gt;" in text
    assert "⚠️" not in text


def test_render_confirm_warns_about_clipping() -> None:
    text = render_confirm(PERSONAL_TITLE, "всем", "x", clipped=True)
    assert f"обрезан до {MAX_TEXT}" in text


def test_render_personal_result_success_and_failure() -> None:
    assert "✅ Отправлено." in render_personal_result("@gleb", True)
    failed = render_personal_result("@gleb", False)
    assert "❌" in failed
    assert "@gleb" in failed
    assert "заблокировал бота" in failed


def test_render_broadcast_result_full_partial_none_and_empty() -> None:
    full = render_broadcast_result(Audience.ACTIVE, DeliveryReport(total=5, delivered=5))
    assert "✅ Доставлено всем: 5." in full

    partial = render_broadcast_result(Audience.ACTIVE, DeliveryReport(total=5, delivered=3))
    assert "Доставлено: 3 из 5" in partial
    assert "Не дошло: 2" in partial

    nothing = render_broadcast_result(Audience.PENDING, DeliveryReport(total=2, delivered=0))
    assert "Не дошло ни одно сообщение из 2" in nothing

    empty = render_broadcast_result(Audience.ALL, DeliveryReport(total=0, delivered=0))
    assert "Отправлять некому" in empty


def test_empty_audience_screen_offers_way_back() -> None:
    text, markup = empty_audience_screen(Audience.PENDING)
    assert "Отправлять некому" in text
    assert AUDIENCE_LABELS[Audience.PENDING] in text
    labels = [b.text for row in markup.inline_keyboard for b in row]
    assert "« К аудиториям" in labels
    assert "« В меню" in labels


def test_cancelled_screen_points_back_to_the_recipient_or_the_menu() -> None:
    text, markup = cancelled_screen(7)
    assert "Отменено" in text
    labels = [b.text for row in markup.inline_keyboard for b in row]
    assert "« К пользователю" in labels
    assert "« В меню" in labels

    _, markup = cancelled_screen(None)
    labels = [b.text for row in markup.inline_keyboard for b in row]
    assert "« К пользователю" not in labels


# ---- audience screen (DB-backed) --------------------------------------------------------------------


async def test_audience_screen_counts_each_audience(users: UserRepo) -> None:
    await _seed(users, 1, Role.USER, "a")
    await _seed(users, 2, Role.ADMIN, "b")
    await _seed(users, 3, Role.PENDING, "c")
    await _seed(users, 4, Role.BLOCKED, "d")

    text, markup = await audience_screen(users)
    assert f"{AUDIENCE_LABELS[Audience.ACTIVE]}: 2" in text
    assert f"{AUDIENCE_LABELS[Audience.PENDING]}: 1" in text
    assert f"{AUDIENCE_LABELS[Audience.ALL]}: 3" in text  # everyone except the blocked

    buttons = [b for row in markup.inline_keyboard for b in row]
    assert f"{AUDIENCE_LABELS[Audience.ACTIVE]} (2)" in [b.text for b in buttons]
    unpacked = [BroadcastTo.unpack(b.callback_data or "") for b in buttons[:-1]]
    assert unpacked == [BroadcastTo(audience=a) for a in Audience]
    assert Home.unpack(buttons[-1].callback_data or "") == Home()


async def test_audience_screen_with_nobody(users: UserRepo) -> None:
    text, markup = await audience_screen(users)
    assert "Подходящих пользователей пока нет." in text
    assert len(markup.inline_keyboard) == len(Audience) + 1  # one audience a row, then «В меню»


# ---- broadcast sending ------------------------------------------------------------------------------


async def test_broadcast_reaches_everyone_and_returns_the_counts(users: UserRepo) -> None:
    await _seed(users, 1, Role.USER)
    await _seed(users, 2, Role.ADMIN)
    await _seed(users, 3, Role.PENDING)  # not in the ACTIVE audience
    await _seed(users, 4, Role.BLOCKED)
    bot = FakeBot()

    report = await broadcast(bot, users, Audience.ACTIVE, "Привет")
    assert (report.total, report.delivered, report.failed) == (2, 2, 0)
    assert [chat_id for chat_id, _ in bot.sent] == [1, 2]
    assert all(text == "Привет" for _, text in bot.sent)


async def test_broadcast_counts_a_blocked_recipient_and_keeps_going(users: UserRepo) -> None:
    await _seed(users, 1, Role.USER)
    await _seed(users, 2, Role.USER)
    await _seed(users, 3, Role.USER)
    bot = FakeBot(blocked={2})  # the second user blocked the bot

    report = await broadcast(bot, users, Audience.ACTIVE, "Привет")
    assert (report.total, report.delivered, report.failed) == (3, 2, 1)
    assert [chat_id for chat_id, _ in bot.sent] == [1, 3]  # the failure did not abort the rest


async def test_broadcast_empty_audience_sends_nothing(users: UserRepo) -> None:
    bot = FakeBot()
    report = await broadcast(bot, users, Audience.ALL, "Привет")
    assert (report.total, report.delivered, report.failed) == (0, 0, 0)
    assert bot.sent == []


async def test_broadcast_pauses_between_messages_but_not_before_the_first(
    users: UserRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    for tg_id in range(1, 4):
        await _seed(users, tg_id, Role.USER)
    pauses: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        pauses.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    await broadcast(FakeBot(), users, Audience.ACTIVE, "Привет")
    assert pauses == [RATE_PAUSE_S, RATE_PAUSE_S]  # one pause fewer than there are recipients
    assert 1 / RATE_PAUSE_S <= 30  # stays under Telegram's ~30 messages a second


async def test_broadcast_does_not_pause_at_all_for_a_single_recipient(
    users: UserRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _seed(users, 1, Role.USER)
    pauses: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        pauses.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    await broadcast(FakeBot(), users, Audience.ACTIVE, "Привет")
    assert pauses == []


# ---- handlers: personal message ----------------------------------------------------------------------


async def test_cb_write_to_asks_for_the_text_and_remembers_the_user(
    users: UserRepo, state: FSMContext
) -> None:
    await _seed(users, 5, Role.USER, "gleb")
    message = _message()
    callback = _callback(message)

    await cb_write_to(callback, WriteTo(tg_id=5), users, state)

    assert await state.get_state() == MessageForm.waiting_for_text.state
    assert await state.get_data() == {"kind": Kind.USER, "tg_id": 5}
    assert "@gleb" in message.answers[0][0]
    assert callback.notes[-1] == (None, False)


async def test_cb_write_to_unknown_user_alerts_and_keeps_no_state(users: UserRepo, state: FSMContext) -> None:
    callback = _callback(_message())

    await cb_write_to(callback, WriteTo(tg_id=999), users, state)

    assert await state.get_state() is None
    assert callback.notes[-1] == ("Пользователь не найден", True)
    assert callback.message is not None and callback.message.answers == []  # type: ignore[attr-defined]


async def test_personal_flow_asks_confirms_and_sends_via_the_main_bot(
    users: UserRepo, state: FSMContext
) -> None:
    await _seed(users, 5, Role.USER, "gleb")
    await state.set_state(MessageForm.waiting_for_text)
    await state.update_data(kind=Kind.USER, tg_id=5)
    message = _message("Привет, <b>друг</b>")

    await on_message_text(message, state, users)

    confirm_text, markup = message.answers[0]
    assert "Привет, &lt;b&gt;друг&lt;/b&gt;" in confirm_text
    assert "<b>друг</b>" not in confirm_text  # the admin's own text is never raw HTML
    assert markup is not None
    labels = [b.text for row in markup.inline_keyboard for b in row]
    assert labels == ["✅ Отправить", "❌ Отмена"]
    assert (await state.get_data())["body"] == "Привет, &lt;b&gt;друг&lt;/b&gt;"
    assert await state.get_state() == MessageForm.waiting_for_text.state  # still waiting for the tap

    bot = FakeBot()
    callback = _callback(_message())
    await cb_send_confirm(callback, SendConfirm(ok=True), state, users, bot)

    assert bot.sent == [(5, "Привет, &lt;b&gt;друг&lt;/b&gt;")]  # same text the admin confirmed
    assert callback.message is not None
    assert "✅ Отправлено." in callback.message.answers[0][0]  # type: ignore[attr-defined]
    assert await state.get_state() is None
    assert await state.get_data() == {}


async def test_personal_send_reports_a_blocked_user(users: UserRepo, state: FSMContext) -> None:
    await _seed(users, 5, Role.USER, "gleb")
    await state.update_data(kind=Kind.USER, tg_id=5, body="Привет")
    bot = FakeBot(blocked={5})
    callback = _callback(_message())

    await cb_send_confirm(callback, SendConfirm(ok=True), state, users, bot)

    assert bot.sent == []
    assert callback.message is not None
    text, _ = callback.message.answers[0]  # type: ignore[attr-defined]
    assert "❌" in text and "заблокировал бота" in text
    assert callback.notes[-1] == ("Не дошло", True)  # an alert, not a toast: this is a real failure
    assert await state.get_data() == {}


async def test_personal_send_to_a_deleted_user_does_not_crash(users: UserRepo, state: FSMContext) -> None:
    await state.update_data(kind=Kind.USER, tg_id=999, body="Привет")
    callback = _callback(_message())

    await cb_send_confirm(callback, SendConfirm(ok=True), state, users, FakeBot())

    assert callback.notes[-1] == ("Пользователь не найден", True)
    assert callback.message is not None and callback.message.answers == []  # type: ignore[attr-defined]
    assert await state.get_data() == {}


async def test_cancel_sends_nothing_and_clears_the_state(users: UserRepo, state: FSMContext) -> None:
    await _seed(users, 5, Role.USER)
    await state.update_data(kind=Kind.USER, tg_id=5, body="Привет")
    bot = FakeBot()
    callback = _callback(_message())

    await cb_send_confirm(callback, SendConfirm(ok=False), state, users, bot)

    assert bot.sent == []
    assert await state.get_data() == {}
    assert await state.get_state() is None
    assert callback.message is not None
    text, markup = callback.message.answers[0]  # type: ignore[attr-defined]
    assert "Отменено" in text
    assert markup is not None
    assert any(b.text == "« К пользователю" for row in markup.inline_keyboard for b in row)
    assert callback.notes[-1] == ("Отменено", False)


async def test_confirm_with_nothing_waiting_does_not_send(users: UserRepo, state: FSMContext) -> None:
    """A double tap on «Отправить», or a button from an old screen, must not resend anything."""
    bot = FakeBot()
    callback = _callback(_message())

    await cb_send_confirm(callback, SendConfirm(ok=True), state, users, bot)

    assert bot.sent == []
    assert callback.notes[-1] == ("Сообщение уже отправлено или отменено.", True)
    assert callback.message is not None and callback.message.answers == []  # type: ignore[attr-defined]


async def test_a_second_confirm_tap_sends_the_message_only_once(users: UserRepo, state: FSMContext) -> None:
    await _seed(users, 5, Role.USER)
    await state.update_data(kind=Kind.USER, tg_id=5, body="Привет")
    bot = FakeBot()

    await cb_send_confirm(_callback(_message()), SendConfirm(ok=True), state, users, bot)
    await cb_send_confirm(_callback(_message()), SendConfirm(ok=True), state, users, bot)

    assert len(bot.sent) == 1


async def test_stale_state_is_reported_instead_of_sending(users: UserRepo, state: FSMContext) -> None:
    await state.update_data(body="Привет")  # no kind: the state outlived its prompt
    message = _message("Привет")

    await on_message_text(message, state, users)

    assert "устарел" in message.answers[0][0]
    assert await state.get_state() is None
    assert await state.get_data() == {}


async def test_personal_text_with_no_text_asks_again(users: UserRepo, state: FSMContext) -> None:
    await _seed(users, 5, Role.USER)
    await state.set_state(MessageForm.waiting_for_text)
    await state.update_data(kind=Kind.USER, tg_id=5)
    message = _message("   ")

    await on_message_text(message, state, users)

    assert "Нужен текст" in message.answers[0][0]
    assert await state.get_state() == MessageForm.waiting_for_text.state  # still waiting
    assert "body" not in await state.get_data()


async def test_personal_text_for_a_deleted_user_clears_and_explains(
    users: UserRepo, state: FSMContext
) -> None:
    await state.update_data(kind=Kind.USER, tg_id=999)
    message = _message("Привет")

    await on_message_text(message, state, users)

    assert message.answers[0][0] == "Пользователь не найден."
    assert await state.get_state() is None


# ---- handlers: broadcast ----------------------------------------------------------------------------


async def test_cb_broadcast_shows_the_audiences(users: UserRepo) -> None:
    await _seed(users, 1, Role.USER)
    callback = _callback(_message())

    await cb_broadcast(callback, users)

    assert callback.message is not None
    text, markup = callback.message.answers[0]  # type: ignore[attr-defined]
    assert AUDIENCE_LABELS[Audience.ACTIVE] in text
    assert markup is not None
    labels = [b.text for row in markup.inline_keyboard for b in row]
    assert any(label.startswith(AUDIENCE_LABELS[Audience.PENDING]) for label in labels)
    assert "« В меню" in labels


async def test_cb_broadcast_to_remembers_the_audience(users: UserRepo, state: FSMContext) -> None:
    await _seed(users, 1, Role.PENDING)
    message = _message()
    callback = _callback(message)

    await cb_broadcast_to(callback, BroadcastTo(audience=Audience.PENDING), users, state)

    assert await state.get_state() == MessageForm.waiting_for_text.state
    assert await state.get_data() == {"kind": Kind.BROADCAST, "audience": Audience.PENDING}
    assert "(1 чел.)" in message.answers[0][0]


async def test_cb_broadcast_to_empty_audience_asks_for_no_text(users: UserRepo, state: FSMContext) -> None:
    callback = _callback(_message())

    await cb_broadcast_to(callback, BroadcastTo(audience=Audience.PENDING), users, state)

    assert await state.get_state() is None
    assert callback.notes[-1] == ("Некому отправлять", True)
    assert callback.message is not None
    text, markup = callback.message.answers[0]  # type: ignore[attr-defined]
    assert "Отправлять некому" in text
    assert markup is not None
    assert any(b.text == "« К аудиториям" for row in markup.inline_keyboard for b in row)


async def test_broadcast_flow_confirms_then_sends_to_everyone(users: UserRepo, state: FSMContext) -> None:
    await _seed(users, 1, Role.USER, "a")
    await _seed(users, 2, Role.USER, "b")
    await _seed(users, 3, Role.BLOCKED, "c")  # outside the ACTIVE audience
    await state.set_state(MessageForm.waiting_for_text)
    await state.update_data(kind=Kind.BROADCAST, audience=Audience.ACTIVE)
    message = _message("Большое объявление")

    await on_message_text(message, state, users)

    confirm_text, _ = message.answers[0]
    assert AUDIENCE_LABELS[Audience.ACTIVE] in confirm_text
    assert "получателей: 2" in confirm_text
    assert "Большое объявление" in confirm_text

    bot = FakeBot(blocked={2})
    callback = _callback(_message())
    await cb_send_confirm(callback, SendConfirm(ok=True), state, users, bot)

    assert [chat_id for chat_id, _ in bot.sent] == [1]  # 2 refused, 1 delivered, no crash
    assert callback.message is not None
    text = callback.message.answers[0][0]  # type: ignore[attr-defined]
    assert "Доставлено: 1 из 2" in text
    assert "Не дошло: 1" in text
    assert await state.get_data() == {}


async def test_broadcast_to_an_audience_emptied_while_typing(users: UserRepo, state: FSMContext) -> None:
    await state.set_state(MessageForm.waiting_for_text)
    await state.update_data(kind=Kind.BROADCAST, audience=Audience.PENDING)
    message = _message("Привет")

    await on_message_text(message, state, users)

    text, markup = message.answers[0]
    assert "Отправлять некому" in text
    assert markup is not None
    assert any(b.text == "« К аудиториям" for row in markup.inline_keyboard for b in row)
    assert await state.get_state() is None


async def test_broadcast_send_with_an_audience_that_emptied_is_a_clear_message(
    users: UserRepo, state: FSMContext
) -> None:
    await state.update_data(kind=Kind.BROADCAST, audience=Audience.ALL, body="Привет")
    bot = FakeBot()
    callback = _callback(_message())

    await cb_send_confirm(callback, SendConfirm(ok=True), state, users, bot)

    assert bot.sent == []
    assert callback.notes[-1] == ("Некому отправлять", True)
    assert callback.message is not None
    assert "Отправлять некому" in callback.message.answers[0][0]  # type: ignore[attr-defined]


# ---- wiring -----------------------------------------------------------------------------------------


def test_callback_data_classes_round_trip_for_the_buttons_other_modules_add() -> None:
    """The buttons added in users.py / home.py must unpack into the handlers defined here."""
    assert WriteTo.unpack(WriteTo(tg_id=1).pack()) == WriteTo(tg_id=1)
    assert Broadcast.unpack(Broadcast().pack()) == Broadcast()
    assert BroadcastTo.unpack(BroadcastTo(audience=Audience.ALL).pack()) == BroadcastTo(audience=Audience.ALL)
    assert SendConfirm.unpack(SendConfirm(ok=True).pack()) == SendConfirm(ok=True)
    assert UserCard.unpack(UserCard(tg_id=1).pack()) == UserCard(tg_id=1)
