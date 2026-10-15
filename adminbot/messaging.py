"""Messaging: a personal note to one user, and a broadcast to a whole audience.

Both send through the *main* bot. The admin bot has its own token, but users only ever talk to the
main bot: with the admin token the message would arrive in a chat nobody has opened, so every send
here goes out through the `main_bot` dependency (wired in adminbot.__main__).
"""

from __future__ import annotations

import asyncio
import html
import re
from dataclasses import dataclass
from enum import StrEnum

from aiogram import Bot, Router
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import CallbackQuery, InlineKeyboardMarkup, Message
from aiogram.utils.keyboard import InlineKeyboardBuilder

from adminbot.callbacks import Broadcast, BroadcastTo, Home, SendConfirm, UserCard, WriteTo
from adminbot.common import AUDIENCE_LABELS, AUDIENCE_ROLES, Audience, show
from bot.render.format import user_label
from bot.services.roles import MessageSender, notify_user
from bot.storage.repositories import UserRepo

router = Router(name="messaging")

PERSONAL_TITLE = "✍️ <b>Личное сообщение</b>"
BROADCAST_TITLE = "📣 <b>Рассылка</b>"

# Telegram caps a message at 4096 characters; the confirmation screen adds a header and a footer
# around the same text, so the body is kept comfortably shorter and the screen still fits.
MAX_TEXT = 3500

# Telegram allows a bot roughly 30 messages a second across all chats. 0.05 s is about 20 a second:
# comfortable headroom under the limit, and the pause only sits *between* two sends, so a long
# broadcast is slowed down without delaying its first message.
RATE_PAUSE_S = 0.05

# A cut in the middle of an escaped entity ("&amp;" -> "&am") makes Telegram reject the whole
# message, so such a tail is dropped instead. Entities only ever start with "&" in escaped text.
_PARTIAL_ENTITY = re.compile(r"&[a-zA-Z0-9#]*$")


class Kind(StrEnum):
    """What the text waiting in the FSM data is for: one user or a whole audience."""

    USER = "u"
    BROADCAST = "b"


class MessageForm(StatesGroup):
    waiting_for_text = State()


@dataclass(frozen=True, slots=True)
class DeliveryReport:
    """How a broadcast went; `failed` counts everyone Telegram refused to deliver to."""

    total: int
    delivered: int

    @property
    def failed(self) -> int:
        return self.total - self.delivered


# ---- pure helpers ---------------------------------------------------------------------------------


def escape_and_clip(raw: str, cap: int = MAX_TEXT) -> tuple[str, bool]:
    """Escape the admin's text, then cut it to `cap`: returns (body, was_clipped).

    Escaping comes first because the result is sent with parse_mode=HTML — the admin writes plain
    text, and an unescaped "<" would either break the message or turn it into markup. Cutting the
    escaped string can then land inside an entity, which Telegram rejects outright, so a trailing
    partial entity is dropped as well.
    """
    escaped = html.escape(raw.strip())
    if len(escaped) <= cap:
        return escaped, False
    cut = escaped[:cap]
    partial = _PARTIAL_ENTITY.search(cut)
    if partial is not None:
        cut = cut[: partial.start()]
    return cut.rstrip() + "…", True


def render_confirm(title: str, subject: str, body: str, *, clipped: bool = False) -> str:
    """Pure: exactly what will go out and to whom. `title` and `subject` are already escaped."""
    note = ""
    if clipped:
        note = f"\n\n⚠️ Текст обрезан до {MAX_TEXT} символов — Telegram не принимает длиннее 4096."
    return (
        f"{title} — {subject}\n"
        "Ниже ровно тот текст, который получит пользователь:\n"
        "─────\n"
        f"{body}\n"
        "─────"
        f"{note}"
    )


def render_personal_result(label: str, delivered: bool) -> str:
    """Pure: the outcome of one personal message. `label` is already escaped."""
    head = f"{PERSONAL_TITLE} — {label}"
    if delivered:
        return f"{head}\n✅ Отправлено."
    return f"{head}\n❌ Отправить не удалось: пользователь заблокировал бота или ещё не запускал его."


def render_broadcast_result(audience: Audience, report: DeliveryReport) -> str:
    """Pure: how much of the audience actually got the broadcast, failures included."""
    head = f"{BROADCAST_TITLE} — {AUDIENCE_LABELS[audience]}"
    if report.total == 0:
        return f"{head}\nОтправлять некому: подходящих пользователей нет."
    if report.failed == 0:
        return f"{head}\n✅ Доставлено всем: {report.delivered}."
    reason = "этот пользователь заблокировал бота или ещё не запускал его"
    if report.delivered == 0:
        return f"{head}\n❌ Не дошло ни одно сообщение из {report.total} ({reason})."
    return (
        f"{head}\n✅ Доставлено: {report.delivered} из {report.total}. Не дошло: {report.failed} ({reason})."
    )


def confirm_keyboard() -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    builder.button(text="✅ Отправить", callback_data=SendConfirm(ok=True))
    builder.button(text="❌ Отмена", callback_data=SendConfirm(ok=False))
    builder.adjust(2)
    return builder.as_markup()


def back_keyboard(tg_id: int | None) -> InlineKeyboardMarkup:
    """After a send: back to the user it went to, or to the menu when the send was a broadcast."""
    builder = InlineKeyboardBuilder()
    if tg_id is not None:
        builder.button(text="« К пользователю", callback_data=UserCard(tg_id=tg_id))
    builder.button(text="« В меню", callback_data=Home())
    builder.adjust(1)
    return builder.as_markup()


def cancelled_screen(tg_id: int | None) -> tuple[str, InlineKeyboardMarkup]:
    """Pure: the screen after «Отмена» — nothing was sent, the admin is one tap from where they were."""
    head = PERSONAL_TITLE if tg_id is not None else BROADCAST_TITLE
    return f"{head}\nОтменено, ничего не отправлено.", back_keyboard(tg_id)


def empty_audience_screen(audience: Audience) -> tuple[str, InlineKeyboardMarkup]:
    """Pure: an audience with nobody in it, shown instead of asking for a text nobody would get."""
    builder = InlineKeyboardBuilder()
    builder.button(text="« К аудиториям", callback_data=Broadcast())
    builder.button(text="« В меню", callback_data=Home())
    builder.adjust(1)
    text = (
        f"{BROADCAST_TITLE} — {AUDIENCE_LABELS[audience]}\nОтправлять некому: подходящих пользователей нет."
    )
    return text, builder.as_markup()


async def audience_screen(users: UserRepo) -> tuple[str, InlineKeyboardMarkup]:
    """The broadcast entry point: every audience with the number of people it will reach."""
    counts = {audience: len(await users.ids_by_roles(AUDIENCE_ROLES[audience])) for audience in Audience}
    text = (
        f"{BROADCAST_TITLE}\n"
        "Кому отправить сообщение? Отправляет основной бот — тот, которому пишут пользователи.\n"
        + " · ".join(f"{AUDIENCE_LABELS[a]}: {n}" for a, n in counts.items())
    )
    if not any(counts.values()):
        text += "\nПодходящих пользователей пока нет."

    builder = InlineKeyboardBuilder()
    for audience in Audience:
        builder.button(
            text=f"{AUDIENCE_LABELS[audience]} ({counts[audience]})",
            callback_data=BroadcastTo(audience=audience),
        )
    builder.button(text="« В меню", callback_data=Home())
    builder.adjust(1)
    return text, builder.as_markup()


# ---- sending --------------------------------------------------------------------------------------


async def broadcast(bot: MessageSender, users: UserRepo, audience: Audience, text: str) -> DeliveryReport:
    """Send the text to every user of the audience, one by one, and count who did not get it.

    Sequential with a pause (see RATE_PAUSE_S) instead of a burst: Telegram counts a bot's messages
    globally, so a tight loop over a big audience gets throttled or banned. One refused recipient
    must never stop the rest, and notify_user already swallows that per-recipient failure for us.
    """
    targets = await users.ids_by_roles(AUDIENCE_ROLES[audience])
    delivered = 0
    for index, tg_id in enumerate(targets):
        if index:  # no pause before the first message
            await asyncio.sleep(RATE_PAUSE_S)
        if await notify_user(bot, tg_id, text):
            delivered += 1
    return DeliveryReport(total=len(targets), delivered=delivered)


# ---- personal message -----------------------------------------------------------------------------


@router.callback_query(WriteTo.filter())
async def cb_write_to(
    callback: CallbackQuery, callback_data: WriteTo, users: UserRepo, state: FSMContext
) -> None:
    """The user card's «Написать»: remember the recipient and ask for the text."""
    user = await users.get(callback_data.tg_id)
    if user is None:
        await callback.answer("Пользователь не найден", show_alert=True)
        return
    await state.set_state(MessageForm.waiting_for_text)
    await state.update_data(kind=Kind.USER, tg_id=user.tg_id)
    if isinstance(callback.message, Message):
        await callback.message.answer(
            f"Напишите текст сообщения для {html.escape(user_label(user))} — он уйдёт от основного бота. "
            "Отмена: /cancel"
        )
    await callback.answer()


# ---- broadcast ------------------------------------------------------------------------------------


@router.callback_query(Broadcast.filter())
async def cb_broadcast(callback: CallbackQuery, users: UserRepo) -> None:
    text, markup = await audience_screen(users)
    await show(callback, text, markup)


@router.callback_query(BroadcastTo.filter())
async def cb_broadcast_to(
    callback: CallbackQuery, callback_data: BroadcastTo, users: UserRepo, state: FSMContext
) -> None:
    """One audience picked: remember it and ask for the text, or say there is nobody to write to."""
    audience = callback_data.audience
    targets = await users.ids_by_roles(AUDIENCE_ROLES[audience])
    if not targets:
        text, markup = empty_audience_screen(audience)
        await show(callback, text, markup, note="Некому отправлять", alert=True)
        return
    await state.set_state(MessageForm.waiting_for_text)
    await state.update_data(kind=Kind.BROADCAST, audience=audience)
    if isinstance(callback.message, Message):
        await callback.message.answer(
            f"Напишите текст рассылки для аудитории «{AUDIENCE_LABELS[audience]}» "
            f"({len(targets)} чел.). Отмена: /cancel"
        )
    await callback.answer()


# ---- the text in between, then the confirmation ----------------------------------------------------


@router.message(MessageForm.waiting_for_text)
async def on_message_text(message: Message, state: FSMContext, users: UserRepo) -> None:
    """The text arrived: keep it in the FSM data and show the confirmation screen."""
    data = await state.get_data()
    kind = data.get("kind")
    body, clipped = escape_and_clip(message.text or "")
    if not body:
        await message.answer("Нужен текст сообщения — пришлите его одним сообщением. Или /cancel.")
        return

    if kind == Kind.USER:
        user = await users.get(data.get("tg_id", 0))
        if user is None:
            await state.clear()
            await message.answer("Пользователь не найден.")
            return
        title, subject = PERSONAL_TITLE, html.escape(user_label(user))
    elif kind == Kind.BROADCAST:
        audience = Audience(data["audience"])
        count = len(await users.ids_by_roles(AUDIENCE_ROLES[audience]))
        if count == 0:
            # The audience emptied while the admin was typing: same answer as picking it directly.
            await state.clear()
            text, markup = empty_audience_screen(audience)
            await message.answer(text, reply_markup=markup)
            return
        title, subject = BROADCAST_TITLE, f"{AUDIENCE_LABELS[audience]} · получателей: {count}"
    else:  # the state outlived its prompt (restart, stale /start) — nothing sensible to send
        await state.clear()
        await message.answer("Ввод устарел, начните заново: /start")
        return

    await state.update_data(body=body)
    await message.answer(
        render_confirm(title, subject, body, clipped=clipped), reply_markup=confirm_keyboard()
    )


@router.callback_query(SendConfirm.filter())
async def cb_send_confirm(
    callback: CallbackQuery,
    callback_data: SendConfirm,
    state: FSMContext,
    users: UserRepo,
    main_bot: Bot,
) -> None:
    """✅ sends what the FSM is holding, ❌ drops it. Either way the state does not survive."""
    data = await state.get_data()
    # Cleared before sending: a double tap on the same button must not send the message twice.
    await state.clear()
    if not callback_data.ok:
        text, markup = cancelled_screen(data.get("tg_id"))
        await show(callback, text, markup, note="Отменено")
        return

    kind = data.get("kind")
    body = data.get("body")
    if not kind or not body:
        await callback.answer("Сообщение уже отправлено или отменено.", show_alert=True)
        return

    if kind == Kind.USER:
        tg_id = data.get("tg_id")
        if tg_id is None:
            await callback.answer("Сообщение уже отправлено или отменено.", show_alert=True)
            return
        user = await users.get(tg_id)
        if user is None:
            await callback.answer("Пользователь не найден", show_alert=True)
            return
        delivered = await notify_user(main_bot, tg_id, body)
        text = render_personal_result(html.escape(user_label(user)), delivered)
        note = "Отправлено" if delivered else "Не дошло"
        await show(callback, text, back_keyboard(tg_id), note=note, alert=not delivered)
        return

    audience = data.get("audience")
    if audience is None:
        await callback.answer("Сообщение уже отправлено или отменено.", show_alert=True)
        return
    audience = Audience(audience)
    report = await broadcast(main_bot, users, audience, body)
    if report.total == 0:
        text, markup = empty_audience_screen(audience)
        await show(callback, text, markup, note="Некому отправлять", alert=True)
        return
    await show(callback, render_broadcast_result(audience, report), back_keyboard(None), note="Готово")
