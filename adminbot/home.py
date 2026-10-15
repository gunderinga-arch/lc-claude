"""Home router: /start and Home (main menu), /cancel, Top (leaderboard)."""

from __future__ import annotations

from datetime import UTC, datetime, timezone

from aiogram import Router
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, InlineKeyboardMarkup, Message
from aiogram.utils.keyboard import InlineKeyboardBuilder

from adminbot.callbacks import (
    Activity,
    AddUser,
    Balance,
    Broadcast,
    Dashboard,
    FindHint,
    Health,
    Home,
    Reserve,
    ServerBackup,
    Top,
    Twin,
    UserCard,
    UserList,
)
from adminbot.common import Period, period_since, show
from bot.domain import Role
from bot.llm.reserve import ReserveSwitch
from bot.render.format import ago, short_number, user_label
from bot.storage.repositories import LEADERBOARD_ORDER, RequestRepo, UserRepo
from bot.telegram.keyboards import LEADERBOARD_LABELS

router = Router(name="home")


async def main_menu_screen(
    users: UserRepo,
    requests: RequestRepo,
    now: datetime,
    tz: timezone,
    reserve_switch: ReserveSwitch | None = None,
) -> tuple[str, InlineKeyboardMarkup]:
    """The admin bot's home screen: counters plus every top-level destination.

    `reserve_switch` is optional only so old callers (and tests) that do not care about the
    emergency switch keep working; every real handler below passes it.
    """
    total = await users.count()
    pending = await users.count(Role.PENDING)
    today = await requests.report(None, period_since(Period.TODAY, now, tz))
    text = (
        "🛠 <b>Админ-панель</b>\n"
        f"Пользователей: {total} · ждут одобрения: {pending}\n"
        f"Сегодня: {today.requests} запросов · {short_number(today.total_tokens)} токенов"
    )
    if reserve_switch is not None:
        mode, _ = await reserve_switch.mode()
        if mode == "auto":
            text += (
                "\n\n🆘 <b>Резерв включился сам</b>: основные модели не отвечали. "
                "Выключится автоматически, когда они вернутся."
            )
        elif mode == "on":
            text += "\n\n🆘 <b>Включён аварийный режим</b>: отвечают резервные модели."
    builder = InlineKeyboardBuilder()
    builder.button(text="📊 Сводка", callback_data=Dashboard(period=Period.TODAY))
    builder.button(text="👥 Пользователи", callback_data=UserList(role=None, page=0))
    builder.button(text=f"⏳ Заявки ({pending})", callback_data=UserList(role=Role.PENDING, page=0))
    builder.button(text="🏆 Рейтинг", callback_data=Top(order="requests"))
    builder.button(text="➕ Добавить по ID", callback_data=AddUser())
    builder.button(text="🔎 Поиск", callback_data=FindHint())
    builder.button(text="📣 Рассылка", callback_data=Broadcast())
    builder.button(text="📈 Активность", callback_data=Activity())
    builder.button(text="🩺 Состояние", callback_data=Health())
    builder.button(text="🛟 Резерв и модели", callback_data=Reserve())
    builder.button(text="🪙 Токены на ключах", callback_data=Balance())
    builder.button(text="🧍 Двойник Глеба", callback_data=Twin())
    builder.button(text="💾 Бэкап сервера", callback_data=ServerBackup())
    builder.adjust(2)
    return text, builder.as_markup()


async def top_screen(order: str, requests: RequestRepo, now: datetime) -> tuple[str, InlineKeyboardMarkup]:
    """Leaderboard for one order key; an unknown order quietly falls back to "requests"."""
    if order not in LEADERBOARD_ORDER:
        order = "requests"
    rows = await requests.leaderboard(order, limit=15)
    text = f"🏆 <b>Рейтинг</b>: {LEADERBOARD_LABELS[order]}"
    if not rows:
        text = "Пользователей пока нет."

    builder = InlineKeyboardBuilder()
    row_sizes: list[int] = []
    for place, row in enumerate(rows, 1):
        label = (
            f"{place}. {user_label(row)} · {row.requests} запр. · "
            f"{short_number(row.approx_tokens)} · {ago(row.last_at, now)}"
        )
        builder.button(text=label, callback_data=UserCard(tg_id=row.tg_id))
        row_sizes.append(1)
    for key in LEADERBOARD_LABELS:
        mark = "✅ " if key == order else ""
        builder.button(text=f"{mark}{LEADERBOARD_LABELS[key]}", callback_data=Top(order=key))
    row_sizes.append(3)
    builder.button(text="« В меню", callback_data=Home())
    row_sizes.append(1)
    builder.adjust(*row_sizes)
    return text, builder.as_markup()


@router.message(CommandStart())
async def cmd_start(
    message: Message,
    state: FSMContext,
    users: UserRepo,
    requests: RequestRepo,
    tz: timezone,
    reserve_switch: ReserveSwitch,
) -> None:
    """/start always opens a fresh menu, dropping whatever the admin was in the middle of."""
    await state.clear()
    text, markup = await main_menu_screen(users, requests, datetime.now(UTC), tz, reserve_switch)
    await message.answer(text, reply_markup=markup)


@router.message(Command("cancel"))
async def cmd_cancel(
    message: Message,
    state: FSMContext,
    users: UserRepo,
    requests: RequestRepo,
    tz: timezone,
    reserve_switch: ReserveSwitch,
) -> None:
    """Drop any FSM input (add-by-id, custom limit) and show the menu again."""
    await state.clear()
    text, markup = await main_menu_screen(users, requests, datetime.now(UTC), tz, reserve_switch)
    await message.answer(f"Отменено.\n\n{text}", reply_markup=markup)


@router.callback_query(Home.filter())
async def cb_home(
    callback: CallbackQuery,
    users: UserRepo,
    requests: RequestRepo,
    tz: timezone,
    reserve_switch: ReserveSwitch,
) -> None:
    text, markup = await main_menu_screen(users, requests, datetime.now(UTC), tz, reserve_switch)
    await show(callback, text, markup)


@router.callback_query(Top.filter())
async def cb_top(callback: CallbackQuery, callback_data: Top, requests: RequestRepo) -> None:
    text, markup = await top_screen(callback_data.order, requests, datetime.now(UTC))
    await show(callback, text, markup)
