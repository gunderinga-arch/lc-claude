"""Users router: list, search, the user card hub, roles, add-by-id and the context limit editor."""

from __future__ import annotations

import html
from datetime import UTC, datetime, timezone

from aiogram import Bot, F, Router
from aiogram.filters import StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import CallbackQuery, InlineKeyboardMarkup, Message
from aiogram.utils.keyboard import InlineKeyboardBuilder

from adminbot.callbacks import (
    IDLE_CUSTOM,
    IDLE_DEFAULT,
    LIMIT_CUSTOM,
    LIMIT_DEFAULT,
    AddUser,
    ContextIdle,
    DailyLimit,
    Dialogs,
    FindHint,
    Home,
    Limit,
    SetContextIdle,
    SetDailyLimit,
    SetLimit,
    SetRole,
    UserCard,
    UserList,
    UserStats,
    WriteTo,
)
from adminbot.common import Period, fmt_date, fmt_tokens, show
from bot.config import Settings
from bot.domain import (
    CONTEXT_LIMIT_MAX,
    CONTEXT_LIMIT_MIN,
    DAILY_LIMIT_MAX,
    DAILY_LIMIT_MIN,
    Mode,
    Role,
    Style,
)
from bot.render.format import ago, short_number, user_label
from bot.services.roles import GRANTED_TEXT, apply_role, notified_note, notify_user
from bot.storage.repositories import ConversationRepo, RequestRepo, UsageRepo, UserRepo

router = Router(name="users")

PAGE_SIZE = 10

ADMIN_GUARD_TEXT = "Это администратор из ADMIN_IDS — его роль не меняется."
# Daily limits are inert for admins (bot.services.quota gives them no limit at all), so a number set on
# one of them would silently do nothing; refuse it with the same shape of alert as the role guard.
ADMIN_LIMIT_GUARD_TEXT = "Это администратор из ADMIN_IDS — дневной лимит на него не действует."

_LIST_TITLES: dict[Role | None, str] = {
    None: "👥 <b>Пользователи</b>",
    Role.PENDING: "⏳ <b>Заявки на доступ</b>",
    Role.BLOCKED: "🚫 <b>Заблокированные</b>",
    Role.ADMIN: "👑 <b>Админы</b>",
    Role.USER: "✅ <b>С доступом</b>",
}

_FILTERS: tuple[tuple[Role | None, str], ...] = ((None, "Все"), (Role.PENDING, "⏳"), (Role.BLOCKED, "🚫"))

ROLE_RU = {
    Role.ADMIN: "админ",
    Role.USER: "есть доступ",
    Role.PENDING: "ждёт одобрения",
    Role.BLOCKED: "заблокирован",
}
STYLE_RU = {Style.SHORT: "кратко", Style.NORMAL: "обычно", Style.DETAILED: "подробно"}

PRESETS = (8_000, 16_000, 32_000, 64_000, 128_000, 256_000, 512_000, 1_048_576)
PRESET_LABELS = {
    8_000: "8k",
    16_000: "16k",
    32_000: "32k",
    64_000: "64k",
    128_000: "128k",
    256_000: "256k",
    512_000: "512k",
    1_048_576: "1M",
}

# Daily request counts an admin is likely to pick: small caps for one noisy user, big ones to lift the
# global default. Eight of them keep the same keyboard shape as the context-limit screen.
DAILY_PRESETS = (10, 25, 50, 100, 250, 500, 1_000, 5_000)

# Hours of silence after which the dialog history is forgotten. 0 means "never reset" (a real,
# pickable value, unlike the daily-limit screen), so it gets its own button next to the presets.
IDLE_HOURS_MIN = 0
IDLE_HOURS_MAX = 168
IDLE_PRESETS = (2, 6, 12, 24, 48, 168)


def fmt_hours(hours: float) -> str:
    """Whole hours without the ".0", fractions keep their digits: "12", "0.5"."""
    text = f"{hours:g}"
    return text


class AddUserForm(StatesGroup):
    waiting_for_id = State()


class LimitForm(StatesGroup):
    waiting_for_value = State()


class DailyLimitForm(StatesGroup):
    waiting_for_value = State()


class ContextIdleForm(StatesGroup):
    waiting_for_value = State()


def parse_tokens(text: str) -> int | None:
    """A token count typed by hand: plain numbers, thousands spaces, or a k/к, m/м suffix.

    None on anything that isn't a positive number ("", "abc", "-5", "0", "1e3x", ...).
    """
    cleaned = text.strip().lower().replace(" ", "").replace(" ", "").replace("_", "")
    if not cleaned:
        return None
    multiplier = 1
    if cleaned[-1] in "kк":
        multiplier = 1_000
        cleaned = cleaned[:-1]
    elif cleaned[-1] in "mм":
        multiplier = 1_000_000
        cleaned = cleaned[:-1]
    try:
        value = float(cleaned.replace(",", "."))
    except ValueError:
        return None
    result = round(value * multiplier)
    return result if result > 0 else None


def parse_count(text: str) -> int | None:
    """A request count typed by hand: digits, optionally grouped with spaces or underscores.

    Deliberately stricter than parse_tokens: a number of requests is not a token count, so "50k" or
    "1.5" must be an error rather than a silent 50 000 the admin never meant. None unless the whole
    text is a positive integer.
    """
    cleaned = text.strip().replace("\u00a0", "").replace(" ", "").replace("_", "")
    if not cleaned or not cleaned.isdigit():
        return None
    value = int(cleaned)
    return value if value > 0 else None


def parse_hours(text: str) -> float | None:
    """Hours of silence typed by hand: an integer or a fraction ("12", "0.5", "6,5").

    None on anything that isn't a non-negative number ("", "abc", "-5", "12ч", ...). 0 is valid —
    it means "never reset" — and is allowed here; the range check happens in the handler.
    """
    cleaned = text.strip().replace("\u00a0", "").replace(" ", "").replace("_", "")
    if not cleaned:
        return None
    try:
        value = float(cleaned.replace(",", "."))
    except ValueError:
        return None
    return value if value >= 0 and value == value else None  # reject NaN too


def find_hint_screen() -> tuple[str, InlineKeyboardMarkup]:
    text = "Пришлите @username, часть ника или Telegram ID — найду пользователя."
    builder = InlineKeyboardBuilder()
    builder.button(text="« В меню", callback_data=Home())
    return text, builder.as_markup()


async def user_list_screen(role: Role | None, page: int, users: UserRepo) -> tuple[str, InlineKeyboardMarkup]:
    """Paginated user list; role=None is everyone."""
    total = await users.count(role)
    pages = max(1, -(-total // PAGE_SIZE))
    page = max(0, min(page, pages - 1))
    rows = await users.list_page(role, PAGE_SIZE, page * PAGE_SIZE)

    if total == 0:
        text = "Заявок нет." if role is Role.PENDING else "Никого нет."
    else:
        text = f"{_LIST_TITLES[role]} ({total}), стр. {page + 1}/{pages}"

    builder = InlineKeyboardBuilder()
    row_sizes = [len(_FILTERS)]
    for filter_role, label in _FILTERS:
        mark = "✅ " if filter_role == role else ""
        builder.button(text=f"{mark}{label}", callback_data=UserList(role=filter_role, page=0))
    for u in rows:
        mark = f" · 🧠 {short_number(u.context_limit)}" if u.context_limit else ""
        builder.button(text=f"{user_label(u)}{mark}", callback_data=UserCard(tg_id=u.tg_id))
        row_sizes.append(1)
    nav = 0
    if page > 0:
        builder.button(text="◀", callback_data=UserList(role=role, page=page - 1))
        nav += 1
    if page < pages - 1:
        builder.button(text="▶", callback_data=UserList(role=role, page=page + 1))
        nav += 1
    if nav:
        row_sizes.append(nav)
    builder.button(text="« В меню", callback_data=Home())
    row_sizes.append(1)
    builder.adjust(*row_sizes)
    return text, builder.as_markup()


async def user_card_screen(
    tg_id: int,
    users: UserRepo,
    requests: RequestRepo,
    daily: UsageRepo,
    dialogs: ConversationRepo,
    settings: Settings,
    tz: timezone,
    now: datetime,
) -> tuple[str, InlineKeyboardMarkup] | None:
    """The hub screen of one user: profile, quota and every action. None if the id is unknown."""
    user = await users.get(tg_id)
    if user is None:
        return None
    report_all = await requests.report(tg_id, None)
    limit = user.context_limit if user.context_limit is not None else settings.history_token_budget
    scope = "личный" if user.context_limit is not None else "общий"
    used = await daily.requests_today(tg_id)
    if user.role is Role.ADMIN:
        quota_line = f"📨 Сегодня: {used} запросов (без лимита)"
    else:
        reset_hour = (0 + settings.admin_utc_offset) % 24
        day_limit = user.daily_limit if user.daily_limit is not None else settings.daily_request_limit
        day_scope = " · личный" if user.daily_limit is not None else ""
        quota_line = f"📨 Сегодня: {used} из {day_limit} запросов{day_scope} (сброс в {reset_hour:02d}:00)"
    dialogs_count = await dialogs.count_for_user(tg_id)
    idle_hours = (
        user.context_idle_hours if user.context_idle_hours is not None else settings.context_idle_hours
    )
    idle_scope = "личный" if user.context_idle_hours is not None else "общий"
    idle_text = "не сбрасывать" if idle_hours == 0 else f"{fmt_hours(idle_hours)} ч"

    text = (
        f"👤 <b>{html.escape(user_label(user))}</b>\n"
        f"ID: <code>{user.tg_id}</code> · роль: {ROLE_RU[user.role]}\n"
        f"С нами с {fmt_date(user.created_at, tz)} · активность: {ago(report_all.last_at, now)}\n"
        f"Модель: {user.model_override or 'по умолчанию'} · стиль: {STYLE_RU[user.style]}\n"
        f"Скиллы: {len(user.skills)} · своя роль: {'задана' if user.persona else 'нет'}\n"
        f"🧠 Лимит контекста: {fmt_tokens(limit)} ток. ({scope})\n"
        f"🕒 Сброс контекста: {idle_text} ({idle_scope})\n"
        f"{quota_line}\n"
        f"💬 Диалогов: {dialogs_count}"
    )

    builder = InlineKeyboardBuilder()
    builder.button(text="📊 Статистика", callback_data=UserStats(tg_id=tg_id, period=Period.WEEK))
    builder.button(text="💬 Переписка", callback_data=Dialogs(tg_id=tg_id, page=0))
    builder.button(text="🧠 Лимит контекста", callback_data=Limit(tg_id=tg_id))
    builder.button(text="📨 Дневной лимит", callback_data=DailyLimit(tg_id=tg_id))
    builder.button(text="✍️ Написать", callback_data=WriteTo(tg_id=tg_id))
    builder.button(text="🕒 Сброс контекста", callback_data=ContextIdle(tg_id=tg_id))
    row_sizes = [2, 4]
    if user.role is Role.PENDING:
        builder.button(text="✅ Одобрить", callback_data=SetRole(tg_id=tg_id, role=Role.USER))
        builder.button(text="🚫 Отклонить", callback_data=SetRole(tg_id=tg_id, role=Role.BLOCKED))
        row_sizes.append(2)
    elif user.role is Role.USER:
        builder.button(text="🚫 Заблокировать", callback_data=SetRole(tg_id=tg_id, role=Role.BLOCKED))
        row_sizes.append(1)
    elif user.role is Role.BLOCKED:
        builder.button(text="🔓 Разблокировать", callback_data=SetRole(tg_id=tg_id, role=Role.USER))
        row_sizes.append(1)
    builder.button(text="« К списку", callback_data=UserList(role=None, page=0))
    row_sizes.append(1)
    builder.adjust(*row_sizes)
    return text, builder.as_markup()


def set_role_guard(tg_id: int, role: Role, admin_ids: frozenset[int]) -> str | None:
    """None means the change is allowed; otherwise the alert to show instead of changing anything."""
    if tg_id in admin_ids:
        return ADMIN_GUARD_TEXT
    if role not in (Role.USER, Role.BLOCKED):
        return "Недопустимая роль."
    return None


async def limit_screen(
    tg_id: int, users: UserRepo, settings: Settings
) -> tuple[str, InlineKeyboardMarkup] | None:
    user = await users.get(tg_id)
    if user is None:
        return None
    current = user.context_limit if user.context_limit is not None else settings.history_token_budget
    scope = "личный" if user.context_limit is not None else "общий"
    text = (
        f"🧠 <b>Лимит контекста</b> — {html.escape(user_label(user))}\n"
        f"Сейчас: {fmt_tokens(current)} ток. ({scope}), "
        f"общий — {fmt_tokens(settings.history_token_budget)}.\n\n"
        "<i>Сколько токенов истории диалога (вместе с системным промптом и новым сообщением) бот "
        "отправляет модели. Более старые сообщения модель не видит. Меньше лимит — дешевле запросы, "
        "но короче память. Больше окна модели всё равно не уйдёт: Opus 5 и Sol ~1M, Fable 5.1 ~850k.</i>"
    )
    builder = InlineKeyboardBuilder()
    for value in PRESETS:
        mark = "✅ " if user.context_limit == value else ""
        builder.button(text=f"{mark}{PRESET_LABELS[value]}", callback_data=SetLimit(tg_id=tg_id, value=value))
    builder.button(text="✏️ Своё значение", callback_data=SetLimit(tg_id=tg_id, value=LIMIT_CUSTOM))
    global_mark = "✅ " if user.context_limit is None else ""
    builder.button(
        text=f"{global_mark}↩️ Общий ({short_number(settings.history_token_budget)})",
        callback_data=SetLimit(tg_id=tg_id, value=LIMIT_DEFAULT),
    )
    builder.button(text="« К пользователю", callback_data=UserCard(tg_id=tg_id))
    builder.adjust(3, 3, 2, 1, 1, 1)
    return text, builder.as_markup()


def set_daily_limit_guard(tg_id: int, admin_ids: frozenset[int]) -> str | None:
    """None means the change is allowed; otherwise the alert to show instead of writing anything.

    Same shape as set_role_guard: the shared mutation path refuses the ids in ADMIN_IDS.
    """
    if tg_id in admin_ids:
        return ADMIN_LIMIT_GUARD_TEXT
    return None


async def daily_limit_screen(
    tg_id: int, users: UserRepo, settings: Settings, daily: UsageRepo
) -> tuple[str, InlineKeyboardMarkup] | None:
    """The sibling of limit_screen for the per-day request cap. None if the id is unknown."""
    user = await users.get(tg_id)
    if user is None:
        return None
    current = user.daily_limit if user.daily_limit is not None else settings.daily_request_limit
    scope = "личный" if user.daily_limit is not None else "общий"
    used = await daily.requests_today(tg_id)
    left = max(0, current - used)
    text = (
        f"📨 <b>Дневной лимит</b> — {html.escape(user_label(user))}\n"
        f"Сейчас: {current} запросов в день ({scope}), "
        f"общий — {settings.daily_request_limit}.\n"
        f"Использовано сегодня: {used} из {current} · осталось {left}.\n\n"
        "<i>Сколько запросов бот принимает от пользователя за одни сутки. День считается по UTC, "
        "лимит обнуляется в 00:00 UTC, как и квота в основном боте. Сверх лимита бот вежливо "
        "отказывает до следующих суток.</i>"
    )
    builder = InlineKeyboardBuilder()
    for value in DAILY_PRESETS:
        mark = "✅ " if user.daily_limit == value else ""
        builder.button(text=f"{mark}{value}", callback_data=SetDailyLimit(tg_id=tg_id, value=value))
    builder.button(text="✏️ Своё значение", callback_data=SetDailyLimit(tg_id=tg_id, value=LIMIT_CUSTOM))
    global_mark = "✅ " if user.daily_limit is None else ""
    builder.button(
        text=f"{global_mark}↩️ Общий ({settings.daily_request_limit})",
        callback_data=SetDailyLimit(tg_id=tg_id, value=LIMIT_DEFAULT),
    )
    builder.button(text="« К пользователю", callback_data=UserCard(tg_id=tg_id))
    builder.adjust(3, 3, 2, 1, 1, 1)
    return text, builder.as_markup()


async def context_idle_screen(
    tg_id: int, users: UserRepo, settings: Settings
) -> tuple[str, InlineKeyboardMarkup] | None:
    """The sibling of limit_screen for the dialog reset timeout. None if the id is unknown."""
    user = await users.get(tg_id)
    if user is None:
        return None
    current = user.context_idle_hours if user.context_idle_hours is not None else settings.context_idle_hours
    scope = "личный" if user.context_idle_hours is not None else "общий"
    current_text = "не сбрасывать" if current == 0 else f"{fmt_hours(current)} ч"
    global_text = (
        "не сбрасывать" if settings.context_idle_hours == 0 else f"{fmt_hours(settings.context_idle_hours)} ч"
    )
    text = (
        f"🕒 <b>Сброс контекста</b> — {html.escape(user_label(user))}\n"
        f"Сейчас: {current_text} ({scope}), общий — {global_text}.\n\n"
        "<i>Сколько часов тишины, после которых история диалога тихо забывается и бот начинает "
        "с чистого листа. 0 — не сбрасывать никогда. Общий параметр задаётся переменной "
        "CONTEXT_IDLE_HOURS в .env.</i>"
    )
    builder = InlineKeyboardBuilder()
    for value in IDLE_PRESETS:
        mark = "✅ " if user.context_idle_hours == value else ""
        builder.button(text=f"{mark}{value} ч", callback_data=SetContextIdle(tg_id=tg_id, value=value))
    off_mark = "✅ " if user.context_idle_hours == 0 else ""
    builder.button(text=f"{off_mark}0 — выкл", callback_data=SetContextIdle(tg_id=tg_id, value=0))
    builder.button(text="✏️ Своё значение", callback_data=SetContextIdle(tg_id=tg_id, value=IDLE_CUSTOM))
    global_mark = "✅ " if user.context_idle_hours is None else ""
    builder.button(
        text=f"{global_mark}↩️ Общий ({global_text})",
        callback_data=SetContextIdle(tg_id=tg_id, value=IDLE_DEFAULT),
    )
    builder.button(text="« К пользователю", callback_data=UserCard(tg_id=tg_id))
    builder.adjust(3, 3, 1, 1, 1, 1)
    return text, builder.as_markup()


# ---- list, search, hub ---------------------------------------------------------------------------


@router.callback_query(UserList.filter())
async def cb_user_list(callback: CallbackQuery, callback_data: UserList, users: UserRepo) -> None:
    text, markup = await user_list_screen(callback_data.role, callback_data.page, users)
    await show(callback, text, markup)


@router.callback_query(FindHint.filter())
async def cb_find_hint(callback: CallbackQuery) -> None:
    text, markup = find_hint_screen()
    await show(callback, text, markup)


@router.callback_query(UserCard.filter())
async def cb_user_card(
    callback: CallbackQuery,
    callback_data: UserCard,
    users: UserRepo,
    requests: RequestRepo,
    daily: UsageRepo,
    dialogs: ConversationRepo,
    settings: Settings,
    tz: timezone,
) -> None:
    now = datetime.now(UTC)
    screen = await user_card_screen(callback_data.tg_id, users, requests, daily, dialogs, settings, tz, now)
    if screen is None:
        await callback.answer("Пользователь не найден", show_alert=True)
        return
    text, markup = screen
    await show(callback, text, markup)


# ---- roles ----------------------------------------------------------------------------------------


@router.callback_query(SetRole.filter())
async def cb_set_role(
    callback: CallbackQuery,
    callback_data: SetRole,
    users: UserRepo,
    requests: RequestRepo,
    daily: UsageRepo,
    dialogs: ConversationRepo,
    settings: Settings,
    main_bot: Bot,
    tz: timezone,
) -> None:
    tg_id, role = callback_data.tg_id, callback_data.role
    guard = set_role_guard(tg_id, role, settings.admin_ids)
    if guard is not None:
        await callback.answer(guard, show_alert=True)
        return
    found, notified = await apply_role(main_bot, users, tg_id, role)
    if not found:
        await callback.answer("Пользователь не найден", show_alert=True)
        return
    screen = await user_card_screen(tg_id, users, requests, daily, dialogs, settings, tz, datetime.now(UTC))
    assert screen is not None
    text, markup = screen
    note = notified_note(notified).strip() or "Готово"
    await show(callback, text, markup, note=note, alert=notified is False)


@router.callback_query(F.data.regexp(r"^(approve|reject):\d+$"))
async def cb_legacy_approval(
    callback: CallbackQuery, users: UserRepo, main_bot: Bot, settings: Settings
) -> None:
    """The old approve/reject buttons that access-request notices carry (bot.telegram.keyboards)."""
    assert callback.data is not None
    action, raw_id = callback.data.split(":", 1)
    target_id = int(raw_id)
    role = Role.USER if action == "approve" else Role.BLOCKED
    guard = set_role_guard(target_id, role, settings.admin_ids)
    if guard is not None:
        await callback.answer(guard, show_alert=True)
        return
    found, notified = await apply_role(main_bot, users, target_id, role)
    if not found:
        await callback.answer("Пользователь не найден", show_alert=True)
        return
    verdict = "одобрен ✅" if action == "approve" else "отклонён 🚫"
    has_text = isinstance(callback.message, Message) and callback.message.text
    original = callback.message.text if has_text else ""
    builder = InlineKeyboardBuilder()
    builder.button(text="👤 Открыть", callback_data=UserCard(tg_id=target_id))
    text = f"{html.escape(original)}\n\nРешение: {verdict}.{notified_note(notified)}"
    await show(callback, text, builder.as_markup())


# ---- add by id --------------------------------------------------------------------------------


@router.callback_query(AddUser.filter())
async def cb_add_user(callback: CallbackQuery, state: FSMContext) -> None:
    await state.set_state(AddUserForm.waiting_for_id)
    if isinstance(callback.message, Message):
        await callback.message.answer("Пришлите числовой Telegram ID — выдам доступ. Отмена: /cancel")
    await callback.answer()


@router.message(AddUserForm.waiting_for_id)
async def on_add_user_id(
    message: Message,
    state: FSMContext,
    users: UserRepo,
    requests: RequestRepo,
    daily: UsageRepo,
    dialogs: ConversationRepo,
    settings: Settings,
    main_bot: Bot,
    tz: timezone,
) -> None:
    text = (message.text or "").strip()
    if not text.isdigit():
        await message.answer("Нужен числовой ID, например 123456789. Или /cancel.")
        return
    target_id = int(text)
    if target_id in settings.admin_ids:
        notified = None
    elif await users.get(target_id) is not None:
        _, notified = await apply_role(main_bot, users, target_id, Role.USER)
    else:
        await users.upsert_seen(target_id, None, default_role=Role.USER, default_mode=Mode.NORMAL)
        notified = await notify_user(main_bot, target_id, GRANTED_TEXT)
    await state.clear()
    await message.answer(f"Готово ✅{notified_note(notified)}")
    now = datetime.now(UTC)
    screen = await user_card_screen(target_id, users, requests, daily, dialogs, settings, tz, now)
    if screen is not None:
        card_text, markup = screen
        await message.answer(card_text, reply_markup=markup)


# ---- context limit ------------------------------------------------------------------------------


@router.callback_query(Limit.filter())
async def cb_limit(
    callback: CallbackQuery, callback_data: Limit, users: UserRepo, settings: Settings
) -> None:
    screen = await limit_screen(callback_data.tg_id, users, settings)
    if screen is None:
        await callback.answer("Пользователь не найден", show_alert=True)
        return
    text, markup = screen
    await show(callback, text, markup)


@router.callback_query(SetLimit.filter())
async def cb_set_limit(
    callback: CallbackQuery,
    callback_data: SetLimit,
    users: UserRepo,
    settings: Settings,
    state: FSMContext,
) -> None:
    tg_id, value = callback_data.tg_id, callback_data.value
    if await users.get(tg_id) is None:
        await callback.answer("Пользователь не найден", show_alert=True)
        return

    if value == LIMIT_CUSTOM:
        await state.set_state(LimitForm.waiting_for_value)
        await state.update_data(tg_id=tg_id)
        if isinstance(callback.message, Message):
            await callback.message.answer(
                "Пришлите лимит в токенах: число от 1 000 до 1 048 576. Можно 64k или 1.5M. Отмена: /cancel"
            )
        await callback.answer()
        return

    if value == LIMIT_DEFAULT:
        await users.set_context_limit(tg_id, None)
        note = "Лимит: общий"
    else:
        if not CONTEXT_LIMIT_MIN <= value <= CONTEXT_LIMIT_MAX:
            await callback.answer("Недопустимое значение лимита.", show_alert=True)
            return
        await users.set_context_limit(tg_id, value)
        note = f"Лимит: {short_number(value)}"

    screen = await limit_screen(tg_id, users, settings)
    assert screen is not None
    text, markup = screen
    await show(callback, text, markup, note=note)


@router.message(LimitForm.waiting_for_value)
async def on_limit_value(message: Message, state: FSMContext, users: UserRepo, settings: Settings) -> None:
    data = await state.get_data()
    tg_id = data["tg_id"]
    value = parse_tokens(message.text or "")
    if value is None or not CONTEXT_LIMIT_MIN <= value <= CONTEXT_LIMIT_MAX:
        await message.answer("Нужно число от 1 000 до 1 048 576 (например 64000 или 64k). Или /cancel.")
        return
    await users.set_context_limit(tg_id, value)
    await state.clear()
    screen = await limit_screen(tg_id, users, settings)
    if screen is not None:
        text, markup = screen
        await message.answer(text, reply_markup=markup)


# ---- daily request limit -------------------------------------------------------------------------


@router.callback_query(DailyLimit.filter())
async def cb_daily_limit(
    callback: CallbackQuery,
    callback_data: DailyLimit,
    users: UserRepo,
    daily: UsageRepo,
    settings: Settings,
) -> None:
    screen = await daily_limit_screen(callback_data.tg_id, users, settings, daily)
    if screen is None:
        await callback.answer("Пользователь не найден", show_alert=True)
        return
    text, markup = screen
    await show(callback, text, markup)


@router.callback_query(SetDailyLimit.filter())
async def cb_set_daily_limit(
    callback: CallbackQuery,
    callback_data: SetDailyLimit,
    users: UserRepo,
    daily: UsageRepo,
    settings: Settings,
    state: FSMContext,
) -> None:
    tg_id, value = callback_data.tg_id, callback_data.value
    guard = set_daily_limit_guard(tg_id, settings.admin_ids)
    if guard is not None:
        await callback.answer(guard, show_alert=True)
        return
    if await users.get(tg_id) is None:
        await callback.answer("Пользователь не найден", show_alert=True)
        return

    if value == LIMIT_CUSTOM:
        await state.set_state(DailyLimitForm.waiting_for_value)
        await state.update_data(tg_id=tg_id)
        if isinstance(callback.message, Message):
            await callback.message.answer(
                f"Пришлите дневной лимит: целое число от {DAILY_LIMIT_MIN} до {DAILY_LIMIT_MAX} "
                "запросов. Буквы k/м не нужны. Отмена: /cancel"
            )
        await callback.answer()
        return

    if value == LIMIT_DEFAULT:
        await users.set_daily_limit(tg_id, None)
        note = "Дневной лимит: общий"
    else:
        if not DAILY_LIMIT_MIN <= value <= DAILY_LIMIT_MAX:
            await callback.answer("Недопустимое значение лимита.", show_alert=True)
            return
        await users.set_daily_limit(tg_id, value)
        note = f"Дневной лимит: {value}"

    screen = await daily_limit_screen(tg_id, users, settings, daily)
    assert screen is not None
    text, markup = screen
    await show(callback, text, markup, note=note)


@router.message(DailyLimitForm.waiting_for_value)
async def on_daily_limit_value(
    message: Message, state: FSMContext, users: UserRepo, daily: UsageRepo, settings: Settings
) -> None:
    data = await state.get_data()
    tg_id = data["tg_id"]
    value = parse_count(message.text or "")
    if value is None or not DAILY_LIMIT_MIN <= value <= DAILY_LIMIT_MAX:
        await message.answer(
            f"Нужно целое число от {DAILY_LIMIT_MIN} до {DAILY_LIMIT_MAX} — например 100. "
            "Буквы k/м тут не работают. Или /cancel."
        )
        return
    await users.set_daily_limit(tg_id, value)
    await state.clear()
    screen = await daily_limit_screen(tg_id, users, settings, daily)
    if screen is not None:
        text, markup = screen
        await message.answer(text, reply_markup=markup)


# ---- context reset timeout ------------------------------------------------------------------------


@router.callback_query(ContextIdle.filter())
async def cb_context_idle(
    callback: CallbackQuery, callback_data: ContextIdle, users: UserRepo, settings: Settings
) -> None:
    screen = await context_idle_screen(callback_data.tg_id, users, settings)
    if screen is None:
        await callback.answer("Пользователь не найден", show_alert=True)
        return
    text, markup = screen
    await show(callback, text, markup)


@router.callback_query(SetContextIdle.filter())
async def cb_set_context_idle(
    callback: CallbackQuery,
    callback_data: SetContextIdle,
    users: UserRepo,
    settings: Settings,
    state: FSMContext,
) -> None:
    tg_id, value = callback_data.tg_id, callback_data.value
    if await users.get(tg_id) is None:
        await callback.answer("Пользователь не найден", show_alert=True)
        return

    if value == IDLE_CUSTOM:
        await state.set_state(ContextIdleForm.waiting_for_value)
        await state.update_data(tg_id=tg_id)
        if isinstance(callback.message, Message):
            await callback.message.answer(
                f"Пришлите часы тишины: число от {IDLE_HOURS_MIN} до {IDLE_HOURS_MAX} "
                "(можно дробное, например 0.5). 0 — не сбрасывать. Отмена: /cancel"
            )
        await callback.answer()
        return

    if value == IDLE_DEFAULT:
        await users.set_context_idle_hours(tg_id, None)
        note = "Сброс контекста: общий"
    else:
        if not IDLE_HOURS_MIN <= value <= IDLE_HOURS_MAX:
            await callback.answer("Недопустимое значение.", show_alert=True)
            return
        await users.set_context_idle_hours(tg_id, value)
        note = "Сброс контекста: не сбрасывать" if value == 0 else f"Сброс контекста: {fmt_hours(value)} ч"

    screen = await context_idle_screen(tg_id, users, settings)
    assert screen is not None
    text, markup = screen
    await show(callback, text, markup, note=note)


@router.message(ContextIdleForm.waiting_for_value)
async def on_context_idle_value(
    message: Message, state: FSMContext, users: UserRepo, settings: Settings
) -> None:
    data = await state.get_data()
    tg_id = data["tg_id"]
    value = parse_hours(message.text or "")
    if value is None or not IDLE_HOURS_MIN <= value <= IDLE_HOURS_MAX:
        await message.answer(
            f"Нужно число от {IDLE_HOURS_MIN} до {IDLE_HOURS_MAX} — например 12 или 0.5. Или /cancel."
        )
        return
    await users.set_context_idle_hours(tg_id, value)
    await state.clear()
    screen = await context_idle_screen(tg_id, users, settings)
    if screen is not None:
        text, markup = screen
        await message.answer(text, reply_markup=markup)


# ---- free-text search: must stay last, it treats any other text as a search query -----------------


@router.message(F.text, StateFilter(None), ~F.text.startswith("/"))
async def on_search_text(
    message: Message,
    users: UserRepo,
    requests: RequestRepo,
    daily: UsageRepo,
    dialogs: ConversationRepo,
    settings: Settings,
    tz: timezone,
) -> None:
    query = (message.text or "").strip()
    matches = await users.search(query)
    if not matches:
        builder = InlineKeyboardBuilder()
        builder.button(text="« В меню", callback_data=Home())
        await message.answer(
            f"Никого не нашёл по запросу «{html.escape(query)}».", reply_markup=builder.as_markup()
        )
        return
    if len(matches) == 1:
        screen = await user_card_screen(
            matches[0].tg_id, users, requests, daily, dialogs, settings, tz, datetime.now(UTC)
        )
        assert screen is not None
        text, markup = screen
        await message.answer(text, reply_markup=markup)
        return
    builder = InlineKeyboardBuilder()
    for u in matches:
        builder.button(text=user_label(u), callback_data=UserCard(tg_id=u.tg_id))
    builder.adjust(1)
    await message.answer(f"Нашёл {len(matches)}:", reply_markup=builder.as_markup())
