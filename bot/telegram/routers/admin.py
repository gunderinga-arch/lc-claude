"""Admin panel: /admin plus /stats, /approve, /block, /unblock and all callbacks."""

from __future__ import annotations

import contextlib
import html
from datetime import UTC, datetime

from aiogram import Bot, Router
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command, CommandObject
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import CallbackQuery, Message

from bot.domain import Mode, Role, User, UserUsage
from bot.render.format import ago, short_number, user_label
from bot.services.roles import GRANTED_TEXT, apply_role, notified_note, notify_user
from bot.storage.repositories import LEADERBOARD_ORDER, RequestRepo, UserRepo
from bot.telegram.keyboards import (
    LEADERBOARD_LABELS,
    admin_panel_keyboard,
    admin_user_keyboard,
    admin_users_keyboard,
    approve_keyboard,
    leaderboard_keyboard,
)
from bot.telegram.panel import show_panel

router = Router(name="admin")

ADMIN_MENU_TEXT = (
    "<b>Админ-панель</b>\n\n"
    "Заявки — кто ждёт доступа. Пользователи — блокировки. "
    "Добавить по ID — сразу выдать доступ, даже если человек ещё не писал боту. "
    "Рейтинг — кто сколько пользуется ботом."
)


class AddUserForm(StatesGroup):
    waiting_for_id = State()


def _denied(message: Message, user: User) -> bool:
    """Admin commands silently ignore non-admins."""
    return not user.is_admin


def _leaderboard_text(rows: list[UserUsage], order: str, now: datetime) -> str:
    lines = [f"<b>🏆 Рейтинг: {LEADERBOARD_LABELS[order].lower()}</b> (за всё время, топ {len(rows)})\n"]
    for place, row in enumerate(rows, 1):
        lines.append(
            f"{place}. {html.escape(user_label(row))} — {row.requests} запр. · "
            f"{short_number(row.approx_tokens)} ток. · {ago(row.last_at, now)}"
        )
    if not rows:
        lines.append("Пользователей пока нет.")
    lines.append("\n<i>Токены — оценка бота (вопрос с контекстом + ответ).</i>")
    return "\n".join(lines)


async def _stats_text(requests: RequestRepo) -> str:
    stats = await requests.stats_today()
    return (
        "<b>Статистика за сегодня (UTC)</b>\n"
        f"Пользователей всего: {stats.users_total}\n"
        f"Ожидают одобрения: {stats.users_pending}\n"
        f"Запросов: {stats.requests_today}\n"
        f"Токенов (оценка): {stats.tokens_today}\n"
        f"Ошибок: {stats.failed_today}\n"
        f"Средняя задержка: {stats.avg_latency_ms_today} мс"
    )


async def _send_pending(message: Message, users: UserRepo) -> None:
    pending = await users.list_by_role(Role.PENDING)
    if not pending:
        await message.answer("Заявок нет.")
        return
    for candidate in pending:
        await message.answer(
            f"{user_label(candidate)} — id <code>{candidate.tg_id}</code>",
            reply_markup=approve_keyboard(candidate.tg_id),
        )


# ---- commands -----------------------------------------------------------------------------------


@router.message(Command("admin"))
async def cmd_admin(message: Message, user: User, bot: Bot, thread_id: int = 0) -> None:
    if _denied(message, user):
        return
    with contextlib.suppress(Exception):
        await message.delete()
    await show_panel(bot, message.chat.id, ADMIN_MENU_TEXT, admin_panel_keyboard(), thread_id=thread_id)


@router.message(Command("stats"))
async def cmd_stats(message: Message, user: User, requests: RequestRepo) -> None:
    if _denied(message, user):
        return
    await message.answer(await _stats_text(requests))


@router.message(Command("pending"))
async def cmd_pending(message: Message, user: User, users: UserRepo) -> None:
    if _denied(message, user):
        return
    await _send_pending(message, users)


async def _set_role_by_id(
    message: Message, command: CommandObject, users: UserRepo, bot: Bot, role: Role, done: str
) -> None:
    if command.args is None or not command.args.strip().isdigit():
        await message.answer("Формат: /команда <telegram_id>")
        return
    target = int(command.args.strip())
    found, notified = await apply_role(bot, users, target, role)
    if found:
        await message.answer(f"Готово: {target} — {done}.{notified_note(notified)}")
    else:
        await message.answer(f"Пользователь {target} не найден.")


@router.message(Command("block"))
async def cmd_block(message: Message, user: User, command: CommandObject, users: UserRepo, bot: Bot) -> None:
    if _denied(message, user):
        return
    await _set_role_by_id(message, command, users, bot, Role.BLOCKED, "заблокирован")


@router.message(Command("unblock"))
async def cmd_unblock(
    message: Message, user: User, command: CommandObject, users: UserRepo, bot: Bot
) -> None:
    if _denied(message, user):
        return
    await _set_role_by_id(message, command, users, bot, Role.USER, "разблокирован")


@router.message(Command("approve"))
async def cmd_approve(
    message: Message, user: User, command: CommandObject, users: UserRepo, bot: Bot
) -> None:
    if _denied(message, user):
        return
    await _set_role_by_id(message, command, users, bot, Role.USER, "одобрен")


# ---- admin panel callbacks ----------------------------------------------------------------------


@router.callback_query(lambda c: c.data and c.data.startswith("admin:"))
async def cb_admin(
    callback: CallbackQuery,
    user: User,
    users: UserRepo,
    requests: RequestRepo,
    state: FSMContext,
    bot: Bot,
) -> None:
    if not user.is_admin:
        await callback.answer("Недостаточно прав", show_alert=True)
        return
    assert callback.data and isinstance(callback.message, Message)
    parts = callback.data.split(":")
    action = parts[1]

    if action == "menu":
        await callback.message.edit_text(ADMIN_MENU_TEXT, reply_markup=admin_panel_keyboard())
    elif action == "stats":
        await callback.message.edit_text(await _stats_text(requests), reply_markup=admin_panel_keyboard())
    elif action == "top" and len(parts) == 3 and parts[2] in LEADERBOARD_ORDER:
        rows = await requests.leaderboard(parts[2])
        with contextlib.suppress(TelegramBadRequest):  # "message is not modified" on a repeated tap
            await callback.message.edit_text(
                _leaderboard_text(rows, parts[2], datetime.now(UTC)),
                reply_markup=leaderboard_keyboard(parts[2]),
            )
    elif action == "pending":
        await callback.message.edit_text("Заявки ниже 👇", reply_markup=admin_panel_keyboard())
        await _send_pending(callback.message, users)
    elif action == "add_prompt":
        await state.set_state(AddUserForm.waiting_for_id)
        await callback.message.answer(
            "Отправьте числовой Telegram ID пользователя — добавлю его в whitelist. Отмена: /cancel"
        )
    elif action == "users":
        all_users = await users.list_all(limit=30)
        if not all_users:
            await callback.message.edit_text("Пользователей пока нет.")
            return
        await callback.message.edit_text(
            "<b>Пользователи</b> (последние 30):",
            reply_markup=admin_users_keyboard([(u.tg_id, user_label(u)) for u in all_users]),
        )
    elif action == "user" and len(parts) == 3:
        target = await users.get(int(parts[2]))
        if target is None:
            await callback.answer("Пользователь не найден", show_alert=True)
            return
        text = (
            f"{user_label(target)}\n"
            f"id: <code>{target.tg_id}</code>\n"
            f"роль: {target.role.value}\n"
            f"режим: {target.mode.value}\n"
            f"модель: {target.model_override or 'по режиму'}"
        )
        await callback.message.edit_text(
            text, reply_markup=admin_user_keyboard(target.tg_id, target.role is Role.BLOCKED)
        )
    elif action in ("block", "unblock") and len(parts) == 3:
        target_id = int(parts[2])
        role = Role.BLOCKED if action == "block" else Role.USER
        found, notified = await apply_role(bot, users, target_id, role)
        if not found:
            await callback.answer("Пользователь не найден", show_alert=True)
            return
        target = await users.get(target_id)
        if target is not None:
            text = f"{user_label(target)}\nid: <code>{target.tg_id}</code>\nроль: {target.role.value}"
            await callback.message.edit_text(
                text, reply_markup=admin_user_keyboard(target_id, role is Role.BLOCKED)
            )
        note = notified_note(notified).strip()
        await callback.answer(note or None, show_alert=notified is False)
        return
    await callback.answer()


# ---- add-user-by-ID FSM -------------------------------------------------------------------------


@router.message(AddUserForm.waiting_for_id)
async def on_add_user_id(message: Message, user: User, users: UserRepo, state: FSMContext, bot: Bot) -> None:
    if _denied(message, user):
        await state.clear()
        return
    text = (message.text or "").strip()
    if not text.isdigit():
        await message.answer("Нужен числовой ID (например 123456789). Попробуйте ещё раз или /cancel.")
        return
    target_id = int(text)
    if await users.get(target_id) is not None:
        _, notified = await apply_role(bot, users, target_id, Role.USER)
    else:
        await users.upsert_seen(target_id, None, default_role=Role.USER, default_mode=Mode.NORMAL)
        notified = await notify_user(bot, target_id, GRANTED_TEXT)
    await state.clear()
    await message.answer(
        f"Пользователь <code>{target_id}</code> добавлен в whitelist ✅{notified_note(notified)}",
        reply_markup=admin_panel_keyboard(),
    )


# ---- pending approval callbacks -----------------------------------------------------------------


@router.callback_query(lambda c: c.data and (c.data.startswith("approve:") or c.data.startswith("reject:")))
async def cb_approval(callback: CallbackQuery, user: User, users: UserRepo, bot: Bot) -> None:
    if not user.is_admin:
        await callback.answer("Недостаточно прав", show_alert=True)
        return
    assert callback.data and isinstance(callback.message, Message)
    action, raw_id = callback.data.split(":", 1)
    target_id = int(raw_id)
    role = Role.USER if action == "approve" else Role.BLOCKED
    found, notified = await apply_role(bot, users, target_id, role)
    if not found:
        await callback.answer("Пользователь не найден", show_alert=True)
        return
    verdict = "одобрен ✅" if action == "approve" else "отклонён 🚫"
    await callback.message.edit_text(
        f"{callback.message.text}\n\nРешение: {verdict}.{notified_note(notified)}"
    )
    await callback.answer()
