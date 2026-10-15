"""Role changes with a notice to the user: shared by the in-bot /admin panel and the admin bot."""

from __future__ import annotations

from typing import Protocol

from bot.domain import Role
from bot.storage.repositories import UserRepo

GRANTED_TEXT = "✅ Вам открыт доступ к боту! Нажмите /start, чтобы начать."
REJECTED_TEXT = "🚫 В доступе к боту отказано."
BLOCKED_NOTICE = "🚫 Администратор закрыл вам доступ к боту."
UNBLOCKED_TEXT = "🔓 Доступ к боту восстановлен! Нажмите /start, чтобы продолжить."


class MessageSender(Protocol):
    """The part of aiogram's Bot used here: the main bot, which is the one users talk to."""

    async def send_message(self, chat_id: int, text: str) -> object: ...


async def notify_user(bot: MessageSender, tg_id: int, text: str) -> bool:
    """Message a user; False if Telegram refuses (never started the bot, or blocked it)."""
    try:
        await bot.send_message(tg_id, text)
    except Exception:
        return False
    return True


def role_notice(old: Role | None, new: Role) -> str | None:
    if new is Role.BLOCKED:
        return REJECTED_TEXT if old is Role.PENDING else BLOCKED_NOTICE
    if new is Role.USER:
        return UNBLOCKED_TEXT if old is Role.BLOCKED else GRANTED_TEXT
    return None


async def apply_role(bot: MessageSender, users: UserRepo, tg_id: int, role: Role) -> tuple[bool, bool | None]:
    """Set the role and tell the user about it.

    Returns (found, notified); notified is None when nothing changed, so nobody gets a duplicate message.
    """
    before = await users.get(tg_id)
    if before is None or not await users.set_role(tg_id, role):
        return False, None
    notice = role_notice(before.role, role)
    if before.role is role or notice is None:
        return True, None
    return True, await notify_user(bot, tg_id, notice)


def notified_note(notified: bool | None) -> str:
    if notified is None:
        return ""
    if notified:
        return " Пользователь уведомлён."
    return " Уведомить не удалось: пользователь ещё не запускал бота или заблокировал его."
