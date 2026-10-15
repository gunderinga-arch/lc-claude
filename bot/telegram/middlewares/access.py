"""Access middleware: resolve the Telegram user into a domain `User` on every update.

New users are registered on the fly (role depends on the access mode). Handlers get
`user` in their data; users without access get a short answer instead of the handler.
Updates from blocked users are dropped silently.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from aiogram import BaseMiddleware, Bot
from aiogram.types import CallbackQuery, Message, TelegramObject

from bot.config import Settings
from bot.domain import Mode, Role, User
from bot.storage.repositories import UserRepo
from bot.telegram.keyboards import approve_keyboard

logger = logging.getLogger(__name__)

PENDING_TEXT = "Заявка отправлена. Администратор скоро рассмотрит ваш запрос."
BLOCKED_TEXT = "Доступ ограничен."


class AccessMiddleware(BaseMiddleware):
    def __init__(self, users: UserRepo, settings: Settings, notifier: Bot | None = None) -> None:
        self._users = users
        self._settings = settings
        self._notifier = notifier  # the separate admin bot, when configured
        self._new_pending = False

    async def _resolve(self, tg_id: int, username: str | None) -> User:
        default_role = (
            Role.ADMIN
            if tg_id in self._settings.admin_ids
            else (Role.USER if self._settings.access_mode == "open" else Role.PENDING)
        )
        known = await self._users.get(tg_id)
        user = await self._users.upsert_seen(
            tg_id, username, default_role=default_role, default_mode=Mode.NORMAL
        )
        # Promote to admin if the id list changed after the user was registered.
        if tg_id in self._settings.admin_ids and user.role not in (Role.ADMIN, Role.BLOCKED):
            await self._users.set_role(tg_id, Role.ADMIN)
            user = await self._users.get(tg_id) or user
        self._new_pending = known is None and user.role is Role.PENDING
        return user

    async def _notify_admins(self, main_bot: Bot, tg_id: int, username: str | None) -> None:
        """Tell every admin about a new access request, with approve buttons.

        Prefers the separate admin bot when configured; falls back to the main bot if an admin
        never pressed Start there, or it is down.
        """
        name = f"@{username}" if username else "(без username)"
        text = f"🔔 Новая заявка: {name} — id <code>{tg_id}</code>"
        for admin_id in self._settings.admin_ids:
            if self._notifier is not None:
                try:
                    await self._notifier.send_message(admin_id, text, reply_markup=approve_keyboard(tg_id))
                    continue
                except Exception:
                    logger.warning("admin bot failed for admin %s, falling back", admin_id, exc_info=True)
            try:
                await main_bot.send_message(admin_id, text, reply_markup=approve_keyboard(tg_id))
            except Exception:
                logger.warning("failed to notify admin %s", admin_id, exc_info=True)

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        tg_user = data.get("event_from_user")
        if tg_user is None or tg_user.is_bot:
            return None

        user = await self._resolve(tg_user.id, tg_user.username)
        data["user"] = user
        if self._new_pending:
            bot = data.get("bot")
            if isinstance(bot, Bot):
                await self._notify_admins(bot, tg_user.id, tg_user.username)

        if user.role is Role.BLOCKED:
            if isinstance(event, Message):
                await event.answer(BLOCKED_TEXT)
            return None
        if user.role is Role.PENDING and not isinstance(event, CallbackQuery):
            if isinstance(event, Message):
                await event.answer(PENDING_TEXT)
            return None
        return await handler(event, data)
