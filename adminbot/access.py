"""Only the ADMIN_IDS, only in private chats: everyone else gets silence."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from aiogram import BaseMiddleware
from aiogram.enums import ChatType
from aiogram.types import TelegramObject


class AdminOnlyMiddleware(BaseMiddleware):
    """Outer update middleware: nothing runs for a stranger, not even a filter.

    The bot shows other people's chats, so it must not answer anyone else, and not in groups either
    (an admin adding it to a group would show the chats to the whole group).
    """

    def __init__(self, admin_ids: frozenset[int]) -> None:
        self._admin_ids = admin_ids

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        user = data.get("event_from_user")
        chat = data.get("event_chat")
        if user is None or user.id not in self._admin_ids:
            return None
        if chat is None or chat.type != ChatType.PRIVATE:
            return None
        return await handler(event, data)
