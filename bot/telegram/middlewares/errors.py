"""Error middleware: never let a handler exception reach the user as silence."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from aiogram import BaseMiddleware
from aiogram.types import Message, TelegramObject

logger = logging.getLogger(__name__)

ERROR_TEXT = "Что-то сломалось на нашей стороне. Попробуйте ещё раз чуть позже."


class ErrorMiddleware(BaseMiddleware):
    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        try:
            return await handler(event, data)
        except Exception:
            logger.exception("handler failed")
            if isinstance(event, Message):
                try:
                    await event.answer(ERROR_TEXT)
                except Exception:
                    logger.warning("failed to send error notice", exc_info=True)
            return None
