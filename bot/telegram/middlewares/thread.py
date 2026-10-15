"""Thread middleware: which chat (topic) an update belongs to, and that chat's own model.

Gives handlers `thread_id` (0 = the main chat) and, inside a topic, a `user` whose `model_override` is the
model of that chat, so the router, the model picker and the answer pipeline need no topic code of their own.
Runs after the access middleware, which resolves `user`.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import Any

from aiogram import BaseMiddleware
from aiogram.types import TelegramObject, Update

from bot.services.conversation import ConversationService
from bot.telegram.topics import thread_of


def _is_service(update: Update) -> bool:
    """A topic created, renamed, closed or reopened: not something the user wrote."""
    m = update.message
    return m is not None and bool(
        m.forum_topic_created or m.forum_topic_edited or m.forum_topic_closed or m.forum_topic_reopened
    )


class ThreadMiddleware(BaseMiddleware):
    def __init__(self, conversations: ConversationService) -> None:
        self._conversations = conversations

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        thread_id = thread_of(event) if isinstance(event, Update) else 0
        data["thread_id"] = thread_id
        user = data.get("user")
        if thread_id and user is not None:
            info = await self._conversations.chat(user.tg_id, thread_id)
            service = isinstance(event, Update) and _is_service(event)
            if info is None and not service:
                # A topic the bot did not open (the user made it): a chat with a name of its own, kept as is.
                # Not on its service messages: the bot's own new topic announces itself before it is saved.
                await self._conversations.open_chat(
                    user.tg_id, thread_id, model_override=user.model_override, title=""
                )
            elif info is not None and info.model_override:
                data["user"] = replace(user, model_override=info.model_override)
        return await handler(event, data)
