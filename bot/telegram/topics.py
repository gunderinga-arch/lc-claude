"""Chats as Telegram topics: a private chat with "Threaded mode" on shows every topic as a tab.

The bot's main chat ("Все", topic 0) is the menu and the place to start from; every chat is a topic of
its own with its own history and model. Without Threaded mode (BotFather) nothing here fires and the bot
keeps its single dialog.
"""

from __future__ import annotations

import contextlib
import logging
import time

from aiogram import Bot
from aiogram.exceptions import TelegramBadRequest
from aiogram.types import Message, ReplyKeyboardMarkup, Update

from bot.domain import User
from bot.services.conversation import ConversationService

logger = logging.getLogger(__name__)

TOPICS_TTL_S = 60  # the owner may switch Threaded mode on while the bot runs
NEW_CHAT_NAME = "Новый чат"
TOPIC_NAME_MAX = 128  # Telegram's limit

_seen: tuple[float, bool] = (0.0, False)


async def topics_enabled(bot: Bot) -> bool:
    """Whether the bot's private chats have topics (BotFather -> Threaded mode)."""
    global _seen
    at, on = _seen
    if time.monotonic() - at < TOPICS_TTL_S:
        return on
    try:
        on = bool((await bot.get_me()).has_topics_enabled)
    except Exception:  # a failed check keeps the last answer
        logger.warning("could not check topics mode", exc_info=True)
    _seen = (time.monotonic(), on)
    return on


def thread_of(update: Update) -> int:
    """The topic an update's message (or its callback's message) belongs to; 0 = the main chat."""
    message = update.message or (update.callback_query.message if update.callback_query else None)
    if message is not None and getattr(message, "is_topic_message", False):
        return message.message_thread_id or 0
    return 0


async def open_chat(
    bot: Bot,
    conversations: ConversationService,
    user: User,
    chat_id: int,
    name: str = NEW_CHAT_NAME,
) -> int | None:
    """A new topic registered as a chat that inherits the user's model; its id, None if Telegram refused."""
    try:
        topic = await bot.create_forum_topic(chat_id, name[:TOPIC_NAME_MAX])
    except Exception:
        logger.warning("could not create a topic for user %s", user.tg_id, exc_info=True)
        return None
    await conversations.open_chat(user.tg_id, topic.message_thread_id, model_override=user.model_override)
    return topic.message_thread_id


async def place_in_chat(
    bot: Bot,
    conversations: ConversationService,
    user: User,
    chat_id: int,
    text: str,
    reply_markup: ReplyKeyboardMarkup | None = None,
) -> tuple[int, bool] | None:
    """Post `text` in the chat the next request belongs to: (topic, reused), None if no topic can be had.

    That is the chat "new chat" opened and nobody wrote in yet, or else a new one. The post doubles as the
    check that a reused topic still exists: Telegram refuses a message to a deleted one ("thread not found"),
    unlike a typing action, and a rename would leave a notice in the chat. A dead topic is forgotten.
    """
    while True:
        thread_id = await conversations.empty_thread(user.tg_id)
        reused = thread_id is not None
        if thread_id is None:
            thread_id = await open_chat(bot, conversations, user, chat_id)
            if thread_id is None:
                return None
        try:
            await bot.send_message(chat_id, text, message_thread_id=thread_id, reply_markup=reply_markup)
        except TelegramBadRequest:
            logger.warning("topic %s of user %s is gone", thread_id, user.tg_id, exc_info=True)
            await conversations.reset(user.tg_id, thread_id)
            if not reused:
                return None
            continue
        return thread_id, reused


async def rename_chat(
    bot: Bot, conversations: ConversationService, user_id: int, chat_id: int, thread_id: int, title: str
) -> None:
    """Name the topic and remember the name (the same name again is not an error)."""
    with contextlib.suppress(Exception):
        await bot.edit_forum_topic(chat_id, thread_id, name=title[:TOPIC_NAME_MAX])
    await conversations.set_title(user_id, thread_id, title)


async def drop_chat(
    bot: Bot, conversations: ConversationService, user_id: int, chat_id: int, thread_id: int
) -> None:
    """Delete the topic with all its messages and end its conversation (the admin bot keeps the history)."""
    with contextlib.suppress(Exception):
        await bot.delete_forum_topic(chat_id, thread_id)
    await conversations.reset(user_id, thread_id)


def moved(message: Message, thread_id: int) -> Message:
    """The same message placed in a topic: its answers (`answer*`) then go there, not to the main chat."""
    return message.model_copy(update={"message_thread_id": thread_id, "is_topic_message": True})
