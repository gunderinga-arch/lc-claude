"""One-message-per-chat UI panel: every screen replaces the previous one.

The bot keeps exactly one live interface message (menu, model picker, help) per chat
(in a chat with topics: per topic).
When a new screen is shown, the previous one is deleted, so the chat stays clean.
Best effort: a message that Telegram refuses to delete (too old, already gone) is skipped.
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path

from aiogram import Bot
from aiogram.types import FSInputFile, InlineKeyboardMarkup, Message

logger = logging.getLogger(__name__)

# (chat_id, topic) -> message_id of the live panel; topic 0 is the chat itself
_panels: dict[tuple[int, int], int] = {}
# Telegram file ids of uploaded assets: upload once per process, then reuse.
_file_ids: dict[Path, str] = {}


def media(path: Path) -> str | FSInputFile:
    """The cached file id of an asset, or the file itself on first use."""
    return _file_ids.get(path) or FSInputFile(path)


def remember(path: Path, sent: Message) -> None:
    item = sent.animation or (sent.photo[-1] if sent.photo else None)
    if item is not None:
        _file_ids[path] = item.file_id


async def _delete_quietly(bot: Bot, chat_id: int, message_id: int) -> None:
    with contextlib.suppress(Exception):
        await bot.delete_message(chat_id, message_id)


async def show_panel(
    bot: Bot,
    chat_id: int,
    text: str,
    reply_markup: InlineKeyboardMarkup | None = None,
    *,
    replace: Message | None = None,
    photo: Path | None = None,
    thread_id: int = 0,
) -> Message:
    """Send `text` as the chat's single panel (in topic `thread_id`), deleting the previous panel.

    `replace` is the callback message to reuse when possible (edit instead of delete+send).
    With `photo`, the panel is that picture (an .mp4 goes as a looping animation) with `text` as its caption.
    """
    if replace is not None and replace.is_topic_message:
        thread_id = replace.message_thread_id or thread_id
    key = (chat_id, thread_id)
    previous = _panels.get(key)
    topic = {"message_thread_id": thread_id} if thread_id else {}

    # A text message can't become a photo and back, so only a panel of the same kind is edited.
    if (
        replace is not None
        and previous == replace.message_id
        and bool(replace.photo or replace.animation) == (photo is not None)
    ):
        try:
            if photo is None:
                edited = await replace.edit_text(text, reply_markup=reply_markup)
            else:
                edited = await replace.edit_caption(caption=text, reply_markup=reply_markup)
            if isinstance(edited, Message):
                return edited
        except Exception:
            logger.debug("panel edit failed, sending a new one", exc_info=True)

    sent = None
    if photo is not None:
        try:
            send = bot.send_animation if photo.suffix == ".mp4" else bot.send_photo
            sent = await send(chat_id, media(photo), caption=text, reply_markup=reply_markup, **topic)
            remember(photo, sent)
        except Exception:  # a missing asset must not take the screen down with it
            logger.warning("could not send panel photo %s", photo.name, exc_info=True)
    if sent is None:
        sent = await bot.send_message(chat_id, text, reply_markup=reply_markup, **topic)
    _panels[key] = sent.message_id
    if previous is not None and previous != sent.message_id:
        await _delete_quietly(bot, chat_id, previous)
    return sent


async def drop_panel(bot: Bot, chat_id: int, thread_id: int = 0) -> None:
    """Delete the live panel of the chat (or of one topic), if any."""
    previous = _panels.pop((chat_id, thread_id), None)
    if previous is not None:
        await _delete_quietly(bot, chat_id, previous)


# Private-chat message ids are sequential for both sides, so a whole chat session is a range.
DELETE_BATCH = 100  # Bot API limit for deleteMessages
MAX_CLEAR_IDS = 5_000  # ponytail: hard cap on one clear; raise if users keep longer sessions


async def delete_range(bot: Bot, chat_id: int, first_id: int, last_id: int) -> None:
    """Delete messages first_id..last_id (inclusive), newest first. Missing ids are skipped by Telegram.

    Telegram refuses to delete messages older than 48 hours; if a batch is refused, its messages are
    retried one by one so the recent ones still go. Old ones simply stay.
    """
    for key in [key for key in _panels if key[0] == chat_id]:
        del _panels[key]
    first_id = max(first_id, last_id - MAX_CLEAR_IDS + 1, 1)
    for top in range(last_id, first_id - 1, -DELETE_BATCH):
        ids = list(range(top, max(top - DELETE_BATCH, first_id - 1), -1))
        try:
            await bot.delete_messages(chat_id, ids)
        except Exception:
            logger.info("batch delete refused in chat %s, deleting one by one", chat_id)
            for message_id in ids:
                await _delete_quietly(bot, chat_id, message_id)
