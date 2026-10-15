"""Twin bot switch: the owner's style clone (twinbot) runs as its own systemd unit on this VPS.

The admin bot cannot start or stop units (the service is sandboxed with NoNewPrivileges), so the
switch is a flag file next to the shared database: twinbot checks it on every update and stays
silent while it is missing. Flipping it takes effect on the next message, no restart needed.

The «История от двойника» button travels the same channel: a request file in the same directory.
twinbot reads it on every polling-loop iteration (not only on incoming messages) and tells the
story through the same pipeline it uses when a person asks in chat.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path

from aiogram import Router
from aiogram.types import CallbackQuery, InlineKeyboardMarkup
from aiogram.utils.keyboard import InlineKeyboardBuilder

from adminbot.callbacks import Home, Twin, TwinSet, TwinStory, TwinStoryCancel, TwinStorySend
from adminbot.common import show
from adminbot.story import (
    STORY_ACTION_CANCEL,
    STORY_ACTION_SEND,
    read_pending_id,
    story_approve_path,
    write_story_approve,
)
from bot.config import Settings

logger = logging.getLogger(__name__)

router = Router(name="twin")

SWITCH_NAME = "twinbot_enabled"
UNIT = "twinbot"

#: Request file name. Duplicated in twinbot/twinbot/story.py — the two packages are deployed
# separately (no runtime import between them); tests/test_adminbot_twin.py cross-checks them.
STORY_REQUEST_NAME = "twinbot_story.request"


def switch_path(settings: Settings) -> Path:
    """The flag file lives in the data dir: the only place the sandboxed admin bot may write."""
    return settings.db_path.parent / SWITCH_NAME


def story_request_path(settings: Settings) -> Path:
    """Where twinbot looks for the story request: beside the switch file, in its own data dir."""
    return settings.db_path.parent / STORY_REQUEST_NAME


def write_story_request(path: Path) -> str:
    """Write "<iso-ts> <uuid>" to `path` and return it; a fresh uuid every press.

    The previous file is removed first so a partial overwrite can never be mistaken for a
    new request. An OSError is raised to the caller: the handler tells the owner it failed.
    """
    body = f"{datetime.now(UTC).isoformat()} {uuid.uuid4()}"
    path.unlink(missing_ok=True)
    path.write_text(body, encoding="utf-8")
    return body


def set_switch(path: Path, on: bool) -> None:
    if on:
        path.touch()
    else:
        path.unlink(missing_ok=True)


async def unit_state(unit: str = UNIT) -> str:
    """`systemctl is-active <unit>`: active / inactive / failed / activating / unknown (no root needed)."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "systemctl", "is-active", unit, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL
        )
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
    except (OSError, TimeoutError):
        return "unknown"
    return out.decode().strip() or "unknown"


def render_twin(on: bool, unit: str) -> tuple[str, InlineKeyboardMarkup]:
    """Pure: the "Двойник" screen."""
    state = "✅ <b>включён</b> — отвечает в чатах" if on else "⛔ <b>выключен</b> — молчит"
    lines = [f"🧍 <b>Двойник Глеба</b> — {state}"]
    if unit != "active":
        lines.append(f"⚠️ Сервис <code>{UNIT}</code>: {unit}. Переключатель сработает, когда он поднимется.")
    lines.append("\nПереключение действует со следующего сообщения, перезапуск не нужен.")
    lines.append(
        "📖 «История от двойника» — двойник сгенерирует историю и пришлёт превью "
        "сюда, в панель: сначала предпросмотр с кнопками «Отправить / Отмена», "
        "и только потом она уйдёт в чат."
    )
    builder = InlineKeyboardBuilder()
    if on:
        builder.button(text="⛔ Выключить", callback_data=TwinSet(on=False))
    else:
        builder.button(text="✅ Включить", callback_data=TwinSet(on=True))
    builder.button(text="📖 История от двойника", callback_data=TwinStory())
    builder.button(text="🔄 Обновить", callback_data=Twin())
    builder.button(text="« В меню", callback_data=Home())
    builder.adjust(1, 2, 2)
    return "\n".join(lines), builder.as_markup()


@router.callback_query(Twin.filter())
async def cb_twin(callback: CallbackQuery, settings: Settings) -> None:
    text, markup = render_twin(switch_path(settings).exists(), await unit_state())
    await show(callback, text, markup)


@router.callback_query(TwinStory.filter())
async def cb_twin_story(callback: CallbackQuery, settings: Settings) -> None:
    """Drop a story request for twinbot and confirm in a toast."""
    path = story_request_path(settings)
    try:
        write_story_request(path)
    except OSError as exc:
        logger.warning("story request %s failed: %s", path, exc)
        await callback.answer(f"Не вышло: {exc}", show_alert=True)
        return
    logger.info("story request written: %s", path)
    text, markup = render_twin(switch_path(settings).exists(), await unit_state())
    await show(callback, text, markup, note="Запрос принят — превью истории придёт сюда")


@router.callback_query(TwinStorySend.filter())
@router.callback_query(TwinStoryCancel.filter())
async def cb_twin_story_verdict(callback: CallbackQuery, settings: Settings) -> None:
    """Write the owner's verdict on the previewed story; a toast confirms it landed.

    The request id is taken from twinbot's preview file at press time (not from the
    callback data — it does not fit the 64-byte limit): the verdict names the story
    twinbot still holds, and if the preview is already gone — decided, timed out, or
    twinbot restarted — there is nothing to approve, which the toast says too.
    """

    send = callback.data.startswith(TwinStorySend.__prefix__)
    action = STORY_ACTION_SEND if send else STORY_ACTION_CANCEL
    request_id = read_pending_id(settings)
    if request_id is None:
        logger.info("story verdict pressed with no preview on disk — nothing to %s", action)
        await callback.answer("Превью уже неактуально — история удалена", show_alert=False)
        return
    path = story_approve_path(settings)
    try:
        write_story_approve(path, request_id, action)
    except OSError as exc:
        logger.warning("story verdict %s %s failed: %s", action, path, exc)
        await callback.answer(f"Не вышло: {exc}", show_alert=True)
        return
    logger.info("story verdict written: %s (%s)", action, request_id)
    if send:
        await callback.answer("Отправляю историю в чат двойника")
    else:
        await callback.answer("История выброшена")


@router.callback_query(TwinSet.filter())
async def cb_twin_set(callback: CallbackQuery, callback_data: TwinSet, settings: Settings) -> None:
    path = switch_path(settings)
    try:
        set_switch(path, callback_data.on)
    except OSError as exc:
        logger.warning("twin switch %s failed: %s", path, exc)
        await callback.answer(f"Не вышло: {exc}", show_alert=True)
        return
    logger.info("twin bot switched %s", "on" if callback_data.on else "off")
    text, markup = render_twin(path.exists(), await unit_state())
    await show(callback, text, markup)
