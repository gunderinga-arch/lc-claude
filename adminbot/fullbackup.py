"""Full server backup on demand: every project, database, VPN and settings, sent to this chat.

The admin bot is sandboxed and cannot start units or read the other projects, so it only drops a
request file into its data dir. play2go-backup.path (root) notices it and runs
/usr/local/sbin/play2go-backup (deploy/play2go-backup.sh), which packs the archives and sends them
here itself through the Bot API, in parts under the 50 MB limit.
"""

from __future__ import annotations

import logging
from pathlib import Path

from aiogram import Router
from aiogram.types import CallbackQuery, InlineKeyboardMarkup
from aiogram.utils.keyboard import InlineKeyboardBuilder

from adminbot.callbacks import Home, ServerBackup
from adminbot.common import show
from adminbot.twin import unit_state
from bot.config import Settings

logger = logging.getLogger(__name__)

router = Router(name="fullbackup")

REQUEST_NAME = "backup.request"
SERVICE = "play2go-backup.service"
PATH_UNIT = "play2go-backup.path"


def request_path(settings: Settings) -> Path:
    """Must match PathExists= in deploy/play2go-backup.path."""
    return settings.db_path.parent / REQUEST_NAME


def is_busy(requested: bool, service: str) -> bool:
    """A backup is queued (file not picked up yet) or the oneshot service is still running."""
    return requested or service in ("activating", "active", "reloading")


def render_backup(requested: bool, service: str, watcher: str) -> tuple[str, InlineKeyboardMarkup]:
    """Pure: the "Бэкап сервера" screen."""
    lines = [
        "💾 <b>Бэкап сервера</b>",
        "Соберу все проекты, базы, VPN и настройки сервера и пришлю архивом сюда (~35 МБ, 1–3 минуты). "
        "Большой архив придёт частями по 45 МБ.",
        "⚠️ В архиве все секреты: токены, ключи, ссылка VPN. Храни у себя и никому не пересылай.",
    ]
    if watcher != "active":
        lines.append(f"\n❌ Служба бэкапа не запущена (<code>{PATH_UNIT}</code>: {watcher}). Нужен деплой.")
    elif is_busy(requested, service):
        lines.append("\n⏳ Бэкап уже собирается — архив придёт сообщением.")
    builder = InlineKeyboardBuilder()
    if watcher == "active" and not is_busy(requested, service):
        builder.button(text="✅ Собрать и прислать", callback_data=ServerBackup(run=True))
    builder.button(text="🔄 Обновить", callback_data=ServerBackup())
    builder.button(text="« В меню", callback_data=Home())
    builder.adjust(1)
    return "\n".join(lines), builder.as_markup()


async def _screen(settings: Settings) -> tuple[str, InlineKeyboardMarkup]:
    return render_backup(
        request_path(settings).exists(), await unit_state(SERVICE), await unit_state(PATH_UNIT)
    )


@router.callback_query(ServerBackup.filter())
async def cb_server_backup(callback: CallbackQuery, callback_data: ServerBackup, settings: Settings) -> None:
    note, alert = None, False
    if callback_data.run:
        path = request_path(settings)
        if path.exists() or is_busy(False, await unit_state(SERVICE)):
            note, alert = "Бэкап уже собирается", True
        else:
            try:
                path.touch()
            except OSError as exc:
                logger.warning("backup request %s failed: %s", path, exc)
                note, alert = f"Не вышло: {exc}", True
            else:
                logger.info("full server backup requested")
                note = "Запрос принят, архив придёт сюда"
    text, markup = await _screen(settings)
    await show(callback, text, markup, note=note, alert=alert)
