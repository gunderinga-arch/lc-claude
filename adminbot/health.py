"""Health screens: the main bot's pulse and free quotas, the error journal, models, DB backup.

Everything here is read-only over the shared database — except the backup, which writes a copy into a
temporary file and sends it. The admin bot may be the only process alive while it is used, so a missing
status row, an empty journal or an unreadable catalog must give a friendly line instead of failing the
screen: an admin opening "Состояние" should never see an exception.
"""

from __future__ import annotations

import contextlib
import html
import logging
import math
import shutil
import sqlite3
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timezone
from pathlib import Path

from aiogram import Router
from aiogram.types import CallbackQuery, FSInputFile, InlineKeyboardMarkup, Message
from aiogram.utils.keyboard import InlineKeyboardBuilder

from adminbot.callbacks import Backup, Errors, Health, Home, Models, Reserve
from adminbot.common import (
    PERIOD_LABELS,
    TELEGRAM_LIMIT,
    Period,
    Quotas,
    bot_alive,
    estimate_cost,
    fmt_dt,
    fmt_money,
    fmt_tokens,
    load_quotas,
    period_since,
    show,
)
from adminbot.usage import period_keyboard
from bot.catalog import CatalogError, load_catalog
from bot.config import Settings
from bot.domain import BotStatus, ErrorEntry, ModelStats, RequestStatus
from bot.render.format import ago, short_number
from bot.storage.db import Database
from bot.storage.repositories import RequestRepo, StatusRepo, UploadRepo

logger = logging.getLogger(__name__)

router = Router(name="health")

ERRORS_PAGE = 5  # journal rows per screen: the reason line can be long, so pages stay small
MAX_MODEL_ROWS = 15  # same cap as usage.py: a pathological DB must not fill the whole message
ERROR_REASON_CHARS = 300  # gateway text is untrusted; the interesting part is at the start
QUOTA_WARN_PERCENT = 80  # the services' docs say to warn above this share of the free plan
_CUT = "\n… [обрезано]"


# ---- shared rendering helpers --------------------------------------------------------------------


def _clip(raw: str, cap: int) -> str:
    """Cut one raw field short. Callers escape the result: truncating escaped text cuts an entity."""
    return raw if len(raw) <= cap else raw[:cap] + "…"


def _one_line(raw: str) -> str:
    """Collapse whitespace: a stored traceback is multi-line and would tear the screen apart."""
    return " ".join(raw.split())


def _safe(raw: str | None) -> str:
    """Untrusted stored text, flattened, truncated raw and only then escaped (the house rule)."""
    return html.escape(_clip(_one_line(raw or ""), ERROR_REASON_CHARS))


def fit_rows(head: str, rows: Sequence[str]) -> str:
    """`head` plus as many already-escaped rows as keep the message inside TELEGRAM_LIMIT."""
    budget = TELEGRAM_LIMIT - len(_CUT)
    parts = [head]
    used = len(head)
    for row in rows:
        chunk = f"\n{row}"
        if used + len(chunk) > budget:
            parts.append(_CUT)
            return "".join(parts)
        parts.append(chunk)
        used += len(chunk)
    return "".join(parts)


def fmt_uptime(started_at: datetime, now: datetime) -> str:
    """Coarse uptime: at a glance days, hours or minutes alone are enough."""
    seconds = max(0, int((now - started_at).total_seconds()))
    days, rest = divmod(seconds, 86_400)
    hours, rest = divmod(rest, 3_600)
    minutes = rest // 60
    if days:
        return f"{days} д {hours} ч"
    if hours:
        return f"{hours} ч {minutes} мин"
    return f"{minutes} мин"


# ---- 1. status and quotas ------------------------------------------------------------------------

# The three answers of common.bot_alive, as they read on screen.
_ALIVE_HEAD = "🩺 <b>Состояние</b> — бот работает ✅"
_STALE_HEAD = "🩺 <b>Состояние</b> — бот молчит ⚠️"
_NEVER_HEAD = "🩺 <b>Состояние</b> — бот ещё не отчитывался ⚪"


def bar(used: int, limit: int) -> str:
    """A ten-cell usage bar; empty without a limit to compare against."""
    if limit <= 0:
        return ""
    filled = min(10, max(0, round(used / limit * 10)))
    return "▰" * filled + "▱" * (10 - filled)


def quota_percent(used: int, limit: int) -> int:
    """Share of a limit used, rounded; 0 without a limit."""
    if limit <= 0:
        return 0
    return min(999, max(0, round(used / limit * 100)))


def quota_mark(used: int, limit: int) -> str:
    """⚠️ near the limit, ⛔ over it, nothing while there is room (docs: warn above 80%)."""
    if limit <= 0:
        return ""
    if used > limit:
        return " ⛔"
    if used * 100 >= limit * QUOTA_WARN_PERCENT:
        return " ⚠️"
    return ""


def quota_line(name: str, used: int, limit: int, unit: str) -> str:
    """One service as "used / limit" with a percentage, a bar and a warning mark."""
    head = f"• {name}: {fmt_tokens(used)} / {fmt_tokens(limit)} {unit}"
    share = f"{head} — {quota_percent(used, limit)}%{quota_mark(used, limit)}"
    cells = bar(used, limit)
    return f"{share}\n{cells}" if cells else share


def render_status(status: BotStatus | None, quotas: Quotas, now: datetime, tz: timezone) -> str:
    """Pure: the whole "Состояние" screen. `status` is None until the main bot ever reported."""
    if status is None:
        lines = [
            _NEVER_HEAD,
            "Живого процесса бота здесь ещё не было: строки статуса нет. Это нормально, пока бот "
            "не запускался после обновления, — иначе он не стартовал вовсе.",
        ]
    else:
        fresh = bot_alive(status, now)
        quiet = "" if fresh else " — молчит дольше нормы, похоже, упал или завис"
        lines = [
            f"{_ALIVE_HEAD if fresh else _STALE_HEAD}{quiet}",
            f"Работает с {fmt_dt(status.started_at, tz)} ({fmt_uptime(status.started_at, now)})",
            f"Последний сигнал: {fmt_dt(status.heartbeat_at, tz)} ({ago(status.heartbeat_at, now)})",
            f"Очередь к шлюзу: {status.queue_depth} · активных ответов: {status.active_jobs}",
        ]

    lines += [
        "",
        "<b>Лимиты бесплатных планов</b>",
        quota_line("🎨 Картинки (Cloudflare)", quotas.images, quotas.neurons_limit, "картинок в день"),
        (
            f"• Нейроны за сегодня: ~{fmt_tokens(quotas.neurons)} / {fmt_tokens(quotas.neurons_limit)}"
            f" — {quota_percent(quotas.neurons, quotas.neurons_limit)}%"
            f"{quota_mark(quotas.neurons, quotas.neurons_limit)}"
        ),
        "",
        "<i>Оценка по нашим логам: у Cloudflare одна картинка — 104 нейрона дня.</i>",
    ]
    return fit_rows("\n".join(lines), [])


def status_keyboard() -> InlineKeyboardMarkup:
    """The ways out of the status screen: journal, models, backup, menu. It is the hub of the four."""
    builder = InlineKeyboardBuilder()
    builder.button(text="🗂 Журнал ошибок", callback_data=Errors(page=0))
    builder.button(text="🧩 Модели", callback_data=Models(period=Period.WEEK))
    builder.button(text="💾 Бэкап базы", callback_data=Backup(confirm=False))
    builder.button(text="🛟 Резерв и модели", callback_data=Reserve())
    builder.button(text="« В меню", callback_data=Home())
    builder.adjust(2, 2, 1)
    return builder.as_markup()


async def status_screen(
    statuses: StatusRepo,
    requests: RequestRepo,
    uploads: UploadRepo,
    settings: Settings,
    now: datetime,
    tz: timezone,
) -> tuple[str, InlineKeyboardMarkup]:
    status = await statuses.get()
    quotas = await load_quotas(uploads, settings, now)
    return render_status(status, quotas, now, tz), status_keyboard()


# ---- 2. error journal ------------------------------------------------------------------------------


def render_errors(
    entries: Sequence[ErrorEntry], page: int, pages: int, total: int, now: datetime, tz: timezone
) -> str:
    """Pure: one page of the journal, newest first (the repo already orders it that way)."""
    head = (
        f"🗂 <b>Журнал ошибок</b> — всего {total}, стр. {page + 1}/{pages}\n"
        "❌ — запрос не удался совсем, 🙊 — шлюз отдал свою заглушку вместо ответа."
    )
    if not entries:
        return head + "\nЗаписей нет — ошибок не было (или все они удачные)."

    rows = []
    for entry in entries:
        mark = "🙊" if entry.status is RequestStatus.STUB else "❌"
        user = html.escape(_one_line(entry.username) if entry.username else f"ID {entry.user_id}")
        model = html.escape(_one_line(entry.model))
        lines = [
            f"{mark} <b>{user}</b> · {model} · {fmt_dt(entry.created_at, tz)} ({ago(entry.created_at, now)})",
            f"попыток: {entry.attempts} · {entry.latency_ms / 1000:.1f} с",
        ]
        reason = _safe(entry.error)
        if reason:
            lines.append(f"<i>{reason}</i>")
        rows.append("\n".join(lines))
    return fit_rows(head, rows)


def errors_keyboard(page: int, pages: int, tg_id: int | None = None) -> InlineKeyboardMarkup:
    """Navigation that disappears at the edges, so a page beyond the last one is unclickable."""
    builder = InlineKeyboardBuilder()
    sizes: list[int] = []
    nav = []
    if page > 0:
        nav.append(("◀ Раньше", Errors(page=page - 1, tg_id=tg_id)))
    if page < pages - 1:
        nav.append(("Позже ▶", Errors(page=page + 1, tg_id=tg_id)))
    for text, data in nav:
        builder.button(text=text, callback_data=data)
    if nav:
        sizes.append(len(nav))
    builder.button(text="🧩 Модели", callback_data=Models(period=Period.WEEK))
    builder.button(text="« К состоянию", callback_data=Health())
    sizes += [1, 1]
    builder.adjust(*sizes)
    return builder.as_markup()


async def errors_screen(
    requests: RequestRepo,
    page: int,
    now: datetime,
    tz: timezone,
    *,
    tg_id: int | None = None,
    failed_only: bool = False,
) -> tuple[str, InlineKeyboardMarkup]:
    """Clamp `page` into range: a stale button after the journal shrank must not fail."""
    total = await requests.count_errors(user_id=tg_id, failed_only=failed_only)
    pages = max(1, math.ceil(total / ERRORS_PAGE))
    page = min(max(page, 0), pages - 1)
    entries = await requests.errors(ERRORS_PAGE, page * ERRORS_PAGE, user_id=tg_id, failed_only=failed_only)
    return render_errors(entries, page, pages, total, now, tz), errors_keyboard(page, pages, tg_id)


# ---- 3. models ----------------------------------------------------------------------------------


def prices_from_catalog(path: Path) -> Mapping[str, float]:
    """Per-model prices from models.yaml; a broken or missing file just means the global price."""
    try:
        catalog = load_catalog(path)
    except (CatalogError, OSError) as exc:
        logger.warning("модели: не удалось прочитать %s: %s", path, exc)
        return {}
    return {spec.id: spec.price for spec in catalog.models() if spec.price is not None}


def model_cost(stats: Sequence[ModelStats], prices: Mapping[str, float], default_price: float) -> float:
    """Estimated money for a list of models: each one priced by its own override when it has one."""
    return estimate_cost(((stat.model, stat.tokens) for stat in stats), prices, default_price)


def render_models(
    stats: Sequence[ModelStats],
    period: Period,
    prices: Mapping[str, float],
    default_price: float,
    currency: str,
) -> str:
    """Pure: the per-model comparison over a period, capped like the usage report."""
    lines = [f"🧩 <b>Модели</b> — {PERIOD_LABELS[period]}"]
    if not stats:
        lines.append("Запросов за период нет.")
        return fit_rows("\n".join(lines), [])

    shown = list(stats[:MAX_MODEL_ROWS])
    lines.append(
        f"Моделей: {len(stats)} · в отчёте {len(shown)} · "
        f"стоимость ≈ {fmt_money(model_cost(shown, prices, default_price), currency)}"
    )
    lines.append("")
    lines.append(
        "<i>Задержка — только по удачным ответам: таймауты и заглушки смазали бы среднее. "
        "Стоимость — оценка по нашим токенам.</i>"
    )

    rows = []
    for stat in shown:
        own = prices.get(stat.model)
        note = f" · своя цена {own:g}" if own is not None else ""
        avg = f"{stat.avg_ok_latency_ms / 1000:.1f} с" if stat.avg_ok_latency_ms else "—"
        top = f"{stat.max_ok_latency_ms / 1000:.1f} с" if stat.max_ok_latency_ms else "—"
        money = fmt_money(model_cost([stat], prices, default_price), currency)
        rows.append(
            f"\n<b>{html.escape(_one_line(stat.model))}</b> — {stat.requests} запр.{note}\n"
            f"✅ {stat.ok} · 🙊 {stat.stub} · ❌ {stat.failed} · ⚡ сред. {avg} · макс. {top}\n"
            f"{short_number(stat.tokens)} ток. · ≈ {money}"
        )
    if len(stats) > len(shown):
        rows.append(f"\n… ещё {len(stats) - len(shown)} моделей не показаны")
    return fit_rows("\n".join(lines), rows)


async def models_screen(
    requests: RequestRepo, period: Period, settings: Settings, now: datetime, tz: timezone
) -> tuple[str, InlineKeyboardMarkup]:
    since = period_since(period, now, tz)
    stats = await requests.model_stats(None, since)
    prices = prices_from_catalog(settings.models_file)
    text = render_models(stats, period, prices, settings.token_price, settings.price_currency)
    markup = period_keyboard(lambda p: Models(period=p), period, "« К состоянию", Health())
    return text, markup


# ---- 4. backup ------------------------------------------------------------------------------------


def backup_warning() -> str:
    return (
        "💾 <b>Бэкап базы</b> — предупреждение\n\n"
        "Файл содержит <b>всё</b>: переписку каждого пользователя, профили, лимиты и логи запросов — "
        "без всякой маскировки.\n\n"
        "• Отправляйте его только себе: это личные данные живых людей.\n"
        "• На сервере файл читает только владелец (права выставлены при создании), "
        "а у файла в чате Telegram никаких прав нет.\n"
        "• После отправки временный файл на сервере удаляется.\n\n"
        "Продолжаем?"
    )


def backup_keyboard() -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    builder.button(text="✅ Да, отправить", callback_data=Backup(confirm=True))
    builder.button(text="« Отмена", callback_data=Health())
    builder.adjust(1)
    return builder.as_markup()


def backup_filename(now: datetime) -> str:
    """The document's name: the date makes copies distinguishable in the admin's chat."""
    return f"bot-backup-{now.astimezone(UTC).strftime('%Y-%m-%d')}.sqlite3"


@dataclass(frozen=True, slots=True)
class Snapshot:
    """A finished copy of the DB: `own_dir` means it sits in a private folder we made ourselves."""

    path: Path
    own_dir: bool


def harden(path: Path) -> None:
    """Owner-only permissions for the copy: it holds every user's messages."""
    with contextlib.suppress(OSError):  # on Windows the ACLs decide; there may be nothing to chmod
        path.chmod(0o600)


def harden_dir(path: Path) -> None:
    """The same for the private folder holding the copy, so even its name stays out of sight."""
    with contextlib.suppress(OSError):
        path.chmod(0o700)


def drop(snapshot: Snapshot) -> None:
    """Remove the copy, and the private folder with it. Errors are ignored: nothing left to do."""
    with contextlib.suppress(OSError):
        snapshot.path.unlink()
    if snapshot.own_dir:
        with contextlib.suppress(OSError):
            shutil.rmtree(snapshot.path.parent, ignore_errors=True)


def reserve_path(now: datetime, directory: Path | None) -> tuple[Path, bool]:
    """A path for the copy that does not exist yet; FileExistsError if the wanted name is taken.

    VACUUM INTO refuses to overwrite, and a stale file would be a trap for whoever restores it later,
    so an existing target is an error rather than a silent overwrite. Without an explicit directory
    the copy goes into a fresh 0700 folder under the system temp dir — with its dated name, so even a
    leaked path carries the date. Synchronous on purpose: async callers must not block on pathlib.
    """
    if directory is not None:
        target = Path(directory) / backup_filename(now)
        if target.exists():
            raise FileExistsError(f"файл уже существует: {target}")
        return target, False
    own = Path(tempfile.mkdtemp(prefix="bot-backup-"))
    harden_dir(own)
    return own / backup_filename(now), True


async def make_snapshot(db: Database, now: datetime, directory: Path | None = None) -> Snapshot:
    """A consistent copy of the DB. Raises FileExistsError if the wanted name is already taken."""
    target, own_dir = reserve_path(now, directory)
    snapshot = Snapshot(target, own_dir)
    try:
        await db.snapshot(target)
    except Exception:
        # A half-written copy must never survive: whoever restores it would get a silent data loss.
        drop(snapshot)
        raise
    harden(target)
    return snapshot


async def send_backup(callback: CallbackQuery, db: Database, now: datetime) -> None:
    """Snapshot, send as a document, delete the temp copy. Never raises at the admin."""
    message = callback.message
    try:
        snapshot = await make_snapshot(db, now)
    except (FileExistsError, OSError, RuntimeError, sqlite3.Error) as exc:
        # A locked DB, a full disk, a name already taken: the admin gets a note, the bot stays up.
        logger.warning("бэкап: не удалось создать снимок: %s", exc)
        await callback.answer("Не получилось сделать бэкап: файл занят или недоступен.", show_alert=True)
        return

    try:
        if isinstance(message, Message):
            await message.answer_document(FSInputFile(snapshot.path, filename=backup_filename(now)))
        await callback.answer("Бэкап отправлен")
    except Exception:
        # aiogram raises TelegramAPIError for a real API problem, but a stalled upload may blow up
        # with anything at all: the admin gets a plain note, the traceback goes to the log.
        logger.exception("бэкап: не удалось отправить файл")
        with contextlib.suppress(Exception):
            await callback.answer("Не получилось отправить файл, подробности в логе.", show_alert=True)
    finally:
        drop(snapshot)  # the copy holds every message: it must not outlive the sending


# ---- handlers -------------------------------------------------------------------------------------


@router.callback_query(Health.filter())
async def cb_health(
    callback: CallbackQuery,
    statuses: StatusRepo,
    requests: RequestRepo,
    uploads: UploadRepo,
    settings: Settings,
    tz: timezone,
) -> None:
    text, markup = await status_screen(statuses, requests, uploads, settings, datetime.now(UTC), tz)
    await show(callback, text, markup)


@router.callback_query(Errors.filter())
async def cb_errors(
    callback: CallbackQuery, callback_data: Errors, requests: RequestRepo, tz: timezone
) -> None:
    text, markup = await errors_screen(
        requests, callback_data.page, datetime.now(UTC), tz, tg_id=callback_data.tg_id
    )
    await show(callback, text, markup)


@router.callback_query(Models.filter())
async def cb_models(
    callback: CallbackQuery, callback_data: Models, requests: RequestRepo, settings: Settings, tz: timezone
) -> None:
    text, markup = await models_screen(requests, callback_data.period, settings, datetime.now(UTC), tz)
    await show(callback, text, markup)


@router.callback_query(Backup.filter())
async def cb_backup(callback: CallbackQuery, callback_data: Backup, db: Database) -> None:
    if not callback_data.confirm:
        await show(callback, backup_warning(), backup_keyboard())
        return
    await send_backup(callback, db, datetime.now(UTC))
