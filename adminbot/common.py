"""Shared bits of the admin bot screens: periods, local time, sizes, editing the screen in place."""

from __future__ import annotations

import contextlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone
from enum import StrEnum

from aiogram.exceptions import TelegramBadRequest
from aiogram.types import CallbackQuery, InlineKeyboardMarkup, Message

from bot.config import Settings
from bot.domain import BOT_STALE_AFTER_S, BotStatus, Role, UploadKind
from bot.storage.repositories import UploadRepo

TELEGRAM_LIMIT = 4096


class Period(StrEnum):
    TODAY = "d"  # since the admin's local midnight
    WEEK = "w"
    MONTH = "m"
    ALL = "a"


PERIOD_LABELS = {
    Period.TODAY: "Сегодня",
    Period.WEEK: "7 дней",
    Period.MONTH: "30 дней",
    Period.ALL: "Всё время",
}


def period_since(period: Period, now: datetime, tz: timezone) -> datetime | None:
    """Start of the period as an aware datetime; None for all time. `now` must be aware."""
    if period is Period.ALL:
        return None
    if period is Period.TODAY:
        return now.astimezone(tz).replace(hour=0, minute=0, second=0, microsecond=0)
    return now - timedelta(days=7 if period is Period.WEEK else 30)


def fmt_dt(moment: datetime, tz: timezone) -> str:
    """Day, month and time in the admin's time zone, e.g. 24.09 14:03."""
    return moment.astimezone(tz).strftime("%d.%m %H:%M")


def fmt_date(moment: datetime, tz: timezone) -> str:
    return moment.astimezone(tz).strftime("%d.%m.%Y")


def fmt_bytes(n: int) -> str:
    if n >= 1024 * 1024:
        return f"{n / 1024 / 1024:.1f} МБ"
    if n >= 1024:
        return f"{n / 1024:.0f} КБ"
    return f"{n} Б"


def fmt_tokens(n: int) -> str:
    """Exact count with thin grouping: "64 000"."""
    return f"{n:,}".replace(",", " ")


async def show(
    callback: CallbackQuery,
    text: str,
    markup: InlineKeyboardMarkup | None = None,
    *,
    note: str | None = None,
    alert: bool = False,
) -> None:
    """Replace the screen the button was pressed on (a document gets a new message), then answer
    the query, with `note` as a toast or, with `alert`, as a popup. Call it once per callback."""
    message = callback.message
    if isinstance(message, Message) and message.text is not None:
        try:
            await message.edit_text(text, reply_markup=markup)
        except TelegramBadRequest as exc:
            if "message is not modified" not in str(exc):
                raise
    elif isinstance(message, Message):
        await message.answer(text, reply_markup=markup)
    with contextlib.suppress(TelegramBadRequest):  # the query may be too old to answer
        await callback.answer(note, show_alert=alert)


# ---- broadcast audiences ----------------------------------------------------------------------


class Audience(StrEnum):
    ACTIVE = "a"  # everyone with access
    PENDING = "p"  # waiting for approval
    ALL = "x"  # everyone except the blocked


AUDIENCE_ROLES = {
    Audience.ACTIVE: (Role.USER, Role.ADMIN),
    Audience.PENDING: (Role.PENDING,),
    Audience.ALL: (Role.USER, Role.ADMIN, Role.PENDING),
}

AUDIENCE_LABELS = {
    Audience.ACTIVE: "Всем с доступом",
    Audience.PENDING: "Ожидающим одобрения",
    Audience.ALL: "Всем, кроме заблокированных",
}


# ---- money --------------------------------------------------------------------------------------


def estimate_cost(
    tokens_by_model: Iterable[tuple[str, int]], prices: Mapping[str, float], default: float
) -> float:
    """Tokens priced per 1M: the model's own price from models.yaml, else the global TOKEN_PRICE."""
    return sum(tokens * prices.get(model, default) for model, tokens in tokens_by_model) / 1_000_000


def fmt_money(amount: float, currency: str) -> str:
    """Enough digits for fractions of a cent: the gateway is cheap."""
    digits = 2 if amount >= 1 else 3 if amount >= 0.01 else 5
    number = f"{amount:.{digits}f}"
    return f"{currency}{number}" if currency in ("$", "€", "£") else f"{number} {currency}"


# ---- external services and the main bot -------------------------------------------------------

NEURONS_PER_IMAGE = 104  # Cloudflare FLUX, one 1024x1024 picture (see bot/services/images.py)


@dataclass(frozen=True, slots=True)
class Quotas:
    """Free-plan usage of the external services, estimated from our own logs."""

    images: int  # generated today, UTC day: Cloudflare resets at 00:00 UTC
    neurons_limit: int

    @property
    def neurons(self) -> int:
        return self.images * NEURONS_PER_IMAGE


async def load_quotas(uploads: UploadRepo, settings: Settings, now: datetime) -> Quotas:
    utc = now.astimezone(UTC)
    day = utc.replace(hour=0, minute=0, second=0, microsecond=0)
    made = (await uploads.stats(None, day)).get(UploadKind.GEN_IMAGE)
    return Quotas(
        images=made.count if made else 0,
        neurons_limit=settings.cloudflare_daily_neurons,
    )


def bot_alive(status: BotStatus | None, now: datetime) -> bool | None:
    """True if the main bot's heartbeat is fresh, False if it went quiet, None if it never reported."""
    if status is None:
        return None
    return (now - status.heartbeat_at).total_seconds() <= BOT_STALE_AFTER_S
