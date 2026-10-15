"""Emergency switch screen: move the main bot to the reserve gateway and back, plus model probes.

`ReserveSwitch` (bot.llm.reserve) is the single row the main bot reads on every call; this screen
only flips it and shows the latest probe results kept by `ModelProber`. Turning the reserve ON needs
one confirmation tap (every user's request goes to a weaker free model meanwhile); turning it back
OFF does not, because returning to the primary gateway is always the safe direction.
"""

from __future__ import annotations

import html
import logging
from collections.abc import Sequence
from datetime import UTC, datetime, timezone

from aiogram import Router
from aiogram.types import CallbackQuery, InlineKeyboardMarkup
from aiogram.utils.keyboard import InlineKeyboardBuilder

from adminbot.callbacks import Health, Home, ProbeNow, Reserve, ReserveSet
from adminbot.common import fmt_dt, show
from adminbot.health import fit_rows
from adminbot.probes import ModelProber, ProbeResult
from bot.config import Settings
from bot.llm.reserve import ReserveSwitch
from bot.render.format import ago

logger = logging.getLogger(__name__)

router = Router(name="reserve")


# ---- rendering (pure) -----------------------------------------------------------------------------


def _result_row(result: ProbeResult) -> str:
    mark = "✅" if result.ok else "❌"
    name = f"{mark} <b>{html.escape(result.title)}</b> <code>{html.escape(result.model)}</code>"
    if not result.ok:
        return f"{name} — {html.escape(result.error or 'ошибка')}"
    tail = f"{result.latency_ms / 1000:.1f} с"
    if result.reserve and result.answered_by:
        tail += f" · {html.escape(result.answered_by)}"
    return f"{name} — {tail}"


def render_reserve(
    configured: bool,
    on: bool,
    since: datetime | None,
    reserve_model: str,
    results: Sequence[ProbeResult],
    last_at: datetime | None,
    probe_minutes: int,
    now: datetime,
    tz: timezone,
) -> str:
    """Pure: the whole "Резерв и модели" screen."""
    if on:
        lines = ["🛟 <b>Резерв и модели</b> — 🆘 <b>АВАРИЙНЫЙ РЕЖИМ</b>"]
        since_line = f"Все запросы идут в резерв (<code>{html.escape(reserve_model)}</code>)"
        if since is not None:
            since_line += f" с {fmt_dt(since, tz)} ({ago(since, now)})"
        lines.append(since_line)
    else:
        lines = ["🛟 <b>Резерв и модели</b> — работают основные модели (BuyTokens) ✅"]
    if not configured:
        lines.append(
            "Резерв не настроен: впишите RESERVE_BASE_URL и RESERVE_API_KEY в .env и перезапустите оба бота."
        )
    lines.append("")
    lines.append("<b>Проверка моделей</b>")

    rows = [_result_row(r) for r in results]
    if last_at is not None:
        rows.append(f"Проверено: {fmt_dt(last_at, tz)} ({ago(last_at, now)})")
    else:
        rows.append("Проверки ещё не было — первая через минуту после запуска.")
    rows.append(
        "Автопроверка выключена (MODEL_PROBE_MINUTES=0)."
        if probe_minutes == 0
        else f"Автопроверка каждые {probe_minutes} мин."
    )
    return fit_rows("\n".join(lines), rows)


def reserve_keyboard(configured: bool, on: bool, has_prober: bool) -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    sizes: list[int] = []
    if configured:
        if on:
            back = ReserveSet(on=False, confirm=True)
            builder.button(text="✅ Вернуть основные модели", callback_data=back)
        else:
            builder.button(text="🆘 Включить резерв", callback_data=ReserveSet(on=True, confirm=False))
        sizes.append(1)
    if has_prober:
        builder.button(text="🔄 Проверить сейчас", callback_data=ProbeNow())
        sizes.append(1)
    builder.button(text="🩺 Состояние", callback_data=Health())
    builder.button(text="« В меню", callback_data=Home())
    sizes.append(2)
    builder.adjust(*sizes)
    return builder.as_markup()


def render_reserve_confirm(
    reserve_model: str, reserve_result: ProbeResult | None, now: datetime, tz: timezone
) -> str:
    lines = [
        "🆘 <b>Включить аварийный резерв?</b>",
        "Все запросы пользователей пойдут в бесплатные резервные модели "
        f"(<code>{html.escape(reserve_model)}</code>) вместо основных: они слабее, с меньшим "
        "контекстом и без гарантии распознавания картинок. В шапке ответа будет "
        "«🛟 Резервная модель».",
    ]
    if reserve_result is not None:
        lines.append("")
        lines.append(f"Последняя проверка резерва: {_result_row(reserve_result)}")
    return "\n".join(lines)


def reserve_confirm_keyboard() -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    builder.button(text="🆘 Да, включить", callback_data=ReserveSet(on=True, confirm=True))
    builder.button(text="« Отмена", callback_data=Reserve())
    builder.adjust(1)
    return builder.as_markup()


async def reserve_screen(
    switch: ReserveSwitch, prober: ModelProber | None, settings: Settings, now: datetime, tz: timezone
) -> tuple[str, InlineKeyboardMarkup]:
    on, since = await switch.state()
    results = prober.last if prober is not None else ()
    last_at = prober.last_at if prober is not None else None
    text = render_reserve(
        settings.reserve_configured,
        on,
        since,
        settings.reserve_model,
        results,
        last_at,
        settings.model_probe_minutes,
        now,
        tz,
    )
    markup = reserve_keyboard(settings.reserve_configured, on, prober is not None)
    return text, markup


# ---- handlers ---------------------------------------------------------------------------------


@router.callback_query(Reserve.filter())
async def cb_reserve(
    callback: CallbackQuery,
    reserve_switch: ReserveSwitch,
    prober: ModelProber | None,
    settings: Settings,
    tz: timezone,
) -> None:
    text, markup = await reserve_screen(reserve_switch, prober, settings, datetime.now(UTC), tz)
    await show(callback, text, markup)


@router.callback_query(ReserveSet.filter())
async def cb_reserve_set(
    callback: CallbackQuery,
    callback_data: ReserveSet,
    reserve_switch: ReserveSwitch,
    prober: ModelProber | None,
    settings: Settings,
    tz: timezone,
) -> None:
    if callback_data.on and not settings.reserve_configured:
        await callback.answer("Резерв не настроен.", show_alert=True)
        return
    if callback_data.on and not callback_data.confirm:
        reserve_result = next((r for r in (prober.last if prober else ()) if r.reserve), None)
        text = render_reserve_confirm(settings.reserve_model, reserve_result, datetime.now(UTC), tz)
        await show(callback, text, reserve_confirm_keyboard())
        return

    await reserve_switch.set(callback_data.on)
    user = getattr(callback, "from_user", None)
    who = user.id if user else "?"
    logger.warning("резерв %s админом %s", "включён" if callback_data.on else "выключен", who)
    note = "Резерв включён" if callback_data.on else "Основные модели возвращены"
    text, markup = await reserve_screen(reserve_switch, prober, settings, datetime.now(UTC), tz)
    await show(callback, text, markup, note=note)


@router.callback_query(ProbeNow.filter())
async def cb_probe_now(
    callback: CallbackQuery,
    reserve_switch: ReserveSwitch,
    prober: ModelProber | None,
    settings: Settings,
    tz: timezone,
) -> None:
    if prober is None:
        await callback.answer("Проверка недоступна: каталог моделей не загрузился.", show_alert=True)
        return
    if prober.busy:
        await callback.answer("Проверка уже идёт, подождите")
        return
    await callback.answer("Проверяю все модели, это до 6 минут…")
    await show(callback, "⏳ Идёт проверка…")
    await prober.probe_all(datetime.now(UTC))
    text, markup = await reserve_screen(reserve_switch, prober, settings, datetime.now(UTC), tz)
    await show(callback, text, markup)
