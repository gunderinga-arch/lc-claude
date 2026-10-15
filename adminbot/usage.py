"""Usage screens: one user's request/token/upload counters, the same for everyone, the money estimate
and the per-day activity chart.
"""

from __future__ import annotations

import html
import re
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone

from aiogram import Router
from aiogram.filters.callback_data import CallbackData
from aiogram.types import CallbackQuery, InlineKeyboardMarkup
from aiogram.utils.keyboard import InlineKeyboardBuilder

from adminbot.callbacks import Activity, Dashboard, Home, UserCard, UserStats
from adminbot.common import (
    PERIOD_LABELS,
    TELEGRAM_LIMIT,
    Period,
    estimate_cost,
    fmt_bytes,
    fmt_money,
    period_since,
    show,
)
from bot.catalog import Catalog
from bot.config import Settings
from bot.domain import GENERATED_KINDS, DayUsage, RequestReport, Role, UploadKind, UploadStat
from bot.render.format import ago, short_number, user_label
from bot.storage.repositories import RequestRepo, UploadRepo, UserRepo

router = Router(name="usage")

# Order and labels of upload kinds in the report; dict order doubles as display order.
_UPLOAD_LABELS: dict[UploadKind, str] = {
    UploadKind.PHOTO: "📷 фото",
    UploadKind.IMAGE: "🖼 картинки файлом",
    UploadKind.PDF: "📕 PDF",
    UploadKind.WORD: "📘 Word",
    UploadKind.EXCEL: "📗 Excel",
    UploadKind.TEXT: "📄 текст и код",
    UploadKind.OTHER: "📦 другие файлы",
    UploadKind.VOICE: "🎤 голосовые",
    UploadKind.AUDIO: "🎵 аудио",
    UploadKind.VIDEO_NOTE: "📹 кружочки",
    UploadKind.GEN_IMAGE: "🎨 картинки FLUX",
    UploadKind.GEN_IMAGE_GEMINI: "🍌 картинки Gemini",
    UploadKind.GEN_FILE: "📎 файлы",
}

MAX_MODEL_ROWS = 15  # keeps the report under TELEGRAM_LIMIT even with many distinct models
MODEL_NAME_CHARS = 60  # a gateway model id is short; a hostile one must not stretch the line
ACTIVITY_DAYS = 14  # what callbacks.Activity promises: the last two weeks
MAX_BAR = 12  # bar cells; 14 rows of a fixed-width bar still fit one screen comfortably

_CUT = "\n… [обрезано]"
# An escaped "<" or "&" that the cut landed inside; Telegram rejects a truncated entity outright.
_PARTIAL_ENTITY = re.compile(r"&[a-zA-Z0-9#]*$")


def _upload_lines(uploads: dict[UploadKind, UploadStat], generated: bool) -> list[str]:
    lines = []
    for kind in _UPLOAD_LABELS:
        if kind in uploads and (kind in GENERATED_KINDS) == generated:
            stat = uploads[kind]
            lines.append(f"• {_UPLOAD_LABELS[kind]} — {stat.count} · {fmt_bytes(stat.bytes)}")
    return lines


def model_label(raw: str) -> str:
    """A model name for the screen: one line, escaped first and only then cut (the house rule).

    Escaping first is what keeps a "<" in a gateway model id from becoming markup; cutting the escaped
    string afterwards may land inside an entity, so a trailing partial one is dropped like messaging.py.
    """
    escaped = html.escape(" ".join(raw.split()))
    if len(escaped) <= MODEL_NAME_CHARS:
        return escaped
    cut = escaped[:MODEL_NAME_CHARS]
    partial = _PARTIAL_ENTITY.search(cut)
    if partial is not None:
        cut = cut[: partial.start()]
    return cut + "…"


def clip_message(text: str) -> str:
    """Last-resort safety net for Telegram's 4096-character cap.

    Row caps should keep every screen short; if a pathological database still blows the budget, cut on
    a line boundary so no half line reaches Telegram, and drop a partial entity like messaging.py does:
    a truncated "&lt" makes Telegram reject the whole message.
    """
    if len(text) <= TELEGRAM_LIMIT:
        return text
    head = text[: TELEGRAM_LIMIT - len(_CUT)]
    cut = head.rfind("\n")
    head = head[:cut] if cut > 0 else head
    partial = _PARTIAL_ENTITY.search(head)
    if partial is not None:
        head = head[: partial.start()]
    return head + _CUT


# ---- money --------------------------------------------------------------------------------------


def catalog_prices(catalog: Catalog | None) -> Mapping[str, float]:
    """`model_id -> price per 1M tokens` from the injected catalog.

    Models without their own price stay out of the mapping, so estimate_cost falls back to the global
    TOKEN_PRICE. Without a catalog (not wired yet, or an unreadable models.yaml) every model uses the
    global price instead of failing the screen.
    """
    if catalog is None:
        return {}
    return {spec.id: spec.price for spec in catalog.models() if spec.price is not None}


def cost_lines(
    report: RequestReport, prices: Mapping[str, float], default_price: float, currency: str
) -> list[str]:
    """Pure: the "Стоимость" section for one report, already escaped.

    The total uses each model's own price and the global one for the rest; the per-model rows are capped
    like the model table, so a database with hundreds of models cannot overflow the message. Empty for a
    period with no tokens: there is nothing to price, and the report already says so.
    """
    if report.total_tokens <= 0 and not report.by_model:
        return []

    rows = [(model.model, model.tokens) for model in report.by_model]
    if rows:
        total = estimate_cost(rows, prices, default_price)
    else:  # tokens without a model split (very old rows): price them at the global rate
        total = report.total_tokens * default_price / 1_000_000
    lines = [
        "<b>Стоимость</b> (оценка, цена за 1M токенов)",
        f"• всего за период — {fmt_money(total, currency)}",
    ]
    for model in report.by_model[:MAX_MODEL_ROWS]:
        own = estimate_cost([(model.model, model.tokens)], prices, default_price)
        lines.append(f"   ◦ {model_label(model.model)} — {fmt_money(own, currency)}")
    return lines


def cost_section(report: RequestReport, settings: Settings | None, catalog: Catalog | None) -> list[str]:
    """The cost lines for a screen. No settings means no price configured: show no money at all."""
    if settings is None:
        return []
    return cost_lines(report, catalog_prices(catalog), settings.token_price, settings.price_currency)


# ---- the usage report ----------------------------------------------------------------------------


def render_report(
    title: str,
    period: Period,
    report: RequestReport,
    uploads: dict[UploadKind, UploadStat],
    now: datetime,
    cost: Sequence[str] = (),
) -> str:
    """Pure: the whole usage screen text for one title (already escaped by the caller).

    `cost` holds the already-rendered money lines (see cost_lines): this function stays free of Settings
    and Catalog, which is what lets the tests render a report without building either.
    """
    lines = [f"📊 <b>{title}</b> — {PERIOD_LABELS[period]}"]
    if report.requests == 0 and not uploads:
        lines.append("Запросов за период нет.")
        return "\n".join(lines)

    lines.append(f"Запросы: {report.requests} · ✅ {report.ok} · 🙊 {report.stub} · ❌ {report.failed}")
    lines.append(
        f"Средняя задержка: {report.avg_latency_ms / 1000:.1f} с · последний: {ago(report.last_at, now)}"
    )
    lines.append("")
    usage = report.usage
    lines.append(f"<b>Токены</b> (оценка бота): {short_number(report.total_tokens)}")
    lines.append(f"• вход — {short_number(usage.input)}")
    if usage.context:
        lines.append(f"   ◦ история диалога — {short_number(usage.context)}")
    if usage.tools:
        lines.append(f"   ◦ результаты инструментов — {short_number(usage.tools)}")
    lines.append(f"• выход — {short_number(usage.output)}")
    lines.append(f"• картинки на входе — {short_number(usage.images)}")
    lines.append(f"• улучшение запроса — {short_number(usage.service)}")
    if report.unsplit_tokens > 0:
        lines.append(f"• без разбивки (старые записи) — {short_number(report.unsplit_tokens)}")

    if cost:
        lines.append("")
        lines.extend(cost)

    if report.web_calls > 0:
        lines.append("")
        lines.append(f"🌐 Веб-поиск: {report.web_calls} вызовов")

    if report.by_model:
        lines.append("")
        lines.append("<b>Модели</b>")
        for model in report.by_model[:MAX_MODEL_ROWS]:
            lines.append(
                f"• {model_label(model.model)} — {model.requests} запр. · {short_number(model.tokens)} ток."
            )

    user_lines = _upload_lines(uploads, generated=False)
    if user_lines:
        lines.append("")
        lines.append("<b>Загрузки пользователя</b>")
        lines.extend(user_lines)

    gen_lines = _upload_lines(uploads, generated=True)
    if gen_lines:
        lines.append("")
        lines.append("<b>Создано ботом</b>")
        lines.extend(gen_lines)

    return clip_message("\n".join(lines))


def bar_cells(requests: int, peak: int, width: int = MAX_BAR) -> str:
    """One fixed-width bar: filled cells scaled to the window's peak, the rest shaded empty."""
    filled = 0 if peak <= 0 else min(width, max(0, round(requests / peak * width)))
    return "█" * filled + "░" * (width - filled)


def render_activity(title: str, days: Sequence[DayUsage], now: datetime, tz: timezone) -> str:
    """Pure: the per-day request/token chart for one user or for everyone.

    Bars are scaled to the busiest day of the window and padded to a fixed width, and the rows sit in a
    `<pre>` block, so in Telegram's monospace rendering the columns line up. Every day of the window gets
    a row, zero days included: a gap is information too ("nothing happened on the 3rd").
    """
    lines = [f"📈 <b>Активность</b> — {title}", f"Последние {len(days)} дней (время админа)"]
    if days and all(day.requests == 0 for day in days):
        lines.append(f"За {len(days)} дней запросов нет — график пустой.")
        return clip_message("\n".join(lines))

    peak = max((day.requests for day in days), default=0)
    today = now.astimezone(tz).date()
    rows = []
    for day in days:
        mark = "·" if day.day == today else " "
        label = f"{day.day:%d.%m}{mark}"
        body = (
            "нет запросов" if day.requests == 0 else f"{day.requests} запр. · {short_number(day.tokens)} ток."
        )
        rows.append(f"{label} {bar_cells(day.requests, peak)} {body}")
    lines.append("<pre>" + "\n".join(rows) + "</pre>")

    total_requests = sum(day.requests for day in days)
    total_tokens = sum(day.tokens for day in days)
    lines.append(
        f"Итого: {total_requests} запр. · {short_number(total_tokens)} ток. · "
        f"пик — {peak} запр. (· отмечен сегодняшний день)"
    )
    return clip_message("\n".join(lines))


def period_keyboard(
    make: Callable[[Period], CallbackData],
    current: Period,
    back_text: str,
    back: CallbackData,
    extra: tuple[str, CallbackData] | None = None,
) -> InlineKeyboardMarkup:
    """One row of the 4 periods (current one marked), an optional extra row, then a back row."""
    builder = InlineKeyboardBuilder()
    for period in Period:
        mark = "✅ " if period is current else ""
        builder.button(text=mark + PERIOD_LABELS[period], callback_data=make(period))
    if extra is not None:
        builder.button(text=extra[0], callback_data=extra[1])
    builder.button(text=back_text, callback_data=back)
    # 4 period buttons; the optional extra (the activity chart) and the back button get their own rows.
    if extra is None:
        builder.adjust(4, 1)
    else:
        builder.adjust(4, 1, 1)
    return builder.as_markup()


async def user_stats_screen(
    users: UserRepo,
    requests: RequestRepo,
    uploads: UploadRepo,
    tg_id: int,
    period: Period,
    now: datetime,
    tz: timezone,
    *,
    settings: Settings | None = None,
    catalog: Catalog | None = None,
) -> tuple[str, InlineKeyboardMarkup] | None:
    """None if the user does not exist (deleted or a stale button)."""
    user = await users.get(tg_id)
    if user is None:
        return None
    since = period_since(period, now, tz)
    report = await requests.report(tg_id, since)
    upload_stats = await uploads.stats(tg_id, since)
    cost = cost_section(report, settings, catalog)
    text = render_report(html.escape(user_label(user)), period, report, upload_stats, now, cost)
    markup = period_keyboard(
        lambda p: UserStats(tg_id=tg_id, period=p),
        period,
        "« К пользователю",
        UserCard(tg_id=tg_id),
        extra=("📈 Активность по дням", Activity(tg_id=tg_id)),
    )
    return text, markup


async def dashboard_screen(
    users: UserRepo,
    requests: RequestRepo,
    uploads: UploadRepo,
    period: Period,
    now: datetime,
    tz: timezone,
    *,
    settings: Settings | None = None,
    catalog: Catalog | None = None,
) -> tuple[str, InlineKeyboardMarkup]:
    since = period_since(period, now, tz)
    report = await requests.report(None, since)
    upload_stats = await uploads.stats(None, since)
    cost = cost_section(report, settings, catalog)
    text = render_report("Все пользователи", period, report, upload_stats, now, cost)
    total, admins, normal, pending, blocked = (
        await users.count(),
        await users.count(Role.ADMIN),
        await users.count(Role.USER),
        await users.count(Role.PENDING),
        await users.count(Role.BLOCKED),
    )
    extra = f"Пользователи: {total} · 👑 {admins} · ✅ {normal} · ⏳ {pending} · 🚫 {blocked}"
    head, _, rest = text.partition("\n")
    text = f"{head}\n{extra}\n{rest}"
    markup = period_keyboard(
        lambda p: Dashboard(period=p),
        period,
        "« В меню",
        Home(),
        extra=("📈 Активность по дням", Activity(tg_id=None)),
    )
    return text, markup


async def activity_screen(
    users: UserRepo,
    requests: RequestRepo,
    tg_id: int | None,
    now: datetime,
    tz: timezone,
) -> tuple[str, InlineKeyboardMarkup] | None:
    """The 14-day chart of one user, or of everyone with tg_id=None.

    None only when a personal chart is asked for an unknown user: a stale button must not draw a chart
    for an id that is gone.
    """
    title = "Все пользователи"
    if tg_id is not None:
        user = await users.get(tg_id)
        if user is None:
            return None
        title = html.escape(user_label(user))
    days = await requests.by_day(tg_id, ACTIVITY_DAYS, now.astimezone(tz).date(), tz)
    text = render_activity(title, days, now, tz)
    return text, activity_keyboard(tg_id)


def activity_keyboard(tg_id: int | None) -> InlineKeyboardMarkup:
    """Back to whoever opened the chart: the user card, or the dashboard for the global one."""
    builder = InlineKeyboardBuilder()
    if tg_id is None:
        builder.button(text="📊 Сводка", callback_data=Dashboard(period=Period.TODAY))
        builder.button(text="« В меню", callback_data=Home())
    else:
        builder.button(text="📊 Статистика", callback_data=UserStats(tg_id=tg_id, period=Period.WEEK))
        builder.button(text="« К пользователю", callback_data=UserCard(tg_id=tg_id))
    builder.adjust(1)
    return builder.as_markup()


@router.callback_query(UserStats.filter())
async def cb_user_stats(
    callback: CallbackQuery,
    callback_data: UserStats,
    users: UserRepo,
    requests: RequestRepo,
    uploads: UploadRepo,
    tz: timezone,
    settings: Settings,
    catalog: Catalog | None = None,
) -> None:
    screen = await user_stats_screen(
        users,
        requests,
        uploads,
        callback_data.tg_id,
        callback_data.period,
        datetime.now(tz),
        tz,
        settings=settings,
        catalog=catalog,
    )
    if screen is None:
        await callback.answer("Пользователь не найден", show_alert=True)
        return
    text, markup = screen
    await show(callback, text, markup)


@router.callback_query(Dashboard.filter())
async def cb_dashboard(
    callback: CallbackQuery,
    callback_data: Dashboard,
    users: UserRepo,
    requests: RequestRepo,
    uploads: UploadRepo,
    tz: timezone,
    settings: Settings,
    catalog: Catalog | None = None,
) -> None:
    text, markup = await dashboard_screen(
        users,
        requests,
        uploads,
        callback_data.period,
        datetime.now(tz),
        tz,
        settings=settings,
        catalog=catalog,
    )
    await show(callback, text, markup)


@router.callback_query(Activity.filter())
async def cb_activity(
    callback: CallbackQuery,
    callback_data: Activity,
    users: UserRepo,
    requests: RequestRepo,
    tz: timezone,
) -> None:
    screen = await activity_screen(users, requests, callback_data.tg_id, datetime.now(tz), tz)
    if screen is None:
        await callback.answer("Пользователь не найден", show_alert=True)
        return
    text, markup = screen
    await show(callback, text, markup)
