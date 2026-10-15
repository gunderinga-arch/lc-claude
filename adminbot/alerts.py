"""Background alert loop of the admin bot: watches the shared database and messages ADMIN_IDS.

The admin bot is the only process the admins talk to, so it is also the one that warns them: the
main bot would have to send to people who never pressed Start on *that* bot, and the admin bot
already knows its own audience (ADMIN_IDS, private chats only).

Five checks run on every pass, each independent of the others:

1. gateway failure burst  - `alert_failed_count` failed requests within the last 15 minutes;
2. user overspending      - one user's tokens for the current local day above `alert_user_tokens`;
3. the main bot went quiet - `StatusRepo.get()` older than BOT_STALE_AFTER_S (or never reported);
4. external quota pressure - Cloudflare neurons above ~80% of the free plan;
5. daily summary          - once a day, in the admin's local `daily_report_hour`.

A check whose setting is off (`alert_failed_count == 0`, `alert_user_tokens <= 0`,
`daily_report_hour == -1`) reads nothing and stays silent.

Notice keys (`NoticeRepo.claim`: a key is claimed exactly once, ever)
--------------------------------------------------------------------
De-duplication lives in the database, not in memory: a restart, a second admin-bot process or the
same condition met again one minute later must not repeat a notice. Every alert claims a key that
is stable under repetition and changes only when a *new* alert is genuinely wanted:

    gw-burst:<YYYY-MM-DDTHH:MM>       local 15-minute bucket the burst was measured in
    user-tokens:<YYYY-MM-DD>:<tg_id>  local day + the user, so once per user per day
    bot-stale:<heartbeat ISO>         the heartbeat we saw: a fresh heartbeat makes a new key,
                                      so a later outage is reported again
    bot-missing:<YYYY-MM-DD>          no heartbeat row at all: one reminder per local day
    quota:neurons:<YYYY-MM-DD>        UTC day, matching when Cloudflare resets
    daily:<YYYY-MM-DD>                local day of the digest

A key is claimed *before* the message goes out: two processes racing at the same moment must not
send twice, and a total delivery failure is logged rather than retried forever. Times follow the
admin's time zone (`admin_utc_offset`) exactly like `adminbot.common` does; the two external quotas
use UTC days/months because that is how those services reset.
"""

from __future__ import annotations

import asyncio
import html
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta, timezone
from typing import Protocol

from aiogram.types import InlineKeyboardMarkup

from adminbot.common import Period, Quotas, bot_alive, fmt_dt, fmt_tokens, load_quotas, period_since
from bot.config import Settings
from bot.domain import BOT_STALE_AFTER_S, BotStatus, ErrorEntry, RequestReport, UserUsage
from bot.render.format import ago, short_number, user_label
from bot.storage.repositories import NoticeRepo, RequestRepo, StatusRepo, UploadRepo

DEFAULT_INTERVAL_S = 60.0  # a pass a minute: fast enough to be useful, cheap enough to ignore
FIRST_PASS_DELAY_S = 60.0  # the main bot writes its first heartbeat right at startup: let it
BURST_WINDOW = timedelta(minutes=15)
QUOTA_ALERT_RATIO = 0.8  # "above ~80% of the free plan" is the point where one still can react
BURST_SAMPLE_ERRORS = 3  # a few examples say more than the bare count
TOP_SPENDERS = 5
MAX_ERROR_CHARS = 120  # one line, never a stack trace pasted into a Telegram message

QUOTA_LABELS = {
    "neurons": "🎨 Квота Cloudflare: нейроны картинок",
}


class AlertSender(Protocol):
    """The part of aiogram's Bot used here: the ADMIN bot, the one admins have started."""

    async def send_message(
        self, chat_id: int, text: str, reply_markup: InlineKeyboardMarkup | None = None
    ) -> object: ...


@dataclass(frozen=True, slots=True)
class SentAlert:
    """One alert of this pass: the notice key, the text and how many admins received it."""

    key: str
    text: str
    delivered: int


def _utc_now() -> datetime:
    return datetime.now(UTC)


# ---- notice keys (pure) -------------------------------------------------------------------------


def burst_key(now: datetime, tz: timezone) -> str:
    """The local 15-minute window the burst belongs to, so the next window may warn again."""
    local = now.astimezone(tz)
    bucket = local.replace(minute=local.minute - local.minute % 15, second=0, microsecond=0)
    return f"gw-burst:{bucket:%Y-%m-%dT%H:%M}"


def overspend_key(now: datetime, tz: timezone, tg_id: int) -> str:
    return f"user-tokens:{now.astimezone(tz):%Y-%m-%d}:{tg_id}"


def stale_key(heartbeat_at: datetime) -> str:
    return f"bot-stale:{heartbeat_at.astimezone(UTC).isoformat()}"


def missing_key(now: datetime, tz: timezone) -> str:
    """There is no heartbeat row to key on, so the reminder is bucketed by the local day."""
    return f"bot-missing:{now.astimezone(tz):%Y-%m-%d}"


def quota_key(kind: str, now: datetime) -> str:
    return f"quota:{kind}:{now.astimezone(UTC):%Y-%m-%d}"


def daily_key(now: datetime, tz: timezone) -> str:
    return f"daily:{now.astimezone(tz):%Y-%m-%d}"


def quota_targets(quotas: Quotas) -> list[tuple[str, int, int]]:
    """The external free plans as (kind, used, limit), whichever of them is over the line."""
    targets = [
        ("neurons", quotas.neurons, quotas.neurons_limit),
    ]
    return [t for t in targets if t[2] > 0 and t[1] >= t[2] * QUOTA_ALERT_RATIO]


# ---- message text (pure) --------------------------------------------------------------------------


def _error_who(entry: ErrorEntry) -> str:
    return f"@{entry.username}" if entry.username else str(entry.user_id)


def render_burst(
    count: int, threshold: int, samples: Sequence[ErrorEntry], now: datetime, tz: timezone
) -> str:
    minutes = int(BURST_WINDOW.total_seconds() // 60)
    lines = [
        "⚠️ <b>Всплеск ошибок шлюза</b>",
        f"За последние {minutes} минут запросов без ответа: {count} (порог {threshold}).",
    ]
    if samples:
        lines.append("")
        lines.append("<b>Последние</b>")
        for entry in samples:
            detail = " ".join((entry.error or entry.status.value).split())[:MAX_ERROR_CHARS]
            lines.append(
                f"• {fmt_dt(entry.created_at, tz)} ({ago(entry.created_at, now)}) · "
                f"{html.escape(_error_who(entry))} · {html.escape(entry.model)}: {html.escape(detail)}"
            )
    lines.append("")
    lines.append("<i>Похоже на сбой шлюза или модели. Проверьте доступность провайдера.</i>")
    return "\n".join(lines)


def render_overspend(user: UserUsage, threshold: int) -> str:
    """Tokens are shown exactly (fmt_tokens): the admin compares them with the threshold."""
    return (
        "💸 <b>Пользователь выше порога расхода</b>\n"
        f"{html.escape(user_label(user))} — {fmt_tokens(user.approx_tokens)} токенов за сегодня "
        f"(порог {fmt_tokens(threshold)}), запросов: {user.requests}.\n\n"
        "<i>Можно снизить лимит контекста или дневной лимит запросов в карточке пользователя.</i>"
    )


def render_bot_quiet(status: BotStatus | None, now: datetime, tz: timezone) -> str:
    if status is None:
        return (
            "🚨 <b>Основной бот не отчитывался ни разу</b>\n"
            "В базе нет записи о его работе: бот не запущен, падает при старте или пишет в другую базу."
        )
    return (
        "🚨 <b>Основной бот молчит</b>\n"
        f"Последний сигнал: {fmt_dt(status.heartbeat_at, tz)} ({ago(status.heartbeat_at, now)}).\n"
        f"Порог: {BOT_STALE_AFTER_S} с. В очереди: {status.queue_depth}, в работе: {status.active_jobs}.\n\n"
        "<i>Пока сигнала нет, запросы пользователей не обрабатываются.</i>"
    )


def render_quota(kind: str, used: int, limit: int) -> str:
    percent = round(used / limit * 100)
    return (
        f"<b>{QUOTA_LABELS[kind]}</b>\n"
        f"Израсходовано {used} из {limit} ({percent}%) — бесплатный лимит почти исчерпан.\n\n"
        "<i>Когда лимит кончится, эта возможность перестанет работать у всех.</i>"
    )


def render_daily(day: date, report: RequestReport, top: Sequence[UserUsage]) -> str:
    """The digest of one local day: counters first, then who spent the most."""
    lines = [f"📊 <b>Итоги дня</b> — {day:%d.%m.%Y}"]
    if report.requests == 0:
        lines.append("Запросов не было.")
    else:
        lines.append(f"Запросы: {report.requests} · ✅ {report.ok} · 🙊 {report.stub} · ❌ {report.failed}")
        lines.append(f"Токены: {short_number(report.total_tokens)}")
    if top:
        lines.append("")
        lines.append("<b>Больше всех</b>")
        for place, user in enumerate(top, 1):
            lines.append(
                f"{place}. {html.escape(user_label(user))} — {user.requests} запр. · "
                f"{short_number(user.approx_tokens)} ток."
            )
    return "\n".join(lines)


def _one(notice: SentAlert | None) -> list[SentAlert]:
    """A one-item list, or nothing when the key was already claimed by an earlier pass."""
    return [] if notice is None else [notice]


# ---- the service -----------------------------------------------------------------------------


class AlertService:
    """Every periodic check of the admin bot in one place.

    The repositories, the admin bot and the settings are injected; the current moment arrives as an
    argument to `check_once`. That seam is what makes the loop testable: a test drives single passes
    with a chosen `now` and reads back what was alerted, without sleeping and without a network.
    """

    def __init__(
        self,
        bot: AlertSender,
        settings: Settings,
        requests: RequestRepo,
        uploads: UploadRepo,
        status: StatusRepo,
        notices: NoticeRepo,
        tz: timezone | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._bot = bot
        self._settings = settings
        self._requests = requests
        self._uploads = uploads
        self._status = status
        self._notices = notices
        # The admin's zone, like the rest of the admin bot; still overridable for tests.
        self._tz = tz if tz is not None else timezone(timedelta(hours=settings.admin_utc_offset))
        # Only the long-running loop reads the real clock; every check takes `now` as an argument.
        self._clock = clock if clock is not None else _utc_now
        self._log = logging.getLogger(__name__)

    # ---- one pass ---------------------------------------------------------------------------

    async def check_once(self, now: datetime) -> list[SentAlert]:
        """Run all five checks once at `now` (aware) and return the alerts that were actually sent."""
        sent: list[SentAlert] = []
        sent += await self._check_burst(now)
        sent += await self._check_overspend(now)
        sent += await self._check_heartbeat(now)
        sent += await self._check_quotas(now)
        sent += await self._check_daily(now)
        return sent

    async def _check_burst(self, now: datetime) -> list[SentAlert]:
        threshold = self._settings.alert_failed_count
        if threshold <= 0:  # 0 turns the check off
            return []
        since = now - BURST_WINDOW
        count = await self._requests.count_errors(since=since, failed_only=True)
        if count < threshold:
            return []
        samples = await self._requests.errors(BURST_SAMPLE_ERRORS, since=since, failed_only=True)
        text = render_burst(count, threshold, samples, now, self._tz)
        return _one(await self._fire(burst_key(now, self._tz), text))

    async def _check_overspend(self, now: datetime) -> list[SentAlert]:
        threshold = self._settings.alert_user_tokens
        if threshold <= 0:  # -1 and 0 both turn the check off
            return []
        since = period_since(Period.TODAY, now, self._tz)
        assert since is not None  # Period.TODAY is never "all time"
        sent = []
        for user in await self._requests.heavy_users(since, threshold):
            text = render_overspend(user, threshold)
            sent += _one(await self._fire(overspend_key(now, self._tz, user.tg_id), text))
        return sent

    async def _check_heartbeat(self, now: datetime) -> list[SentAlert]:
        status = await self._status.get()
        if status is None:  # no heartbeat row at all: a reminder per local day, not once forever
            return _one(await self._fire(missing_key(now, self._tz), render_bot_quiet(None, now, self._tz)))
        if bot_alive(status, now):
            return []
        return _one(await self._fire(stale_key(status.heartbeat_at), render_bot_quiet(status, now, self._tz)))

    async def _check_quotas(self, now: datetime) -> list[SentAlert]:
        quotas = await load_quotas(self._uploads, self._settings, now)
        sent = []
        for kind, used, limit in quota_targets(quotas):
            text = render_quota(kind, used, limit)
            sent += _one(await self._fire(quota_key(kind, now), text))
        return sent

    async def _check_daily(self, now: datetime) -> list[SentAlert]:
        hour = self._settings.daily_report_hour
        if hour < 0:  # -1 turns the digest off
            return []
        if now.astimezone(self._tz).hour != hour:  # the digest waits for its own local hour
            return []
        since = period_since(Period.TODAY, now, self._tz)
        assert since is not None
        report = await self._requests.report(None, since)
        top = await self._requests.heavy_users(since, 0)  # everyone with requests, most tokens first
        text = render_daily(now.astimezone(self._tz).date(), report, top[:TOP_SPENDERS])
        return _one(await self._fire(daily_key(now, self._tz), text))

    # ---- delivery ---------------------------------------------------------------------------

    async def fire(
        self, key: str, text: str, reply_markup: InlineKeyboardMarkup | None = None
    ) -> SentAlert | None:
        """Claim `key` and message every admin; None when an earlier pass already claimed it.

        Public so callers outside the five scheduled checks (the model probe loop) can reuse the
        same claim-then-notify de-duplication instead of re-implementing it.
        """
        return await self._fire(key, text, reply_markup)

    async def _fire(
        self, key: str, text: str, reply_markup: InlineKeyboardMarkup | None = None
    ) -> SentAlert | None:
        """Claim `key` and message every admin; None when an earlier pass already claimed it.

        The claim happens first on purpose: a restart or a second process must never repeat a
        notice, even if this process dies right after sending.
        """
        if not await self._notices.claim(key):
            return None
        delivered = await self._notify(text, reply_markup)
        if delivered == 0:
            self._log.warning("alert %s reached no admin", key)
        return SentAlert(key=key, text=text, delivered=delivered)

    async def _notify(self, text: str, reply_markup: InlineKeyboardMarkup | None = None) -> int:
        """Send to every admin in turn; a blocked admin or a Telegram hiccup is logged, not fatal."""
        delivered = 0
        for admin_id in sorted(self._settings.admin_ids):
            try:
                if reply_markup is not None:
                    await self._bot.send_message(admin_id, text, reply_markup=reply_markup)
                else:
                    await self._bot.send_message(admin_id, text)
            except Exception as exc:  # broad on purpose: one admin must not mute the others
                self._log.warning("admin %s did not get the alert: %s", admin_id, exc)
                continue
            delivered += 1
        return delivered

    # ---- the loop ---------------------------------------------------------------------------

    async def run(
        self, interval_s: float = DEFAULT_INTERVAL_S, first_delay_s: float = FIRST_PASS_DELAY_S
    ) -> None:
        """Check every `interval_s` until cancelled; start it with `start_alerts`.

        The very first pass waits `first_delay_s`: both bots usually start together, and a pass at
        t=0 would read the database before the main bot has written its first heartbeat. A failing
        pass is logged and the next one follows: one bad night of the database or of Telegram must
        never silence the alerts. Cancellation propagates (CancelledError is not an Exception), so
        `task.cancel()` stops the loop right away.
        """
        self._log.info("alert loop started, one pass every %s s", interval_s)
        await asyncio.sleep(max(0.0, first_delay_s))
        while True:
            try:
                sent = await self.check_once(self._clock())
            except Exception:
                self._log.exception("alert pass failed")
            else:
                for notice in sent:
                    self._log.info("alert %s delivered to %s admins", notice.key, notice.delivered)
            await asyncio.sleep(interval_s)


def start_alerts(
    service: AlertService,
    interval_s: float = DEFAULT_INTERVAL_S,
    first_delay_s: float = FIRST_PASS_DELAY_S,
) -> asyncio.Task[None]:
    """Start the loop as a named background task; `task.cancel()` (or its own cancel) stops it."""
    return asyncio.create_task(service.run(interval_s, first_delay_s), name="adminbot-alerts")
