"""adminbot.alerts: each alert fires under its condition, stays silent otherwise, and never repeats.

Everything runs against an in-memory database and a fake bot, and the clock is a plain argument of
one pass, so no test waits for a real interval and none touches the network. Most tests drive a pass
with `pass_live`, which keeps the main bot's heartbeat fresh: that way only the alert under test can
possibly fire, and the heartbeat alert has its own dedicated tests.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone

import pytest

from adminbot.alerts import (
    AlertService,
    SentAlert,
    burst_key,
    daily_key,
    missing_key,
    overspend_key,
    quota_key,
    quota_targets,
    render_bot_quiet,
    render_burst,
    render_daily,
    render_overspend,
    render_quota,
    stale_key,
    start_alerts,
)
from adminbot.common import Quotas, fmt_dt
from bot.domain import (
    BOT_STALE_AFTER_S,
    BotStatus,
    ErrorEntry,
    Mode,
    RequestRecord,
    RequestReport,
    RequestStatus,
    Role,
    TokenUsage,
    UploadKind,
    UserUsage,
)
from bot.storage.db import Database
from bot.storage.repositories import NoticeRepo, RequestRepo, StatusRepo, UploadRepo, UserRepo

TZ = timezone(timedelta(hours=3))  # the admin's local zone, as in the shipped defaults
NOW = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)  # 15:00 in the admin's zone
LOCAL_DAY = "2026-01-15"


@dataclass(frozen=True)
class FakeSettings:
    """Just the attributes the alert service reads off Settings."""

    admin_ids: frozenset[int] = frozenset({100, 200})
    admin_utc_offset: int = 3
    alert_failed_count: int = 5
    alert_user_tokens: int = 1_000_000
    daily_report_hour: int = 9
    cloudflare_daily_neurons: int = 10_000


class FakeBot:
    """Records the admin messages; the ids in `refuse` are the admins Telegram turns away."""

    def __init__(self, refuse: frozenset[int] = frozenset()) -> None:
        self.refuse = refuse
        self.sent: list[tuple[int, str]] = []

    async def send_message(self, chat_id: int, text: str) -> None:
        if chat_id in self.refuse:
            raise RuntimeError("Forbidden: bot was blocked by the user")
        self.sent.append((chat_id, text))

    @property
    def texts(self) -> list[str]:
        return [text for _, text in self.sent]

    @property
    def targets(self) -> list[int]:
        return [chat_id for chat_id, _ in self.sent]


@dataclass
class Repos:
    db: Database
    users: UserRepo
    requests: RequestRepo
    uploads: UploadRepo
    status: StatusRepo
    notices: NoticeRepo


@pytest.fixture
async def bundle() -> Database:
    database = Database(":memory:")
    await database.connect()
    try:
        yield database
    finally:
        await database.close()


@pytest.fixture
async def repos(bundle: Database) -> Repos:
    return Repos(
        db=bundle,
        users=UserRepo(bundle),
        requests=RequestRepo(bundle),
        uploads=UploadRepo(bundle),
        status=StatusRepo(bundle),
        notices=NoticeRepo(bundle),
    )


def make_service(
    repos: Repos,
    settings: FakeSettings | None = None,
    bot: FakeBot | None = None,
    clock: object = None,
) -> tuple[AlertService, FakeBot]:
    """The service under test with a fresh fake bot unless the caller brings its own."""
    sender = FakeBot() if bot is None else bot
    service = AlertService(
        sender,
        settings or FakeSettings(),
        repos.requests,
        repos.uploads,
        repos.status,
        repos.notices,
        TZ,
        clock,  # type: ignore[arg-type]  # only the loop test passes a clock
    )
    return service, sender


async def pass_live(service: AlertService, repos: Repos, now: datetime) -> list[SentAlert]:
    """One pass at `now` with a fresh heartbeat, so only the alert under test can fire."""
    await _beat(repos, now)
    return await service.check_once(now)


async def _request(
    repos: Repos,
    *,
    user_id: int = 1,
    username: str | None = "gleb",
    status: RequestStatus = RequestStatus.OK,
    tokens: int = 100,
    web_calls: int = 0,
    error: str | None = None,
    created_at: datetime | None = None,
) -> None:
    """Add one request row; `created_at` is rewritten in SQL because `add` stamps the real now()."""
    if await repos.users.get(user_id) is None:
        await repos.users.upsert_seen(user_id, username, default_role=Role.USER, default_mode=Mode.NORMAL)
    await repos.requests.add(
        RequestRecord(
            user_id=user_id,
            conversation_id=None,
            model="m1",
            status=status,
            attempts=1,
            latency_ms=1000,
            approx_tokens=tokens,
            error=error,
            usage=TokenUsage(),
            web_calls=web_calls,
        )
    )
    if created_at is not None:
        await _backdate(repos.db, "requests", created_at)


async def _backdate(db: Database, table: str, created_at: datetime) -> None:
    await db.conn.execute(
        f"UPDATE {table} SET created_at = ? WHERE rowid = (SELECT MAX(rowid) FROM {table})",
        (created_at.astimezone(UTC).isoformat(),),
    )
    await db.conn.commit()


async def _images(repos: Repos, count: int, created_at: datetime) -> None:
    """`count` generated pictures, stamped at `created_at` (the repo stamps the real now)."""
    if await repos.users.get(1) is None:
        await repos.users.upsert_seen(1, "gleb", default_role=Role.USER, default_mode=Mode.NORMAL)
    for _ in range(count):
        await repos.uploads.add(1, UploadKind.GEN_IMAGE, "i.png", 1000)
        await _backdate(repos.db, "uploads", created_at)


async def _beat(repos: Repos, heartbeat_at: datetime) -> None:
    """A heartbeat row of a chosen age (StatusRepo.beat itself always stamps the real now)."""
    await repos.status.beat(heartbeat_at, 0, 0)
    await repos.db.conn.execute(
        "UPDATE bot_status SET started_at = ?, heartbeat_at = ? WHERE id = 1",
        (heartbeat_at.astimezone(UTC).isoformat(), heartbeat_at.astimezone(UTC).isoformat()),
    )
    await repos.db.conn.commit()


def keys(sent: list[SentAlert]) -> list[str]:
    return [notice.key for notice in sent]


# ---- the notice keys (pure) -----------------------------------------------------------------------


def test_burst_key_is_bucketed_into_local_quarters() -> None:
    """A second pass within the same window must produce the very same key."""
    assert burst_key(NOW, TZ) == "gw-burst:2026-01-15T15:00"
    assert burst_key(NOW + timedelta(minutes=14), TZ) == "gw-burst:2026-01-15T15:00"
    assert burst_key(NOW + timedelta(minutes=59, seconds=59), TZ) == "gw-burst:2026-01-15T15:45"
    assert burst_key(NOW + timedelta(hours=1), TZ) == "gw-burst:2026-01-15T16:00"


def test_per_user_and_per_day_keys_change_with_the_user_and_the_local_day() -> None:
    assert overspend_key(NOW, TZ, 7) == f"user-tokens:{LOCAL_DAY}:7"
    assert overspend_key(NOW, TZ, 7) != overspend_key(NOW + timedelta(days=1), TZ, 7)
    assert overspend_key(NOW, TZ, 7) != overspend_key(NOW, TZ, 8)
    assert daily_key(NOW, TZ) == f"daily:{LOCAL_DAY}"
    assert daily_key(NOW + timedelta(days=1), TZ) == "daily:2026-01-16"


def test_quota_keys_use_the_utc_day_and_the_service() -> None:
    assert quota_key("web", NOW) == "quota:web:2026-01-15"
    assert quota_key("neurons", NOW) == "quota:neurons:2026-01-15"
    # 21:30 UTC is already tomorrow locally (+03): the services still reset on the UTC day, not ours
    late = datetime(2026, 1, 15, 21, 30, tzinfo=UTC)
    assert quota_key("web", late) == "quota:web:2026-01-15"


def test_stale_key_carries_the_heartbeat_it_saw() -> None:
    """A newer heartbeat means a new outage, which must be reported again."""
    old = NOW - timedelta(minutes=10)
    assert stale_key(old) != stale_key(old + timedelta(hours=1))
    assert stale_key(old).startswith("bot-stale:2026-01-15T11:50")
    assert missing_key(NOW, TZ) == f"bot-missing:{LOCAL_DAY}"
    assert missing_key(NOW, TZ) != missing_key(NOW + timedelta(days=1), TZ)


def test_quota_targets_report_only_the_plans_over_the_line() -> None:
    assert quota_targets(Quotas(images=1, neurons_limit=10_000)) == []
    assert quota_targets(Quotas(images=100, neurons_limit=10_000)) == [("neurons", 10_400, 10_000)]


# ---- alert 1: gateway failure burst ---------------------------------------------------------------


async def test_burst_silent_below_the_threshold_and_fires_at_it(repos: Repos) -> None:
    service, bot = make_service(repos, FakeSettings(alert_failed_count=3))
    for _ in range(2):
        await _request(
            repos, status=RequestStatus.FAILED, error="timeout", created_at=NOW - timedelta(minutes=5)
        )
    assert await pass_live(service, repos, NOW) == []
    assert bot.sent == []

    await _request(repos, status=RequestStatus.FAILED, error="429", created_at=NOW - timedelta(minutes=1))
    sent = await pass_live(service, repos, NOW)
    assert keys(sent) == ["gw-burst:2026-01-15T15:00"]
    assert sent[0].delivered == 2  # both admins
    assert "Всплеск ошибок" in bot.texts[0]
    assert "3 (порог 3)" in bot.texts[0]
    assert "429" in bot.texts[0]  # a sample error is quoted, so the admin sees the reason


async def test_burst_ignores_old_failures_stubs_and_ok_requests(repos: Repos) -> None:
    service, _ = make_service(repos, FakeSettings(alert_failed_count=2))
    await _request(repos, status=RequestStatus.FAILED, created_at=NOW - timedelta(minutes=40))  # too old
    await _request(repos, status=RequestStatus.STUB, created_at=NOW - timedelta(minutes=1))
    await _request(repos, status=RequestStatus.OK, created_at=NOW - timedelta(minutes=1))
    assert await pass_live(service, repos, NOW) == []


async def test_burst_fires_once_per_window_not_once_per_pass(repos: Repos) -> None:
    """The point of the dedup: the loop passes every minute and must not repeat the same burst."""
    service, bot = make_service(repos, FakeSettings(alert_failed_count=1))
    await _request(repos, status=RequestStatus.FAILED, created_at=NOW - timedelta(minutes=1))

    assert len(await pass_live(service, repos, NOW)) == 1
    assert await pass_live(service, repos, NOW + timedelta(minutes=1)) == []
    assert await pass_live(service, repos, NOW + timedelta(minutes=14)) == []
    assert len(bot.sent) == 2  # one alert, two admins

    await _request(repos, status=RequestStatus.FAILED, created_at=NOW + timedelta(minutes=16))
    assert keys(await pass_live(service, repos, NOW + timedelta(minutes=16))) == ["gw-burst:2026-01-15T15:15"]


async def test_burst_disabled_with_zero(repos: Repos) -> None:
    service, bot = make_service(repos, FakeSettings(alert_failed_count=0))
    for _ in range(10):
        await _request(repos, status=RequestStatus.FAILED, created_at=NOW - timedelta(minutes=1))
    assert await pass_live(service, repos, NOW) == []
    assert bot.sent == []


# ---- alert 2: user overspending -------------------------------------------------------------------


async def test_overspend_fires_once_per_user_per_day(repos: Repos) -> None:
    service, bot = make_service(repos, FakeSettings(alert_user_tokens=10_000))
    await _request(repos, user_id=1, username="gleb", tokens=6_000, created_at=NOW - timedelta(minutes=5))
    await _request(repos, user_id=1, username="gleb", tokens=6_000, created_at=NOW - timedelta(minutes=4))
    await _request(repos, user_id=2, username="other", tokens=100, created_at=NOW - timedelta(minutes=4))
    await _request(repos, user_id=3, username="heavy", tokens=50_000, created_at=NOW - timedelta(minutes=4))

    sent = await pass_live(service, repos, NOW)
    assert sorted(keys(sent)) == [f"user-tokens:{LOCAL_DAY}:1", f"user-tokens:{LOCAL_DAY}:3"]
    assert all("порога расхода" in text for text in bot.texts)
    assert any("12 000 токенов" in text for text in bot.texts)  # fmt_tokens: exact and grouped
    assert any("50 000" in text for text in bot.texts)

    assert await pass_live(service, repos, NOW + timedelta(minutes=30)) == []  # same local day


async def test_overspend_counts_only_the_current_local_day(repos: Repos) -> None:
    """At +03 the local day starts at 21:00 UTC: a spend yesterday must not carry over."""
    service, _ = make_service(repos, FakeSettings(alert_user_tokens=10_000))
    await _request(repos, tokens=50_000, created_at=NOW - timedelta(hours=20))
    assert await pass_live(service, repos, NOW) == []

    await _request(repos, tokens=50_000, created_at=NOW - timedelta(minutes=2))
    assert keys(await pass_live(service, repos, NOW)) == [f"user-tokens:{LOCAL_DAY}:1"]


async def test_overspend_alerts_again_on_the_next_day(repos: Repos) -> None:
    service, _ = make_service(repos, FakeSettings(alert_user_tokens=10_000))
    await _request(repos, tokens=50_000, created_at=NOW - timedelta(minutes=2))
    assert len(await pass_live(service, repos, NOW)) == 1

    tomorrow = NOW + timedelta(days=1)
    await _request(repos, tokens=50_000, created_at=tomorrow - timedelta(minutes=2))
    assert keys(await pass_live(service, repos, tomorrow)) == ["user-tokens:2026-01-16:1"]


@pytest.mark.parametrize("threshold", [-1, 0])
async def test_overspend_disabled(repos: Repos, threshold: int) -> None:
    service, bot = make_service(repos, FakeSettings(alert_user_tokens=threshold))
    await _request(repos, tokens=9_000_000, created_at=NOW - timedelta(minutes=2))
    assert await pass_live(service, repos, NOW) == []
    assert bot.sent == []


# ---- alert 3: the main bot went quiet -------------------------------------------------------------


async def test_bot_quiet_fires_on_a_stale_heartbeat(repos: Repos) -> None:
    service, bot = make_service(repos)
    stale = NOW - timedelta(seconds=BOT_STALE_AFTER_S + 30)
    await _beat(repos, stale)

    sent = await service.check_once(NOW)
    assert keys(sent) == [stale_key(stale)]
    assert "Основной бот молчит" in bot.texts[0]
    assert fmt_dt(stale, TZ) in bot.texts[0]

    assert await service.check_once(NOW + timedelta(minutes=5)) == []  # the same outage, once


async def test_bot_quiet_silent_while_the_heartbeat_is_fresh(repos: Repos) -> None:
    service, bot = make_service(repos)
    await _beat(repos, NOW - timedelta(seconds=BOT_STALE_AFTER_S))
    assert await service.check_once(NOW) == []
    assert bot.sent == []


async def test_a_recovered_then_again_silent_bot_alerts_twice(repos: Repos) -> None:
    """A newer heartbeat is a different outage, so it deserves its own notice."""
    service, _ = make_service(repos)
    stale = NOW - timedelta(seconds=BOT_STALE_AFTER_S + 30)
    await _beat(repos, stale)
    assert len(await service.check_once(NOW)) == 1

    live = NOW + timedelta(hours=2)
    await _beat(repos, live)
    assert await service.check_once(live + timedelta(minutes=1)) == []  # fresh again: silence

    second = live + timedelta(hours=1)
    await _beat(repos, second)
    assert keys(await service.check_once(second + timedelta(hours=1))) == [stale_key(second)]


async def test_bot_quiet_when_it_never_reported(repos: Repos) -> None:
    """No heartbeat row at all: a reminder in every local day, but not on every pass."""
    service, bot = make_service(repos)
    assert keys(await service.check_once(NOW)) == [missing_key(NOW, TZ)]
    assert "не отчитывался" in bot.texts[0]
    assert await service.check_once(NOW + timedelta(minutes=5)) == []  # the same day
    assert len(bot.sent) == 2  # one alert, two admins

    tomorrow = NOW + timedelta(days=1)
    assert keys(await service.check_once(tomorrow)) == [missing_key(tomorrow, TZ)]


# ---- alert 4: external quota pressure -------------------------------------------------------------


async def test_quota_pressure_fires_per_service_once_a_utc_day(repos: Repos) -> None:
    settings = FakeSettings(cloudflare_daily_neurons=2_000)
    service, bot = make_service(repos, settings)
    await _images(repos, 16, NOW - timedelta(minutes=30))  # 16 * 104 = 1664 neurons, 83%

    sent = await pass_live(service, repos, NOW)
    assert sorted(keys(sent)) == ["quota:neurons:2026-01-15"]
    assert any("Квота Cloudflare" in text and "83%" in text for text in bot.texts)

    assert await pass_live(service, repos, NOW + timedelta(minutes=10)) == []  # same UTC day


async def test_quota_pressure_silent_under_the_line(repos: Repos) -> None:
    settings = FakeSettings(cloudflare_daily_neurons=10_000)
    service, bot = make_service(repos, settings)
    await _images(repos, 76, NOW - timedelta(minutes=30))  # 7904 neurons: 79%
    assert await pass_live(service, repos, NOW) == []
    assert bot.sent == []


async def test_quota_pressure_uses_the_utc_reset_not_the_local_day(repos: Repos) -> None:
    """Pictures made in the previous UTC day are already forgotten by Cloudflare."""
    service, _ = make_service(repos, FakeSettings(cloudflare_daily_neurons=2_000))
    await _images(repos, 30, NOW - timedelta(days=1))
    assert await pass_live(service, repos, NOW) == []


# ---- alert 5: daily summary -----------------------------------------------------------------------


async def test_daily_summary_fires_only_in_its_local_hour(repos: Repos) -> None:
    bot = FakeBot()
    early, _ = make_service(repos, FakeSettings(daily_report_hour=14), bot)
    await _request(repos, tokens=5_000, created_at=NOW - timedelta(minutes=5))
    assert await pass_live(early, repos, NOW) == []  # 15:00 locally: the digest waits for 14:00

    late, _ = make_service(repos, FakeSettings(daily_report_hour=16), bot)
    assert await pass_live(late, repos, NOW) == []
    assert bot.sent == []

    on_time, _ = make_service(repos, FakeSettings(daily_report_hour=15), bot)
    sent = await pass_live(on_time, repos, NOW)
    assert keys(sent) == [f"daily:{LOCAL_DAY}"]
    assert "Итоги дня" in bot.texts[0]
    assert "15.01.2026" in bot.texts[0]

    assert await pass_live(on_time, repos, NOW + timedelta(minutes=20)) == []  # once a day
    assert len(bot.sent) == 2  # one digest, two admins


async def test_daily_summary_counts_and_ranks_the_local_day(repos: Repos) -> None:
    service, bot = make_service(repos, FakeSettings(daily_report_hour=15))
    await _request(repos, user_id=1, username="small", tokens=10, created_at=NOW - timedelta(minutes=5))
    await _request(repos, user_id=2, username="big", tokens=9_000, created_at=NOW - timedelta(minutes=5))
    await _request(
        repos, user_id=2, username="big", status=RequestStatus.FAILED, created_at=NOW - timedelta(minutes=2)
    )
    await _request(repos, user_id=3, username="old", tokens=9_999, created_at=NOW - timedelta(days=2))

    await pass_live(service, repos, NOW)
    text = bot.texts[0]
    assert "Запросы: 3 · ✅ 2 · 🙊 0 · ❌ 1" in text
    assert "1. @big" in text and "2. @small" in text
    assert "@old" not in text  # yesterday's spender
    assert text.index("@big") < text.index("@small")


async def test_daily_summary_on_a_quiet_day(repos: Repos) -> None:
    service, bot = make_service(repos, FakeSettings(daily_report_hour=15))
    assert keys(await pass_live(service, repos, NOW)) == [f"daily:{LOCAL_DAY}"]
    assert "Запросов не было." in bot.texts[0]
    assert "Больше всех" not in bot.texts[0]


async def test_daily_summary_disabled(repos: Repos) -> None:
    service, bot = make_service(repos, FakeSettings(daily_report_hour=-1))
    await _request(repos, tokens=5_000, created_at=NOW - timedelta(minutes=5))
    assert await pass_live(service, repos, NOW) == []
    assert bot.sent == []


# ---- delivery -------------------------------------------------------------------------------------


async def test_one_blocked_admin_does_not_stop_the_others(repos: Repos) -> None:
    settings = FakeSettings(admin_ids=frozenset({1, 2, 3}), alert_failed_count=1)
    bot = FakeBot(refuse=frozenset({2}))
    service, _ = make_service(repos, settings, bot)
    await _request(repos, status=RequestStatus.FAILED, created_at=NOW - timedelta(minutes=1))

    sent = await pass_live(service, repos, NOW)
    assert len(sent) == 1
    assert sent[0].delivered == 2  # 1 and 3 got it, 2 was refused
    assert bot.targets == [1, 3]


async def test_an_alert_that_reached_nobody_is_not_retried_forever(repos: Repos) -> None:
    """The claim is spent on purpose: an admin who blocked the bot must not flood the loop."""
    settings = FakeSettings(admin_ids=frozenset({2}), alert_failed_count=1)
    bot = FakeBot(refuse=frozenset({2}))
    service, _ = make_service(repos, settings, bot)
    await _request(repos, status=RequestStatus.FAILED, created_at=NOW - timedelta(minutes=1))

    assert (await pass_live(service, repos, NOW))[0].delivered == 0
    assert await pass_live(service, repos, NOW + timedelta(minutes=1)) == []
    assert bot.sent == []


async def test_alerts_go_only_to_the_admins_of_this_bot(repos: Repos) -> None:
    """ADMIN_IDS is the whole audience; nobody else is ever messaged."""
    service, bot = make_service(repos, FakeSettings(admin_ids=frozenset({100, 200}), daily_report_hour=15))
    assert len(await pass_live(service, repos, NOW)) == 1
    assert bot.targets == [100, 200]


class Exploding(AlertService):
    """The real service with a first pass that throws: the loop has to survive that."""

    def __init__(self, bot: FakeBot, settings: FakeSettings, repos: Repos, now: datetime) -> None:
        super().__init__(
            bot, settings, repos.requests, repos.uploads, repos.status, repos.notices, TZ, lambda: now
        )
        self.calls = 0

    async def check_once(self, now: datetime) -> list[SentAlert]:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("database is locked")
        return await super().check_once(now)


async def test_the_loop_survives_a_failing_pass(repos: Repos) -> None:
    bot = FakeBot()
    service = Exploding(bot, FakeSettings(daily_report_hour=15), repos, NOW)

    task = start_alerts(service, interval_s=0.005, first_delay_s=0)
    try:
        await asyncio.sleep(0.05)
        assert service.calls >= 2  # the first pass blew up, the later passes still ran
        assert any("Итоги дня" in text for text in bot.texts)  # the digest got out anyway
    finally:
        task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


class _Broken(RequestRepo):
    """A request repo whose every read fails, as a bad night of the database would."""

    def __init__(self) -> None:  # deliberately not calling the parent: no database is needed
        pass

    async def count_errors(self, **_: object) -> int:
        raise RuntimeError("database is locked")


async def test_a_broken_check_does_not_stop_the_next_alerts(repos: Repos) -> None:
    """The loop catches whatever one check throws; the alerts after it still go out."""
    bot = FakeBot()
    broken = AlertService(
        bot, FakeSettings(daily_report_hour=15), _Broken(), repos.uploads, repos.status, repos.notices, TZ
    )
    with pytest.raises(RuntimeError):
        await broken.check_once(NOW)

    healthy, _ = make_service(repos, FakeSettings(daily_report_hour=15), bot)
    assert keys(await pass_live(healthy, repos, NOW)) == [f"daily:{LOCAL_DAY}"]
    assert "Итоги дня" in bot.texts[0]


# ---- the loop -------------------------------------------------------------------------------------


async def test_start_alerts_makes_passes_and_cancels_cleanly(repos: Repos) -> None:
    """The real `run` loop: a frozen clock makes every pass identical, so nothing repeats."""
    service, bot = make_service(repos, FakeSettings(alert_failed_count=1), clock=lambda: NOW)
    await _request(repos, status=RequestStatus.FAILED, created_at=NOW - timedelta(minutes=1))
    await _beat(repos, NOW)

    task = start_alerts(service, interval_s=0.005, first_delay_s=0)  # a 5 ms interval: 50 ms total
    try:
        await asyncio.sleep(0.05)  # several passes
        assert not task.done()
        assert len(bot.sent) == 2  # one alert, two admins: the later passes repeat nothing
    finally:
        task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    assert task.cancelled()


async def test_the_loop_waits_before_its_first_pass(repos: Repos) -> None:
    """Both bots start together: a pass at t=0 would see no heartbeat and cry wolf."""
    service, bot = make_service(repos, FakeSettings(alert_failed_count=0, daily_report_hour=-1))

    task = start_alerts(service, interval_s=0.005, first_delay_s=0.05)
    try:
        await asyncio.sleep(0.02)
        assert bot.sent == []  # still inside the grace period, nothing looked at the database
    finally:
        task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


# ---- render helpers (pure) ------------------------------------------------------------------------


def test_render_burst_escapes_usernames_models_and_errors() -> None:
    entry = ErrorEntry(
        id=1,
        user_id=1,
        username="a<b>",
        model="m<x>",
        status=RequestStatus.FAILED,
        error="boom <500>",
        attempts=3,
        latency_ms=100,
        created_at=NOW - timedelta(minutes=1),
    )
    text = render_burst(7, 5, [entry], NOW, TZ)
    assert "&lt;b&gt;" in text and "&lt;x&gt;" in text and "&lt;500&gt;" in text
    assert "7 (порог 5)" in text


def test_render_overspend_names_the_user_and_both_numbers() -> None:
    user = UserUsage(tg_id=5, username="gleb", role=Role.USER, requests=12, approx_tokens=64_000, last_at=NOW)
    text = render_overspend(user, 50_000)
    assert "@gleb" in text
    assert "64 000 токенов" in text and "порог 50 000" in text
    assert "запросов: 12" in text


def test_render_bot_quiet_has_both_phrasings() -> None:
    status = BotStatus(
        started_at=NOW - timedelta(hours=2),
        heartbeat_at=NOW - timedelta(minutes=10),
        queue_depth=3,
        active_jobs=1,
    )
    text = render_bot_quiet(status, NOW, TZ)
    assert "молчит" in text and "10 мин назад" in text
    assert "В очереди: 3, в работе: 1" in text
    assert "не отчитывался" in render_bot_quiet(None, NOW, TZ)


def test_render_quota_shows_the_percentage() -> None:
    text = render_quota("neurons", 9_000, 10_000)
    assert "9000 из 10000 (90%)" in text
    assert "Cloudflare" in text


def test_render_daily_reports_counts_and_numbers() -> None:
    report = RequestReport(
        requests=4,
        ok=3,
        stub=0,
        failed=1,
        avg_latency_ms=0,
        total_tokens=1234,
        usage=TokenUsage(),
        web_calls=0,
        by_model=(),
        last_at=NOW,
    )
    text = render_daily(NOW.date(), report, ())
    assert "15.01.2026" in text
    assert "Запросы: 4 · ✅ 3 · 🙊 0 · ❌ 1" in text
    assert "1.2k" in text  # short_number(1234)
    assert "Больше всех" not in text


# ---- public fire(): reused by the model probe loop ------------------------------------------------


class MarkupCapturingBot:
    """A bot whose send_message also accepts reply_markup, to check `fire` passes it through."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, str, object | None]] = []

    async def send_message(self, chat_id: int, text: str, reply_markup: object | None = None) -> None:
        self.calls.append((chat_id, text, reply_markup))


async def test_fire_passes_reply_markup_when_given(repos: Repos) -> None:
    bot = MarkupCapturingBot()
    service, _ = make_service(repos, bot=bot)  # type: ignore[arg-type]
    markup = object()

    sent = await service.fire("k1", "text", markup)

    assert sent is not None
    assert all(m is markup for _, _, m in bot.calls)


async def test_fire_omits_reply_markup_when_not_given(repos: Repos) -> None:
    """The existing fakes in this file define send_message(chat_id, text) with no markup param:
    `fire` must not pass reply_markup at all when it is None, or those fakes would break."""
    service, bot = make_service(repos)

    sent = await service.fire("k2", "text")

    assert sent is not None
    assert bot.sent == [(100, "text"), (200, "text")]


async def test_fire_deduplicates_like_the_scheduled_checks(repos: Repos) -> None:
    service, bot = make_service(repos)

    first = await service.fire("k3", "text")
    second = await service.fire("k3", "text")

    assert first is not None
    assert second is None
    assert len(bot.sent) == 2  # only the first call reached the admins
