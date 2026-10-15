"""WP-5: the admin-bot storage layer - reports, errors, models, activity, limits, races.

Every test drives the real repositories against an in-memory database, so the SQL itself
is under test: the admin bot has no other way to get its numbers.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from bot.domain import (
    CONTEXT_LIMIT_MAX,
    Mode,
    RequestRecord,
    RequestStatus,
    Role,
    TokenUsage,
    UploadKind,
    User,
)
from bot.storage.db import Database
from bot.storage.repositories import (
    LEADERBOARD_ORDER,
    ConversationRepo,
    NoticeRepo,
    RequestRepo,
    StatusRepo,
    UploadRepo,
    UsageRepo,
    UserRepo,
)

TZ = timezone(timedelta(hours=3))  # the admin's local zone, as in the shipped defaults


@pytest.fixture
async def db() -> Database:
    database = Database(":memory:")
    await database.connect()
    try:
        yield database
    finally:
        await database.close()


async def _user(db: Database, tg_id: int = 1, username: str | None = "gleb", role: Role = Role.USER) -> None:
    await UserRepo(db).upsert_seen(tg_id, username, default_role=role, default_mode=Mode.NORMAL)


async def _request(
    db: Database,
    *,
    user_id: int = 1,
    model: str = "m1",
    status: RequestStatus = RequestStatus.OK,
    tokens: int = 100,
    latency_ms: int = 1000,
    usage: TokenUsage | None = None,
    web_calls: int = 0,
    error: str | None = None,
    created_at: datetime | None = None,
) -> None:
    """Insert one request row; `created_at` is backdated directly because `add` stamps now()."""
    repo = RequestRepo(db)
    await repo.add(
        RequestRecord(
            user_id=user_id,
            conversation_id=None,
            model=model,
            status=status,
            attempts=1,
            latency_ms=latency_ms,
            approx_tokens=tokens,
            error=error,
            usage=usage or TokenUsage(),
            web_calls=web_calls,
        )
    )
    if created_at is not None:
        await db.conn.execute(
            "UPDATE requests SET created_at = ? WHERE id = (SELECT MAX(id) FROM requests)",
            (created_at.astimezone(UTC).isoformat(),),
        )
        await db.conn.commit()


# ---- search escapes LIKE wildcards ---------------------------------------------------------------


async def test_search_escapes_percent_and_underscore(db: Database) -> None:
    """A username containing % or _ must match literally, not as a LIKE wildcard."""
    repo = UserRepo(db)
    await _user(db, 1, "100%real")
    await _user(db, 2, "a_b")
    await _user(db, 3, "plain")

    assert [u.tg_id for u in await repo.search("%")] == [1]
    assert [u.tg_id for u in await repo.search("_")] == [2]
    assert [u.tg_id for u in await repo.search("plain")] == [3]
    assert await repo.search("nothing-like-this") == []


async def test_search_matches_exact_id_or_partial_username(db: Database) -> None:
    """The admin types either a full id or a piece of the username: both must work."""
    repo = UserRepo(db)
    await _user(db, 123456, "someone")

    assert [u.tg_id for u in await repo.search("123456")] == [123456]
    assert [u.tg_id for u in await repo.search("someone")] == [123456]
    assert [u.tg_id for u in await repo.search("@some")] == [123456]  # the @ form is accepted too

    for i in range(5):
        await _user(db, 1000 + i, f"same{i}")
    assert len(await repo.search("same", limit=3)) == 3
    assert await repo.search("") == []  # an empty query is not a "select everything"


# ---- user pages, counters, limits ----------------------------------------------------------------


async def test_list_page_and_count_by_role(db: Database) -> None:
    repo = UserRepo(db)
    await _user(db, 1, "a", Role.USER)
    await _user(db, 2, "b", Role.USER)
    await _user(db, 3, "c", Role.PENDING)

    assert await repo.count() == 3
    assert await repo.count(Role.USER) == 2
    assert await repo.count(Role.PENDING) == 1

    first = await repo.list_page(Role.USER, limit=1, offset=0)
    second = await repo.list_page(Role.USER, limit=1, offset=1)
    assert [u.tg_id for u in first] == [2]  # newest first
    assert [u.tg_id for u in second] == [1]
    assert len(await repo.list_page(None, limit=10, offset=0)) == 3


async def test_ids_by_roles_is_the_broadcast_audience(db: Database) -> None:
    repo = UserRepo(db)
    await _user(db, 1, "a", Role.USER)
    await _user(db, 2, "b", Role.ADMIN)
    await _user(db, 3, "c", Role.PENDING)
    await _user(db, 4, "d", Role.BLOCKED)

    assert sorted(await repo.ids_by_roles((Role.USER, Role.ADMIN))) == [1, 2]
    assert await repo.ids_by_roles((Role.PENDING,)) == [3]
    assert await repo.ids_by_roles(()) == []


async def test_set_context_limit_round_trip_and_unknown_user(db: Database) -> None:
    repo = UserRepo(db)
    await _user(db, 1)

    assert await repo.set_context_limit(1, 50_000) is True
    user = await repo.get(1)
    assert user is not None and user.context_limit == 50_000

    assert await repo.set_context_limit(1, None) is True  # back to the global budget
    user = await repo.get(1)
    assert user is not None and user.context_limit is None

    assert await repo.set_context_limit(999, 1_000) is False


async def test_set_daily_limit_round_trip(db: Database) -> None:
    repo = UserRepo(db)
    await _user(db, 1)

    assert await repo.set_daily_limit(1, 7) is True
    user = await repo.get(1)
    assert user is not None and user.daily_limit == 7

    assert await repo.set_daily_limit(999, 7) is False


async def test_context_limit_bounds_are_sane() -> None:
    """The admin bot clamps against these, so they must bracket the shipping global budget."""
    assert CONTEXT_LIMIT_MAX >= 350_000


# ---- request reports -----------------------------------------------------------------------------


async def test_report_splits_tokens_by_kind(db: Database) -> None:
    await _user(db, 1)
    await _request(
        db,
        usage=TokenUsage(input=10, output=20, images=30, service=40, context=50, tools=60),
        web_calls=2,
    )
    await _request(db, status=RequestStatus.FAILED, tokens=5, error="boom")

    report = await RequestRepo(db).report(1, None)
    assert report.requests == 2
    assert (report.ok, report.failed) == (1, 1)
    assert report.total_tokens == 105
    assert report.usage.input == 10
    assert report.usage.output == 20
    assert report.usage.images == 30
    assert report.usage.service == 40
    assert report.usage.context == 50
    assert report.usage.tools == 60
    assert report.web_calls == 2
    assert report.last_at is not None


async def test_report_groups_by_model_most_tokens_first(db: Database) -> None:
    await _user(db, 1)
    await _request(db, model="small", tokens=10)
    await _request(db, model="big", tokens=900)
    await _request(db, model="big", tokens=100)

    report = await RequestRepo(db).report(1, None)
    assert [(m.model, m.tokens, m.requests) for m in report.by_model] == [
        ("big", 1000, 2),
        ("small", 10, 1),
    ]


async def test_report_filters_by_user_and_period(db: Database) -> None:
    await _user(db, 1)
    await _user(db, 2)
    now = datetime.now(UTC)
    await _request(db, user_id=1, tokens=10, created_at=now - timedelta(days=10))
    await _request(db, user_id=1, tokens=20, created_at=now)
    await _request(db, user_id=2, tokens=40, created_at=now)

    repo = RequestRepo(db)
    assert (await repo.report(1, None)).requests == 2
    assert (await repo.report(1, now - timedelta(days=1))).total_tokens == 20
    assert (await repo.report(None, None)).requests == 3  # everyone
    assert (await repo.report(None, now - timedelta(days=1))).total_tokens == 60


async def test_report_on_empty_db_is_all_zeros(db: Database) -> None:
    report = await RequestRepo(db).report(None, None)
    assert report.requests == 0
    assert report.total_tokens == 0
    assert report.avg_latency_ms == 0
    assert report.by_model == ()
    assert report.last_at is None


# ---- error journal -------------------------------------------------------------------------------


async def test_errors_include_failures_and_stubs_newest_first(db: Database) -> None:
    await _user(db, 1, "gleb")
    await _request(db, status=RequestStatus.OK)
    await _request(db, status=RequestStatus.FAILED, error="timeout")
    await _request(db, status=RequestStatus.STUB, error=None)

    repo = RequestRepo(db)
    entries = await repo.errors(limit=10)
    assert [e.status for e in entries] == [RequestStatus.STUB, RequestStatus.FAILED]
    assert entries[1].error == "timeout"
    assert entries[1].username == "gleb"  # joined from users for the admin's convenience
    assert await repo.count_errors() == 2


async def test_errors_failed_only_excludes_stubs(db: Database) -> None:
    await _user(db, 1)
    await _request(db, status=RequestStatus.FAILED, error="boom")
    await _request(db, status=RequestStatus.STUB)

    repo = RequestRepo(db)
    assert await repo.count_errors(failed_only=True) == 1
    entries = await repo.errors(limit=10, failed_only=True)
    assert [e.status for e in entries] == [RequestStatus.FAILED]


async def test_errors_paging_and_unknown_user_filter(db: Database) -> None:
    await _user(db, 1)
    for i in range(5):
        await _request(db, status=RequestStatus.FAILED, error=f"e{i}")

    repo = RequestRepo(db)
    first = await repo.errors(limit=2, offset=0)
    second = await repo.errors(limit=2, offset=2)
    assert len(first) == 2 and len(second) == 2
    assert {e.id for e in first}.isdisjoint({e.id for e in second})

    # A user with no failures yet: an empty page, not an error.
    assert await repo.errors(limit=10, user_id=999) == []


async def test_errors_expose_the_username_for_the_journal(db: Database) -> None:
    """The journal shows who it was: the join must carry the username through."""
    await _user(db, 1, "gleb")
    await _request(db, status=RequestStatus.FAILED, error="boom")

    entries = await RequestRepo(db).errors(limit=10)
    assert len(entries) == 1
    assert entries[0].username == "gleb"
    assert entries[0].user_id == 1
    assert entries[0].attempts >= 1


# ---- model stats ---------------------------------------------------------------------------------


async def test_model_stats_split_and_latency_only_for_ok(db: Database) -> None:
    await _user(db, 1)
    await _request(db, model="a", status=RequestStatus.OK, latency_ms=100)
    await _request(db, model="a", status=RequestStatus.OK, latency_ms=300)
    await _request(db, model="a", status=RequestStatus.FAILED, latency_ms=90_000)
    await _request(db, model="b", status=RequestStatus.STUB, latency_ms=5)

    stats = {s.model: s for s in await RequestRepo(db).model_stats(None, None)}
    assert stats["a"].requests == 3
    assert (stats["a"].ok, stats["a"].failed) == (2, 1)
    assert stats["a"].avg_ok_latency_ms == 200  # the 90s failure is excluded
    assert stats["a"].max_ok_latency_ms == 300
    assert stats["b"].stub == 1
    assert stats["b"].avg_ok_latency_ms == 0  # no answered request at all


async def test_model_stats_empty(db: Database) -> None:
    assert await RequestRepo(db).model_stats(None, None) == []


# ---- activity by day -----------------------------------------------------------------------------


async def test_by_day_keeps_empty_days_and_uses_local_midnights(db: Database) -> None:
    await _user(db, 1)
    today = date(2026, 1, 15)
    # 21:00 UTC on the 14th is 00:00 on the 15th in +3: it must land on the local 15th.
    await _request(db, tokens=5, created_at=datetime(2026, 1, 14, 21, 0, tzinfo=UTC))
    await _request(db, tokens=7, created_at=datetime(2026, 1, 15, 9, 0, tzinfo=UTC))

    rows = await RequestRepo(db).by_day(None, 3, today, TZ)
    assert [(r.day, r.requests, r.tokens) for r in rows] == [
        (date(2026, 1, 13), 0, 0),
        (date(2026, 1, 14), 0, 0),
        (date(2026, 1, 15), 2, 12),
    ]


async def test_by_day_filters_by_user(db: Database) -> None:
    await _user(db, 1)
    await _user(db, 2)
    today = date(2026, 1, 15)
    await _request(db, user_id=1, tokens=1, created_at=datetime(2026, 1, 15, 9, 0, tzinfo=UTC))
    await _request(db, user_id=2, tokens=9, created_at=datetime(2026, 1, 15, 9, 0, tzinfo=UTC))

    rows = await RequestRepo(db).by_day(1, 1, today, TZ)
    assert rows == rows and len(rows) == 1
    assert rows[0].requests == 1 and rows[0].tokens == 1


# ---- heavy users and leaderboard -----------------------------------------------------------------


async def test_heavy_users_respects_the_threshold(db: Database) -> None:
    await _user(db, 1, "big")
    await _user(db, 2, "small")
    since = datetime.now(UTC) - timedelta(hours=1)
    await _request(db, user_id=1, tokens=5001)
    await _request(db, user_id=2, tokens=10)

    rows = await RequestRepo(db).heavy_users(since, min_tokens=5000)
    assert [r.tg_id for r in rows] == [1]


async def test_leaderboard_orders_are_all_queryable(db: Database) -> None:
    await _user(db, 1, "one")
    await _user(db, 2, "two")
    await _request(db, user_id=1, tokens=100, latency_ms=100)
    await _request(db, user_id=2, tokens=10, latency_ms=900)

    repo = RequestRepo(db)
    for order in LEADERBOARD_ORDER:
        rows = await repo.leaderboard(order, limit=10)
        assert rows, f"order {order} returned nothing"
        assert len(rows) == 2

    by_requests = await repo.leaderboard("requests", limit=10)
    assert {r.tg_id for r in by_requests} == {1, 2}


async def test_leaderboard_rejects_unknown_order_so_the_screen_can_fall_back(db: Database) -> None:
    """The repo refuses an unknown key; `home.top_screen` is the layer that falls back.

    Callback data is user-supplied, so the SQL builder must never take an arbitrary string,
    and the screen must not let one reach it.
    """
    from adminbot.home import top_screen

    await _user(db, 1, "one")
    await _request(db, user_id=1)

    with pytest.raises(KeyError):
        await RequestRepo(db).leaderboard("'; DROP TABLE users; --", limit=5)

    text, _ = await top_screen("nonsense", RequestRepo(db), datetime.now(UTC))
    assert "Рейтинг" in text  # quietly rendered as the default ordering instead of crashing


# ---- uploads -------------------------------------------------------------------------------------


async def test_uploads_stats_per_kind_and_period(db: Database) -> None:
    await _user(db, 1)
    await _user(db, 2)
    repo = UploadRepo(db)
    await repo.add(1, UploadKind.PHOTO, None, 100)
    await repo.add(1, UploadKind.PHOTO, None, 50)
    await repo.add(1, UploadKind.PDF, "doc.pdf", 1_000)
    await repo.add(2, UploadKind.VOICE, None, 7)

    mine = await repo.stats(1, None)
    assert mine[UploadKind.PHOTO].count == 2
    assert mine[UploadKind.PHOTO].bytes == 150
    assert mine[UploadKind.PDF].bytes == 1_000
    assert UploadKind.VOICE not in mine

    everyone = await repo.stats(None, None)
    assert everyone[UploadKind.VOICE].count == 1
    assert everyone[UploadKind.PHOTO].count == 2


async def test_uploads_stats_empty(db: Database) -> None:
    assert await UploadRepo(db).stats(None, None) == {}


# ---- status and notices --------------------------------------------------------------------------


async def test_status_beat_then_read(db: Database) -> None:
    repo = StatusRepo(db)
    assert await repo.get() is None  # never reported yet

    started = datetime.now(UTC) - timedelta(minutes=5)
    await repo.beat(started, queue_depth=3, active_jobs=2)
    status = await repo.get()
    assert status is not None
    assert status.queue_depth == 3
    assert status.active_jobs == 2
    assert status.started_at == pytest.approx(started, abs=timedelta(seconds=1))

    await repo.beat(started, queue_depth=0, active_jobs=0)  # single row, rewritten
    again = await repo.get()
    assert again is not None and again.queue_depth == 0


async def test_notice_claim_is_first_time_only(db: Database) -> None:
    """Every alert must fire once: this is the whole de-duplication mechanism."""
    repo = NoticeRepo(db)
    assert await repo.claim("alert:failed:2026-01-15") is True
    assert await repo.claim("alert:failed:2026-01-15") is False
    assert await repo.claim("alert:failed:2026-01-16") is True


# ---- conversations for the admin bot -------------------------------------------------------------


async def test_dialogs_survive_a_reset_and_are_listed_for_the_admin(db: Database) -> None:
    await _user(db, 1)
    convo = ConversationRepo(db)
    conv_id = await convo.active_id(1)
    await convo.add_message(conv_id, "user", "hi", 1)
    await convo.add_message(conv_id, "assistant", "hello", 1)

    await convo.reset(1)  # /new: the dialog ends but stays readable for admins

    assert await convo.count_for_user(1) == 1
    summaries = await convo.list_for_user(1, limit=10)
    assert len(summaries) == 1
    assert summaries[0].id == conv_id
    assert summaries[0].user_id == 1
    assert summaries[0].messages == 2
    assert summaries[0].active is False


async def test_messages_page_and_summary(db: Database) -> None:
    await _user(db, 1)
    convo = ConversationRepo(db)
    conv_id = await convo.active_id(1)
    for i in range(10):
        await convo.add_message(conv_id, "user", f"m{i}", 1)

    summary = await convo.summary(conv_id)
    assert summary is not None and summary.messages == 10
    assert await convo.summary(999999) is None

    page = await convo.messages_page(conv_id, limit=3, offset=0)
    assert [m.content for m in page] == ["m0", "m1", "m2"]
    assert page[0].role == "user"

    beyond = await convo.messages_page(conv_id, limit=3, offset=100)
    assert beyond == []


async def test_dialogs_count_for_unknown_user(db: Database) -> None:
    assert await ConversationRepo(db).count_for_user(999) == 0
    assert await ConversationRepo(db).list_for_user(999, limit=10) == []


async def test_usage_repo_is_per_day(db: Database) -> None:
    await _user(db, 1)
    usage = UsageRepo(db)
    await usage.increment(1, 10)
    await usage.increment(1, 10)
    assert await usage.requests_today(1) == 2

    # A row from another day must not leak into today's count.
    await db.conn.execute(
        "INSERT INTO daily_usage (user_id, day, requests, approx_tokens) VALUES (1, '2000-01-01', 99, 0)"
    )
    await db.conn.commit()
    assert await usage.requests_today(1) == 2


# ---- migrations ----------------------------------------------------------------------------------


async def test_migrations_set_the_schema_version(db: Database) -> None:
    from bot.storage.db import MIGRATIONS

    cursor = await db.conn.execute("PRAGMA user_version")
    (version,) = await cursor.fetchone()
    assert version == len(MIGRATIONS)


async def test_migration_race_between_two_processes(tmp_path: Path) -> None:
    """The main bot and the admin bot may start together on one file.

    The loser of the race hits "duplicate column" and must swallow it: the schema is correct
    either way, so a failed start would be a false alarm.
    """
    path = tmp_path / "shared.sqlite3"

    first, second = Database(path), Database(path)
    await first.connect()
    await asyncio.gather(second.connect())  # re-applying every migration on a live file
    try:
        await _user(first, 1)
        user = await UserRepo(second).get(1)
        assert user is not None and user.username == "gleb"
    finally:
        await second.close()
        await first.close()


async def test_reconnect_after_close(tmp_path: Path) -> None:
    path = tmp_path / "bot.sqlite3"
    db1 = Database(path)
    await db1.connect()
    await _user(db1, 1, "persisted")
    await db1.close()

    db2 = Database(path)
    await db2.connect()
    try:
        user = await UserRepo(db2).get(1)
        assert user is not None and user.username == "persisted"
    finally:
        await db2.close()


async def test_snapshot_is_a_usable_copy(db: Database, tmp_path: Path) -> None:
    """The backup button relies on this: the copy must hold the data and build a valid DB."""
    await _user(db, 1, "backed-up")
    await _request(db, user_id=1, tokens=42)
    target = tmp_path / "backup.sqlite3"

    await db.snapshot(target)

    assert target.exists() and target.stat().st_size > 0
    copy = Database(target)
    await copy.connect()
    try:
        user = await UserRepo(copy).get(1)
        assert user is not None and user.username == "backed-up"
        assert (await RequestRepo(copy).report(1, None)).total_tokens == 42
    finally:
        await copy.close()


async def test_row_to_user_reads_every_column(db: Database) -> None:
    """A column added by a migration without the mapper would silently show as a default."""
    repo = UserRepo(db)
    await _user(db, 1)
    await repo.set_context_limit(1, 12_345)
    await repo.set_daily_limit(1, 6)
    await repo.set_persona(1, "вежливый")
    await repo.set_skills(1, ("web", "images"))

    user = await repo.get(1)
    assert user is not None
    assert isinstance(user, User)
    assert user.context_limit == 12_345
    assert user.daily_limit == 6
    assert user.persona == "вежливый"
    assert user.skills == ("web", "images")
