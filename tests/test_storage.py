from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from bot.domain import Mode, RequestRecord, RequestStatus, Role
from bot.storage.db import MIGRATIONS, Database
from bot.storage.repositories import ConversationRepo, RequestRepo, UsageRepo, UserRepo


@pytest.fixture
async def db() -> Database:
    database = Database(":memory:")
    await database.connect()
    try:
        yield database
    finally:
        await database.close()


async def _make_user(db: Database, tg_id: int = 1, username: str = "gleb") -> None:
    await UserRepo(db).upsert_seen(tg_id, username, default_role=Role.PENDING, default_mode=Mode.NORMAL)


def test_conn_raises_before_connect() -> None:
    database = Database(":memory:")
    with pytest.raises(RuntimeError):
        _ = database.conn


async def test_migrations_idempotent_on_file_db(tmp_path: Path) -> None:
    path = tmp_path / "sub" / "bot.sqlite3"
    db1 = Database(path)
    await db1.connect()
    await _make_user(db1, 1)
    await db1.close()

    db2 = Database(path)
    await db2.connect()  # must not fail re-applying migrations
    user = await UserRepo(db2).get(1)
    assert user is not None
    assert user.username == "gleb"
    await db2.close()


async def test_upsert_seen_insert_and_refresh_username(db: Database) -> None:
    repo = UserRepo(db)
    user = await repo.upsert_seen(1, "old_name", default_role=Role.PENDING, default_mode=Mode.NORMAL)
    assert user.tg_id == 1
    assert user.username == "old_name"
    assert user.role is Role.PENDING
    assert user.mode is Mode.NORMAL
    assert user.model_override is None

    await repo.set_role(1, Role.USER)
    await repo.set_mode(1, Mode.DEEP)

    refreshed = await repo.upsert_seen(1, "new_name", default_role=Role.PENDING, default_mode=Mode.NORMAL)
    assert refreshed.username == "new_name"
    assert refreshed.role is Role.USER  # untouched by re-seeing
    assert refreshed.mode is Mode.DEEP  # untouched by re-seeing


async def test_set_role_set_mode_set_model_override(db: Database) -> None:
    repo = UserRepo(db)
    await _make_user(db, 1)

    assert await repo.set_role(1, Role.ADMIN) is True
    assert await repo.set_role(999, Role.ADMIN) is False

    await repo.set_mode(1, Mode.FAST)
    await repo.set_model_override(1, "gpt-x")

    user = await repo.get(1)
    assert user is not None
    assert user.role is Role.ADMIN
    assert user.mode is Mode.FAST
    assert user.model_override == "gpt-x"

    await repo.set_model_override(1, None)
    user = await repo.get(1)
    assert user is not None
    assert user.model_override is None


async def test_get_missing_user_returns_none(db: Database) -> None:
    assert await UserRepo(db).get(42) is None


async def test_list_by_role_ordered_by_created_at(db: Database) -> None:
    repo = UserRepo(db)
    await repo.upsert_seen(1, "a", default_role=Role.USER, default_mode=Mode.NORMAL)
    await repo.upsert_seen(2, "b", default_role=Role.USER, default_mode=Mode.NORMAL)
    await repo.upsert_seen(3, "c", default_role=Role.PENDING, default_mode=Mode.NORMAL)

    users = await repo.list_by_role(Role.USER)
    assert [u.tg_id for u in users] == [1, 2]

    pending = await repo.list_by_role(Role.PENDING)
    assert [u.tg_id for u in pending] == [3]


async def test_active_id_stable_then_new_after_reset(db: Database) -> None:
    await _make_user(db, 1)
    convo = ConversationRepo(db)

    first = await convo.active_id(1)
    again = await convo.active_id(1)
    assert first == again

    await convo.add_message(first, "user", "hi", 1)
    await convo.reset(1)

    second = await convo.active_id(1)
    assert second != first
    assert await convo.recent_messages(second, 10) == []
    # The ended dialog stays in the log for the admin bot.
    assert [m.content for m in await convo.recent_messages(first, 10)] == ["hi"]


async def test_reset_without_active_conversation_is_noop(db: Database) -> None:
    await _make_user(db, 1)
    convo = ConversationRepo(db)
    await convo.reset(1)  # nothing active yet, should not raise


async def test_recent_messages_last_n_oldest_first(db: Database) -> None:
    await _make_user(db, 1)
    convo = ConversationRepo(db)
    conv_id = await convo.active_id(1)

    for i in range(5):
        await convo.add_message(conv_id, "user", f"msg{i}", i)

    recent = await convo.recent_messages(conv_id, 3)
    assert [m.content for m in recent] == ["msg2", "msg3", "msg4"]
    assert recent[0].role == "user"
    assert recent[0].approx_tokens == 2


async def test_purge_older_than(db: Database) -> None:
    await _make_user(db, 1)
    convo = ConversationRepo(db)
    conv_id = await convo.active_id(1)

    old_ts = (datetime.now(UTC) - timedelta(days=40)).isoformat()
    await db.conn.execute(
        "INSERT INTO messages (conversation_id, role, content, approx_tokens, created_at) "
        "VALUES (?, 'user', 'old', 1, ?)",
        (conv_id, old_ts),
    )
    await db.conn.commit()
    await convo.add_message(conv_id, "user", "new", 1)

    deleted = await convo.purge_older_than(30)
    assert deleted == 1

    remaining = await convo.recent_messages(conv_id, 10)
    assert [m.content for m in remaining] == ["new"]


async def test_usage_repo_increment_and_requests_today(db: Database) -> None:
    await _make_user(db, 1)
    usage = UsageRepo(db)

    assert await usage.requests_today(1) == 0

    await usage.increment(1, 10)
    await usage.increment(1, 5)

    assert await usage.requests_today(1) == 2


async def test_request_repo_add_and_stats_today(db: Database) -> None:
    await _make_user(db, 1, "a")
    await _make_user(db, 2, "b")
    await UserRepo(db).set_role(1, Role.ADMIN)

    requests = RequestRepo(db)
    await requests.add(
        RequestRecord(
            user_id=1,
            conversation_id=None,
            model="m1",
            status=RequestStatus.OK,
            attempts=1,
            latency_ms=100,
            approx_tokens=20,
        )
    )
    await requests.add(
        RequestRecord(
            user_id=2,
            conversation_id=None,
            model="m1",
            status=RequestStatus.FAILED,
            attempts=3,
            latency_ms=200,
            approx_tokens=5,
            error="boom",
        )
    )

    stats = await requests.stats_today()
    assert stats.users_total == 2
    assert stats.users_pending == 1  # user 2 stayed pending
    assert stats.requests_today == 2
    assert stats.tokens_today == 25
    assert stats.failed_today == 1
    assert stats.avg_latency_ms_today == 150


async def test_migration_caps_personal_daily_limits(db: Database) -> None:
    # The owner raised the global limit to 100; personal limits above it are capped (idempotently).
    await _make_user(db, 1, "high")  # daily_limit 250 -> 100
    await _make_user(db, 2, "low")  # below the cap, stays
    await _make_user(db, 3, "null")  # NULL = the global limit, must stay NULL
    await db.conn.execute("UPDATE users SET daily_limit = 250 WHERE tg_id = 1")
    await db.conn.execute("UPDATE users SET daily_limit = 40 WHERE tg_id = 2")
    await db.conn.commit()

    async def limits() -> list[int | None]:
        cursor = await db.conn.execute("SELECT tg_id, daily_limit FROM users ORDER BY tg_id")
        return [row[1] for row in await cursor.fetchall()]

    # On the live DB the version was 7 with the limits already in place, and the cap arrived with
    # migration 8: apply that one script, like the restart after its deploy did.
    assert await limits() == [250, 40, None]
    await db.conn.executescript(f"BEGIN;\n{MIGRATIONS[7]}\nCOMMIT;")
    assert await limits() == [100, 40, None]

    # Re-applying every script by hand must be a no-op for the cap.
    for script in MIGRATIONS:
        try:
            await db.conn.executescript(f"BEGIN;\n{script}\nCOMMIT;")
        except sqlite3.OperationalError:
            await db.conn.rollback()
    assert await limits() == [100, 40, None]


async def test_stats_today_with_no_requests(db: Database) -> None:
    stats = await RequestRepo(db).stats_today()
    assert stats.users_total == 0
    assert stats.users_pending == 0
    assert stats.requests_today == 0
    assert stats.tokens_today == 0
    assert stats.failed_today == 0
    assert stats.avg_latency_ms_today == 0
