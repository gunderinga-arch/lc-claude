"""The main bot's heartbeat: the row the admin bot reads to tell alive from hung."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from bot.domain import BOT_STALE_AFTER_S, HEARTBEAT_S
from bot.services.heartbeat import Heartbeat
from bot.storage.db import Database
from bot.storage.repositories import StatusRepo


@pytest.fixture
async def db() -> Database:
    database = Database(":memory:")
    await database.connect()
    try:
        yield database
    finally:
        await database.close()


def _hb(db: Database, queue: int = 0, jobs: int = 0, now: datetime | None = None) -> Heartbeat:
    moment = now or datetime.now(UTC)
    return Heartbeat(
        StatusRepo(db),
        queue_depth=lambda: queue,
        active_jobs=lambda: jobs,
        now=lambda: moment,
    )


async def test_beat_once_writes_the_row(db: Database) -> None:
    status = StatusRepo(db)
    assert await status.get() is None

    await _hb(db, queue=3, jobs=2).beat_once()

    row = await status.get()
    assert row is not None
    assert (row.queue_depth, row.active_jobs) == (3, 2)


async def test_counters_are_read_live_not_captured(db: Database) -> None:
    """The values must come from the bot's current state, not from construction time."""
    status = StatusRepo(db)
    state = {"queue": 0, "jobs": 0}
    hb = Heartbeat(
        status,
        queue_depth=lambda: state["queue"],
        active_jobs=lambda: state["jobs"],
    )

    await hb.beat_once()
    state.update(queue=5, jobs=1)
    await hb.beat_once()

    row = await status.get()
    assert row is not None
    assert (row.queue_depth, row.active_jobs) == (5, 1)


async def test_started_at_stays_put_across_beats(db: Database) -> None:
    """Uptime would be meaningless if every heartbeat moved the start."""
    status = StatusRepo(db)
    start = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)
    hb = _hb(db, now=start)

    await hb.beat_once()
    first = await status.get()
    await hb.beat_once()
    second = await status.get()

    assert first is not None and second is not None
    assert first.started_at == second.started_at == start


async def test_fresh_heartbeat_is_not_stale(db: Database) -> None:
    """The admin bot's `bot_alive` must read a just-written row as alive."""
    from adminbot.common import bot_alive

    now = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)
    await _hb(db, now=now).beat_once()

    status = await StatusRepo(db).get()
    assert status is not None
    assert bot_alive(status, now) is True
    # The repository stamps heartbeat_at with the real clock, so compare against real time.
    assert bot_alive(await StatusRepo(db).get(), datetime.now(UTC)) is True


async def test_never_reported_is_unknown_not_dead() -> None:
    """None is a distinct third state: the admin bot shows "no data" rather than an alarm."""
    from adminbot.common import bot_alive

    assert bot_alive(None, datetime.now(UTC)) is None


async def test_stale_row_is_detected(db: Database) -> None:
    from adminbot.common import bot_alive

    status = StatusRepo(db)
    await status.beat(datetime(2026, 1, 15, 11, 0, tzinfo=UTC), queue_depth=0, active_jobs=0)
    row = await status.get()
    assert row is not None

    fresh = row.heartbeat_at + timedelta(seconds=BOT_STALE_AFTER_S - 1)
    assert bot_alive(row, fresh) is True

    stale = row.heartbeat_at + timedelta(seconds=BOT_STALE_AFTER_S + 1)
    assert bot_alive(row, stale) is False


async def test_write_failure_is_swallowed(db: Database) -> None:
    """A database hiccup must never take the bot down with it."""

    class Exploding(StatusRepo):
        async def beat(self, *args, **kwargs):  # type: ignore[override]
            raise RuntimeError("database is locked")

    hb = Heartbeat(Exploding(db), queue_depth=lambda: 0, active_jobs=lambda: 0)
    await hb.beat_once()  # must not raise


async def test_start_beat_and_stop(db: Database) -> None:
    """start() writes immediately and keeps writing; stop() cancels cleanly and is idempotent."""
    status = StatusRepo(db)
    hb = Heartbeat(
        status,
        queue_depth=lambda: 0,
        active_jobs=lambda: 0,
        interval_s=0.01,
    )

    await hb.start()
    assert (await status.get()) is not None  # the first beat is synchronous

    await hb.stop()
    await hb.stop()  # calling twice must be safe


async def test_heartbeat_interval_is_sane() -> None:
    """Stale detection must be slower than the write interval, or the admin bot sees flapping."""
    assert BOT_STALE_AFTER_S > HEARTBEAT_S
