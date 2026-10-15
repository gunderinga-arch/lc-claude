from datetime import UTC, datetime, timedelta

from bot.domain import Mode, RequestRecord, RequestStatus, Role, UserUsage
from bot.render.format import ago, short_number
from bot.storage.db import Database
from bot.storage.repositories import RequestRepo, UserRepo
from bot.telegram.keyboards import leaderboard_keyboard
from bot.telegram.routers.admin import _leaderboard_text


async def _db_with_usage() -> tuple[Database, RequestRepo]:
    db = Database(":memory:")
    await db.connect()
    users, requests = UserRepo(db), RequestRepo(db)
    for tg_id in (1, 2, 3):
        await users.upsert_seen(tg_id, f"u{tg_id}", default_role=Role.USER, default_mode=Mode.NORMAL)
    # user 1: many small requests, user 2: one huge request (added last), user 3: never asked
    for _ in range(3):
        await requests.add(RequestRecord(1, None, "m", RequestStatus.OK, 1, 10, 100))
    await requests.add(RequestRecord(2, None, "m", RequestStatus.FAILED, 3, 10, 50_000))
    return db, requests


async def test_leaderboard_orders_by_each_key() -> None:
    db, requests = await _db_with_usage()
    by_requests = await requests.leaderboard("requests")
    assert [(r.tg_id, r.requests, r.approx_tokens) for r in by_requests] == [
        (1, 3, 300),
        (2, 1, 50_000),
        (3, 0, 0),
    ]
    assert [r.tg_id for r in await requests.leaderboard("tokens")] == [2, 1, 3]
    by_last = await requests.leaderboard("last")
    assert [r.tg_id for r in by_last] == [2, 1, 3]  # never-active user goes last
    assert by_last[-1].last_at is None and by_last[0].last_at is not None
    assert len(await requests.leaderboard("requests", limit=2)) == 2
    await db.close()


def test_numbers_and_relative_time() -> None:
    now = datetime(2026, 9, 23, 12, 0, tzinfo=UTC)
    assert [short_number(n) for n in (999, 1_500, 2_345_678)] == ["999", "1.5k", "2.3M"]
    assert ago(None, now) == "не пользовался"
    assert ago(now - timedelta(seconds=10), now) == "только что"
    assert ago(now - timedelta(minutes=5), now) == "5 мин назад"
    assert ago(now - timedelta(hours=3, minutes=59), now) == "3 ч назад"
    assert ago(now - timedelta(days=2, hours=5), now) == "2 дн назад"


def test_leaderboard_text_escapes_and_shows_counters_only() -> None:
    now = datetime.now(UTC)
    rows = [UserUsage(1, "a<b>", Role.ADMIN, 12, 1_200, now - timedelta(minutes=3))]
    text = _leaderboard_text(rows, "tokens", now)
    assert "@a&lt;b&gt; 👑" in text
    assert "1. " in text and "12 запр." in text and "1.2k ток." in text and "3 мин назад" in text
    assert "Пользователей пока нет" in _leaderboard_text([], "requests", now)


def test_leaderboard_keyboard_marks_current_order() -> None:
    buttons = [b for row in leaderboard_keyboard("tokens").inline_keyboard for b in row]
    assert [b.callback_data for b in buttons] == [
        "admin:top:requests",
        "admin:top:tokens",
        "admin:top:last",
        "admin:menu",
    ]
    assert [b.text.startswith("✅") for b in buttons[:3]] == [False, True, False]
