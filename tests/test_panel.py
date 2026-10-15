from pathlib import Path

import aiosqlite

from bot.domain import Mode, Role
from bot.storage.db import MIGRATIONS, Database
from bot.storage.repositories import UserRepo
from bot.telegram import panel
from bot.telegram.keyboards import REPLY_KEYBOARD


class FakeBot:
    def __init__(self, refuse_below: int = 0) -> None:
        self.refuse_below = refuse_below  # ids below this are "older than 48h"
        self.batches: list[list[int]] = []
        self.singles: list[int] = []

    async def delete_messages(self, chat_id: int, message_ids: list[int]) -> bool:
        if min(message_ids) < self.refuse_below:
            raise RuntimeError("Bad Request: message can't be deleted")
        self.batches.append(message_ids)
        return True

    async def delete_message(self, chat_id: int, message_id: int) -> bool:
        if message_id < self.refuse_below:
            raise RuntimeError("Bad Request: message can't be deleted")
        self.singles.append(message_id)
        return True


async def test_delete_range_covers_every_id_newest_first_in_batches() -> None:
    bot = FakeBot()
    await panel.delete_range(bot, 1, 10, 260)
    assert [len(b) for b in bot.batches] == [100, 100, 51]
    flat = [i for batch in bot.batches for i in batch]
    assert flat == list(range(260, 9, -1))


async def test_refused_batch_falls_back_to_single_deletes() -> None:
    bot = FakeBot(refuse_below=150)
    await panel.delete_range(bot, 1, 101, 300)
    assert bot.batches == [list(range(300, 200, -1))]
    assert bot.singles == list(range(200, 149, -1))


async def test_delete_range_is_capped_and_forgets_the_panel() -> None:
    panel._panels[(7, 0)] = 42
    bot = FakeBot()
    await panel.delete_range(bot, 7, 1, 100_000)
    flat = [i for batch in bot.batches for i in batch]
    assert len(flat) == panel.MAX_CLEAR_IDS and flat[0] == 100_000
    assert (7, 0) not in panel._panels


def test_reply_keyboard_is_persistent() -> None:
    assert REPLY_KEYBOARD.is_persistent


async def test_v1_database_migrates_and_keeps_users(tmp_path: Path) -> None:
    path = tmp_path / "bot.db"
    async with aiosqlite.connect(path) as conn:  # a database created before migration 2 existed
        await conn.executescript(MIGRATIONS[0])
        await conn.execute(
            "INSERT INTO users VALUES (5, 'old', 'user', 'normal', NULL, '2026-01-01T00:00:00+00:00')"
        )
        await conn.execute("PRAGMA user_version = 1")
        await conn.commit()

    db = Database(path)
    await db.connect()
    users = UserRepo(db)
    assert (await users.get(5)).username == "old"
    assert await users.get_chat_floor(5) is None
    await users.set_chat_floor(5, 123)
    assert await users.get_chat_floor(5) == 123
    new = await users.upsert_seen(6, None, default_role=Role.USER, default_mode=Mode.NORMAL)
    assert new.tg_id == 6
    await db.close()
