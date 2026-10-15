import pytest

from bot.domain import Mode, Role
from bot.services.roles import (
    BLOCKED_NOTICE,
    GRANTED_TEXT,
    REJECTED_TEXT,
    UNBLOCKED_TEXT,
    apply_role,
    notified_note,
)
from bot.storage.db import Database
from bot.storage.repositories import UserRepo


class FakeBot:
    def __init__(self, refuse: bool = False) -> None:
        self.refuse = refuse
        self.sent: list[tuple[int, str]] = []

    async def send_message(self, chat_id: int, text: str) -> None:
        if self.refuse:
            raise RuntimeError("Forbidden: bot can't initiate conversation with a user")
        self.sent.append((chat_id, text))


@pytest.fixture
async def users():
    db = Database(":memory:")
    await db.connect()
    repo = UserRepo(db)
    await repo.upsert_seen(7, "u", default_role=Role.PENDING, default_mode=Mode.NORMAL)
    yield repo
    await db.close()


@pytest.mark.parametrize(
    ("start", "new", "text"),
    [
        (Role.PENDING, Role.USER, GRANTED_TEXT),
        (Role.PENDING, Role.BLOCKED, REJECTED_TEXT),
        (Role.USER, Role.BLOCKED, BLOCKED_NOTICE),
        (Role.BLOCKED, Role.USER, UNBLOCKED_TEXT),
    ],
)
async def test_each_role_change_sends_its_notice(users, start, new, text) -> None:
    await users.set_role(7, start)
    bot = FakeBot()
    assert await apply_role(bot, users, 7, new) == (True, True)
    assert bot.sent == [(7, text)]
    assert (await users.get(7)).role is new


async def test_no_change_no_message_and_unknown_user(users) -> None:
    await users.set_role(7, Role.BLOCKED)
    bot = FakeBot()
    assert await apply_role(bot, users, 7, Role.BLOCKED) == (True, None)
    assert await apply_role(bot, users, 999, Role.USER) == (False, None)
    assert bot.sent == []


async def test_undeliverable_notice_is_reported(users) -> None:
    found, notified = await apply_role(FakeBot(refuse=True), users, 7, Role.USER)
    assert (found, notified) == (True, False)
    assert (await users.get(7)).role is Role.USER  # the role changes even if the message can't be sent
    assert "не удалось" in notified_note(notified)
    assert notified_note(None) == "" and "уведомлён" in notified_note(True)
