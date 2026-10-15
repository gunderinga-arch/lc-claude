from __future__ import annotations

from datetime import UTC, datetime

import pytest

from bot.catalog import Catalog, load_catalog
from bot.config import DEFAULT_MODELS_FILE
from bot.domain import Mode, Role, User
from bot.llm.base import ChatMessage
from bot.services.conversation import ConversationService
from bot.services.model_router import ModelRouter
from bot.services.quota import QuotaService
from bot.storage.db import Database
from bot.storage.repositories import ConversationRepo, UsageRepo, UserRepo


@pytest.fixture
async def db() -> Database:
    database = Database(":memory:")
    await database.connect()
    try:
        yield database
    finally:
        await database.close()


async def _make_user(
    db: Database,
    tg_id: int = 1,
    *,
    role: Role = Role.USER,
    mode: Mode = Mode.NORMAL,
) -> User:
    await UserRepo(db).upsert_seen(tg_id, "gleb", default_role=role, default_mode=mode)
    if role is not Role.PENDING:
        await UserRepo(db).set_role(tg_id, role)
    user = await UserRepo(db).get(tg_id)
    assert user is not None
    return user


def _user(*, mode: Mode, model_override: str | None) -> User:
    return User(
        tg_id=1,
        username=None,
        role=Role.USER,
        mode=mode,
        model_override=model_override,
        created_at=datetime.now(UTC),
    )


# ---- ConversationService ----------------------------------------------------------------------


async def test_build_context_empty_history(db: Database) -> None:
    await _make_user(db, 1)
    service = ConversationService(ConversationRepo(db), system_prompt="sys", token_budget=1000)
    conv_id, messages = await service.build_context(1, "hello")
    assert conv_id > 0
    assert messages == [
        ChatMessage(role="system", content="sys"),
        ChatMessage(role="user", content="hello"),
    ]


async def test_build_context_trims_to_budget_keeps_newest_chronological(db: Database) -> None:
    await _make_user(db, 1)
    repo = ConversationRepo(db)
    conv_id = await repo.active_id(1)
    full_order: list[str] = []
    for i in range(5):
        await repo.add_message(conv_id, "user", f"u{i}", 1)
        await repo.add_message(conv_id, "assistant", f"a{i}", 1)
        full_order.extend([f"u{i}", f"a{i}"])

    # system("sys")=1 + user("new")=1 -> used=2; budget=5 fits 3 more history messages (a3, u4, a4),
    # but the leading "a3" is an assistant orphan and gets dropped, leaving [u4, a4].
    service = ConversationService(ConversationRepo(db), system_prompt="sys", token_budget=5)
    _, messages = await service.build_context(1, "new")

    assert messages[0] == ChatMessage(role="system", content="sys")
    assert messages[-1] == ChatMessage(role="user", content="new")
    kept = [m.content for m in messages[1:-1]]
    assert kept == ["u4", "a4"]
    assert kept == full_order[-3:][1:]  # suffix of chronological history, orphan dropped

    # budget=6 fits 4 messages (u3, a3, u4, a4); no leading orphan, so nothing is dropped.
    service2 = ConversationService(ConversationRepo(db), system_prompt="sys", token_budget=6)
    _, messages2 = await service2.build_context(1, "new")
    kept2 = [m.content for m in messages2[1:-1]]
    assert kept2 == full_order[-4:]


async def test_build_context_drops_leading_assistant_orphan(db: Database) -> None:
    await _make_user(db, 1)
    repo = ConversationRepo(db)
    conv_id = await repo.active_id(1)
    await repo.add_message(conv_id, "user", "u0", 1)
    await repo.add_message(conv_id, "assistant", "a0", 1)

    # used starts at approx_tokens("s")=1 + approx_tokens("new")=1 = 2; budget=3 allows one more token.
    # newest->oldest walk: a0 (used=3, fits) then u0 (used=4, doesn't fit) -> kept=[a0] -> orphan dropped.
    service = ConversationService(ConversationRepo(db), system_prompt="s", token_budget=3)
    _, messages = await service.build_context(1, "new")
    kept = messages[1:-1]
    assert kept == []


async def test_build_context_oversized_new_message_still_included(db: Database) -> None:
    await _make_user(db, 1)
    service = ConversationService(ConversationRepo(db), system_prompt="sys", token_budget=1)
    huge = "x" * 100
    conv_id, messages = await service.build_context(1, huge)
    assert conv_id > 0
    assert messages == [
        ChatMessage(role="system", content="sys"),
        ChatMessage(role="user", content=huge),
    ]


async def test_build_context_token_budget_overrides_global_and_none_falls_back(db: Database) -> None:
    await _make_user(db, 1)
    repo = ConversationRepo(db)
    conv_id = await repo.active_id(1)
    for i in range(5):
        await repo.add_message(conv_id, "user", f"u{i}", 10)
        await repo.add_message(conv_id, "assistant", f"a{i}", 10)

    service = ConversationService(ConversationRepo(db), system_prompt="sys", token_budget=1000)

    _, limited = await service.build_context(1, "new", token_budget=15)
    assert limited == [ChatMessage(role="system", content="sys"), ChatMessage(role="user", content="new")]

    _, global_budget = await service.build_context(1, "new", token_budget=None)
    assert len(global_budget) == 12  # system + all 10 history messages (fit the 1000 global budget) + user


async def test_save_exchange_then_build_context_contains_pair(db: Database) -> None:
    await _make_user(db, 1)
    service = ConversationService(ConversationRepo(db), system_prompt="sys", token_budget=1000)
    conv_id, _ = await service.build_context(1, "hi")
    await service.save_exchange(conv_id, "hi", "hello there")

    _, messages = await service.build_context(1, "again")
    contents = [(m.role, m.content) for m in messages]
    assert ("user", "hi") in contents
    assert ("assistant", "hello there") in contents


async def test_reset_clears_history(db: Database) -> None:
    await _make_user(db, 1)
    service = ConversationService(ConversationRepo(db), system_prompt="sys", token_budget=1000)
    conv_id, _ = await service.build_context(1, "hi")
    await service.save_exchange(conv_id, "hi", "hello")

    await service.reset(1)

    new_conv_id, messages = await service.build_context(1, "again")
    assert new_conv_id != conv_id
    assert messages == [
        ChatMessage(role="system", content="sys"),
        ChatMessage(role="user", content="again"),
    ]


# ---- personal context idle hours (repository round trip) ----------------------------------------


async def test_set_context_idle_hours_round_trip_and_unknown_user(db: Database) -> None:
    repo = UserRepo(db)
    await _make_user(db, 1)

    assert await repo.set_context_idle_hours(1, 6.0) is True
    user = await repo.get(1)
    assert user is not None and user.context_idle_hours == 6.0

    assert await repo.set_context_idle_hours(1, 0) is True  # personal "never reset"
    user = await repo.get(1)
    assert user is not None and user.context_idle_hours == 0.0

    assert await repo.set_context_idle_hours(1, None) is True  # back to the global value
    user = await repo.get(1)
    assert user is not None and user.context_idle_hours is None

    assert await repo.set_context_idle_hours(999, 12.0) is False


async def test_new_user_has_no_personal_context_idle_hours(db: Database) -> None:
    """A fresh user must read as None so the global CONTEXT_IDLE_HOURS applies."""
    user = await _make_user(db, 1)
    assert user.context_idle_hours is None


# ---- QuotaService -------------------------------------------------------------------------------


async def test_quota_admin_unlimited(db: Database) -> None:
    user = await _make_user(db, 1, role=Role.ADMIN)
    service = QuotaService(UsageRepo(db), daily_limit=2)
    decision = await service.check(user)
    assert decision.allowed is True
    assert decision.reason == "ok"
    assert decision.limit is None


async def test_quota_pending_denied(db: Database) -> None:
    user = await _make_user(db, 1, role=Role.PENDING)
    service = QuotaService(UsageRepo(db), daily_limit=2)
    decision = await service.check(user)
    assert decision.allowed is False
    assert decision.reason == "pending"


async def test_quota_blocked_denied(db: Database) -> None:
    user = await _make_user(db, 1, role=Role.BLOCKED)
    service = QuotaService(UsageRepo(db), daily_limit=2)
    decision = await service.check(user)
    assert decision.allowed is False
    assert decision.reason == "blocked"


async def test_quota_user_boundary_at_daily_limit(db: Database) -> None:
    user = await _make_user(db, 1, role=Role.USER)
    service = QuotaService(UsageRepo(db), daily_limit=2)

    decision = await service.check(user)
    assert decision.allowed is True
    assert decision.used == 0
    assert decision.limit == 2

    await service.record(1, 10)
    decision = await service.check(user)
    assert decision.allowed is True
    assert decision.used == 1

    await service.record(1, 10)
    decision = await service.check(user)
    assert decision.allowed is False
    assert decision.reason == "daily_limit"
    assert decision.used == 2


# ---- ModelRouter --------------------------------------------------------------------------------


def _catalog() -> Catalog:
    return load_catalog(DEFAULT_MODELS_FILE)


def test_model_router_available_override_wins() -> None:
    catalog = _catalog()
    router = ModelRouter(catalog)
    user = _user(mode=Mode.FAST, model_override="claude-opus-5")
    spec = router.resolve(user)
    assert spec.id == "claude-opus-5"


@pytest.mark.parametrize("override", ["claude-fable-5", "no-such-model", "gpt-5.6-luna"])
def test_model_router_disabled_unknown_or_unavailable_override_falls_back(override: str) -> None:
    catalog = _catalog()
    router = ModelRouter(catalog)
    user = _user(mode=Mode.DEEP, model_override=override)
    spec = router.resolve(user)
    assert spec.id == catalog.for_mode(Mode.DEEP).id


def test_model_router_marked_unavailable_override_falls_back() -> None:
    catalog = _catalog()
    catalog.mark_unavailable("claude-opus-5")
    router = ModelRouter(catalog)
    user = _user(mode=Mode.NORMAL, model_override="claude-opus-5")
    spec = router.resolve(user)
    assert spec.id == catalog.for_mode(Mode.NORMAL).id


@pytest.mark.parametrize("mode", list(Mode))
def test_model_router_no_override_uses_mode_model(mode: Mode) -> None:
    catalog = _catalog()
    router = ModelRouter(catalog)
    user = _user(mode=mode, model_override=None)
    spec = router.resolve(user)
    assert spec.id == catalog.for_mode(mode).id
