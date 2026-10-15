"""CONTEXT_IDLE_HOURS: the dialog history is quietly dropped after hours of silence."""

from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from bot.catalog import load_catalog
from bot.config import DEFAULT_MODELS_FILE
from bot.domain import Mode, RequestStatus, Role
from bot.llm.base import ChatMessage, ChatResult, ChatStatus
from bot.services.chat import ChatService
from bot.services.conversation import ConversationService
from bot.services.model_router import ModelRouter
from bot.services.quota import QuotaService
from bot.storage.db import Database
from bot.storage.repositories import ConversationRepo, RequestRepo, UsageRepo, UserRepo

STALE = (datetime.now(UTC) - timedelta(hours=13)).isoformat()


class FakeLLM:
    def __init__(self, *results: ChatResult) -> None:
        self.results = list(results)
        self.calls: list[tuple[list[ChatMessage], str]] = []

    async def chat(self, messages: Sequence[ChatMessage], model_id: str, tools=()) -> ChatResult:
        self.calls.append((list(messages), model_id))
        return self.results.pop(0)

    def queue_depth(self) -> int:
        return 0


def ok(text: str) -> ChatResult:
    return ChatResult(ChatStatus.OK, text, "claude-opus-4.8", 1, 10, False)


@pytest.fixture
async def env():
    db = Database(":memory:")
    await db.connect()
    users = UserRepo(db)
    user = await users.upsert_seen(1, "u", default_role=Role.USER, default_mode=Mode.NORMAL)
    yield {
        "db": db,
        "repo": ConversationRepo(db),
        "user": user,
        "catalog": load_catalog(DEFAULT_MODELS_FILE),
    }
    await db.close()


def make_service(env, llm: FakeLLM, idle_hours: float) -> ChatService:
    return ChatService(
        llm,
        ConversationService(env["repo"], system_prompt="sys", token_budget=10_000, idle_hours=idle_hours),
        QuotaService(UsageRepo(env["db"]), daily_limit=10),
        ModelRouter(env["catalog"]),
        RequestRepo(env["db"]),
        env["catalog"],
    )


async def age_everything(env, conversation_id: int) -> None:
    """Push the dialog 13 hours into the past: last message and started_at."""
    conn = env["db"].conn
    await conn.execute(
        "UPDATE messages SET created_at = ? WHERE conversation_id = ?", (STALE, conversation_id)
    )
    await conn.execute("UPDATE conversations SET started_at = ? WHERE id = ?", (STALE, conversation_id))
    await conn.commit()


async def active_row(env) -> tuple[int, int]:
    cursor = await env["db"].conn.execute(
        "SELECT id, active FROM conversations WHERE user_id = ? ORDER BY id DESC LIMIT 1",
        (env["user"].tg_id,),
    )
    row = await cursor.fetchone()
    return int(row["id"]), int(row["active"])


async def test_idle_over_12_hours_resets_history_quietly(env) -> None:
    llm = FakeLLM(ok("a1"), ok("a2"))
    service = make_service(env, llm, idle_hours=12)
    await service.answer(env["user"], "q1")
    old_id, active = await active_row(env)
    assert active == 1

    await age_everything(env, old_id)
    reply = await service.answer(env["user"], "q2")

    assert reply.status is RequestStatus.OK
    assert not reply.context_reset  # the reset is silent: no header note for the user
    assert [(m.role, m.content) for m in llm.calls[1][0]] == [("system", "sys"), ("user", "q2")]
    # The old dialog is closed in the DB, a new empty one took its place.
    cursor = await env["db"].conn.execute("SELECT active FROM conversations WHERE id = ?", (old_id,))
    assert int((await cursor.fetchone())["active"]) == 0
    new_id, active = await active_row(env)
    assert new_id != old_id and active == 1


async def test_fresh_history_is_kept(env) -> None:
    llm = FakeLLM(ok("a1"), ok("a2"))
    service = make_service(env, llm, idle_hours=12)
    await service.answer(env["user"], "q1")
    await service.answer(env["user"], "q2")
    assert [(m.role, m.content) for m in llm.calls[1][0]] == [
        ("system", "sys"),
        ("user", "q1"),
        ("assistant", "a1"),
        ("user", "q2"),
    ]
    _, active = await active_row(env)
    assert active == 1


async def test_zero_idle_hours_disables_the_reset(env) -> None:
    llm = FakeLLM(ok("a1"), ok("a2"))
    service = make_service(env, llm, idle_hours=0)
    await service.answer(env["user"], "q1")
    conversation_id, _ = await active_row(env)

    await age_everything(env, conversation_id)
    await service.answer(env["user"], "q2")

    assert [(m.role, m.content) for m in llm.calls[1][0]] == [
        ("system", "sys"),
        ("user", "q1"),
        ("assistant", "a1"),
        ("user", "q2"),
    ]
    same_id, active = await active_row(env)
    assert same_id == conversation_id and active == 1


async def test_exchange_after_reset_lands_in_the_new_dialog(env) -> None:
    llm = FakeLLM(ok("a1"), ok("a2"), ok("a3"))
    service = make_service(env, llm, idle_hours=12)
    await service.answer(env["user"], "q1")
    old_id, _ = await active_row(env)

    await age_everything(env, old_id)
    await service.answer(env["user"], "q2")
    await service.answer(env["user"], "q3")
    new_id, _ = await active_row(env)

    cursor = await env["db"].conn.execute(
        "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY id", (new_id,)
    )
    rows = [(r["role"], r["content"]) for r in await cursor.fetchall()]
    assert rows == [("user", "q2"), ("assistant", "a2"), ("user", "q3"), ("assistant", "a3")]
    # The new dialog is fresh: the third turn sees only the post-reset exchange.
    assert [(m.role, m.content) for m in llm.calls[2][0]] == [
        ("system", "sys"),
        ("user", "q2"),
        ("assistant", "a2"),
        ("user", "q3"),
    ]


async def test_empty_dialog_ages_from_started_at(env) -> None:
    conversations = ConversationService(env["repo"], system_prompt="sys", token_budget=10_000, idle_hours=12)
    conversation_id = await env["repo"].active_id(env["user"].tg_id)
    conn = env["db"].conn
    await conn.execute("UPDATE conversations SET started_at = ? WHERE id = ?", (STALE, conversation_id))
    await conn.commit()

    built_id, messages = await conversations.build_context(env["user"].tg_id, "привет")

    assert built_id != conversation_id
    assert [(m.role, m.content) for m in messages] == [("system", "sys"), ("user", "привет")]


async def test_personal_idle_hours_reset(env) -> None:
    """A personal value below the global one resets the history after that, not 12 hours."""
    llm = FakeLLM(ok("a1"), ok("a2"))
    service = make_service(env, llm, idle_hours=12)
    await service.answer(env["user"], "q1")
    old_id, _ = await active_row(env)
    # 5 hours old: below the global 12, above the personal 2.
    fresh = (datetime.now(UTC) - timedelta(hours=5)).isoformat()
    conn = env["db"].conn
    await conn.execute("UPDATE messages SET created_at = ? WHERE conversation_id = ?", (fresh, old_id))
    await conn.execute("UPDATE conversations SET started_at = ? WHERE id = ?", (fresh, old_id))
    await conn.commit()

    personal = replace(env["user"], context_idle_hours=2)
    reply = await service.answer(personal, "q2")

    assert reply.status is RequestStatus.OK
    assert not reply.context_reset
    assert [(m.role, m.content) for m in llm.calls[1][0]] == [("system", "sys"), ("user", "q2")]
    new_id, active = await active_row(env)
    assert new_id != old_id and active == 1


async def test_personal_none_keeps_global(env) -> None:
    """None means the global CONTEXT_IDLE_HOURS: 5 hours old is fresh with a 12-hour global."""
    llm = FakeLLM(ok("a1"), ok("a2"))
    service = make_service(env, llm, idle_hours=12)
    await service.answer(env["user"], "q1")
    old_id, _ = await active_row(env)
    fresh = (datetime.now(UTC) - timedelta(hours=5)).isoformat()
    conn = env["db"].conn
    await conn.execute("UPDATE messages SET created_at = ? WHERE conversation_id = ?", (fresh, old_id))
    await conn.execute("UPDATE conversations SET started_at = ? WHERE id = ?", (fresh, old_id))
    await conn.commit()

    default_user = replace(env["user"], context_idle_hours=None)  # as loaded from the DB: NULL
    await service.answer(default_user, "q2")

    assert [(m.role, m.content) for m in llm.calls[1][0]] == [
        ("system", "sys"),
        ("user", "q1"),
        ("assistant", "a1"),
        ("user", "q2"),
    ]
    same_id, active = await active_row(env)
    assert same_id == old_id and active == 1


async def test_personal_zero_disables_the_reset_for_that_user(env) -> None:
    """A personal 0 turns the reset off for this user while the global stays 12 hours."""
    llm = FakeLLM(ok("a1"), ok("a2"))
    service = make_service(env, llm, idle_hours=12)
    await service.answer(env["user"], "q1")
    conversation_id, _ = await active_row(env)

    await age_everything(env, conversation_id)  # 13 hours: over the global timeout
    never_reset = replace(env["user"], context_idle_hours=0)
    await service.answer(never_reset, "q2")

    assert [(m.role, m.content) for m in llm.calls[1][0]] == [
        ("system", "sys"),
        ("user", "q1"),
        ("assistant", "a1"),
        ("user", "q2"),
    ]
    same_id, active = await active_row(env)
    assert same_id == conversation_id and active == 1


async def test_personal_value_larger_than_global_keeps_history(env) -> None:
    """A personal value above the global one keeps the history past the global timeout."""
    llm = FakeLLM(ok("a1"), ok("a2"))
    service = make_service(env, llm, idle_hours=12)
    await service.answer(env["user"], "q1")
    conversation_id, _ = await active_row(env)

    await age_everything(env, conversation_id)  # 13 hours: over 12, below 48
    patient = replace(env["user"], context_idle_hours=48)
    await service.answer(patient, "q2")

    assert [(m.role, m.content) for m in llm.calls[1][0]] == [
        ("system", "sys"),
        ("user", "q1"),
        ("assistant", "a1"),
        ("user", "q2"),
    ]
    same_id, active = await active_row(env)
    assert same_id == conversation_id and active == 1
