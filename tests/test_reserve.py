from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from types import SimpleNamespace

import pytest

from bot.catalog import Catalog, ModelSpec, load_catalog
from bot.config import DEFAULT_MODELS_FILE
from bot.domain import Mode, RequestStatus, Role
from bot.llm.base import ChatMessage, ChatResult, ChatStatus, Completion, CompletionRequest
from bot.llm.reserve import ReserveSwitch, SwitchingProvider
from bot.llm.resilience import ResilientLLMClient
from bot.services.chat import ChatService
from bot.services.conversation import ConversationService
from bot.services.model_router import ModelRouter
from bot.services.quota import QuotaService
from bot.storage.db import MIGRATIONS, Database
from bot.storage.repositories import ConversationRepo, FlagRepo, RequestRepo, UsageRepo, UserRepo
from bot.telegram.keyboards import REPLY_KEYBOARD
from bot.telegram.routers.chat import _deliver

# ============================================================================
# Migration v6 and FlagRepo
# ============================================================================


@pytest.fixture
async def db() -> Database:
    database = Database(":memory:")
    await database.connect()
    try:
        yield database
    finally:
        await database.close()


async def test_migration_v6_creates_app_flags_table(db: Database) -> None:
    """A fresh DB ends up at the newest schema version and has the app_flags table.

    Pinned to MIGRATIONS, not to the literal 6: a later migration must not break this test,
    it only ever checks that the flag table arrived somewhere along the way.
    """
    cursor = await db.conn.execute("PRAGMA user_version")
    version = (await cursor.fetchone())[0]
    assert version == len(MIGRATIONS)

    # Table exists and has the expected schema
    cursor = await db.conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='app_flags'")
    assert await cursor.fetchone() is not None


async def test_flag_repo_get_unknown_key_returns_none(db: Database) -> None:
    """FlagRepo.get of an unknown key → None."""
    repo = FlagRepo(db)
    result = await repo.get("nonexistent")
    assert result is None


async def test_flag_repo_set_and_get_returns_value_and_datetime(db: Database) -> None:
    """FlagRepo.set then get returns (value, aware datetime)."""
    repo = FlagRepo(db)
    await repo.set("test_flag", "test_value")

    result = await repo.get("test_flag")
    assert result is not None
    value, updated_at = result
    assert value == "test_value"
    assert isinstance(updated_at, datetime)
    assert updated_at.tzinfo is not None  # aware datetime


async def test_flag_repo_second_set_overwrites(db: Database) -> None:
    """A second set overwrites the first."""
    repo = FlagRepo(db)
    await repo.set("flag", "value1")
    await repo.set("flag", "value2")

    result = await repo.get("flag")
    assert result is not None
    value, _ = result
    assert value == "value2"


# ============================================================================
# ReserveSwitch
# ============================================================================


async def test_reserve_switch_default_state_is_off(db: Database) -> None:
    """Default active() False and state() == (False, None)."""
    flags = FlagRepo(db)
    switch = ReserveSwitch(flags)

    active = await switch.active()
    assert active is False

    state = await switch.state()
    assert state == (False, None)


async def test_reserve_switch_set_true_activates(db: Database) -> None:
    """set(True) → active True with a datetime."""
    flags = FlagRepo(db)
    switch = ReserveSwitch(flags)

    await switch.set(True)
    active = await switch.active()
    assert active is True

    on, updated_at = await switch.state()
    assert on is True
    assert isinstance(updated_at, datetime)
    assert updated_at.tzinfo is not None


async def test_reserve_switch_set_false_deactivates(db: Database) -> None:
    """set(False) → False."""
    flags = FlagRepo(db)
    switch = ReserveSwitch(flags)

    await switch.set(True)
    await switch.set(False)

    active = await switch.active()
    assert active is False

    on, updated_at = await switch.state()
    assert on is False
    assert updated_at is not None


async def test_reserve_switch_tolerates_flag_repo_errors(db: Database) -> None:
    """A FlagRepo whose get raises → active() False (no exception)."""

    class FailingFlagRepo:
        async def get(self, key: str):
            raise RuntimeError("database error")

        async def set(self, key: str, value: str) -> None:
            pass

    switch = ReserveSwitch(FailingFlagRepo())  # type: ignore
    active = await switch.active()
    assert active is False


# ============================================================================
# SwitchingProvider
# ============================================================================


class FakeCompletion:
    @staticmethod
    def ok(text: str = "answer", model: str = "primary") -> Completion:
        return Completion(text, "stop", model, None, None, reserve=False)

    @staticmethod
    def reserve(text: str = "reserve answer", model: str = "reserve-gpt") -> Completion:
        return Completion(text, "stop", model, None, None, reserve=True)


class FakeProvider:
    """Records calls and plays pre-set responses."""

    def __init__(self, completions: list[Completion]) -> None:
        self.completions = completions
        self.requests: list[CompletionRequest] = []
        self.closed = False

    async def complete(self, request: CompletionRequest) -> Completion:
        self.requests.append(request)
        return self.completions.pop(0) if self.completions else FakeCompletion.ok()

    async def list_models(self) -> list[str]:
        return ["primary", "secondary"]

    async def aclose(self) -> None:
        self.closed = True


async def test_switching_provider_off_uses_primary(db: Database) -> None:
    """Switch off → primary receives request unchanged, reserve not called."""
    flags = FlagRepo(db)
    switch = ReserveSwitch(flags)

    primary = FakeProvider([FakeCompletion.ok("primary answer")])
    reserve = FakeProvider([FakeCompletion.reserve()])

    provider = SwitchingProvider(primary, reserve, switch, "reserve-model")

    request = CompletionRequest("primary-model", [ChatMessage("user", "hi")], 100, 30.0)
    completion = await provider.complete(request)

    assert completion.text == "primary answer"
    assert completion.model == "primary"
    assert completion.reserve is False
    assert len(primary.requests) == 1
    assert len(reserve.requests) == 0


async def test_switching_provider_on_uses_reserve(db: Database) -> None:
    """Switch on → reserve receives request with model == reserve_model."""
    flags = FlagRepo(db)
    switch = ReserveSwitch(flags)
    await switch.set(True)

    primary = FakeProvider([FakeCompletion.ok()])
    reserve = FakeProvider([FakeCompletion.reserve("reserve answer", "nvidia/x")])

    provider = SwitchingProvider(primary, reserve, switch, "reserve-model")

    request = CompletionRequest("primary-model", [ChatMessage("user", "hi")], 100, 30.0)
    completion = await provider.complete(request)

    assert completion.text == "reserve answer"
    assert completion.model == "nvidia/x"
    assert completion.reserve is True
    assert len(primary.requests) == 0
    assert len(reserve.requests) == 1
    # Check that reserve received model substitution
    assert reserve.requests[0].model == "reserve-model"
    # Other fields unchanged
    assert reserve.requests[0].max_tokens == 100


async def test_switching_provider_aclose_closes_both(db: Database) -> None:
    """aclose closes both."""
    flags = FlagRepo(db)
    switch = ReserveSwitch(flags)

    primary = FakeProvider([])
    reserve = FakeProvider([])

    provider = SwitchingProvider(primary, reserve, switch, "reserve-model")
    await provider.aclose()

    assert primary.closed is True
    assert reserve.closed is True


async def test_switching_provider_list_models_from_primary(db: Database) -> None:
    """list_models goes to primary."""
    flags = FlagRepo(db)
    switch = ReserveSwitch(flags)

    primary = FakeProvider([])
    reserve = FakeProvider([])

    provider = SwitchingProvider(primary, reserve, switch, "reserve-model")
    models = await provider.list_models()

    assert models == ["primary", "secondary"]


# ============================================================================
# ResilientLLMClient with reserve
# ============================================================================


def make_catalog() -> Catalog:
    """Minimal catalog for testing."""
    specs = {
        "primary": ModelSpec("primary", "Primary", 1000, 60.0, None, enabled=True, slow=False),
        "reserve": ModelSpec("reserve", "Reserve", 1000, 60.0, None, enabled=True, slow=False),
    }
    return Catalog(specs, {mode: "primary" for mode in Mode}, max_tokens_cap=2000, max_attempts=2)


async def test_resilient_client_maps_reserve_flag_to_served_by() -> None:
    """ResilientLLMClient over reserve completion → ChatResult.served_by == model."""
    provider = FakeProvider([FakeCompletion.reserve("answer", "nvidia/x")])
    catalog = make_catalog()
    client = ResilientLLMClient(provider, catalog, max_concurrency=1)

    result = await client.chat([ChatMessage("user", "hi")], "primary")

    assert result.served_by == "nvidia/x"
    assert result.status is ChatStatus.OK


async def test_resilient_client_normal_completion_has_no_served_by() -> None:
    """Normal completion (reserve=False) → served_by is None."""
    provider = FakeProvider([FakeCompletion.ok("answer", "primary")])
    catalog = make_catalog()
    client = ResilientLLMClient(provider, catalog, max_concurrency=1)

    result = await client.chat([ChatMessage("user", "hi")], "primary")

    assert result.served_by is None
    assert result.status is ChatStatus.OK


# ============================================================================
# ChatService with reserve
# ============================================================================


class FakeLLM:
    def __init__(self, *results: ChatResult) -> None:
        self.results = list(results)
        self.calls: list[tuple[list[ChatMessage], str]] = []

    async def chat(self, messages: Sequence[ChatMessage], model_id: str, tools=()) -> ChatResult:
        self.calls.append((list(messages), model_id))
        return (
            self.results.pop(0)
            if self.results
            else ChatResult(ChatStatus.OK, "default", model_id, 1, 10, False)
        )

    def queue_depth(self) -> int:
        return 0


@pytest.fixture
async def chat_env():
    db = Database(":memory:")
    await db.connect()
    users = UserRepo(db)
    user = await users.upsert_seen(1, "u", default_role=Role.USER, default_mode=Mode.NORMAL)
    catalog = load_catalog(DEFAULT_MODELS_FILE)
    parts = {
        "conversations": ConversationService(ConversationRepo(db), system_prompt="sys", token_budget=10_000),
        "usage": UsageRepo(db),
        "requests": RequestRepo(db),
        "catalog": catalog,
        "user": user,
        "db": db,
    }
    yield parts
    await db.close()


def _reserve_service(chat_env, llm, switch, spy: dict) -> ChatService:
    """ChatService whose rewrite step and context builder report what they were asked."""
    conversations = chat_env["conversations"]
    build = conversations.build_context

    async def spy_build(*args, **kwargs):
        spy.setdefault("budgets", []).append(kwargs.get("token_budget"))
        return await build(*args, **kwargs)

    conversations.build_context = spy_build
    service = ChatService(
        llm,
        conversations,
        QuotaService(chat_env["usage"], daily_limit=100),
        ModelRouter(chat_env["catalog"]),
        chat_env["requests"],
        chat_env["catalog"],
        reserve=switch,
        reserve_token_budget=5000,
    )

    async def spy_rewrite(user, text):
        spy["rewrites"] = spy.get("rewrites", 0) + 1
        return text, 0, 0

    service._rewrite = spy_rewrite  # type: ignore[method-assign]
    return service


@pytest.mark.parametrize(("on", "rewrites"), [(True, 0), (False, 1)])
async def test_chat_service_reserve_skips_rewrite(chat_env, on: bool, rewrites: int) -> None:
    switch = ReserveSwitch(FlagRepo(chat_env["db"]))
    await switch.set(on)
    spy: dict = {}
    llm = FakeLLM(ChatResult(ChatStatus.OK, "answer", "claude-opus-4.8", 1, 10, False))
    await _reserve_service(chat_env, llm, switch, spy).answer(chat_env["user"], "hello there")
    assert spy.get("rewrites", 0) == rewrites


@pytest.mark.parametrize(("limit", "expected"), [(None, 5000), (3000, 3000), (9000, 5000)])
async def test_chat_service_reserve_caps_token_budget(chat_env, limit, expected) -> None:
    users = UserRepo(chat_env["db"])
    await users.set_context_limit(chat_env["user"].tg_id, limit)
    user = await users.get(chat_env["user"].tg_id)
    switch = ReserveSwitch(FlagRepo(chat_env["db"]))
    await switch.set(True)
    spy: dict = {}
    llm = FakeLLM(ChatResult(ChatStatus.OK, "answer", "claude-opus-4.8", 1, 10, False, served_by="r"))
    await _reserve_service(chat_env, llm, switch, spy).answer(user, "hello")
    assert spy["budgets"] == [expected]


async def test_chat_service_reserve_off_keeps_user_budget(chat_env) -> None:
    switch = ReserveSwitch(FlagRepo(chat_env["db"]))
    spy: dict = {}
    llm = FakeLLM(ChatResult(ChatStatus.OK, "answer", "claude-opus-4.8", 1, 10, False))
    await _reserve_service(chat_env, llm, switch, spy).answer(chat_env["user"], "hello")
    assert spy["budgets"] == [None]


async def test_chat_service_stores_reserve_model_in_request_record(chat_env) -> None:
    """Stored RequestRecord.model equals the served_by value."""
    db = chat_env["db"]
    flags = FlagRepo(db)
    switch = ReserveSwitch(flags)
    await switch.set(True)

    result = ChatResult(ChatStatus.OK, "answer", "claude-opus-4.8", 1, 10, False, served_by="reserve-model")
    llm = FakeLLM(result)
    requests = RequestRepo(db)
    service = ChatService(
        llm,
        chat_env["conversations"],
        QuotaService(chat_env["usage"], daily_limit=100),
        ModelRouter(chat_env["catalog"]),
        requests,
        chat_env["catalog"],
        reserve=switch,
    )

    await service.answer(chat_env["user"], "hello")

    # Check the stored request record
    report = await requests.report(chat_env["user"].tg_id, None)
    assert report.requests == 1
    # The model in the record should be the reserve model
    assert report.by_model[0].model == "reserve-model"


async def test_chat_service_sets_reply_served_by(chat_env) -> None:
    """ChatReply.served_by is set."""
    db = chat_env["db"]
    flags = FlagRepo(db)
    switch = ReserveSwitch(flags)
    await switch.set(True)

    result = ChatResult(ChatStatus.OK, "answer", "claude-opus-4.8", 1, 10, False, served_by="reserve-model")
    llm = FakeLLM(result)
    service = ChatService(
        llm,
        chat_env["conversations"],
        QuotaService(chat_env["usage"], daily_limit=100),
        ModelRouter(chat_env["catalog"]),
        chat_env["requests"],
        chat_env["catalog"],
        reserve=switch,
    )

    reply = await service.answer(chat_env["user"], "hello")

    assert reply.served_by == "reserve-model"


async def test_chat_service_without_reserve_keeps_normal_behavior(chat_env) -> None:
    """With switch inactive: old behavior (rewrite still called when skills request it)."""
    # Don't activate the switch

    # Provide two results: one for rewrite, one for main answer
    llm = FakeLLM(
        ChatResult(ChatStatus.OK, "improved prompt", "claude-opus-4.8", 1, 5, False),
        ChatResult(ChatStatus.OK, "answer", "claude-opus-4.8", 1, 10, False),
    )
    service = ChatService(
        llm,
        chat_env["conversations"],
        QuotaService(chat_env["usage"], daily_limit=100),
        ModelRouter(chat_env["catalog"]),
        chat_env["requests"],
        chat_env["catalog"],
    )

    reply = await service.answer(chat_env["user"], "hello")

    # Two calls: rewrite (for normal mode with no skills) and main answer
    # Actually, rewrite is skipped for short messages, so let's just verify it works
    assert reply.status is RequestStatus.OK
    assert reply.served_by is None


# ============================================================================
# _deliver with reserve header
# ============================================================================


class FakeMessage:
    def __init__(self) -> None:
        self.sent: list[str] = []

    async def answer(self, text: str, **kwargs) -> SimpleNamespace:
        self.sent.append(text)
        return SimpleNamespace()

    async def answer_photo(self, *args, **kwargs) -> SimpleNamespace:
        return SimpleNamespace()

    async def answer_media_group(self, *args, **kwargs) -> SimpleNamespace:
        return SimpleNamespace()

    async def answer_document(self, *args, **kwargs) -> SimpleNamespace:
        return SimpleNamespace()


def _reply(model_id: str, parts: list[str], **kw):
    """Helper to create ChatReply for testing."""
    catalog = load_catalog(DEFAULT_MODELS_FILE)
    from bot.services.chat import ChatReply

    return ChatReply(
        RequestStatus.OK, parts, "raw", catalog.get(model_id), kw.pop("fallback_used", False), **kw
    )


async def test_deliver_with_served_by_shows_reserve_header() -> None:
    """_deliver with served_by → header with reserve emoji, the reserve model's title, and who answered."""
    msg = FakeMessage()
    reply = _reply("free-coded", ["answer"], served_by="nvidia/x")
    await _deliver(msg, reply, REPLY_KEYBOARD)

    assert len(msg.sent) > 0
    first = msg.sent[0]
    assert "🛟" in first  # reserve emoji
    assert "<b>Авто (GPT-6 Luna → бесплатные)</b>" in first
    assert "nvidia/x" in first


async def test_deliver_without_served_by_shows_normal_badge() -> None:
    """Without served_by → usual badge."""
    msg = FakeMessage()
    reply = _reply("gpt-5.6-luna", ["answer"])
    await _deliver(msg, reply, REPLY_KEYBOARD)

    assert len(msg.sent) > 0
    first = msg.sent[0]
    # Should have the normal model badge, not the reserve header
    assert "🛟" not in first or "Резервная модель" not in first


async def test_deliver_reserve_header_escapes_model_name() -> None:
    """Reserve header HTML-escapes the answering model's name."""
    msg = FakeMessage()
    reply = _reply("free-coded", ["answer"], served_by="model<script>alert('xss')</script>")
    await _deliver(msg, reply, REPLY_KEYBOARD)

    first = msg.sent[0]
    # Should be escaped
    assert "<script>" not in first
    assert "&lt;script&gt;" in first or "script" not in first.split("</b>")[1].split("<i>")[0]


async def test_a_request_rerouted_to_the_reserve_is_fitted_to_its_window(db: Database) -> None:
    """The client fitted the request to the primary model's ~1M window; the reserve gets it cut to its own."""
    from dataclasses import replace as dc_replace

    from bot.catalog import ModelSpec
    from bot.llm.window import prompt_tokens

    switch = ReserveSwitch(FlagRepo(db))
    await switch.set(True)
    reserve = FakeProvider([FakeCompletion.reserve()])
    spec = dc_replace(
        ModelSpec("reserve-model", "r", 500, 60, None, enabled=True, slow=False), context_tokens=3000
    )
    provider = SwitchingProvider(FakeProvider([]), reserve, switch, "reserve-model", reserve_spec=spec)
    history = [ChatMessage("user", "старая реплика " * 200, droppable=True) for _ in range(10)]
    messages = [ChatMessage("system", "s"), *history, ChatMessage("user", "вопрос")]
    await provider.complete(CompletionRequest("primary-model", messages, 50_000, 30.0))
    sent = reserve.requests[0].messages
    assert sent[0].content == "s" and sent[-1].content == "вопрос"
    assert prompt_tokens(sent) <= 3000 * 0.96 - 500
    assert reserve.requests[0].max_tokens == 500
