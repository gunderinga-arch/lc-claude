from collections.abc import Sequence
from datetime import UTC, datetime

import pytest

from bot.catalog import load_catalog
from bot.config import DEFAULT_MODELS_FILE
from bot.domain import IMAGE_TOKENS, Mode, RequestStatus, Role, User, approx_tokens
from bot.llm.base import ChatMessage, ChatResult, ChatStatus, ToolCall
from bot.services.chat import ChatService
from bot.services.conversation import ConversationService
from bot.services.model_router import ModelRouter
from bot.services.quota import QuotaService
from bot.storage.db import Database
from bot.storage.repositories import ConversationRepo, RequestRepo, UsageRepo, UserRepo


class FakeLLM:
    def __init__(self, *results: ChatResult) -> None:
        self.results = list(results)
        self.calls: list[tuple[list[ChatMessage], str]] = []
        self.tools: list[tuple] = []

    async def chat(self, messages: Sequence[ChatMessage], model_id: str, tools=()) -> ChatResult:
        self.calls.append((list(messages), model_id))
        self.tools.append(tuple(tools))
        return self.results.pop(0)

    def queue_depth(self) -> int:
        return 0


def ok(text: str, model: str = "claude-opus-5", fallback: bool = False) -> ChatResult:
    return ChatResult(ChatStatus.OK, text, model, 1, 10, fallback)


@pytest.fixture
async def env():
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


def make_service(env, llm: FakeLLM) -> ChatService:
    return ChatService(
        llm,
        env["conversations"],
        QuotaService(env["usage"], daily_limit=10),
        ModelRouter(env["catalog"]),
        env["requests"],
        env["catalog"],
    )


async def test_ok_answer_is_rendered_saved_and_counted(env) -> None:
    llm = FakeLLM(ok("**hi**"), ok("again"))
    service = make_service(env, llm)

    reply = await service.answer(env["user"], "hello")

    assert reply.status is RequestStatus.OK
    assert reply.parts == ["<b>hi</b>"]
    assert reply.text == "**hi**"
    assert reply.model.id == "claude-opus-5"
    assert llm.calls[0][1] == "claude-opus-5"
    assert await env["usage"].requests_today(1) == 1
    assert (await env["requests"].stats_today()).requests_today == 1

    await service.answer(env["user"], "next")
    history = [(m.role, m.content) for m in llm.calls[1][0]]
    assert history == [("system", "sys"), ("user", "hello"), ("assistant", "**hi**"), ("user", "next")]


async def test_failed_and_stub_are_counted_but_not_saved(env) -> None:
    llm = FakeLLM(
        ChatResult(ChatStatus.FAILED, "", "gpt-5.6-sol", 6, 50, True, "timeout"),
        ChatResult(ChatStatus.STUB, "", "claude-opus-5", 1, 5, False),
        ok("fine"),
    )
    service = make_service(env, llm)

    failed = await service.answer(env["user"], "q1")
    assert failed.status is RequestStatus.FAILED
    assert failed.parts == []
    assert failed.error == "timeout"
    assert failed.model.id == "gpt-5.6-sol"
    assert failed.fallback_used

    stub = await service.answer(env["user"], "q2")
    assert stub.status is RequestStatus.STUB

    await service.answer(env["user"], "q3")
    assert [m.content for m in llm.calls[2][0]] == ["sys", "q3"]
    stats = await env["requests"].stats_today()
    assert (stats.requests_today, stats.failed_today) == (3, 1)
    assert await env["usage"].requests_today(1) == 3


async def test_no_models_keeps_requested_spec(env) -> None:
    llm = FakeLLM(ChatResult(ChatStatus.FAILED, "", "claude-opus-5", 0, 0, False, "no_models"))
    reply = await make_service(env, llm).answer(env["user"], "q")
    assert reply.model.id == "claude-opus-5"
    assert reply.error == "no_models"


async def test_mode_and_override_pick_the_model(env) -> None:
    llm = FakeLLM(ok("a", "claude-fable-5.1"), ok("b", "gpt-6-sol"))
    service = make_service(env, llm)
    deep = User(1, "u", Role.USER, Mode.DEEP, None, env["user"].created_at)
    await service.answer(deep, "q")
    overridden = User(1, "u", Role.USER, Mode.DEEP, "gpt-6-sol", env["user"].created_at)
    await service.answer(overridden, "q")
    assert [model for _, model in llm.calls] == ["claude-fable-5.1", "gpt-6-sol"]


async def test_files_are_cut_from_answer_but_kept_in_history(env) -> None:
    answer = 'Вот:\n<<<FILE name="a.csv">>>\nx,y\n<<<END>>>'
    llm = FakeLLM(ok(answer), ok("ок"))
    service = make_service(env, llm)
    reply = await service.answer(env["user"], "сделай csv", ("data:image/png;base64,AA",))
    assert [f.name for f in reply.files] == ["a.csv"]
    assert "<<<FILE" not in "".join(reply.parts) and "a.csv" in "".join(reply.parts)
    assert llm.calls[0][0][-1].images == ("data:image/png;base64,AA",)

    await service.answer(env["user"], "поменяй")
    history = llm.calls[1][0]
    assert history[2].content == answer  # the model sees its own file for follow-up edits
    assert history[1].images == ()  # images are not re-sent from history


async def test_filtered_history_is_dropped_and_question_retried_alone(env) -> None:
    stub = ChatResult(ChatStatus.STUB, "", "claude-opus-5", 1, 5, False)
    llm = FakeLLM(ok("про Тяньаньмэнь"), stub, ok("про фотоаппараты"), ok("дальше"))
    service = make_service(env, llm)
    await service.answer(env["user"], "политика")

    reply = await service.answer(env["user"], "фотоаппарат?")
    assert reply.status is RequestStatus.OK and reply.context_reset
    assert [m.content for m in llm.calls[2][0]] == ["sys", "фотоаппарат?"]  # retried without history

    await service.answer(env["user"], "а ещё?")
    history = [m.content for m in llm.calls[3][0]]
    assert history == ["sys", "фотоаппарат?", "про фотоаппараты", "а ещё?"]  # poisoned pair is gone
    stats = await env["requests"].stats_today()
    assert stats.requests_today == 3  # the retry is part of the same request


async def test_stub_without_history_is_not_retried(env) -> None:
    stub = ChatResult(ChatStatus.STUB, "", "claude-opus-5", 1, 5, False)
    llm = FakeLLM(stub)
    reply = await make_service(env, llm).answer(env["user"], "запретная тема")
    assert reply.status is RequestStatus.STUB and not reply.context_reset
    assert len(llm.calls) == 1


async def test_stub_after_retry_keeps_history(env) -> None:
    stub = ChatResult(ChatStatus.STUB, "", "claude-opus-5", 1, 5, False)
    llm = FakeLLM(ok("ответ"), stub, stub, ok("снова"))
    service = make_service(env, llm)
    await service.answer(env["user"], "q1")
    reply = await service.answer(env["user"], "запретная тема")
    assert reply.status is RequestStatus.STUB and not reply.context_reset
    await service.answer(env["user"], "q3")
    assert [m.content for m in llm.calls[3][0]] == ["sys", "q1", "ответ", "q3"]  # history untouched


async def test_rewrite_skill_improves_prompt_but_history_keeps_original(env) -> None:
    from dataclasses import replace

    from bot.skills import Skill

    skills = {"pm": Skill("pm", "PM", "REWRITER", rewrite=True), "x": Skill("x", "X", "EXTRA")}
    env["conversations"] = ConversationService(
        ConversationRepo(env["db"]), system_prompt="sys", token_budget=10_000, skills=skills
    )
    user = replace(env["user"], skills=("pm", "x"))
    original = "расскажи про чёрные дыры подробно"
    llm = FakeLLM(ok("<request>IMPROVED</request>"), ok("answer"), ok("second"))
    service = make_service(env, llm)

    await service.answer(user, original)
    (rewrite_msgs, rewrite_model), (main_msgs, _) = llm.calls
    assert rewrite_model == env["catalog"].for_mode(Mode.FAST).id
    assert rewrite_msgs[0].content == "REWRITER" and original in rewrite_msgs[1].content
    assert main_msgs[0].content == "sys\n\nEXTRA"  # the rewriter prompt never reaches the main model
    assert main_msgs[-1].content == "IMPROVED"

    await service.answer(user, "спасибо")  # too short: no rewrite call
    assert len(llm.calls) == 3
    assert [m.content for m in llm.calls[2][0][1:]] == [original, "answer", "спасибо"]


async def test_failed_rewrite_falls_back_to_user_text(env) -> None:
    from dataclasses import replace

    from bot.skills import Skill

    env["conversations"] = ConversationService(
        ConversationRepo(env["db"]),
        system_prompt="sys",
        token_budget=10_000,
        skills={"pm": Skill("pm", "PM", "REWRITER", rewrite=True)},
    )
    failed = ChatResult(ChatStatus.FAILED, "", "gpt-5.6-luna", 3, 10, False, "timeout")
    llm = FakeLLM(failed, ok("answer"))
    text = "объясни разницу между TCP и UDP"
    reply = await make_service(env, llm).answer(replace(env["user"], skills=("pm",)), text)
    assert reply.status is RequestStatus.OK
    assert llm.calls[1][0][-1].content == text


# ---- WP-2: token accounting by kind, web calls, per-user context limit -------------------------


class FakeWeb:
    def __init__(self) -> None:
        self.calls: list[ToolCall] = []

    async def run(self, call: ToolCall) -> str:
        self.calls.append(call)
        return f"RESULT {call.id}"


def tool_turn(text: str, *calls: ToolCall) -> ChatResult:
    return ChatResult(ChatStatus.OK, text, "claude-opus-5", 1, 10, False, tool_calls=calls)


async def test_usage_split_by_kind_and_context_grows_with_history(env) -> None:
    llm = FakeLLM(ok("hi"), ok("again"))
    service = make_service(env, llm)

    t0 = datetime.now(UTC)
    await service.answer(env["user"], "hello")
    first = await env["requests"].report(env["user"].tg_id, t0)
    assert first.usage.input == approx_tokens("sys") + approx_tokens("hello")
    assert first.usage.output == approx_tokens("hi")
    assert (first.usage.images, first.usage.service, first.usage.tools, first.usage.context) == (0, 0, 0, 0)
    assert first.total_tokens == first.usage.total

    t1 = datetime.now(UTC)
    await service.answer(env["user"], "next")
    second = await env["requests"].report(env["user"].tg_id, t1)
    assert second.usage.context > 0  # history (the first exchange) is now sent along
    assert second.total_tokens == second.usage.total


async def test_image_input_counts_image_tokens(env) -> None:
    llm = FakeLLM(ok("desc"))
    service = make_service(env, llm)
    t0 = datetime.now(UTC)
    await service.answer(env["user"], "что на фото?", ("data:image/png;base64,AA",))
    report = await env["requests"].report(env["user"].tg_id, t0)
    assert report.usage.images == IMAGE_TOKENS


async def test_rewrite_skill_records_service_tokens(env) -> None:
    from dataclasses import replace

    from bot.skills import Skill

    skills = {"pm": Skill("pm", "PM", "REWRITER", rewrite=True)}
    env["conversations"] = ConversationService(
        ConversationRepo(env["db"]), system_prompt="sys", token_budget=10_000, skills=skills
    )
    user = replace(env["user"], skills=("pm",))
    original = "расскажи про чёрные дыры подробно"
    llm = FakeLLM(ok("<request>IMPROVED</request>"), ok("answer"))
    service = make_service(env, llm)

    await service.answer(user, original)

    rewrite_msgs = llm.calls[0][0]
    expected_service = sum(approx_tokens(m.content) for m in rewrite_msgs) + approx_tokens(
        "<request>IMPROVED</request>"
    )
    report = await env["requests"].report(user.tg_id, None)
    assert report.usage.service == expected_service > 0


async def test_web_tool_round_counts_tools_and_web_calls(env) -> None:
    from dataclasses import replace

    from bot.skills import Skill

    call = ToolCall("call_1", "web_search", '{"query": "x"}')
    llm = FakeLLM(tool_turn("ищу...", call), ok("итог"))
    web = FakeWeb()
    env["conversations"] = ConversationService(
        ConversationRepo(env["db"]),
        system_prompt="sys",
        token_budget=10_000,
        skills={"net": Skill("net", "Net", "NET", web=True)},
    )
    service = ChatService(
        llm,
        env["conversations"],
        QuotaService(env["usage"], daily_limit=10),
        ModelRouter(env["catalog"]),
        env["requests"],
        env["catalog"],
        web,
    )
    user = replace(env["user"], skills=("net",))

    reply = await service.answer(user, "вопрос")

    assert reply.status is RequestStatus.OK
    assert web.calls == [call]
    report = await env["requests"].report(user.tg_id, None)
    assert report.web_calls == 1
    assert report.usage.tools > 0
    assert report.usage.output == approx_tokens("ищу...") + approx_tokens("итог")


async def test_stub_retry_sums_usage_of_both_attempts(env) -> None:
    llm = FakeLLM(ok("first"), ChatResult(ChatStatus.STUB, "", "claude-opus-5", 1, 5, False), ok("second"))
    service = make_service(env, llm)
    await service.answer(env["user"], "q1")

    t0 = datetime.now(UTC)
    reply = await service.answer(env["user"], "q2")
    assert reply.status is RequestStatus.OK and reply.context_reset

    stub_messages, retry_messages = llm.calls[1][0], llm.calls[2][0]
    expected_input = sum(approx_tokens(m.content) for m in stub_messages) + sum(
        approx_tokens(m.content) for m in retry_messages
    )
    report = await env["requests"].report(env["user"].tg_id, t0)
    assert report.usage.input == expected_input
    assert report.usage.output == approx_tokens("second")  # the STUB attempt wrote no text
    assert report.usage.total == report.total_tokens


async def test_user_context_limit_trims_history_more_than_global(env) -> None:
    from dataclasses import replace

    repo = ConversationRepo(env["db"])
    conv_id = await repo.active_id(env["user"].tg_id)
    for i in range(5):
        await repo.add_message(conv_id, "user", f"u{i}", 100)
        await repo.add_message(conv_id, "assistant", f"a{i}", 100)

    llm = FakeLLM(ok("a"), ok("b"))
    service = make_service(env, llm)

    # system(1) + user(1) + one 100-token message = 102 fits, a second one (202) does not, and the
    # single kept message is an assistant one with no user before it, so it is dropped as an orphan.
    limited = replace(env["user"], context_limit=150)
    await service.answer(limited, "q")
    assert len(llm.calls[0][0]) == 2

    # The first answer added its exchange to the history, so 12 stored messages are all kept here.
    await service.answer(env["user"], "q")  # context_limit=None falls back to the global 10_000 budget
    assert len(llm.calls[1][0]) == 14  # system + all 12 history messages + user


async def test_chat_title_uses_haiku_on_the_reserve_gateway_and_hides_it_from_choice(env) -> None:
    from bot.services.chat import TITLE_MODEL

    catalog = env["catalog"]
    assert TITLE_MODEL not in {m.id for m in catalog.reserve_selectable() + catalog.selectable()}

    llm = FakeLLM(ok("Авиастроение", TITLE_MODEL))
    service = make_service(env, llm)
    service._reserve = object()  # a reserve gateway is configured
    assert await service.chat_title("Давай поговорим про авиастроение") == "Авиастроение"
    assert llm.calls[-1][1] == TITLE_MODEL

    llm = FakeLLM(ok("Мамонты", "gpt-6-sol"))
    service = make_service(env, llm)  # no reserve gateway: the fast mode's model names it
    assert await service.chat_title("Хочу поговорить про мамонтов") == "Мамонты"
    assert llm.calls[-1][1] == catalog.for_mode(Mode.FAST).id


async def test_pictures_lost_by_the_gateway_are_described_by_a_model_that_sees_them(env) -> None:
    from bot.services.chat import SEEN_NOTE, VISION_FALLBACK

    blind = ok("Картинку я не получил — служба распознавания изображений не ответила.", "claude-opus-5")
    seen = ok("Крупная надпись «ШТРИХ-КОД», чёрный гротеск на светлом фоне #F2EEE6.", VISION_FALLBACK)
    llm = FakeLLM(blind, seen, ok("На фото слово «ШТРИХ-КОД»."), ok("сводка"), ok("сводка"))
    picture = "data:image/jpeg;base64,AAAA"
    reply = await make_service(env, llm).answer(env["user"], "что на фото?", (picture,))
    assert reply.status is RequestStatus.OK and "ШТРИХ-КОД" in reply.text
    assert [c[1] for c in llm.calls[:3]] == ["claude-opus-5", VISION_FALLBACK, "claude-opus-5"]
    assert llm.calls[1][0][-1].images == (picture,)  # the describer gets the pictures
    redo = llm.calls[2][0][-1]  # the user's model gets them as words
    assert redo.images == () and SEEN_NOTE in redo.content and "#F2EEE6" in redo.content


async def test_without_pictures_such_words_are_just_an_answer(env) -> None:
    llm = FakeLLM(
        ok("Фото не дошло до редакции, пишет автор статьи.", "claude-opus-5"), ok("сводка"), ok("сводка")
    )
    reply = await make_service(env, llm).answer(env["user"], "перескажи статью", ())
    assert "редакции" in reply.text and llm.calls[0][1] == "claude-opus-5" and len(llm.calls) <= 2


async def test_broken_deck_json_is_mended_around_the_error_only(env) -> None:
    import json as _json

    broken = "\n".join([
        "{", ' "title": "a",', ' "slides": [', '  {"elements": [',
        '   {"type": "text", "size": 10."size": 10.5, "text": "x"}', "  ]}", " ]", "}",
    ])  # fmt: skip
    good_line = '   {"type": "text", "size": 10.5, "text": "x"}'
    numbered = "\n".join(f"{n}: {line}" for n, line in enumerate(broken.split("\n")[2:7], 3))
    numbered = numbered.replace('10."size": 10.5', "10.5")  # the model returns the lines fixed, numbers kept
    llm = FakeLLM(ok(numbered))
    fixed = await make_service(env, llm)._mend_json(broken, "claude-opus-5")
    assert fixed is not None and _json.loads(fixed)["slides"][0]["elements"][0]["size"] == 10.5
    assert good_line in fixed.split("\n") and len(llm.calls) == 1
    assert "строка 5" in llm.calls[0][0][0].content  # only the lines around the error were sent

    llm = FakeLLM(ok("всё ещё {сломано"), ok("всё ещё {сломано"), ok("всё ещё {сломано"))
    assert await make_service(env, llm)._mend_json(broken, "claude-opus-5") is None


async def test_a_long_dialog_is_compacted_at_the_edge_of_its_budget(env) -> None:
    from dataclasses import replace

    from bot.services import conversation as conv
    from bot.services.chat import SUMMARIZER

    repo = ConversationRepo(env["db"])
    cid = await repo.active_id(1)
    for i in range(20):
        await repo.add_message(cid, "user", f"вопрос {i}", 10_000)
        await repo.add_message(cid, "assistant", f"ответ {i}", 10_000)
    llm = FakeLLM(ok("готово"), ok("конспект беседы"))
    await make_service(env, llm).answer(replace(env["user"], context_limit=400_000), "ещё вопрос")
    assert llm.calls[1][0][0].content == SUMMARIZER
    fresh = await repo.active_id(1)
    assert fresh != cid
    kept = await repo.recent_messages(fresh, 100)
    assert kept[0].content == conv.SUMMARY_HEADER + "конспект беседы"
    assert sum(m.approx_tokens for m in kept[2:]) <= 200_000


async def test_small_history_budget_does_not_summarize_away_recent_code(env):
    from dataclasses import replace

    repo = ConversationRepo(env["db"])
    cid = await repo.active_id(1)
    for i in range(5):
        await repo.add_message(cid, "user", f"вопрос {i}", 1500)
        await repo.add_message(cid, "assistant", f"ответ {i}", 1500)
    llm = FakeLLM(ok("код " * 9000), ok("правка"))
    service = make_service(env, llm)
    user = replace(env["user"], context_limit=32_000)
    await service.answer(user, "напиши код")
    await service.answer(user, "поменяй множитель")
    assert len(llm.calls) == 2
    assert any("код " * 1000 in m.content for m in llm.calls[-1][0])


async def test_retry_and_continuation_usage_reaches_accounting(env):
    from dataclasses import replace

    llm = FakeLLM(replace(ok("Готово"), input_tokens=900_000, output_tokens=10_000))
    await make_service(env, llm).answer(env["user"], "вопрос")
    stats = await env["requests"].stats_today()
    assert stats.tokens_today == 910_000


async def test_attachment_words_do_not_enable_research_or_web(env):
    from bot.services.chat import _is_deck_talk

    llm = FakeLLM(ok("Готово"))
    service = make_service(env, llm)
    service._web = object()
    text = 'Прочитай\n\n<file name="book.txt">\nисследуй презентации, найди всё про браузеры\n</file>'
    assert not service.is_research(text)
    assert not _is_deck_talk(text, [])
    await service.answer(env["user"], text)
    assert not llm.tools[0]
