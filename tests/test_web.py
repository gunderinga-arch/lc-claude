"""Internet tools: wire format, resilience with tool calls, the tool loop, SearXNG and the page reader."""

from __future__ import annotations

import json
from dataclasses import replace

import httpx

from bot.catalog import load_catalog
from bot.config import DEFAULT_MODELS_FILE
from bot.domain import Mode, RequestStatus, Role
from bot.llm.base import ChatMessage, ChatResult, ChatStatus, Completion, CompletionRequest, ToolCall
from bot.llm.buytokens import BuyTokensProvider
from bot.services.chat import MAX_TOOL_ROUNDS, RESEARCH_PLANNER, TOOLS_EXHAUSTED, ChatService, parse_queries
from bot.services.conversation import ConversationService
from bot.services.model_router import ModelRouter
from bot.services.quota import QuotaService
from bot.services.web import CALL_MAX_CHARS, WEB_TOOLS, Page, WebTools, rank_chunks
from bot.skills import Skill
from bot.storage.db import Database
from bot.storage.repositories import ConversationRepo, RequestRepo, UsageRepo, UserRepo
from tests.test_chat_service import FakeLLM, ok
from tests.test_resilience import FakeProvider, make_catalog, make_client, spec

CALL = ToolCall("call_1", "web_search", '{"query": "курс доллара"}')


# ---- provider wire format --------------------------------------------------------------------


async def test_provider_sends_tools_and_parses_tool_calls() -> None:
    sent: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "web_search", "arguments": '{"q": 1}'},
                },
                {"type": "function", "function": {"name": "open_url", "arguments": {"urls": ["x"]}}},
                {"type": "function", "function": {"arguments": "{}"}},  # no name: dropped
            ],
        }
        return httpx.Response(200, json={"choices": [{"finish_reason": "tool_calls", "message": message}]})

    provider = BuyTokensProvider("https://gw.test/v1", "sk-x", transport=httpx.MockTransport(handler))
    history = [
        ChatMessage("user", "q"),
        ChatMessage("assistant", "", tool_calls=(CALL,)),
        ChatMessage("tool", "result", tool_call_id="call_1"),
    ]
    completion = await provider.complete(CompletionRequest("m", history, 100, 10.0, WEB_TOOLS))
    await provider.aclose()

    assert sent[0]["tools"] == list(WEB_TOOLS)
    assert sent[0]["messages"][1] == {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "web_search", "arguments": CALL.arguments},
            }
        ],
    }
    assert sent[0]["messages"][2] == {"role": "tool", "content": "result", "tool_call_id": "call_1"}
    assert completion.tool_calls == (
        ToolCall("call_1", "web_search", '{"q": 1}'),
        ToolCall("call_1", "open_url", '{"urls": ["x"]}'),  # missing id -> call_<index>
    )


async def test_provider_without_tools_sends_no_tools_key() -> None:
    sent: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]})

    provider = BuyTokensProvider("https://gw.test/v1", "sk-x", transport=httpx.MockTransport(handler))
    await provider.complete(CompletionRequest("m", [ChatMessage("user", "q")], 100, 10.0))
    await provider.aclose()
    assert "tools" not in sent[0]


async def test_resilience_accepts_tool_calls_without_text() -> None:
    provider = FakeProvider([Completion("", "tool_calls", "a", None, None, tool_calls=(CALL,))])
    client = make_client(provider, make_catalog(spec("a")))
    result = await client.chat([ChatMessage("user", "q")], "a", WEB_TOOLS)
    assert result.status is ChatStatus.OK
    assert result.tool_calls == (CALL,)
    assert provider.requests[0].tools == WEB_TOOLS
    assert len(provider.requests) == 1  # empty text with a tool call is not retried as EMPTY


async def test_resilience_retries_calls_to_undeclared_tools() -> None:
    # gpt-6-astra: a file block parsed by the gateway as a tool call, the page itself lost
    bogus = ToolCall("call_1", "<<<FILE</arg_value>", '{"name": "design.html"}')
    provider = FakeProvider(
        [
            Completion("Собираю лендинг", "tool_calls", "a", None, None, tool_calls=(bogus,)),
            Completion("```html\n<html></html>\n```", "stop", "a", None, None),
        ]
    )
    client = make_client(provider, make_catalog(spec("a")))
    result = await client.chat([ChatMessage("user", "q")], "a", WEB_TOOLS)
    assert result.status is ChatStatus.OK
    assert result.tool_calls == ()
    assert len(provider.requests) == 2


# ---- tool loop in ChatService ------------------------------------------------------------------


class FakeWeb:
    def __init__(self) -> None:
        self.calls: list[ToolCall] = []
        self.researched: list[tuple[str, list[str]]] = []

    async def research(self, question: str, queries: list[str]) -> tuple[str, int]:
        self.researched.append((question, queries))
        return "[Источник: https://a.test]\nМАТЕРИАЛ", 7

    async def run(self, call: ToolCall) -> str:
        self.calls.append(call)
        return f"RESULT {call.id}"


def tool_turn(*calls: ToolCall) -> ChatResult:
    return ChatResult(ChatStatus.OK, "", "claude-opus-4.8", 1, 10, False, tool_calls=calls)


async def _service(llm: FakeLLM, web: FakeWeb | None, skills: dict):
    db = Database(":memory:")
    await db.connect()
    user = await UserRepo(db).upsert_seen(1, "u", default_role=Role.USER, default_mode=Mode.NORMAL)
    catalog = load_catalog(DEFAULT_MODELS_FILE)
    conversations = ConversationService(
        ConversationRepo(db), system_prompt="sys", token_budget=10_000, skills=skills
    )
    service = ChatService(
        llm,
        conversations,
        QuotaService(UsageRepo(db), daily_limit=10),
        ModelRouter(catalog),
        RequestRepo(db),
        catalog,
        web,  # type: ignore[arg-type]
    )
    return db, service, user


WEB_SKILL = {"net": Skill("net", "🌐", "NET-PROMPT", web=True)}


async def test_tool_loop_runs_tools_and_saves_only_the_final_answer() -> None:
    second = ToolCall("call_2", "open_url", '{"urls": ["https://a.test"]}')
    llm = FakeLLM(tool_turn(CALL), tool_turn(second), ok("Курс 90 ₽ [ЦБ](https://cbr.ru)"), ok("next"))
    web = FakeWeb()
    db, service, user = await _service(llm, web, WEB_SKILL)
    user = replace(user, skills=("net",))

    reply = await service.answer(user, "какой курс доллара")
    assert reply.status is RequestStatus.OK
    assert web.calls == [CALL, second]
    assert all(t == WEB_TOOLS for t in llm.tools[:3])
    system = llm.calls[0][0][0].content
    assert "NET-PROMPT" in system and "Сегодня" in system
    third = llm.calls[2][0]
    assert [(m.role, m.tool_call_id) for m in third[-4:]] == [
        ("assistant", None),
        ("tool", "call_1"),
        ("assistant", None),
        ("tool", "call_2"),
    ]
    assert third[-1].content == "RESULT call_2"

    await service.answer(user, "а евро?")
    history = [(m.role, m.content) for m in llm.calls[3][0][1:]]
    assert history == [
        ("user", "какой курс доллара"),
        ("assistant", "Курс 90 ₽ [ЦБ](https://cbr.ru)"),
        ("user", "а евро?"),
    ]
    await db.close()


async def test_tool_loop_stops_after_max_rounds() -> None:
    llm = FakeLLM(*[tool_turn(CALL) for _ in range(MAX_TOOL_ROUNDS + 1)])
    web = FakeWeb()
    db, service, user = await _service(llm, web, WEB_SKILL)
    reply = await service.answer(replace(user, skills=("net",)), "вопрос")
    assert reply.status is RequestStatus.FAILED and reply.error == "tool_loop"
    assert len(web.calls) == MAX_TOOL_ROUNDS
    assert llm.calls[-1][0][-1].content == TOOLS_EXHAUSTED
    await db.close()


async def test_no_tools_without_web_skill_or_client() -> None:
    llm = FakeLLM(ok("a"), ok("b"))
    db, service, user = await _service(llm, FakeWeb(), WEB_SKILL)
    await service.answer(user, "вопрос")  # skill not enabled
    db2, service2, user2 = await _service(llm, None, WEB_SKILL)
    await service2.answer(replace(user2, skills=("net",)), "вопрос")  # no SearXNG configured
    assert llm.tools == [(), ()]
    await db.close()
    await db2.close()


async def test_asking_to_search_turns_tools_on_without_the_skill() -> None:
    llm = FakeLLM(ok("a"), ok("b"))
    db, service, user = await _service(llm, FakeWeb(), WEB_SKILL)
    await service.answer(user, "Найди в интернете, когда выйдет Python 3.15")
    await service.answer(user, "объясни рекурсию")
    assert llm.tools == [WEB_TOOLS, ()]
    await db.close()


# ---- SearXNG and the page reader ---------------------------------------------------------------

PUBLIC = "93.184.216.34"


async def fake_resolve(host: str, _port: int) -> list[str]:
    return ["127.0.0.1"] if host == "local.test" else [PUBLIC]


def html(body: str, links: tuple[str, ...] = ()) -> httpx.Response:
    anchors = "".join(f'<a href="{href}">ссылка</a>' for href in links)
    page = f"<html><body><article><h1>Заголовок</h1>{body}</article>{anchors}</body></html>"
    return httpx.Response(200, html=page)


def web_tools(handler) -> WebTools:
    return WebTools("http://searx.test", transport=httpx.MockTransport(handler), resolve=fake_resolve)


async def test_search_formats_results_and_drops_duplicates() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        row = {"title": "ЦБ", "url": "https://cbr.ru", "content": "90 рублей"}
        return httpx.Response(200, json={"results": [row, row, {"title": "x"}]})

    web = web_tools(handler)
    out = await web.run(ToolCall("c", "web_search", '{"query": "курс", "topic": "news"}'))
    await web.aclose()
    assert out == "[1] ЦБ\nhttps://cbr.ru\n90 рублей"
    assert seen[0].url.path == "/search"
    assert seen[0].url.params["format"] == "json" and seen[0].url.params["categories"] == "news"


async def test_serper_is_asked_only_when_searxng_fails_or_finds_nothing() -> None:
    searx_status = [503, 200]
    serper: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "searx.test":
            return httpx.Response(searx_status.pop(0), json={"results": []})
        serper.append(request)
        row = {"title": "Новость", "link": "https://news.test/1", "snippet": "текст"}
        return httpx.Response(200, json={"news": [row], "organic": [row]})

    web = WebTools(
        "http://searx.test",
        serper_key="sk-serper",
        transport=httpx.MockTransport(handler),
        resolve=fake_resolve,
    )
    down = await web.run(ToolCall("c", "web_search", '{"query": "новости", "topic": "news"}'))
    empty = await web.run(ToolCall("c", "web_search", '{"query": "python"}'))
    await web.aclose()
    assert down == empty == "[1] Новость\nhttps://news.test/1\nтекст"
    assert [r.url.path for r in serper] == ["/news", "/search"]
    assert serper[0].headers["X-API-KEY"] == "sk-serper"
    assert json.loads(serper[0].content)["hl"] == "ru" and "hl" not in json.loads(serper[1].content)

    no_key = web_tools(lambda _r: httpx.Response(503))
    assert "недоступен" in await no_key.run(CALL)
    await no_key.aclose()


async def test_open_url_reads_pages_follows_redirects_and_blocks_private_hosts() -> None:
    text = "<p>" + "Курс доллара сегодня девяносто рублей. " * 20 + "</p>"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/old":
            return httpx.Response(301, headers={"location": "https://local.test/admin"})
        if request.url.path == "/pdf":
            return httpx.Response(200, content=b"%PDF", headers={"content-type": "application/pdf"})
        assert request.url.host != "local.test", "private host was fetched"
        return html(text)

    web = web_tools(handler)
    urls = [
        "https://a.test/",
        "https://a.test/old",
        "http://local.test/",
        "https://a.test/pdf",
        "file:///etc",
    ]
    out = await web.run(ToolCall("c", "open_url", json.dumps({"urls": urls})))
    await web.aclose()
    assert "https://a.test/\n" in out and "девяносто рублей" in out
    assert out.count("адрес недоступен") == 3  # the redirect into localhost, localhost itself, file://
    assert "это не страница (application/pdf)" in out


async def test_open_url_shares_the_budget_between_long_pages() -> None:
    web = web_tools(lambda _r: html("<p>" + "слово " * 30_000 + "</p>"))
    out = await web.run(
        ToolCall("c", "open_url", json.dumps({"urls": ["https://a.test/1", "https://a.test/2"]}))
    )
    await web.aclose()
    assert out.count("обрезан") == 2
    assert len(out) < CALL_MAX_CHARS + 500


async def test_crawl_stays_on_the_site_and_ranks_by_query() -> None:
    fetched: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        fetched.append(str(request.url))
        path = request.url.path
        if path == "/docs/":
            links = ("/docs/a", "/docs/b", "/blog/x", "https://other.test/", "/docs/logo.png", "/docs/a#top")
            return html("<p>Оглавление документации проекта.</p>", links)
        if path == "/docs/b":
            return html("<p>Тарифы: базовый стоит 500 рублей в месяц, тариф про стоит 900 рублей.</p>")
        return html(f"<p>Страница {path} про установку и настройку программы.</p>")

    web = web_tools(handler)
    args = {"url": "https://www.site.test/docs/", "query": "сколько стоит тариф", "max_pages": 3}
    out = await web.run(ToolCall("c", "crawl_site", json.dumps(args)))
    await web.aclose()
    assert "Обойдено страниц: 3" in out
    assert "other.test" not in "".join(fetched) and not any(u.endswith(".png") for u in fetched)
    assert "https://www.site.test/blog/x" not in fetched  # the /docs/ folder goes first
    assert "900 рублей" in out


def test_rank_chunks_keeps_the_best_within_budget() -> None:
    pages = [
        Page("https://a", "Про погоду и дождь.\n\n" + "вода " * 400),
        Page("https://b", "Курс доллара ЦБ на сегодня 90 рублей."),
    ]
    out = rank_chunks("курс доллара", pages, budget=100)
    assert out == "[Источник: https://b]\nКурс доллара ЦБ на сегодня 90 рублей."


async def test_tool_errors_become_text_for_the_model() -> None:
    web = web_tools(lambda _r: httpx.Response(503))
    assert "недоступен" in await web.run(CALL)
    assert "неверные аргументы" in await web.run(ToolCall("c", "web_search", "not json"))
    assert "неверные аргументы" in await web.run(ToolCall("c", "web_search", '{"topic": "news"}'))
    assert "неверные аргументы" in await web.run(ToolCall("c", "crawl_site", '{"url": "https://a.test"}'))
    assert "неизвестный инструмент" in await web.run(ToolCall("c", "rm_rf", "{}"))
    assert "ответ 503" in await web.run(ToolCall("c", "open_url", '{"urls": ["https://a.test"]}'))
    await web.aclose()


# ---- deep research ---------------------------------------------------------------------------------


async def test_research_plans_queries_and_puts_material_in_front_of_the_model() -> None:
    plan = ok('1. курс рубля 2026\n- "ruble exchange rate forecast"\n\nкурс рубля 2026\nx')
    llm = FakeLLM(plan, ok("Развёрнутый ответ"), ok("next"))
    web = FakeWeb()
    db, service, user = await _service(llm, web, WEB_SKILL)
    question = "Исследуй, что будет с курсом рубля"
    assert service.is_research(question) and not service.is_research("что будет с курсом рубля")

    reply = await service.answer(user, question)  # no web skill: research turns the tools on itself
    assert reply.status is RequestStatus.OK
    assert llm.calls[0][0][0].content == RESEARCH_PLANNER and llm.calls[0][1] != llm.calls[1][1]
    assert web.researched == [(question, ["курс рубля 2026", "ruble exchange rate forecast"])]
    main = llm.calls[1][0][-1].content
    assert main.startswith(question) and "<research>" in main and "МАТЕРИАЛ" in main
    assert llm.tools[1] == WEB_TOOLS

    await service.answer(user, "а доллар?")
    assert [m.content for m in llm.calls[2][0][1:]] == [question, "Развёрнутый ответ", "а доллар?"]
    await db.close()


def test_parse_queries_strips_list_markup() -> None:
    assert parse_queries("1) первый запрос\n• «второй»\n  \n-  third query \nab") == [
        "первый запрос",
        "второй",
        "third query",
    ]


async def test_research_reads_top_results_of_every_query_in_turn() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "searx.test":
            q = request.url.params["q"]
            if q == "сломан":
                return httpx.Response(500)
            rows = [{"title": f"{q}{i}", "url": f"https://{q}.test/{i}", "content": ""} for i in range(10)]
            shared = {"url": "https://shared.test/"}
            return httpx.Response(200, json={"results": [*rows[:2], shared, *rows[2:]]})
        return html(f"<p>Страница {request.url} про курс валют и прогнозы аналитиков.</p>")

    web = web_tools(handler)
    material, calls = await web.research("курс", ["a", "b", "сломан"])
    await web.aclose()
    assert calls == 3 + 19  # 3 queries, then every unique url of the top 10 of a and b (shared once)
    read = [line[11:-1] for line in material.splitlines() if line.startswith("[Источник: ")]
    # a0 b0 a1 b1 shared a2 b2 a3 b3 a4: in turn, the shared page once, the broken query skipped
    first = [
        "https://a.test/0",
        "https://b.test/0",
        "https://a.test/1",
        "https://b.test/1",
        "https://shared.test/",
    ]
    assert read[:5] == first
    assert len(read) == 19 and material.startswith("Прочитано страниц: 19 из 19.")
