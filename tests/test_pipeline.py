"""Pipeline "models": the bot plans the answer and picks skills with the model before it answers."""

from __future__ import annotations

from dataclasses import replace

import pytest

from bot.catalog import Catalog, CatalogError, load_catalog
from bot.config import DEFAULT_MODELS_FILE
from bot.domain import Mode, RequestStatus
from bot.llm.base import ChatResult, ChatStatus
from bot.services.chat import PLANNER, parse_plan
from bot.skills import Skill
from tests.test_chat_service import FakeLLM, ok
from tests.test_resilience import spec
from tests.test_web import _service

SKILLS = {
    "coder": Skill("coder", "💻 Код", "CODER-PROMPT", description="пишет код"),
    "polish": Skill("polish", "✨", "REWRITER", rewrite=True),
}


async def test_pipeline_plans_then_answers_with_the_picked_skills() -> None:
    plan = ok("SKILLS: coder, polish, bogus\nPLAN:\n1. Разобрать задачу\n2. Написать код", "gpt-6-sol")
    llm = FakeLLM(plan, ok("Готовый код", "gpt-6-sol"), ok("next"))
    db, service, user = await _service(llm, None, SKILLS)
    user = replace(user, model_override="plan-opus")

    reply = await service.answer(user, "Сделай бота")
    assert reply.status is RequestStatus.OK and reply.model.id == "plan-opus"
    (planner, planner_model), (main, main_model) = llm.calls[0], llm.calls[1]
    assert planner_model == "gpt-6-sol" and main_model == "gpt-6-sol"
    assert planner[0].content.startswith(PLANNER.split("\n")[0])
    assert "- coder: 💻 Код — пишет код" in planner[0].content and "polish" not in planner[0].content
    assert planner[-1].content.endswith("Сделай бота")
    assert "CODER-PROMPT" in main[0].content  # the picked skill, only for this answer
    assert main[-1].content.startswith("Сделай бота") and "2. Написать код" in main[-1].content
    assert "SKILLS" not in main[-1].content

    await service.answer(replace(user, model_override=None), "а тесты?")
    assert "CODER-PROMPT" not in llm.calls[2][0][0].content
    assert [m.content for m in llm.calls[2][0][1:]] == ["Сделай бота", "Готовый код", "а тесты?"]
    await db.close()


async def test_pipeline_answers_without_a_plan_when_the_planner_fails() -> None:
    failed = ChatResult(ChatStatus.FAILED, "", "gpt-6-astra", 3, 10, False, error="timeout")
    llm = FakeLLM(failed, ok("ответ", "gpt-6-sol", fallback=True))
    db, service, user = await _service(llm, None, SKILLS)
    reply = await service.answer(replace(user, model_override="plan-opus"), "вопрос")
    assert reply.status is RequestStatus.OK
    assert reply.model.id == "gpt-6-sol" and reply.fallback_used  # a fallback shows the real model
    assert "<plan>" not in llm.calls[1][0][-1].content
    await db.close()


def test_parse_plan() -> None:
    assert parse_plan("SKILLS: -\nPLAN:\nшаг", {"a"}) == ("шаг", [])
    assert parse_plan("PLAN: сразу план", {"a"}) == ("сразу план", [])
    assert parse_plan("вступление\nSKILLS: a; b a\nPLAN:\nx", {"a", "b"}) == ("вступление\n\nx", ["a", "b"])


def test_pipeline_in_the_catalog() -> None:
    catalog = load_catalog(DEFAULT_MODELS_FILE)
    assert (
        catalog.get("plan-opus").pipeline == "gpt-6-sol"
        and catalog.get("plan-opus").answer == "gpt-6-sol"
        and not catalog.get("plan-opus").probe
    )
    catalog.mark_unavailable("gpt-6-sol")
    assert not catalog.is_available("plan-opus")  # no worker, no pipeline in the menu

    def build(extra) -> Catalog:
        models = {"p": spec("p"), extra.id: extra}
        return Catalog(models, dict.fromkeys(Mode, "p"), max_tokens_cap=10**6, max_attempts=1)

    assert build(replace(spec("x"), pipeline="p", probe=False)).get("x").pipeline == "p"
    with pytest.raises(CatalogError, match="not probed"):
        build(replace(spec("x"), pipeline="p"))  # probed by default: pings would hit the gateway
    with pytest.raises(CatalogError, match="plain gateway model"):
        build(replace(spec("x"), pipeline="x", probe=False))
