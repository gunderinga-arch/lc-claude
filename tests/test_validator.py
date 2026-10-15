import pytest

from bot.llm.base import Verdict
from bot.llm.validator import CUT_FRAGMENTS, STUB_SIGNATURES, classify


@pytest.mark.parametrize("text", ["", "   ", "\n\t \r\n"])
def test_empty(text: str) -> None:
    assert classify(text) is Verdict.EMPTY


def test_required_cut_fragments_present() -> None:
    assert {"Let", "##", "#", "Looking", "**", "```"} <= CUT_FRAGMENTS


@pytest.mark.parametrize("fragment", sorted(CUT_FRAGMENTS))
def test_each_cut_fragment(fragment: str) -> None:
    assert classify(fragment) is Verdict.CUT
    assert classify(f"  {fragment}\n") is Verdict.CUT


@pytest.mark.parametrize("text", ["---", "> ", "***", "# #", "~~", "__", "-\n-\n"])
def test_markdown_punctuation_only_is_cut(text: str) -> None:
    assert classify(text) is Verdict.CUT


@pytest.mark.parametrize(
    "text", ["Все картинки готовы. Собираю лендинг:", "Готово — текстура тоже есть. Собираю лендинг:\n"]
)
def test_lead_in_without_content_is_cut(text: str) -> None:
    assert classify(text) is Verdict.CUT


def test_long_answer_ending_in_colon_is_ok() -> None:
    assert classify("Разбор по пунктам. " * 40 + "Итог:") is Verdict.OK


@pytest.mark.parametrize("signature", STUB_SIGNATURES)
def test_stub_inside_longer_text(signature: str) -> None:
    assert classify(f"Well... {signature}! Let's talk about something else.") is Verdict.STUB


def test_stub_is_case_sensitive() -> None:
    assert classify("oh, that island is lovely") is Verdict.OK


@pytest.mark.parametrize(
    "text",
    [
        "42",
        "Да",
        "No.",
        "-1",
        "Let's go",
        "## Heading\n\nBody",
        "Looking at the data, the answer is 7.",
        "Первый абзац.\n\nВторой абзац с **жирным** текстом.",
    ],
)
def test_ok(text: str) -> None:
    assert classify(text) is Verdict.OK


@pytest.mark.parametrize(
    "text",
    [
        "Я не хочу обсуждать политические темы.",
        "  я не хочу обсуждать политические темы  ",
        "Я не хочу обсуждать политические темы!",
    ],
)
def test_political_refusal_is_stub(text: str) -> None:
    assert classify(text) is Verdict.STUB


def test_answer_quoting_the_refusal_stays_ok() -> None:
    text = "Бот может ответить «Я не хочу обсуждать политические темы.», если в истории есть политика."
    assert classify(text) is Verdict.OK
