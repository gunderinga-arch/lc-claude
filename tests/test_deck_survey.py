from types import SimpleNamespace

import pytest

from bot.telegram import deck_survey as ds


@pytest.fixture(autouse=True)
def _clean():
    ds._surveys.clear()
    yield
    ds._surveys.clear()


@pytest.mark.parametrize(
    "text",
    [
        "Сделай презентацию про мамонтов",
        "нужна презентация по истории Томска",
        "хочу слайды для защиты диплома",
    ],
)
def test_a_new_deck_asks_first(text):
    assert ds.wants_survey(text)


@pytest.mark.parametrize(
    "text",
    [
        "что такое презентация",  # no request to make one
        "переделай презентацию в тёмную тему",  # an edit of an existing deck
        "сделай слайд 3 покороче",
        "Сделай презентацию: " + "очень подробное ТЗ, " * 20,  # a detailed brief needs no survey
    ],
)
def test_questions_and_edits_go_straight_to_the_model(text):
    assert not ds.wants_survey(text)


def test_answers_become_wishes_in_the_request():
    request = SimpleNamespace(text="Сделай презентацию про кофе")
    survey = ds.start(1, 0, request, can_draw=True)
    assert [s.key for s in survey.steps] == ["who", "look", "size", "pics"]
    assert ds.answer(1, 0, "deck:who:1") == (survey, False)
    assert ds.answer(1, 0, "deck:who:0") == (survey, False)  # a stale button of a past step is ignored
    ds.answer(1, 0, "deck:look:1")
    ds.answer(1, 0, "deck:size:0")
    assert ds.answer(1, 0, "deck:pics:1") == (survey, True)
    final = ds.final_request(survey)
    assert final.startswith("Сделай презентацию про кофе\n\nПожелания к презентации")
    assert (
        "учебный доклад" in final and '"dark"' in final and "5–7 слайдов" in final and "без картинок" in final
    )
    assert ds.answer(1, 0, "deck:who:0") == (None, False)  # finished surveys are gone


def test_skip_sends_what_was_answered_and_no_pictures_step_without_drawing():
    survey = ds.start(2, 7, SimpleNamespace(text="сделай презентацию"), can_draw=False)
    assert "pics" not in [s.key for s in survey.steps]
    ds.answer(2, 7, "deck:who:2")
    assert ds.answer(2, 7, "deck:skip") == (survey, True)
    assert ds.final_request(survey).endswith("убедить и продать.")
    assert "вопрос 1 из 3" in ds.question(ds.start(2, 7, SimpleNamespace(text="x"), can_draw=False))
