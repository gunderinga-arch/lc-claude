"""A short button survey before a new presentation: audience, look, length, pictures.

The answers are appended to the user's request as wishes, so the model builds the deck the user had in mind
instead of guessing. Pure logic here; the handlers live in routers/chat.py.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from aiogram.types import InlineKeyboardMarkup, Message
from aiogram.utils.keyboard import InlineKeyboardBuilder

SKIP = "skip"
TTL_S = 30 * 60  # an unanswered survey is forgotten after this long
DETAILED_CHARS = 250  # a request this long already says what the user wants: no survey

# A new deck is asked for: a "make" verb near "презентация/слайды". ponytail: a regex, not a classifier;
# edits of an existing deck ("убери слайд 3", "переделай презентацию") and long briefs skip the survey.
_MAKE = re.compile(
    r"(сдела|созда|подготов|собер|сгенер|нужн|хочу|напиш|оформ)\w*[^.!?\n]{0,60}"
    r"(презентаци|слайд|pptx|powerpoint|пауэрпоинт)",
    re.I,
)
_EDIT = re.compile(
    r"(передела|исправ|поменя|измени|убер|удали|добав|покороче|подлинне|слайд\w*\s*№?\s*\d)", re.I
)


@dataclass(frozen=True, slots=True)
class Step:
    key: str
    question: str
    options: tuple[tuple[str, str], ...]  # (button label, wish text for the model)


STEPS = (
    Step(
        "who",
        "Для кого презентация?",
        (
            ("💼 Работа, бизнес", "аудитория — коллеги или руководство, деловой тон"),
            ("🎓 Учёба", "аудитория — преподаватель и одногруппники, учебный доклад"),
            ("🤝 Клиенты", "аудитория — клиенты, нужно убедить и продать"),
            ("🎉 Для себя, друзей", "аудитория — друзья, лёгкий живой тон"),
        ),
    ),
    Step(
        "look",
        "Какой стиль?",
        (
            ("⚪ Светлый строгий", 'тема оформления "light"'),
            ("⚫ Тёмный", 'тема оформления "dark"'),
            ("🟠 Тёплый", 'тема оформления "warm"'),
            ("🟢 Природный", 'тема оформления "forest"'),
            ("🎲 На твой вкус", "тему оформления выбери сам под содержание"),
        ),
    ),
    Step(
        "size",
        "Сколько слайдов?",
        (
            ("5–7", "5–7 слайдов"),
            ("8–12", "8–12 слайдов"),
            ("13–20", "13–20 слайдов"),
        ),
    ),
    Step(
        "pics",
        "Картинки на слайдах?",
        (
            ("🎨 Нарисовать", "нарисуй 1–3 картинки для слайдов image"),
            ("Без картинок", "без картинок"),
        ),
    ),
)


@dataclass(slots=True)
class Survey:
    request: Message  # the user's own message: the job answers it, in its chat or new topic
    steps: tuple[Step, ...]
    answers: list[str] = field(default_factory=list)
    started: float = field(default_factory=time.monotonic)

    @property
    def step(self) -> Step | None:
        return self.steps[len(self.answers)] if len(self.answers) < len(self.steps) else None


# (user id, topic) -> the survey in progress. ponytail: in memory, a restart forgets it (the user asks again).
_surveys: dict[tuple[int, int], Survey] = {}


def wants_survey(text: str) -> bool:
    return len(text) < DETAILED_CHARS and bool(_MAKE.search(text)) and not _EDIT.search(text)


def start(user_id: int, thread_id: int, request: Message, can_draw: bool) -> Survey:
    now = time.monotonic()
    for key in [k for k, s in _surveys.items() if now - s.started > TTL_S]:
        del _surveys[key]
    steps = STEPS if can_draw else tuple(s for s in STEPS if s.key != "pics")
    survey = _surveys[(user_id, thread_id)] = Survey(request, steps)
    return survey


def answer(user_id: int, thread_id: int, data: str) -> tuple[Survey | None, bool]:
    """Apply a pressed button ("deck:<step>:<n>" or "deck:skip"): (survey, finished); None if it is gone."""
    survey = _surveys.get((user_id, thread_id))
    if survey is None or survey.step is None:
        return None, False
    _, key, *rest = data.split(":")
    if key == SKIP:
        del _surveys[(user_id, thread_id)]
        return survey, True
    step = survey.step
    if key != step.key or not rest or not rest[0].isdigit() or int(rest[0]) >= len(step.options):
        return survey, False  # a stale button of an earlier step: ignore
    survey.answers.append(step.options[int(rest[0])][1])
    if survey.step is None:
        del _surveys[(user_id, thread_id)]
        return survey, True
    return survey, False


def keyboard(step: Step) -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    for n, (label, _) in enumerate(step.options):
        builder.button(text=label, callback_data=f"deck:{step.key}:{n}")
    builder.button(text="Пропустить — делай так", callback_data=f"deck:{SKIP}")
    builder.adjust(2)
    return builder.as_markup()


def question(survey: Survey) -> str:
    step = survey.step
    assert step is not None
    n = len(survey.answers) + 1
    return f"📊 <b>Презентация</b> · вопрос {n} из {len(survey.steps)}\n\n{step.question}"


def final_request(survey: Survey) -> str:
    """The user's request plus the survey answers, as the model will read it."""
    text = survey.request.text or ""
    if not survey.answers:
        return text
    return f"{text}\n\nПожелания к презентации (ответы на короткий опрос): " + "; ".join(survey.answers) + "."
