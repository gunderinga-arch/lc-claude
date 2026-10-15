from datetime import UTC, datetime
from pathlib import Path

import pytest

from bot.config import DEFAULT_SKILLS_FILE
from bot.domain import Mode, Role, Style, User
from bot.services.conversation import PERSONA_HEADER, ConversationService, compose_system_prompt
from bot.skills import Skill, SkillsError, load_skills, toggle_skill
from bot.storage.db import Database
from bot.storage.repositories import ConversationRepo, UserRepo
from bot.telegram.keyboards import settings_keyboard, skills_keyboard
from bot.telegram.routers.settings import _settings_text

SKILLS = {
    "a": Skill("a", "🅰 Альфа", "PROMPT-A"),
    "b": Skill("b", "🅱 Бета", "PROMPT-B"),
}


def test_shipped_skills_file_is_valid_and_fits_callback_data() -> None:
    skills = load_skills(DEFAULT_SKILLS_FILE)
    assert skills
    kb = skills_keyboard(list(skills.values()), ())
    assert all(len(b.callback_data.encode()) <= 64 for row in kb.inline_keyboard for b in row)


def test_load_skills_rejects_bad_files(tmp_path: Path) -> None:
    assert load_skills(tmp_path / "missing.yaml") == {}
    bad = tmp_path / "skills.yaml"
    for body in (
        "skills:\n  'Bad Id':\n    title: t\n    prompt: p\n",
        "skills:\n  ok:\n    prompt: p\n",
        "skills:\n  ok:\n    title: t\n",
    ):
        bad.write_text(body, encoding="utf-8")
        with pytest.raises(SkillsError):
            load_skills(bad)


def test_toggle_skill_keeps_order_and_drops_stale_ids() -> None:
    assert toggle_skill((), "a", SKILLS) == ("a",)
    assert toggle_skill(("a",), "b", SKILLS) == ("a", "b")
    assert toggle_skill(("a", "b"), "a", SKILLS) == ("b",)
    assert toggle_skill(("gone", "a"), "b", SKILLS) == ("a", "b")


def test_skills_go_between_style_and_persona() -> None:
    prompt = compose_system_prompt("BASE", "Ты пират", Style.SHORT, ["PROMPT-A"])
    assert prompt.startswith("BASE")
    assert prompt.index("PROMPT-A") < prompt.index(PERSONA_HEADER)


async def test_skills_are_stored_and_reach_the_system_prompt() -> None:
    db = Database(":memory:")
    await db.connect()
    users = UserRepo(db)
    await users.upsert_seen(1, "u", default_role=Role.USER, default_mode=Mode.NORMAL)
    assert (await users.get(1)).skills == ()

    await users.set_skills(1, ("b", "gone"))
    assert (await users.get(1)).skills == ("b", "gone")

    repo = ConversationRepo(db)
    service = ConversationService(repo, system_prompt="sys", token_budget=10_000, skills=SKILLS)
    _, messages = await service.build_context(1, "q", skills=("b", "gone"))
    assert messages[0].content == "sys\n\nPROMPT-B"
    await db.close()


def test_settings_panel_lists_enabled_skills() -> None:
    user = User(1, "u", Role.USER, Mode.NORMAL, None, datetime.now(UTC), skills=("b",))
    assert "🅱 Бета" in _settings_text(user, SKILLS)
    assert "Скиллы" not in _settings_text(user, {})  # no skills file: nothing to mention

    def data(kb):
        return [b.callback_data for row in kb.inline_keyboard for b in row]

    assert "set:skills" in data(settings_keyboard(Style.NORMAL, False, skills_available=True))
    assert "set:skills" not in data(settings_keyboard(Style.NORMAL, False))
    texts = [b.text for row in skills_keyboard(list(SKILLS.values()), ("b",)).inline_keyboard for b in row]
    assert texts[:2] == ["▫️ 🅰 Альфа", "✅ 🅱 Бета"]


def test_skills_panel_shows_state_counter_and_off_button() -> None:
    from bot.telegram.routers.settings import _skills_text

    text = _skills_text(SKILLS, ("b",))
    assert "▫️ <b>🅰 Альфа</b>" in text and "✅ <b>🅱 Бета</b>" in text
    assert "Включено: 1 из 2" in text
    assert len(_skills_text(load_skills(DEFAULT_SKILLS_FILE), ())) < 4096  # Telegram message limit

    def data(enabled):
        return [
            b.callback_data
            for row in skills_keyboard(list(SKILLS.values()), enabled).inline_keyboard
            for b in row
        ]

    assert "set:skills_off" in data(("a",))
    assert "set:skills_off" not in data(())
