from datetime import UTC, datetime

from bot.domain import PERSONA_MAX_CHARS, Mode, Role, Style, User, approx_tokens
from bot.services.conversation import (
    PERSONA_HEADER,
    STYLE_INSTRUCTIONS,
    ConversationService,
    compose_system_prompt,
)
from bot.storage.db import Database
from bot.storage.repositories import ConversationRepo, UserRepo
from bot.telegram.keyboards import settings_keyboard
from bot.telegram.routers.settings import PERSONA_PREVIEW_CHARS, _settings_text


def _user(**kw) -> User:
    return User(1, "u", Role.USER, Mode.NORMAL, None, datetime.now(UTC), **kw)


def test_system_prompt_keeps_bot_rules_first() -> None:
    assert compose_system_prompt("BASE", None, Style.NORMAL) == "BASE"
    prompt = compose_system_prompt("BASE", "Ты пират", Style.SHORT)
    assert prompt.startswith("BASE")
    assert prompt.index(STYLE_INSTRUCTIONS[Style.SHORT]) < prompt.index(PERSONA_HEADER)
    assert prompt.endswith("Ты пират")


async def test_persona_and_style_are_stored_and_counted_in_budget() -> None:
    db = Database(":memory:")
    await db.connect()
    users = UserRepo(db)
    await users.upsert_seen(1, "u", default_role=Role.USER, default_mode=Mode.NORMAL)
    fresh = await users.get(1)
    assert (fresh.persona, fresh.style) == (None, Style.NORMAL)

    await users.set_persona(1, "Ты пират")
    await users.set_style(1, Style.DETAILED)
    stored = await users.get(1)
    assert (stored.persona, stored.style) == ("Ты пират", Style.DETAILED)

    repo = ConversationRepo(db)
    budget = 2_500
    service = ConversationService(repo, system_prompt="sys", token_budget=budget)
    conv = await repo.active_id(1)
    for i in range(10):  # 200 tokens per message: the whole history (2000) fits without a persona
        await repo.add_message(conv, "user" if i % 2 == 0 else "assistant", "x" * 700, 200)

    _, plain = await service.build_context(1, "q")
    persona = "p" * PERSONA_MAX_CHARS  # ~570 tokens pushes the oldest history out of the budget
    _, personal = await service.build_context(1, "q", persona=persona, style=Style.SHORT)
    assert len(plain) == 12
    assert personal[0].content.endswith(persona)
    assert len(personal) < len(plain)
    assert sum(approx_tokens(m.content) for m in personal) <= budget
    await db.close()


def test_settings_panel_escapes_and_truncates_persona() -> None:
    text = _settings_text(_user(persona="<b>злой</b> & " + "x" * 1000))
    assert "<b>злой</b>" not in text and "&lt;b&gt;злой&lt;/b&gt; &amp;" in text
    assert text.count("x") <= PERSONA_PREVIEW_CHARS
    assert "не задана" in _settings_text(_user())


def test_settings_keyboard_values_are_valid_styles() -> None:
    kb = settings_keyboard(Style.SHORT, has_persona=True)
    data = [b.callback_data for row in kb.inline_keyboard for b in row]
    styles = [d.removeprefix("set:style:") for d in data if d.startswith("set:style:")]
    assert [Style(s) for s in styles] == list(Style)
    assert "set:persona_reset" in data
    assert "set:persona_reset" not in [
        b.callback_data
        for row in settings_keyboard(Style.SHORT, has_persona=False).inline_keyboard
        for b in row
    ]
    assert any(b.text.startswith("✅") and "Кратко" in b.text for row in kb.inline_keyboard for b in row)
