from types import SimpleNamespace

import pytest
from aiogram.types import FSInputFile

from bot.telegram import panel
from bot.telegram.routers import commands


class FakeBot:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.sent: list[tuple] = []  # (kind, media, text)
        self.next_id = 100

    def _msg(self) -> SimpleNamespace:
        return SimpleNamespace(message_id=self._next(), animation=None, photo=None)

    async def send_animation(self, chat_id, animation, caption=None, reply_markup=None):
        if self.fail:
            raise RuntimeError("Bad Request: wrong file identifier")
        self.sent.append(("animation", animation, caption))
        return SimpleNamespace(message_id=self._next(), animation=SimpleNamespace(file_id="anim"), photo=None)

    async def send_photo(self, chat_id, photo, caption=None, reply_markup=None):
        self.sent.append(("photo", photo, caption))
        pic = [SimpleNamespace(file_id="pic")]
        return SimpleNamespace(message_id=self._next(), animation=None, photo=pic)

    async def send_message(self, chat_id, text, reply_markup=None):
        self.sent.append(("text", None, text))
        return self._msg()

    async def delete_message(self, chat_id, message_id):
        pass

    def _next(self) -> int:
        self.next_id += 1
        return self.next_id


@pytest.fixture(autouse=True)
def _fresh_cache():
    panel._file_ids.clear()
    panel._panels.clear()
    yield
    panel._file_ids.clear()
    panel._panels.clear()


async def test_start_is_one_animation_with_the_welcome_caption(tmp_path, monkeypatch) -> None:
    animation = tmp_path / "welcome.mp4"
    animation.write_bytes(b"test fixture: FakeBot does not decode media")
    monkeypatch.setattr(commands, "START_ANIMATION", animation)
    bot = FakeBot()
    first = await commands.send_start(bot, 1, user=None)
    kind, media, caption = bot.sent[0]
    assert (kind, caption, first.message_id) == ("animation", commands.START_TEXT, 101)
    assert isinstance(media, FSInputFile) and len(bot.sent) == 1
    await commands.send_start(bot, 1, user=None)
    assert bot.sent[1][1] == "anim"  # uploaded once, then reused by file id


async def test_start_without_animation_still_greets(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(commands, "START_ANIMATION", tmp_path / "missing.mp4")
    bot = FakeBot()
    await commands.send_start(bot, 1, user=None)
    assert bot.sent == [("text", None, commands.START_TEXT)]
    animation = tmp_path / "welcome.mp4"
    animation.write_bytes(b"test fixture: FakeBot does not decode media")
    monkeypatch.setattr(commands, "START_ANIMATION", animation)
    bot = FakeBot(fail=True)
    await commands.send_start(bot, 1, user=None)
    assert bot.sent == [("text", None, commands.START_TEXT)]


async def test_model_menu_is_the_roster_photo() -> None:
    bot = FakeBot()
    await panel.show_panel(bot, 1, commands.MODEL_MENU_TEXT, None, photo=commands.ROSTER)
    await panel.show_panel(bot, 1, "help", None)
    await panel.show_panel(bot, 1, commands.MODEL_MENU_TEXT, None, photo=commands.ROSTER)
    kinds = [(k, m if m is None or isinstance(m, str) else "file") for k, m, _ in bot.sent]
    assert kinds == [("animation", "file"), ("text", None), ("animation", "anim")]
    assert len(commands.MODEL_MENU_TEXT) <= 1024  # Telegram caption limit


def test_buttons_accept_pre_redesign_labels():
    """Chats opened before the redesign keep the old keyboard: its presses must still work."""
    from bot.telegram import keyboards as kb

    for label, legacy in kb.LEGACY_LABELS.items():
        assert label != legacy
        f = kb.pressed(label)
        assert f.resolve(type("M", (), {"text": label})()) and f.resolve(type("M", (), {"text": legacy})())
        assert not f.resolve(type("M", (), {"text": "привет"})())


def test_clear_ignores_a_floor_from_the_old_bots_chat():
    assert commands.clear_from(40, 120) == 40
    assert commands.clear_from(None, 120) == 120 - commands.UNKNOWN_FLOOR_SPAN
    assert commands.clear_from(1306, 120) == 120 - commands.UNKNOWN_FLOOR_SPAN  # old chat's id


async def test_start_with_topics_explains_chats_and_fits_a_caption(monkeypatch) -> None:
    async def on(_bot):
        return True

    monkeypatch.setattr(commands, "topics_enabled", on)
    bot = FakeBot()
    await commands.send_start(bot, 1, user=None)
    caption = bot.sent[0][2]
    assert caption == commands.START_TEXT + commands.START_CHATS_TEXT
    assert len(caption) <= 1024  # Telegram caption limit
