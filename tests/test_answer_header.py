from types import SimpleNamespace

from bot.catalog import load_catalog
from bot.config import DEFAULT_MODELS_FILE
from bot.domain import RequestStatus
from bot.services.chat import HEADER_RESERVE, ChatReply
from bot.telegram.keyboards import REPLY_KEYBOARD
from bot.telegram.routers.chat import CONTEXT_RESET_TEXT, _deliver

CATALOG = load_catalog(DEFAULT_MODELS_FILE)


class FakeMessage:
    def __init__(self) -> None:
        self.sent: list[str] = []

    async def answer(self, text: str, **kwargs) -> SimpleNamespace:
        self.sent.append(text)
        return SimpleNamespace()


def _reply(model_id: str, parts: list[str], **kw) -> ChatReply:
    return ChatReply(
        RequestStatus.OK, parts, "raw", CATALOG.get(model_id), kw.pop("fallback_used", False), **kw
    )


async def test_first_message_starts_with_bold_model_name() -> None:
    msg = FakeMessage()
    await _deliver(msg, _reply("gpt-5.6-luna", ["часть 1", "часть 2"]), REPLY_KEYBOARD)
    assert msg.sent == ["<b>GPT-Luna</b> 🌒\n\nчасть 1", "часть 2"]


async def test_fallback_and_context_reset_notes_follow_the_name() -> None:
    msg = FakeMessage()
    reply = _reply("gpt-5.6-sol", ["ответ"], fallback_used=True, context_reset=True)
    await _deliver(msg, reply, REPLY_KEYBOARD)
    first = msg.sent[0]
    assert first.startswith("<b>GPT-Sol</b> ☀️ <i>(запасная модель)</i>\n")
    assert CONTEXT_RESET_TEXT in first and first.endswith("\n\nответ")


def test_longest_header_fits_the_reserve() -> None:
    longest = max(len(f"<b>{m.title}</b> 🌒") for m in CATALOG.selectable()) + len(
        " <i>(запасная модель)</i>\n"
    )
    assert longest + len(f"<i>{CONTEXT_RESET_TEXT}</i>\n\n") < HEADER_RESERVE


async def test_a_very_long_answer_comes_as_its_start_and_a_page() -> None:
    from bot.telegram.routers.chat import LONG_ANSWER_FILE, MAX_ANSWER_MESSAGES

    class WithDocs(FakeMessage):
        def __init__(self) -> None:
            super().__init__()
            self.docs: list = []

        async def answer_document(self, document, **kwargs) -> SimpleNamespace:
            self.docs.append(document)
            return SimpleNamespace()

    msg = WithDocs()
    from dataclasses import replace

    from bot.render.telegram_html import markdown_to_html, render_markdown

    text = (("## Заголовок\nСтрока с <>& и пробелами.\n" + "слово " * 1000 + "\n\n") * 20).strip()
    parts = render_markdown(text, limit=3796)
    assert len(parts) > MAX_ANSWER_MESSAGES
    await _deliver(msg, replace(_reply("claude-opus-5", parts), text=text), REPLY_KEYBOARD)
    assert len(msg.sent) == 1 and "Заголовок" in msg.sent[0] and "целиком" in msg.sent[0]
    assert len(msg.sent[0]) <= 4096
    assert f"{len(text):,}".replace(",", " ") in msg.sent[0]
    (page,) = msg.docs
    assert page.filename == LONG_ANSWER_FILE
    html = page.data.decode("utf-8")
    assert html.startswith("<!doctype html>") and markdown_to_html(text) in html

    from bot.llm.resilience import CUT_NOTE

    msg = WithDocs()
    text += CUT_NOTE
    await _deliver(msg, replace(_reply("claude-opus-5", render_markdown(text)), text=text), REPLY_KEYBOARD)
    assert "Ответ оборвался" in msg.sent[0] and "продолжи" in msg.sent[0]
    assert "целиком" not in msg.sent[0]

    msg = WithDocs()
    text = "```json\n" + '"value": "' + "a" * 20_000 + '"\n```'
    await _deliver(msg, replace(_reply("claude-opus-5", render_markdown(text)), text=text), REPLY_KEYBOARD)
    assert len(msg.docs) == 1 and all(len(part) <= 4096 for part in msg.sent)
    assert "a" * 20_000 in msg.docs[0].data.decode("utf-8")


async def test_a_flood_wait_is_waited_out_once(monkeypatch) -> None:
    import asyncio

    from aiogram.exceptions import TelegramRetryAfter
    from aiogram.methods import SendMessage

    slept: list[float] = []

    async def fake_sleep(s: float) -> None:
        slept.append(s)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    class Flooded(FakeMessage):
        tries = 0

        async def answer(self, text: str, **kwargs) -> SimpleNamespace:
            self.tries += 1
            if self.tries == 1:
                raise TelegramRetryAfter(SendMessage(chat_id=1, text="x"), "flood", retry_after=3)
            return await super().answer(text, **kwargs)

    msg = Flooded()
    await _deliver(msg, _reply("claude-opus-5", ["ответ"]), REPLY_KEYBOARD)
    assert slept == [4] and msg.sent and msg.sent[0].endswith("ответ")
