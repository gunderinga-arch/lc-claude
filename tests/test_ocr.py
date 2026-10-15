"""Scanned PDFs: read from their pages as pictures by the vision model, in batches."""

import pytest

from bot.files import ocr
from bot.files.extract import ScannedPdf, UnsupportedFile, extract_text
from bot.llm.base import ChatResult, ChatStatus
from bot.services.chat import ChatService
from bot.telegram.routers.chat import _collect
from tests.test_chat_service import FakeLLM, env, make_service, ok  # noqa: F401  (env is a fixture)
from tests.test_extract import tiny_pdf
from tests.test_files_flow import FakeBot, _doc


def test_a_pdf_without_text_is_a_scan_and_still_unsupported_for_old_callers() -> None:
    with pytest.raises(ScannedPdf):
        extract_text("scan.pdf", tiny_pdf(""))
    assert issubclass(ScannedPdf, UnsupportedFile)


def test_batches_cover_the_pages_up_to_the_cap_and_the_estimate_is_fair() -> None:
    assert ocr.batches(14) == [(1, 6), (7, 12), (13, 14)]
    capped = ocr.batches(1000)
    assert capped[-1][1] == ocr.OCR_MAX_PAGES and len(capped) == -(
        -ocr.OCR_MAX_PAGES // ocr.OCR_PAGES_PER_CALL
    )
    assert ocr.minutes_for(1) == 1 and ocr.minutes_for(150) == ocr.minutes_for(10_000)


async def test_read_scan_sends_pictures_in_batches_and_joins_the_text(env, monkeypatch) -> None:  # noqa: F811
    asked: list[tuple[int, int]] = []

    async def render(pdf, first, last):
        asked.append((first, last))
        return [b"jpeg"] * (last - first + 1)

    monkeypatch.setattr("bot.services.chat.render_pages", render)
    monkeypatch.setattr("bot.services.chat.can_read_scans", lambda: True)
    monkeypatch.setattr("bot.services.chat.page_count", lambda data: 8)
    blind = ChatResult(
        ChatStatus.OK, "Image could not be recognized: the vision service did not respond", "f", 1, 1, False
    )
    llm = FakeLLM(ok("--- страница 1 ---\nПервая"), blind)
    service: ChatService = make_service(env, llm)
    text = await service.read_scan(b"%PDF", "скан.pdf")
    assert sorted(asked) == [(1, 6), (7, 8)]
    assert all(model == "claude-fable-5.1" for _, model in llm.calls)
    assert len(llm.calls[0][0][0].images) + len(llm.calls[1][0][0].images) == 8
    assert "Первая" in text and "страницы 7–8: распознать не удалось" in text

    llm.results = [blind, blind]
    assert await service.read_scan(b"%PDF", "скан.pdf") is None  # nothing read: the old answer stands
    actual_text = "--- страница 1 ---\nФото объекта заказчику не получено, акт не подписан."
    llm.results = [ok(actual_text), ok("--- страница 7 ---\nКонец договора")]
    assert actual_text in await service.read_scan(b"%PDF", "договор.pdf")
    monkeypatch.setattr("bot.services.chat.can_read_scans", lambda: False)
    assert not service.can_read_scan()


async def test_collect_reads_a_scan_and_tells_the_user_how_long_it_takes(monkeypatch) -> None:
    monkeypatch.setattr("bot.telegram.routers.chat.page_count", lambda data: 12)
    told: list[str] = []

    async def notify(text: str) -> None:
        told.append(text)

    async def read_scan(data: bytes, name: str) -> str:
        return "--- страница 1 ---\nТекст со скана"

    bot = FakeBot({"s": tiny_pdf("")})
    text, _ = await _collect(bot, [_doc("s", "лекция.pdf", "application/pdf", caption="о чём?")], None, None,
                             None, read_scan, notify)  # fmt: skip
    assert '<file name="лекция.pdf">\n--- страница 1 ---\nТекст со скана\n</file>' in text
    assert told and "12 стр." in told[0] and "мин" in told[0]
    monkeypatch.setattr("bot.telegram.routers.chat.page_count", lambda data: 400)
    await _collect(bot, [_doc("s", "лекция.pdf", "application/pdf")], read_scan=read_scan, notify=notify)
    assert "первые 150 из 400" in told[-1]

    async def nothing(data: bytes, name: str) -> None:
        return None

    with pytest.raises(ScannedPdf):
        await _collect(
            bot, [_doc("s", "лекция.pdf", "application/pdf", caption="?")], None, None, None, nothing
        )
