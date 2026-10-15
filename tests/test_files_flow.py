import io
import json
import time
import zipfile
from types import SimpleNamespace

import httpx
import pytest

from bot.files.blocks import OutFile, safe_filename, split_files
from bot.files.extract import UnsupportedFile, extract_text
from bot.llm.base import ChatMessage, CompletionRequest
from bot.llm.buytokens import BuyTokensProvider
from bot.telegram.routers.chat import DEFAULT_FILE_PROMPT, TELEGRAM_DOWNLOAD_LIMIT, _collect

# ---- files the model sends back --------------------------------------------------------------


def test_split_files_cuts_blocks_and_leaves_a_marker() -> None:
    answer = (
        "Готово:\n"
        '<<<FILE name="report.md">>>\n# Отчёт\nтекст\n<<<END>>>\n'
        '<<<FILE name="data.csv">>>\nгород,население\nМосква,13\n<<<END>>>\n'
        "Что-то ещё?"
    )
    text, files = split_files(answer)
    assert text == "Готово:\n📎 report.md\n📎 data.csv\nЧто-то ещё?"
    assert files[0] == OutFile("report.md", "# Отчёт\nтекст".encode())
    assert files[1].data.startswith(b"\xef\xbb\xbf")  # BOM so Excel reads UTF-8 CSV
    assert files[1].data.decode("utf-8-sig") == "город,население\nМосква,13"


def test_unclosed_block_stays_text() -> None:
    answer = '<<<FILE name="a.txt">>>\nобрыв ответа'
    assert split_files(answer) == (answer, [])


def test_filenames_are_sanitized() -> None:
    assert safe_filename("../../etc/passwd") == "passwd"
    assert safe_filename("C:\\Windows\\evil.bat") == "evil.bat"
    assert safe_filename("отчёт 2026 (итог).md") == "отчёт 2026 (итог).md"
    assert safe_filename("a<>|:*?.txt") == "a_.txt"
    assert safe_filename("...") == "file.txt"
    long = safe_filename("x" * 200 + ".json")
    assert len(long) <= 64 and long.endswith(".json")


# ---- attachments from the user ---------------------------------------------------------------


class FakeBot:
    def __init__(self, blobs: dict[str, bytes]) -> None:
        self.blobs = blobs

    async def download(self, file: SimpleNamespace, destination=None) -> io.BytesIO:
        destination = destination if destination is not None else io.BytesIO()
        destination.write(self.blobs[file.file_id])
        destination.seek(0)
        return destination


def _doc(file_id: str, name: str, mime: str, size: int = 10, caption: str | None = None) -> SimpleNamespace:
    document = SimpleNamespace(file_id=file_id, file_name=name, mime_type=mime, file_size=size)
    return SimpleNamespace(photo=None, document=document, caption=caption)


def _photo(file_id: str, caption: str | None = None) -> SimpleNamespace:
    sizes = [SimpleNamespace(file_id="small", file_size=1), SimpleNamespace(file_id=file_id, file_size=5)]
    return SimpleNamespace(photo=sizes, document=None, caption=caption)


async def test_collect_mixes_images_and_documents() -> None:
    bot = FakeBot({"p1": b"\xff\xd8jpeg", "t1": "привет, файл".encode(), "i1": b"png"})
    messages = [
        _photo("p1", caption="Что тут?"),
        _doc("t1", "notes.txt", "text/plain"),
        _doc("i1", "shot.PNG", "image/png"),
    ]
    text, images = await _collect(bot, messages)
    assert text.startswith("Что тут?\n\n")
    assert '<file name="notes.txt">\nпривет, файл\n</file>' in text
    assert text.count("[Приложено изображение:") == 2
    assert images[0].startswith("data:image/jpeg;base64,") and images[1].startswith("data:image/png;base64,")


async def test_collect_default_prompt_and_errors() -> None:
    text, _ = await _collect(FakeBot({"t": b"x"}), [_doc("t", "a.md", "text/markdown")])
    assert text.startswith(DEFAULT_FILE_PROMPT)
    with pytest.raises(UnsupportedFile, match="20 МБ"):
        await _collect(
            FakeBot({}), [_doc("big", "big.pdf", "application/pdf", size=TELEGRAM_DOWNLOAD_LIMIT + 1)]
        )
    with pytest.raises(UnsupportedFile, match="DOC"):
        await _collect(FakeBot({"d": b"\xd0\xcf\x11\xe0"}), [_doc("d", "old.doc", "application/msword")])


def _xlsx(cell_ref: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(
            "xl/workbook.xml",
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            '<sheets><sheet name="S" r:id="r1"/></sheets></workbook>',
        )
        z.writestr(
            "xl/_rels/workbook.xml.rels",
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="r1" Target="worksheets/s.xml"/></Relationships>',
        )
        rows = f'<row><c r="{cell_ref}"><v>1</v></c></row>' * 20_000
        z.writestr(
            "xl/worksheets/s.xml",
            f'<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            f"<sheetData>{rows}</sheetData></worksheet>",
        )
    return buf.getvalue()


def test_xlsx_forged_or_huge_columns_are_bounded() -> None:
    with pytest.raises(UnsupportedFile):
        extract_text("a.xlsx", _xlsx("ZZZZZZZ1"))
    wide = extract_text("b.xlsx", _xlsx("XFD1"))  # 16384 columns x 20k rows would be ~330 MB of commas
    assert wide.truncated


# ---- images reach the gateway as content parts ----------------------------------------------


async def test_provider_sends_images_as_content_parts() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    provider = BuyTokensProvider("https://gw.test/v1", "k", transport=httpx.MockTransport(handler))
    messages = [
        ChatMessage("system", "sys"),
        ChatMessage("user", "что это?", images=("data:image/png;base64,AA",)),
    ]
    await provider.complete(CompletionRequest("m", messages, 100, 10))
    await provider.aclose()
    assert seen["messages"][0] == {"role": "system", "content": "sys"}
    assert seen["messages"][1]["content"] == [
        {"type": "text", "text": "что это?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}},
    ]


async def test_file_without_caption_waits_for_the_next_message(monkeypatch) -> None:
    from bot.telegram.routers import chat

    chat._waiting.clear()
    answers: list[str] = []
    submitted: list[object] = []

    async def answer(text, reply_markup=None):
        answers.append(text)

    async def submit(message, user, quota, jobs, job, **kw):
        submitted.append(job)

    monkeypatch.setattr(chat, "_submit", submit)
    user = SimpleNamespace(tg_id=7)
    doc = SimpleNamespace(**vars(_doc("t", "a.md", "text/markdown")), media_group_id=None, answer=answer)

    await chat.on_file(doc, user, None, None, None, None, None, None)
    await chat.on_file(doc, user, None, None, None, None, None, None)
    assert answers == [chat._waiting_note(1), chat._waiting_note(2)] and submitted == []  # no model call

    assert len(chat._take_waiting(7)) == 2 and chat._take_waiting(7) == []  # the next message takes them
    chat._waiting[(7, 0)] = (time.monotonic() - chat.WAITING_TTL_S - 1, [doc])  # sent too long ago
    assert chat._take_waiting(7) == []

    text, _ = await _collect(FakeBot({"t": b"x"}), [_doc("t", "a.md", "text/markdown")], request="переведи")
    assert text.startswith("переведи\n\n")  # the later request replaces the default prompt
