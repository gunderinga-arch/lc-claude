import io
import json
import struct
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen
from PIL import Image
from pptx import Presentation

from bot.files import deck_qa, fonts, slides


def tiny_ttf(family: str, bold: bool = False) -> bytes:
    """A real TrueType font with a few glyphs (box shapes), enough for metrics, wrapping and embedding."""
    chars = "абвгдежзиклмнопрстуфхцчшщыэюяABCDEFGHIJKLMNOPQRSTUVWXYZ abc0123456789.,-"
    names = [".notdef", *[f"g{i}" for i in range(len(chars))]]
    fb = FontBuilder(1000, isTTF=True)
    fb.setupGlyphOrder(names)
    fb.setupCharacterMap({ord(c): f"g{i}" for i, c in enumerate(chars)})
    pen = TTGlyphPen(None)
    pen.moveTo((50, 0)), pen.lineTo((50, 700)), pen.lineTo((450, 700)), pen.lineTo((450, 0)), pen.closePath()
    box = pen.glyph()
    fb.setupGlyf({n: box for n in names})
    fb.setupHorizontalMetrics({n: (300 if n == "g29" else 500, 50) for n in names})  # g29 is the space
    fb.setupHorizontalHeader(ascent=800, descent=-200)
    fb.setupNameTable({"familyName": family, "styleName": "Bold" if bold else "Regular"})
    fb.setupOS2(
        sTypoAscender=800, usWinAscent=900, usWinDescent=300, usWeightClass=700 if bold else 400, fsType=0
    )
    fb.setupPost()
    out = io.BytesIO()
    fb.save(out)
    return out.getvalue()


@pytest.fixture
def catalog(tmp_path: Path, monkeypatch):
    """A one-family font catalog like deploy/fonts.py writes, in place of the server's."""
    folder = tmp_path / "Testa"
    folder.mkdir()
    (folder / "400.ttf").write_bytes(tiny_ttf("Testa"))
    (folder / "700.ttf").write_bytes(tiny_ttf("Testa", bold=True))
    item = {
        "family": "Testa",
        "category": "Display",
        "stroke": "Sans Serif",
        "classifications": ["display"],
        "rank": 1,
        "files": {"400": str(folder / "400.ttf"), "700": str(folder / "700.ttf")},
    }
    (tmp_path / "catalog.json").write_text(json.dumps([item]), encoding="utf-8")
    monkeypatch.setattr(fonts, "FONTS_DIR", tmp_path)
    fonts.catalog.cache_clear()
    real = fonts.catalog
    monkeypatch.setattr(fonts, "catalog", lambda root=tmp_path: real(tmp_path))
    yield tmp_path
    real.cache_clear()


def test_eot_header_wraps_the_font_unchanged() -> None:
    ttf = tiny_ttf("Testa")
    eot = fonts.to_eot(ttf)
    size, data_size, version = struct.unpack_from("<LLL", eot)
    assert size == len(eot) and data_size == len(ttf) and version == 0x00020001
    assert struct.unpack_from("<H", eot, 34)[0] == 0x504C  # the magic number
    assert eot.endswith(ttf) and "Testa".encode("utf-16-le") in eot


CANVAS = {
    "title": "Тест",
    "fonts": {"heading": "Testa", "body": "Testa"},
    "slides": [
        {
            "background": "#0B0B0B",
            "elements": [
                {
                    "type": "text",
                    "role": "heading",
                    "x": 0.6,
                    "y": 0.6,
                    "w": 8,
                    "h": 2,
                    "text": "ЖИМ",
                    "size": 96,
                    "bold": True,
                    "caps": True,
                    "spacing": 2,
                    "color": "#FFFFFF",
                },
                {
                    "type": "rect",
                    "x": 9,
                    "y": 1,
                    "w": 3,
                    "h": 3,
                    "fill": "#E63312",
                    "opacity": 0.5,
                    "radius": 0.3,
                },
                {"type": "ellipse", "x": 1, "y": 4, "w": 1, "h": 1, "fill": "#FFFFFF"},
                {"type": "line", "x": 0.6, "y": 6.8, "x2": 12.7, "y2": 6.8, "color": "#FFFFFF", "width": 1},
                {"type": "text", "x": 0.6, "y": 5, "w": 4, "h": 1, "font": "Нет-такого", "text": "подпись"},
            ],
            "notes": "речь",
        }
    ],
}


def test_canvas_deck_is_native_pptx_with_its_fonts_embedded(catalog) -> None:
    built = slides.build_deck(json.dumps(CANVAS, ensure_ascii=False), {})
    assert any("Нет-такого" in w for w in built.warnings)  # unknown font replaced and reported

    prs = Presentation(io.BytesIO(built.pptx))
    shapes = list(prs.slides[0].shapes)
    title = next(s for s in shapes if s.has_text_frame and "ЖИМ" in s.text_frame.text)
    rpr = title.text_frame.paragraphs[0].runs[0]._r.rPr
    assert rpr.get("spc") == "200" and rpr.get("cap") == "all"
    assert title.text_frame.paragraphs[0].runs[0].font.name == "Testa"
    assert sum(1 for s in shapes if s.shape_type == 9) == 1  # the line (a connector)
    assert any("alpha" in s._element.xml for s in shapes)  # the half-transparent rectangle
    assert prs.slides[0].notes_slide.notes_text_frame.text == "речь"

    z = zipfile.ZipFile(io.BytesIO(built.pptx))
    pres = z.read("ppt/presentation.xml").decode()
    assert 'embedTrueTypeFonts="1"' in pres
    assert pres.index("<p:embeddedFontLst>") < pres.index("<p:defaultTextStyle")
    assert '<p:font typeface="Testa"' in pres and "<p:regular " in pres and "<p:bold " in pres
    fntdata = [n for n in z.namelist() if n.endswith(".fntdata")]
    assert len(fntdata) == 2 and z.read(fntdata[0])[34:36] == b"\x4c\x50"
    assert b'Extension="fntdata"' in z.read("[Content_Types].xml")
    assert b"relationships/font" in z.read("ppt/_rels/presentation.xml.rels")

    page = built.preview
    assert "@font-face{font-family:'Testa'" in page and "text-transform:uppercase" in page
    assert "border-radius:50%" in page and "rgba(230,51,18,0.5)" in page and "rotate(0.00deg)" in page


def test_text_that_does_not_fit_is_shrunk_and_reported(catalog) -> None:
    long = {
        "title": "x",
        "fonts": {"body": "Testa"},
        "slides": [
            {
                "elements": [
                    {
                        "type": "text",
                        "x": 1,
                        "y": 1,
                        "w": 3,
                        "h": 0.6,
                        "size": 40,
                        "text": "очень длинный текст " * 3,
                    }
                ]
            }
        ],
    }
    built = slides.build_deck(json.dumps(long, ensure_ascii=False), {})
    assert any("уменьш" in w or "не влезает" in w for w in built.warnings)
    box = slides.layout(slides.parse_deck(json.dumps(long, ensure_ascii=False)), {})[0][0]
    assert box.size < 40


def test_sheets_number_the_slides_six_to_a_page(tmp_path: Path) -> None:
    pngs = []
    for i in range(7):
        p = tmp_path / f"s-{i}.png"
        Image.new("RGB", (160, 90), "#123456").save(p)
        pngs.append(p)
    sheets = deck_qa._sheets(pngs)
    assert len(sheets) == 2
    with Image.open(io.BytesIO(sheets[0])) as img:
        assert img.size == (2 * deck_qa.SLIDE_W + 30, 3 * deck_qa.SLIDE_H + 40)


async def test_no_render_without_libreoffice(monkeypatch) -> None:
    monkeypatch.setattr(deck_qa.shutil, "which", lambda _: None)
    assert await deck_qa.render_sheets(b"pptx") == []


async def test_review_replaces_the_deck_only_with_a_working_one(monkeypatch) -> None:
    from bot.llm.base import ChatMessage, ChatResult, ChatStatus
    from bot.services import chat as chat_mod

    async def sheets(_pptx):
        return [b"png"]

    monkeypatch.setattr(chat_mod, "render_sheets", sheets)
    deck = {
        "title": "a",
        "slides": [{"elements": [{"type": "text", "x": 1, "y": 1, "w": 5, "h": 1, "text": "раз"}]}],
    }
    fixed = {
        "title": "a",
        "slides": [{"elements": [{"type": "text", "x": 1, "y": 1, "w": 5, "h": 1, "text": "два"}]}],
    }
    raw, new = json.dumps(deck, ensure_ascii=False), json.dumps(fixed, ensure_ascii=False)
    answer = f'Готово.\n<<<FILE name="d.pptx">>>\n{raw}\n<<<END>>>'

    class FakeLLM:
        def __init__(self, reply: str) -> None:
            self.reply, self.calls = reply, []

        async def chat(self, messages, model_id, tools=()):
            self.calls.append(messages)
            return ChatResult(ChatStatus.OK, self.reply, model_id, 1, 1, False)

    service = SimpleNamespace(
        _is_blind=lambda *_: False, _llm=FakeLLM(f'<<<FILE name="d.pptx">>>\n{new}\n<<<END>>>')
    )
    msgs = [ChatMessage(role="user", content="сделай презентацию")]
    out = await chat_mod.ChatService._polish_decks(
        service, answer, msgs, ("data:image/png;base64,ref",), "m", {}
    )
    assert new in out and raw not in out
    last = service._llm.calls[0][-1]
    assert last.images[-1] == "data:image/png;base64,ref" and "референсы" in last.content

    service = SimpleNamespace(_is_blind=lambda *_: False, _llm=FakeLLM("OK"))
    assert await chat_mod.ChatService._polish_decks(service, answer, msgs, (), "m", {}) == answer
    service = SimpleNamespace(
        _is_blind=lambda *_: False, _llm=FakeLLM('<<<FILE name="d.pptx">>>\n{broken\n<<<END>>>')
    )
    assert await chat_mod.ChatService._polish_decks(service, answer, msgs, (), "m", {}) == answer


def test_deck_talk_includes_edits_without_the_word() -> None:
    from bot.llm.base import ChatMessage
    from bot.services.chat import _is_deck_talk, _with_deck_guide

    history = [
        ChatMessage(role="system", content="sys"),
        ChatMessage(role="assistant", content='<<<FILE name="a.pptx">>>\n{}\n<<<END>>>'),
    ]
    assert _is_deck_talk("переделай её в таком стиле", history)
    assert _is_deck_talk("сделай презентацию", history[:1]) and not _is_deck_talk("привет", history[:1])
    guided = _with_deck_guide(history)
    assert guided[0].content.startswith("sys") and "Каталог шрифтов" in guided[0].content
