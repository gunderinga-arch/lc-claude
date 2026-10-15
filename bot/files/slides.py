"""Presentations: the model writes a deck as JSON in a <<<FILE name="x.pptx">>> block, the bot builds a real
.pptx and an HTML preview from it.

Every slide is first laid out as boxes in inches on a 13.333×7.5 (16:9) page; the PPTX and the HTML renderer
draw the very same boxes, so the preview matches the file. In the HTML each slide is a fixed 16:9 frame
scaled as a whole (container units), so it looks the same on a phone and a computer, only smaller.
"""

from __future__ import annotations

import base64
import html
import io
import json
import logging
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, replace

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

from bot.files import fonts
from bot.files.blocks import OutFile

logger = logging.getLogger(__name__)

W, H = 13.333, 7.5  # inches, 16:9
FONT = "Arial"  # present on every PowerPoint; the preview falls back to metric-compatible sans fonts
MAX_SLIDES = 30
MAX_BULLETS = 8

# Always in the system prompt: just enough to recognise the format; the full guide comes with deck requests.
SLIDES_INSTRUCTIONS = (
    'Презентацию PowerPoint верни одним блоком <<<FILE name="имя.pptx">>> с JSON вместо содержимого — '
    "бот соберёт из него настоящий редактируемый PPTX и HTML-превью. Подробный формат и правила дизайна "
    "приходят вместе с запросом о презентации."
)

DECK_GUIDE = """# Как делать презентацию
Ты — дизайнер презентаций уровня сильной студии. Слайд 16:9 — это холст 13.333 × 7.5 дюйма, координаты в дюймах
от левого верхнего угла. Ты сам расставляешь элементы; бот превращает их в родные объекты PowerPoint (текст
остаётся редактируемым, использованные шрифты встраиваются в файл) и в HTML-превью, которое выглядит так же.
После сборки ты увидишь снимки слайдов и сможешь исправить вёрстку.

## Формат: один блок <<<FILE name="имя.pptx">>> с JSON
{"title": "…",
 "style": {"palette": ["#111111", …], "fonts": "…", "idea": "…"},   ← сначала зафиксируй стиль (см. референсы)
 "fonts": {"heading": "Семейство", "body": "Семейство"},            ← шрифты только из каталога ниже
 "slides": [{"background": "#0B0B0B" | {"image": "имя.jpg"}, "elements": [ … ], "notes": "речь выступающего"}]}

Элементы (любой может иметь "opacity": 0–1):
- {"type":"text","x","y","w","h","text":"строки через \\n" | "runs":[{"text","bold","italic","color","size","font"}],
   "role":"heading"|"body", "font","size"(pt),"bold","italic","color":"#RRGGBB","align":"left|center|right",
   "valign":"top|middle|bottom","caps":true,"spacing"(pt между буквами),"line"(множитель интерлиньяжа, 0.8–1.4),
   "gap"(pt после абзаца),"bullet":"—"|"•"(знак перед каждым абзацем),"fill":"#RRGGBB"(подложка)}
- {"type":"rect"|"ellipse","x","y","w","h","fill","stroke","stroke_width","radius"(дюймы, скругление)}
- {"type":"line","x","y","x2","y2","color","width"(pt)}
- {"type":"image","image":"имя","x","y","w","h","fit":"cover"|"contain"} — только картинки, которые ты нарисовал
  инструментом в этом ответе (или раньше в этом чате).
Текстовый блок задавай с запасом по высоте: бот меряет текст настоящим шрифтом и, если не влезает, уменьшит кегль
и сообщит тебе. Старые раскладки ("layout": "title|section|bullets|two_columns|stat|quote|image") тоже работают,
но для дизайнерской работы используй elements.

## Референсы
Если пользователь прислал картинки-образцы, сначала разбери их и запиши в "style": точные цвета (HEX), какие шрифты
из каталога ближе всего (гротеск/антиква/акцидентный, ширина, контраст штриха, капс или нет), пропорции
заголовка к тексту, сетку и поля, фактуры и приёмы. Потом верстай строго по этому — без «своего» акцента и цветов.

## Дизайн
- Одна сквозная идея и строгая сетка: поля не меньше 0.5″, одинаковые отступы (0.3″ или 0.5″), заголовки одного
  типа слайдов — в одной позиции, одним кеглем.
- Типографика — главный инструмент: контраст размеров (заголовок 40–120 pt рядом с подписью 10–14 pt), текст
  от 14 pt, иерархия 2–3 уровня, «ёлочки» и тире. Не больше двух семейств.
- Палитра под тему: один доминирующий цвет, 1–2 поддерживающих, один акцент. Контраст текста к фону — высокий.
- Разные композиции: обложка, разделитель, большая цифра, цитата, сравнение, схема из фигур и линий, финал.
  Каждому слайду — визуальный элемент (фигура, линия, цифра, картинка), а не только текст.
- Мало текста на слайде, подробности — в notes.
Не делай ИИ-штампов: подчёркивания под заголовком, цветных полос по краю слайда или карточки, фиолетово-синих
градиентов, сеток из одинаковых карточек с тенями, всего по центру, эмодзи-иконок.

## Каталог шрифтов (с кириллицей; * — без жирного начертания)
{catalog}
"""  # noqa: E501 — prose for the model, wrapped for reading, not for the linter


def deck_guide() -> str:
    """The full guide for a deck request, with the fonts this server has."""
    return DECK_GUIDE.replace("{catalog}", fonts.catalog_text() or "Arial, Georgia, Times New Roman")


class DeckError(ValueError):
    """The model's JSON is not a usable deck."""


@dataclass(frozen=True, slots=True)
class Theme:
    bg: str
    text: str
    muted: str
    accent: str
    panel: str  # quiet fill for cards and image placeholders


THEMES = {
    "light": Theme("FFFFFF", "1F2430", "6B7280", "2563EB", "EEF2FA"),
    "dark": Theme("14161F", "F2F3F7", "9AA0B4", "7C9CFF", "1F2230"),
    "warm": Theme("FBF6EE", "2B2118", "7A6A58", "D9692B", "F1E6D6"),
    "forest": Theme("0F2A24", "EAF5EF", "9CC3B4", "3DDC97", "17382F"),
}


@dataclass(frozen=True, slots=True)
class Run:
    text: str
    color: str
    bold: bool = False
    italic: bool = False
    size: float | None = None  # pt; None = the box's size
    font: str | None = None  # None = the box's font


@dataclass(frozen=True, slots=True)
class Box:
    """A rectangle on the page (inches). Text boxes carry paragraphs of runs; `fill` paints the box.

    `shape` draws a figure instead of a plain fill: rect | round | ellipse | line (a line runs from (x, y)
    to (x + w, y + h), so w and h may be negative).
    """

    x: float
    y: float
    w: float
    h: float
    paragraphs: tuple[tuple[Run, ...], ...] = ()
    size: float = 20  # pt
    align: str = "l"  # l | c | r
    anchor: str = "t"  # t | m | b
    fill: str | None = None
    image: str | None = None  # name in the deck's images
    gap: float = 0.0  # pt of space after each paragraph
    font: str = "Arial"
    spacing: float = 0.0  # pt between letters
    line: float = 1.0  # line spacing, a multiple of the font's single spacing
    caps: bool = False
    shape: str | None = None
    opacity: float = 1.0  # of the fill or the picture
    radius: float = 0.0  # inches, for "round"
    stroke: str | None = None  # outline color (the color of a line)
    stroke_w: float = 0.0  # pt
    fit: str = "cover"  # pictures: cover | contain


@dataclass(slots=True)
class Slide:
    layout: str
    data: dict
    notes: str = ""


@dataclass(slots=True)
class Deck:
    title: str
    theme: Theme
    slides: list[Slide] = field(default_factory=list)
    # Problems found while building (unknown fonts, text shrunk or still not fitting): shown to the model
    # when it reviews the rendered slides.
    warnings: list[str] = field(default_factory=list)


# ---- parsing ----------------------------------------------------------------------------------------


def _text(value: object, limit: int = 300) -> str:
    return " ".join(str(value or "").split())[:limit]


def _bullets(value: object) -> list[str]:
    items = value if isinstance(value, list) else []
    return [t for t in (_text(v, 200) for v in items) if t][:MAX_BULLETS]


def parse_deck(raw: str) -> Deck:
    """The model's JSON -> a Deck; unknown layouts become bullet slides, empty slides are dropped."""
    body = raw.strip()
    if body.startswith("```"):  # a fence inside the block
        body = body.strip("`").removeprefix("json").strip()
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise DeckError(f"не JSON: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("slides"), list):
        raise DeckError("нет списка slides")
    theme = THEMES.get(str(data.get("theme", "")).lower(), THEMES["light"])
    deck = Deck(_text(data.get("title"), 120), theme)
    fonts = data.get("fonts") if isinstance(data.get("fonts"), dict) else {}
    body_font = _font(fonts.get("body"), FONT, deck.warnings)
    head_font = _font(fonts.get("heading"), body_font, deck.warnings)
    for n, item in enumerate(data["slides"][:MAX_SLIDES], 1):
        if not isinstance(item, dict):
            continue
        if isinstance(item.get("elements"), list):  # free canvas: the model places every element itself
            canvas = _canvas(item, n, theme, body_font, head_font, deck.warnings)
            if canvas["elements"]:
                deck.slides.append(Slide("canvas", canvas, _text(item.get("notes"), 3000)))
            continue
        layout = str(item.get("layout", "bullets")).lower()
        if layout not in LAYOUTS:
            layout = "bullets"
        clean = {
            "title": _text(item.get("title"), 140),
            "subtitle": _text(item.get("subtitle"), 220),
            "bullets": _bullets(item.get("bullets")),
            "value": _text(item.get("value"), 24),
            "caption": _text(item.get("caption"), 200),
            "quote": _text(item.get("quote"), 400),
            "author": _text(item.get("author"), 100),
            "image": _text(item.get("image"), 100),
        }
        for side in ("left", "right"):
            col = item.get(side) if isinstance(item.get(side), dict) else {}
            clean[side] = {"heading": _text(col.get("heading"), 80), "bullets": _bullets(col.get("bullets"))}
        if any(v for k, v in clean.items() if k not in ("left", "right")) or clean["left"]["bullets"]:
            deck.slides.append(Slide(layout, clean, _text(item.get("notes"), 3000)))
    if not deck.slides:
        raise DeckError("пустая презентация")
    return deck


# ---- the free canvas --------------------------------------------------------------------------------

MAX_ELEMENTS = 80
_HEX = re.compile(r"#?([0-9a-fA-F]{6}|[0-9a-fA-F]{3})")
_ALIGNS = {"l": "l", "left": "l", "c": "c", "center": "c", "centre": "c", "r": "r", "right": "r"}
_ANCHORS = {"t": "t", "top": "t", "m": "m", "middle": "m", "center": "m", "b": "b", "bottom": "b"}
_SHAPES = {
    "rect": "rect",
    "rectangle": "rect",
    "round": "round",
    "rounded": "round",
    "ellipse": "ellipse",
    "circle": "ellipse",
    "oval": "ellipse",
    "line": "line",
}


def _color(value: object, default: str | None) -> str | None:
    m = _HEX.fullmatch(str(value or "").strip())
    if not m:
        return default
    hex_ = m.group(1)
    return (hex_ if len(hex_) == 6 else "".join(c * 2 for c in hex_)).upper()


def _num(value: object, default: float, lo: float, hi: float) -> float:
    try:
        return min(hi, max(lo, float(value)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _font(name: object, default: str, warnings: list[str]) -> str:
    """A catalog family (exact casing), or `default` with a warning; any name passes without a catalog."""
    wanted = _text(name, 60)
    if not wanted:
        return default
    if not fonts.catalog():
        return wanted
    fam = fonts.resolve(wanted)
    if fam is None:
        warnings.append(f"шрифта «{wanted}» нет в каталоге — заменён на {default}")
        return default
    return fam.name


def _runs(el: dict, color: str, font: str, warnings: list[str]) -> list[Run]:
    raw = el.get("runs") if isinstance(el.get("runs"), list) else [{"text": el.get("text", "")}]
    runs = []
    for r in raw[:40]:
        if not isinstance(r, dict):
            r = {"text": r}
        text = str(r.get("text", ""))[:2000]
        if text:
            size = _num(r["size"], 20, 4, 400) if "size" in r else None
            runs.append(
                Run(
                    text,
                    _color(r.get("color"), color) or color,
                    bool(r.get("bold", el.get("bold"))),
                    bool(r.get("italic", el.get("italic"))),
                    size,
                    _font(r["font"], font, warnings) if r.get("font") else None,
                )
            )
    return runs


def _paragraphs(runs: list[Run], bullet: str) -> tuple[tuple[Run, ...], ...]:
    """Split runs at "\n" into paragraphs; `bullet` (a sign like "•") starts every paragraph."""
    paras: list[list[Run]] = [[]]
    for run in runs:
        for i, piece in enumerate(run.text.split("\n")):
            if i:
                paras.append([])
            if piece:
                paras[-1].append(Run(piece, run.color, run.bold, run.italic, run.size, run.font))
    out = []
    for p in paras:
        if p and bullet:
            p.insert(0, Run(bullet + "  ", p[0].color, p[0].bold, False, p[0].size, p[0].font))
        out.append(tuple(p))
    return tuple(out)


def _canvas(item: dict, n: int, t: Theme, body: str, head: str, warnings: list[str]) -> dict:
    """A canvas slide's checked elements, as Boxes ready for both renderers."""
    bg = item.get("background")
    bg_image = _text(bg.get("image"), 100) if isinstance(bg, dict) else ""
    out: dict = {
        "bg": _color(bg if isinstance(bg, str) else (bg or {}).get("color"), t.bg),
        "bg_image": bg_image,
        "elements": [],
    }
    for el in item["elements"][:MAX_ELEMENTS]:
        if not isinstance(el, dict):
            continue
        kind = str(el.get("type", "text")).lower()
        x, y = _num(el.get("x"), 0, -W, 2 * W), _num(el.get("y"), 0, -H, 2 * H)
        w, h = _num(el.get("w"), 1, 0.01, 3 * W), _num(el.get("h"), 1, 0.01, 3 * H)
        opacity = _num(el.get("opacity"), 1, 0, 1)
        if kind == "text":
            role_font = head if str(el.get("role", "")).lower() in ("heading", "title") else body
            font = _font(el.get("font"), role_font, warnings) if el.get("font") else role_font
            color = _color(el.get("color"), t.text) or t.text
            runs = _runs(el, color, font, warnings)
            if not runs:
                continue
            out["elements"].append(
                Box(
                    x,
                    y,
                    w,
                    h,
                    _paragraphs(runs, _text(el.get("bullet"), 3)),
                    size=_num(el.get("size"), 20, 4, 400),
                    align=_ALIGNS.get(str(el.get("align", "l")).lower(), "l"),
                    anchor=_ANCHORS.get(str(el.get("valign", "t")).lower(), "t"),
                    fill=_color(el.get("fill"), None),
                    gap=_num(el.get("gap"), 0, 0, 200),
                    font=font,
                    spacing=_num(el.get("spacing"), 0, -10, 100),
                    line=_num(el.get("line"), 1, 0.6, 4),
                    caps=bool(el.get("caps")),
                    opacity=opacity,
                )
            )
        elif kind == "image":
            name = _text(el.get("image"), 100)
            if name:
                fit = "contain" if str(el.get("fit", "")).lower() == "contain" else "cover"
                out["elements"].append(Box(x, y, w, h, image=name, opacity=opacity, fit=fit))
        elif kind == "line":
            x2, y2 = _num(el.get("x2"), x + w, -W, 2 * W), _num(el.get("y2"), y, -H, 2 * H)
            out["elements"].append(
                Box(
                    x,
                    y,
                    x2 - x,
                    y2 - y,
                    shape="line",
                    opacity=opacity,
                    stroke=_color(el.get("color"), t.text),
                    stroke_w=_num(el.get("width"), 1, 0.25, 40),
                )
            )
        else:
            shape = _SHAPES.get(str(el.get("shape", kind)).lower(), "rect")
            radius = _num(el.get("radius"), 0, 0, min(w, h) / 2)
            if shape == "rect" and radius:
                shape = "round"
            out["elements"].append(
                Box(
                    x,
                    y,
                    w,
                    h,
                    fill=_color(el.get("fill"), None),
                    shape=shape,
                    opacity=opacity,
                    radius=radius,
                    stroke=_color(el.get("stroke"), None),
                    stroke_w=_num(el.get("stroke_width"), 0, 0, 40),
                )
            )
    return out


# ---- layout: slide -> boxes -------------------------------------------------------------------------


GLYPH = 0.52  # average Arial glyph width, in em, for Russian text (measured 0.49–0.53; bold 0.55)
LINE = 1.18  # line height, in em (the preview's line-height; PowerPoint's single spacing is close)
BULLET_SIZES = (28, 26, 24, 22, 20, 18, 16, 14)


def _fit(
    texts: list[str], w: float, h: float, sizes: tuple[float, ...], gap: float = 0.55, indent: float = 0
) -> float:
    """The largest size whose wrapped text fits a w×h box (inches); `indent` em go to a bullet sign.

    ponytail: the wrap is estimated from an average glyph width with 8% slack for word breaks, not measured
    with the font; measure with Pillow's ImageFont if real decks still overflow or look sparse.
    """
    for size in sizes:
        em = size / 72
        per_line = max(1, int((w - indent * em) / (em * GLYPH) * 0.92))
        lines = sum(-(-max(1, len(t)) // per_line) for t in texts)  # ceil division
        if lines * em * LINE + len(texts) * em * gap <= h:
            return size
    return sizes[-1]


def _bullet_box(x, y, w, h, items: list[str], t: Theme, size: float) -> Box:
    paras = tuple((Run("●  ", t.accent, bold=True), Run(i, t.text)) for i in items)
    return Box(x, y, w, h, paras, size=size, gap=size * 0.55)


def _title_box(s: dict, t: Theme) -> list[Box]:
    return [
        Box(0.8, 0.55, W - 1.6, 1.0, ((Run(s["title"], t.text, bold=True),),), size=32, anchor="m"),
        Box(0.8, 1.6, 1.1, 0.07, fill=t.accent),
    ]


def _lay_title(s, t, images):
    boxes = [Box(0, 0, 0.28, H, fill=t.accent)]
    boxes.append(Box(1.0, 2.0, W - 2.0, 2.0, ((Run(s["title"], t.text, bold=True),),), size=46, anchor="b"))
    if s["subtitle"]:
        boxes.append(Box(1.0, 4.25, W - 2.0, 1.4, ((Run(s["subtitle"], t.muted),),), size=22))
    return boxes


def _lay_section(s, t, images):
    boxes = [Box(0, 0, W, H, fill=t.accent)]
    boxes.append(Box(1.0, 2.4, W - 2.0, 1.6, ((Run(s["title"], t.bg, bold=True),),), size=42, anchor="b"))
    if s["subtitle"]:
        boxes.append(Box(1.0, 4.15, W - 2.0, 1.2, ((Run(s["subtitle"], t.bg),),), size=22))
    return boxes


def _lay_bullets(s, t, images):
    items = s["bullets"] or [x for x in (s["subtitle"] or s["caption"],) if x]
    w, h = W - 1.6, H - 2.6
    size = _fit(items, w, h, BULLET_SIZES, indent=1.6)
    return [*_title_box(s, t), _bullet_box(0.8, 2.05, w, h, items, t, size)]


def _lay_two_columns(s, t, images):
    boxes = _title_box(s, t)
    colw = (W - 1.6 - 0.6) / 2
    for i, side in enumerate(("left", "right")):
        col, x = s[side], 0.8 + i * (colw + 0.6)
        boxes.append(Box(x, 2.0, colw, H - 2.5, fill=t.panel))
        boxes.append(
            Box(x + 0.35, 2.25, colw - 0.7, 0.7, ((Run(col["heading"], t.accent, bold=True),),), size=22)
        )
        size = _fit(col["bullets"], colw - 0.7, H - 3.8, BULLET_SIZES[2:], indent=1.6)
        boxes.append(_bullet_box(x + 0.35, 3.05, colw - 0.7, H - 3.8, col["bullets"], t, size))
    return boxes


def _lay_stat(s, t, images):
    boxes = []
    if s["title"]:
        boxes.append(
            Box(0.8, 0.7, W - 1.6, 0.9, ((Run(s["title"], t.muted, bold=True),),), size=24, align="c")
        )
    boxes.append(
        Box(
            0.8,
            1.7,
            W - 1.6,
            2.9,
            ((Run(s["value"], t.accent, bold=True),),),
            size=110,
            align="c",
            anchor="m",
        )
    )
    if s["caption"]:
        boxes.append(Box(1.5, 4.8, W - 3.0, 1.6, ((Run(s["caption"], t.text),),), size=24, align="c"))
    return boxes


def _lay_quote(s, t, images):
    boxes = [Box(0.8, 0.6, 1.5, 1.6, ((Run("“", t.accent, bold=True),),), size=120)]
    size = _fit([s["quote"]], W - 3.2, 3.6, (40, 36, 32, 28, 24, 20), gap=0)
    boxes.append(
        Box(1.6, 1.7, W - 3.2, 3.6, ((Run(s["quote"], t.text, italic=True),),), size=size, anchor="m")
    )
    if s["author"]:
        boxes.append(Box(1.6, 5.5, W - 3.2, 0.8, ((Run("— " + s["author"], t.muted),),), size=20))
    return boxes


def _lay_image(s, t, images):
    if s["image"] not in images:
        return _lay_bullets(s, t, images)
    half = W / 2
    boxes = [Box(half, 0, half, H, image=s["image"])]
    boxes.append(
        Box(0.7, 0.8, half - 1.2, 1.4, ((Run(s["title"], t.text, bold=True),),), size=30, anchor="b")
    )
    boxes.append(Box(0.7, 2.35, 1.0, 0.07, fill=t.accent))
    if s["bullets"]:
        boxes.append(
            _bullet_box(
                0.7,
                2.75,
                half - 1.2,
                H - 3.3,
                s["bullets"],
                t,
                _fit(s["bullets"], half - 1.2, H - 3.3, BULLET_SIZES[2:], indent=1.6),
            )
        )
    elif s["subtitle"] or s["caption"]:
        boxes.append(
            Box(0.7, 2.75, half - 1.2, H - 3.3, ((Run(s["subtitle"] or s["caption"], t.text),),), size=20)
        )
    return boxes


LAYOUTS = {
    "title": _lay_title,
    "section": _lay_section,
    "bullets": _lay_bullets,
    "two_columns": _lay_two_columns,
    "stat": _lay_stat,
    "quote": _lay_quote,
    "image": _lay_image,
}


def text_height(b: Box) -> tuple[float, bool]:
    """(inches the text needs, a word is wider than the box), measured with the real font files."""
    need, too_wide, width_pt = 0.0, False, abs(b.w) * 72
    for para in b.paragraphs:
        size = max((r.size or b.size) for r in para) if para else b.size
        text = "".join(r.text for r in para)
        text = text.upper() if b.caps else text
        fam = fonts.resolve((para[0].font if para else None) or b.font)
        if fam is not None:
            path = fam.file(bool(para and para[0].bold))
            lines, wide = fonts.wrapped_lines(text, path, size, width_pt, b.spacing)
            factor = fonts.line_factor(path)
        else:  # no font file (no catalog on this machine): the Arial estimate
            lines, wide, factor = max(1, math.ceil(len(text) * size * GLYPH / width_pt)), False, LINE
        need += lines * size * factor * b.line + b.gap
        too_wide = too_wide or wide
    return need / 72, too_wide


def _fit_text(b: Box, n: int, warnings: list[str]) -> Box:
    """Shrink a canvas text that does not fit its box, 5% at a time down to 60%; report what was done."""
    need, wide = text_height(b)
    if need <= b.h + 0.02 and not wide:
        return b
    snippet = "".join(r.text for p in b.paragraphs for r in p)[:40]
    for k in range(1, 9):
        f = 1 - 0.05 * k
        smaller = replace(
            b,
            size=b.size * f,
            spacing=b.spacing * f,
            paragraphs=tuple(
                tuple(replace(r, size=r.size * f if r.size else None) for r in p) for p in b.paragraphs
            ),
        )
        need2, wide2 = text_height(smaller)
        if need2 <= b.h + 0.02 and not wide2:
            warnings.append(
                f"слайд {n}: текст «{snippet}» не влезал в блок — кегль уменьшен до {round(f * 100)}%"
            )
            return smaller
    warnings.append(
        f"слайд {n}: текст «{snippet}» не влезает в блок {b.w:.1f}×{b.h:.1f}″ даже при 60% кегля "
        f"(нужно ~{need:.1f}″ высоты) — сократите текст или увеличьте блок"
    )
    return smaller


def _lay_canvas(s: dict, n: int, images: Mapping[str, bytes], warnings: list[str]) -> list[Box]:
    boxes = [Box(0, 0, W, H, image=s["bg_image"])] if s["bg_image"] in images else []
    for b in s["elements"]:
        if b.image and b.image not in images:
            warnings.append(
                f"слайд {n}: картинки «{b.image}» нет — её надо нарисовать инструментом в этом ответе"
            )
            continue
        if (
            min(b.x, b.x + b.w) > W
            or max(b.x, b.x + b.w) < 0
            or min(b.y, b.y + b.h) > H
            or max(b.y, b.y + b.h) < 0
        ):
            warnings.append(f"слайд {n}: элемент целиком за краем слайда ({b.x:.1f}, {b.y:.1f})")
            continue
        boxes.append(_fit_text(b, n, warnings) if b.paragraphs else b)
    return boxes


def layout(deck: Deck, images: Mapping[str, bytes]) -> list[list[Box]]:
    """Boxes of every slide, page numbers included (not on title and section slides)."""
    pages = []
    t = deck.theme
    for n, slide in enumerate(deck.slides, 1):
        if slide.layout == "canvas":
            pages.append(_lay_canvas(slide.data, n, images, deck.warnings))
            continue
        boxes = LAYOUTS[slide.layout](slide.data, t, images)
        if slide.layout not in ("title", "section") and not any(b.image for b in boxes):  # not over a picture
            boxes.append(Box(W - 1.6, H - 0.6, 1.1, 0.4, ((Run(str(n), t.muted),),), size=12, align="r"))
        pages.append(boxes)
    return pages


# ---- PPTX -------------------------------------------------------------------------------------------

_ALIGN = {"l": PP_ALIGN.LEFT, "c": PP_ALIGN.CENTER, "r": PP_ALIGN.RIGHT}
_ANCHOR = {"t": MSO_ANCHOR.TOP, "m": MSO_ANCHOR.MIDDLE, "b": MSO_ANCHOR.BOTTOM}


def _cover_crop(data: bytes, w: float, h: float) -> tuple[float, float, float, float]:
    """Crops (l, r, t, b) that make the picture fill a w×h box without stretching."""
    with Image.open(io.BytesIO(data)) as img:
        iw, ih = img.size
    want, have = w / h, iw / ih
    if have > want:  # too wide: cut the sides
        cut = (1 - want / have) / 2
        return cut, cut, 0.0, 0.0
    cut = (1 - have / want) / 2
    return 0.0, 0.0, cut, cut


def _contain(data: bytes, b: Box) -> tuple[float, float, float, float]:
    """The largest x, y, w, h inside the box that keeps the picture's proportions, centred."""
    with Image.open(io.BytesIO(data)) as img:
        iw, ih = img.size
    scale = min(b.w / iw, b.h / ih)
    w, h = iw * scale, ih * scale
    return b.x + (b.w - w) / 2, b.y + (b.h - h) / 2, w, h


def _alpha(shape, opacity: float) -> None:
    """Fill transparency: python-pptx has no API, so <a:alpha> goes into the fill's color by hand."""
    if opacity < 1:
        for clr in shape._element.spPr.iter(qn("a:srgbClr")):
            clr.append(clr.makeelement(qn("a:alpha"), {"val": str(round(opacity * 100000))}))


_SHAPE_KIND = {"rect": MSO_SHAPE.RECTANGLE, "round": MSO_SHAPE.ROUNDED_RECTANGLE, "ellipse": MSO_SHAPE.OVAL}


def _draw_shape(page, b: Box) -> None:
    x, y, w, h = Inches(b.x), Inches(b.y), Inches(b.w), Inches(b.h)
    if b.shape == "line":
        line = page.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, x, y, Inches(b.x + b.w), Inches(b.y + b.h))
        line.line.color.rgb = RGBColor.from_string(b.stroke or "000000")
        line.line.width = Pt(b.stroke_w)
        _alpha(line, b.opacity)
        return
    shape = page.shapes.add_shape(_SHAPE_KIND.get(b.shape or "rect", MSO_SHAPE.RECTANGLE), x, y, w, h)
    style = shape._element.find(qn("p:style"))  # the theme's shadow and outline: the deck sets its own
    if style is not None:
        shape._element.remove(style)
    if b.shape == "round" and min(b.w, b.h) > 0:
        shape.adjustments[0] = min(0.5, b.radius / min(b.w, b.h))
    if b.fill:
        shape.fill.solid()
        shape.fill.fore_color.rgb = RGBColor.from_string(b.fill)
    else:
        shape.fill.background()
    if b.stroke and b.stroke_w:
        shape.line.color.rgb = RGBColor.from_string(b.stroke)
        shape.line.width = Pt(b.stroke_w)
    else:
        shape.line.fill.background()
    shape.shadow.inherit = False
    _alpha(shape, b.opacity)


def _draw_picture(page, b: Box, data: bytes) -> None:
    if b.fit == "contain":
        x, y, w, h = _contain(data, b)
        pic = page.shapes.add_picture(io.BytesIO(data), Inches(x), Inches(y), Inches(w), Inches(h))
    else:
        pic = page.shapes.add_picture(io.BytesIO(data), Inches(b.x), Inches(b.y), Inches(b.w), Inches(b.h))
        pic.crop_left, pic.crop_right, pic.crop_top, pic.crop_bottom = _cover_crop(data, b.w, b.h)
    if b.opacity < 1:
        blip = pic._element.blipFill.find(qn("a:blip"))
        blip.append(blip.makeelement(qn("a:alphaModFix"), {"amt": str(round(b.opacity * 100000))}))


def _draw_text(page, b: Box) -> None:
    frame = page.shapes.add_textbox(Inches(b.x), Inches(b.y), Inches(b.w), Inches(b.h)).text_frame
    frame.word_wrap = True
    frame.vertical_anchor = _ANCHOR[b.anchor]
    frame.margin_left = frame.margin_right = frame.margin_top = frame.margin_bottom = Emu(0)
    for i, runs in enumerate(b.paragraphs):
        para = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
        para.alignment = _ALIGN[b.align]
        para.space_after = Pt(b.gap)
        if b.line != 1:
            para.line_spacing = b.line
        for r in runs:
            run = para.add_run()
            run.text = r.text
            font = run.font
            font.name, font.size = r.font or b.font, Pt(r.size or b.size)
            font.bold, font.italic = r.bold, r.italic
            font.color.rgb = RGBColor.from_string(r.color)
            rpr = run._r.get_or_add_rPr()
            if b.spacing:
                rpr.set("spc", str(round(b.spacing * 100)))  # hundredths of a point
            if b.caps:
                rpr.set("cap", "all")


def build_pptx(deck: Deck, images: Mapping[str, bytes], pages: list[list[Box]] | None = None) -> bytes:
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(W), Inches(H)
    prs.core_properties.title = deck.title
    blank = prs.slide_layouts[6]
    for slide, boxes in zip(deck.slides, pages or layout(deck, images), strict=True):
        page = prs.slides.add_slide(blank)
        page.background.fill.solid()
        page.background.fill.fore_color.rgb = RGBColor.from_string(slide.data.get("bg") or deck.theme.bg)
        for b in boxes:
            if b.image:
                _draw_picture(page, b, images[b.image])
                continue
            if b.shape or b.fill:
                _draw_shape(page, b)
            if b.paragraphs:
                _draw_text(page, b)
        if slide.notes:
            page.notes_slide.notes_text_frame.text = slide.notes
    out = io.BytesIO()
    prs.save(out)
    return out.getvalue()


def used_fonts(pages: list[list[Box]]) -> dict[str, set[bool]]:
    """Family -> {bold flags} of every run, for embedding into the PPTX and the preview."""
    used: dict[str, set[bool]] = {}
    for boxes in pages:
        for b in boxes:
            for para in b.paragraphs:
                for r in para:
                    used.setdefault(r.font or b.font, set()).add(r.bold)
    return used


# ---- HTML preview -----------------------------------------------------------------------------------

_CQ = 100 / W  # container units per inch: the slide is 100cqw wide
_PT = _CQ / 72  # container units per point

_PAGE = """<!doctype html><html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title><style>
*{{box-sizing:border-box;margin:0}}html,body{{height:100%;background:#0d0e12;overflow:hidden}}
body{{font-family:Arial,"Helvetica Neue",Arimo,"Liberation Sans",Roboto,sans-serif}}
.deck{{display:flex;height:100dvh;overflow-x:auto;scroll-snap-type:x mandatory;scrollbar-width:none}}
.deck::-webkit-scrollbar{{display:none}}
.cell{{flex:0 0 100vw;height:100dvh;padding-bottom:64px;display:grid;place-items:center;
scroll-snap-align:center}}
.slide{{position:relative;width:min(94vw,calc((100dvh - 96px) * 16 / 9));aspect-ratio:16/9;
container-type:inline-size;overflow:hidden;border-radius:clamp(4px,1vw,12px);
box-shadow:0 10px 40px rgba(0,0,0,.45)}}
.b{{position:absolute;display:flex;flex-direction:column;overflow:hidden;line-height:1.18;
white-space:pre-wrap;overflow-wrap:anywhere}}
.b img{{width:100%;height:100%;object-fit:cover}}
.bar{{position:fixed;left:0;right:0;bottom:12px;display:flex;justify-content:center;align-items:center;
gap:14px;color:#cfd3e0;font:500 15px/1 Arial,sans-serif;user-select:none}}
.bar button{{width:42px;height:42px;border-radius:50%;border:0;background:#262a36;color:#fff;
font-size:20px;cursor:pointer}}
.bar button:disabled{{opacity:.35;cursor:default}}
</style></head><body><div class="deck" id="deck">{slides}</div>
<div class="bar"><button id="prev" aria-label="Назад">‹</button><span id="pos"></span>
<button id="next" aria-label="Вперёд">›</button></div>
<script>
const prev=document.getElementById("prev"),next=document.getElementById("next");
const d=document.getElementById("deck"),n=d.children.length,pos=document.getElementById("pos");
const cur=()=>Math.round(d.scrollLeft/d.clientWidth);
let want=0,settle=0;/* the slide the strip is heading to: quick taps add up during a smooth scroll */
const go=i=>{{want=Math.max(0,Math.min(n-1,i));d.scrollTo({{left:want*d.clientWidth,behavior:"smooth"}})}};
const show=()=>{{const i=cur();pos.textContent=(i+1)+" / "+n;prev.disabled=i==0;next.disabled=i==n-1}};
prev.onclick=()=>go(want-1);next.onclick=()=>go(want+1);
d.addEventListener("scroll",()=>{{show();clearTimeout(settle);settle=setTimeout(()=>{{want=cur()}},160)}});
addEventListener("keydown",e=>{{
if(["ArrowRight","PageDown"," "].includes(e.key)){{e.preventDefault();go(want+1)}}
if(["ArrowLeft","PageUp"].includes(e.key)){{e.preventDefault();go(want-1)}}}});
let lock=0;d.addEventListener("wheel",e=>{{
if(Math.abs(e.deltaY)<=Math.abs(e.deltaX))return;e.preventDefault();
const t=Date.now();if(t-lock>450){{lock=t;go(want+(e.deltaY>0?1:-1))}}}},{{passive:false}});
addEventListener("resize",()=>{{d.scrollLeft=want*d.clientWidth}});show();
</script></body></html>"""


def _geo(b: Box) -> str:
    return (
        f"left:{b.x * _CQ:.3f}cqw;top:{b.y * _CQ:.3f}cqw;width:{b.w * _CQ:.3f}cqw;height:{b.h * _CQ:.3f}cqw;"
    )


def _rgba(hex_: str, opacity: float) -> str:
    r, g, b = (int(hex_[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{opacity:g})"


def _family(name: str) -> str:
    return f"'{html.escape(name)}',Arial,sans-serif"


def _line_height(b: Box) -> float:
    fam = fonts.resolve(b.font)
    return round((fonts.line_factor(fam.files[fonts.REGULAR]) if fam else LINE) * b.line, 3)


def _shape_html(b: Box) -> str:
    if b.shape == "line":  # a rotated bar from (x, y), as long as the line
        length, angle = math.hypot(b.w, b.h), math.degrees(math.atan2(b.h, b.w))
        thick = b.stroke_w * _PT
        return (
            f'<div class="b" style="left:{b.x * _CQ:.3f}cqw;'
            f"top:calc({b.y * _CQ:.3f}cqw - {thick / 2:.3f}cqw);"
            f"width:{length * _CQ:.3f}cqw;height:{thick:.3f}cqw;"
            f"background:{_rgba(b.stroke or '000000', b.opacity)};"
            f'transform-origin:0 50%;transform:rotate({angle:.2f}deg)"></div>'
        )
    style = _geo(b) + (f"background:{_rgba(b.fill, b.opacity)};" if b.fill else "")
    if b.shape == "ellipse":
        style += "border-radius:50%;"
    elif b.shape == "round":
        style += f"border-radius:{b.radius * _CQ:.3f}cqw;"
    if b.stroke and b.stroke_w:
        style += f"box-shadow:inset 0 0 0 {b.stroke_w * _PT:.3f}cqw #{b.stroke};"  # inside, like PowerPoint's
    return f'<div class="b" style="{style}"></div>'


def _text_html(b: Box) -> str:
    style = (
        _geo(b)
        + f"font-size:{b.size * _PT:.3f}cqw;font-family:{_family(b.font)};line-height:{_line_height(b)};"
        f"justify-content:{ {'t': 'flex-start', 'm': 'center', 'b': 'flex-end'}[b.anchor] };"
        f"text-align:{ {'l': 'left', 'c': 'center', 'r': 'right'}[b.align] };"
        + (f"letter-spacing:{b.spacing * _PT:.3f}cqw;" if b.spacing else "")
        + ("text-transform:uppercase;" if b.caps else "")
    )
    paras = "".join(
        f'<p style="margin-bottom:{b.gap * _PT:.3f}cqw">'
        + "".join(
            f'<span style="color:#{r.color};font-weight:{700 if r.bold else 400};'
            f"font-style:{'italic' if r.italic else 'normal'}"
            + (f";font-size:{r.size * _PT:.3f}cqw" if r.size else "")
            + (f";font-family:{_family(r.font)}" if r.font else "")
            + f'">{html.escape(r.text)}</span>'
            for r in runs
        )
        + "</p>"
        for runs in b.paragraphs
    )
    return f'<div class="b" style="{style}">{paras}</div>'


def _image_html(b: Box, data: bytes) -> str:
    mime = "image/png" if data.startswith(b"\x89PNG") else "image/jpeg"
    uri = f"data:{mime};base64," + base64.b64encode(data).decode()
    style = f"object-fit:{b.fit};" + (f"opacity:{b.opacity:g};" if b.opacity < 1 else "")
    return f'<div class="b" style="{_geo(b)}"><img src="{uri}" alt="" style="{style}"></div>'


def build_preview(deck: Deck, images: Mapping[str, bytes], pages: list[list[Box]] | None = None) -> str:
    pages = pages or layout(deck, images)
    slides = []
    for slide, boxes in zip(deck.slides, pages, strict=True):
        parts = []
        for b in boxes:
            if b.image:
                parts.append(_image_html(b, images[b.image]))
                continue
            if b.shape or b.fill:
                parts.append(_shape_html(b))
            if b.paragraphs:
                parts.append(_text_html(b))
        bg = slide.data.get("bg") or deck.theme.bg
        slides.append(
            f'<div class="cell"><div class="slide" style="background:#{bg}">{"".join(parts)}</div></div>'
        )
    page = _PAGE.format(title=html.escape(deck.title or "Презентация"), slides="".join(slides))
    return page.replace("</style>", fonts.font_faces(used_fonts(pages)) + "</style>", 1)


# ---- the hook for ChatService -------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Built:
    deck: Deck
    pptx: bytes  # with the used catalog fonts embedded
    preview: str
    warnings: tuple[str, ...]


def build_deck(raw: str, images: Mapping[str, bytes]) -> Built:
    """The model's JSON -> PPTX + preview from one layout pass; raises DeckError for unusable JSON."""
    deck = parse_deck(raw)
    pages = layout(deck, images)
    pptx = fonts.embed_fonts(build_pptx(deck, images, pages), used_fonts(pages))
    return Built(deck, pptx, build_preview(deck, images, pages), tuple(dict.fromkeys(deck.warnings)))


def render_decks(files: list[OutFile], images: Mapping[str, bytes]) -> list[OutFile]:
    """Turn each .pptx block (JSON) into the .pptx and its "… (превью).html"; a broken one stays as .json."""
    out: list[OutFile] = []
    for f in files:
        if not f.name.lower().endswith(".pptx"):
            out.append(f)
            continue
        stem = f.name[: -len(".pptx")]
        try:
            built = build_deck(f.data.decode("utf-8"), images)
            out.append(OutFile(f.name, built.pptx))
            out.append(OutFile(f"{stem} (превью).html", built.preview.encode("utf-8")))
        except Exception:
            logger.warning("deck %s could not be built, sent as JSON", f.name, exc_info=True)
            out.append(OutFile(f"{stem}.json", f.data))
    return out
