"""Fonts for decks: the server's catalog (deploy/fonts.py), embedding into PPTX, the preview's @font-face,
and measuring text so a deck can be checked for overflow before anyone sees it.

PowerPoint keeps embedded fonts as EOT ("Embedded OpenType") in ppt/fonts/*.fntdata, listed in
<p:embeddedFontLst> of presentation.xml (the same as the pptx-embed-fonts library and PowerPoint itself
write). With the fonts inside, the deck opens with its own typography on a computer that lacks them and
stays editable. Only fonts whose license allows embedding are in the catalog.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import struct
import zipfile
from dataclasses import dataclass
from functools import cache, lru_cache
from pathlib import Path

from fontTools.ttLib import TTFont
from PIL import ImageFont

logger = logging.getLogger(__name__)

FONTS_DIR = Path(os.environ.get("FONTS_DIR", "/usr/local/share/fonts/lc"))
REGULAR, BOLD = "400", "700"


@dataclass(frozen=True, slots=True)
class Family:
    name: str  # the typeface name runs and <p:embeddedFont> use
    category: str  # Sans Serif | Serif | Display | Handwriting | Monospace
    tags: tuple[str, ...]  # Google's stroke and classifications: "Sans Serif", "Display", ...
    files: dict[str, Path]  # weight -> TTF

    def file(self, bold: bool) -> Path:
        return self.files.get(BOLD if bold else REGULAR) or self.files[REGULAR]


@cache
def catalog(root: Path = FONTS_DIR) -> dict[str, Family]:
    """Families by lower-case name, in popularity order; empty when the fonts were never downloaded."""
    path = root / "catalog.json"
    if not path.exists():
        logger.warning("no font catalog at %s: decks use their fallback font", path)
        return {}
    out: dict[str, Family] = {}
    for item in json.loads(path.read_text(encoding="utf-8")):
        files = {w: Path(p) for w, p in item["files"].items() if Path(p).exists()}
        if REGULAR in files:
            tags = tuple(t for t in [item.get("stroke", ""), *item.get("classifications", [])] if t)
            out[item["family"].lower()] = Family(item["family"], item["category"], tags, files)
    return out


def resolve(name: str | None, root: Path = FONTS_DIR) -> Family | None:
    return catalog(root).get((name or "").strip().lower()) if name else None


def catalog_text(root: Path = FONTS_DIR) -> str:
    """The catalog for the model: one line per category, most popular first."""
    groups: dict[str, list[str]] = {}
    for fam in catalog(root).values():
        groups.setdefault(fam.category, []).append(fam.name + ("" if BOLD in fam.files else "*"))
    return "\n".join(f"{cat}: {', '.join(names)}" for cat, names in groups.items())


# ---- metrics and measuring --------------------------------------------------------------------------


@cache
def line_factor(path: Path) -> float:
    """One line of "single" spacing in em, from the Windows metrics PowerPoint uses."""
    font = TTFont(path, lazy=True)
    os2, upm = font["OS/2"], font["head"].unitsPerEm
    return round((os2.usWinAscent + os2.usWinDescent) / upm, 3)


@lru_cache(maxsize=256)
def _pil(path: Path) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), 100)  # widths scale linearly: measured at 100, scaled to the size


def text_width(text: str, path: Path, size: float, spacing: float = 0.0) -> float:
    """Width in points of `text` at `size` pt with `spacing` pt between letters."""
    return _pil(path).getlength(text) * size / 100 + spacing * len(text)


def wrapped_lines(
    text: str, path: Path, size: float, width_pt: float, spacing: float = 0.0
) -> tuple[int, bool]:
    """(lines a paragraph wraps into, a single word wider than the box) — greedy wrap like PowerPoint's."""
    lines, too_wide, current = 0, False, ""
    for word in text.split(" "):
        trial = f"{current} {word}" if current else word
        if text_width(trial, path, size, spacing) <= width_pt or not current:
            too_wide = too_wide or text_width(word, path, size, spacing) > width_pt
            current = trial
        else:
            lines += 1
            current = word
    return lines + 1, too_wide


# ---- EOT and PPTX embedding -------------------------------------------------------------------------

_EOT_VERSION = 0x00020001
_MAGIC = 0x504C


def _name(font: TTFont, name_id: int) -> str:
    record = font["name"].getName(name_id, 3, 1, 0x409) or font["name"].getName(name_id, 1, 0, 0)
    return str(record) if record else ""


def to_eot(ttf: bytes) -> bytes:
    """Uncompressed EOT 2.1 (PowerPoint's .fntdata): a header from the font's own tables, then the TTF."""
    font = TTFont(io.BytesIO(ttf), lazy=True)
    os2, head = font["OS/2"], font["head"]
    p = os2.panose
    panose = bytes(
        [
            p.bFamilyType,
            p.bSerifStyle,
            p.bWeight,
            p.bProportion,
            p.bContrast,
            p.bStrokeVariation,
            p.bArmStyle,
            p.bLetterForm,
            p.bMidline,
            p.bXHeight,
        ]
    )

    def string(value: str) -> bytes:
        data = value.encode("utf-16-le")
        return struct.pack("<H", len(data)) + data

    names = (
        b"".join(struct.pack("<H", 0) + string(_name(font, nid)) for nid in (1, 2, 5, 4))
        + struct.pack("<H", 0)
        + struct.pack("<H", 0)
    )  # padding5 + empty RootString
    fixed = (
        struct.pack("<LLL", 0, len(ttf), _EOT_VERSION)
        + struct.pack("<L", 0)
        + panose
        + struct.pack("<BBLHH", 1, 1 if os2.fsSelection & 1 else 0, os2.usWeightClass, os2.fsType, _MAGIC)
        + struct.pack(
            "<LLLL", os2.ulUnicodeRange1, os2.ulUnicodeRange2, os2.ulUnicodeRange3, os2.ulUnicodeRange4
        )
        + struct.pack("<LL", os2.ulCodePageRange1, os2.ulCodePageRange2)
        + struct.pack("<L", head.checkSumAdjustment)
        + struct.pack("<LLLL", 0, 0, 0, 0)
    )
    header = fixed + names
    size = len(header) + len(ttf)
    return struct.pack("<L", size) + header[4:] + ttf


_FONT_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/font"


def embed_fonts(pptx: bytes, used: dict[str, set[bool]], root: Path = FONTS_DIR) -> bytes:
    """Put the catalog fonts a deck uses into the PPTX: family name -> {False (regular), True (bold)}."""
    entries, parts, rels = [], {}, []
    for n, name in enumerate(sorted(used), 1):
        fam = resolve(name, root)
        if fam is None:
            continue
        slots = []
        for bold in sorted(used[name] | {False}):
            if bold and BOLD not in fam.files:
                continue  # PowerPoint makes a faux bold from the regular face
            rid = f"rIdFont{n}{'b' if bold else 'r'}"
            parts[f"ppt/fonts/font{n}{'b' if bold else 'r'}.fntdata"] = to_eot(fam.file(bold).read_bytes())
            rels.append(
                f'<Relationship Id="{rid}" Type="{_FONT_REL}" '
                f'Target="fonts/font{n}{"b" if bold else "r"}.fntdata"/>'
            )
            slots.append(f'<p:{"bold" if bold else "regular"} r:id="{rid}"/>')
        entries.append(
            f'<p:embeddedFont><p:font typeface="{fam.name}" charset="0"/>{"".join(slots)}</p:embeddedFont>'
        )
    if not entries:
        return pptx
    src, out = zipfile.ZipFile(io.BytesIO(pptx)), io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as dst:
        for item in src.infolist():
            data = src.read(item.filename)
            if item.filename == "[Content_Types].xml" and b'Extension="fntdata"' not in data:
                data = data.replace(
                    b"<Default ",
                    b'<Default Extension="fntdata" ContentType="application/x-fontdata"/><Default ',
                    1,
                )
            elif item.filename == "ppt/_rels/presentation.xml.rels":
                data = data.replace(b"</Relationships>", "".join(rels).encode() + b"</Relationships>")
            elif item.filename == "ppt/presentation.xml":
                text = data.decode("utf-8")
                text = re.sub(r"<p:presentation\b", '<p:presentation embedTrueTypeFonts="1"', text, count=1)
                block = f"<p:embeddedFontLst>{''.join(entries)}</p:embeddedFontLst>"
                # Schema order: ..., notesSz, smartTags, embeddedFontLst, custShowLst, ..., defaultTextStyle
                text = (
                    text.replace("<p:defaultTextStyle", block + "<p:defaultTextStyle", 1)
                    if ("<p:defaultTextStyle" in text)
                    else text.replace("</p:presentation>", block + "</p:presentation>")
                )
                data = text.encode("utf-8")
            dst.writestr(item, data)
        for name, blob in parts.items():
            dst.writestr(name, blob)
    return out.getvalue()


def font_faces(used: dict[str, set[bool]], root: Path = FONTS_DIR) -> str:
    """@font-face rules with the used fonts inline, so the preview shows the deck's own typography offline."""
    rules = []
    for name, weights in used.items():
        fam = resolve(name, root)
        if fam is None:
            continue
        for bold in sorted(weights | {False}):
            if bold and BOLD not in fam.files:
                continue
            data = base64.b64encode(fam.file(bold).read_bytes()).decode()
            rules.append(
                f"@font-face{{font-family:'{fam.name}';font-weight:{700 if bold else 400};"
                f"src:url(data:font/ttf;base64,{data}) format('truetype')}}"
            )
    return "".join(rules)
