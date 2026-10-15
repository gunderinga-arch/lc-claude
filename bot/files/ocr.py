"""Scanned PDFs (no text layer): the pages as pictures, read out verbatim by a vision model.

Fable 5.1 receives pictures on BuyTokens (Opus 5 and Sol lose them, 14.10.2026) and transcribed 4 rendered
pages in 113 s word for word (15.10.2026): ~30 s a page, so batches of pages go out a few at a time.
"""

from __future__ import annotations

import asyncio
import io
import math
import shutil
from pathlib import Path

from pypdf import PdfReader

from bot.files.deck_qa import _run

OCR_DPI = 110  # small print stays legible, a page is ~200 KB of JPEG
OCR_PAGES_PER_CALL = 6
OCR_CONCURRENCY = 2  # shared bulk-work cap in ChatService (also used by file digests)
OCR_MAX_PAGES = 150  # ~38 minutes at concurrency 2; the rest of a longer scan is not read
OCR_SECONDS_PER_PAGE = 30
OCR_PROMPT = (
    "Это страницы {first}–{last} скана документа «{name}» (всего в нём {total} стр.). Перепиши весь текст "
    "с них дословно, в порядке чтения, ничего не добавляя и не пропуская; таблицы — строками через « | », "
    "формулы — как написаны. Перед каждой страницей строка «--- страница N ---», где N — её номер в "
    "документе (первая здесь — {first}). Если на странице нет текста, напиши после её строки «(без текста)». "
    "Только текст страниц, без вступления и пояснений."
)


def page_count(data: bytes) -> int:
    """Pages in the PDF; 0 when it cannot be read at all."""
    try:
        return len(PdfReader(io.BytesIO(data)).pages)
    except Exception:
        return 0


def can_read_scans() -> bool:
    return shutil.which("pdftoppm") is not None


def minutes_for(pages: int) -> int:
    """A fair estimate of the reading time for the user's notice."""
    return max(1, math.ceil(min(pages, OCR_MAX_PAGES) * OCR_SECONDS_PER_PAGE / OCR_CONCURRENCY / 60))


async def render_pages(pdf: Path, first: int, last: int) -> list[bytes]:
    """JPEG pictures of pages `first`..`last` (1-based) of the PDF file; [] if pdftoppm fails."""
    prefix = pdf.parent / f"b{first}"
    if not await _run(
        "pdftoppm", "-jpeg", "-r", str(OCR_DPI), "-f", str(first), "-l", str(last),
        str(pdf), str(prefix), cwd=str(pdf.parent),
    ):  # fmt: skip
        return []

    def collect() -> list[bytes]:
        files = sorted(pdf.parent.glob(f"{prefix.name}-*.jpg"))
        shots = [f.read_bytes() for f in files]
        for f in files:
            f.unlink()
        return shots

    return await asyncio.to_thread(collect)


def batches(pages: int) -> list[tuple[int, int]]:
    """(first, last) page ranges of OCR_PAGES_PER_CALL, up to OCR_MAX_PAGES."""
    pages = min(pages, OCR_MAX_PAGES)
    return [(f, min(f + OCR_PAGES_PER_CALL - 1, pages)) for f in range(1, pages + 1, OCR_PAGES_PER_CALL)]
