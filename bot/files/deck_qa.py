"""See a built deck the way the user will: LibreOffice draws the PPTX, the slides go to the model as pictures.

The model then reviews its own layout (overflow, overlaps, contrast, rhythm, the references' style) and
returns a corrected deck. Without LibreOffice on the machine the review works from the automatic warnings
alone.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import shutil
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)

RENDER_TIMEOUT_S = 120
SLIDE_W, SLIDE_H = 800, 450  # one slide on a sheet
COLS, ROWS = 2, 3  # slides per sheet: enough detail to spot a cut word, few pictures per review
_render_lock = asyncio.Lock()  # one LibreOffice at a time: it is heavy and its profile is not concurrent

REVIEW_PROMPT = (
    "Ты — арт-директор. Выше — снимки слайдов презентации, собранной из твоего JSON ровно так, "
    "PowerPoint (номер слайда в углу){refs}. Проверь каждый слайд: текст вылезает, обрезан или налезает на "
    "другие элементы; элементы уехали за край, слиплись или стоят криво относительно сетки; слабый контраст; "
    "пустые или перегруженные места; заголовки одного типа слайдов стоят по-разному; стиль расходится "
    "с референсами (цвета, шрифты, пропорции, композиция). Автоматическая проверка нашла: {warnings}\n\n"
    "Исправь всё найденное и верни JSON презентации целиком одним блоком "
    "без пояснений. Если исправлять нечего — ответь одним словом OK."
)


async def _run(*cmd: str, cwd: str) -> bool:
    # The bot's unit makes its home read-only (ProtectSystem=strict): caches and profiles go to the temp dir.
    env = {**os.environ, "HOME": cwd, "XDG_CACHE_HOME": cwd, "XDG_CONFIG_HOME": cwd}
    proc = await asyncio.create_subprocess_exec(
        *cmd, cwd=cwd, env=env, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL
    )
    try:
        await asyncio.wait_for(proc.wait(), RENDER_TIMEOUT_S)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        return False
    return proc.returncode == 0


def _sheets(pngs: list[Path]) -> list[bytes]:
    """Slides side by side, numbered, COLS × ROWS per sheet, as PNG."""
    out, per = [], COLS * ROWS
    for start in range(0, len(pngs), per):
        chunk = pngs[start : start + per]
        rows = -(-len(chunk) // COLS)
        sheet = Image.new(
            "RGB", (COLS * SLIDE_W + (COLS + 1) * 10, rows * SLIDE_H + (rows + 1) * 10), "#808080"
        )
        draw = ImageDraw.Draw(sheet)
        for i, path in enumerate(chunk):
            x, y = 10 + (i % COLS) * (SLIDE_W + 10), 10 + (i // COLS) * (SLIDE_H + 10)
            with Image.open(path) as img:
                sheet.paste(img.convert("RGB").resize((SLIDE_W, SLIDE_H)), (x, y))
            draw.rectangle((x, y, x + 34, y + 22), fill="#FF00FF")
            draw.text((x + 6, y + 5), str(start + i + 1), fill="#FFFFFF")
        buf = io.BytesIO()
        sheet.save(buf, "PNG", optimize=True)
        out.append(buf.getvalue())
    return out


async def render_sheets(pptx: bytes) -> list[bytes]:
    """PNG sheets of the deck as LibreOffice renders it; [] if LibreOffice or pdftoppm is missing or fails."""
    if not shutil.which("soffice") or not shutil.which("pdftoppm"):
        return []
    async with _render_lock:
        with tempfile.TemporaryDirectory(prefix="deck-") as tmp:
            deck = Path(tmp) / "deck.pptx"
            deck.write_bytes(pptx)
            converted = await _run(
                "soffice", f"-env:UserInstallation=file://{tmp}/profile", "--headless",
                "--convert-to", "pdf", "--outdir", tmp, str(deck), cwd=tmp,
            )  # fmt: skip
            pdf = Path(tmp) / "deck.pdf"
            if not converted or not pdf.exists():
                logger.warning("LibreOffice could not render a deck")
                return []
            if not await _run(
                "pdftoppm",
                "-png",
                "-scale-to-x",
                str(SLIDE_W),
                "-scale-to-y",
                "-1",
                str(pdf),
                str(Path(tmp) / "s"),
                cwd=tmp,
            ):
                return []
            return await asyncio.to_thread(lambda: _sheets(sorted(Path(tmp).glob("s-*.png"))))


def review_prompt(name: str, warnings: tuple[str, ...], with_refs: bool) -> str:
    refs = " и картинки-референсы пользователя (стиль надо повторить)" if with_refs else ""
    found = "; ".join(warnings) if warnings else "ничего."
    return REVIEW_PROMPT.format(refs=refs, warnings=found, name=name)
