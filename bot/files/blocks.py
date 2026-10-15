"""Files the model sends back: <<<FILE name="x.ext">>> ... <<<END>>> blocks in its answer.

The protocol was verified on the live gateway: Luna, Sol, Opus 4.8 and Fable 5 all follow it.
"""

from __future__ import annotations

import base64
import re
from collections.abc import Mapping
from dataclasses import dataclass

FILE_INSTRUCTIONS = (
    "Если пользователь просит создать, изменить или переделать файл, верни каждый файл блоком строго такого "
    'вида (маркеры на отдельных строках):\n<<<FILE name="имя.расширение">>>\nсодержимое файла\n<<<END>>>\n'
    "Бот пришлёт такой блок пользователю отдельным документом. Можно только текстовые форматы: txt, md, csv, "
    "json, код, html, svg, xml, yaml и т. п., а также презентации PowerPoint (как — ниже). "
    "Word, Excel, PDF и картинки создать нельзя: предложи текстовый аналог (CSV вместо Excel, "
    "Markdown вместо Word). Вне блоков пиши обычный ответ, содержимое файла там не повторяй."
)

_BLOCK = re.compile(r'^<<<FILE name="([^"\n]{1,200})">>>\n(.*?)\n?<<<END>>>[ \t]*$', re.S | re.M)
_UNSAFE = re.compile(r"[^\w.\- ()]+")
_HTML_FENCE = re.compile(r"^```html[ \t]*\n(.*?)\n```[ \t]*$", re.S | re.M | re.I)
_FULL_PAGE = re.compile(r"<!doctype html|<html[\s>]", re.I)
_IMAGE_EXT = r"(?:\.(?:jpe?g|png|webp))?"
MAX_FILES = 10  # blocks beyond this stay in the text
MAX_NAME = 64
BOM_EXTENSIONS = (".csv",)  # Excel on Windows needs the BOM to read UTF-8 CSV correctly


@dataclass(frozen=True, slots=True)
class OutFile:
    name: str
    data: bytes


def safe_filename(name: str) -> str:
    """Basename only, harmless characters, bounded length, extension kept."""
    base = re.split(r"[\\/]", name)[-1]
    base = _UNSAFE.sub("_", base).strip(" .")
    if not base:
        return "file.txt"
    if len(base) > MAX_NAME:
        stem, dot, ext = base.rpartition(".")
        base = f"{stem[: MAX_NAME - len(ext) - 1]}.{ext}" if dot and len(ext) <= 10 else base[:MAX_NAME]
    return base


def split_files(text: str) -> tuple[str, list[OutFile]]:
    """Cut file blocks out of the answer. Each block leaves a '📎 name' line in its place.

    A whole HTML page in a plain ```html fence (models sometimes forget the protocol) becomes a file too:
    as chat text it is unreadable and gets cut by Telegram's length limit.
    """
    files: list[OutFile] = []
    pages = 0

    def promote(match: re.Match[str]) -> str:
        nonlocal pages
        body = match.group(1)
        if not _FULL_PAGE.search(body):
            return match.group(0)
        pages += 1
        name = "page.html" if pages == 1 else f"page-{pages}.html"
        return f'<<<FILE name="{name}">>>\n{body}\n<<<END>>>'

    if "<<<FILE" not in text:  # a fence inside a real file block is that file's content
        text = _HTML_FENCE.sub(promote, text)

    def take(match: re.Match[str]) -> str:
        if len(files) >= MAX_FILES:
            return match.group(0)
        name = safe_filename(match.group(1))
        encoding = "utf-8-sig" if name.lower().endswith(BOM_EXTENSIONS) else "utf-8"
        files.append(OutFile(name, match.group(2).encode(encoding)))
        return f"📎 {name}"

    rest = _BLOCK.sub(take, text)
    return rest.strip(), files


def embed_images(html: str, images: Mapping[str, bytes]) -> str:
    """Replace references to generated images ("hero.jpg", or just "hero") with data: URIs.

    The HTML then works as one file, offline. Only src="…", href="…" and url(…) are touched.
    """
    for filename, data in images.items():
        stem, _, ext = filename.rpartition(".")
        mime = "image/png" if ext == "png" else "image/jpeg"
        uri = f"data:{mime};base64,{base64.b64encode(data).decode()}"
        name = re.escape(stem) + _IMAGE_EXT
        html = re.sub(
            rf"""((?:src|href)\s*=\s*)(["']){name}\2""", lambda m, uri=uri: f"{m[1]}{m[2]}{uri}{m[2]}", html
        )
        html = re.sub(
            rf"""url\(\s*(["']?){name}\1\s*\)""", lambda m, uri=uri: f"url({m[1]}{uri}{m[1]})", html
        )
    return html
