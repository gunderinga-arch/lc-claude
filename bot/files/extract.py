"""Extract plain text from uploaded files for the model prompt."""

from __future__ import annotations

import contextlib
import csv
import io
import os
import shutil
import string
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from xml.etree import ElementTree as ET

from pypdf import PdfReader

# Text taken from one file. The model gets as much of it as its window holds (~1.5–2 million chars of Russian
# for a ~1M-token model); the rest it reads with its file tools (bot.services.filetools). The cap only keeps
# memory sane: a 20 MB upload (Telegram's limit for bots) rarely holds more text.
MAX_TEXT_CHARS = 20_000_000
MAX_ZIP_MEMBER_BYTES = 50 * 1024 * 1024  # zip-bomb guard

_W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_S_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"

_SUPPORTED_HINT = (
    "Поддерживаются: PDF, DOCX, XLSX, PPTX, ZIP-архивы с ними, "
    "текстовые файлы и код (txt, md, csv, json, py …)."
)
_A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"

# A zip of a project or a folder of documents: every text-like member, each under its path.
ZIP_MAX_MEMBERS = 5_000
ZIP_MAX_TOTAL_BYTES = 300 * 1024 * 1024
_ZIP_SKIP_DIRS = (
    "__MACOSX/", ".git/", "node_modules/", "__pycache__/", ".venv/", "venv/", ".idea/", ".vscode/", ".next/",
    "dist/", "build/", "target/", ".gradle/", "vendor/",
)  # fmt: skip
_ZIP_SKIP_EXT = frozenset((
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".ico", ".tif", ".tiff", ".psd", ".svgz",
    ".mp3", ".mp4", ".wav", ".ogg", ".avi", ".mov", ".mkv", ".webm",
    ".zip", ".rar", ".7z", ".gz", ".tgz", ".bz2", ".xz", ".tar", ".jar", ".war",
    ".class", ".pyc", ".pyo", ".so", ".dll", ".dylib", ".exe", ".bin", ".o", ".a", ".lib",
    ".woff", ".woff2", ".ttf", ".otf", ".eot",
    ".db", ".sqlite", ".sqlite3", ".pkl", ".npy", ".npz", ".pt", ".onnx",
    ".doc", ".xls", ".ppt",
))  # fmt: skip

_KNOWN_BINARY = {
    ".doc": "DOC",
    ".xls": "XLS",
    ".ppt": "PPT",
    ".rar": "RAR",
    ".7z": "7Z",
    ".exe": "EXE",
    ".mp3": "MP3",
    ".mp4": "MP4",
}

_CONVERT_HINT = {
    ".doc": "Сохраните файл как .docx и отправьте снова.",
    ".xls": "Сохраните файл как .xlsx или CSV и отправьте снова.",
    ".ppt": "Сохраните файл как .pptx и отправьте снова.",
}


class UnsupportedFile(Exception):
    """str(exc) is a short Russian message shown to the user as is."""


class ScannedPdf(UnsupportedFile):
    """A PDF without a text layer: its pages can still be read as pictures (bot.files.ocr)."""


@dataclass(frozen=True, slots=True)
class ExtractedText:
    text: str
    kind: str  # "pdf" | "docx" | "xlsx" | "pptx" | "text"
    truncated: bool  # True if cut to MAX_TEXT_CHARS


def extract_text(filename: str, data: bytes) -> ExtractedText:
    """Extract plain text from uploaded file bytes or raise UnsupportedFile."""
    ext = _extension(filename)
    if ext == ".pdf" or data.startswith(b"%PDF"):
        text = _extract_pdf(data)
        kind = "pdf"
    elif ext == ".docx":
        text = _extract_docx(data)
        kind = "docx"
    elif ext == ".xlsx":
        text = _extract_xlsx(data)
        kind = "xlsx"
    elif ext == ".pptx":
        text = _extract_pptx(data)
        kind = "pptx"
    elif ext == ".zip":
        text = _extract_zip(data)
        kind = "zip"
    elif ext in _KNOWN_BINARY:
        raise UnsupportedFile(_binary_message(ext))
    else:
        text = _extract_text_file(data)
        kind = "text"
    return _finalize(text, kind)


def _extension(filename: str) -> str:
    name = filename.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    if "." not in name:
        return ""
    return "." + name.rsplit(".", 1)[-1].lower()


def _binary_message(ext: str) -> str:
    fmt = _KNOWN_BINARY[ext]
    hint = _CONVERT_HINT.get(ext, "Конвертируйте файл в поддерживаемый формат.")
    return f"Формат {fmt} не поддерживается. {hint} {_SUPPORTED_HINT}"


def _finalize(text: str, kind: str) -> ExtractedText:
    text = text.replace("\r\n", "\n").rstrip()
    truncated = len(text) > MAX_TEXT_CHARS
    if truncated:
        text = text[:MAX_TEXT_CHARS]
    return ExtractedText(text=text, kind=kind, truncated=truncated)


# --- PDF -------------------------------------------------------------------


PDFTOTEXT_TIMEOUT_S = 300


def _pdftotext_pages(data: bytes) -> list[str] | None:
    """Page texts by poppler's pdftotext (the server has it): a book of a thousand pages takes seconds, where
    pypdf takes minutes. None when it is missing or fails (a password, a broken file): pypdf decides then.
    """
    exe = shutil.which("pdftotext")
    if exe is None:
        return None
    fd, path = tempfile.mkstemp(suffix=".pdf")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        run = subprocess.run(
            [exe, "-enc", "UTF-8", "-q", path, "-"],
            capture_output=True,
            timeout=PDFTOTEXT_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    finally:
        with contextlib.suppress(OSError):
            os.unlink(path)
    if run.returncode != 0:
        return None
    pages = [page.strip() for page in run.stdout.decode("utf-8", errors="replace").split("\f")]
    while pages and not pages[-1]:
        pages.pop()
    return pages if any(pages) else None


def _extract_pdf(data: bytes) -> str:
    pages = _pdftotext_pages(data)
    if pages is not None:
        return _join_pages(pages)
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception as exc:
                raise UnsupportedFile("PDF защищён паролем, снимите защиту и отправьте файл снова.") from exc
            if reader.is_encrypted:
                raise UnsupportedFile("PDF защищён паролем, снимите защиту и отправьте файл снова.")
        pages = [(page.extract_text() or "").strip() for page in reader.pages]
    except UnsupportedFile:
        raise
    except Exception as exc:
        raise UnsupportedFile(
            "Не удалось прочитать PDF: файл повреждён или имеет неподдерживаемый формат."
        ) from exc
    if not any(pages):
        raise ScannedPdf(
            "В этом PDF нет текстового слоя (похоже, это скан). Пришлите страницы как фотографии."
        )
    return _join_pages(pages)


def _join_pages(pages: list[str]) -> str:
    return "\n\n".join(f"--- страница {i} ---\n\n{p}" for i, p in enumerate(pages, 1))


# --- DOCX --------------------------------------------------------------------


def _extract_docx(data: bytes) -> str:
    xml = _read_zip_member(data, "word/document.xml", "DOCX")
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as exc:
        raise UnsupportedFile("Не удалось прочитать DOCX: файл повреждён.") from exc
    body = root.find(f"{{{_W_NS}}}body")
    if body is None:
        raise UnsupportedFile("Не удалось прочитать DOCX: файл повреждён.")
    lines: list[str] = []
    for child in body:
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            lines.append(_docx_paragraph_text(child))
        elif tag == "tbl":
            lines.extend(_docx_table_lines(child))
    return "\n".join(lines)


def _docx_paragraph_text(p: ET.Element) -> str:
    parts: list[str] = []
    for node in p.iter():
        tag = node.tag.rsplit("}", 1)[-1]
        if tag == "t":
            parts.append(node.text or "")
        elif tag == "tab":
            parts.append("\t")
        elif tag == "br":
            parts.append("\n")
    return "".join(parts)


def _docx_table_lines(tbl: ET.Element) -> list[str]:
    lines: list[str] = []
    for tr in tbl.iter(f"{{{_W_NS}}}tr"):
        cells = []
        for tc in tr.findall(f"{{{_W_NS}}}tc"):
            texts = [_docx_paragraph_text(p) for p in tc.findall(f"{{{_W_NS}}}p")]
            cells.append("\n".join(texts))
        lines.append(" | ".join(cells))
    return lines


# --- XLSX --------------------------------------------------------------------


def _extract_xlsx(data: bytes) -> str:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            _guard_zip(zf)
            wb_xml = _zip_read_required(zf, "xl/workbook.xml", "XLSX")
            rels_xml = _zip_read_optional(zf, "xl/_rels/workbook.xml.rels")
            shared_xml = _zip_read_optional(zf, "xl/sharedStrings.xml")
            shared = _parse_shared_strings(shared_xml) if shared_xml else []
            sheets = _parse_workbook_sheets(wb_xml, rels_xml)
            out: list[str] = []
            for name, path in sheets:
                sheet_xml = _zip_read_optional(zf, path)
                if sheet_xml is None:
                    continue
                out.append(f"## Лист: {name}")
                out.append(_render_sheet(sheet_xml, shared))
            return "\n".join(out)
    except UnsupportedFile:
        raise
    except zipfile.BadZipFile as exc:
        raise UnsupportedFile("Не удалось прочитать XLSX: файл повреждён.") from exc
    except ET.ParseError as exc:
        raise UnsupportedFile("Не удалось прочитать XLSX: файл повреждён.") from exc


def _parse_shared_strings(xml: bytes) -> list[str]:
    root = ET.fromstring(xml)
    result = []
    for si in root.findall(f"{{{_S_NS}}}si"):
        texts = [t.text or "" for t in si.iter(f"{{{_S_NS}}}t")]
        result.append("".join(texts))
    return result


def _parse_workbook_sheets(wb_xml: bytes, rels_xml: bytes | None) -> list[tuple[str, str]]:
    wb_root = ET.fromstring(wb_xml)
    rid_to_target: dict[str, str] = {}
    if rels_xml is not None:
        rels_root = ET.fromstring(rels_xml)
        for rel in rels_root:
            rid = rel.get("Id")
            target = rel.get("Target")
            if rid and target:
                target = "xl/" + target if not target.startswith("/") else target.lstrip("/")
                rid_to_target[rid] = target
    sheets = []
    sheets_el = wb_root.find(f"{{{_S_NS}}}sheets")
    if sheets_el is None:
        return sheets
    for sheet in sheets_el.findall(f"{{{_S_NS}}}sheet"):
        name = sheet.get("name", "")
        rid = sheet.get(f"{{{_R_NS}}}id")
        target = rid_to_target.get(rid, "") if rid else ""
        if target:
            sheets.append((name, target))
    return sheets


MAX_XLSX_COLUMNS = 16_384  # Excel's own limit (column XFD)


def _col_letters_to_index(ref: str) -> int:
    """Convert the column letters of a cell reference like 'C5' to a 0-based index."""
    letters = "".join(ch for ch in ref if ch in string.ascii_letters)
    idx = 0
    for ch in letters.upper():
        idx = idx * 26 + (ord(ch) - ord("A") + 1)
    # A forged reference like "ZZZZZZZ1" would otherwise build a row of billions of fields.
    if idx > MAX_XLSX_COLUMNS:
        raise UnsupportedFile("Не удалось прочитать XLSX: файл повреждён.")
    return idx - 1


def _render_sheet(xml: bytes, shared: list[str]) -> str:
    root = ET.fromstring(xml)
    buf = io.StringIO()
    writer = csv.writer(buf)
    for row in root.iter(f"{{{_S_NS}}}row"):
        cells: dict[int, str] = {}
        max_idx = -1
        for c in row.findall(f"{{{_S_NS}}}c"):
            ref = c.get("r", "")
            idx = _col_letters_to_index(ref) if ref else max_idx + 1
            cells[idx] = _cell_value(c, shared)
            max_idx = max(max_idx, idx)
        line = [cells.get(i, "") for i in range(max_idx + 1)]
        writer.writerow(line)
        if buf.tell() > MAX_TEXT_CHARS:  # the result is cut to this anyway: stop before memory blows up
            break
    return buf.getvalue().rstrip("\r\n")


def _cell_value(c: ET.Element, shared: list[str]) -> str:
    t = c.get("t")
    if t == "inlineStr":
        is_el = c.find(f"{{{_S_NS}}}is")
        if is_el is not None:
            return "".join(node.text or "" for node in is_el.iter(f"{{{_S_NS}}}t"))
        return ""
    v = c.find(f"{{{_S_NS}}}v")
    if v is None or v.text is None:
        return ""
    if t == "s":
        try:
            return shared[int(v.text)]
        except (ValueError, IndexError):
            return ""
    if t == "b":
        return "TRUE" if v.text == "1" else "FALSE"
    return v.text


# --- PPTX --------------------------------------------------------------------


def _slide_number(path: str) -> int:
    digits = "".join(ch for ch in path.rsplit("/", 1)[-1] if ch.isdigit())
    return int(digits or 0)


def _pptx_paragraphs(xml: bytes) -> list[str]:
    root = ET.fromstring(xml)
    lines = ["".join(t.text or "" for t in p.iter(f"{{{_A_NS}}}t")) for p in root.iter(f"{{{_A_NS}}}p")]
    return [line for line in lines if line.strip()]


def _extract_pptx(data: bytes) -> str:
    """Slide texts in slide order (file order, ppt/slides/slideN.xml), speaker notes after each slide."""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            _guard_zip(zf)
            names = zf.namelist()
            slides = [n for n in names if n.startswith("ppt/slides/slide") and n.endswith(".xml")]
            slides.sort(key=_slide_number)
            if not slides:
                raise UnsupportedFile("Не удалось прочитать PPTX: в файле нет слайдов.")
            out: list[str] = []
            for path in slides:
                num = _slide_number(path)
                out.append(f"--- слайд {num} ---")
                out.extend(_pptx_paragraphs(zf.read(path)))
                # ponytail: notes matched by file number (PowerPoint and python-pptx keep them equal), not via
                # the slide's .rels; read the rels if decks with reordered notes show up.
                notes = f"ppt/notesSlides/notesSlide{num}.xml"
                if notes in names:
                    # The notes page repeats the slide number as a field; real notes are the longer lines.
                    said = [line for line in _pptx_paragraphs(zf.read(notes)) if line.strip() != str(num)]
                    if said:
                        out.append("Заметки: " + " ".join(said))
            return "\n".join(out)
    except UnsupportedFile:
        raise
    except (zipfile.BadZipFile, ET.ParseError, KeyError) as exc:
        raise UnsupportedFile("Не удалось прочитать PPTX: файл повреждён.") from exc


# --- plain text --------------------------------------------------------------


def _extract_text_file(data: bytes) -> str:
    if b"\x00" in data[:8192]:
        raise UnsupportedFile(f"Не удалось распознать формат файла. {_SUPPORTED_HINT}")
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        pass
    try:
        return data.decode("cp1251")
    except UnicodeDecodeError as exc:
        raise UnsupportedFile(f"Не удалось распознать формат файла. {_SUPPORTED_HINT}") from exc


# --- ZIP archives ----------------------------------------------------------------


def _extract_zip(data: bytes) -> str:
    """Every readable member under a "=== path ===" line; pictures, binaries, nested archives only listed."""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            members = [i for i in zf.infolist() if not i.is_dir()]
            if len(members) > ZIP_MAX_MEMBERS or sum(i.file_size for i in members) > ZIP_MAX_TOTAL_BYTES:
                raise UnsupportedFile(
                    f"Архив слишком большой: не больше {ZIP_MAX_MEMBERS} файлов и "
                    f"{ZIP_MAX_TOTAL_BYTES // 2**20} МБ в распакованном виде."
                )
            parts: list[str] = []
            skipped: list[str] = []
            size = 0
            for info in members:
                path = info.filename
                if any(f"/{d}" in f"/{path}" for d in _ZIP_SKIP_DIRS):
                    continue
                if _extension(path) in _ZIP_SKIP_EXT or info.file_size > MAX_ZIP_MEMBER_BYTES:
                    skipped.append(path)
                    continue
                if size > MAX_TEXT_CHARS:  # the rest would be cut off anyway
                    skipped.append(path)
                    continue
                try:
                    text = extract_text(path, zf.read(info)).text
                except UnsupportedFile:
                    skipped.append(path)
                    continue
                parts.append(f"=== {path} ===\n{text}")
                size += len(parts[-1])
            if not parts:
                raise UnsupportedFile("В архиве нет файлов, которые я умею читать. " + _SUPPORTED_HINT)
            if skipped:
                listed = "\n".join(skipped[:300]) + (
                    f"\n… и ещё {len(skipped) - 300}" if len(skipped) > 300 else ""
                )
                parts.append(f"=== не прочитаны (картинки, двоичные, вложенные архивы) ===\n{listed}")
    except UnsupportedFile:
        raise
    except (zipfile.BadZipFile, RuntimeError) as exc:  # RuntimeError: an encrypted member
        raise UnsupportedFile("Не удалось прочитать ZIP: архив повреждён или защищён паролем.") from exc
    return "\n\n".join(parts)


# --- zip helpers ---------------------------------------------------------------


def _guard_zip(zf: zipfile.ZipFile) -> None:
    members = zf.infolist()
    if len(members) > ZIP_MAX_MEMBERS or sum(i.file_size for i in members) > ZIP_MAX_TOTAL_BYTES:
        raise UnsupportedFile("Файл слишком большой или повреждён (превышен предел распаковки).")
    for info in members:
        if info.file_size > MAX_ZIP_MEMBER_BYTES:
            raise UnsupportedFile("Файл слишком большой или повреждён (превышен предел распаковки).")


def _zip_read_required(zf: zipfile.ZipFile, name: str, fmt: str) -> bytes:
    data = _zip_read_optional(zf, name)
    if data is None:
        raise UnsupportedFile(
            f"Не удалось прочитать {fmt}: файл повреждён или имеет неподдерживаемую структуру."
        )
    return data


def _zip_read_optional(zf: zipfile.ZipFile, name: str) -> bytes | None:
    try:
        info = zf.getinfo(name)
    except KeyError:
        return None
    if info.file_size > MAX_ZIP_MEMBER_BYTES:
        raise UnsupportedFile("Файл слишком большой или повреждён (превышен предел распаковки).")
    return zf.read(name)


def _read_zip_member(data: bytes, name: str, fmt: str) -> bytes:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            _guard_zip(zf)
            return _zip_read_required(zf, name, fmt)
    except UnsupportedFile:
        raise
    except zipfile.BadZipFile as exc:
        raise UnsupportedFile(f"Не удалось прочитать {fmt}: файл повреждён.") from exc
