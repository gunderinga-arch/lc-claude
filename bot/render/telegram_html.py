"""Render the LLM's markdown answer as Telegram-safe HTML chunks.

Telegram messages are limited to 4096 chars and only support a small HTML subset:
<b>, <i>, <u>, <s>, <code>, <pre>, <a>, <blockquote>, <tg-spoiler>.
Tables and headings from the model are flattened to plain text.
Splitting prefers paragraph boundaries and never breaks inside a code block.
"""

from __future__ import annotations

import re
from html import escape

TELEGRAM_LIMIT = 4096

_FENCE_RE = re.compile(r"^```(\w*)\s*$")
_INLINE_RE = re.compile(
    r"""
    (?P<link>\[(?P<link_text>[^\]]+)\]\((?P<link_url>https?://[^)\s]+)\))
    | `(?P<code>[^`\n]+)`
    | \*\*(?P<bold>[^*]+)\*\*
    | (?<![\w*])\*(?P<italic>[^*\n]+)\*(?![\w*])
    | ~~(?P<strike>[^~]+)~~
    """,
    re.VERBOSE,
)
_LIST_BULLET_RE = re.compile(r"^(\s*)[-*+]\s+")
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+")


def _render_inline(text: str) -> str:
    """Escape HTML and convert inline markdown of one line to Telegram HTML."""
    out: list[str] = []
    pos = 0
    for match in _INLINE_RE.finditer(text):
        out.append(escape(text[pos : match.start()]))
        if match.group("link"):
            url = escape(match.group("link_url"), quote=True)
            out.append(f'<a href="{url}">{escape(match.group("link_text"))}</a>')
        elif match.group("code") is not None:
            out.append(f"<code>{escape(match.group('code'))}</code>")
        elif match.group("bold") is not None:
            out.append(f"<b>{escape(match.group('bold'))}</b>")
        elif match.group("italic") is not None:
            out.append(f"<i>{escape(match.group('italic'))}</i>")
        else:
            out.append(f"<s>{escape(match.group('strike'))}</s>")
        pos = match.end()
    out.append(escape(text[pos:]))
    return "".join(out)


def _render_block(lines: list[str], in_code: bool) -> str:
    if in_code:
        return "\n".join(escape(line) for line in lines)
    rendered: list[str] = []
    for line in lines:
        stripped = line.strip()
        if _HEADING_RE.match(line):
            rendered.append(f"<b>{_render_inline(_HEADING_RE.sub('', stripped))}</b>")
        elif stripped == "":
            rendered.append("")
        elif bullet := _LIST_BULLET_RE.match(line):
            indent = " " * len(bullet.group(1))
            rendered.append(f"{indent}• {_render_inline(line[bullet.end() :])}")
        elif set(stripped) <= {"-", "|", ":", " "}:
            continue  # markdown table separator row
        elif stripped.startswith("|") and stripped.endswith("|"):
            cells = [cell.strip() for cell in stripped.strip("|").split("|")]
            rendered.append(" | ".join(_render_inline(cell) for cell in cells))
        else:
            rendered.append(_render_inline(line))
    return "\n".join(rendered)


def markdown_to_html(text: str) -> str:
    """Convert the model's markdown answer to a single Telegram HTML string."""
    lines = text.split("\n")
    out: list[str] = []
    in_code = False
    for line in lines:
        if _FENCE_RE.match(line.strip()):
            in_code = not in_code
            out.append("<pre>" if in_code else "</pre>")
        else:
            out.append(_render_block([line], in_code))
    if in_code:  # unclosed fence: close the tag so Telegram accepts the message
        out.append("</pre>")
    return "\n".join(out)


def _split_long_plain_line(line: str, limit: int) -> list[str]:
    """Hard-wrap one oversized plain-text line (split on spaces when possible)."""
    parts: list[str] = []
    while len(line) > limit:
        cut = line.rfind(" ", 0, limit + 1)
        if cut <= 0:
            cut = limit
        parts.append(line[:cut])
        line = line[cut:].lstrip(" ")
    parts.append(line)
    return parts


def split_html(html: str, limit: int = TELEGRAM_LIMIT) -> list[str]:
    """Split a rendered HTML answer into Telegram-sized chunks.

    Paragraphs are preferred as split points. Code blocks are re-opened/closed
    around a split so every chunk is valid HTML.
    """
    if len(html) <= limit:
        return [html]

    lines = html.split("\n")
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    in_code = False

    def flush() -> None:
        nonlocal current, current_len
        if not current:
            return
        text = "\n".join(current)
        if in_code:
            text += "\n</pre>"
        chunks.append(text)
        current = ["<pre>"] if in_code else []
        current_len = len(current[0]) + 1 if current else 0

    for line in lines:
        if line == "<pre>":
            in_code = True
        elif line == "</pre>":
            in_code = False
            line = "\n</pre>"  # lead with a newline so it lands on its own line after a split
        extra = len(line) + (1 if current else 0)
        if current_len + extra > limit:
            if current:
                flush()
            if len(line) > limit and not in_code:
                for piece in _split_long_plain_line(line, limit):
                    if current_len + len(piece) + (1 if current else 0) > limit:
                        flush()
                    current.append(piece)
                    current_len += len(piece) + 1
                continue
        current.append(line)
        current_len += extra

    if current:
        text = "\n".join(current)
        if in_code and not text.endswith("</pre>"):
            text += "\n</pre>"
        chunks.append(text)
    return [chunk for chunk in chunks if chunk.strip()]


def render_markdown(text: str, limit: int = TELEGRAM_LIMIT) -> list[str]:
    """Markdown answer -> list of Telegram HTML chunks, each within `limit`. Blank text -> []."""
    if not text.strip():
        return []
    html = markdown_to_html(text)
    return split_html(html, limit) or [escape(text)[:limit]]
