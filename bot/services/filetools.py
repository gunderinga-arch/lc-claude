"""read_file / search_file / digest_file: the model works with a chat's files bigger than its context.

Files go into the message whole while they fit the model's window (~1.5–2 million chars of Russian for a
~1M-token model); a bigger one goes in cut, and its full text waits in the `FileStore`. These tools give the
model any part of it by character offset, the fragments that best match a query, or — for a task about the
whole file (a summary, notes on a textbook) — a digest: the bot reads the file in big parts with a model and
hands back what each part holds for the task.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field

from bot.domain import approx_tokens
from bot.files.store import FileStore
from bot.llm.base import ChatMessage, ChatStatus, LLMClient, ToolCall, ToolSpec

READ_DEFAULT_CHARS = 100_000
READ_MAX_CHARS = 200_000  # ~50–100k tokens: every later tool round resends it
SEARCH_MAX_CHARS = 40_000
PAGE_CHARS = 1_500  # search granularity: one fragment, labelled with its char offsets
# Files from this size are kept for the tools even when they fit the message: the history may drop them later.
STORE_MIN_CHARS = 30_000
# digest_file: parts of this many tokens, two at a time (the main key takes ~10 requests a minute).
DIGEST_PART_TOKENS = 250_000
DIGEST_MAX_PARTS = 40  # ~10M tokens: a 20 MB text file; more is not read
DIGEST_CONCURRENCY = 2
DIGEST_MAX_CHARS = 150_000  # what the model gets back: the notes of all parts together
DIGEST_PROMPT = (
    "Ты читаешь часть {n} из {total} большого файла «{name}» (символы {start}–{end} из {size}). "
    "Другая модель "
    "решает по этому файлу задачу пользователя, но весь файл ей не виден: она увидит только твои выписки.\n"
    "Задача: {task}\n\n"
    "Выпиши из этой части всё, что нужно для задачи: суть, факты, цифры, даты, имена, определения, формулы, "
    "выводы, важные цитаты дословно; сохраняй порядок и заголовки разделов. "
    "Чем полнее для задачи, тем лучше, "
    "но без воды и пересказа ненужного. Если для задачи в части ничего нет — ответь одним символом «—».\n\n"
    "<part>\n{text}\n</part>"
)

FILE_TOOLS: tuple[ToolSpec, ...] = (
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Прочитать часть файла этого чата по номерам символов (с 0). Для больших файлов, которые не "
                f"поместились в переписку целиком. Не больше {READ_MAX_CHARS} символов за вызов."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Имя файла"},
                    "start": {"type": "integer", "description": "С какого символа (по умолчанию 0)"},
                    "length": {
                        "type": "integer",
                        "description": f"Сколько символов (по умолчанию {READ_DEFAULT_CHARS})",
                    },
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_file",
            "description": (
                "Найти в файле этого чата места, подходящие к запросу (по словам), с номерами символов, "
                "чтобы дочитать их read_file. Для больших файлов, которые не поместились в переписку."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Имя файла"},
                    "query": {"type": "string", "description": "Что искать: слова, имена, термины"},
                },
                "required": ["name", "query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "digest_file",
            "description": (
                "Для задач про весь большой файл (конспект, пересказ, анализ, все упоминания чего-то): бот "
                "прочитает файл целиком частями и вернёт по каждой части выписки под задачу. Долго (минуты) "
                "и дорого: вызывай один раз на задачу, только если файл не поместился в переписку."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Имя файла"},
                    "task": {
                        "type": "string",
                        "description": "Что нужно от всего файла — подробно, словами задачи пользователя",
                    },
                },
                "required": ["name", "task"],
            },
        },
    },
)
FILE_TOOL_NAMES = frozenset(t["function"]["name"] for t in FILE_TOOLS)

FILE_GUIDE = (
    "<files>\nУ этого чата есть сохранённые файлы с полным текстом: {names}.\n"
    "Если файл целиком есть в переписке — работай по ней. Если он показан не целиком "
    "(есть пометка, что дальше "
    "не поместилось) или выпал из переписки — читай нужные места read_file по номерам символов и ищи "
    "search_file. Если задача про весь такой файл (конспект, пересказ, анализ всего текста) — "
    "вызови один раз "
    "digest_file и работай по выпискам. Ничего не придумывай о частях, которых не читал и не получил "
    "в выписках.\n</files>"
)


def file_guide(names: list[str]) -> str:
    return FILE_GUIDE.format(names=", ".join(f"«{n}»" for n in names))


@dataclass(slots=True)
class FileTools:
    """The tools of one chat: its stored files; `llm` and `model` read them for digest_file."""

    store: FileStore
    user_id: int
    thread_id: int
    llm: LLMClient | None = None
    model: str | None = None
    spent: int = 0  # tokens digest_file sent and got back, for the request's accounting
    bulk_gate: asyncio.Semaphore = field(default_factory=lambda: asyncio.Semaphore(DIGEST_CONCURRENCY))
    digested: bool = False

    def names(self) -> list[str]:
        return self.store.names(self.user_id, self.thread_id)

    async def run(self, call: ToolCall) -> str:
        try:
            args = json.loads(call.arguments or "{}")
        except ValueError:
            args = None
        if not isinstance(args, dict) or not isinstance(args.get("name"), str):
            return f"Ошибка: у {call.name} обязателен name — имя файла."
        found = await asyncio.to_thread(self.store.read, self.user_id, self.thread_id, args["name"])
        if found is None:
            names = ", ".join(f"«{n}»" for n in self.names()) or "нет"
            return f"Ошибка: файла «{args['name']}» в этом чате нет. Сохранённые файлы: {names}."
        name, text = found
        if call.name == "read_file":
            return read_part(
                name, text, _int(args.get("start"), 0), _int(args.get("length"), READ_DEFAULT_CHARS)
            )
        if call.name == "digest_file":
            task = args.get("task")
            if not isinstance(task, str) or not task.strip():
                return "Ошибка: у digest_file обязателен task — что нужно от файла."
            if self.llm is None or self.model is None:
                return "Ошибка: выжимка сейчас недоступна, читай файл read_file и ищи search_file."
            if self.digested:
                return "Ошибка: одна выжимка за ответ. Используй выписки, read_file и search_file."
            self.digested = True
            return await self.digest(name, text, task)
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            return "Ошибка: у search_file обязателен query — что искать."
        return await asyncio.to_thread(search, name, text, query)

    async def digest(self, name: str, text: str, task: str) -> str:
        """The file read part by part for `task`; the notes of every part, in file order."""
        assert self.llm is not None and self.model is not None
        parts = (await asyncio.to_thread(split_parts, text, DIGEST_PART_TOKENS))[:DIGEST_MAX_PARTS]
        llm, model = self.llm, self.model

        async def one(n: int, start: int, end: int) -> str:
            ask = DIGEST_PROMPT.format(
                n=n,
                total=len(parts),
                name=name,
                start=start,
                end=end,
                size=len(text),
                task=task[:2000],
                text=text[start:end],
            )
            async with self.bulk_gate:
                result = await llm.chat([ChatMessage("user", ask, truncatable=False)], model)
            self.spent += (result.input_tokens if result.input_tokens is not None else approx_tokens(ask)) + (
                result.output_tokens if result.output_tokens is not None else approx_tokens(result.text)
            )
            notes = result.text.strip() if result.status is ChatStatus.OK else ""
            return f"### Часть {n} (символы {start}–{end})\n{notes or '(не прочиталась: модель не ответила)'}"

        notes = await asyncio.gather(*(one(n, s, e) for n, (s, e) in enumerate(parts, 1)))
        read_to = parts[-1][1] if parts else 0
        tail = (
            ""
            if read_to >= len(text)
            else f"\n\n[Прочитаны символы 0–{read_to} из {len(text)}: дальше не влезло.]"
        )
        body = "\n\n".join(notes)
        if len(body) > DIGEST_MAX_CHARS:
            body = body[:DIGEST_MAX_CHARS] + "\n\n[…выписки обрезаны по объёму]"
        return f"Выписки из файла «{name}» по задаче «{task[:300]}», {len(parts)} частей:\n\n{body}{tail}"


def split_parts(text: str, tokens: int) -> list[tuple[int, int]]:
    """(start, end) of consecutive parts of about `tokens` each, cut at paragraph or line ends."""
    out: list[tuple[int, int]] = []
    start = 0
    while start < len(text):
        chunk = text[start : start + tokens * 6]  # more than a part can hold: ~2–5 chars a token
        size = approx_tokens(chunk)
        end = start + (len(chunk) if size <= tokens else int(len(chunk) * tokens / size))
        if end < len(text):
            cut = text.rfind("\n", start + (end - start) // 2, end)
            end = cut + 1 if cut > start else end
        out.append((start, end))
        start = end
    return out


_FILE_BLOCK = re.compile(r'<file name="([^"\n]*)"([^>\n]*)>\n(.*?)\n</file>', re.S)
FILE_CUT_NOTE = (
    "\n\n[…Файл длиннее, чем помещается в переписку: выше символы 0–{shown} из {total}. "
    'Полный текст сохранён — читай дальше read_file(name="{name}", start={shown}) '
    'или ищи search_file(name="{name}", query=…).]'
)


def fit_files(text: str, budget: int) -> tuple[str, dict[str, str]]:
    """`text` with its <file> blocks cut to fit `budget` tokens, and the full texts to keep for the tools.

    Kept: every cut file and every file of STORE_MIN_CHARS or more. The room left besides the message's own
    words is shared out so that small files stay whole and the big ones get equal shares of what remains.
    """
    blocks = list(_FILE_BLOCK.finditer(text))
    keep = {m.group(1): m.group(3) for m in blocks if len(m.group(3)) >= STORE_MIN_CHARS}
    total = approx_tokens(text)
    if not blocks or total <= budget:
        return text, keep
    sizes = [approx_tokens(m.group(3)) for m in blocks]
    room = max(0, budget - (total - sum(sizes)))
    caps = [0] * len(blocks)
    for n, i in enumerate(sorted(range(len(blocks)), key=lambda k: sizes[k])):
        caps[i] = min(sizes[i], room // (len(blocks) - n))
        room -= caps[i]
    out, last = [], 0
    for i, m in enumerate(blocks):
        name, attrs, body = m.group(1), m.group(2), m.group(3)
        if sizes[i] <= caps[i]:
            continue
        chars = int(len(body) * caps[i] / sizes[i] * 0.97)
        line_end = body.rfind("\n", 0, chars)
        shown = line_end if line_end > chars * 0.8 else chars
        note = FILE_CUT_NOTE.format(shown=shown, total=len(body), name=name)
        out += [
            text[last : m.start()],
            f'<file name="{name}"{attrs} chars="{len(body)}">\n{body[:shown]}{note}\n</file>',
        ]
        last = m.end()
        keep[name] = body
    out.append(text[last:])
    return "".join(out), keep


def _int(value: object, default: int) -> int:
    try:
        return int(value)  # type: ignore[call-overload]
    except (TypeError, ValueError):
        return default


def read_part(name: str, text: str, start: int, length: int) -> str:
    total = len(text)
    start = min(max(0, start), total)
    end = min(total, start + max(1, min(length, READ_MAX_CHARS)))
    header = f"Файл «{name}», символы {start}–{end} из {total}:"
    more = f"\n\n[Дальше: read_file(name, start={end})]" if end < total else "\n\n[Конец файла]"
    return f"{header}\n\n{text[start:end]}{more}"


def pages(text: str) -> list[tuple[int, int, str]]:
    """The text in fragments of about PAGE_CHARS, cut at paragraph or line ends: (start, end, fragment)."""
    out = []
    start, total = 0, len(text)
    while start < total:
        end = min(total, start + PAGE_CHARS)
        if end < total:
            cut = max(
                text.rfind("\n\n", start + PAGE_CHARS // 2, end),
                text.rfind("\n", start + PAGE_CHARS // 2, end),
            )
            end = cut + 1 if cut > start else end
        out.append((start, end, text[start:end]))
        start = end
    return out


_WORD = re.compile(r"\w{3,}")


def _terms(text: str) -> list[str]:
    # a 6-letter prefix stands in for a stemmer, as in web
    return [w[:6] for w in _WORD.findall(text.lower())]


def search(name: str, text: str, query: str) -> str:
    """BM25 over the file's fragments that hold a query word at all; the best ones up to SEARCH_MAX_CHARS.

    The substring pre-check keeps it fast on tens of megabytes: only candidate fragments are tokenized.
    """
    words = set(_terms(query))
    if not words:
        return "Ошибка: в запросе нет слов длиннее двух букв."
    lowered = text.lower()
    hits = [(s, e, text[s:e]) for s, e, _ in pages(text) if any(w in lowered[s:e] for w in words)]
    if not hits:
        return f"В файле «{name}» по запросу «{query}» ничего не нашлось. Попробуй другие слова."
    docs = [Counter(_terms(part)) for _, _, part in hits]
    avg = sum(sum(d.values()) for d in docs) / len(docs) or 1
    df = Counter(t for d in docs for t in words & d.keys())
    idf = {t: math.log(1 + (len(docs) - n + 0.5) / (n + 0.5)) for t, n in df.items()}

    def score(d: Counter[str]) -> float:
        size = sum(d.values())
        return sum(idf[t] * d[t] * 2.2 / (d[t] + 1.2 * (0.25 + 0.75 * size / avg)) for t in idf if t in d)

    ranked = sorted((i for i in range(len(hits)) if score(docs[i]) > 0), key=lambda i: -score(docs[i]))
    picked, used = [], 0
    for i in ranked:
        if used + len(hits[i][2]) > SEARCH_MAX_CHARS:
            break
        picked.append(i)
        used += len(hits[i][2])
    parts = [f"[символы {hits[i][0]}–{hits[i][1]}]\n{hits[i][2].strip()}" for i in sorted(picked)]
    shown = f"{len(picked)} из {len(ranked)} подходящих мест"
    return f"Файл «{name}», запрос «{query}»: {shown}, в порядке файла.\n\n" + "\n\n".join(parts)
