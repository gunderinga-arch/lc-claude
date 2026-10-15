"""Big files: cut to the model's window, kept whole in the file store, read and searched by tools."""

import itertools
import json
import os
import time

from bot.domain import Mode, Role, approx_tokens
from bot.files.store import FileStore
from bot.llm.base import ChatResult, ChatStatus, ToolCall
from bot.services.chat import ChatService
from bot.services.conversation import ConversationService
from bot.services.filetools import (
    FILE_CUT_NOTE,
    FILE_TOOL_NAMES,
    READ_MAX_CHARS,
    FileTools,
    fit_files,
    read_part,
    search,
)
from bot.services.model_router import ModelRouter
from bot.services.quota import QuotaService
from bot.storage.db import Database
from bot.storage.repositories import ConversationRepo, RequestRepo, UsageRepo, UserRepo
from tests.test_chat_service import FakeLLM, ok

BOOK = "\n".join(f"Глава {n}. " + ("Обычный текст главы про погоду и дорогу. " * 30) for n in range(1, 400))
BOOK = BOOK.replace("Глава 321. ", "Глава 321. Здесь капитан Немо впервые называет своё имя. ")


def test_a_message_that_fits_is_left_alone_and_only_big_files_are_kept() -> None:
    small = '<file name="a.txt">\nкоротко\n</file>'
    text, keep = fit_files("вопрос\n\n" + small, 10_000)
    assert text == "вопрос\n\n" + small and keep == {}
    big = f'<file name="book.txt">\n{BOOK}\n</file>'
    text, keep = fit_files("вопрос\n\n" + big, 10_000_000)
    assert text == "вопрос\n\n" + big and keep == {"book.txt": BOOK}  # whole, and kept for later turns


def test_a_file_over_the_budget_is_cut_with_a_note_and_the_small_one_stays_whole() -> None:
    note = '<file name="note.txt">\nкраткая заметка\n</file>'
    message = f'Сравни\n\n{note}\n\n<file name="book.txt">\n{BOOK}\n</file>'
    text, keep = fit_files(message, 20_000)
    assert approx_tokens(text) <= 20_000
    assert '<file name="note.txt">\nкраткая заметка\n</file>' in text
    assert f'<file name="book.txt" chars="{len(BOOK)}">\nГлава 1.' in text
    assert "read_file" in text and "Глава 321" not in text
    assert keep["book.txt"] == BOOK
    shown = int(text.split("выше символы 0–")[1].split(" ")[0])
    assert BOOK[:shown] in text
    assert FILE_CUT_NOTE.split("{")[0] in text


def test_read_part_shows_offsets_and_where_to_go_on() -> None:
    part = read_part("book.txt", BOOK, 1000, 500)
    assert part.startswith(f"Файл «book.txt», символы 1000–1500 из {len(BOOK)}:")
    assert BOOK[1000:1500] in part and "start=1500" in part
    assert read_part("b", "abc", 2, 10).endswith("c\n\n[Конец файла]")
    assert len(read_part("b", "x" * 10**6, 0, 10**6)) < READ_MAX_CHARS + 300


def test_search_finds_the_place_with_its_offsets() -> None:
    found = search("book.txt", BOOK, "капитан Немо")
    assert "Здесь капитан Немо впервые называет своё имя" in found
    at = BOOK.index("Здесь капитан")
    start, end = (int(x) for x in found.split("[символы ")[1].split("]")[0].split("–"))
    assert start <= at < end
    assert "ничего не нашлось" in search("book.txt", BOOK, "звездолёт")


def test_store_keeps_reads_forgets_and_prunes(tmp_path) -> None:
    store = FileStore(tmp_path, keep_days=1)
    store.save(1, 0, "Отчёт 2026.pdf", "текст отчёта")
    store.save(1, 7, "other.txt", "другой чат")
    assert store.names(1, 0) == ["Отчёт 2026.pdf"]
    assert store.read(1, 0, "отчёт 2026.PDF") == ("Отчёт 2026.pdf", "текст отчёта")
    assert store.read(1, 0, "что-то не то") == ("Отчёт 2026.pdf", "текст отчёта")  # the only file of the chat
    store.clear(1, 0)
    assert store.names(1, 0) == [] and store.names(1, 7) == ["other.txt"]
    old = time.time() - 3 * 86_400
    for path in (tmp_path / "1" / "7").glob("*"):
        os.utime(path, (old, old))
    store.prune()
    assert store.names(1, 7) == []


async def test_file_tools_read_and_search_by_the_models_arguments(tmp_path) -> None:
    store = FileStore(tmp_path)
    store.save(5, 3, "book.txt", BOOK)
    tools = FileTools(store, 5, 3)

    def call(tool: str, **args) -> ToolCall:
        return ToolCall("c1", tool, json.dumps(args, ensure_ascii=False))

    assert BOOK[:100] in await tools.run(call("read_file", name="book.txt", length=100))
    assert "капитан Немо" in await tools.run(call("search_file", name="book.txt", query="Немо"))
    assert "нет" in await FileTools(store, 5, 4).run(call("read_file", name="book.txt"))
    assert "обязателен" in await tools.run(ToolCall("c2", "read_file", "{oops"))


async def test_a_huge_file_reaches_the_model_cut_and_its_tools_read_the_rest(tmp_path, monkeypatch) -> None:
    db = Database(":memory:")
    await db.connect()
    user = await UserRepo(db).upsert_seen(1, "u", default_role=Role.USER, default_mode=Mode.NORMAL)
    store = FileStore(tmp_path)
    conversations = ConversationService(
        ConversationRepo(db), system_prompt="sys", token_budget=20_000, files=store
    )
    from bot.catalog import load_catalog
    from bot.config import DEFAULT_MODELS_FILE

    catalog = load_catalog(DEFAULT_MODELS_FILE)
    read = ToolCall("r1", "search_file", json.dumps({"name": "book.txt", "query": "капитан Немо"}))
    llm = FakeLLM(
        ChatResult(ChatStatus.OK, "", "claude-opus-5", 1, 10, False, tool_calls=(read,)),
        ok("Имя он называет в главе 321."),
        ok("второй ответ"),
    )
    service = ChatService(
        llm,
        conversations,
        QuotaService(UsageRepo(db), daily_limit=10),
        ModelRouter(catalog),
        RequestRepo(db),
        catalog,
        files=store,
    )
    message = f'Где капитан называет имя?\n\n<file name="book.txt">\n{BOOK}\n</file>'
    monkeypatch.setattr(service, "_history_window", lambda _: 25_000)
    reply = await service.answer(user, message)
    assert reply.status.value == "ok"
    first, second = llm.calls[0][0], llm.calls[1][0]
    assert "Глава 321" not in first[-1].content and "read_file" in first[-1].content  # cut, with the note
    assert "сохранённые файлы" in first[0].content and "«book.txt»" in first[0].content
    assert {t["function"]["name"] for t in llm.tools[0]} >= FILE_TOOL_NAMES
    assert "капитан Немо впервые" in second[-1].content  # the tool's result went back to the model
    assert store.read(1, 0, "book.txt") == ("book.txt", BOOK)
    # The history keeps the cut message, and the next answer still has the tools.
    await service.answer(user, "А в какой главе он уходит?")
    history = llm.calls[2][0]
    assert any("read_file" in m.content for m in history if m.role == "user")
    assert {t["function"]["name"] for t in llm.tools[2]} >= FILE_TOOL_NAMES
    await conversations.reset(1)
    assert store.names(1, 0) == []
    await db.close()


def test_split_parts_covers_the_text_in_line_cut_parts_of_the_size() -> None:
    from bot.services.filetools import split_parts

    parts = split_parts(BOOK, 20_000)
    assert parts[0][0] == 0 and parts[-1][1] == len(BOOK)
    assert all(a[1] == b[0] for a, b in itertools.pairwise(parts))  # no gaps, no overlaps
    assert all(BOOK[e - 1] == "\n" for _, e in parts[:-1])
    assert all(approx_tokens(BOOK[s:e]) <= 20_000 for s, e in parts)


async def test_digest_reads_every_part_and_counts_the_tokens(tmp_path) -> None:
    from bot.services.filetools import split_parts

    store = FileStore(tmp_path)
    store.save(5, 3, "book.txt", BOOK)
    n = len(split_parts(BOOK, 20_000))
    llm = FakeLLM(*(ok(f"выписка {i}") for i in range(n)))
    tools = FileTools(store, 5, 3, llm, "claude-opus-5")
    import bot.services.filetools as ft

    old, ft.DIGEST_PART_TOKENS = ft.DIGEST_PART_TOKENS, 20_000
    try:
        out = await tools.run(
            ToolCall("d", "digest_file", json.dumps({"name": "book.txt", "task": "конспект"}))
        )
    finally:
        ft.DIGEST_PART_TOKENS = old
    assert len(llm.calls) == n and all(model == "claude-opus-5" for _, model in llm.calls)
    assert out.startswith(f"Выписки из файла «book.txt» по задаче «конспект», {n} частей")
    assert all(f"### Часть {i + 1} " in out and f"выписка {i}" in out for i in range(n))
    assert "конспект" in llm.calls[0][0][0].content and "Глава 1." in llm.calls[0][0][0].content
    assert tools.spent > approx_tokens(BOOK)
    repeated = await tools.run(
        ToolCall("again", "digest_file", json.dumps({"name": "book.txt", "task": "x"}))
    )
    assert "одна выжимка" in repeated and len(llm.calls) == n
    assert "недоступна" in await FileTools(store, 5, 3).run(
        ToolCall("d", "digest_file", json.dumps({"name": "book.txt", "task": "x"}))
    )
