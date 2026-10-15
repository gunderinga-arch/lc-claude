"""Regressions found while completing the interrupted long-context review."""

import asyncio
import io
import json
import zipfile
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from bot.domain import Mode, RequestStatus
from bot.files.extract import UnsupportedFile, extract_text
from bot.files.store import FileStore
from bot.llm.base import ChatMessage, ProviderTooLong, ToolCall
from bot.llm.window import fit_window
from bot.services.conversation import ConversationService
from bot.services.filetools import FileTools
from bot.storage.repositories import ConversationRepo
from tests.test_chat_service import FakeLLM, env, make_service, ok  # noqa: F401


async def test_idle_reset_keeps_the_new_attachment(env, tmp_path, monkeypatch):  # noqa: F811
    store = FileStore(tmp_path)
    repo = ConversationRepo(env["db"])
    old_id = await repo.active_id(1)
    await repo.add_message(old_id, "user", "старый вопрос", 10)

    async def old_activity(_):
        return datetime.now(UTC) - timedelta(hours=13)

    monkeypatch.setattr(repo, "last_activity", old_activity)
    env["conversations"] = ConversationService(
        repo, system_prompt="s", token_budget=20_000, files=store, idle_hours=12
    )
    llm = FakeLLM(ok("ответ"))
    service = make_service(env, llm)
    service._files = store
    monkeypatch.setattr(service, "_history_window", lambda _: 20_000)
    body = "Данные новой книги. " * 20_000
    await service.answer(env["user"], f'Прочитай\n\n<file name="new.txt">\n{body}\n</file>')
    assert await repo.active_id(1) != old_id
    assert store.read(1, 0, "new.txt") == ("new.txt", body)
    assert "read_file" in llm.calls[0][0][-1].content
    assert "new.txt" in llm.calls[0][0][0].content


async def test_personal_history_budget_does_not_cut_a_file_that_fits_model(env, tmp_path):  # noqa: F811
    llm = FakeLLM(ok("ответ"))
    service = make_service(env, llm)
    service._files = FileStore(tmp_path)
    body = "Данные документа. " * 12_000
    message = f'Прочитай\n\n<file name="report.txt">\n{body}\n</file>'
    await service.answer(replace(env["user"], context_limit=8_000), message)
    assert llm.calls[0][0][-1].content == message


async def test_reserve_disables_digest_instead_of_summarizing_truncated_parts(env, tmp_path):  # noqa: F811
    from bot.llm.base import ChatResult, ChatStatus

    class Reserve:
        async def active(self):
            return True

    llm = FakeLLM(
        ChatResult(
            ChatStatus.OK,
            "",
            "free-coded",
            1,
            1,
            False,
            tool_calls=(ToolCall("d", "digest_file", json.dumps({"name": "book.txt", "task": "конспект"})),),
        ),
        ok("Прочитаю частями", model="free-coded"),
    )
    service = make_service(env, llm)
    service._reserve = Reserve()
    service._files = FileStore(tmp_path)
    body = "Текст книги. " * 30_000
    reply = await service.answer(env["user"], f'Конспект\n\n<file name="book.txt">\n{body}\n</file>')
    assert reply.status is RequestStatus.OK
    assert len(llm.calls) == 2
    assert "выжимка сейчас недоступна" in llm.calls[-1][0][-1].content


def test_required_digest_part_cannot_be_silently_cut_for_a_small_fallback():
    message = ChatMessage("user", "Полная часть книги. " * 10_000, truncatable=False)
    with pytest.raises(ProviderTooLong):
        fit_window([message], 1000)


async def test_bulk_gate_is_shared_between_users(env, tmp_path, monkeypatch):  # noqa: F811
    import bot.services.filetools as ft

    active = peak = 0

    class CountingLLM:
        async def chat(self, *args, **kwargs):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.01)
            active -= 1
            return ok("Текст страниц")

    llm = CountingLLM()
    service = make_service(env, llm)
    monkeypatch.setattr("bot.services.chat.page_count", lambda _: 12)
    monkeypatch.setattr("bot.services.chat.can_read_scans", lambda: True)

    async def render(*args):
        return [b"jpeg"]

    monkeypatch.setattr("bot.services.chat.render_pages", render)
    monkeypatch.setattr(ft, "DIGEST_PART_TOKENS", 1000)
    store = FileStore(tmp_path)
    model = env["catalog"].for_mode(Mode.NORMAL).id
    files = FileTools(store, 1, 0, llm, model, bulk_gate=service._bulk_gate)
    await asyncio.gather(
        service.read_scan(b"pdf", "one.pdf"),
        service.read_scan(b"pdf", "two.pdf"),
        files.digest("book.txt", "Текст книги. " * 1000, "конспект"),
    )
    assert peak == 2


def test_office_archives_share_the_total_decompression_guard(monkeypatch):
    import bot.files.extract as extract

    monkeypatch.setattr(extract, "ZIP_MAX_TOTAL_BYTES", 100)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("word/document.xml", "a" * 60)
        archive.writestr("unused.xml", "b" * 60)
    with pytest.raises(UnsupportedFile, match="предел распаковки"):
        extract_text("document.docx", buf.getvalue())
