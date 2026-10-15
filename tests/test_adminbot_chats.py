"""adminbot.chats: collapse_files, dialog rendering, export text, and the DB-backed screens."""

from __future__ import annotations

import re
from datetime import UTC, datetime

import pytest

from adminbot.callbacks import LAST_PAGE
from adminbot.chats import (
    DIALOGS_PAGE,
    PAGE_MESSAGES,
    collapse_files,
    dialog_page_screen,
    dialogs_screen,
    export_text,
    render_dialog_page,
)
from adminbot.common import TELEGRAM_LIMIT
from bot.domain import ConversationSummary, LoggedMessage, Mode, Role
from bot.storage.db import Database
from bot.storage.repositories import ConversationRepo, UserRepo

UTC_TZ = UTC
NOW = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)
_BAD_AMP = re.compile(r"&(?!amp;|lt;|gt;|quot;|#39;)")  # a lone '&' means an entity got cut


# ---- collapse_files -------------------------------------------------------------------------------


def test_collapse_files_user_block() -> None:
    content = 'before <file name="report.pdf">\nhello world\n</file> after'
    assert collapse_files(content) == "before 📎 report.pdf (11 симв.) after"


def test_collapse_files_user_block_with_truncation_note() -> None:
    content = 'x <file name="report.pdf" (показаны первые 200000 символов)>\nabc\n</file> y'
    assert collapse_files(content) == "x 📎 report.pdf (3 симв.) y"


def test_collapse_files_generated_block() -> None:
    content = '<<<FILE name="page.html">>>\n<html></html>\n<<<END>>>'
    assert collapse_files(content) == "📎 page.html"


def test_collapse_files_multiple_blocks_and_plain_text_untouched() -> None:
    content = 'Hi\n<file name="a.txt">\nAAA\n</file>\n<<<FILE name="b.txt">>>\nBBB\n<<<END>>>\nbye'
    result = collapse_files(content)
    assert "📎 a.txt (3 симв.)" in result
    assert "📎 b.txt" in result
    assert "Hi" in result
    assert "bye" in result
    assert "AAA" not in result and "BBB" not in result


# ---- render_dialog_page (pure) ---------------------------------------------------------------------


def test_render_dialog_page_escapes_user_text() -> None:
    summary = ConversationSummary(id=1, user_id=1, active=False, started_at=NOW, messages=1, last_at=NOW)
    messages = [LoggedMessage(id=1, role="user", content="<script>", approx_tokens=1, created_at=NOW)]
    text = render_dialog_page(summary, "label", messages, 0, 1, UTC_TZ)
    assert "<script>" not in text
    assert "&lt;script&gt;" in text


def test_render_dialog_page_empty_dialog() -> None:
    summary = ConversationSummary(id=1, user_id=1, active=True, started_at=NOW, messages=0, last_at=None)
    text = render_dialog_page(summary, "label", [], 0, 1, UTC_TZ)
    assert "Сообщений нет." in text


def test_render_dialog_page_large_messages_stay_within_limit_with_no_broken_entity() -> None:
    summary = ConversationSummary(id=1, user_id=1, active=True, started_at=NOW, messages=6, last_at=NOW)
    messages = [
        LoggedMessage(
            id=i,
            role="user" if i % 2 == 0 else "assistant",
            content="<>&" * 5000,
            approx_tokens=10,
            created_at=NOW,
        )
        for i in range(6)
    ]
    text = render_dialog_page(summary, "label", messages, 0, 1, UTC_TZ)
    assert len(text) <= TELEGRAM_LIMIT
    assert not _BAD_AMP.search(text)


# ---- export_text (pure) -----------------------------------------------------------------------------


def test_export_text_has_full_uncollapsed_content_and_both_speakers() -> None:
    summary = ConversationSummary(id=5, user_id=1, active=True, started_at=NOW, messages=2, last_at=NOW)
    messages = [
        LoggedMessage(
            id=1,
            role="user",
            content='<file name="a.txt">\nSECRET\n</file>',
            approx_tokens=1,
            created_at=NOW,
        ),
        LoggedMessage(id=2, role="assistant", content="the answer", approx_tokens=1, created_at=NOW),
    ]
    text = export_text(summary, "@gleb", messages, UTC_TZ)
    assert "Диалог #5 — @gleb" in text
    assert "Пользователь" in text
    assert "Бот" in text
    assert "SECRET" in text
    assert '<file name="a.txt">' in text  # not collapsed, unlike the screen


# ---- DB-backed screens ------------------------------------------------------------------------------


@pytest.fixture
async def db() -> Database:
    database = Database(":memory:")
    await database.connect()
    try:
        yield database
    finally:
        await database.close()


async def _make_dialog(dialogs: ConversationRepo, user_id: int, n_messages: int) -> int:
    conv_id = await dialogs.active_id(user_id)
    for i in range(n_messages):
        role = "user" if i % 2 == 0 else "assistant"
        await dialogs.add_message(conv_id, role, f"message {i}", 5)
    return conv_id


async def test_dialogs_screen_unknown_user_returns_none(db: Database) -> None:
    assert await dialogs_screen(UserRepo(db), ConversationRepo(db), 999, 0, UTC_TZ) is None


async def test_dialogs_screen_marks_active_and_orders_newest_first(db: Database) -> None:
    users = UserRepo(db)
    dialogs = ConversationRepo(db)
    await users.upsert_seen(1, "gleb", default_role=Role.USER, default_mode=Mode.NORMAL)
    old_id = await dialogs.active_id(1)
    await dialogs.add_message(old_id, "user", "hi", 1)
    await dialogs.reset(1)
    new_id = await dialogs.active_id(1)
    await dialogs.add_message(new_id, "user", "hi again", 1)

    screen = await dialogs_screen(users, dialogs, 1, 0, UTC_TZ)
    assert screen is not None
    text, markup = screen
    assert "Диалоги @gleb" in text
    buttons = [btn.text for row in markup.inline_keyboard for btn in row]
    idx_new = next(i for i, t in enumerate(buttons) if f"#{new_id}" in t)
    idx_old = next(i for i, t in enumerate(buttons) if f"#{old_id}" in t)
    assert idx_new < idx_old  # newest first
    assert any(t.startswith("🟢") and f"#{new_id}" in t for t in buttons)
    assert any(t.startswith("⚪") and f"#{old_id}" in t for t in buttons)


async def test_dialogs_screen_pagination(db: Database) -> None:
    users = UserRepo(db)
    dialogs = ConversationRepo(db)
    await users.upsert_seen(1, "gleb", default_role=Role.USER, default_mode=Mode.NORMAL)
    for _ in range(DIALOGS_PAGE + 2):  # two pages
        await dialogs.active_id(1)
        await dialogs.reset(1)

    screen = await dialogs_screen(users, dialogs, 1, 0, UTC_TZ)
    assert screen is not None
    texts = [btn.text for row in screen[1].inline_keyboard for btn in row]
    assert "▶" in texts and "◀" not in texts

    screen = await dialogs_screen(users, dialogs, 1, 1, UTC_TZ)
    assert screen is not None
    texts = [btn.text for row in screen[1].inline_keyboard for btn in row]
    assert "◀" in texts and "▶" not in texts


async def test_dialog_page_screen_unknown_dialog_returns_none(db: Database) -> None:
    assert await dialog_page_screen(UserRepo(db), ConversationRepo(db), 999, 0, UTC_TZ) is None


async def test_dialog_page_screen_clamps_page(db: Database) -> None:
    users = UserRepo(db)
    dialogs = ConversationRepo(db)
    await users.upsert_seen(1, "gleb", default_role=Role.USER, default_mode=Mode.NORMAL)
    conv_id = await _make_dialog(dialogs, 1, PAGE_MESSAGES * 2 + 2)  # 3 pages

    screen = await dialog_page_screen(users, dialogs, conv_id, LAST_PAGE, UTC_TZ)
    assert screen is not None and "стр. 3/3" in screen[0]

    screen = await dialog_page_screen(users, dialogs, conv_id, 99, UTC_TZ)
    assert screen is not None and "стр. 3/3" in screen[0]

    screen = await dialog_page_screen(users, dialogs, conv_id, -5, UTC_TZ)
    assert screen is not None and "стр. 1/3" in screen[0]


async def test_dialog_page_screen_nav_buttons(db: Database) -> None:
    users = UserRepo(db)
    dialogs = ConversationRepo(db)
    await users.upsert_seen(1, "gleb", default_role=Role.USER, default_mode=Mode.NORMAL)
    conv_id = await _make_dialog(dialogs, 1, PAGE_MESSAGES * 2 + 2)  # 3 pages: 0, 1, 2

    def texts(markup: object) -> list[str]:
        return [btn.text for row in markup.inline_keyboard for btn in row]  # type: ignore[attr-defined]

    _, markup = await dialog_page_screen(users, dialogs, conv_id, 0, UTC_TZ)  # type: ignore[misc]
    assert "Позже ▶" in texts(markup) and "◀ Раньше" not in texts(markup)

    _, markup = await dialog_page_screen(users, dialogs, conv_id, 1, UTC_TZ)  # type: ignore[misc]
    assert "Позже ▶" in texts(markup) and "◀ Раньше" in texts(markup)

    _, markup = await dialog_page_screen(users, dialogs, conv_id, 2, UTC_TZ)  # type: ignore[misc]
    assert "Позже ▶" not in texts(markup) and "◀ Раньше" in texts(markup)
