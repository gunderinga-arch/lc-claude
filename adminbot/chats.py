"""Reading a user's dialogs: the list, one page of messages, and a full-text export."""

from __future__ import annotations

import html
import math
import re
from datetime import datetime, timezone

from aiogram import Router
from aiogram.types import BufferedInputFile, CallbackQuery, InlineKeyboardMarkup, Message
from aiogram.utils.keyboard import InlineKeyboardBuilder

from adminbot.callbacks import LAST_PAGE, DialogPage, Dialogs, Export, UserCard
from adminbot.common import TELEGRAM_LIMIT, fmt_dt, show
from bot.domain import ConversationSummary, LoggedMessage
from bot.render.format import short_number, user_label
from bot.storage.repositories import ConversationRepo, UserRepo

router = Router(name="chats")

DIALOGS_PAGE = 8
PAGE_MESSAGES = 6
EXPORT_CHUNK = 500

# A user's uploaded file, stored as <file name="report.pdf">text</file> (sometimes with an extra
# attribute after the name, e.g. `(показаны первые 200000 символов)`).
_FILE_BLOCK = re.compile(r'<file name="([^"\n]+?)"[^>\n]*>\n?(.*?)\n?</file>', re.S)
# A file the bot generated, stored as <<<FILE name="page.html">>>...content...<<<END>>> (bot/files/blocks.py).
_GEN_BLOCK = re.compile(r'<<<FILE name="([^"\n]{1,200})">>>\n(.*?)\n?<<<END>>>', re.S)


def collapse_files(content: str) -> str:
    """Pure: replace stored file blocks with a short placeholder; other text is untouched.

    Returns raw (unescaped) text — callers escape after truncating, so an entity is never cut.
    """
    content = _FILE_BLOCK.sub(lambda m: f"📎 {m.group(1)} ({len(m.group(2))} симв.)", content)
    return _GEN_BLOCK.sub(lambda m: f"📎 {m.group(1)}", content)


def _truncate(raw: str, cap: int) -> str:
    if len(raw) <= cap:
        return raw
    return raw[:cap] + "… [обрезано, полностью — в файле]"


def render_dialog_page(
    summary: ConversationSummary,
    label: str,
    messages: list[LoggedMessage],
    page: int,
    pages: int,
    tz: timezone,
) -> str:
    """Pure: one page of a dialog. `label` is the already-escaped user label."""
    status = "🟢 текущий" if summary.active else "⚪ завершён"
    header = (
        f"💬 <b>Диалог #{summary.id}</b> · {label} · {status}\n"
        f"Начат {fmt_dt(summary.started_at, tz)} · {summary.messages} сообщ. · стр. {page + 1}/{pages}\n"
    )
    if not messages:
        return header + "\nСообщений нет."

    cap = max(1, (TELEGRAM_LIMIT - 200 - len(header)) // max(1, len(messages)))
    while True:
        text = header
        for message in messages:
            icon = "👤" if message.role == "user" else "🤖"
            tokens = f" · ~{short_number(message.approx_tokens)} ток." if message.role == "assistant" else ""
            body = html.escape(_truncate(collapse_files(message.content), cap))
            text += f"\n{icon} <b>{fmt_dt(message.created_at, tz)}</b>{tokens}\n{body}\n"
        if len(text) <= TELEGRAM_LIMIT or cap <= 1:
            return text
        cap = max(1, int(cap * 0.8))


def _fmt_full(moment: datetime, tz: timezone) -> str:
    """Day, month, year and time — for the exported file, unlike the screens' short fmt_dt."""
    return moment.astimezone(tz).strftime("%d.%m.%Y %H:%M")


def export_text(summary: ConversationSummary, label: str, messages: list[LoggedMessage], tz: timezone) -> str:
    """Pure: the whole dialog as plain text, uncollapsed, for the downloadable .txt."""
    lines = [
        f"Диалог #{summary.id} — {label}",
        f"Начат: {_fmt_full(summary.started_at, tz)}",
        f"Сообщений: {summary.messages}",
        "",
    ]
    for message in messages:
        speaker = "Пользователь" if message.role == "user" else "Бот"
        lines.append(f"[{_fmt_full(message.created_at, tz)}] {speaker}:\n{message.content}\n")
    return "\n".join(lines)


async def dialogs_screen(
    users: UserRepo, dialogs: ConversationRepo, tg_id: int, page: int, tz: timezone
) -> tuple[str, InlineKeyboardMarkup] | None:
    user = await users.get(tg_id)
    if user is None:
        return None
    total = await dialogs.count_for_user(tg_id)
    pages = max(1, math.ceil(total / DIALOGS_PAGE))
    page = min(max(page, 0), pages - 1)
    rows = await dialogs.list_for_user(tg_id, DIALOGS_PAGE, page * DIALOGS_PAGE)

    text = (
        f"💬 <b>Диалоги {html.escape(user_label(user))}</b> — всего {total}\n"
        "🟢 — текущий, его видит модель; ⚪ — завершённые (после «Новый диалог» или очистки)."
    )
    if not rows:
        text += "\nДиалогов пока нет."

    builder = InlineKeyboardBuilder()
    for row in rows:
        mark = "🟢" if row.active else "⚪"
        moment = row.last_at or row.started_at
        builder.button(
            text=f"{mark} #{row.id} · {fmt_dt(moment, tz)} · {row.messages} сообщ.",
            callback_data=DialogPage(conv_id=row.id, page=LAST_PAGE),
        )
    sizes = [1] * len(rows)

    nav = []
    if page > 0:
        nav.append(("◀", Dialogs(tg_id=tg_id, page=page - 1)))
    if page < pages - 1:
        nav.append(("▶", Dialogs(tg_id=tg_id, page=page + 1)))
    for nav_text, callback_data in nav:
        builder.button(text=nav_text, callback_data=callback_data)
    if nav:
        sizes.append(len(nav))

    builder.button(text="« К пользователю", callback_data=UserCard(tg_id=tg_id))
    sizes.append(1)
    builder.adjust(*sizes)
    return text, builder.as_markup()


async def dialog_page_screen(
    users: UserRepo, dialogs: ConversationRepo, conv_id: int, page: int, tz: timezone
) -> tuple[str, InlineKeyboardMarkup] | None:
    summary = await dialogs.summary(conv_id)
    if summary is None:
        return None
    pages = max(1, math.ceil(summary.messages / PAGE_MESSAGES))
    if page == LAST_PAGE:
        page = pages - 1
    elif page < 0:
        page = 0
    elif page >= pages:
        page = pages - 1

    user = await users.get(summary.user_id)
    label = html.escape(user_label(user)) if user is not None else str(summary.user_id)
    messages = await dialogs.messages_page(conv_id, PAGE_MESSAGES, page * PAGE_MESSAGES)
    text = render_dialog_page(summary, label, messages, page, pages, tz)

    builder = InlineKeyboardBuilder()
    sizes = []
    nav = []
    if page > 0:
        nav.append(("◀ Раньше", DialogPage(conv_id=conv_id, page=page - 1)))
    if page < pages - 1:
        nav.append(("Позже ▶", DialogPage(conv_id=conv_id, page=page + 1)))
    for nav_text, callback_data in nav:
        builder.button(text=nav_text, callback_data=callback_data)
    if nav:
        sizes.append(len(nav))
    builder.button(text="📄 Скачать целиком", callback_data=Export(conv_id=conv_id))
    sizes.append(1)
    builder.button(text="« К диалогам", callback_data=Dialogs(tg_id=summary.user_id, page=0))
    sizes.append(1)
    builder.adjust(*sizes)
    return text, builder.as_markup()


async def _all_messages(dialogs: ConversationRepo, conv_id: int) -> list[LoggedMessage]:
    """Every message of the dialog, chronological, fetched in chunks so one dialog is never held twice."""
    messages: list[LoggedMessage] = []
    offset = 0
    while True:
        chunk = await dialogs.messages_page(conv_id, EXPORT_CHUNK, offset)
        messages.extend(chunk)
        if len(chunk) < EXPORT_CHUNK:
            return messages
        offset += EXPORT_CHUNK


@router.callback_query(Dialogs.filter())
async def cb_dialogs(
    callback: CallbackQuery, callback_data: Dialogs, users: UserRepo, dialogs: ConversationRepo, tz: timezone
) -> None:
    screen = await dialogs_screen(users, dialogs, callback_data.tg_id, callback_data.page, tz)
    if screen is None:
        await callback.answer("Пользователь не найден", show_alert=True)
        return
    text, markup = screen
    await show(callback, text, markup)


@router.callback_query(DialogPage.filter())
async def cb_dialog_page(
    callback: CallbackQuery,
    callback_data: DialogPage,
    users: UserRepo,
    dialogs: ConversationRepo,
    tz: timezone,
) -> None:
    screen = await dialog_page_screen(users, dialogs, callback_data.conv_id, callback_data.page, tz)
    if screen is None:
        await callback.answer("Диалог не найден", show_alert=True)
        return
    text, markup = screen
    await show(callback, text, markup)


@router.callback_query(Export.filter())
async def cb_export(
    callback: CallbackQuery, callback_data: Export, users: UserRepo, dialogs: ConversationRepo, tz: timezone
) -> None:
    summary = await dialogs.summary(callback_data.conv_id)
    if summary is None:
        await callback.answer("Диалог не найден", show_alert=True)
        return
    user = await users.get(summary.user_id)
    label = user_label(user) if user is not None else str(summary.user_id)
    messages = await _all_messages(dialogs, callback_data.conv_id)
    document = BufferedInputFile(
        export_text(summary, label, messages, tz).encode("utf-8"), filename=f"dialog-{summary.id}.txt"
    )
    assert isinstance(callback.message, Message)
    await callback.message.answer_document(document)
    await callback.answer("Готово")
