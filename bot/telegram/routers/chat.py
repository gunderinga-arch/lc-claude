"""Chat: text, photos and documents. The handler answers at once, generation runs as a background job."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import html
import logging
import time
from collections.abc import Awaitable, Callable

from aiogram import Bot, F, Router
from aiogram.enums import ChatAction
from aiogram.exceptions import TelegramBadRequest, TelegramRetryAfter
from aiogram.types import (
    Audio,
    BufferedInputFile,
    CallbackQuery,
    InputMediaPhoto,
    Message,
    ReplyKeyboardMarkup,
    VideoNote,
    Voice,
)

from bot.config import Settings
from bot.domain import RequestStatus, UploadKind, User, upload_kind
from bot.files.blocks import split_files
from bot.files.extract import MAX_TEXT_CHARS, ExtractedText, ScannedPdf, UnsupportedFile, extract_text
from bot.files.ocr import OCR_MAX_PAGES, minutes_for, page_count
from bot.llm.base import LLMClient
from bot.llm.resilience import CUT_NOTE
from bot.render.telegram_html import TELEGRAM_LIMIT, markdown_to_html, split_html
from bot.services.chat import ChatReply, ChatService, title_from
from bot.services.conversation import ConversationService
from bot.services.jobs import JobBusyError, JobManager, ProgressNotifier
from bot.services.quota import QuotaService
from bot.services.speech import SpeechRecognizer, SpeechUnavailable
from bot.storage.repositories import UploadRepo
from bot.telegram import deck_survey
from bot.telegram.api import download_limit
from bot.telegram.download import download_bytes, too_big
from bot.telegram.keyboards import BTN_STOP, REPLY_KEYBOARD, model_badge
from bot.telegram.topics import moved, place_in_chat, rename_chat, topics_enabled

logger = logging.getLogger(__name__)

router = Router(name="chat")

STUB_TEXT = "Эта тема недоступна: шлюз отказывается её обсуждать. Попробуйте переформулировать."
FAILED_TEXT = "Не удалось получить ответ от шлюза (причина: {error}). Попробуйте позже или смените модель."
LIMIT_TEXT = "Дневной лимит исчерпан ({used}/{limit}). Лимит обновится завтра."
BUSY_TEXT = f"Уже обрабатываю ваше предыдущее сообщение — подождите или нажмите «{BTN_STOP}»."
QUEUE_TEXT = "Шлюз занят, вы в очереди (перед вами: {depth})…"
SLOW_TEXT = "Глубокий режим: ответ может занять 1–3 минуты ⏳"
RESEARCH_TEXT = "🔎 Глубокое исследование: ищу по нескольким запросам и читаю страницы, это займёт 1–3 минуты"
TOO_BIG_TEXT = "Файл «{name}» больше 20 МБ — Telegram не даёт ботам скачивать такие файлы."
SCAN_TEXT = (
    "📄 «{name}» — скан без текста. Распознаю {pages} стр. по картинкам, это займёт около {minutes} мин."
)
CONTEXT_RESET_TEXT = (
    "Контекст диалога пришлось сбросить: сервис блокировал ответы из-за темы, "
    "которая обсуждалась раньше. Отвечаю на ваш вопрос с чистого листа."
)
VOICE_OFF_TEXT = "Голосовые сообщения пока не поддерживаются — напишите, пожалуйста, текстом."
VOICE_LONG_TEXT = (
    "Голосовое длиннее {minutes} мин — столько я не распознаю. Разбейте на части или напишите текстом."
)
VOICE_EMPTY_TEXT = "Не получилось разобрать речь 🙉 Запишите ещё раз или напишите текстом."
VOICE_FAIL_TEXT = "Распознавание речи сейчас недоступно — напишите, пожалуйста, текстом."
TRANSCRIPT_SHOWN_CHARS = 3500  # the full transcript goes to the model; the echo in chat is capped
DEFAULT_FILE_PROMPT = "Посмотри приложенные файлы и кратко скажи, что в них."
FILE_WAITING_TEXT = "📎 {what} получил. Напиши, что с {obj} сделать, — отвечу на следующее сообщение."
QUOTE_MAX_CHARS = 1000  # the first request quoted into a new topic
WAITING_TTL_S = 30 * 60  # files sent without a request are forgotten after this long

TELEGRAM_DOWNLOAD_LIMIT = 20 * 1024 * 1024  # Bot API getFile limit
# An answer longer than this many messages (~4k chars each) comes as its start plus the whole as a page.
MAX_ANSWER_MESSAGES = 8
LONG_ANSWER_FILE = "ответ.html"
LONG_ANSWER_NOTE = (
    "<i>…Ответ длинный ({chars} символов): целиком — в файле ниже, он открывается в браузере.</i>"
)
ANSWER_PAGE = (
    '<!doctype html><html lang="ru"><head><meta charset="utf-8">'
    '<meta name="viewport" content="width=device-width, initial-scale=1"><title>Ответ</title><style>'
    "body{{max-width:760px;margin:24px auto;padding:0 16px;font:17px/1.6 system-ui,sans-serif;"
    "white-space:pre-wrap;overflow-wrap:anywhere;color:#1b1b1f;background:#fff}}"
    "pre,code{{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:15px;background:#f3f3f6;"
    "border-radius:6px}}pre{{padding:12px;overflow-x:auto}}code{{padding:1px 4px}}pre code{{padding:0}}"
    "blockquote{{margin:0;padding-left:12px;border-left:3px solid #c9c9d3;color:#555}}a{{color:#3b5bdb}}"
    "</style></head><body>{body}</body></html>"
)
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".gif")
ALBUM_WAIT_S = 1.0  # Telegram delivers an album as separate messages in quick succession

_albums: dict[str, list[Message]] = {}
# Files sent without a caption wait for the user's next message, which becomes their request
# (on a phone Telegram sends the file first and the text as a separate message).
# (user id, topic) -> (monotonic time of the last file, the file messages).
# ponytail: in memory, a restart forgets them.
_waiting: dict[tuple[int, int], tuple[float, list[Message]]] = {}
_naming: set[asyncio.Task[None]] = set()  # fire-and-forget chat naming; a task must be referenced


def drop_waiting(user_id: int, thread_id: int = 0) -> None:
    """Forget files waiting for a request ("new dialog", "clear chat")."""
    _waiting.pop((user_id, thread_id), None)


def _take_waiting(user_id: int, thread_id: int = 0) -> list[Message]:
    at, messages = _waiting.pop((user_id, thread_id), (0.0, []))
    return messages if time.monotonic() - at < WAITING_TTL_S else []


def _waiting_note(count: int) -> str:
    if count == 1:
        return FILE_WAITING_TEXT.format(what="Файл", obj="ним")
    return FILE_WAITING_TEXT.format(what=f"Файлы ({count})", obj="ними")


async def _send_typing(bot: Bot, chat_id: int, thread_id: int = 0) -> None:
    topic = {"message_thread_id": thread_id} if thread_id else {}
    await bot.send_chat_action(chat_id, ChatAction.TYPING, **topic)


async def _deliver(message: Message, reply: ChatReply, markup: ReplyKeyboardMarkup) -> None:
    # Answers carry the persistent keyboard too, so users who never pressed /start get it.
    if reply.status is RequestStatus.STUB:
        await message.answer(STUB_TEXT, reply_markup=markup)
        return
    if reply.status is RequestStatus.FAILED:
        await message.answer(FAILED_TEXT.format(error=reply.error or "unknown"), reply_markup=markup)
        return
    # First line of every answer: the model that actually answered, in bold with its emoji.
    if reply.served_by:  # emergency switch: name the reserve model, no "outage" wording anywhere
        header = f"🛟 <b>{html.escape(reply.model.title)}</b>"
        if reply.model.id == "free-coded":  # the combo: also say which of its members answered
            header += f" <i>({html.escape(reply.served_by)})</i>"
    else:
        header = model_badge(reply.model.id, reply.model.title)
        if reply.fallback_used:
            header += " <i>(запасная модель)</i>"
    header += "\n"
    if reply.context_reset:
        header += f"<i>{CONTEXT_RESET_TEXT}</i>\n"
    header += "\n"
    parts, page = reply.parts, None
    if len(parts) > MAX_ANSWER_MESSAGES or any(len(part) + len(header) > TELEGRAM_LIMIT for part in parts):
        # A very long answer (it can now run to a million chars) would be dozens of messages and hit
        # Telegram's flood limit midway: the start goes to the chat, the whole answer as a page to read.
        answer_text = split_files(reply.text)[0]
        note = LONG_ANSWER_NOTE.format(chars=f"{len(answer_text):,}".replace(",", " "))
        if reply.text.rstrip().endswith(CUT_NOTE.strip()):
            note = "<i>Ответ оборвался. Полученная часть — в файле ниже. Напишите «продолжи» — допишу.</i>"
        preview_limit = TELEGRAM_LIMIT - len(header) - len(note) - 2
        first = split_html(parts[0], limit=preview_limit)[0]
        if len(first) > preview_limit:
            # A single code line cannot be split safely as HTML. The file still keeps it whole.
            first = html.escape(answer_text[: max(1, preview_limit // 6)])
        parts = [first + "\n\n" + note]
        page = BufferedInputFile(
            answer_page(markdown_to_html(answer_text)).encode("utf-8"), filename=LONG_ANSWER_FILE
        )
    for part in parts:
        try:
            await _with_retry(lambda text=header + part: message.answer(text, reply_markup=markup))
        except TelegramBadRequest:
            # Telegram rejected the HTML: resend the raw text without markup.
            logger.warning("telegram rejected rendered HTML, resending as plain text")
            await message.answer(reply.text[:4096], reply_markup=markup, parse_mode=None)
            break
        header = ""
    if page is not None:
        await _with_retry(lambda: message.answer_document(page, reply_markup=markup))
    # Telegram recompresses photos to JPEG and loses transparency, so PNG cutouts go as documents.
    photos = [p for p in reply.images if not p.name.endswith(".png")]
    if len(photos) == 1:
        await message.answer_photo(
            BufferedInputFile(photos[0].data, filename=photos[0].name), reply_markup=markup
        )
    elif photos:  # an album; Telegram takes at most 10 per group
        await message.answer_media_group(
            [InputMediaPhoto(media=BufferedInputFile(p.data, filename=p.name)) for p in photos[:10]]
        )
    for out in [p for p in reply.images if p.name.endswith(".png")] + reply.files:
        await message.answer_document(BufferedInputFile(out.data, filename=out.name), reply_markup=markup)


async def _with_retry(send):
    """One Telegram call, repeated once after the pause Telegram asks for when it says "too many requests"."""
    try:
        return await send()
    except TelegramRetryAfter as exc:
        await asyncio.sleep(exc.retry_after + 1)
        return await send()


def answer_page(body: str) -> str:
    """A long answer's Telegram HTML as a page that reads well on a phone."""
    return ANSWER_PAGE.format(body=body)


def _is_image(message: Message) -> bool:
    if message.photo:
        return True
    doc = message.document
    name = (doc.file_name or "").lower() if doc else ""
    return bool(doc and (doc.mime_type or "").startswith("image/") and name.endswith(IMAGE_EXTENSIONS))


async def _collect(
    bot: Bot,
    messages: list[Message],
    uploads: UploadRepo | None = None,
    user_id: int | None = None,
    request: str | None = None,
    read_scan: Callable[[bytes, str], Awaitable[str | None]] | None = None,
    notify: Callable[[str], Awaitable[object]] | None = None,
) -> tuple[str, tuple[str, ...]]:
    """Download the attachments and turn them into prompt text plus image data URIs.

    Raises UnsupportedFile with a message for the user. Logs every downloaded attachment to
    `uploads` (skipped when None, for callers that don't care, e.g. existing tests). A scanned PDF is read
    from its pages with `read_scan` (the user hears first how long it takes, through `notify`).
    """
    caption = request or next((m.caption for m in messages if m.caption), "") or DEFAULT_FILE_PROMPT
    blocks: list[str] = []
    images: list[str] = []
    for m in messages:
        if m.photo:
            file, name, mime = m.photo[-1], "фото.jpg", "image/jpeg"
        elif m.document:
            file, name = m.document, m.document.file_name or "файл"
            mime = m.document.mime_type or "application/octet-stream"
        else:
            continue
        data = await download_bytes(bot, file, name)
        if uploads is not None:
            # Logged before image/text handling so unsupported files still count.
            if m.photo:
                kind, log_name = UploadKind.PHOTO, None
            else:
                assert m.document is not None
                kind, log_name = upload_kind(m.document.file_name, m.document.mime_type), m.document.file_name
            await uploads.add(user_id, kind, log_name, len(data))
        if _is_image(m):
            images.append(f"data:{mime};base64,{base64.b64encode(data).decode()}")
            blocks.append(f"[Приложено изображение: {name}]")
            continue
        try:
            extracted = await asyncio.to_thread(extract_text, name, data)  # PDF parsing is CPU-bound
        except ScannedPdf:
            if read_scan is None:
                raise
            pages = await asyncio.to_thread(page_count, data)
            if notify is not None and pages:
                shown = str(pages) if pages <= OCR_MAX_PAGES else f"первые {OCR_MAX_PAGES} из {pages}"
                await notify(
                    SCAN_TEXT.format(name=html.escape(name), pages=shown, minutes=minutes_for(pages))
                )
            text = await read_scan(data, name)
            if not text:
                raise
            extracted = ExtractedText(text=text, kind="pdf", truncated=False)
        cut = f" (показаны первые {MAX_TEXT_CHARS} символов)" if extracted.truncated else ""
        blocks.append(f'<file name="{name}"{cut}>\n{extracted.text}\n</file>')
    return caption + "\n\n" + "\n\n".join(blocks), tuple(images)


def _media(message: Message) -> Voice | Audio | VideoNote | None:
    return message.voice or message.audio or message.video_note


async def _transcribe(
    bot: Bot,
    message: Message,
    speech: SpeechRecognizer,
    uploads: UploadRepo | None = None,
    user_id: int | None = None,
) -> str | None:
    """Voice -> text, echoed back so the user sees what was understood. None if there is nothing to answer."""
    media = _media(message)
    assert media is not None
    try:
        data = await download_bytes(bot, media, "аудио")
    except UnsupportedFile as exc:
        await message.answer(html.escape(str(exc)), reply_markup=REPLY_KEYBOARD)
        return None
    if uploads is not None:
        if message.voice:
            kind, name = UploadKind.VOICE, None
        elif message.audio:
            kind, name = UploadKind.AUDIO, message.audio.file_name
        else:
            kind, name = UploadKind.VIDEO_NOTE, None
        await uploads.add(user_id, kind, name, len(data))
    try:
        text = await speech.transcribe(data)
    except SpeechUnavailable:
        await message.answer(VOICE_FAIL_TEXT, reply_markup=REPLY_KEYBOARD)
        return None
    if not text:
        await message.answer(VOICE_EMPTY_TEXT, reply_markup=REPLY_KEYBOARD)
        return None
    shown = text if len(text) <= TRANSCRIPT_SHOWN_CHARS else text[:TRANSCRIPT_SHOWN_CHARS] + "…"
    echo = f"🎤 <i>{html.escape(shown)}</i>"
    try:
        await message.reply(echo)
    except TelegramBadRequest:  # the original moved to a new topic and is gone: nothing to reply to
        await message.answer(echo)
    return f"{message.caption}\n\n{text}" if message.caption else text


async def _generate(
    bot: Bot,
    message: Message,
    user: User,
    chat: ChatService,
    llm: LLMClient,
    uploads: UploadRepo,
    text: str | None = None,
    attachments: list[Message] | None = None,
    speech: SpeechRecognizer | None = None,
    thread_id: int = 0,
    conversations: ConversationService | None = None,
) -> None:
    chat_id = message.chat.id
    spec = chat.model_for(user)
    status_note: Message | None = None
    try:
        if llm.queue_depth() > 0:
            status_note = await message.answer(QUEUE_TEXT.format(depth=llm.queue_depth()))
        elif spec.slow:
            status_note = await message.answer(SLOW_TEXT)
        async with ProgressNotifier(lambda: _send_typing(bot, chat_id, thread_id)):
            images: tuple[str, ...] = ()
            if speech is not None:
                text = await _transcribe(bot, message, speech, uploads, user.tg_id)
                if text is None:
                    return
            if attachments:
                try:
                    # a text or voice request that came after the files is their caption
                    text, images = await _collect(
                        bot,
                        attachments,
                        uploads,
                        user.tg_id,
                        text,
                        chat.read_scan if chat.can_read_scan() else None,
                        message.answer,
                    )
                except UnsupportedFile as exc:
                    await message.answer(html.escape(str(exc)), reply_markup=REPLY_KEYBOARD)
                    return
            assert text is not None
            if status_note is None and chat.is_research(text):
                status_note = await message.answer(RESEARCH_TEXT)
            naming = None
            if thread_id and conversations is not None:
                naming = await _start_naming(bot, chat, conversations, user.tg_id, chat_id, thread_id, text)
            reply = await chat.answer(user, text, images, thread_id)
            if naming is not None:
                await naming  # the topic gets its name before the answer appears in it
        # Generated files/images only come from an OK reply; logged before delivery.
        for out in reply.images:
            kind = UploadKind.GEN_IMAGE_GEMINI if out.name in reply.gemini_images else UploadKind.GEN_IMAGE
            await uploads.add(user.tg_id, kind, out.name, len(out.data))
        for out in reply.files:
            await uploads.add(user.tg_id, UploadKind.GEN_FILE, out.name, len(out.data))
        await _deliver(message, reply, REPLY_KEYBOARD)
    finally:
        if status_note is not None:
            with contextlib.suppress(Exception):
                await bot.delete_message(chat_id, status_note.message_id)


async def _start_naming(
    bot: Bot,
    chat: ChatService,
    conversations: ConversationService,
    user_id: int,
    chat_id: int,
    thread_id: int,
    text: str,
) -> asyncio.Task[None] | None:
    """The first request of a chat the bot opened: the fast model names the topic while the answer is made.

    Returns the naming task (the answer waits for it, so the new name always shows first), or None
    if the chat is named already, or is a topic the user named.
    """
    info = await conversations.chat(user_id, thread_id)
    if info is None or info.title is not None:
        return None
    await conversations.set_title(
        user_id, thread_id, title_from(text)
    )  # claims the naming: one call per chat

    async def name() -> None:
        await rename_chat(bot, conversations, user_id, chat_id, thread_id, await chat.chat_title(text))

    task = asyncio.create_task(name())
    _naming.add(task)
    task.add_done_callback(_naming.discard)
    return task


def _quote_html(messages: list[Message]) -> str:
    """The first request of a chat as a quote for its topic: the text, then the names of the files."""
    text = next((m.text or m.caption for m in messages if m.text or m.caption), "") or ""
    lines = []
    if text:
        cut = "…" if len(text) > QUOTE_MAX_CHARS else ""
        lines.append(f"<blockquote>{html.escape(text[:QUOTE_MAX_CHARS])}{cut}</blockquote>")
    for m in messages:
        if m.photo:
            lines.append("🖼 фото")
        elif m.document:
            lines.append(f"📎 {html.escape(m.document.file_name or 'файл')}")
        elif m.voice or m.audio or m.video_note:
            lines.append("🎤 голосовое")
    return "\n".join(lines) or "…"


async def _land(
    bot: Bot,
    message: Message,
    user: User,
    conversations: ConversationService,
    carried: list[Message],
) -> tuple[Message, int]:
    """A message in the main chat of a bot with topics opens a chat of its own: a new topic.

    The user's message stays where they wrote it (Telegram cannot move it); the topic gets it as a quote,
    with the names of the files, and the answer goes there. Returns (message to answer, topic), or the
    message itself and 0 if Telegram refused a topic.
    """
    unique = {m.message_id: m for m in [*carried, message]}  # an album is both carried and the message
    placed = await place_in_chat(
        bot, conversations, user, message.chat.id, _quote_html(list(unique.values()))
    )
    if placed is None:
        return message, 0
    return moved(message, placed[0]), placed[0]


async def _submit(
    message: Message,
    user: User,
    quota: QuotaService,
    jobs: JobManager,
    make_job,
    *,
    bot: Bot | None = None,
    conversations: ConversationService | None = None,
    thread_id: int = 0,
    carried: list[Message] | None = None,
) -> None:
    """Quota check, then run `make_job(message, thread_id)` in the background, one job per chat.

    In the main chat of a bot with topics, the message first moves to a topic of its own.
    """
    decision = await quota.check(user)
    if not decision.allowed:
        if decision.reason == "daily_limit":
            await message.answer(LIMIT_TEXT.format(used=decision.used, limit=decision.limit))
        return  # pending/blocked are answered by the access middleware
    if not thread_id and bot is not None and conversations is not None and await topics_enabled(bot):
        message, thread_id = await _land(bot, message, user, conversations, carried or [])
    try:
        jobs.submit((user.tg_id, thread_id), make_job(message, thread_id))
    except JobBusyError:
        await message.answer(BUSY_TEXT)


@router.message(F.text, ~F.text.startswith("/"))
async def on_text(
    message: Message,
    user: User,
    bot: Bot,
    chat: ChatService,
    llm: LLMClient,
    quota: QuotaService,
    jobs: JobManager,
    uploads: UploadRepo,
    conversations: ConversationService | None = None,
    thread_id: int = 0,
) -> None:
    files = _take_waiting(user.tg_id, thread_id)
    if not files and deck_survey.wants_survey(message.text or ""):
        # A new presentation: ask how it should look first; the buttons submit the request (on_deck_answer).
        survey = deck_survey.start(user.tg_id, thread_id, message, chat.can_draw(user))
        assert survey.step is not None
        await message.reply(deck_survey.question(survey), reply_markup=deck_survey.keyboard(survey.step))
        return

    def job(target: Message, thread: int):
        return _generate(
            bot, target, user, chat, llm, uploads, text=message.text, attachments=files or None,
            thread_id=thread, conversations=conversations,
        )  # fmt: skip

    await _submit(
        message, user, quota, jobs, job,
        bot=bot, conversations=conversations, thread_id=thread_id,
        carried=files,
    )  # fmt: skip


@router.callback_query(F.data.startswith("deck:"))
async def on_deck_answer(
    callback: CallbackQuery,
    user: User,
    bot: Bot,
    chat: ChatService,
    llm: LLMClient,
    quota: QuotaService,
    jobs: JobManager,
    uploads: UploadRepo,
    conversations: ConversationService | None = None,
    thread_id: int = 0,
) -> None:
    """A button of the presentation survey: the next question, or the request goes to the model."""
    survey, finished = deck_survey.answer(user.tg_id, thread_id, callback.data or "")
    await callback.answer()
    panel = callback.message if isinstance(callback.message, Message) else None
    if survey is None:
        if panel is not None:
            with contextlib.suppress(TelegramBadRequest):
                await panel.edit_text("Этот опрос уже неактуален — напишите запрос заново.")
        return
    if not finished:
        if panel is not None and survey.step is not None:
            markup = deck_survey.keyboard(survey.step)
            with contextlib.suppress(TelegramBadRequest):  # "message is not modified" on a double tap
                await panel.edit_text(deck_survey.question(survey), reply_markup=markup)
        return
    text = deck_survey.final_request(survey)
    if panel is not None:
        new_tab = not thread_id and await topics_enabled(bot)
        where = " — ответ придёт в новую вкладку сверху" if new_tab else ""
        with contextlib.suppress(TelegramBadRequest):
            await panel.edit_text(f"📊 Делаю презентацию, это займёт пару минут{where}.")

    def job(target: Message, thread: int):
        return _generate(
            bot, target, user, chat, llm, uploads, text=text, thread_id=thread, conversations=conversations
        )

    await _submit(
        survey.request, user, quota, jobs, job, bot=bot, conversations=conversations, thread_id=thread_id
    )


@router.message(F.voice | F.audio | F.video_note)
async def on_voice(
    message: Message,
    user: User,
    bot: Bot,
    chat: ChatService,
    llm: LLMClient,
    quota: QuotaService,
    jobs: JobManager,
    uploads: UploadRepo,
    speech: SpeechRecognizer | None,
    settings: Settings,
    conversations: ConversationService | None = None,
    thread_id: int = 0,
) -> None:
    media = _media(message)
    assert media is not None
    if speech is None:
        await message.answer(VOICE_OFF_TEXT)
        return
    if (media.duration or 0) > settings.stt_max_seconds:
        await message.answer(VOICE_LONG_TEXT.format(minutes=settings.stt_max_seconds // 60))
        return
    if (media.file_size or 0) > download_limit(bot):
        await message.answer(too_big(bot, "аудио"))
        return
    files = _take_waiting(user.tg_id, thread_id)

    def job(target: Message, thread: int):
        return _generate(
            bot, target, user, chat, llm, uploads, speech=speech, attachments=files or None,
            thread_id=thread, conversations=conversations,
        )  # fmt: skip

    await _submit(
        message, user, quota, jobs, job,
        bot=bot, conversations=conversations, thread_id=thread_id,
        carried=files,
    )  # fmt: skip


@router.message(F.photo | F.document)
async def on_file(
    message: Message,
    user: User,
    bot: Bot,
    chat: ChatService,
    llm: LLMClient,
    quota: QuotaService,
    jobs: JobManager,
    uploads: UploadRepo,
    conversations: ConversationService | None = None,
    thread_id: int = 0,
) -> None:
    messages = [message]
    if message.media_group_id:
        album = _albums.setdefault(message.media_group_id, [])
        album.append(message)
        if len(album) > 1:
            return  # the first message of the album handles all of them
        await asyncio.sleep(ALBUM_WAIT_S)
        messages = sorted(_albums.pop(message.media_group_id), key=lambda m: m.message_id)
    if not any(m.caption for m in messages):
        _, earlier = _waiting.get((user.tg_id, thread_id), (0.0, []))
        waiting = [*earlier, *messages]
        _waiting[(user.tg_id, thread_id)] = (time.monotonic(), waiting)
        await message.answer(_waiting_note(len(waiting)), reply_markup=REPLY_KEYBOARD)
        return
    messages = [
        *_take_waiting(user.tg_id, thread_id),
        *messages,
    ]  # files sent before this captioned one go too

    def job(target: Message, thread: int):
        return _generate(
            bot, target, user, chat, llm, uploads, attachments=messages,
            thread_id=thread, conversations=conversations,
        )  # fmt: skip

    await _submit(
        message, user, quota, jobs, job,
        bot=bot, conversations=conversations, thread_id=thread_id,
        carried=messages,
    )  # fmt: skip
