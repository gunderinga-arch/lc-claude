"""Start screen, help, new dialog, clear chat, stop — as commands and persistent-keyboard buttons."""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path

from aiogram import Bot, F, Router
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, Message

from bot.domain import User
from bot.services.chat import ChatService
from bot.services.conversation import ConversationService
from bot.services.jobs import JobManager
from bot.storage.repositories import UserRepo
from bot.telegram.keyboards import (
    BTN_CLEAR,
    BTN_HELP,
    BTN_MODEL,
    BTN_NEW,
    BTN_SETTINGS,
    BTN_STOP,
    REPLY_KEYBOARD,
    clear_confirm_keyboard,
    close_keyboard,
    model_badge,
    pressed,
)
from bot.telegram.panel import delete_range, drop_panel, media, remember, show_panel
from bot.telegram.routers.chat import drop_waiting
from bot.telegram.topics import drop_chat, place_in_chat, topics_enabled

logger = logging.getLogger(__name__)

router = Router(name="commands")

START_TEXT = (
    "<b>▶ Игра началась.</b>\n"
    "Задавай вопрос текстом или голосом, кидай фото, PDF и таблицы — отвечу.\n\n"
    "В команде три бойца-нейросети. Сравнить их и выбрать, кто играет: "
    f"кнопка «{BTN_MODEL}»."
)
# Appended to the welcome when the bot has topics: new users kept missing how chats work (14.10.2026).
START_CHATS_TEXT = (
    "\n\n<b>💬 Чаты</b>\n"
    "Напиши во «Все», о чём хочешь поговорить, — я открою под это отдельную вкладку сверху и сам её назову. "
    "Новая тема — снова пиши во «Все». Переключайся между вкладками когда угодно: "
    "каждый чат помнит свою переписку. Подробнее — «" + BTN_HELP + "»."
)

# Brand assets rendered by promo/src/telegram.tsx in the Play2Go workspace: the looping /start
# cover (the welcome text is its caption) and the looping fighter table for the model picker.
ASSETS = Path(__file__).resolve().parents[3] / "assets"
START_ANIMATION = ASSETS / "start.mp4"
# Stats 1–10, 10 = best in the roster (Fable 5.1, Opus 5, GPT-6 Sol; brand.js in the Play2Go workspace,
# our runs 11–13.10.2026): ЛОГИКА — ARC, ЗНАНИЯ — HLE, СКОРОСТЬ — answer time on hard tasks, ПАМЯТЬ — context.
# ponytail: baked into the picture; re-render it if the roster or the benchmarks change.
ROSTER = ASSETS / "roster.mp4"

HELP_TEXT = (
    "<b>Как это работает</b>\n"
    "Сообщения обрабатываются в фоне: пока модель думает, вы видите индикатор набора. "
    "Самая сильная модель, Fable 5.1, думает дольше остальных — на трудных задачах до нескольких минут.\n\n"
    "<b>Файлы</b>\n"
    "Присылайте фото, PDF, DOCX, XLSX, PPTX, ZIP-архивы, текстовые файлы и код (до 20 МБ, можно альбомом) — "
    "подпись к файлу станет вопросом. Книгу или учебник модель прочитает целиком: что не помещается в её "
    "память сразу, она дочитает по частям. Попросите создать или переделать файл — пришлю его документом "
    "(текстовые форматы: CSV, Markdown, код, JSON…).\n\n"
    "<b>Презентации</b>\n"
    "Попросите презентацию — пришлю файл PowerPoint и страницу-превью: слайды листаются на телефоне и "
    "компьютере. Картинки на слайдах — со скиллом «🎨 Дизайнер».\n\n"
    "<b>Голосовые</b>\n"
    "Можно говорить вместо того, чтобы печатать: отправьте голосовое, кружочек или аудиофайл "
    "до 5 минут. Я покажу расшифровку и отвечу на неё, как на обычное сообщение.\n\n"
    "<b>Кнопки</b>\n"
    f"{BTN_NEW} — новый чат (без тем — забыть контекст, переписка остаётся)\n"
    f"{BTN_CLEAR} — удалить чат с перепиской (без тем — всю переписку, начать с чистого листа)\n"
    f"{BTN_MODEL} — выбор модели\n"
    f"{BTN_SETTINGS} — стиль ответов и своя роль для бота\n"
    f"{BTN_STOP} — прервать текущий ответ\n\n"
    "Telegram не даёт ботам удалять сообщения старше 48 часов — такие останутся в чате.\n\n"
    "<b>Связь с администратором</b>\n"
    "Вопросы, идеи и проблемы с ботом — пишите @gonetake"
)

CHATS_HELP_TEXT = (
    "<b>💬 Чаты — как пользоваться</b>\n"
    "1. Напишите во «Все», о чём хотите поговорить, — откроется новый чат-вкладка сверху, "
    "название он получит сам.\n"
    "2. Нажмите на вкладку — ответ там. Дальше переписывайтесь внутри неё.\n"
    "3. Новая тема — снова пишите во «Все», рядом появится ещё одна вкладка.\n"
    "4. Возвращайтесь в любой чат когда угодно: у каждого своя история и своя модель.\n"
    f"«{BTN_NEW}» — пустой чат с той же моделью, «{BTN_CLEAR}» внутри чата — удалить его.\n\n"
)
CHAT_GREETING = (
    "✦ <b>Новый чат</b>\n\n"
    "Задайте свой запрос — ответит {badge}.\n"
    "Тему я назову по первому сообщению. Модель этого чата меняется кнопкой «" + BTN_MODEL + "»."
)
EMPTY_CHAT_TEXT = "Пустой чат уже открыт — вкладка «Новый чат» сверху. Первое сообщение попадёт в неё."
CLEAR_CHAT_CONFIRM_TEXT = "Удалить этот чат вместе с перепиской? Это нельзя отменить."
CLEAR_ALL_CONFIRM_TEXT = "Удалить всю переписку, включая все чаты-темы, и начать заново? Это нельзя отменить."

MODEL_MENU_TEXT = (
    "<b>Выбери бойца</b>\n"
    "Характеристики — на карточке, 10 — лучший в команде. Цифры — из наших замеров.\n"
    "💎 Fable 5.1 — сильнее всех в логике и помнит вдвое больше, но на трудных задачах думает дольше.\n"
    "🔱 Opus 5 и 🌞 GPT-6 Sol — заметно быстрее на трудных задачах, а в знаниях Fable не уступают.\n"
    "🧠 Sol с планом — сначала составляет план, потом отвечает по нему: для многоэтапных задач."
)
RESERVE_MODEL_MENU_TEXT = "Сейчас работают резервные модели — сверху самая сильная."
CLEAR_CONFIRM_TEXT = "Удалить всю переписку в этом чате и начать заново? Это нельзя отменить."

# Without a stored floor (chats from before this feature) clear only this many recent ids.
UNKNOWN_FLOOR_SPAN = 300


def clear_from(floor: int | None, last_id: int) -> int:
    """First message id "clear chat" deletes: the stored floor, or a recent span when it is unusable."""
    if floor is None or floor > last_id:
        return last_id - UNKNOWN_FLOOR_SPAN
    return floor


async def _delete_trigger(message: Message, state: FSMContext | None = None) -> None:
    """Remove the user's command/button message so the chat stays clean.

    With `state`, also drop any pending input (e.g. waiting for a persona): otherwise the next
    ordinary message would silently be taken as that input.
    """
    if state is not None:
        await state.clear()
    with contextlib.suppress(Exception):
        await message.delete()


async def send_start(bot: Bot, chat_id: int, user: User, thread_id: int = 0) -> Message:
    """The looping cover with the welcome text as its caption, plus the persistent keyboard.

    Returns the message, so "clear chat" can start its range there. Falls back to plain text.
    """
    topic = {"message_thread_id": thread_id} if thread_id else {}
    text = START_TEXT + (START_CHATS_TEXT if await topics_enabled(bot) else "")
    if START_ANIMATION.exists():
        try:
            sent = await bot.send_animation(
                chat_id, media(START_ANIMATION), caption=text, reply_markup=REPLY_KEYBOARD, **topic
            )
            remember(START_ANIMATION, sent)
            return sent
        except Exception:
            logger.warning("could not send the start animation", exc_info=True)
    return await bot.send_message(chat_id, text, reply_markup=REPLY_KEYBOARD, **topic)


@router.message(CommandStart())
async def cmd_start(
    message: Message, user: User, users: UserRepo, state: FSMContext, bot: Bot, thread_id: int = 0
) -> None:
    await _delete_trigger(message, state)
    await drop_panel(bot, message.chat.id, thread_id)
    await send_start(bot, message.chat.id, user, thread_id)
    if await users.get_chat_floor(user.tg_id) is None:
        await users.set_chat_floor(user.tg_id, message.message_id)


@router.message(Command("help"))
@router.message(pressed(BTN_HELP))
async def cmd_help(message: Message, user: User, state: FSMContext, bot: Bot, thread_id: int = 0) -> None:
    await _delete_trigger(message, state)
    from bot.telegram.api import download_limit

    text = HELP_TEXT.replace("до 20 МБ", f"до {download_limit(bot) // (1024 * 1024)} МБ")
    if await topics_enabled(bot):
        text = CHATS_HELP_TEXT + text
    if user.is_admin:
        text += "\n\n<b>Админам</b>: /admin — панель управления, /stats — статистика."
    await show_panel(bot, message.chat.id, text, close_keyboard(), thread_id=thread_id)


@router.message(Command("new"))
@router.message(pressed(BTN_NEW))
async def cmd_new(
    message: Message,
    user: User,
    conversations: ConversationService,
    chat: ChatService,
    state: FSMContext,
    bot: Bot,
    thread_id: int = 0,
) -> None:
    await _delete_trigger(message, state)
    if await topics_enabled(bot):
        # A new chat is a new topic; the old ones stay as tabs. Telegram itself announces the new tab.
        # An empty one is not duplicated: the first message (here or in "Все") goes to it.
        spec = chat.model_for(user)
        greeting = CHAT_GREETING.format(badge=model_badge(spec.id, spec.title))
        placed = await place_in_chat(bot, conversations, user, message.chat.id, greeting, REPLY_KEYBOARD)
        if placed is not None:
            if placed[1]:
                await show_panel(bot, message.chat.id, EMPTY_CHAT_TEXT, close_keyboard(), thread_id=thread_id)
            return
    await conversations.reset(user.tg_id, thread_id)
    drop_waiting(user.tg_id, thread_id)
    text = "▶ Новый раунд. Прошлый разговор я забыл, переписка осталась в чате."
    await show_panel(bot, message.chat.id, text, close_keyboard(), thread_id=thread_id)


@router.message(Command("cancel"))
@router.message(pressed(BTN_STOP))
async def cmd_cancel(
    message: Message, user: User, jobs: JobManager, state: FSMContext, bot: Bot, thread_id: int = 0
) -> None:
    await _delete_trigger(message)
    if await state.get_state() is not None:
        await state.clear()
        text = "Отменено."
    elif jobs.cancel((user.tg_id, thread_id)):
        text = "Останавливаю ответ…"
    else:
        text = "Сейчас ничего не генерируется."
    await show_panel(bot, message.chat.id, text, close_keyboard(), thread_id=thread_id)


@router.message(Command("clear"))
@router.message(pressed(BTN_CLEAR))
async def cmd_clear(
    message: Message,
    user: User,
    conversations: ConversationService,
    state: FSMContext,
    bot: Bot,
    thread_id: int = 0,
) -> None:
    await _delete_trigger(message, state)
    if thread_id:
        text = CLEAR_CHAT_CONFIRM_TEXT
    elif await conversations.threads(user.tg_id):
        text = CLEAR_ALL_CONFIRM_TEXT
    else:
        text = CLEAR_CONFIRM_TEXT
    await show_panel(bot, message.chat.id, text, clear_confirm_keyboard(), thread_id=thread_id)


@router.callback_query(F.data == "chat:clear")
async def cb_clear(
    callback: CallbackQuery,
    user: User,
    users: UserRepo,
    conversations: ConversationService,
    jobs: JobManager,
    state: FSMContext,
    bot: Bot,
    thread_id: int = 0,
) -> None:
    assert isinstance(callback.message, Message)
    chat_id = callback.message.chat.id
    await callback.answer("Очищаю…")
    await state.clear()
    if thread_id:  # a chat is a topic: deleting the topic deletes its messages
        jobs.cancel((user.tg_id, thread_id))
        drop_waiting(user.tg_id, thread_id)
        await drop_panel(bot, chat_id, thread_id)
        await drop_chat(bot, conversations, user.tg_id, chat_id, thread_id)
        return
    # In the main chat everything goes: the topics first, then the messages of the chat itself.
    for topic in await conversations.threads(user.tg_id):
        jobs.cancel((user.tg_id, topic))
        drop_waiting(user.tg_id, topic)
        await drop_chat(bot, conversations, user.tg_id, chat_id, topic)
    # A little past the confirm panel: catches messages that raced in; ids that don't exist are skipped.
    last_id = callback.message.message_id + 20
    jobs.cancel((user.tg_id, 0))
    await conversations.reset(user.tg_id)
    drop_waiting(user.tg_id)
    floor = await users.get_chat_floor(user.tg_id)
    # A floor past this chat's last message came from another chat (the old @lccluade_bot's ids).
    await delete_range(bot, chat_id, clear_from(floor, last_id), last_id)
    start = await send_start(bot, chat_id, user)
    await users.set_chat_floor(user.tg_id, start.message_id)


@router.callback_query(F.data == "panel:close")
async def cb_close(callback: CallbackQuery, bot: Bot, thread_id: int = 0) -> None:
    assert isinstance(callback.message, Message)
    await drop_panel(bot, callback.message.chat.id, thread_id)
    with contextlib.suppress(Exception):  # not the tracked panel (e.g. after a restart)
        await callback.message.delete()
    await callback.answer()
