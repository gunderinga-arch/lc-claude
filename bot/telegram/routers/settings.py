"""Per-user settings: model picker, answer style, persona and skills."""

from __future__ import annotations

import contextlib
import html
from dataclasses import replace
from pathlib import Path

from aiogram import Bot, F, Router
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import CallbackQuery, InlineKeyboardMarkup, Message

from bot.catalog import Catalog, UnknownModelError
from bot.domain import PERSONA_MAX_CHARS, ImageModel, Style, User
from bot.llm.reserve import ReserveSwitch
from bot.services.conversation import ConversationService
from bot.services.images import ImageGenerator
from bot.services.model_router import ModelRouter
from bot.skills import Skill, toggle_skill
from bot.storage.repositories import UserRepo
from bot.telegram.keyboards import (
    BTN_MODEL,
    BTN_SETTINGS,
    STYLE_LABELS,
    image_model_keyboard,
    model_keyboard,
    persona_prompt_keyboard,
    pressed,
    reserve_model_keyboard,
    settings_keyboard,
    skills_keyboard,
)
from bot.telegram.panel import show_panel
from bot.telegram.routers.commands import MODEL_MENU_TEXT, RESERVE_MODEL_MENU_TEXT, ROSTER

router = Router(name="settings")

THIS_CHAT_NOTE = "\n\n<i>Выбор действует только для этого чата.</i>"
PERSONA_PREVIEW_CHARS = 600
PERSONA_PROMPT_TEXT = (
    "<b>✏️ Роль бота</b>\n\n"
    "Пришлите одним сообщением, каким должен быть бот: кто он, как общается, что учитывать. "
    f"До {PERSONA_MAX_CHARS} символов.\n\n"
    "<i>Например: «Ты опытный Python-разработчик. Отвечай коротко, всегда с примером кода.»</i>"
)


class PersonaForm(StatesGroup):
    waiting = State()


IMAGE_MODEL_MENU_TEXT = (
    "<b>🖼 Модели для фото</b>\n\n"
    "Чем рисовать картинки в скиллах «📸 Фото» и «🎨 Дизайнер».\n\n"
    "<b>Gemini · Nano Banana</b> — качество выше, лучше понимает сложные сцены и надписи. "
    "Если Gemini недоступен или кончился его лимит, картинку нарисует FLUX.\n"
    "<b>FLUX · Cloudflare</b> — быстрый, общий бесплатный лимит на день.\n\n"
    "<i>Редактирование присланных фото (фон, цвет, объекты) всегда делает FLUX.</i>"
)


def _picker(catalog: Catalog, user: User, photo: bool = False) -> InlineKeyboardMarkup:
    return model_keyboard(catalog.selectable(), user.model_override, catalog.for_mode(user.mode).id, photo)


def _reserve_picker(
    catalog: Catalog, router: ModelRouter, user: User, photo: bool = False
) -> InlineKeyboardMarkup:
    current = router.resolve(user, reserve=True).id
    return reserve_model_keyboard(catalog.reserve_selectable(), current, photo)


def _image_picker(user: User, images: ImageGenerator) -> InlineKeyboardMarkup:
    current = user.image_model if images.has_gemini else ImageModel.FLUX
    return image_model_keyboard(current, images.has_gemini)


def _settings_text(user: User, skills: dict[str, Skill] | None = None) -> str:
    if user.persona:
        preview = user.persona[:PERSONA_PREVIEW_CHARS]
        more = "…" if len(user.persona) > PERSONA_PREVIEW_CHARS else ""
        persona = f"\n<blockquote>{html.escape(preview)}{more}</blockquote>"  # user text: always escaped
    else:
        persona = " не задана, бот ведёт себя по умолчанию"
    return (
        "<b>⚙️ Настройки ответов</b>\n\n"
        f"<b>Стиль:</b> {STYLE_LABELS[user.style]}\n"
        f"<b>Роль:</b>{persona}\n"
        f"{_skills_line(user, skills or {})}\n"
        "Настройки действуют только на ваши диалоги."
    )


def _skills_line(user: User, skills: dict[str, Skill]) -> str:
    if not skills:
        return ""
    titles = [html.escape(skills[s].title) for s in user.skills if s in skills]
    return f"<b>Скиллы:</b> {', '.join(titles) if titles else 'не выбраны'}\n"


def _skills_text(skills: dict[str, Skill], enabled: tuple[str, ...]) -> str:
    blocks = [
        f"{'✅' if s.id in enabled else '▫️'} <b>{html.escape(s.title)}</b>"
        + (f"\n<i>{html.escape(s.description)}</i>" if s.description else "")
        for s in skills.values()
    ]
    count = sum(s in skills for s in enabled)
    return (
        "<b>🧩 Скиллы 🔥</b>\n\n"
        "Скиллы — готовые навыки для модели. Включайте сколько угодно: они работают вместе "
        "друг с другом, с ролью и стилем ответа. Нажмите на скилл, чтобы включить или выключить.\n\n"
        + "\n\n".join(blocks)
        + f"\n\n<b>Включено: {count} из {len(skills)}</b>"
    )


async def _show_skills(
    bot: Bot, callback: CallbackQuery, skills: dict[str, Skill], enabled: tuple[str, ...]
) -> None:
    assert isinstance(callback.message, Message)
    active = tuple(s for s in enabled if s in skills)
    await show_panel(
        bot,
        callback.message.chat.id,
        _skills_text(skills, active),
        skills_keyboard(list(skills.values()), active),
        replace=callback.message,
    )


async def _show_settings(
    bot: Bot,
    chat_id: int,
    user: User,
    skills: dict[str, Skill],
    replace_message: Message | None = None,
    thread_id: int = 0,
) -> None:
    await show_panel(
        bot,
        chat_id,
        _settings_text(user, skills),
        settings_keyboard(user.style, bool(user.persona), skills_available=bool(skills)),
        replace=replace_message,
        thread_id=thread_id,
    )


# ---- model ------------------------------------------------------------------------------------


@router.message(Command("model"))
@router.message(pressed(BTN_MODEL))
async def cmd_model(
    message: Message,
    user: User,
    catalog: Catalog,
    state: FSMContext,
    bot: Bot,
    reserve: ReserveSwitch | None,
    router: ModelRouter,
    images: ImageGenerator | None = None,
    thread_id: int = 0,
) -> None:
    with contextlib.suppress(Exception):
        await message.delete()
    await state.clear()
    text, markup, photo = await _model_menu(catalog, router, reserve, user, images, thread_id)
    await show_panel(bot, message.chat.id, text, markup, photo=photo, thread_id=thread_id)


async def _model_menu(
    catalog: Catalog,
    router: ModelRouter,
    reserve: ReserveSwitch | None,
    user: User,
    images: ImageGenerator | None,
    thread_id: int = 0,
) -> tuple[str, InlineKeyboardMarkup, Path | None]:
    """Text, buttons and picture of the model picker; the reserve list has no fighter card.

    In a topic the pick is that chat's own model, and the text says so.
    """
    photo = images is not None
    if reserve is not None and await reserve.active():
        return RESERVE_MODEL_MENU_TEXT, _reserve_picker(catalog, router, user, photo), None
    return MODEL_MENU_TEXT + (THIS_CHAT_NOTE if thread_id else ""), _picker(catalog, user, photo), ROSTER


@router.callback_query(F.data.startswith("imodel:"))
async def cb_image_model(
    callback: CallbackQuery,
    user: User,
    users: UserRepo,
    catalog: Catalog,
    reserve: ReserveSwitch | None,
    router: ModelRouter,
    bot: Bot,
    images: ImageGenerator | None = None,
    thread_id: int = 0,
) -> None:
    """The photo tab of /model: "menu" opens it, "back" returns to the text models, else a pick."""
    assert callback.data and isinstance(callback.message, Message)
    action = callback.data.removeprefix("imodel:")
    chat_id = callback.message.chat.id
    if images is None:
        await callback.answer("Картинки сейчас выключены", show_alert=True)
        return
    if action == "back":
        text, markup, photo = await _model_menu(catalog, router, reserve, user, images, thread_id)
        await show_panel(bot, chat_id, text, markup, replace=callback.message, photo=photo)
        await callback.answer()
        return
    note = None
    if action != "menu":
        try:
            model = ImageModel(action)
        except ValueError:
            await callback.answer("Неизвестная модель", show_alert=True)
            return
        if model is ImageModel.GEMINI and not images.has_gemini:
            await callback.answer("Gemini не подключён", show_alert=True)
            return
        await users.set_image_model(user.tg_id, model)
        user, note = replace(user, image_model=model), "Сохранено"
    await show_panel(
        bot, chat_id, IMAGE_MODEL_MENU_TEXT, _image_picker(user, images), replace=callback.message
    )
    await callback.answer(note)


@router.callback_query(F.data.startswith("model:"))
async def cb_model(
    callback: CallbackQuery,
    user: User,
    users: UserRepo,
    catalog: Catalog,
    reserve: ReserveSwitch | None,
    images: ImageGenerator | None = None,
    conversations: ConversationService | None = None,
    thread_id: int = 0,
) -> None:
    assert callback.data and isinstance(callback.message, Message)
    choice = callback.data.removeprefix("model:")
    try:
        spec = catalog.get(choice)
    except UnknownModelError:
        await callback.answer("Неизвестная модель", show_alert=True)
        return
    if not catalog.is_available(choice) or spec.reserve:
        await callback.answer("Модель сейчас недоступна", show_alert=True)
        return
    if thread_id and conversations is not None:  # inside a chat: only this chat changes its model
        await conversations.set_model(user.tg_id, thread_id, choice)
    else:
        await users.set_model_override(user.tg_id, choice)
    user = replace(user, model_override=choice)
    if reserve is not None and await reserve.active():
        # The picker still shows the reserve list; a stale normal-model button just saves the
        # choice for when the switch goes back off, without touching the panel on screen.
        await callback.answer("Сохранено, применится, когда выключат резерв")
        return
    await callback.message.edit_reply_markup(reply_markup=_picker(catalog, user, images is not None))
    await callback.answer("Сохранено")


@router.callback_query(F.data.startswith("rmodel:"))
async def cb_reserve_model(
    callback: CallbackQuery,
    user: User,
    users: UserRepo,
    catalog: Catalog,
    router: ModelRouter,
    images: ImageGenerator | None = None,
) -> None:
    assert callback.data and isinstance(callback.message, Message)
    choice = callback.data.removeprefix("rmodel:")
    try:
        spec = catalog.get(choice)
    except UnknownModelError:
        await callback.answer("Неизвестная модель", show_alert=True)
        return
    if not spec.reserve or spec.service or not catalog.is_available(choice):
        await callback.answer("Модель сейчас недоступна", show_alert=True)
        return
    await users.set_reserve_model(user.tg_id, choice)
    user = replace(user, reserve_model=choice)
    markup = _reserve_picker(catalog, router, user, images is not None)
    await callback.message.edit_reply_markup(reply_markup=markup)
    await callback.answer("Сохранено")


# ---- style and persona --------------------------------------------------------------------------


@router.message(Command("settings"))
@router.message(pressed(BTN_SETTINGS))
async def cmd_settings(
    message: Message, user: User, skills: dict[str, Skill], state: FSMContext, bot: Bot, thread_id: int = 0
) -> None:
    with contextlib.suppress(Exception):
        await message.delete()
    await state.clear()
    await _show_settings(bot, message.chat.id, user, skills, thread_id=thread_id)


@router.callback_query(F.data.startswith("set:"))
async def cb_settings(
    callback: CallbackQuery,
    user: User,
    users: UserRepo,
    skills: dict[str, Skill],
    state: FSMContext,
    bot: Bot,
) -> None:
    assert callback.data and isinstance(callback.message, Message)
    action = callback.data.removeprefix("set:")

    if action == "persona":
        await state.set_state(PersonaForm.waiting)
        await show_panel(
            bot,
            callback.message.chat.id,
            PERSONA_PROMPT_TEXT,
            persona_prompt_keyboard(),
            replace=callback.message,
        )
        await callback.answer()
        return

    if action == "skills":
        await _show_skills(bot, callback, skills, user.skills)
        await callback.answer()
        return

    if action == "skills_off":
        await users.set_skills(user.tg_id, ())
        await _show_skills(bot, callback, skills, ())
        await callback.answer("Все скиллы выключены")
        return

    if action.startswith("skill:"):
        skill_id = action.removeprefix("skill:")
        if skill_id not in skills:
            await callback.answer("Этого скилла больше нет", show_alert=True)
            return
        enabled = toggle_skill(user.skills, skill_id, skills)
        await users.set_skills(user.tg_id, enabled)
        await _show_skills(bot, callback, skills, enabled)
        await callback.answer("Включено" if skill_id in enabled else "Выключено")
        return

    if action.startswith("style:"):
        try:
            style = Style(action.removeprefix("style:"))
        except ValueError:
            await callback.answer("Неизвестный стиль", show_alert=True)
            return
        await users.set_style(user.tg_id, style)
        user = replace(user, style=style)
    elif action == "persona_reset":
        await users.set_persona(user.tg_id, None)
        user = replace(user, persona=None)
    elif action == "back":
        await state.clear()

    await _show_settings(bot, callback.message.chat.id, user, skills, replace_message=callback.message)
    await callback.answer()


@router.message(PersonaForm.waiting, F.text, ~F.text.startswith("/"))
async def on_persona(
    message: Message,
    user: User,
    users: UserRepo,
    skills: dict[str, Skill],
    state: FSMContext,
    bot: Bot,
    thread_id: int = 0,
) -> None:
    assert message.text is not None
    persona = message.text.strip()
    with contextlib.suppress(Exception):
        await message.delete()
    if len(persona) > PERSONA_MAX_CHARS:
        await show_panel(
            bot,
            message.chat.id,
            f"Слишком длинно: {len(persona)} символов, максимум {PERSONA_MAX_CHARS}. Пришлите короче.",
            persona_prompt_keyboard(),
            thread_id=thread_id,
        )
        return
    await users.set_persona(user.tg_id, persona or None)
    await state.clear()
    await _show_settings(
        bot, message.chat.id, replace(user, persona=persona or None), skills, thread_id=thread_id
    )
