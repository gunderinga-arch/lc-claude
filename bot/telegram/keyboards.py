"""Keyboards: the persistent reply keyboard, /model picker, admin panels."""

from __future__ import annotations

import html

from aiogram import F
from aiogram.types import InlineKeyboardMarkup, KeyboardButton, ReplyKeyboardMarkup
from aiogram.utils.keyboard import InlineKeyboardBuilder

from bot.catalog import ModelSpec
from bot.domain import ImageModel, Style
from bot.skills import Skill

# Buttons of the always-visible keyboard under the input field. Their presses arrive as plain
# text messages, so these exact labels are routed to handlers before the chat router sees them.
# Arcade style: game-console glyphs (promo/brand in the Play2Go workspace).
BTN_NEW = "▶ Новый диалог"
BTN_CLEAR = "✖ Очистить чат"
BTN_MODEL = "★ Модель"
BTN_SETTINGS = "⚙ Настройки"
BTN_STOP = "■ Стоп"
BTN_HELP = "? Помощь"

# Labels of the previous design: chats opened before the redesign still show that keyboard.
LEGACY_LABELS = {
    BTN_NEW: "🔄 Новый диалог",
    BTN_CLEAR: "🧹 Очистить чат",
    BTN_MODEL: "🤖 Модель",
    BTN_SETTINGS: "⚙️ Настройки",
    BTN_STOP: "⛔ Стоп",
    BTN_HELP: "ℹ️ Помощь",
}


def pressed(label: str):
    """Filter for a keyboard button press: the current label or its pre-redesign twin."""
    return F.text.in_({label, LEGACY_LABELS[label]})


REPLY_KEYBOARD = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text=BTN_NEW), KeyboardButton(text=BTN_CLEAR)],
        [KeyboardButton(text=BTN_MODEL), KeyboardButton(text=BTN_SETTINGS)],
        [KeyboardButton(text=BTN_STOP), KeyboardButton(text=BTN_HELP)],
    ],
    resize_keyboard=True,
    is_persistent=True,
    input_field_placeholder="Сообщение…",
)


def close_keyboard() -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    builder.button(text="✖ Закрыть", callback_data="panel:close")
    return builder.as_markup()


STYLE_LABELS = {Style.SHORT: "Кратко", Style.NORMAL: "Обычно", Style.DETAILED: "Подробно"}


def settings_keyboard(
    style: Style, has_persona: bool, skills_available: bool = False
) -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    for value, label in STYLE_LABELS.items():
        builder.button(text=("✅ " if value is style else "") + label, callback_data=f"set:style:{value}")
    builder.button(text="✏️ Задать роль", callback_data="set:persona")
    if has_persona:
        builder.button(text="🗑 Сбросить роль", callback_data="set:persona_reset")
    if skills_available:
        builder.button(text="🧩 Скиллы 🔥", callback_data="set:skills")
    builder.button(text="✖ Закрыть", callback_data="panel:close")
    builder.adjust(3, 2 if has_persona else 1, 1)
    return builder.as_markup()


def skills_keyboard(skills: list[Skill], enabled: tuple[str, ...]) -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    for skill in skills:
        mark = "✅ " if skill.id in enabled else "▫️ "
        builder.button(text=mark + skill.title, callback_data=f"set:skill:{skill.id}")
    if enabled:
        builder.button(text="🚫 Выключить все", callback_data="set:skills_off")
    builder.button(text="« Назад к настройкам", callback_data="set:back")
    builder.adjust(1)
    return builder.as_markup()


def persona_prompt_keyboard() -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    builder.button(text="« Назад", callback_data="set:back")
    return builder.as_markup()


def clear_confirm_keyboard() -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    builder.button(text="🧹 Да, удалить всё", callback_data="chat:clear")
    builder.button(text="Отмена", callback_data="panel:close")
    builder.adjust(2)
    return builder.as_markup()


# Отображение моделей в /model: id -> (подпись со смайликом). Порядок = порядок в списке.
MODEL_LABELS: dict[str, str] = {
    "claude-fable-5.1": "💎 Claude Fable 5.1 — самая сильная, медленная",
    "gpt-6-sol": "🌞 GPT-6 Sol",
    "gpt-6-astra": "⚡ GPT-Astra",
    "gpt-5.6-sol": "☀️ GPT-Sol",
    "claude-opus-5": "🔱 Claude Opus 5",
    "gpt-5.6-luna": "🌒 GPT-Luna",
    "claude-opus-4.8": "🔱 Claude Opus 4.8",
    "claude-fable-5": "⏳ Claude Fable 5",
    "plan-opus": "🧠 GPT-6 Sol с планом — многоэтапные задачи",
}


# Short name + emoji shown above every answer (the model that actually answered).
MODEL_BADGES: dict[str, str] = {
    "claude-fable-5.1": "<b>Claude Fable 5.1</b> 💎",
    "gpt-6-sol": "<b>GPT-6 Sol</b> 🌞",
    "claude-opus-5": "<b>Claude Opus 5</b> 🔱",
    "gpt-6-astra": "<b>GPT-Astra</b> ⚡",
    "gpt-5.6-sol": "<b>GPT-Sol</b> ☀️",
    "gpt-5.6-luna": "<b>GPT-Luna</b> 🌒",
    "claude-opus-4.8": "<b>Claude Opus 4.8</b> 🔱",
    "claude-fable-5": "<b>Claude Fable 5</b> ⏳",
    "plan-opus": "<b>GPT-6 Sol + план</b> 🧠",
}


def model_badge(model_id: str, title: str) -> str:
    return MODEL_BADGES.get(model_id, f"<b>{html.escape(title)}</b>")


def model_keyboard(
    models: list[ModelSpec], current_id: str | None, mode_model_id: str, photo: bool = False
) -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    labeled = [s for s in models if s.id in MODEL_LABELS]
    rest = [s for s in models if s.id not in MODEL_LABELS]
    ordered = sorted(labeled, key=lambda s: list(MODEL_LABELS).index(s.id)) + rest
    for spec in ordered:
        # current_id = явный выбор пользователя; если его нет, подсвечиваем модель режима
        mark = "✅ " if spec.id == (current_id or mode_model_id) else ""
        label = MODEL_LABELS.get(spec.id, spec.title)
        builder.button(text=f"{mark}{label}", callback_data=f"model:{spec.id}")
    if photo:
        builder.button(text=BTN_PHOTO_MODELS, callback_data="imodel:menu")
    builder.button(text="✖ Закрыть", callback_data="panel:close")
    builder.adjust(1)
    return builder.as_markup()


def reserve_model_keyboard(
    models: list[ModelSpec], current_id: str | None, photo: bool = False
) -> InlineKeyboardMarkup:
    """/model picker shown while the emergency reserve switch is on: `models` in strength order."""
    builder = InlineKeyboardBuilder()
    for spec in models:
        mark = "✅ " if spec.id == current_id else ""
        builder.button(text=f"{mark}{spec.title}", callback_data=f"rmodel:{spec.id}")
    if photo:
        builder.button(text=BTN_PHOTO_MODELS, callback_data="imodel:menu")
    builder.button(text="✖ Закрыть", callback_data="panel:close")
    builder.adjust(1)
    return builder.as_markup()


BTN_PHOTO_MODELS = "🖼 Модели для фото »"
IMAGE_MODEL_LABELS = {
    ImageModel.GEMINI: "Gemini · Nano Banana 🍌",
    ImageModel.FLUX: "FLUX · Cloudflare ⚡",
}


def image_model_keyboard(current: ImageModel, gemini: bool) -> InlineKeyboardMarkup:
    """The photo tab of /model; without a Gemini gateway only FLUX is offered."""
    builder = InlineKeyboardBuilder()
    for model, label in IMAGE_MODEL_LABELS.items():
        if model is ImageModel.GEMINI and not gemini:
            continue
        # Without Gemini everyone draws with FLUX, whatever they picked before.
        mark = "✅ " if model is current or (not gemini and model is ImageModel.FLUX) else ""
        builder.button(text=f"{mark}{label}", callback_data=f"imodel:{model}")
    builder.button(text="« Назад к моделям", callback_data="imodel:back")
    builder.adjust(1)
    return builder.as_markup()


def approve_keyboard(tg_id: int) -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    builder.button(text="✅ Одобрить", callback_data=f"approve:{tg_id}")
    builder.button(text="🚫 Отклонить", callback_data=f"reject:{tg_id}")
    builder.adjust(2)
    return builder.as_markup()


# ---- admin panel ------------------------------------------------------------------------------


def admin_panel_keyboard() -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    builder.button(text="📋 Заявки на доступ", callback_data="admin:pending")
    builder.button(text="👥 Пользователи", callback_data="admin:users")
    builder.button(text="➕ Добавить по ID", callback_data="admin:add_prompt")
    builder.button(text="📊 Статистика", callback_data="admin:stats")
    builder.button(text="🏆 Рейтинг пользователей", callback_data="admin:top:requests")
    builder.adjust(1)
    return builder.as_markup()


LEADERBOARD_LABELS = {"requests": "Запросы", "tokens": "Токены", "last": "Активность"}


def leaderboard_keyboard(order: str) -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    for key, label in LEADERBOARD_LABELS.items():
        builder.button(text=("✅ " if key == order else "") + label, callback_data=f"admin:top:{key}")
    builder.button(text="« В меню", callback_data="admin:menu")
    builder.adjust(3, 1)
    return builder.as_markup()


def admin_user_keyboard(tg_id: int, is_blocked: bool) -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    if is_blocked:
        builder.button(text="🔓 Разблокировать", callback_data=f"admin:unblock:{tg_id}")
    else:
        builder.button(text="🚫 Заблокировать", callback_data=f"admin:block:{tg_id}")
    builder.button(text="« Назад", callback_data="admin:users")
    builder.adjust(1)
    return builder.as_markup()


def admin_users_keyboard(users: list[tuple[int, str]]) -> InlineKeyboardMarkup:
    """users: (tg_id, display label)."""
    builder = InlineKeyboardBuilder()
    for tg_id, label in users:
        builder.button(text=label, callback_data=f"admin:user:{tg_id}")
    builder.button(text="« В меню", callback_data="admin:menu")
    builder.adjust(1)
    return builder.as_markup()
