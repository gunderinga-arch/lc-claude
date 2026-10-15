"""Tokens left on the configured gateway keys, as shown on its account page.

The OpenAI-style API has no balance call; the site's page posts the key to /public/key-info and
gets quota (total, used, remaining) and usage (today, week, month) back. The main key feeds the
bots, the second one (BUYTOKENS_API_KEY_2) the benchmarks.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx
from aiogram import Router
from aiogram.types import CallbackQuery, InlineKeyboardMarkup
from aiogram.utils.keyboard import InlineKeyboardBuilder

from adminbot.callbacks import Balance, Home
from adminbot.common import show
from bot.config import Settings
from bot.render.format import short_number

logger = logging.getLogger(__name__)

router = Router(name="balance")


def info_url(base_url: str) -> str:
    """Map the configured API base URL to its key-info endpoint."""
    return base_url.rstrip("/").removesuffix("/v1") + "/public/key-info"


async def fetch_info(client: httpx.AsyncClient, url: str, key: str) -> dict[str, Any] | str:
    """The site's answer, or a short reason why there is none."""
    try:
        response = await client.post(url, json={"key": key})
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as exc:
        return f"сайт ответил {exc.response.status_code}"
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("key-info %s failed: %r", url, exc)
        return "сайт не ответил"


def _num(value: Any) -> str:
    return short_number(int(value or 0))


def render_key(label: str, info: dict[str, Any] | str) -> list[str]:
    """Pure: the block of one key."""
    if isinstance(info, str):
        return [f"🔑 <b>{label}</b>", f"❌ Не удалось узнать: {info}"]
    quota, usage = info.get("quota") or {}, info.get("usage") or {}
    total, used = quota.get("total"), quota.get("used") or 0
    remaining = quota.get("remaining")
    if remaining is None and total is not None:
        remaining = max(0, total - used)
    lines = [f"🔑 <b>{label}</b>" + ("" if info.get("active") else " — ⛔️ ключ неактивен")]
    if remaining is None:
        lines.append("Общей квоты нет")
    else:
        share = f" ({100 * remaining / total:.0f}%)" if total else ""
        lines.append(f"Осталось: <b>{_num(remaining)}</b> из {_num(total)}{share}")
    lines.append(
        f"Расход: сегодня {_num(usage.get('today_tokens'))} · 7 дней {_num(usage.get('weekly_tokens'))} · "
        f"месяц {_num(usage.get('month_tokens'))}"
    )
    per_day = (usage.get("weekly_tokens") or 0) / 7
    if remaining is not None and per_day > 0:
        lines.append(f"Хватит примерно на {remaining / per_day:.0f} дн. при расходе как за неделю")
    photo = (info.get("media") or {}).get("photo") or {}
    if photo.get("quota"):
        lines.append(f"Фото: {photo.get('remaining', 0)} из {photo['quota']}")
    return lines


def render_balance(keys: list[tuple[str, dict[str, Any] | str]]) -> tuple[str, InlineKeyboardMarkup]:
    """Pure: the «Токены на ключах» screen."""
    blocks = ["🪙 <b>Токены на ключах BuyTokens</b>"]
    for label, info in keys:
        blocks.append("\n".join(render_key(label, info)))
    builder = InlineKeyboardBuilder()
    builder.button(text="🔄 Обновить", callback_data=Balance())
    builder.button(text="« В меню", callback_data=Home())
    builder.adjust(2)
    return "\n\n".join(blocks), builder.as_markup()


def configured_keys(settings: Settings) -> list[tuple[str, str]]:
    keys = [("Основной — боты", settings.buytokens_api_key.get_secret_value())]
    if settings.buytokens_api_key_2 is not None:
        keys.append(("Второй — тесты", settings.buytokens_api_key_2.get_secret_value()))
    return keys


@router.callback_query(Balance.filter())
async def cb_balance(callback: CallbackQuery, settings: Settings) -> None:
    keys = configured_keys(settings)
    url = info_url(settings.buytokens_base_url)
    async with httpx.AsyncClient(timeout=20) as client:
        infos = await asyncio.gather(*(fetch_info(client, url, key) for _, key in keys))
    text, markup = render_balance([(label, info) for (label, _), info in zip(keys, infos, strict=True)])
    await show(callback, text, markup)
