"""The main bot's transport; the admin bot keeps its own cloud connection."""

from aiogram import Bot
from aiogram.client.default import DefaultBotProperties
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.client.telegram import TelegramAPIServer
from aiogram.enums import ParseMode

from bot.config import Settings


def main_bot(settings: Settings) -> Bot:
    session = None
    if settings.telegram_api_base:
        session = AiohttpSession(
            api=TelegramAPIServer.from_base(settings.telegram_api_base, is_local=True),
            timeout=300,
        )
    return Bot(
        settings.telegram_bot_token.get_secret_value(),
        session=session,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )


def download_limit(bot: Bot) -> int:
    # Limit in-memory extraction on this 4 GB VPS even though local Bot API permits larger files.
    api = getattr(getattr(bot, "session", None), "api", None)
    return (100 if getattr(api, "is_local", False) is True else 20) * 1024 * 1024
