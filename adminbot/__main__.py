"""Admin bot entry point: python -m adminbot [--log-file PATH].

A second Telegram bot (ADMIN_BOT_TOKEN) over the main bot's SQLite file. The main bot keeps polling
its own token; this process only reads and writes the shared database, and uses the main bot's token
to send notices to users (access granted, blocked), because users talk to the main bot, not this one.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
from datetime import timedelta, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import BotCommand, InlineKeyboardMarkup

from adminbot import balance, chats, fullbackup, health, home, messaging, reserve, twin, usage, users
from adminbot.access import AdminOnlyMiddleware
from adminbot.alerts import AlertService, start_alerts
from adminbot.probes import ModelProber, start_probes
from adminbot.story import (
    STORY_PREVIEW_INTERVAL_S,
    TWINBOT_DATA_DIR,
    StoryPreviewPoller,
    preview_path,
    start_story_previews,
)
from bot.catalog import CatalogError, load_catalog
from bot.config import Settings
from bot.llm.buytokens import BuyTokensProvider
from bot.llm.reserve import ReserveSwitch
from bot.storage.db import Database
from bot.storage.repositories import (
    ConversationRepo,
    FlagRepo,
    NoticeRepo,
    RequestRepo,
    StatusRepo,
    UploadRepo,
    UsageRepo,
    UserRepo,
)
from bot.telegram.api import main_bot as create_main_bot
from bot.telegram.middlewares.errors import ErrorMiddleware

COMMANDS = [
    BotCommand(command="start", description="Главное меню"),
    BotCommand(command="cancel", description="Отменить ввод"),
]


async def _notify_story(settings: Settings, admin_bot: Bot, text: str, markup: InlineKeyboardMarkup) -> int:
    """Deliver one story preview to every admin; a failure is logged, not fatal."""

    delivered = 0
    for admin_id in sorted(settings.admin_ids):
        try:
            await admin_bot.send_message(admin_id, text, reply_markup=markup)
        except Exception as exc:  # one admin must not mute the others, same as AlertService
            logging.getLogger(__name__).warning("admin %s did not get the story preview: %s", admin_id, exc)
            continue
        delivered += 1
    return delivered


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-file", type=Path, help="log to a rotating file instead of stderr")
    args = parser.parse_args()

    settings = Settings()  # type: ignore[call-arg]  # values come from env/.env
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if args.log_file is not None:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers = [RotatingFileHandler(args.log_file, maxBytes=1_000_000, backupCount=3, encoding="utf-8")]
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
    )
    if settings.admin_bot_token is None:
        raise SystemExit("ADMIN_BOT_TOKEN is not set: create a second bot with @BotFather, put it in .env")
    if not settings.admin_ids:
        raise SystemExit("ADMIN_IDS is empty: nobody could use the admin bot")

    db = Database(settings.db_path)
    await db.connect()
    defaults = DefaultBotProperties(parse_mode=ParseMode.HTML)
    admin_bot = Bot(settings.admin_bot_token.get_secret_value(), default=defaults)
    main_bot = create_main_bot(settings)

    # One repo per table, shared by the handlers and the alert loop: they all wrap the same connection.
    users_repo = UserRepo(db)
    requests = RequestRepo(db)
    uploads = UploadRepo(db)
    statuses = StatusRepo(db)

    # The catalog is only needed for per-model prices; a broken models.yaml must not stop the panel.
    try:
        catalog = load_catalog(settings.models_file)
    except (CatalogError, OSError) as exc:
        logging.getLogger(__name__).warning("models.yaml недоступен, цены будут общими: %s", exc)
        catalog = None

    dp = Dispatcher(storage=MemoryStorage())
    dp.update.outer_middleware(AdminOnlyMiddleware(settings.admin_ids))
    dp.update.middleware(ErrorMiddleware())
    # users goes last on purpose: its catch-all text handler treats any other text as a search
    # query, so every router that expects typed input (messaging) must be included before it.
    # reserve goes before it for the same reason it is listed before users below: no typed input of
    # its own, but the rule is simplest kept uniform.
    routers = (
        home.router,
        usage.router,
        chats.router,
        messaging.router,
        health.router,
        reserve.router,
        twin.router,
        fullbackup.router,
        balance.router,
        users.router,
    )
    dp.include_routers(*routers)

    # Dependencies injected into every handler by name.
    dp["users"] = users_repo
    dp["requests"] = requests
    dp["daily"] = UsageRepo(db)
    dp["uploads"] = uploads
    dp["dialogs"] = ConversationRepo(db)
    dp["statuses"] = statuses
    dp["db"] = db  # the backup screen snapshots the shared file
    dp["catalog"] = catalog  # per-model price overrides; None falls back to the global price
    dp["main_bot"] = main_bot
    dp["settings"] = settings
    dp["tz"] = timezone(timedelta(hours=settings.admin_utc_offset))

    reserve_switch = ReserveSwitch(FlagRepo(db))
    dp["reserve_switch"] = reserve_switch

    # The primary and (if configured) reserve providers, used only to probe models: the main bot
    # has its own instances and its own routing (bot.llm.reserve.SwitchingProvider).
    buytokens_key = settings.buytokens_api_key.get_secret_value()
    primary_provider = BuyTokensProvider(settings.buytokens_base_url, buytokens_key)
    reserve_key = settings.reserve_api_key.get_secret_value() if settings.reserve_api_key else ""
    reserve_provider = (
        BuyTokensProvider(settings.reserve_base_url, reserve_key) if settings.reserve_configured else None
    )
    prober = (
        ModelProber(primary_provider, catalog, reserve_provider, settings.reserve_model)
        if catalog is not None
        else None
    )
    dp["prober"] = prober

    await admin_bot.set_my_commands(COMMANDS)
    await admin_bot.delete_webhook(drop_pending_updates=True)

    # Watches the shared database and warns the admins: gateway failures, overspending users,
    # a silent main bot, external quotas running out, and a daily digest. Started before polling
    # so the first pass lands while the bot is already answering.
    alerts = AlertService(
        admin_bot,
        settings,
        requests,
        uploads,
        statuses,
        NoticeRepo(db),
        tz=dp["tz"],
    )
    alerts_task = start_alerts(alerts)

    # Same idea, but for the models themselves: probes every model on a schedule and alerts on
    # up/down changes, and publishes the dead ones so the main bot hides them from the choice.
    # Off when there is no prober (broken catalog) or MODEL_PROBE_MINUTES=0.
    probe_interval_s = settings.model_probe_minutes * 60
    probes_task = (
        start_probes(
            prober, alerts, reserve_switch, settings.reserve_configured, probe_interval_s, flags=FlagRepo(db)
        )
        if prober is not None and settings.model_probe_minutes > 0
        else None
    )

    # The twin bot's story preview: twinbot generates the story but does not send it; the loop
    # watches its preview file and shows a fresh story to the admins with Send/Cancel buttons.
    # A send callback here (adminbot.twin) turns the button press into a verdict file.
    story_poller = StoryPreviewPoller(
        preview_path(TWINBOT_DATA_DIR),
        lambda text, markup: _notify_story(settings, admin_bot, text, markup),
        interval_s=STORY_PREVIEW_INTERVAL_S,
    )
    story_task = start_story_previews(story_poller)

    logging.getLogger(__name__).info("admin bot polling started")
    try:
        await dp.start_polling(admin_bot)
    finally:
        story_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await story_task
        alerts_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await alerts_task
        if probes_task is not None:
            probes_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await probes_task
        await db.close()
        await primary_provider.aclose()
        if reserve_provider is not None:
            await reserve_provider.aclose()
        await main_bot.session.close()
        await admin_bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
