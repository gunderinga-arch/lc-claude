"""Entry point: config, DI wiring and polling."""

from __future__ import annotations

import argparse
import asyncio
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.fsm.storage.memory import MemoryStorage

from bot.catalog import load_catalog
from bot.config import Settings
from bot.files.store import FileStore
from bot.llm.base import LLMProvider
from bot.llm.buytokens import BuyTokensProvider
from bot.llm.reserve import ReserveSwitch, SwitchingProvider, track_down_models
from bot.llm.resilience import ResilientLLMClient
from bot.services.chat import TITLE_MODEL, ChatService
from bot.services.conversation import ConversationService
from bot.services.heartbeat import Heartbeat
from bot.services.images import ImageGenerator
from bot.services.jobs import JobManager
from bot.services.model_router import ModelRouter
from bot.services.quota import QuotaService
from bot.services.speech import SpeechRecognizer
from bot.services.web import WebTools
from bot.skills import load_skills
from bot.storage.db import Database
from bot.storage.repositories import (
    ConversationRepo,
    FlagRepo,
    RequestRepo,
    StatusRepo,
    UploadRepo,
    UsageRepo,
    UserRepo,
)
from bot.telegram.api import main_bot
from bot.telegram.middlewares.access import AccessMiddleware
from bot.telegram.middlewares.errors import ErrorMiddleware
from bot.telegram.middlewares.thread import ThreadMiddleware
from bot.telegram.routers import build_router


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
    logger = logging.getLogger(__name__)

    catalog = load_catalog(settings.models_file)
    skills = load_skills(settings.skills_file)
    serper = settings.serper_api_key.get_secret_value() if settings.serper_api_key else None
    web = WebTools(settings.searxng_url, serper_key=serper) if settings.searxng_url else None
    if web is None and any(s.web for s in skills.values()):
        logger.warning("SEARXNG_URL is not set: internet skills are hidden")
        skills = {k: s for k, s in skills.items() if not s.web}
    images = (
        ImageGenerator(
            settings.cloudflare_account_id,
            settings.cloudflare_api_token.get_secret_value(),
            # Gemini pictures go through the reserve gateway (OmniRoute); without it only FLUX is offered.
            # A failed Gemini picture falls back to FLUX; the server's signed-in profile feeds it.
            gemini_url=settings.reserve_base_url,
            gemini_key=settings.reserve_api_key.get_secret_value() if settings.reserve_api_key else "",
            # Written by the server's gemini-refresh timer: gemini-web links open only with these cookies.
            gemini_cookies=settings.db_path.parent / "gemini-cookies.json",
        )
        if settings.cloudflare_account_id and settings.cloudflare_api_token
        else None
    )
    if images is None and any(s.images for s in skills.values()):
        logger.warning("CLOUDFLARE_ACCOUNT_ID / CLOUDFLARE_API_TOKEN are not set: image skills are hidden")
        skills = {k: s for k, s in skills.items() if not s.images}
    db = Database(settings.db_path)
    await db.connect()

    provider: LLMProvider = BuyTokensProvider(
        settings.buytokens_base_url, settings.buytokens_api_key.get_secret_value()
    )
    # The admin bot's emergency switch: with a reserve gateway configured, every call checks the flag.
    reserve = ReserveSwitch(FlagRepo(db)) if settings.reserve_configured else None
    # Every reserve-pool id goes to the reserve gateway by name, service ones (chat titles) too.
    reserve_ids = frozenset(spec.id for spec in catalog.models() if spec.reserve)
    if reserve is not None:
        assert settings.reserve_base_url is not None  # reserve_configured
        if reserve_ids and settings.reserve_model not in reserve_ids:
            logger.warning(
                "RESERVE_MODEL %r is not a reserve model in the catalog, falling back to the strongest one",
                settings.reserve_model,
            )
        reserve_key = settings.reserve_api_key.get_secret_value() if settings.reserve_api_key else ""
        provider = SwitchingProvider(
            provider,
            BuyTokensProvider(settings.reserve_base_url, reserve_key),  # any OpenAI-compatible gateway
            reserve,
            settings.reserve_model,
            reserve_ids=reserve_ids,
            reserve_spec=catalog.get(settings.reserve_model)
            if settings.reserve_model in reserve_ids
            else None,
        )
    # The whole fallback chain failed (main gateway down, timeouts): one last try on the reserve gateway,
    # so the user gets an answer before the admin bot's probes switch everyone to the reserve.
    rescue = settings.reserve_model if reserve is not None and settings.reserve_model in reserve_ids else None
    # Haiku 5.5 on the reserve gateway mends the seams of long answers written in parts (~700 chars a side).
    glue = TITLE_MODEL if reserve is not None and TITLE_MODEL in reserve_ids else None
    llm = ResilientLLMClient(
        provider, catalog, max_concurrency=settings.max_concurrency, rescue_model=rescue, glue_model=glue
    )

    users = UserRepo(db)
    # Full texts of the chats' big files, for the model's file tools (data/ is the unit's writable dir).
    files = FileStore(settings.db_path.parent / "files")
    conversations = ConversationService(
        ConversationRepo(db),
        system_prompt=settings.system_prompt,
        token_budget=settings.history_token_budget,
        skills=skills,
        idle_hours=settings.context_idle_hours,
        files=files,
    )
    quota = QuotaService(
        UsageRepo(db), daily_limit=settings.daily_request_limit, daily_images=settings.daily_image_limit
    )
    requests = RequestRepo(db)
    router = ModelRouter(catalog, reserve_default=settings.reserve_model)
    chat = ChatService(
        llm,
        conversations,
        quota,
        router,
        requests,
        catalog,
        web,
        images,
        reserve=reserve,
        reserve_token_budget=settings.reserve_token_budget,
        files=files,
    )
    jobs = JobManager()
    speech = SpeechRecognizer(settings.stt_model, settings.stt_models_dir) if settings.stt_model else None
    # The admin bot reads this row: without it, a hung process looks exactly like a healthy one.
    heartbeat = Heartbeat(StatusRepo(db), queue_depth=llm.queue_depth, active_jobs=jobs.active_count)

    # The separate admin bot: with a token set, access-request notices go through it first.
    notifier = (
        Bot(
            settings.admin_bot_token.get_secret_value(),
            default=DefaultBotProperties(parse_mode=ParseMode.HTML),
        )
        if settings.admin_bot_token
        else None
    )

    bot = main_bot(settings)
    dp = Dispatcher(storage=MemoryStorage())
    dp.update.middleware(ErrorMiddleware())
    dp.update.middleware(AccessMiddleware(users, settings, notifier))
    dp.update.middleware(ThreadMiddleware(conversations))
    dp.include_router(build_router())

    # Dependencies injected into every handler by name.
    dp["chat"] = chat
    dp["llm"] = llm
    dp["quota"] = quota
    dp["jobs"] = jobs
    dp["users"] = users
    dp["conversations"] = conversations
    dp["catalog"] = catalog
    dp["skills"] = skills
    dp["requests"] = requests
    dp["uploads"] = UploadRepo(db)
    dp["speech"] = speech
    dp["settings"] = settings
    dp["reserve"] = reserve
    dp["router"] = router
    dp["images"] = images

    await bot.delete_webhook(drop_pending_updates=True)
    await heartbeat.start()
    warm_up = asyncio.create_task(speech.warm_up()) if speech else None  # keep a reference
    # Models the admin bot's probes found dead leave /model and the fallback chains until they answer.
    down_models = asyncio.create_task(track_down_models(FlagRepo(db), catalog))
    logger.info("polling started")
    try:
        await dp.start_polling(bot)
    finally:
        down_models.cancel()
        await heartbeat.stop()
        if warm_up is not None:
            warm_up.cancel()
        await jobs.shutdown()
        await provider.aclose()
        if web is not None:
            await web.aclose()
        if images is not None:
            await images.aclose()
        if notifier is not None:
            await notifier.session.close()
        await db.close()
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
