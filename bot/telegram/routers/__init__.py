"""All bot routers, ordered: commands first, plain chat last."""

from __future__ import annotations

from aiogram import Router

from bot.telegram.routers import admin, chat, commands, settings


def build_router() -> Router:
    router = Router(name="root")
    router.include_router(commands.router)
    router.include_router(admin.router)
    router.include_router(settings.router)
    router.include_router(chat.router)
    return router
