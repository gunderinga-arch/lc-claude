"""Daily request quota: per role, with an optional personal limit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from bot.domain import Role, User
from bot.storage.repositories import UsageRepo

QuotaReason = Literal["ok", "pending", "blocked", "daily_limit"]


@dataclass(frozen=True, slots=True)
class QuotaDecision:
    allowed: bool
    reason: QuotaReason
    used: int  # requests today
    limit: int | None  # None = unlimited (admins)


class QuotaService:
    def __init__(self, usage: UsageRepo, *, daily_limit: int, daily_images: int = 6) -> None:
        self._usage = usage
        self._daily_limit = daily_limit
        self._daily_images = daily_images

    async def check(self, user: User) -> QuotaDecision:
        limit = user.daily_limit if user.daily_limit is not None else self._daily_limit
        if user.role is Role.PENDING:
            return QuotaDecision(allowed=False, reason="pending", used=0, limit=limit)
        if user.role is Role.BLOCKED:
            return QuotaDecision(allowed=False, reason="blocked", used=0, limit=limit)

        used = await self._usage.requests_today(user.tg_id)
        if user.role is Role.ADMIN:
            return QuotaDecision(allowed=True, reason="ok", used=used, limit=None)

        if used < limit:
            return QuotaDecision(allowed=True, reason="ok", used=used, limit=limit)
        return QuotaDecision(allowed=False, reason="daily_limit", used=used, limit=limit)

    async def record(self, user_id: int, approx_tokens: int) -> None:
        await self._usage.increment(user_id, approx_tokens)

    async def images_left(self, user: User) -> int | None:
        """Pictures the user may still have drawn today; None = unlimited (admins)."""
        if user.role is Role.ADMIN:
            return None
        return max(0, self._daily_images - await self._usage.images_today(user.tg_id))

    async def record_images(self, user_id: int, count: int) -> None:
        if count:
            await self._usage.add_images(user_id, count)

    @property
    def daily_images(self) -> int:
        return self._daily_images
