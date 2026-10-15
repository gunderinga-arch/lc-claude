"""The main bot's heartbeat: the admin bot reads it to tell "alive" from "hung"."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable
from datetime import UTC, datetime

from bot.domain import HEARTBEAT_S
from bot.storage.repositories import StatusRepo

logger = logging.getLogger(__name__)


class Heartbeat:
    """Rewrites one row of `bot_status` every HEARTBEAT_S seconds.

    A process that polls Telegram but is stuck inside a handler still answers nothing: only the
    row tells the admin bot the difference. `started_at` never changes, so the admin can show uptime.
    """

    def __init__(
        self,
        status: StatusRepo,
        *,
        queue_depth: Callable[[], int],
        active_jobs: Callable[[], int],
        interval_s: float = HEARTBEAT_S,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._status = status
        self._queue_depth = queue_depth
        self._active_jobs = active_jobs
        self._interval_s = interval_s
        self._now = now
        self._started_at = now()
        self._task: asyncio.Task[None] | None = None

    async def beat_once(self) -> None:
        """Write one heartbeat. A database hiccup is logged, never raised: the bot keeps serving."""
        try:
            await self._status.beat(
                self._started_at, queue_depth=self._queue_depth(), active_jobs=self._active_jobs()
            )
        except Exception:
            logger.warning("heartbeat write failed", exc_info=True)

    async def _loop(self) -> None:
        while True:
            await self.beat_once()
            await asyncio.sleep(self._interval_s)

    async def start(self) -> None:
        """Write the first heartbeat immediately, then keep the row fresh in the background."""
        await self.beat_once()
        if self._task is None:
            self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
