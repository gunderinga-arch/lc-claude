"""Background job manager and typing-heartbeat notifier (pure asyncio, no Telegram)."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable, Coroutine, Hashable
from typing import Any

logger = logging.getLogger(__name__)


class JobBusyError(Exception):
    """The chat already has a running job."""


class JobManager:
    """One running job per key: the caller picks it, e.g. (user id, topic) so chats run side by side."""

    def __init__(self) -> None:
        self._jobs: dict[Hashable, asyncio.Task[None]] = {}
        self._closed = False

    def submit(self, user_id: Hashable, coro: Coroutine[Any, Any, None]) -> asyncio.Task[None]:
        if self._closed:
            coro.close()
            raise RuntimeError("JobManager is shut down")
        if user_id in self._jobs:
            coro.close()
            raise JobBusyError(f"{user_id} already has a running job")
        task = asyncio.create_task(coro)
        self._jobs[user_id] = task
        task.add_done_callback(lambda t: self._on_done(user_id, t))
        return task

    def _on_done(self, user_id: Hashable, task: asyncio.Task[None]) -> None:
        if self._jobs.get(user_id) is task:
            del self._jobs[user_id]
        exc = task.exception() if not task.cancelled() else None
        if exc is not None:
            logger.error("job %s failed", user_id, exc_info=exc)

    def active_count(self) -> int:
        return len(self._jobs)

    def is_running(self, user_id: Hashable) -> bool:
        return user_id in self._jobs

    def cancel(self, user_id: Hashable) -> bool:
        task = self._jobs.get(user_id)
        if task is None:
            return False
        return task.cancel()

    async def shutdown(self, timeout_s: float = 10.0) -> None:
        self._closed = True
        tasks = list(self._jobs.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.wait(tasks, timeout=timeout_s)


class ProgressNotifier:
    """async with ProgressNotifier(send): calls `send()` immediately, then every interval_s."""

    def __init__(self, send: Callable[[], Awaitable[None]], interval_s: float = 4.0) -> None:
        self._send = send
        self._interval_s = interval_s
        self._task: asyncio.Task[None] | None = None

    async def _safe_send(self) -> None:
        try:
            await self._send()
        except Exception:
            logger.warning("ProgressNotifier send failed", exc_info=True)

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(self._interval_s)
            await self._safe_send()

    async def __aenter__(self) -> ProgressNotifier:
        await self._safe_send()
        self._task = asyncio.create_task(self._loop())
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
