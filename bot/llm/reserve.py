"""Emergency switch: send every model call to the reserve gateway instead of BuyTokens.

The admin bot flips one row of `app_flags`; the main bot reads it on every provider call, so the
switch takes effect on the very next request of any user, without a restart. The catalog, modes,
fallbacks and retries stay as they are: only the wire target and the model name change.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Literal

from bot.catalog import Catalog, ModelSpec
from bot.llm.base import Completion, CompletionRequest, LLMProvider
from bot.llm.window import WINDOW_SLACK, fit_window
from bot.storage.repositories import FlagRepo

logger = logging.getLogger(__name__)

RESERVE_FLAG = "llm_reserve"
_ON = "on"
_AUTO = "auto"  # flipped by the admin bot itself, not a human; counts as "on" everywhere
_OFF = "off"
_VALUES = frozenset((_ON, _AUTO, _OFF))

# Comma-separated models the admin bot's last probe round found dead; the main bot hides them.
DOWN_FLAG = "models_down"
# A round older than this means the admin bot stopped probing: stale verdicts hide nothing.
DOWN_STALE_AFTER = timedelta(hours=1)


async def load_down(flags: FlagRepo, now: datetime) -> frozenset[str]:
    """Models to hide right now; empty when never probed or the last round is stale."""
    row = await flags.get(DOWN_FLAG)
    if row is None or now - row[1] > DOWN_STALE_AFTER:
        return frozenset()
    return frozenset(m for m in row[0].split(",") if m)


async def track_down_models(flags: FlagRepo, catalog: Catalog, interval_s: float = 60.0) -> None:
    """Keep `catalog`'s hidden set in sync with the admin bot's probes until cancelled."""
    while True:
        try:
            catalog.set_down(await load_down(flags, datetime.now(UTC)))
        except Exception:  # a DB hiccup keeps the previous set
            logger.warning("down-models flag read failed", exc_info=True)
        await asyncio.sleep(interval_s)


class ReserveSwitch:
    """The `llm_reserve` flag. A database hiccup reads as "off": BuyTokens is the safe default."""

    def __init__(self, flags: FlagRepo) -> None:
        self._flags = flags

    async def mode(self) -> tuple[Literal["off", "on", "auto"], datetime | None]:
        """(flag value, when it was last flipped); ("off", None) if it never was or on a DB error."""
        try:
            row = await self._flags.get(RESERVE_FLAG)
        except Exception:
            logger.warning("reserve flag read failed, using the primary gateway", exc_info=True)
            return _OFF, None
        if row is None:
            return _OFF, None
        value, updated_at = row
        if value not in _VALUES:
            value = _OFF
        return value, updated_at  # type: ignore[return-value]

    async def state(self) -> tuple[bool, datetime | None]:
        """(on, when it was last flipped); "auto" counts as on, same as "on"."""
        value, updated_at = await self.mode()
        return value != _OFF, updated_at

    async def active(self) -> bool:
        on, _ = await self.state()
        return on

    async def set(self, on: bool, *, auto: bool = False) -> None:
        value = _OFF if not on else (_AUTO if auto else _ON)
        await self._flags.set(RESERVE_FLAG, value)


class SwitchingProvider:
    """`LLMProvider` that routes to `reserve` while the switch is on, or for a request already
    naming one of `reserve_ids` (the caller, usually `ChatService`, already decided to use it).
    """

    def __init__(
        self,
        primary: LLMProvider,
        reserve: LLMProvider,
        switch: ReserveSwitch,
        reserve_model: str,
        reserve_ids: frozenset[str] = frozenset(),
        reserve_spec: ModelSpec | None = None,
    ) -> None:
        """`reserve_spec` (the reserve model's catalog entry) fits a rerouted request into its window."""
        self._primary = primary
        self._reserve = reserve
        self._switch = switch
        self._reserve_model = reserve_model
        self._reserve_ids = reserve_ids
        self._reserve_spec = reserve_spec

    async def complete(self, request: CompletionRequest) -> Completion:
        if request.model in self._reserve_ids:
            completion = await self._reserve.complete(request)
            return replace(completion, reserve=True)
        if not await self._switch.active():
            return await self._primary.complete(request)
        # The rewrite step and anything else that still names a primary model while the switch is
        # on: sent to the reserve anyway, with its own default model, cut to the reserve's window (the client
        # fitted it to the primary model's ~1M).
        rerouted = replace(request, model=self._reserve_model)
        if self._reserve_spec is not None:
            spec = self._reserve_spec
            room = min(request.max_tokens, spec.max_tokens)
            limit = max(1, int(spec.context_tokens * (1 - WINDOW_SLACK)) - room)
            rerouted = replace(
                rerouted,
                messages=await asyncio.to_thread(fit_window, request.messages, limit, spec.tokenizer),
                max_tokens=room,
            )
        completion = await self._reserve.complete(rerouted)
        # OmniRoute reports the combo member that answered; keep it for the header and the stats.
        return replace(completion, reserve=True)

    async def list_models(self) -> list[str]:
        return await self._primary.list_models()

    async def aclose(self) -> None:
        await self._primary.aclose()
        await self._reserve.aclose()
