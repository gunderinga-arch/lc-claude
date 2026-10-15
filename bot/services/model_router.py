"""Picks the model for a user's request."""

from __future__ import annotations

from bot.catalog import Catalog, ModelSpec
from bot.domain import User


class ModelRouter:
    def __init__(self, catalog: Catalog, reserve_default: str | None = None) -> None:
        self._catalog = catalog
        self._reserve_default = reserve_default

    def resolve(self, user: User, reserve: bool = False) -> ModelSpec:
        """The model for a request.

        Normal (`reserve=False`): `user.model_override` if set and available, else the mode's model.
        Reserve (`reserve=True`): `user.reserve_model` if it is an available reserve model, else the
        configured `reserve_default` if it is one, else the strongest available reserve model. A
        catalog with no reserve pool at all falls back to the normal resolution.
        """
        if reserve:
            pool = self._catalog.reserve_selectable()
            if pool:
                return self._pick_reserve(user, pool)
        override = user.model_override
        if override is not None and self._catalog.is_available(override):
            return self._catalog.get(override)
        return self._catalog.for_mode(user.mode)

    def _pick_reserve(self, user: User, pool: list[ModelSpec]) -> ModelSpec:
        choice = user.reserve_model
        if choice is not None and self._is_reserve_available(choice):
            return self._catalog.get(choice)
        if self._reserve_default is not None and self._is_reserve_available(self._reserve_default):
            return self._catalog.get(self._reserve_default)
        return pool[0]

    def _is_reserve_available(self, model_id: str) -> bool:
        return self._catalog.is_available(model_id) and self._catalog.get(model_id).reserve
