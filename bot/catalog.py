"""Model catalog: which gateway models exist, their limits, modes and fallback chains."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

from bot.domain import TOKEN_WEIGHTS, Mode

logger = logging.getLogger(__name__)

DEFAULT_CONTEXT_TOKENS = 128_000  # a model without a measured window in models.yaml


class CatalogError(ValueError):
    """The catalog file is inconsistent."""


class UnknownModelError(KeyError):
    pass


@dataclass(frozen=True, slots=True)
class ModelSpec:
    id: str
    title: str
    max_tokens: int
    timeout_s: float
    fallback: str | None
    enabled: bool
    slow: bool  # answers take minutes: warn the user up front
    note: str = ""
    price: float | None = None  # per 1M tokens; None = the global TOKEN_PRICE
    reserve: bool = False  # lives in `reserve_models:`, offered only while the emergency switch is on
    probe: bool = True  # pinged by the admin bot's prober; false for weekly-quota models the pings would eat
    service: bool = False  # internal calls only (chat titles): never offered in /model
    # Not a gateway model: the bot plans the answer and picks skills with this model, then it answers.
    pipeline: str | None = None
    # Optional gateway model that executes a pipeline's plan. When omitted, `pipeline` answers too.
    answer: str | None = None
    # The model's window in our token estimate (bot.domain.count_tokens): prompt + max_tokens fit in it.
    context_tokens: int = DEFAULT_CONTEXT_TOKENS
    tokenizer: str = "claude"  # whose token counts the estimate follows: "claude" or "gpt"


class Catalog:
    def __init__(
        self,
        models: Mapping[str, ModelSpec],
        modes: Mapping[Mode, str],
        *,
        max_tokens_cap: int,
        max_attempts: int,
    ) -> None:
        self._models = dict(models)
        self._modes = dict(modes)
        self.max_tokens_cap = max_tokens_cap
        self.max_attempts = max_attempts
        self._unavailable: set[str] = set()  # disabled at runtime, e.g. after a 403
        self._down: frozenset[str] = frozenset()  # failed the admin bot's last probe round
        self._validate()

    def _validate(self) -> None:
        for mode in Mode:
            model_id = self._modes.get(mode)
            if model_id is None:
                raise CatalogError(f"mode {mode} has no model")
            spec = self._models.get(model_id)
            if spec is None or not spec.enabled or spec.reserve:
                raise CatalogError(f"mode {mode} points to unknown, disabled or reserve model {model_id}")
        for spec in self._models.values():
            if spec.fallback is not None:
                fallback = self._models.get(spec.fallback)
                if fallback is None:
                    raise CatalogError(f"{spec.id}: unknown fallback {spec.fallback}")
                if spec.reserve != fallback.reserve:
                    raise CatalogError(f"{spec.id}: fallback {spec.fallback} crosses the reserve boundary")
            if spec.max_tokens > self.max_tokens_cap:
                raise CatalogError(f"{spec.id}: max_tokens exceeds max_tokens_cap")
            if spec.max_tokens >= spec.context_tokens:
                raise CatalogError(f"{spec.id}: max_tokens must leave room for the prompt in context_tokens")
            if spec.tokenizer not in TOKEN_WEIGHTS:
                raise CatalogError(f"{spec.id}: unknown tokenizer {spec.tokenizer}")
            if spec.pipeline is not None:
                worker = self._models.get(spec.pipeline)
                if worker is None or worker.pipeline is not None or worker.reserve or spec.reserve:
                    raise CatalogError(f"{spec.id}: pipeline must name a plain gateway model")
                if spec.probe or spec.fallback is not None:
                    raise CatalogError(f"{spec.id}: a pipeline has no fallback and is not probed")
            if spec.answer is not None:
                worker = self._models.get(spec.answer)
                if worker is None or worker.pipeline is not None or worker.reserve or spec.reserve:
                    raise CatalogError(f"{spec.id}: answer must name a plain gateway model")
                if spec.pipeline is None:
                    raise CatalogError(f"{spec.id}: answer requires pipeline")
        if self.max_attempts < 1:
            raise CatalogError("max_attempts must be >= 1")

    def get(self, model_id: str) -> ModelSpec:
        try:
            return self._models[model_id]
        except KeyError:
            raise UnknownModelError(model_id) from None

    def for_mode(self, mode: Mode) -> ModelSpec:
        return self._models[self._modes[mode]]

    def is_available(self, model_id: str) -> bool:
        spec = self._models.get(model_id)
        if spec is None or not spec.enabled or model_id in self._unavailable or model_id in self._down:
            return False
        return (spec.pipeline is None or self.is_available(spec.pipeline)) and (
            spec.answer is None or self.is_available(spec.answer)
        )

    def mark_unavailable(self, model_id: str) -> None:
        self._unavailable.add(model_id)

    def set_down(self, model_ids: frozenset[str]) -> None:
        """Replace the probe-failed set: these models leave the choice and the fallback chains."""
        if model_ids != self._down:
            logger.info("models hidden after probes: %s", ", ".join(sorted(model_ids)) or "none")
        self._down = model_ids

    def models(self) -> list[ModelSpec]:
        """Every model in the file, disabled ones too, in catalog order."""
        return list(self._models.values())

    def selectable(self) -> list[ModelSpec]:
        """Models a user may pick with /model, in catalog order.

        If the probes hid every one of them (the main gateway was down; seen 14.10.2026 when the reserve
        was switched off by hand before the next probe round), the menu shows them anyway, not empty buttons.
        """
        picked = [
            spec
            for spec in self._models.values()
            if self.is_available(spec.id) and not spec.reserve and not spec.service
        ]
        if not picked and self._down:
            down, self._down = self._down, frozenset()
            try:
                return self.selectable()
            finally:
                self._down = down
        return picked

    def reserve_selectable(self) -> list[ModelSpec]:
        """Reserve-gateway models a user may pick while the emergency switch is on, strongest first."""
        return [
            spec
            for spec in self._models.values()
            if self.is_available(spec.id) and spec.reserve and not spec.service
        ]

    def fallback_chain(self, model_id: str) -> list[ModelSpec]:
        """`model_id` followed by its fallbacks, skipping unavailable models, without cycles."""
        chain: list[ModelSpec] = []
        seen: set[str] = set()
        current: str | None = model_id
        while current is not None and current not in seen:
            seen.add(current)
            spec = self.get(current)
            if self.is_available(current):
                chain.append(spec)
            current = spec.fallback
        if not chain and self._down:  # the probes hid every model here: trying beats refusing outright
            down, self._down = self._down, frozenset()
            try:
                return self.fallback_chain(model_id)
            finally:
                self._down = down
        return chain


def _load_models(items: Mapping, *, reserve: bool) -> dict[str, ModelSpec]:
    return {
        model_id: ModelSpec(
            id=model_id,
            title=str(item.get("title", model_id)),
            max_tokens=int(item["max_tokens"]),
            timeout_s=float(item["timeout_s"]),
            fallback=item.get("fallback"),
            # Reserve entries have no `enabled` key: the reserve pool is always on, availability is
            # only about whether the model actually answers (tracked at runtime, not in the file).
            enabled=True if reserve else bool(item.get("enabled", True)),
            slow=bool(item.get("slow", False)),
            note=str(item.get("note", "")),
            price=float(item["price"]) if item.get("price") is not None else None,
            reserve=reserve,
            probe=bool(item.get("probe", True)),
            service=bool(item.get("service", False)),
            pipeline=item.get("pipeline"),
            answer=item.get("answer"),
            context_tokens=int(item.get("context_tokens", DEFAULT_CONTEXT_TOKENS)),
            tokenizer=str(item.get("tokenizer", "claude")),
        )
        for model_id, item in items.items()
    }


def load_catalog(path: Path) -> Catalog:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    try:
        models = _load_models(raw["models"], reserve=False)
        reserve_models = _load_models(raw.get("reserve_models") or {}, reserve=True)
        clash = set(models) & set(reserve_models)
        if clash:
            raise CatalogError(f"model ids in both models: and reserve_models: {sorted(clash)}")
        modes = {Mode(name): model_id for name, model_id in raw["modes"].items()}
        return Catalog(
            {**models, **reserve_models},
            modes,
            max_tokens_cap=int(raw.get("max_tokens_cap", 128_000)),
            max_attempts=int(raw.get("max_attempts", 3)),
        )
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, CatalogError):
            raise
        raise CatalogError(f"invalid catalog {path}: {exc!r}") from exc
