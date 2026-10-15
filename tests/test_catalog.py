from pathlib import Path

import pytest

from bot.catalog import Catalog, CatalogError, ModelSpec, load_catalog
from bot.config import DEFAULT_MODELS_FILE, Settings
from bot.domain import Mode


def _spec(model_id: str, fallback: str | None = None, enabled: bool = True) -> ModelSpec:
    return ModelSpec(model_id, model_id, 1000, 10, fallback, enabled, slow=False)


def _catalog(**models: ModelSpec) -> Catalog:
    return Catalog(models, {mode: "a" for mode in Mode}, max_tokens_cap=128_000, max_attempts=3)


def test_shipped_catalog_loads() -> None:
    catalog = load_catalog(DEFAULT_MODELS_FILE)
    assert catalog.for_mode(Mode.NORMAL).id == "claude-opus-5"
    assert catalog.for_mode(Mode.DEEP).slow
    assert {m.id for m in catalog.selectable()} == {
        "claude-fable-5.1",
        "gpt-6-sol",
        "claude-opus-5",
        "plan-opus",
    }


def test_fallback_chain_skips_unavailable_and_stops_on_cycle() -> None:
    catalog = _catalog(a=_spec("a", "b"), b=_spec("b", "c"), c=_spec("c", "a"))
    assert [m.id for m in catalog.fallback_chain("a")] == ["a", "b", "c"]
    catalog.mark_unavailable("b")
    assert [m.id for m in catalog.fallback_chain("a")] == ["a", "c"]
    assert [m.id for m in catalog.selectable()] == ["a", "c"]


def test_disabled_model_is_skipped_in_chain() -> None:
    catalog = _catalog(a=_spec("a", "x"), x=_spec("x", enabled=False))
    assert [m.id for m in catalog.fallback_chain("a")] == ["a"]


def test_mode_on_disabled_model_is_rejected() -> None:
    with pytest.raises(CatalogError):
        _catalog(a=_spec("a", enabled=False))


def test_unknown_fallback_is_rejected() -> None:
    with pytest.raises(CatalogError):
        _catalog(a=_spec("a", "missing"))


def test_broken_file_raises_catalog_error(tmp_path: Path) -> None:
    path = tmp_path / "models.yaml"
    path.write_text("modes: {}\nmodels: {}\n", encoding="utf-8")
    with pytest.raises(CatalogError):
        load_catalog(path)


def test_settings_parse_admin_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "t")
    monkeypatch.setenv("BUYTOKENS_API_KEY", "k")
    monkeypatch.setenv("ADMIN_IDS", "1, 22;333")
    monkeypatch.setenv("BUYTOKENS_BASE_URL", "https://x/v1/")
    settings = Settings(_env_file=None)
    assert settings.admin_ids == frozenset({1, 22, 333})
    assert settings.buytokens_base_url == "https://x/v1"


def test_menu_never_empty_when_probes_hid_every_main_model() -> None:
    catalog = load_catalog(DEFAULT_MODELS_FILE)
    every = {m.id for m in catalog.selectable()}
    catalog.set_down(frozenset(every))  # the main gateway was down at the last probe round
    assert {m.id for m in catalog.selectable()} == every
    catalog.set_down(frozenset({"gpt-6-sol"}))  # one down: that one is hidden as before
    assert "gpt-6-sol" not in {m.id for m in catalog.selectable()}
