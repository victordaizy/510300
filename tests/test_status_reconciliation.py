from __future__ import annotations

import json
from pathlib import Path

from scripts.render_research_status import DEFAULT_OUTPUT, render_status
from scripts.validate_research_registry import DEFAULT_REGISTRY, load_registry


ROOT = Path(__file__).resolve().parents[1]


def _models_by_id(payload: dict) -> dict[str, dict]:
    return {model["model_id"]: model for model in payload["models"]}


def test_rendered_status_is_exact_registry_derivation() -> None:
    records = load_registry(DEFAULT_REGISTRY)
    expected = render_status(
        records,
        registry_path=DEFAULT_REGISTRY,
        root=ROOT,
    )
    actual = json.loads(DEFAULT_OUTPUT.read_text(encoding="utf-8"))
    assert actual == expected


def test_rendered_status_contains_no_unicode_replacement_character() -> None:
    assert "\ufffd" not in DEFAULT_OUTPUT.read_text(encoding="utf-8")


def test_operational_success_does_not_upgrade_research_or_view() -> None:
    payload = render_status(
        load_registry(DEFAULT_REGISTRY),
        registry_path=DEFAULT_REGISTRY,
        as_of_date="2026-08-18",
        root=ROOT,
    )
    models = _models_by_id(payload)
    primary = models["510300_PRIMARY_SECONDARY_FORWARD_V1"]
    assert primary["run_status"] == "SUCCESS"
    assert primary["collection_status"] == "COLLECTING"
    assert primary["research_status"] == "DISCOVERY_ONLY"
    assert primary["view_status"] == "NO_VIEW"
    assert primary["authorization"] == "REAL_DISABLED"


def test_failed_run_does_not_become_research_rejection_or_view() -> None:
    payload = render_status(
        load_registry(DEFAULT_REGISTRY),
        registry_path=DEFAULT_REGISTRY,
        as_of_date="2026-08-18",
        root=ROOT,
    )
    v3 = _models_by_id(payload)["V3_FORWARD_2"]
    assert v3["run_status"] == "FAILED"
    assert v3["collection_status"] == "BLOCKED"
    assert v3["research_status"] == "FORWARD_COLLECTING"
    assert v3["view_status"] == "NO_VIEW"
    assert v3["authorization"] == "PAPER_ONLY"
    assert payload["safety"]["live_trading_authorized"] is False
