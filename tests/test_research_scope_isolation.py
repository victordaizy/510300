from __future__ import annotations

import copy
from pathlib import Path

from scripts.validate_research_registry import (
    DEFAULT_REGISTRY,
    load_registry,
    validate_records,
)


ROOT = Path(__file__).resolve().parents[1]


def test_registered_dependencies_do_not_cross_research_scope() -> None:
    records = load_registry(DEFAULT_REGISTRY)
    result = validate_records(
        records,
        root=ROOT,
        verify_hashes=False,
        audit_coverage=False,
    )
    assert not [error for error in result["errors"] if "跨研究范围" in error]
    assert {record["research_scope"] for record in records} == {
        "ETF_510300",
        "INTERNATIONAL_BROKER",
    }


def test_international_broker_dependency_cannot_enter_510300_graph() -> None:
    records = load_registry(DEFAULT_REGISTRY)
    contaminated = copy.deepcopy(records)
    etf_record = next(
        record
        for record in contaminated
        if record["model_id"] == "510300_PREDICTABLE_FISH_MIDDLE_DECISION_V1"
    )
    etf_record["dependencies"].append(
        {
            "model_id": "INTERNATIONAL_BROKER_THEME_RESEARCH_V1",
            "research_scope": "INTERNATIONAL_BROKER",
            "relation": "ILLEGAL_FEATURE_INPUT",
        }
    )
    result = validate_records(
        contaminated,
        root=ROOT,
        verify_hashes=False,
        audit_coverage=False,
    )
    assert any("跨研究范围依赖" in error for error in result["errors"])


def test_international_broker_feature_namespace_cannot_enter_510300() -> None:
    records = load_registry(DEFAULT_REGISTRY)
    contaminated = copy.deepcopy(records)
    etf_record = next(
        record
        for record in contaminated
        if record["model_id"] == "510300_PREDICTABLE_FISH_MIDDLE_DECISION_V1"
    )
    etf_record["feature_dependencies"].append(
        {
            "namespace": "INTERNATIONAL_BROKER::G3_PROFIT_BRIDGE",
            "research_scope": "INTERNATIONAL_BROKER",
        }
    )
    result = validate_records(
        contaminated,
        root=ROOT,
        verify_hashes=False,
        audit_coverage=False,
    )
    assert any("特征跨研究范围" in error for error in result["errors"])
