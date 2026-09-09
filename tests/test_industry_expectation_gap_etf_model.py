from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from research.industry_expectation_gap_etf_model import (
    ModelRules,
    ResearchInputError,
    build_industry_map,
    build_research_result,
    evaluate_etf_snapshot,
    evaluate_market_state,
    select_sector_snapshot,
    validate_contract,
    validate_source_registry,
)


def _rules() -> ModelRules:
    return ModelRules(
        required_latest_weight_coverage=0.98,
        maximum_sector_weight_age_calendar_days=31,
        minimum_expectation_gap_weight_coverage=0.65,
        minimum_abs_weighted_gap_for_direction=0.08,
        minimum_sources_per_industry=2,
        primary_horizon_trading_days=60,
        tail_horizon_trading_days=120,
        minimum_calibration_observations=20,
        minimum_model_comparison_observations=40,
        allow_historical_return_read=False,
        allow_model_fitting=False,
        allow_failed_family_reuse=False,
        allow_proxy_identity_inference=False,
        require_market_gate_for_directional_view=True,
        require_all_latest_industries=True,
    )


def _panel() -> pd.DataFrame:
    shared = {
        "date": pd.Timestamp("2026-07-31"),
        "component_count": 10,
        "core_feature_coverage_ratio": 1.0,
        "weighted_earnings_yield": 0.05,
        "weighted_book_yield": 0.5,
        "weighted_ttm_roe": 0.1,
        "weighted_ttm_profit_growth_yoy": 0.2,
        "weighted_ttm_revenue_growth_yoy": 0.1,
    }
    return pd.DataFrame(
        [
            {**shared, "industry_l1": "行业甲", "sector_weight": 0.6},
            {**shared, "industry_l1": "行业乙", "sector_weight": 0.4},
        ]
    )


def _sources() -> dict[str, dict[str, str]]:
    return {
        "S1": {"available_at": "2026-08-01T00:00:00+08:00"},
        "S2": {"available_at": "2026-08-02T00:00:00+08:00"},
    }


def _ledger() -> dict:
    shared = {
        "fundamental_60d": "IMPROVEMENT",
        "fundamental_120d": "IMPROVEMENT",
        "confidence": "MEDIUM",
        "valuation_method": "行业内估值锚",
        "thesis": "经营改善但只作待验证判断。",
        "invalidation": "后续经营数据转弱。",
        "source_ids": ["S1", "S2"],
    }
    return {
        "as_of_date": "2026-08-18",
        "information_cutoff": "2026-08-18T16:30:00+08:00",
        "weight_snapshot_date": "2026-07-31",
        "historical_return_read": False,
        "model_fitting_performed": False,
        "judgments": [
            {**shared, "industry_l1": "行业甲", "expectation_gap": "POSITIVE"},
            {**shared, "industry_l1": "行业乙", "expectation_gap": "NEGATIVE"},
        ],
    }


def _market() -> dict:
    dimension = {
        "state": "NO_VIEW",
        "data_as_of": "2026-08-12",
        "failure_category": "STALE",
        "reason": "输入已过期。",
        "source_ids": ["S1"],
    }
    return {
        "as_of_date": "2026-08-18",
        "information_cutoff": "2026-08-18T16:30:00+08:00",
        "funding_liquidity": dict(dimension),
        "equity_liquidity": dict(dimension),
        "risk_bearing": dict(dimension),
        "national_team": {
            "disclosed_holdings_state": "UNOBSERVED",
            "holdings_report_period": None,
            "holdings_publication_date": None,
            "activity_proxy_state": "UNOBSERVED",
            "failure_category": "NO_DISCLOSURE",
            "reason": "没有当前披露。",
            "source_ids": [],
        },
    }


def _etf() -> dict:
    return {
        "asset": "510300",
        "as_of_date": "2026-08-18",
        "information_cutoff": "2026-08-18T16:30:00+08:00",
        "source_ids": ["S1", "S2"],
        "collector_status": "PASS",
        "collector_role": "OBSERVATION_EVIDENCE_ONLY",
        "pcf": {
            "trading_date": "2026-08-18",
            "creation_allowed": True,
            "redemption_allowed": True,
        },
        "iopv": {"timestamp": "2026-08-18T14:59:00+08:00"},
        "forward_readiness": {
            "full_coverage_days": 2,
            "eligible_for_research_evaluation": False,
        },
        "small_account_primary_market_feasible": False,
        "use_as_directional_signal": False,
    }


def test_select_sector_snapshot_uses_latest_non_future_date() -> None:
    panel = pd.concat(
        [
            _panel().assign(date=pd.Timestamp("2026-06-30")),
            _panel(),
            _panel().assign(date=pd.Timestamp("2026-08-31")),
        ],
        ignore_index=True,
    )
    snapshot, metadata = select_sector_snapshot(panel, "2026-08-18", _rules())
    assert metadata["snapshot_date"] == "2026-07-31"
    assert metadata["weight_coverage"] == pytest.approx(1.0)
    assert snapshot["date"].max() == pd.Timestamp("2026-07-31")


def test_industry_map_requires_exact_latest_industry_set() -> None:
    snapshot, metadata = select_sector_snapshot(_panel(), "2026-08-18", _rules())
    ledger = _ledger()
    ledger["judgments"] = ledger["judgments"][:1]
    with pytest.raises(ResearchInputError, match="精确覆盖"):
        build_industry_map(
            ledger,
            snapshot,
            metadata,
            _sources(),
            _rules(),
            "2026-08-18",
            "2026-08-18T16:30:00+08:00",
        )


def test_industry_ledger_rejects_outcome_fields() -> None:
    snapshot, metadata = select_sector_snapshot(_panel(), "2026-08-18", _rules())
    ledger = _ledger()
    ledger["judgments"][0]["future_return_60d"] = 0.12
    with pytest.raises(ResearchInputError, match="结果字段"):
        build_industry_map(
            ledger,
            snapshot,
            metadata,
            _sources(),
            _rules(),
            "2026-08-18",
            "2026-08-18T16:30:00+08:00",
        )


def test_source_registry_rejects_future_available_source(tmp_path: Path) -> None:
    registry = {
        "sources": {
            "FUTURE": {
                "available_at": "2026-08-19T00:00:00+08:00",
                "url": "https://example.com/future",
            }
        }
    }
    with pytest.raises(ResearchInputError, match="晚于信息截止时间"):
        validate_source_registry(registry, tmp_path, "2026-08-18T16:30:00+08:00")


def test_national_team_present_requires_disclosure_dates() -> None:
    market = _market()
    market["national_team"]["disclosed_holdings_state"] = "PRESENT"
    market["national_team"]["source_ids"] = ["S1"]
    with pytest.raises(ResearchInputError, match="报告期和公开日"):
        evaluate_market_state(
            market,
            _sources(),
            "2026-08-18",
            "2026-08-18T16:30:00+08:00",
        )


def test_etf_observation_and_predictive_readiness_are_separate() -> None:
    result = evaluate_etf_snapshot(
        _etf(),
        _sources(),
        "2026-08-18",
        "2026-08-18T16:30:00+08:00",
        _rules(),
    )
    assert result["observation_state"] == "PASS"
    assert result["predictive_state"] == "INSUFFICIENT_FORWARD_HISTORY"
    assert result["execution_mode"] == "OBSERVE_ONLY"


def test_balanced_industry_map_still_produces_no_view() -> None:
    snapshot, metadata = select_sector_snapshot(_panel(), "2026-08-18", _rules())
    balanced_ledger = _ledger()
    for judgment in balanced_ledger["judgments"]:
        judgment["expectation_gap"] = "BALANCED"
    frame, industry = build_industry_map(
        balanced_ledger,
        snapshot,
        metadata,
        _sources(),
        _rules(),
        "2026-08-18",
        "2026-08-18T16:30:00+08:00",
    )
    market = evaluate_market_state(
        _market(),
        _sources(),
        "2026-08-18",
        "2026-08-18T16:30:00+08:00",
    )
    etf = evaluate_etf_snapshot(
        _etf(),
        _sources(),
        "2026-08-18",
        "2026-08-18T16:30:00+08:00",
        _rules(),
    )
    contract = {
        "version": "V1",
        "status": "FROZEN_BEFORE_FORWARD_OUTCOME",
        "research_mode": "RESEARCH_ONLY",
        "signal_mode": "SHADOW_ONLY",
        "asset": "510300",
        "benchmark": "000300",
        "as_of_date": "2026-08-18",
        "information_cutoff": "2026-08-18T16:30:00+08:00",
        "first_forward_observation_date": "2026-08-19",
    }
    result = build_research_result(
        contract,
        frame,
        industry,
        market,
        etf,
        {
            "position_mapping_enabled": False,
            "order_generation_enabled": False,
            "broker_connection_enabled": False,
            "live_trading_enabled": False,
            "current_holdings_read_enabled": False,
        },
    )
    assert industry["state"] == "BALANCED_NO_EDGE"
    assert result["final_state"] == "NO_VIEW"
    assert "ETF_FORWARD_HISTORY_INSUFFICIENT" in result["failure_categories"]


def test_contract_rejects_any_enabled_execution_switch() -> None:
    contract = {
        "status": "FROZEN_BEFORE_FORWARD_OUTCOME",
        "asset": "510300",
        "benchmark": "000300",
        "rules": _rules().__dict__,
        "safety": {
            "position_mapping_enabled": True,
            "order_generation_enabled": False,
            "broker_connection_enabled": False,
            "live_trading_enabled": False,
            "current_holdings_read_enabled": False,
        },
    }
    with pytest.raises(ResearchInputError, match="安全开关"):
        validate_contract(contract)
