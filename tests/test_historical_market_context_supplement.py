from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.historical_market_context_supplement import (
    HistoricalContextError,
    HistoricalContextRules,
    build_current_market_context,
    build_historical_context_report,
    build_industry_price_context,
    prepare_market_features,
    select_historical_analogues,
    validate_context_config,
)


def _rules() -> HistoricalContextRules:
    return HistoricalContextRules(
        annualization_trading_days=242,
        trend_return_windows=(20, 60, 120),
        moving_average_windows=(20, 60),
        volatility_windows=(20, 60),
        drawdown_window=60,
        valuation_percentile_window=120,
        participation_percentile_window=60,
        analogue_count=8,
        analogue_minimum_separation_trading_days=20,
        analogue_current_embargo_trading_days=120,
        analogue_outcome_horizons=(20, 60),
        minimum_analogue_count=5,
        industry_return_windows=(5, 20, 60),
        industry_percentile_windows=(20, 60),
        high_percentile=0.80,
        extreme_percentile=0.95,
        low_percentile=0.20,
        allowed_market_staleness_trading_days=0,
        allowed_industry_staleness_trading_days=0,
    )


def _config() -> dict:
    return {
        "status": "FROZEN_BEFORE_ANALOGUE_OUTCOME_READ",
        "asset": "510300",
        "benchmark": "H00300",
        "rules": {
            **_rules().__dict__,
            "trend_return_windows": [20, 60, 120],
            "moving_average_windows": [20, 60],
            "volatility_windows": [20, 60],
            "analogue_outcome_horizons": [20, 60],
            "industry_return_windows": [5, 20, 60],
            "industry_percentile_windows": [20, 60],
        },
        "prohibitions": {
            "may_fill_funding_liquidity": False,
            "may_fill_equity_flow": False,
            "may_infer_national_team_identity": False,
            "may_change_industry_expectation_gap": False,
            "may_upgrade_original_no_view": False,
            "may_call_analogue_outcomes_oos": False,
        },
        "safety": {
            "position_mapping_enabled": False,
            "order_generation_enabled": False,
            "broker_connection_enabled": False,
            "live_trading_enabled": False,
        },
    }


def _market(periods: int = 1500) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-02", periods=periods)
    steps = 0.0002 + 0.006 * np.sin(np.arange(periods) / 17.0)
    close = 1000.0 * np.exp(np.cumsum(steps))
    return pd.DataFrame(
        {
            "date": dates,
            "close": close,
            "amount": 100.0 + 20.0 * np.cos(np.arange(periods) / 23.0),
            "pe_ttm": 12.0 + 1.5 * np.sin(np.arange(periods) / 31.0),
        }
    )


def _blocks() -> dict[str, list[str]]:
    return {
        "trend": [
            "return_20d",
            "return_60d",
            "return_120d",
            "close_vs_ma20",
            "close_vs_ma60",
        ],
        "risk": ["rv_20d", "rv_60d", "drawdown_60d"],
        "valuation": ["pe_ttm_percentile_120d"],
        "participation": ["amount_percentile_60d"],
    }


def test_config_rejects_permission_to_upgrade_original_view() -> None:
    config = _config()
    config["prohibitions"]["may_upgrade_original_no_view"] = True
    with pytest.raises(HistoricalContextError, match="必须为false"):
        validate_context_config(config)


def test_market_features_stop_exactly_at_context_cutoff() -> None:
    market = _market()
    cutoff = market.iloc[-2]["date"]
    features = prepare_market_features(market, cutoff, _rules())
    assert features["date"].max() == cutoff
    assert len(features) == len(market) - 1
    assert pd.notna(features.iloc[-1]["return_120d"])
    assert 0 <= features.iloc[-1]["pe_ttm_percentile_120d"] <= 1


def test_historical_analogues_are_spaced_and_outcomes_mature() -> None:
    market = _market()
    features = prepare_market_features(market, market.iloc[-1]["date"], _rules())
    records, summary = select_historical_analogues(features, _blocks(), _rules())
    assert len(records) == 8
    positions = {
        date.date().isoformat(): index for index, date in enumerate(pd.to_datetime(features["date"]))
    }
    selected_positions = [positions[row["analogue_date"]] for row in records]
    for left_index, left in enumerate(selected_positions):
        for right in selected_positions[left_index + 1 :]:
            assert abs(left - right) >= 20
        assert left + 60 < len(features)
    assert summary["interpretation"] == "DESCRIPTIVE_CONDITIONAL_SAMPLE_NOT_OOS"
    assert 0 <= summary["horizon_60d"]["positive_share"] <= 1


def test_duplicate_feature_across_blocks_is_rejected() -> None:
    market = _market()
    features = prepare_market_features(market, market.iloc[-1]["date"], _rules())
    blocks = _blocks()
    blocks["risk"].append("return_20d")
    with pytest.raises(HistoricalContextError, match="重复"):
        select_historical_analogues(features, blocks, _rules())


def test_current_market_context_does_not_claim_liquidity_or_identity() -> None:
    market = _market()
    features = prepare_market_features(market, market.iloc[-1]["date"], _rules())
    result = build_current_market_context(features, _rules(), _blocks())
    assert result["can_fill_market_liquidity"] is False
    assert result["can_infer_investor_identity"] is False
    assert result["trend_state"] in {
        "CONFIRMED_UPTREND",
        "SHORT_RECOVERY_NOT_CONFIRMED",
        "CONFIRMED_DOWNTREND",
        "MIXED_TREND",
    }


def test_industry_context_retains_stale_trading_day_gate() -> None:
    dates = pd.bdate_range("2026-01-02", periods=150)
    rows = []
    for date in dates[:-2]:
        rows.extend(
            [
                {"date": date, "industry_l1": "行业甲", "industry_return_1d": 0.001},
                {"date": date, "industry_l1": "行业乙", "industry_return_1d": -0.0005},
            ]
        )
    forecast = [
        {"industry_l1": "行业甲", "sector_weight": 0.6, "expectation_gap": "POSITIVE"},
        {"industry_l1": "行业乙", "sector_weight": 0.4, "expectation_gap": "NEGATIVE"},
    ]
    result = build_industry_price_context(
        pd.DataFrame(rows), forecast, dates[-1], dates, _rules()
    )
    assert result["stale_trading_days"] == 2
    assert result["state"] == "STALE_HISTORICAL_CONTEXT"
    assert result["eligible_to_fill_current_industry_state"] is False
    assert len(result["industry_rows"]) == 2


def test_report_preserves_original_no_view() -> None:
    report = build_historical_context_report(
        {
            "version": "V1",
            "as_of_date": "2026-08-18",
            "information_cutoff": "2026-08-18T20:30:00+08:00",
        },
        _rules(),
        _config()["prohibitions"],
        _config()["safety"],
        {"price_risk_state": "PRICE_RISK_MIXED"},
        [],
        {"selected_count": 0},
        {
            "eligible_to_fill_current_industry_state": False,
            "state": "STALE_HISTORICAL_CONTEXT",
        },
        {
            "final_state": "NO_VIEW",
            "industry_aggregation": {"state": "BALANCED_NO_EDGE"},
        },
        {},
    )
    assert report["original_prediction_state"] == "NO_VIEW"
    assert report["original_prediction_changed"] is False
    assert report["status"] == "PARTIAL_STALE_INDUSTRY_CONTEXT"
