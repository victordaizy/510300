from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from research.val01_raw_ey_5y_v1 import (
    FORBIDDEN_OUTPUT_TOKENS,
    apply_frozen_percentile,
    build_full_snapshot_prices,
    empirical_midrank_percentile,
    load_config,
)


ROOT = Path(__file__).resolve().parents[1]


def test_empirical_midrank_uses_frozen_tie_rule() -> None:
    assert empirical_midrank_percentile(pd.Series([1.0, 2.0, 2.0])) == 2.0 / 3.0
    assert empirical_midrank_percentile(pd.Series([2.0, 3.0, 1.0])) == 0.5 / 3.0
    assert empirical_midrank_percentile(pd.Series([1.0, 2.0, 3.0])) == 2.5 / 3.0


def test_percentile_requires_complete_window_and_includes_current() -> None:
    frame = pd.DataFrame(
        {
            "weighted_raw_earnings_yield": np.arange(1.0, 62.0),
            "input_status": ["PASS"] * 61,
        }
    )
    result = apply_frozen_percentile(frame, 60)
    assert result.loc[:58, "raw_ey_percentile_60m"].isna().all()
    assert result.loc[59, "raw_ey_percentile_60m"] == 59.5 / 60.0
    assert result.loc[60, "raw_ey_percentile_60m"] == 59.5 / 60.0
    assert result["percentile_ready"].sum() == 2


def test_failed_month_breaks_following_complete_window() -> None:
    frame = pd.DataFrame(
        {
            "weighted_raw_earnings_yield": np.arange(1.0, 62.0),
            "input_status": ["PASS"] * 30 + ["BLOCKED"] + ["PASS"] * 30,
        }
    )
    result = apply_frozen_percentile(frame, 60)
    assert not result["percentile_ready"].any()


def test_config_preserves_governance_and_existing_trial_count() -> None:
    config = load_config()
    assert config["protocol"]["upstream_registered_trial_id"] == "VAL01_RAW_EY_5Y"
    assert config["trial_registration"]["existing_registered_valuation_trial_count"] == 8
    assert config["trial_registration"]["consumes_one_existing_trial"] is True
    for key, value in config["protocol"].items():
        if key.endswith("_enabled"):
            assert value is False


def test_full_snapshot_price_panel_has_no_future_prices() -> None:
    config = load_config()
    weights = pd.read_parquet(
        ROOT / config["data_contracts"]["historical_weights"]["file"]
    )
    early = pd.read_parquet(ROOT / config["data_contracts"]["snapshot_prices"]["file"])
    current = pd.read_parquet(
        ROOT / config["data_contracts"]["current_constituent_daily"]["file"]
    )
    panel = build_full_snapshot_prices(weights, early, current, config)
    assert len(panel) == 36_000
    assert panel.groupby("date")["con_code"].nunique().eq(300).all()
    assert not (panel["price_trade_date"] > panel["date"]).any()
    panel["valid_price_weight"] = panel["official_weight_pct"].where(
        panel["raw_close"].notna() & panel["raw_close"].gt(0), 0.0
    ) / 100.0
    coverage = panel.groupby("date")["valid_price_weight"].sum()
    assert coverage.ge(config["signal_definition"]["minimum_price_weight_coverage"]).all()


def test_generated_output_contains_no_return_position_or_order_columns() -> None:
    config = load_config()
    output = ROOT / config["artifacts"]["signal_inputs"]
    if not output.exists():
        return
    data = pd.read_parquet(output)
    lowered = [str(column).lower() for column in data.columns]
    assert not [
        column for column in lowered if any(token in column for token in FORBIDDEN_OUTPUT_TOKENS)
    ]
    ready = data.loc[data["percentile_ready"]]
    assert data["valid_price_constituent_count"].min() >= 299
    assert data["valid_ttm_constituent_count"].min() >= 296
    assert ready.iloc[0]["date"] == pd.Timestamp("2021-07-30")
    assert ready.iloc[0]["earliest_execution_date"] == pd.Timestamp("2021-08-02")
