from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.systemic_sell_pressure_cascade_v1 import (  # noqa: E402
    _strict_prior_rolling_percentile,
    build_daily_features_from_frame,
    build_policy_states,
    calibrate_and_apply_signal,
    load_config,
    validate_config,
)


def _master(codes: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_code": codes,
            "list_status": ["L"] * len(codes),
            "list_date": pd.to_datetime(["2020-01-01"] * len(codes)),
            "delist_date": pd.to_datetime([None] * len(codes)),
            "split_group": ["TRAIN"] * len(codes),
        }
    )


def _panel() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=5)
    rows: list[dict[str, object]] = []
    returns_by_day = [
        [-10.0, -5.0, -2.0, 1.0],
        [-1.0, 0.0, 1.0, 2.0],
        [-6.0, -3.0, -1.0, 1.0],
        [-2.5, -2.1, -1.0, 0.5],
        [-9.8, -5.5, -2.2, -0.5],
    ]
    for date, returns in zip(dates, returns_by_day, strict=True):
        for index, value in enumerate(returns):
            close = 10.0 * (1.0 + value / 100.0)
            low = close if value < 0.0 else close - 0.1
            high = max(10.0, close + 0.1)
            rows.append(
                {
                    "con_code": f"00000{index + 1}.SZ",
                    "date": date,
                    "raw_open": 10.0,
                    "raw_high": high,
                    "raw_low": low,
                    "raw_close": close,
                    "pct_chg": value,
                    "amount": float((index + 1) * 100.0),
                    "is_suspended": False,
                }
            )
    return pd.DataFrame(rows)


def test_config_freezes_binary_scope_and_twenty_percent_goal() -> None:
    config = load_config()
    validate_config(config)
    assert config["scope"]["allowed_target_states"] == [0, 1]
    assert config["scope"]["cash_exit_holding_days"] == 1
    assert config["objective"]["minimum_annualized_excess"] == 0.20
    assert config["signal"]["score_cutoff_quantile"] == 0.95


def test_strict_prior_percentile_excludes_current_and_future_values() -> None:
    values = pd.Series([1.0, 3.0, 2.0, 4.0, 0.0])
    result = _strict_prior_rolling_percentile(values, 3)
    assert result.iloc[:3].isna().all()
    assert result.iloc[3] == 1.0
    assert result.iloc[4] == 0.0
    changed_future = _strict_prior_rolling_percentile(pd.Series([1.0, 3.0, 2.0, 4.0, 999.0]), 3)
    assert changed_future.iloc[3] == result.iloc[3]


def test_daily_feature_aggregation_has_expected_sell_pressure_counts() -> None:
    config = deepcopy(load_config())
    config["universe"]["minimum_observed_history_days"] = 1
    config["signal"]["raw_feature_prior_window_trading_days"] = 2
    config["signal"]["raw_feature_prior_minimum_observations"] = 2
    panel = _panel()
    codes = sorted(panel["con_code"].unique())
    features, audit = build_daily_features_from_frame(
        panel,
        _master(codes),
        required_split_group="TRAIN",
        feature_start="2024-01-02",
        feature_end="2024-01-08",
        config=config,
    )
    first = features.iloc[0]
    assert first["eligible_stock_count"] == 4
    assert first["down_2_count"] == 3
    assert first["down_5_count"] == 2
    assert first["lower_limit_close_count"] == 1
    assert first["close_low_down_count"] == 3
    assert np.isclose(first["amount_weighted_down_share"], 0.6)
    assert np.isclose(first["amount_weighted_down_5_share"], 0.3)
    assert audit["status"] == "PASS"
    assert features["risk_feature_percentiles_available"].tolist() == [False, False, True, True, True]


def test_calibration_cutoff_uses_only_calibration_scores() -> None:
    config = load_config()
    dates = pd.bdate_range("2020-01-02", periods=300)
    scores = np.linspace(0.0, 1.0, 300)
    features = pd.DataFrame(
        {
            "date": dates,
            "risk_feature_percentiles_available": True,
            "systemic_sell_pressure_risk_score": scores,
            "cross_section_median_return": -0.01,
            "amount_weighted_down_share": 0.75,
        }
    )
    scored, calibration = calibrate_and_apply_signal(
        features,
        calibration_start=str(dates[0].date()),
        calibration_end=str(dates[249].date()),
        signal_start=str(dates[250].date()),
        signal_end=str(dates[-1].date()),
        config=config,
    )
    expected = np.quantile(scores[:250], 0.95, method="linear")
    assert np.isclose(calibration["score_cutoff"], expected)
    assert scored.loc[scored["date"] < dates[250], "cash_next_trading_day"].sum() == 0
    assert calibration["cash_signal_days"] > 0


def test_policy_state_uses_strictly_previous_trading_day_signal() -> None:
    dates = pd.bdate_range("2026-01-05", periods=5)
    market = pd.DataFrame({"date": dates})
    signals = pd.DataFrame(
        {
            "date": dates[:-1],
            "cash_next_trading_day": [False, True, False, True],
        }
    )
    states = build_policy_states(market, signals)
    assert states.tolist() == [0, 1, 0, 1, 0]
