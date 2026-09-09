"""十因子点时构造、目标和走步模型防前视测试。"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from research.csi300_ten_factor_alpha_v1 import (
    FACTOR_COLUMNS,
    TenFactorRules,
    build_forward_relative_outcomes,
    build_ten_factor_panel,
    walk_forward_predictions,
)


def _rules(**overrides: object) -> TenFactorRules:
    values: dict[str, object] = {
        "feature_start": pd.Timestamp("2025-05-01"),
        "pseudo_oos_start": pd.Timestamp("2025-08-01"),
        "end_date": pd.Timestamp("2025-12-31"),
        "rebalance_step": 10,
        "target_horizon": 2,
        "feature_columns": FACTOR_COLUMNS,
        "learning_rate": 0.05,
        "max_iter": 20,
        "max_leaf_nodes": 5,
        "min_samples_leaf": 2,
        "l2_regularization": 1.0,
        "max_bins": 16,
        "random_state": 20260818,
        "target_clip": (-0.5, 0.5),
        "half_life_days": 730.0,
        "minimum_training_rows": 4,
        "minimum_training_signal_dates": 2,
        "minimum_average_amount": 1.0,
    }
    values.update(overrides)
    return TenFactorRules(**values)


def _factor_inputs() -> tuple[pd.DataFrame, ...]:
    dates = pd.bdate_range("2025-01-02", periods=180)
    symbols = ["A", "B"]
    history = pd.DataFrame(
        [
            {
                "date": date,
                "con_code": symbol,
                "raw_open": 10.0 + symbol_index + date_index * 0.01,
                "raw_close": 10.05 + symbol_index + date_index * 0.01,
                "total_return_open": 10.0 + symbol_index + date_index * 0.01,
                "total_return_close": 10.05 + symbol_index + date_index * 0.01,
                "volume": 100000.0,
                "amount": 10000000.0 + symbol_index * 1000.0,
            }
            for date_index, date in enumerate(dates)
            for symbol_index, symbol in enumerate(symbols)
        ]
    )
    members = history[["date", "con_code", "raw_open", "raw_close"]].copy()
    members["is_index_member"] = True
    members["is_suspended"] = False
    flow = pd.DataFrame(
        [
            {
                "date": date,
                "ts_code": symbol,
                "large_extra_large_net_amount_10k_cny": 100.0 + symbol_index,
                "small_net_amount_10k_cny": -20.0,
            }
            for date in dates
            for symbol_index, symbol in enumerate(symbols)
        ]
    )
    financials = pd.DataFrame(
        {
            "con_code": symbols,
            "announcement_date": [dates[0], dates[0]],
            "report_period": [pd.Timestamp("2024-12-31")] * 2,
            "bps": [5.0, 6.0],
            "roe": [10.0, 12.0],
        }
    )
    index = pd.DataFrame({"date": dates, "close": np.linspace(1000.0, 1200.0, len(dates))})
    breadth = pd.DataFrame({"date": dates, "equal_above_ma60_share": 0.6})
    return history, members, flow, financials, index, breadth


def test_factor_count_is_ten_and_future_price_cannot_rewrite_past() -> None:
    inputs = _factor_inputs()
    rules = _rules(feature_start=pd.Timestamp("2025-07-01"))
    first = build_ten_factor_panel(*inputs, rules)
    changed_history = inputs[0].copy()
    changed_history.loc[changed_history["date"].eq(changed_history["date"].max()), "total_return_close"] *= 0.5
    second = build_ten_factor_panel(changed_history, *inputs[1:], rules)
    cutoff = first["date"].max() - pd.Timedelta(days=10)
    columns = ["date", "con_code", *FACTOR_COLUMNS]
    left = first.loc[first["date"].le(cutoff), columns].reset_index(drop=True)
    right = second.loc[second["date"].le(cutoff), columns].reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right)
    assert len(FACTOR_COLUMNS) == 10
    assert first["announcement_date"].le(first["date"]).all()


def test_outcome_starts_next_open_and_exits_after_fixed_horizon() -> None:
    history, members, *_ = _factor_inputs()
    dates = pd.DatetimeIndex(sorted(members["date"].unique()))
    signal_date = dates[20]
    factors = pd.DataFrame({"date": [signal_date], "con_code": ["A"]})
    benchmark = pd.DataFrame({"date": dates, "close": 100.0})
    result = build_forward_relative_outcomes(factors, history, benchmark, dates, _rules())
    row = result.iloc[0]
    assert row["entry_date"] == dates[21]
    assert row["maturity_date"] == dates[23]
    expected = np.log(row["exit_total_return_open"] / row["entry_total_return_open"])
    assert np.isclose(row["future_excess_log_return_10d"], expected)


def test_walk_forward_prediction_does_not_use_unmatured_labels() -> None:
    dates = pd.bdate_range("2025-01-02", periods=8)
    rows = []
    outcomes = []
    for date_index, date in enumerate(dates):
        for symbol_index, symbol in enumerate(("A", "B", "C", "D")):
            row = {
                "date": date,
                "con_code": symbol,
                "signal_output": "SIGNAL_READY",
                "average_amount_20d": 1e8,
                "is_suspended": False,
            }
            for factor_index, column in enumerate(FACTOR_COLUMNS):
                row[column] = (symbol_index + 1 + factor_index * 0.01) / 4.0
            rows.append(row)
            outcomes.append(
                {
                    "date": date,
                    "con_code": symbol,
                    "maturity_date": date + pd.Timedelta(days=2),
                    "future_excess_log_return_10d": 0.01 * (symbol_index - 1),
                }
            )
    factors = pd.DataFrame(rows)
    outcome_frame = pd.DataFrame(outcomes)
    rules = replace(_rules(), pseudo_oos_start=dates[4])
    first = walk_forward_predictions(factors, outcome_frame, rules)
    changed = outcome_frame.copy()
    changed.loc[changed["date"].ge(dates[4]), "future_excess_log_return_10d"] = 0.49
    second = walk_forward_predictions(factors, changed, rules)
    columns = ["con_code", "predicted_excess_log_return_10d"]
    left = first.loc[first["date"].eq(dates[4]), columns].sort_values("con_code").reset_index(drop=True)
    right = second.loc[second["date"].eq(dates[4]), columns].sort_values("con_code").reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right)

