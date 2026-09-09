from __future__ import annotations

import numpy as np
import pandas as pd

from research.csi300_all_etf_momentum_alpha_v1 import FACTOR_COLUMNS, build_all_etf_targets


def contract() -> dict:
    linear = {name: 0.96 / len(FACTOR_COLUMNS) for name in FACTOR_COLUMNS}
    return {
        "periods": {"warmup_start": "2019-01-01", "evaluation_start": "2020-01-01", "evaluation_end": "2022-12-31", "rebalance_every_trading_days": 20},
        "universe": {"defensive_asset": "D", "minimum_history_days": 252, "minimum_20d_average_amount_cny": 1.0},
        "formula": {"linear_weights": linear, "nonlinear_interactions": {"momentum_efficiency_rank_product": 0.02, "medium_short_momentum_rank_product": 0.02}, "retention_rank": 3},
    }


def test_nine_factors_and_no_future_reaction() -> None:
    assert len(FACTOR_COLUMNS) == 9
    dates = pd.bdate_range("2019-01-01", periods=900)
    benchmark = pd.DataFrame({"date": dates, "close": np.arange(len(dates)) + 100})
    master = pd.DataFrame({"ts_code": ["A", "B", "D"], "list_date": ["2018-01-01"] * 3, "delist_date": [pd.NaT] * 3})
    rows = []
    for symbol, slope in (("A", 0.001), ("B", 0.0005), ("D", 0.0001)):
        values = 100 * np.exp(np.arange(len(dates)) * slope)
        rows.extend({"date": date, "con_code": symbol, "total_return_close": value, "amount": 1e9} for date, value in zip(dates, values, strict=True))
    panel = pd.DataFrame(rows)
    before_features, before_targets, _ = build_all_etf_targets(panel, master, benchmark, contract())
    cutoff = pd.Timestamp("2021-09-01")
    changed = panel.copy()
    changed.loc[changed["date"].gt(cutoff) & changed["con_code"].eq("B"), "total_return_close"] *= 50
    after_features, after_targets, _ = build_all_etf_targets(changed, master, benchmark, contract())
    pd.testing.assert_frame_equal(before_features.loc[before_features["date"].le(cutoff)].reset_index(drop=True), after_features.loc[after_features["date"].le(cutoff)].reset_index(drop=True))
    pd.testing.assert_frame_equal(before_targets.loc[before_targets["signal_date"].le(cutoff)].reset_index(drop=True), after_targets.loc[after_targets["signal_date"].le(cutoff)].reset_index(drop=True))
