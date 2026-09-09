from __future__ import annotations

import numpy as np
import pandas as pd

from research.csi300_etf_rotation_alpha_v1 import FACTOR_COLUMNS, build_rotation_targets


def _contract() -> dict:
    return {
        "periods": {"evaluation_start": "2021-01-01", "evaluation_end": "2022-12-31", "rebalance_every_trading_days": 20},
        "universe": {"risk_assets": {"A": "A", "B": "B"}, "defensive_asset": {"D": "D"}},
        "formula": {"weights": {name: 1 / 8 for name in FACTOR_COLUMNS}, "retention_rank": 2},
    }


def test_factor_budget_and_no_future_reaction() -> None:
    assert len(FACTOR_COLUMNS) == 8
    dates = pd.bdate_range("2019-01-01", periods=800)
    rows = []
    for symbol, slope in (("A", 0.001), ("B", 0.0005), ("D", 0.0001)):
        close = 100 * np.exp(np.arange(len(dates)) * slope)
        rows.extend({"date": date, "con_code": symbol, "total_return_close": value} for date, value in zip(dates, close, strict=True))
    panel = pd.DataFrame(rows)
    features_before, targets_before, _ = build_rotation_targets(panel, _contract())
    cutoff = pd.Timestamp("2021-09-01")
    changed = panel.copy()
    changed.loc[changed["date"].gt(cutoff) & changed["con_code"].eq("B"), "total_return_close"] *= 50
    features_after, targets_after, _ = build_rotation_targets(changed, _contract())
    pd.testing.assert_frame_equal(
        features_before.loc[features_before["date"].le(cutoff)].reset_index(drop=True),
        features_after.loc[features_after["date"].le(cutoff)].reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(
        targets_before.loc[targets_before["signal_date"].le(cutoff)].reset_index(drop=True),
        targets_after.loc[targets_after["signal_date"].le(cutoff)].reset_index(drop=True),
    )
