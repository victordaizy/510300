"""桶3+4分层低波动选择规则测试。"""

from __future__ import annotations

import pandas as pd

from research.a_share_hash_holdout_alpha_v1 import FEATURE_COLUMNS
from scripts.run_a_share_bucket34_time_holdout_stratified_lowvol_v1 import select_targets


def test_each_bucket_selects_exactly_one_lowest_volatility_stock() -> None:
    rows = []
    for date in pd.to_datetime(["2024-01-02", "2024-03-28"]):
        for bucket, values in ((3, (("A", 0.9), ("B", 0.7))), (4, (("C", 0.6), ("D", 0.8)))):
            for code, score in values:
                rows.append(
                    {
                        "date": date,
                        "con_code": code,
                        "split_bucket": bucket,
                        "signal_output": "SIGNAL_READY",
                        "raw_close": 20.0,
                        FEATURE_COLUMNS[4]: score,
                    }
                )
    contract = {"universe": {"minimum_signal_price_cny": 10.0}}
    targets = select_targets(pd.DataFrame(rows), contract)
    assert targets.groupby("signal_date").size().eq(2).all()
    assert targets.groupby(["signal_date", "split_bucket"]).size().eq(1).all()
    assert set(targets.loc[targets["split_bucket"].eq(3), "con_code"]) == {"A"}
    assert set(targets.loc[targets["split_bucket"].eq(4), "con_code"]) == {"D"}
    assert targets.groupby("signal_date")["selection_rank"].apply(lambda values: list(values) == [1, 2]).all()


def test_minimum_price_is_applied_before_ranking() -> None:
    frame = pd.DataFrame(
        [
            {"date": pd.Timestamp("2024-01-02"), "con_code": "A", "split_bucket": 3, "signal_output": "SIGNAL_READY", "raw_close": 9.0, FEATURE_COLUMNS[4]: 1.0},
            {"date": pd.Timestamp("2024-01-02"), "con_code": "B", "split_bucket": 3, "signal_output": "SIGNAL_READY", "raw_close": 20.0, FEATURE_COLUMNS[4]: 0.8},
            {"date": pd.Timestamp("2024-01-02"), "con_code": "C", "split_bucket": 4, "signal_output": "SIGNAL_READY", "raw_close": 20.0, FEATURE_COLUMNS[4]: 0.7},
        ]
    )
    targets = select_targets(frame, {"universe": {"minimum_signal_price_cny": 10.0}})
    assert set(targets["con_code"]) == {"B", "C"}
