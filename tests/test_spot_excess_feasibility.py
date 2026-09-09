"""现货超额能力需求模拟测试。"""

from __future__ import annotations

import pandas as pd

from research.spot_excess_feasibility import build_non_overlapping_blocks


def test_non_overlapping_blocks_use_horizon_stride() -> None:
    dates = pd.bdate_range("2024-01-02", periods=500)
    targets = pd.DataFrame(
        {
            "date": dates,
            "exec_total_return_20d_net": 0.01,
            "label_end_date_20d": pd.Series(dates).shift(-20),
        }
    )
    benchmark = pd.DataFrame({"date": dates, "close": range(100, 600)})
    blocks = build_non_overlapping_blocks(targets, benchmark, horizon=20)
    assert blocks
    first = blocks[0]
    assert (first["date"].diff().dropna().dt.days >= 20).all()
