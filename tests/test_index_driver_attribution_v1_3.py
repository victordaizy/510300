"""点时指数驱动归因 V1.3 研究窗口测试。"""

from __future__ import annotations

import pandas as pd

from research.index_driver_attribution_v1_3 import (
    restrict_industry_intervals_to_research_window,
)


def test_out_of_window_ambiguity_is_excluded_before_resolution() -> None:
    intervals = pd.DataFrame(
        {
            "con_code": ["A.SH", "A.SH", "B.SZ"],
            "industry_l1": ["旧一", "旧二", "当前"],
            "in_date": pd.to_datetime(["2014-01-01", "2014-01-01", "2020-01-01"]),
            "out_date": pd.to_datetime(["2015-01-01", "2015-01-01", None]),
        }
    )
    relevant = restrict_industry_intervals_to_research_window(
        intervals, pd.Timestamp("2019-12-23"), pd.Timestamp("2026-08-14")
    )
    assert relevant["con_code"].tolist() == ["B.SZ"]


def test_interval_touching_start_at_open_end_is_excluded() -> None:
    intervals = pd.DataFrame(
        {
            "in_date": pd.to_datetime(["2019-01-01", "2019-01-01"]),
            "out_date": pd.to_datetime(["2019-12-23", "2019-12-24"]),
        }
    )
    relevant = restrict_industry_intervals_to_research_window(
        intervals, pd.Timestamp("2019-12-23"), pd.Timestamp("2026-08-14")
    )
    assert len(relevant) == 1
    assert relevant.iloc[0]["out_date"] == pd.Timestamp("2019-12-24")
