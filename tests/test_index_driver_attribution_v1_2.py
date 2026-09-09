"""点时指数驱动归因 V1.2 冲突区间测试。"""

from __future__ import annotations

import pandas as pd
import pytest

from research.index_driver_attribution_v1_2 import (
    truncate_superseded_l1_intervals,
)


def _conflicting_intervals(same_start: bool = False) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "con_code": ["300442.SZ", "300442.SZ", "300442.SZ"],
            "industry_l1": ["机械", "机械", "环保"],
            "in_date": pd.to_datetime(
                ["2015-05-04", "2015-08-03", "2015-05-04" if same_start else "2022-09-19"]
            ),
            "out_date": pd.to_datetime(["2022-09-16", "2022-10-10", None]),
            "classification_usage": ["POINT_IN_TIME_INTERVAL_CITIC"] * 3,
            "source": ["中信测试"] * 3,
        }
    )


def test_latest_starting_l1_truncates_superseded_l1() -> None:
    resolved = truncate_superseded_l1_intervals(_conflicting_intervals())
    old = resolved.loc[resolved["industry_l1"].eq("机械")].iloc[0]
    new = resolved.loc[resolved["industry_l1"].eq("环保")].iloc[0]
    assert old["out_date"] == pd.Timestamp("2022-09-19")
    assert new["in_date"] == pd.Timestamp("2022-09-19")
    date = pd.Timestamp("2022-09-19")
    active = resolved.loc[
        resolved["in_date"].le(date)
        & (resolved["out_date"].isna() | resolved["out_date"].gt(date))
    ]
    assert active["industry_l1"].tolist() == ["环保"]


def test_same_latest_start_with_different_l1_is_rejected() -> None:
    with pytest.raises(ValueError, match="同一最新生效日存在多个一级行业"):
        truncate_superseded_l1_intervals(_conflicting_intervals(same_start=True))
