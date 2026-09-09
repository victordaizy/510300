"""早期外部验证的等权广度测试。"""

from __future__ import annotations

import pandas as pd

from scripts.run_csi300_small_account_external_2015_2021 import build_external_breadth


def test_external_breadth_uses_only_same_day_members() -> None:
    dates = pd.bdate_range("2025-01-02", periods=65)
    rows = []
    for symbol, slope in (("A", 0.1), ("B", -0.1)):
        for index, date in enumerate(dates):
            rows.append(
                {
                    "date": date,
                    "con_code": symbol,
                    "total_return_close": 10.0 + slope * index,
                    "is_index_member": symbol == "A" or index < 60,
                }
            )
    breadth = build_external_breadth(pd.DataFrame(rows))
    latest = breadth.iloc[-1]
    assert latest["member_count"] == 1
    assert latest["equal_above_ma60_share"] == 1.0

