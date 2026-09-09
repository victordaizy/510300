"""日频ETF申赎与融资方向候选测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.daily_etf_flow_direction_challenger import prepare_daily_flow_features


def test_daily_flow_features_use_only_current_and_prior_records() -> None:
    dates = pd.bdate_range("2024-01-02", periods=10)
    base = pd.DataFrame(
        {
            "date": dates,
            "etf_amount": 1000.0,
            "exec_total_return_20d_net": np.nan,
        }
    )
    shares = pd.DataFrame(
        {
            "date": [dates[0], dates[5]],
            "fund_shares": [100.0, 110.0],
        }
    )
    margin = pd.DataFrame(
        {
            "date": dates,
            "rzye": np.arange(10, dtype=float) + 100.0,
            "financing_net_buy_cny": 10.0,
        }
    )
    result = prepare_daily_flow_features(base, shares, margin)
    assert result.loc[4, "fund_shares"] == 100.0
    assert result.loc[5, "fund_shares"] == 110.0
    assert result.loc[5, "share_snapshot_date"] == dates[5]
    assert np.isclose(result.loc[5, "fund_share_change_5d_pct"], 0.10)
    assert np.isclose(result.loc[4, "financing_net_buy_5d_to_turnover"], 0.01)
