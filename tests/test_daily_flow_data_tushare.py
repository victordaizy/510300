"""日频资金流下载标准化测试。"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from scripts.download_daily_flow_data_tushare import (
    normalize_fund_share,
    normalize_margin,
    normalize_northbound,
)


def test_fund_share_unit_converts_from_ten_thousand() -> None:
    raw = pd.DataFrame(
        {
            "ts_code": ["510300.SH"],
            "trade_date": ["20240102"],
            "fd_share": [123.45],
            "fund_type": ["ETF"],
            "market": ["E"],
        }
    )
    result = normalize_fund_share(raw, datetime.now(timezone.utc))
    assert result.loc[0, "fund_shares"] == 1_234_500.0


def test_margin_net_buy_equals_buy_minus_repayment() -> None:
    raw = pd.DataFrame(
        {
            "trade_date": ["20240102"],
            "ts_code": ["510300.SH"],
            "rzye": [100.0],
            "rqye": [2.0],
            "rzmre": [20.0],
            "rqyl": [1.0],
            "rzche": [12.0],
            "rqchl": [0.0],
            "rqmcl": [0.0],
            "rzrqye": [102.0],
        }
    )
    result = normalize_margin(raw, datetime.now(timezone.utc))
    assert result.loc[0, "financing_net_buy_cny"] == 8.0


def test_northbound_semantics_break_on_2024_08_19() -> None:
    raw = pd.DataFrame(
        {
            "trade_date": ["20240816", "20240819"],
            "ggt_ss": [1.0, 1.0],
            "ggt_sz": [1.0, 1.0],
            "hgt": [1.0, 100.0],
            "sgt": [1.0, 100.0],
            "north_money": [2.0, 200.0],
            "south_money": [2.0, 2.0],
        }
    )
    result = normalize_northbound(raw, datetime.now(timezone.utc))
    assert bool(result.loc[0, "usable_as_net_flow"])
    assert not bool(result.loc[1, "usable_as_net_flow"])
