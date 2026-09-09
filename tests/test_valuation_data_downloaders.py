"""长期估值数据下载器的纯函数测试。"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from scripts.download_china_government_bond_yields import (
    build_download_windows,
    normalize_curve,
)
from scripts.download_csi300_point_in_time_financials import (
    build_point_in_time_events,
    build_quarter_periods,
)


def test_国债下载窗口均小于一年且首尾连续() -> None:
    windows = build_download_windows(pd.Timestamp("2020-01-01"), pd.Timestamp("2022-02-01"))
    assert windows[0][0] == pd.Timestamp("2020-01-01")
    assert windows[-1][1] == pd.Timestamp("2022-02-01")
    assert all((end - start).days < 365 for start, end in windows)
    assert all(windows[index][1] + pd.Timedelta(days=1) == windows[index + 1][0] for index in range(len(windows) - 1))


def test_国债曲线只保留国债并标准化字段() -> None:
    raw = pd.DataFrame(
        {
            "曲线名称": ["中债国债收益率曲线", "中债商业银行普通债收益率曲线(AAA)"],
            "日期": ["2021-08-02", "2021-08-02"],
            "1年": [2.12, 3.0],
            "10年": [2.82, 4.0],
        }
    )
    result = normalize_curve(raw, datetime(2026, 8, 13, tzinfo=ZoneInfo("Asia/Shanghai")))
    assert len(result) == 1
    assert result.iloc[0]["cgb_1y"] == 2.12
    assert result.iloc[0]["cgb_10y"] == 2.82


def test_季度报告期生成覆盖完整季度() -> None:
    assert build_quarter_periods(pd.Timestamp("2020-07-01"), pd.Timestamp("2021-04-01")) == [
        "20200930", "20201231", "20210331"
    ]


def test_点时合并不会把后续修订提前到首次公告() -> None:
    income = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ"],
            "ann_date": ["20210420", "20210520"],
            "f_ann_date": ["20210420", "20210520"],
            "end_date": ["20210331", "20210331"],
            "report_type": ["1", "4"],
            "revenue": [100.0, 110.0],
            "n_income_attr_p": [10.0, 11.0],
        }
    )
    balance = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "ann_date": ["20210420"],
            "f_ann_date": ["20210420"],
            "end_date": ["20210331"],
            "report_type": ["1"],
            "total_hldr_eqy_exc_min_int": [200.0],
            "total_share": [20.0],
        }
    )
    indicator = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "ann_date": ["20210420"],
            "end_date": ["20210331"],
            "eps": [0.5],
            "dt_eps": [0.5],
            "bps": [10.0],
            "roe": [5.0],
        }
    )
    events = build_point_in_time_events(
        income,
        balance,
        indicator,
        datetime(2026, 8, 13, tzinfo=ZoneInfo("Asia/Shanghai")),
    )
    first = events.loc[events["available_at"].eq(pd.Timestamp("2021-04-20"))].iloc[0]
    revised = events.loc[events["available_at"].eq(pd.Timestamp("2021-05-20"))].iloc[0]
    assert first["revenue_cny"] == 100.0
    assert revised["revenue_cny"] == 110.0
    assert revised["equity_parent_cny"] == 200.0
