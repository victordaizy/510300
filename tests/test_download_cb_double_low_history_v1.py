"""可转债双低原始历史采集器的纯合成测试。"""

from __future__ import annotations

import pandas as pd

from scripts.download_cb_double_low_history_v1 import (
    normalize_master,
    normalize_price_history,
    normalize_value_history,
)


def test_normalize_master_uses_listing_date_and_exchange_without_market_outcome() -> None:
    raw = pd.DataFrame(
        [
            [
                "113001",
                "测试转债",
                "2017-01-01",
                "783001",
                100,
                "600001",
                "测试股份",
                10,
                9,
                111,
                105,
                5,
                "2016-12-30",
                1,
                5,
                "2017-01-03",
                0.01,
                "2017-01-10",
                "AA",
            ],
            [
                "123001",
                "深市转债",
                "2018-01-01",
                "370001",
                100,
                "300001",
                "深市股份",
                20,
                18,
                111,
                101,
                10,
                "2017-12-30",
                1,
                4,
                "2018-01-03",
                0.02,
                "2018-01-10",
                "AA-",
            ],
            [
                "404001",
                "北交转债",
                "2023-01-01",
                "889001",
                100,
                "830001",
                "北交股份",
                10,
                9,
                111,
                105,
                5,
                "2022-12-30",
                1,
                5,
                "2023-01-03",
                0.01,
                "2023-01-10",
                "AA",
            ],
        ]
    )
    result = normalize_master(raw)
    assert result["bond_code"].tolist() == ["113001", "123001"]
    assert result["exchange"].tolist() == ["SSE", "SZSE"]
    assert "current_price" not in result.columns
    assert "delist_date" not in result.columns


def test_normalize_price_history_rejects_post_cutoff_and_invalid_prices() -> None:
    raw = pd.DataFrame(
        {
            "date": ["2020-01-02", "2020-01-03", "2026-08-17"],
            "open": [100.0, -1.0, 101.0],
            "high": [101.0, 101.0, 102.0],
            "low": [99.0, 99.0, 100.0],
            "close": [100.5, 100.0, 101.5],
            "volume": [1000, 1000, 1000],
        }
    )
    result = normalize_price_history(raw, "113001")
    assert len(result) == 1
    assert result.iloc[0]["date"] == pd.Timestamp("2020-01-02")


def test_normalize_value_history_uses_canonical_position_mapping() -> None:
    raw = pd.DataFrame(
        [
            ["2020-01-02", 101.0, 92.0, 98.0, 9.78, 3.06],
            ["2026-08-17", 102.0, 92.5, 99.0, 10.27, 3.03],
        ],
        columns=["日期", "收盘", "纯债价值", "转股价值", "纯债溢价", "转股溢价"],
    )
    result = normalize_value_history(raw, "113001")
    assert len(result) == 1
    row = result.iloc[0]
    assert row["value_table_close"] == 101.0
    assert row["bond_value"] == 92.0
    assert row["conversion_value"] == 98.0
    assert row["conversion_premium_rate_pct"] == 3.06
