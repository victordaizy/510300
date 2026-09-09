"""510300 五年数据管线的回归测试。"""

from __future__ import annotations

import unittest

import pandas as pd

from research.return_anatomy import add_return_columns
from scripts.quality_check_daily import check_daily_data


class DailyQualityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = {
            "project": {"start_date": "2021-08-12", "end_date": "2026-08-12"},
            "symbols": {"etf": "510300.SH"},
            "quality": {"price_tick": 0.001},
        }
        self.data = pd.DataFrame(
            {
                "symbol": ["510300.SH", "510300.SH"],
                "date": pd.to_datetime(["2021-08-12", "2021-08-13"]),
                "open": [5.000, 5.050],
                "high": [5.100, 5.100],
                "low": [4.950, 5.000],
                "close": [5.050, 5.080],
                "volume": [1000, 2000],
                "amount": [5020, 10120],
                "source": ["source", "source"],
                "retrieved_at": ["2026-08-12T16:00:00+08:00"] * 2,
                "volume_unit": ["share"] * 2,
                "amount_unit": ["CNY"] * 2,
            }
        )

    def test_隐含均价超出高低区间必须失败(self) -> None:
        data = self.data.copy()
        data.loc[1, "amount"] = 30000
        errors, _ = check_daily_data(data, self.config)
        self.assertIn("IMPLIED_VWAP_OUTSIDE_RANGE", {item["code"] for item in errors})

    def test_非最小价格单位必须失败(self) -> None:
        data = self.data.copy()
        data.loc[1, "close"] = 5.0805
        errors, _ = check_daily_data(data, self.config)
        self.assertIn("INVALID_PRICE_TICK", {item["code"] for item in errors})


class TotalReturnTests(unittest.TestCase):
    def test_除息现金必须计入总回报(self) -> None:
        prices = pd.DataFrame(
            {
                "date": pd.to_datetime(["2025-06-17", "2025-06-18"]),
                "open": [4.000, 3.910],
                "high": [4.020, 3.950],
                "low": [3.980, 3.900],
                "close": [4.000, 3.920],
            }
        )
        dividends = pd.DataFrame(
            {"ex_date": pd.to_datetime(["2025-06-18"]), "cash_dividend_per_share": [0.088]}
        )
        result = add_return_columns(prices, dividends)
        self.assertAlmostEqual(result.loc[1, "price_return"], -0.02)
        self.assertAlmostEqual(result.loc[1, "total_return"], 0.002)
        self.assertAlmostEqual(
            result.loc[1, "total_return"],
            result.loc[1, "overnight_total_contribution"] + result.loc[1, "intraday_contribution"],
        )


if __name__ == "__main__":
    unittest.main()
