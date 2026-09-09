"""沪深300市场状态特征的防未来函数回归测试。"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from research.build_market_state_features import build_market_state_features


class MarketStateFeatureTests(unittest.TestCase):
    def setUp(self) -> None:
        dates = pd.bdate_range("2020-01-01", periods=400)
        close = pd.Series(np.linspace(3000.0, 4200.0, len(dates)))
        self.index = pd.DataFrame(
            {
                "date": dates,
                "open": close - 5,
                "high": close + 10,
                "low": close - 10,
                "close": close,
                "volume": np.linspace(1e8, 2e8, len(dates)),
            }
        )
        self.valuation = pd.DataFrame(
            {
                "date": dates,
                "pe_ttm": np.linspace(10.0, 18.0, len(dates)),
                "pe_static": np.linspace(11.0, 19.0, len(dates)),
                "pb": np.linspace(1.1, 1.9, len(dates)),
            }
        )

    def test_信号使用当日收盘并记录信息截止日(self) -> None:
        features = build_market_state_features(self.index, self.valuation, minimum_history=252)
        target_date = features.iloc[-1]["date"]
        current_pe = self.valuation.loc[self.valuation["date"] == target_date, "pe_ttm"].iloc[0]
        self.assertEqual(features.iloc[-1]["signal_asof_date"], target_date)
        self.assertAlmostEqual(features.iloc[-1]["signal_pe_ttm"], current_pe)

    def test_修改当日收盘只能改变当日及以后特征(self) -> None:
        original = build_market_state_features(self.index, self.valuation, minimum_history=252)
        changed_index = self.index.copy()
        changed_index.loc[changed_index.index[-1], ["open", "high", "low", "close"]] *= 2
        changed_valuation = self.valuation.copy()
        changed_valuation.loc[changed_valuation.index[-1], ["pe_ttm", "pe_static", "pb"]] *= 10
        changed = build_market_state_features(changed_index, changed_valuation, minimum_history=252)
        self.assertAlmostEqual(
            original.iloc[-2]["signal_trend_close_over_ma120"],
            changed.iloc[-2]["signal_trend_close_over_ma120"],
        )
        self.assertNotAlmostEqual(
            original.iloc[-1]["signal_trend_close_over_ma120"],
            changed.iloc[-1]["signal_trend_close_over_ma120"],
        )
        self.assertNotAlmostEqual(original.iloc[-1]["signal_pe_ttm"], changed.iloc[-1]["signal_pe_ttm"])

    def test_五年百分位必须使用至少252日预热(self) -> None:
        features = build_market_state_features(self.index, self.valuation, minimum_history=252)
        first_valid = features.loc[features["signal_pe_ttm_percentile_5y"].notna()].index.min()
        self.assertGreaterEqual(first_valid, 251)


if __name__ == "__main__":
    unittest.main()
