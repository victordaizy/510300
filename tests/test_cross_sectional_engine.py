"""多股票下一开盘回测引擎测试。"""

from __future__ import annotations

import unittest

import pandas as pd

from backtest.cross_sectional_engine import CrossSectionalCosts, run_weighted_open_backtest


class CrossSectionalEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dates = pd.bdate_range("2025-01-02", periods=4)
        rows = []
        prices = {
            "000001.SZ": [(10.0, 10.0), (10.0, 11.0), (11.0, 11.0), (11.0, 11.0)],
            "000002.SZ": [(20.0, 20.0), (20.0, 20.0), (20.0, 22.0), (22.0, 22.0)],
        }
        for symbol, values in prices.items():
            for date, (open_price, close_price) in zip(self.dates, values, strict=True):
                rows.append(
                    {
                        "date": date,
                        "con_code": symbol,
                        "total_return_open": open_price,
                        "total_return_close": close_price,
                        "raw_open": open_price,
                        "is_suspended": False,
                    }
                )
        self.panel = pd.DataFrame(rows)
        self.zero_costs = CrossSectionalCosts(
            commission_rate=0.0,
            minimum_commission_cny=0.0,
            stamp_duty_sell_rate=0.0,
            slippage_bps_per_leg=0.0,
            cash_annual_rate=0.0,
            lot_size=100,
        )

    def test_卖出印花税按2023年政策生效日分段(self) -> None:
        costs = CrossSectionalCosts(
            stamp_duty_sell_rate=0.0005,
            stamp_duty_sell_rate_before_reduction=0.001,
            stamp_duty_reduction_effective_date="2023-08-28",
        )
        self.assertAlmostEqual(costs.stamp_duty_rate_for_date("2023-08-25"), 0.001)
        self.assertAlmostEqual(costs.stamp_duty_rate_for_date("2023-08-28"), 0.0005)

    def test_收盘信号在下一交易日开盘成交并获得当日盘中收益(self) -> None:
        targets = pd.DataFrame(
            {
                "signal_date": [self.dates[0]],
                "con_code": ["000001.SZ"],
                "target_weight": [1.0],
                "regime": ["BULL"],
            }
        )
        ledger, trades = run_weighted_open_backtest(
            self.panel,
            targets,
            10000.0,
            self.zero_costs,
            self.dates[0],
            self.dates[-1],
        )
        self.assertEqual(trades.iloc[0]["signal_date"], self.dates[0])
        self.assertEqual(trades.iloc[0]["date"], self.dates[1])
        self.assertAlmostEqual(ledger.loc[1, "equity"], 11000.0)

    def test_近似涨停的新目标不假设可以买入(self) -> None:
        panel = self.panel.copy()
        mask = panel["con_code"].eq("000001.SZ") & panel["date"].eq(self.dates[1])
        panel.loc[mask, ["total_return_open", "total_return_close", "raw_open"]] = 11.0
        targets = pd.DataFrame(
            {
                "signal_date": [self.dates[0]],
                "con_code": ["000001.SZ"],
                "target_weight": [1.0],
            }
        )
        ledger, trades = run_weighted_open_backtest(
            panel,
            targets,
            10000.0,
            self.zero_costs,
            self.dates[0],
            self.dates[-1],
            maximum_open_gap_for_trade=0.095,
        )
        self.assertTrue(trades.empty)
        self.assertEqual(int(ledger.loc[1, "blocked_buy_count"]), 1)
        self.assertAlmostEqual(ledger.loc[1, "cash"], 10000.0)

    def test_停牌旧仓无法卖出时继续保留(self) -> None:
        panel = self.panel.copy()
        suspended = panel["con_code"].eq("000001.SZ") & panel["date"].eq(self.dates[2])
        panel.loc[suspended, "is_suspended"] = True
        targets = pd.DataFrame(
            {
                "signal_date": [self.dates[0], self.dates[1]],
                "con_code": ["000001.SZ", "000002.SZ"],
                "target_weight": [1.0, 1.0],
            }
        )
        ledger, trades = run_weighted_open_backtest(
            panel,
            targets,
            10000.0,
            self.zero_costs,
            self.dates[0],
            self.dates[-1],
        )
        day_three_sells = trades.loc[
            trades["date"].eq(self.dates[2]) & trades["con_code"].eq("000001.SZ")
        ]
        self.assertTrue(day_three_sells.empty)
        self.assertEqual(int(ledger.loc[2, "blocked_sell_count"]), 1)
        self.assertGreaterEqual(int(ledger.loc[2, "position_count"]), 1)


if __name__ == "__main__":
    unittest.main()
