"""日线二次测试的顺序、风险与真实账户边界检查。"""
import unittest
import numpy as np
import pandas as pd
from research import daily_supply_test_v1 as m


def frame(n=44):
    d = pd.DataFrame({"date": pd.bdate_range("2015-01-01", periods=n), "open": 10.4, "close": 10.5,
                      "high": 11., "low": 10., "volume": 100., "dividend": 0., "cash_shift": 0.,
                      "ao": 10.4, "ac": 10.5, "ah": 11., "al": 10., "atr20": .5,
                      "support20": 10., "rv20": .01, "context_available": True,
                      "weekly_macd_rising": True, "daily_macd_rising": True,
                      "low_volatility": True, "relative_volume_high": True, "relative_volume": 1., "prior_rv_ratio": .8})
    return d


def example(volume=70.):
    d = frame()
    d.loc[22, ["al", "ac", "ah", "volume"]] = [9.8, 10.25, 10.7, 120.]
    d.loc[23, ["al", "ac", "ah"]] = [10., 10.6, 10.8]
    d.loc[24, ["al", "ac", "ah", "volume"]] = [9.9, 10.1, 10.3, volume]
    d.loc[25, ["al", "ac", "ah"]] = [10.05, 10.5, 10.7]
    return d


def account_input():
    d = frame(12)
    d.loc[:, ["open", "close", "ao", "ac"]] = 10.
    d.loc[3, ["close", "ac"]] = 10.6
    sig = pd.DataFrame([{"policy": "SUPPLY_TEST", "signal_idx": 2, "signal_date": d.date.iloc[2],
                         "event_id": "人工事件", "stop_index": 9., "target_index": 10.3,
                         "daily_macd_rising": True, "weekly_macd_rising": True,
                         "low_volatility": True, "relative_volume_high": True}])
    risk = pd.DataFrame({"idx": range(len(d)), "es95": .02})
    dividends = pd.DataFrame(columns=["record_date", "ex_date", "payment_date", "cash_dividend_per_share"])
    return d, sig, risk, dividends


class SupplyTestCases(unittest.TestCase):
    def test_先收复后回测再确认(self):
        episodes, _, signals = m.detect(example())
        actual = signals.set_index("policy").signal_idx.to_dict()
        self.assertEqual(actual, {"DIRECT_RECLAIM": 22, "PRICE_TEST": 25, "SUPPLY_TEST": 25})
        self.assertEqual(episodes.iloc[0].test_idx, 24)
        self.assertTrue(episodes.iloc[0].supply_pass)

    def test_放量回测只留价格对照(self):
        _, _, signals = m.detect(example(150.))
        self.assertEqual(set(signals.policy), {"DIRECT_RECLAIM", "PRICE_TEST"})

    def test_未来价格不改历史触发(self):
        d = example()
        before = m.detect(d.iloc[:27].copy())[2]
        d.loc[30:, ["al", "ac", "ah"]] = [1., 100., 200.]
        after = m.detect(d)[2]
        pd.testing.assert_frame_equal(before, after.loc[after.signal_idx < 27].reset_index(drop=True))

    def test_次日才能兑现收盘退出(self):
        d, s, r, dv = account_input()
        ledger, trades, orders, _, _ = m.account(d, dv, s, r, "SUPPLY_TEST", "STRESS", start=d.date.iloc[1])
        self.assertEqual(int(trades.iloc[0].entry_idx), 3)
        self.assertEqual(int(trades.iloc[0].exit_idx), 4)
        self.assertLess(trades.iloc[0].net_pnl, 0.)
        self.assertLess(ledger.accounting_error.abs().max(), 1e-7)
        self.assertEqual(list(orders.side), ["BUY", "SELL"])

    def test_跌停顺延退出(self):
        d, s, r, dv = account_input()
        d.loc[4, ["open", "ao", "close", "ac"]] = [9.54, 9.54, 10.4, 10.4]
        _, trades, _, rejects, _ = m.account(d, dv, s, r, "SUPPLY_TEST", "STRESS", start=d.date.iloc[1])
        self.assertEqual(int(trades.iloc[0].exit_idx), 5)
        self.assertIn("SELL_DEFERRED_T1_OR_LIMIT", rejects.reason.tolist())

    def test_除息与到账分开(self):
        d, s, r, _ = account_input()
        d.loc[:, ["open", "close", "ao", "ac"]] = 10.
        d.loc[4:, ["open", "close"]] = 9.9
        d.loc[4, "dividend"] = .1
        s["target_index"] = 20.
        dv = pd.DataFrame([{"record_date": d.date.iloc[3], "ex_date": d.date.iloc[4],
                            "payment_date": d.date.iloc[6], "cash_dividend_per_share": .1}])
        ledger, _, _, _, terminal = m.account(d, dv, s, r, "SUPPLY_TEST", "STRESS", start=d.date.iloc[1])
        byday = ledger.set_index("idx")
        amount = byday.loc[3, "shares"] * .1
        self.assertAlmostEqual(byday.loc[4, "dividend_accrual"], amount)
        self.assertAlmostEqual(byday.loc[4, "receivable"], amount)
        self.assertAlmostEqual(byday.loc[6, "dividend_paid"], amount)
        self.assertAlmostEqual(byday.loc[6, "receivable"], 0.)
        self.assertGreater(terminal["open_shares"], 0)

    def test_开盘涨价不能增加原份额(self):
        budget = m.limits(200000., 200000., .08)
        plan = m.cap_quantity(4., budget, 10000000, 200000.)
        actual = m.cap_quantity(4.4, budget, plan, 200000.)
        self.assertLessEqual(actual, plan)
        self.assertTrue(m.risk_ok(actual, 4.4, budget, True))
        self.assertFalse(m.risk_ok(actual + 100, 4.4, budget, True))

    def test_风险标签成熟且历史前缀一致(self):
        d = frame(1000)
        d["date"] = pd.bdate_range("2012-01-02", periods=len(d))
        d["open"] = 10. + np.sin(np.arange(len(d)) / 15)
        d["ao"] = d.open
        d["rv20"] = .01 + .002 * np.cos(np.arange(len(d)) / 20)
        full, members = m.risk_estimates(d)
        early, _ = m.risk_estimates(d.iloc[:900].copy())
        self.assertTrue((members.label_exit_idx <= members.decision_idx).all())
        pd.testing.assert_frame_equal(early, full.loc[full.idx < 900].reset_index(drop=True))


if __name__ == "__main__":
    unittest.main()
