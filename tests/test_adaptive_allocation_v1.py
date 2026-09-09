"""检查时间可用性、分红权益、组合选择与账户费用，均不读取真实收益。"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from research.adaptive_allocation_v1 import (
    adaptive_targets, eligible_training, factors, simulate, spec,
)
from research.intraday_overnight_increment_v1 import holding_total_return


def synthetic(n: int = 900):
    rng = np.random.default_rng(47)
    dates = pd.bdate_range("2011-01-03", periods=n)
    close = 3 * np.exp(np.cumsum(rng.normal(0.0002, 0.009, n)))
    op = close * np.exp(rng.normal(0, 0.002, n))
    prices = pd.DataFrame({"date": dates, "open": op, "close": close,
                           "high": np.maximum(op, close) * 1.005, "low": np.minimum(op, close) * 0.995,
                           "volume": np.full(n, 1000000.0), "symbol": "510300.SH"})
    dividends = pd.DataFrame({"record_date": [dates[499]], "ex_date": [dates[500]],
                              "payment_date": [dates[502]], "cash_dividend_per_share": [0.10]})
    prices.loc[500:, ["open", "close", "high", "low"]] -= 0.10
    return prices, dividends


class AdaptiveAllocationTests(unittest.TestCase):
    def test_future_prices_do_not_change_earlier_factors(self):
        prices, div = synthetic()
        full, columns = factors(prices, div)
        altered = prices.copy()
        altered.loc[700:, ["open", "close", "high", "low"]] *= 1.2
        other, _ = factors(altered, div)
        np.testing.assert_allclose(full[columns].iloc[:700], other[columns].iloc[:700], equal_nan=True)
        prefix, _ = factors(prices.iloc[:700], div)
        np.testing.assert_allclose(full[columns].iloc[:700], prefix[columns], equal_nan=True)

    def test_labels_only_enter_after_exit_open_and_rolling_cutoff(self):
        prices, div = synthetic()
        data, _ = factors(prices, div)
        train = eligible_training(data, origin=800, horizon=20, window=400)
        self.assertEqual(train[-1] + 21, 800)
        self.assertGreaterEqual(train[0], 400)
        self.assertNotIn(780, train)

    def test_dividend_eligibility_before_and_after_record(self):
        prices, div = synthetic()
        before, entitlement = holding_total_return(prices, div, 499, 501)
        self.assertEqual(entitlement, 0.1)
        after, no_entitlement = holding_total_return(prices, div, 500, 502)
        self.assertEqual(no_entitlement, 0)
        self.assertAlmostEqual(before, (prices.open.iloc[501] + 0.1) / prices.open.iloc[499] - 1)
        self.assertAlmostEqual(after, prices.open.iloc[502] / prices.open.iloc[500] - 1)

    def test_orders_fixed_before_changed_next_open(self):
        prices, div = synthetic()
        data, _ = factors(prices, div)
        config = spec()
        start = str(data.date.iloc[490].date())
        targets = np.ones(len(data))
        ledger, decisions = simulate(data, div, config, config["costs"]["BASE"], start, "TEST", targets=targets)
        changed = data.copy()
        changed.loc[490, "open"] *= 1.04
        second, second_decisions = simulate(changed, div, config, config["costs"]["BASE"], start, "TEST", targets=targets)
        self.assertEqual(decisions.requested_quantity.iloc[0], second_decisions.requested_quantity.iloc[0])
        self.assertGreater(ledger.filled_quantity.iloc[0], second.filled_quantity.iloc[0])
        self.assertLess(float(ledger.accounting_error.abs().max()), 1e-6)

    def test_receivable_not_spendable_and_terminal_fees_present(self):
        prices, div = synthetic(520)
        data, _ = factors(prices, div)
        config = spec()
        targets = np.ones(len(data))
        ledger, _ = simulate(data, div, config, config["costs"]["BASE"], str(data.date.iloc[490].date()), "TEST", targets=targets)
        indexed = ledger.set_index("date")
        record = indexed.loc[data.date.iloc[499]]
        ex = indexed.loc[data.date.iloc[500]]
        pay = indexed.loc[data.date.iloc[502]]
        self.assertAlmostEqual(ex.dividend_receivable, record.shares * 0.1)
        self.assertEqual(ex.dividend_paid, 0)
        self.assertAlmostEqual(pay.dividend_paid, record.shares * 0.1)
        self.assertEqual(pay.dividend_receivable, 0)
        self.assertEqual(ledger.shares.iloc[-1], 0)
        self.assertGreater(ledger.commission.iloc[-1], 0)
        self.assertEqual(ledger.mark_clock.iloc[-1], "OPEN_TERMINAL")
        self.assertTrue((ledger.cash >= -1e-7).all())

    def test_flat_cash_sharpe_not_fabricated(self):
        from research.adaptive_allocation_v1 import summarize
        prices, div = synthetic(520)
        data, _ = factors(prices, div)
        config = spec()
        ledger, _ = simulate(data, div, config, config["costs"]["BASE"], str(data.date.iloc[490].date()), "TEST", targets=np.zeros(len(data)))
        self.assertIsNone(summarize(ledger, config)["net_sharpe"])
        self.assertEqual(ledger.commission.sum(), 0)

    def test_selector_future_tail_does_not_change_earlier_choices(self):
        rng = np.random.default_rng(91)
        dates = pd.bdate_range("2015-01-01", periods=900)
        returns = rng.normal(0.0005, 0.01, (900, 3))
        components = np.tile([0.0, 0.5, 1.0], (900, 1))
        origins = np.arange(504, 899)
        first, _ = adaptive_targets(returns, components, dates, origins, ["A", "B", "C"], 504, 1)
        changed = returns.copy()
        changed[700:, 0] = 0.10
        second, _ = adaptive_targets(changed, components, dates, origins, ["A", "B", "C"], 504, 1)
        np.testing.assert_array_equal(first[:700], second[:700])
        self.assertTrue(np.isfinite(first).all())


if __name__ == "__main__":
    unittest.main()
