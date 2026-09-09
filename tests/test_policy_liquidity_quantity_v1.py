"""央行规模缺失、资金时钟与完整账户的针对性测试。"""
import unittest

import numpy as np
import pandas as pd

from research.adaptive_allocation_v1 import factors, spec, simulate
from research.policy_liquidity_quantity_v1 import quantity_events, prepare_features, simulate_policy
from tests.test_adaptive_allocation_v1 import synthetic


def notices_for(dates):
    dates = pd.DatetimeIndex(dates)
    n = len(dates)
    return pd.DataFrame({"notice_date": dates, "published_at": dates + pd.Timedelta(hours=9, minutes=20),
                         "has_seven_day_row": True, "seven_day_operation_amount_100m": np.arange(1, n + 1) * 100.0,
                         "seven_day_rate_percent": np.full(n, 2.0), "source_url": "合成测试", "raw_path": "合成测试"})


class PolicyLiquidityQuantityTests(unittest.TestCase):
    def test_absent_quantity_is_not_zero_and_explicit_zero_survives(self):
        notices = notices_for(pd.bdate_range("2015-01-01", periods=45))
        notices.loc[30, "has_seven_day_row"] = False
        notices.loc[30, "seven_day_operation_amount_100m"] = np.nan
        notices.loc[31, "seven_day_operation_amount_100m"] = 0
        actual = quantity_events(notices)
        self.assertEqual(len(actual), 44)
        self.assertNotIn(notices.notice_date.iloc[30], set(actual.notice_date))
        zero = actual.loc[actual.notice_date == notices.notice_date.iloc[31]].iloc[0]
        self.assertEqual(zero.quantity_log, 0)

    def test_surprise_uses_twenty_previous_confirmed_announcements(self):
        notices = notices_for(pd.bdate_range("2015-01-01", periods=45))
        actual = quantity_events(notices)
        values = np.log1p(notices.seven_day_operation_amount_100m.to_numpy())
        expected = (values[25] - values[5:25].mean()) / values[5:25].std(ddof=1)
        self.assertAlmostEqual(actual.quantity_surprise20.iloc[25], expected)
        modified = notices.copy()
        modified.loc[35:, "seven_day_operation_amount_100m"] *= 100
        np.testing.assert_allclose(actual.quantity_surprise20.iloc[:35], quantity_events(modified).quantity_surprise20.iloc[:35], equal_nan=True)

    def test_rate_next_open_and_late_announcement_do_not_leak(self):
        prices, dividends = synthetic()
        data, _ = factors(prices, dividends)
        notices = notices_for(data.date.iloc[300:400])
        notices.loc[50, "published_at"] = notices.notice_date.iloc[50] + pd.Timedelta(hours=16)
        dr = pd.DataFrame({"date": data.date, "dr007": 2.0 + np.arange(len(data)) / 10000})
        config = {**spec(), "maximum_quantity_age_days": 10, "maximum_dr007_age_days": 7}
        actual = prepare_features(data, notices, dr, config)
        self.assertEqual(actual.dr_date.iloc[350], data.date.iloc[349])
        self.assertEqual(actual.quantity_notice_date.iloc[350], data.date.iloc[349])
        self.assertEqual(actual.quantity_notice_date.iloc[351], data.date.iloc[351])
        self.assertLessEqual(actual.dr_available_at.iloc[350], actual.origin_time.iloc[350])

    def test_complete_predictions_match_previous_account_implementation(self):
        prices, dividends = synthetic()
        data, _ = factors(prices, dividends)
        config = spec()
        config["evaluation_start"] = str(data.date.iloc[700].date())
        prediction = np.sin(np.arange(len(data)) / 31) * 0.03
        a, _ = simulate_policy(data, dividends, config, config["costs"]["BASE"], "TEST", prediction, 20)
        b, _ = simulate(data, dividends, config, config["costs"]["BASE"], config["evaluation_start"], "TEST", prediction=prediction, horizon=20, rebalance=20)
        for column in ("equity", "cash", "shares", "net_return", "commission", "slippage_cost"):
            np.testing.assert_allclose(a[column], b[column], atol=1e-10, rtol=0)

    def test_no_view_keeps_inventory_and_unsets_target(self):
        prices, dividends = synthetic()
        data, _ = factors(prices, dividends)
        config = spec()
        config["evaluation_start"] = str(data.date.iloc[700].date())
        prediction = np.full(len(data), 0.1)
        prediction[700] = np.nan
        ledger, decisions = simulate_policy(data, dividends, config, config["costs"]["BASE"], "TEST", prediction, 1)
        self.assertGreater(ledger.shares.iloc[0], 0)
        self.assertEqual(ledger.shares.iloc[1], ledger.shares.iloc[0])
        self.assertEqual(ledger.filled_quantity.iloc[1], 0)
        self.assertEqual(decisions.view.iloc[1], "NO_VIEW")
        self.assertTrue(pd.isna(decisions.reference_weight.iloc[1]))


if __name__ == "__main__":
    unittest.main()
