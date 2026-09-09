"""日历条件的日期前缀、执行日、休市边界和账户一致性检查。"""
import unittest

import numpy as np
import pandas as pd

from research.adaptive_allocation_v1 import factors, spec, simulate
from research.calendar_liquidity_timing_v1 import CALENDAR_COLUMNS, calendar_features, prepare, rule_targets, simulate_policy
from tests.test_adaptive_allocation_v1 import synthetic


def config_for_test():
    config = spec()
    config.update({"calendar_first_month_sessions": 3, "calendar_month_end_natural_days": 5,
                   "long_break_minimum_date_gap_days": 4, "post_break_sessions": 3})
    return config


class CalendarLiquidityTests(unittest.TestCase):
    def test_calendar_features_are_unchanged_when_future_sessions_are_removed(self):
        dates = pd.bdate_range("2019-01-02", "2021-12-31")
        dates = dates[~((dates >= "2020-01-24") & (dates <= "2020-02-02"))]
        full = calendar_features(dates, config_for_test())
        prefix = calendar_features(dates[:350], config_for_test())
        pd.testing.assert_frame_equal(full.iloc[:350].reset_index(drop=True), prefix)
        self.assertTrue(np.isfinite(full[CALENDAR_COLUMNS]).all().all())

    def test_month_end_uses_calendar_days_and_holiday_counter_uses_only_past(self):
        dates = pd.to_datetime(["2024-01-29", "2024-01-30", "2024-01-31", "2024-02-01", "2024-02-02",
                                "2024-02-05", "2024-02-06", "2024-02-07", "2024-02-08", "2024-02-19",
                                "2024-02-20", "2024-02-21", "2024-02-22", "2024-02-23", "2024-02-26"])
        features = calendar_features(dates, config_for_test()).set_index("execution_date")
        self.assertEqual(features.loc["2024-02-01", "month_session_ordinal"], 1)
        self.assertEqual(features.loc["2024-02-05", "month_first3"], 1)
        self.assertEqual(features.loc["2024-02-06", "month_first3"], 0)
        self.assertEqual(features.loc["2024-02-19", "previous_session_gap_days"], 11)
        self.assertEqual(features.loc["2024-02-21", "post_break_first3"], 1)
        self.assertEqual(features.loc["2024-02-22", "post_break_first3"], 0)
        self.assertEqual(features.loc["2024-02-23", "month_last5_calendar"], 0)
        self.assertEqual(features.loc["2024-02-26", "month_last5_calendar"], 1)

    def test_rule_belongs_to_execution_day_not_previous_close_date(self):
        dates = pd.bdate_range("2023-12-01", "2024-02-16")
        data = pd.DataFrame({"date": dates, "feature_valid": True})
        calendar = pd.DataFrame({"trade_date": dates, "is_open": True})
        prepared, _ = prepare(data, calendar, config_for_test())
        row = prepared.loc[prepared.date == "2024-01-31"].iloc[0]
        self.assertEqual(row.execution_date, pd.Timestamp("2024-02-01"))
        self.assertEqual(row.month_session_ordinal, 1)
        self.assertEqual(row.decision_time, pd.Timestamp("2024-02-01 09:00"))
        targets = rule_targets(prepared)
        where = prepared.index[prepared.date == "2024-01-31"][0]
        self.assertEqual(targets["K1_PRIMARY_MONTH_START3"][where], 1)
        self.assertFalse(prepared.feature_valid.iloc[-1])

    def test_rule_and_prediction_accounts_match_existing_engine(self):
        prices, dividends = synthetic()
        data, _ = factors(prices, dividends)
        config = config_for_test()
        config["evaluation_start"] = str(data.date.iloc[700].date())
        targets = (np.arange(len(data)) % 21 < 3).astype(float)
        prediction = np.sin(np.arange(len(data)) / 31) * 0.003
        for key, values in (("targets", targets), ("prediction", prediction)):
            supplied = {key: values}
            a, _ = simulate_policy(data, dividends, config, config["costs"]["BASE"], "TEST", **supplied)
            b, _ = simulate(data, dividends, config, config["costs"]["BASE"], config["evaluation_start"], "TEST", **supplied)
            for col in ("equity", "cash", "shares", "net_return", "commission", "slippage_cost", "dividend_receivable"):
                np.testing.assert_allclose(a[col], b[col], atol=1e-10, rtol=0)

    def test_no_view_keeps_inventory_without_setting_cash_target(self):
        prices, dividends = synthetic()
        data, _ = factors(prices, dividends)
        config = config_for_test()
        config["evaluation_start"] = str(data.date.iloc[700].date())
        targets = np.ones(len(data))
        targets[700] = np.nan
        ledger, decisions = simulate_policy(data, dividends, config, config["costs"]["BASE"], "TEST", targets=targets)
        self.assertGreater(ledger.shares.iloc[0], 0)
        self.assertEqual(ledger.shares.iloc[1], ledger.shares.iloc[0])
        self.assertEqual(decisions.view.iloc[1], "NO_VIEW")
        self.assertTrue(np.isnan(decisions.reference_weight.iloc[1]))
        self.assertEqual(ledger.filled_quantity.iloc[1], 0)


if __name__ == "__main__":
    unittest.main()
