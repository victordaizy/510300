"""检验资金字段源时钟与旧分位数支持分离后的必要边界。"""
import unittest

import numpy as np
import pandas as pd

from research.funding_availability_semantics_v1 import source_mask


def frame():
    return pd.DataFrame({"date": pd.to_datetime(["2020-02-04"]),
                         "fund_stat_date": pd.to_datetime(["2020-02-03"]),
                         "available_at": ["2020-02-04T09:30:00+08:00"],
                         "policy_known_at": ["2020-01-01T09:20:00+08:00"],
                         "dr007": [2.0], "rate": [2.4], "gap_pp": [-.4],
                         "K05_q90": [np.nan], "K05_q50": [np.nan], "fund_known": [False]})


class FundingSourceContractTests(unittest.TestCase):
    def test_known_source_does_not_require_old_quantile_signal(self):
        data = frame()
        self.assertTrue(bool(source_mask(data).iloc[0]))
        self.assertFalse(bool(data.fund_known.iloc[0]))

    def test_after_decision_publication_is_unknown(self):
        data = frame()
        data.loc[0, "available_at"] = "2020-02-04T17:00:00+08:00"
        self.assertFalse(bool(source_mask(data).iloc[0]))

    def test_same_day_statistics_are_not_admitted(self):
        data = frame()
        data.loc[0, "fund_stat_date"] = pd.Timestamp("2020-02-04")
        self.assertFalse(bool(source_mask(data).iloc[0]))

    def test_stale_source_and_unknown_value_are_not_zero(self):
        stale = frame()
        stale.loc[0, "fund_stat_date"] = pd.Timestamp("2020-01-23")
        self.assertFalse(bool(source_mask(stale).iloc[0]))
        unknown = frame()
        unknown.loc[0, "dr007"] = np.nan
        self.assertFalse(bool(source_mask(unknown).iloc[0]))

    def test_future_policy_clock_and_future_append_do_not_enter_old_origin(self):
        future_policy = frame()
        future_policy.loc[0, "policy_known_at"] = "2020-02-05T09:20:00+08:00"
        self.assertFalse(bool(source_mask(future_policy).iloc[0]))
        current, future = frame(), frame()
        future.loc[0, "date"] = pd.Timestamp("2021-01-05")
        future.loc[0, "fund_stat_date"] = pd.Timestamp("2021-01-04")
        future.loc[0, "available_at"] = "2021-01-05T09:30:00+08:00"
        joined = pd.concat([current, future], ignore_index=True)
        self.assertEqual(bool(source_mask(current).iloc[0]), bool(source_mask(joined).iloc[0]))


if __name__ == "__main__":
    unittest.main()
