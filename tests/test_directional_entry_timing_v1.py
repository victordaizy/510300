"""多空符号、股息和收盘后才能退出的实质边界。"""
import unittest

import numpy as np
import pandas as pd

from research.directional_entry_timing_v1 import friction_result, evaluate_point, directional_episodes


class DirectionalTests(unittest.TestCase):
    def test_short_dividend_is_cost_not_gain(self):
        long = friction_result(10., 9.5, .2, 1)
        short = friction_result(10., 9.5, .2, -1)
        self.assertAlmostEqual(long["gross_return"], -.03)
        self.assertAlmostEqual(short["gross_return"], .03)
        self.assertLess(short["net_reference_return"], .03)

    def test_intraday_target_does_not_make_close_confirmation(self):
        d = pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=5),
                          "open": [10., 10., 10., 10., 10.2], "close": [10., 10., 10., 10.4, 10.2],
                          "ao": [10., 10., 10., 10., 10.2], "ac": [10., 10., 10., 10.4, 10.2],
                          "high": [10., 11., 11., 11., 10.5], "volume": 10000., "dividend": 0., "cash_shift": 0.})
        div = pd.DataFrame({"record_date": pd.to_datetime([]), "ex_date": pd.to_datetime([]), "cash_dividend_per_share": []})
        sig = {"signal_idx": 0, "direction": 1, "stop_index": 9.8, "target_index": 10.3}
        point = evaluate_point(sig, d, div)
        self.assertEqual(point["exit_reason"], "TARGET_CONFIRMED")
        self.assertEqual(point["decision_idx"], 3)
        self.assertEqual(point["exit_idx"], 4)

    def test_downward_extreme_is_not_confirmed_until_rebound(self):
        d = pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=7),
                          "total_return_index": [100., 104., 98., 92., 95., 98., 97.], "available": True})
        ep = directional_episodes(d)
        down = ep.loc[ep.direction.eq(-1)].iloc[0]
        self.assertEqual((down.start_idx, down.end_idx, down.confirm_idx, down.end_confirm_idx), (1, 3, 2, 5))
        self.assertEqual(ep.iloc[-1].status, "RIGHT_CENSORED")


if __name__ == "__main__":
    unittest.main()
