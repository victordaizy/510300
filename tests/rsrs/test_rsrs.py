"""RSRS 最低限度单元测试；运行: python -m unittest tests.rsrs.test_rsrs -v"""
import unittest
import numpy as np
import pandas as pd

from research.rsrs.rsrs import calculate_rsrs, rsrs_target_weight


class TestRSRS(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        base = 3 + np.cumsum(rng.normal(0, 0.02, 800))
        low = base - rng.uniform(0.01, 0.04, 800)
        high = base + rng.uniform(0.01, 0.04, 800)
        self.bars = pd.DataFrame({"low": low, "high": high},
                                 index=pd.bdate_range("2020-01-01", periods=800))

    def test_warmup_and_bounds(self):
        result = calculate_rsrs(self.bars)
        self.assertTrue(result["right_skew_rsrs"].iloc[:616].isna().all())
        r2 = result["r_squared"].dropna()
        self.assertTrue(r2.between(0, 1).all())
        self.assertTrue(result["right_skew_rsrs"].iloc[-1] == result["right_skew_rsrs"].iloc[-1])

    def test_no_future_leakage(self):
        a = calculate_rsrs(self.bars).iloc[:700]
        altered = self.bars.copy()
        altered.iloc[700:, :] *= 1.5
        b = calculate_rsrs(altered).iloc[:700]
        pd.testing.assert_frame_equal(a, b)

    def test_hysteresis(self):
        x = pd.Series([np.nan, 0.8, 0.1, np.nan, -0.8, 0.9])
        self.assertEqual(rsrs_target_weight(x).tolist(), [0, 1, 1, 1, 0, 1])


if __name__ == "__main__":
    unittest.main()
