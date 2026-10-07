"""检验最容易把事后上涨错当成可交易信号的边界。"""
import unittest

import numpy as np
import pandas as pd

from research.upward_episode_anatomy_v1 import upward_episodes, features, label_origins


class EpisodeTests(unittest.TestCase):
    def frame(self, values):
        return pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=len(values)),
                             "total_return_index": values, "available": True})

    def test_peak_needs_later_fall_confirmation(self):
        d = self.frame([100., 99., 105., 110., 109., 104.])
        ep = upward_episodes(d).iloc[0]
        self.assertEqual((ep.bottom_idx, ep.confirm_up_idx, ep.peak_idx, ep.confirm_down_idx), (1, 2, 3, 5))
        partial = upward_episodes(d.iloc[:5]).iloc[0]
        self.assertEqual(partial.status, "RIGHT_CENSORED")
        self.assertTrue(pd.isna(partial.confirm_down_idx))

    def test_small_bounce_does_not_create_upward_episode(self):
        d = self.frame([100., 94., 90., 94., 88., 91., 86.])
        self.assertTrue(upward_episodes(d).empty)

    def test_two_upward_episodes_are_not_split_at_small_pullback(self):
        d = self.frame([100., 106., 104., 112., 105., 101., 108., 114., 107.])
        ep = upward_episodes(d)
        self.assertEqual(len(ep), 2)
        self.assertEqual(ep.bottom_idx.tolist(), [0, 5])
        self.assertEqual(ep.peak_idx.tolist(), [3, 7])

    def test_fixed_horizon_excludes_buying_on_ex_date_dividend(self):
        dates = pd.bdate_range("2020-01-01", periods=50)
        p = pd.DataFrame({"date": dates, "open": 10., "close": 10., "high": 10.1,
                          "low": 9.9, "volume": 10000., "symbol": "510300.SH"})
        p.loc[1:, ["open", "close", "high", "low"]] -= 1.
        div = pd.DataFrame({"record_date": [dates[0]], "ex_date": [dates[1]],
                            "payment_date": [dates[4]], "cash_dividend_per_share": [1.]})
        d, _ = features(p, div)
        d["available"] = True
        labels = label_origins(d, div)
        first = labels.iloc[0]
        self.assertEqual(first.dividend_per_share, 0.)
        self.assertAlmostEqual(first.gross_return, 0.)
        self.assertEqual(int(first.end_idx) - int(first.entry_idx) + 1, 20)
        self.assertEqual(int(labels.status.eq("RIGHT_CENSORED").sum()), 20)
        self.assertAlmostEqual(d.return1.iloc[1], 0.)
        self.assertTrue(np.allclose(d.ac, 10.))


if __name__ == "__main__":
    unittest.main()
