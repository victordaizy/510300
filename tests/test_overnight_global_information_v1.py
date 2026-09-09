"""盘后源时钟、夏令时以及无有效判断的保留语义。"""
import unittest
import numpy as np
import pandas as pd

from research.overnight_global_information_v1 import available_times, align_one, binary_targets


class OvernightGlobalInformationTests(unittest.TestCase):
    def test_us_daylight_saving_clock(self):
        times = available_times(pd.Series(pd.to_datetime(["2024-03-08", "2024-03-11"])), "US")
        self.assertEqual(times[0], pd.Timestamp("2024-03-08 22:00"))
        self.assertEqual(times[1], pd.Timestamp("2024-03-11 21:00"))

    def test_holiday_alignment_does_not_use_upcoming_us_close(self):
        chinese = pd.DataFrame({"date": pd.to_datetime(["2024-04-03", "2024-04-08", "2024-04-09"])})
        source = pd.DataFrame({"date": pd.to_datetime(["2024-04-03", "2024-04-04", "2024-04-05", "2024-04-08"]),
                               "close": [100.0, 101.0, 102.0, 103.0]})
        aligned, _ = align_one(chinese, source, "GSPC", "US")
        self.assertEqual(aligned.source_date.iloc[0], pd.Timestamp("2024-04-05"))
        self.assertEqual(aligned.source_date.iloc[1], pd.Timestamp("2024-04-08"))
        self.assertTrue((aligned.available_at.dropna() <= aligned.decision_time.dropna()).all())
        changed = source.copy()
        changed.loc[3, "close"] = 500.0
        altered, _ = align_one(chinese, changed, "GSPC", "US")
        self.assertEqual(aligned.GSPC_mom1.iloc[0], altered.GSPC_mom1.iloc[0])

    def test_no_view_keeps_previous_target(self):
        probability = np.array([np.nan, 0.6, np.nan, 0.5, np.nan, 0.7])
        np.testing.assert_array_equal(binary_targets(probability), [0, 1, 1, 0, 0, 1])


if __name__ == "__main__":
    unittest.main()
