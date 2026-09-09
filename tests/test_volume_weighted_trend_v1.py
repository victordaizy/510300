"""核对成交量权重、单位缩放、缺失窗口和未来信息边界。"""
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from research.volume_weighted_trend_v1 import build_factors


def data(n=100):
    t = np.arange(n)
    return pd.DataFrame({"date": pd.bdate_range("2010-01-01", periods=n), "wealth": 10. + .03 * t + .2 * np.sin(t / 5),
                         "volume": 100. + 10 * (t % 7)})


def test_weighted_mean_has_known_value_and_uniform_volume_matches_simple_mean():
    frame = data(5)
    frame["wealth"] = [1., 2., 3., 4., 5.]
    frame["volume"] = [1., 2., 3., 4., 5.]
    f = build_factors(frame, windows=(3,))
    np.testing.assert_allclose(f.VWMA_3.iloc[2], 14. / 6)
    frame["volume"] = 100.
    f = build_factors(frame, windows=(3,))
    np.testing.assert_allclose(f.VWMA_3, f.SMA_3, equal_nan=True)


def test_fixed_volume_unit_rescaling_cannot_change_weighted_trend():
    frame = data()
    original = build_factors(frame)
    frame["volume"] *= 100.
    altered = build_factors(frame)
    for window in [20, 60]:
        np.testing.assert_allclose(original[f"VWMA_{window}"], altered[f"VWMA_{window}"], equal_nan=True)


def test_missing_or_negative_volume_requires_contiguous_valid_window():
    frame = data(20)
    frame.loc[5, "volume"] = np.nan
    frame.loc[12, "volume"] = -1
    f = build_factors(frame, windows=(3,))
    assert f.VWMA_3.iloc[5:8].isna().all()
    assert f.VWMA_3.iloc[12:15].isna().all()
    assert np.isfinite(f.VWMA_3.iloc[8]) and np.isfinite(f.VWMA_3.iloc[15])
    assert f.SMA_3.iloc[2:].notna().all()


def test_zero_total_volume_has_no_factor_and_future_rows_cannot_change_past():
    frame = data()
    frame.loc[:20, "volume"] = 0.
    assert build_factors(frame).VWMA_20.iloc[:21].isna().all()
    prefix = build_factors(frame.iloc[:80])
    frame.loc[80:, "wealth"] *= 2
    frame.loc[80:, "volume"] *= 8
    assert_frame_equal(prefix, build_factors(frame).iloc[:80])
