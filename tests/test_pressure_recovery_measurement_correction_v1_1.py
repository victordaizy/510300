"""用明确边界的合成消息和盘口，验证单位修正与价格带归因。"""

import numpy as np
import pandas as pd
import pytest

from research.pressure_recovery_measurement_correction_v1_1 import depth_decomposition, trailing_values


@pytest.mark.parametrize("message_unit,end_unit", [("us", "us"), ("ns", "ns"), ("us", "ns"), ("ns", "us")])
def test_exact_five_minutes_excludes_left_and_future_in_all_datetime_units(message_unit, end_unit):
    times = pd.DatetimeIndex(["2026-07-09 09:00:00", "2026-07-09 09:55:00", "2026-07-09 09:55:00.001", "2026-07-09 10:00:00", "2026-07-09 10:00:00.001"], tz="Asia/Shanghai").as_unit(message_unit)
    messages = pd.DataFrame({"_at": times})
    ends = pd.DatetimeIndex(["2026-07-09 10:00:00"], tz="Asia/Shanghai").as_unit(end_unit)
    result = trailing_values(messages, ends, [100000, 10000, 100, 200, 1000000])
    assert result.tolist() == [300]


def test_signed_weights_and_empty_window():
    messages = pd.DataFrame({"_at": pd.DatetimeIndex(["2026-07-09 10:00:00", "2026-07-09 10:01:00"], tz="Asia/Shanghai")})
    ends = pd.DatetimeIndex(["2026-07-09 09:59:00", "2026-07-09 10:01:00"], tz="Asia/Shanghai")
    assert trailing_values(messages, ends, [100, -200]).tolist() == [0, -100]


def test_unsorted_messages_fail_instead_of_silently_miscounting():
    times = pd.DatetimeIndex(["2026-07-09 10:01:00", "2026-07-09 10:00:00"], tz="Asia/Shanghai")
    with pytest.raises(ValueError, match="排序"):
        trailing_values(pd.DataFrame({"_at": times}), times, [100, 200])


def book(best: int, size: float = 1000) -> pd.Series:
    result = {"ask_px1": best + 10}
    for i in range(1, 11):
        result[f"bid_px{i}"] = best - (i - 1) * 10
        result[f"bid_vol{i}"] = size
    return pd.Series(result)


def test_unchanged_prices_depth_increase_is_same_absolute_band_change():
    result = depth_decomposition(book(40000), book(40000, 2000))
    assert result["moving_depth_ratio"] == 2
    assert result["initial_band_depth_ratio"] == 2
    assert result["band_shift_net_component_cny"] == 0
    assert np.isclose(result["same_absolute_band_net_change_cny"], result["moving_depth_change_cny"])


def test_falling_price_can_double_moving_depth_without_any_original_band_bid():
    result = depth_decomposition(book(40000), book(39900, 2000))
    assert result["moving_depth_ratio"] > 1
    assert result["initial_band_depth_ratio"] == 0
    assert result["band_overlap_fraction"] == 0
    assert result["same_absolute_band_net_change_cny"] == 0
    assert np.isclose(result["band_shift_net_component_cny"], result["moving_depth_change_cny"])


def test_original_price_band_below_visible_ten_levels_is_unknown_not_zero():
    result = depth_decomposition(book(40000), book(40200))
    assert not result["initial_band_at_endpoint_fully_visible"]
    assert result["initial_band_endpoint_depth_cny"] is None
    assert result["initial_band_depth_ratio"] is None
