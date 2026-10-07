"""验证确认延迟、严格顺序、缺失隔离、分红与份额单位，不使用收益标签。"""
import numpy as np
import pandas as pd

from research.volume_lead_price_confirm_inputs_v1 import volume_lead_states


def sample():
    close = np.array([10, 11, 13, 12, 11, 10, 11, 12, 12, 14, 13, 12], dtype=float)
    return pd.DataFrame({"date": pd.date_range("2020-01-01", periods=len(close)),
                         "open": close, "high": close+.1, "low": close-.1, "close": close,
                         "dividend": 0., "volume": [10]*6+[100, 100]+[10]*4})


def test_reference_is_known_two_days_later_and_lead_precedes_confirmation():
    states, paths, _ = volume_lead_states(sample())
    assert not states.new_reference_high.iloc[:4].any()
    assert states.reference_center_index.iloc[4] == 2
    assert states.reference_confirmation_index.iloc[4] == 4
    assert states.volume_lead_event[states.volume_lead_event].index.tolist() == [7]
    assert states.price_after_volume_event[states.price_after_volume_event].index.tolist() == [9]
    assert states.fixed_low_ticks.iloc[9] == 10000
    assert paths.iloc[0].status == "CONFIRMED_PRICE_AFTER_VOLUME"


def test_equality_does_not_confirm_and_consumed_anchor_never_retries():
    data = sample()
    data.loc[8:11, ["open", "close"]] = np.repeat(np.array([13., 14., 13., 14.])[:, None], 2, axis=1)
    data["high"], data["low"] = data.close+.1, data.close-.1
    states, _, _ = volume_lead_states(data)
    assert not states.price_after_volume_event.iloc[8]
    assert states.price_after_volume_event.iloc[9]
    assert not states.price_after_volume_event.iloc[10:].any()


def test_first_lead_loss_consumes_anchor_without_rescue():
    data = sample()
    data.loc[8, ["open", "close"]] = 11.
    data.loc[8, ["high", "low", "volume"]] = [11.1, 10.9, 500.]
    data.loc[9, "volume"] = 1000.
    states, paths, _ = volume_lead_states(data)
    assert paths.iloc[0].status == "VOLUME_LEAD_LOST_BEFORE_CONFIRM"
    assert paths.iloc[0].terminal_index == 8
    assert not states.price_after_volume_event.any()


def test_missing_volume_is_absorbing_and_zero_volume_is_known():
    data = sample()
    data.loc[8, "volume"] = 0.
    states, _, _ = volume_lead_states(data)
    assert states.cumulative_volume_known.all()
    assert states.signed_volume.iloc[8] == 0
    data.loc[8, "volume"] = np.nan
    states, paths, _ = volume_lead_states(data)
    assert not states.cumulative_volume_known.iloc[8:].any()
    assert states.cumulative_signed_volume.iloc[8:].isna().all()
    assert paths.iloc[0].status == "UNKNOWN_INPUT_NO_BRIDGE"
    assert not states.price_after_volume_event.any()


def test_cash_dividend_direction_and_volume_units_preserve_events():
    base = sample()
    original, _, _ = volume_lead_states(base)
    base.loc[7, "dividend"] = .25
    base.loc[7:, ["open", "high", "low", "close"]] -= .25
    base["volume"] *= 10
    transformed, _, _ = volume_lead_states(base)
    for field in ["volume_lead_event", "price_after_volume_event", "reference_high_ticks", "fixed_low_ticks"]:
        pd.testing.assert_series_equal(original[field], transformed[field])
    pd.testing.assert_series_equal(original.cumulative_signed_volume*10, transformed.cumulative_signed_volume)


def test_future_prices_and_volume_do_not_change_prior_states():
    data = sample()
    states, _, _ = volume_lead_states(data)
    for end in [5, 8, 10]:
        prefix, _, _ = volume_lead_states(data.iloc[:end].copy())
        pd.testing.assert_frame_equal(prefix, states.iloc[:end].reset_index(drop=True), check_exact=True)
    data.loc[9:, "volume"] *= 100
    data.loc[9:, ["open", "high", "low", "close"]] += 100
    altered, _, _ = volume_lead_states(data)
    pd.testing.assert_frame_equal(altered.iloc[:9], states.iloc[:9], check_exact=True)
