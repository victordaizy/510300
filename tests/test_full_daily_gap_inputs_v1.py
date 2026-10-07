"""验证完整缺口定义、除息整数坐标与过去状态不可被未来改写。"""
import numpy as np
import pandas as pd

from research.full_daily_gap_inputs_v1 import gap_states


def fixture(n=24):
    close = np.array([10.] * 15 + [10.02, 10.021, 10.021, 10.05, 10.052, 10.052] + [10.021] * 3)[:n]
    return pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=n), "open": close,
                         "high": close+.002, "low": close-.002, "close": close, "dividend": 0.})


def test_full_range_gap_is_not_just_opening_above_prior_high():
    data = fixture(3)
    data.loc[0, ["open", "high", "low", "close"]] = [10., 10.1, 9.9, 10.]
    data.loc[1, ["open", "high", "low", "close"]] = [10.2, 10.3, 10., 10.25]
    data.loc[2, ["open", "high", "low", "close"]] = [10.45, 10.5, 10.4, 10.45]
    states = gap_states(data)
    assert not states.gap_event.iloc[1]
    assert states.gap_event.iloc[2]
    assert states.gap_lower_ticks.iloc[2] == 10300
    assert states.gap_lower_reference_index.iloc[2] == 1


def test_ex_dividend_economic_touch_is_exactly_not_a_gap():
    data = fixture(2)
    data.loc[0, ["open", "high", "low", "close"]] = [10., 10.001, 9.998, 10.]
    data.loc[1, ["open", "high", "low", "close", "dividend"]] = [9.972, 9.98, 9.968, 9.975, .033]
    states = gap_states(data)
    assert states.known_cash_low_ticks.iloc[1] == states.known_cash_high_ticks.iloc[0] == 10001
    assert not states.gap_event.iloc[1]
    data.loc[1, "low"] += .001
    assert gap_states(data).gap_event.iloc[1]


def test_missing_quote_does_not_backfill_and_unknown_dividend_poison_is_preserved():
    data = fixture()
    data.loc[14, "high"] = np.nan
    states = gap_states(data)
    assert not states.gap_event.iloc[15]
    assert states.gap_status.iloc[15] == "NO_VIEW_CURRENT_OR_PRIOR_QUOTE"
    data.loc[17, "dividend"] = np.nan
    states = gap_states(data)
    assert not states.current_quote_known.iloc[17:].any()
    assert not states.gap_event.iloc[17:].any()


def test_every_prefix_and_future_prices_dividends_labels_leave_past_unchanged():
    data = fixture()
    full = gap_states(data)
    for n in range(1, len(data)+1):
        pd.testing.assert_frame_equal(gap_states(data.iloc[:n].reset_index(drop=True)), full.iloc[:n].reset_index(drop=True), check_exact=True)
    altered = data.copy()
    altered.loc[19:, ["open", "high", "low", "close"]] *= 1.1
    altered.loc[19, "dividend"] = .023
    altered["future_net_return"] = np.arange(len(altered))*100.
    pd.testing.assert_frame_equal(gap_states(altered).iloc[:19], full.iloc[:19], check_exact=True)
