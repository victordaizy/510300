"""验证整日收盘跨区间而非普通收复、整数除息、未知及未来不可改写。"""
import numpy as np
import pandas as pd

from research.full_range_reversal_inputs_v1 import reversal_states


def fixture():
    return pd.DataFrame({"date": pd.bdate_range("2020-01-02", periods=5),
                         "open": [10., 10., 10.02, 10., 10.],
                         "high": [10.01, 10.05, 10.06, 10.02, 10.02],
                         "low": [9.99, 9.98, 9.97, 9.99, 9.99],
                         "close": [10., 10.04, 9.979, 10., 10.], "dividend": 0.})


def test_full_range_close_requires_both_strict_endpoints_not_low_reclaim():
    data = fixture()
    assert reversal_states(data).bullish_range_reversal.iloc[1]
    for key, boundary in (("low", 9.99), ("close", 10.01)):
        changed = data.copy()
        changed.loc[1, key] = boundary
        assert not reversal_states(changed).bullish_range_reversal.iloc[1]
    changed = data.copy()
    changed.loc[1, "close"] = 10.
    assert not reversal_states(changed).bullish_range_reversal.iloc[1]


def test_symmetric_opposing_event_and_unknown_pair_are_preserved():
    data = fixture()
    assert reversal_states(data).bearish_range_reversal.iloc[2]
    data.loc[1, "high"] = np.nan
    states = reversal_states(data)
    assert not states.current_quote_known.iloc[1]
    assert not states.bearish_range_reversal.iloc[2]
    data.loc[3, "dividend"] = np.nan
    states = reversal_states(data)
    assert not states.current_quote_known.iloc[3:].any()
    assert not states.bullish_range_reversal.iloc[3:].any()


def test_ex_dividend_equal_high_is_not_strict_crossing():
    data = fixture().iloc[:2].copy()
    data.loc[1, ["open", "high", "low", "close", "dividend"]] = [9.967, 9.98, 9.946, 9.977, .033]
    states = reversal_states(data)
    assert states.known_cash_close_ticks.iloc[1] == states.known_cash_high_ticks.iloc[0] == 10010
    assert not states.bullish_range_reversal.iloc[1]
    data.loc[1, "close"] += .001
    assert reversal_states(data).bullish_range_reversal.iloc[1]


def test_all_prefixes_and_future_prices_dividends_labels_cannot_rewrite_past():
    data = fixture()
    full = reversal_states(data)
    for n in range(1, len(data)+1):
        pd.testing.assert_frame_equal(reversal_states(data.iloc[:n].reset_index(drop=True)),
                                      full.iloc[:n].reset_index(drop=True), check_exact=True)
    changed = data.copy()
    changed.loc[3:, ["open", "high", "low", "close"]] *= 1.1
    changed.loc[3, "dividend"] = .023
    changed["future_return"] = 12345.
    pd.testing.assert_frame_equal(reversal_states(changed).iloc[:3], full.iloc[:3], check_exact=True)
