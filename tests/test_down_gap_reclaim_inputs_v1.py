"""验证完整缺口、严格收复、最新锚、除息和未来不可改写。"""
import numpy as np
import pandas as pd

from research.down_gap_reclaim_inputs_v1 import reclaim_states, anchor_paths


def fixture():
    return pd.DataFrame({"date": pd.bdate_range("2020-01-02", periods=6),
                         "open": [10.05, 9.8, 9.85, 10.01, 10.02, 10.02],
                         "high": [10.1, 9.9, 10., 10.2, 10.1, 10.1],
                         "low": [10., 9.7, 9.8, 9.9, 9.9, 9.9],
                         "close": [10.05, 9.8, 10., 10.01, 10.02, 10.02], "dividend": 0.})


def test_whole_down_gap_and_first_strict_upper_close_consumes_once():
    data = fixture()
    states = reclaim_states(data)
    assert states.new_down_gap.iloc[1]
    assert not states.reclaim_event.iloc[1:3].any()
    assert states.reclaim_event.iloc[3]
    assert states.reclaim_floor_ticks.iloc[3] == 9900
    assert states.reclaim_upper_ticks.iloc[3] == 10000
    assert not states.reclaim_event.iloc[4:].any()
    for high in (10., 10.01):
        changed = data.copy()
        changed.loc[1, "high"] = high
        assert not reclaim_states(changed).new_down_gap.iloc[1]


def test_new_down_gap_replaces_unreclaimed_old_anchor_no_old_anchor_selection():
    data = fixture()
    data.loc[2, ["open", "high", "low", "close"]] = [9.55, 9.6, 9.4, 9.5]
    data.loc[3, ["open", "high", "low", "close"]] = [9.65, 9.8, 9.6, 9.701]
    states = reclaim_states(data)
    assert states.new_down_gap.iloc[2] and states.reclaim_event.iloc[3]
    assert states.reclaim_reference_index.iloc[3] == 2
    assert states.reclaim_floor_ticks.iloc[3] == 9600
    paths = anchor_paths(states)
    assert paths.status.tolist() == ["REPLACED_BEFORE_RECLAIM", "RECLAIMED"]
    assert paths.terminal_index.tolist() == [2, 3]


def test_ex_dividend_equal_boundary_and_unknown_anchor_state_are_preserved():
    data = fixture()
    data.loc[2, ["open", "high", "low", "close", "dividend"]] = [9.817, 9.967, 9.767, 9.967, .033]
    states = reclaim_states(data)
    assert states.known_cash_close_ticks.iloc[2] == 10000
    assert not states.reclaim_event.iloc[2]
    data.loc[2, "close"] = 9.968
    data.loc[2, "high"] = 9.968
    assert reclaim_states(data).reclaim_event.iloc[2]
    data = fixture()
    data.loc[3, "high"] = np.nan
    states = reclaim_states(data)
    assert not states.reclaim_event.iloc[3] and states.anchor_active_after_origin.iloc[3]
    assert states.reclaim_event.iloc[4]
    data.loc[2, "dividend"] = np.nan
    states = reclaim_states(data)
    assert not states.current_quote_known.iloc[2:].any()
    assert not states.reclaim_event.any()
    assert anchor_paths(states).status.iloc[0] == "RIGHT_CENSORED"


def test_all_prefixes_future_prices_dividends_and_labels_cannot_rewrite_past():
    data = fixture()
    full = reclaim_states(data)
    for n in range(1, len(data)+1):
        pd.testing.assert_frame_equal(reclaim_states(data.iloc[:n].reset_index(drop=True)),
                                      full.iloc[:n].reset_index(drop=True), check_exact=True)
    changed = data.copy()
    changed.loc[4:, ["open", "high", "low", "close"]] *= 1.1
    changed.loc[4, "dividend"] = .023
    changed["future_return"] = 999.
    pd.testing.assert_frame_equal(reclaim_states(changed).iloc[:4], full.iloc[:4], check_exact=True)
