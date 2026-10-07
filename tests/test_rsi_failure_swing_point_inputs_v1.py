"""RSI确认不得前移；真实末日不强平，价格保护按随后实际开盘执行。"""
import numpy as np
import pandas as pd

from research import rsi_failure_swing_point_inputs_v1 as study


def test_wilder_seed_then_recursive_value():
    prices = [10, 11, 10, 12, 11]
    rsi = study.wilder_rsi(prices, 3)
    assert np.isnan(rsi[:3]).all()
    assert np.isclose(rsi[3], 75.)
    assert np.isclose(rsi[4], 100.-100./(1.+(2./3.)/(5./9.)))


def test_swing_waits_for_break_after_pullback_and_future_does_not_rewrite():
    dates = pd.bdate_range("2024-01-02", periods=9)
    rsi = [28, 25, 32, 45, 40, 35, 44, 46, 49]
    lows = [10, 9, 9.5, 10, 9.8, 9.6, 9.7, 10, 10.2]
    full = study.detect(dates, rsi, lows)
    assert np.flatnonzero(full.entry_signal).tolist() == [7]
    assert full.structural_stop_index.iloc[7] == 9.6
    short = study.detect(dates[:7], rsi[:7], lows[:7])
    pd.testing.assert_frame_equal(short, full.iloc[:7])


def test_pullback_touching_thirty_does_not_confirm_old_setup():
    dates = pd.bdate_range("2024-01-02", periods=7)
    frame = study.detect(dates, [25, 35, 45, 30, 35, 44, 46], np.ones(7)*10)
    assert not frame.entry_signal.any()


def example():
    dates = pd.bdate_range("2024-01-02", periods=5)
    data = pd.DataFrame({"date": dates, "open": [10., 10., 9.1, 9.2, 9.3], "close": [10., 9.4, 9.1, 9.2, 9.3],
                         "volume": 1000., "dividend": 0., "cash_shift": 0.})
    data["ao"], data["ac"] = data.open, data.close
    signals = pd.DataFrame({"date": dates, "entry_signal": [True, False, False, False, False], "rsi14": 45., "structural_stop_index": 9.5})
    dividends = pd.DataFrame(columns=["record_date", "ex_date", "cash_dividend_per_share"])
    return data, dividends, signals


def test_stop_is_next_open_and_no_forced_terminal_exit():
    data, div, signals = example()
    points, _ = study.observe(data, div, signals, data.date.iloc[1])
    assert points.entry_date.iloc[0] == data.date.iloc[1]
    assert points.exit_origin.iloc[0] == data.date.iloc[1]
    assert points.exit_date.iloc[0] == data.date.iloc[2]
    assert points.exit_raw.iloc[0] == 9.1
    assert points.point_net_return.iloc[0] < -.09
    short, _ = study.observe(data.iloc[:2], div, signals.iloc[:2], data.date.iloc[1])
    assert short.status.iloc[0] == "RIGHT_CENSORED" and pd.isna(short.point_net_return.iloc[0])


def test_gap_below_structure_cancels_entry_once():
    data, div, signals = example()
    data.loc[1, ["open", "ao"]] = 9.4
    points, events = study.observe(data, div, signals, data.date.iloc[1])
    assert points.empty
    assert events.event.to_list() == ["CANCEL_ENTRY"]
