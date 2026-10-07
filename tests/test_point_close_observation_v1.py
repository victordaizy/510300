"""点位最后真实交易日照常观察，未知状态和未来扩展不会伪造退出。"""
import numpy as np
import pandas as pd

from research import point_close_observation_v1 as study


def sample(targets):
    dates = pd.bdate_range("2024-03-04", periods=len(targets)+1)
    data = pd.DataFrame({"date": dates, "open": 10., "close": 10., "volume": 1000000., "dividend": 0.})
    signals = pd.DataFrame({"origin": dates[:-1], "execution_date": dates[1:], "target": targets,
                            "core_target": targets, "aux_target": 0., "effective_auxiliary": 0., "gate": 1.})
    div = pd.DataFrame(columns=["record_date", "ex_date", "cash_dividend_per_share"])
    return data, div, signals


def test_last_real_day_exit_executes_and_is_complete():
    data, div, signals = sample([1., 1., 0.])
    points, events = study.observe_points(data, div, signals, data.date.iloc[1])
    assert points.status.to_list() == ["COMPLETE"]
    assert points.exit_date.iloc[0] == data.date.iloc[-1]
    assert events.event.to_list() == ["ENTER", "EXIT"]


def test_last_real_day_entry_remains_censored():
    data, div, signals = sample([0., 0., 1.])
    points, _ = study.observe_points(data, div, signals, data.date.iloc[1])
    assert points.status.to_list() == ["RIGHT_CENSORED"]
    assert points.entry_date.iloc[0] == data.date.iloc[-1]
    assert points.point_net_return.isna().all()
    assert points.exit_date.isna().all()


def test_unknown_last_signal_keeps_holding_without_profit_label():
    data, div, signals = sample([1., 1., np.nan])
    points, events = study.observe_points(data, div, signals, data.date.iloc[1])
    assert points.status.to_list() == ["RIGHT_CENSORED"]
    assert events.event.iloc[-1] == "UNKNOWN_KEEP_POSITION"


def test_future_exit_cannot_change_previous_events_or_complete_early():
    data, div, signals = sample([1., 1., 1., 0.])
    short_points, short_events = study.observe_points(data.iloc[:4], div, signals.iloc[:3], data.date.iloc[1])
    full_points, full_events = study.observe_points(data, div, signals, data.date.iloc[1])
    assert short_points.status.to_list() == ["RIGHT_CENSORED"]
    assert full_points.status.to_list() == ["COMPLETE"]
    pd.testing.assert_frame_equal(short_events, full_events.loc[full_events.date.le(data.date.iloc[3])].reset_index(drop=True))
