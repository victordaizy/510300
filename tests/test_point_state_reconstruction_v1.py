"""检验组合未知边界、下一开盘及受阻后最新目标，避免伪造可复现点位。"""
import numpy as np
import pandas as pd

from research.point_state_reconstruction_v1 import compose, confirmed_direction, reconstruct_market, replay


def fixture(targets, opens=None):
    n = len(targets) + 1
    dates = pd.bdate_range("2025-03-03", periods=n)
    data = pd.DataFrame({"date": dates, "open": opens if opens is not None else np.full(n, 4.),
                         "close": np.full(n, 4.), "volume": np.full(n, 100000), "dividend": np.zeros(n)})
    signals = pd.DataFrame({"origin": dates[:-1], "execution_date": dates[1:], "target": targets,
                            "core_target": targets, "aux_target": np.zeros(n-1),
                            "effective_auxiliary": np.zeros(n-1), "gate": np.ones(n-1)})
    div = pd.DataFrame({"record_date": pd.to_datetime([]), "ex_date": pd.to_datetime([]), "cash_dividend_per_share": []})
    return data, signals, div


def test_known_zero_gate_does_not_discard_core_and_unknown_is_not_zero():
    target, effective = compose([.3, 0, .3, .3], [.5, .5, .5, np.nan], [0, 0, np.nan, 1])
    assert np.allclose(target[:2], [.3, 0])
    assert np.isnan(target[2:]).all()
    assert effective[0] == 0


def test_hold_positive_weights_and_unknown_until_known_zero_next_open():
    data, signals, div = fixture([.2, .9, np.nan, 0, 0, 0])
    points, events = replay(data, div, signals, data.date.iloc[1])
    assert len(points) == 1 and points.iloc[0].status == "COMPLETE"
    assert points.iloc[0].entry_date == data.date.iloc[1]
    assert points.iloc[0].exit_date == data.date.iloc[4]
    assert list(events.event).count("ENTER") == 1
    assert "UNKNOWN_KEEP_POSITION" in list(events.event)


def test_blocked_entry_is_cancelled_by_latest_zero_target():
    data, signals, div = fixture([1., 0., 0., 0.], [4., 4.4, 4., 4., 4.])
    points, events = replay(data, div, signals, data.date.iloc[1])
    assert points.empty
    assert events.iloc[0].event == "ENTRY_BLOCKED_DIRECTIONAL_OPEN_LIMIT"


def test_blocked_exit_rechecks_latest_target_without_stale_forced_exit():
    data, signals, div = fixture([1., 0., 1., 0., 0., 0.], [4., 4., 3.6, 4., 4., 4., 4.])
    points, events = replay(data, div, signals, data.date.iloc[1])
    assert points.iloc[0].exit_date == data.date.iloc[4]
    assert "EXIT_BLOCKED_DIRECTIONAL_OPEN_LIMIT" in list(events.event)


def test_terminal_mark_is_censored_not_a_successful_closed_trade():
    data, signals, div = fixture([1., 1., 1., 1.])
    points, _ = replay(data, div, signals, data.date.iloc[1])
    assert points.iloc[0].status == "RIGHT_CENSORED"
    assert np.isnan(points.iloc[0].point_net_return)
    assert pd.isna(points.iloc[0].exit_date)


def test_separate_two_day_exits_and_strict_run_entry_boundary():
    direction = confirmed_direction([-1., -1.2, -1.3, 0., -.5, 0., 0.], [1., 1., 1., 1., -1., 1., 1.], "RUNS")
    assert list(direction) == [0., 0., 1., 1., 1., 1., 0.]


def test_market_reconstruction_prefix_does_not_use_future_prices():
    rng = np.random.default_rng(20261001)
    dates = pd.bdate_range("2020-01-02", periods=260)
    price = 4 * np.cumprod(1 + rng.normal(.0003, .008, len(dates)))
    data = pd.DataFrame({"date": dates, "open": price, "close": price, "volume": np.full(len(dates), 100000)})
    dividends = pd.DataFrame({"record_date": pd.to_datetime([]), "ex_date": pd.to_datetime([]), "cash_dividend_per_share": []})
    full = reconstruct_market(data, dividends)
    prefix = reconstruct_market(data.iloc[:190], dividends)
    pd.testing.assert_frame_equal(full.iloc[:190], prefix)
