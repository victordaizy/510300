"""收盘位置入场的计量、因果顺序和真实再入场状态。"""
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from research.close_location_entry_gate_v1 import LocationGate, location_factors
from research.rearmed_cycle_exit_account_v1 import simulate_rearmed_exit
from research.self_performance_entry_account_v1 import simulate_self_performance_exit


def prices():
    return pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=9), "high": 12., "low": 10.,
                         "close": [12., 10., 11., 12., 10., 11., 12., 10., 11.], "volume": [3., 1., 2.] * 3})


def test_weighted_hand_calculation_and_equal_control():
    f = location_factors(prices(), 3)
    assert not f.location_available.iloc[:2].any()
    assert np.isclose(f.volume_location21.iloc[2], (3 - 1) / 6)
    assert f.equal_location21.iloc[2] == 0
    assert LocationGate(f, "volume_location21", "加权")(2, None, 1, 100)["entry_allowed"]
    assert not LocationGate(f, "equal_location21", "等权")(2, None, 1, 100)["entry_allowed"]


def test_price_affine_and_volume_unit_invariance():
    p = prices()
    a = location_factors(p, 3)
    changes = np.arange(len(p)) + 1.
    for c in ["high", "low", "close"]:
        p[c] = p[c] * changes + changes * .7
    p.volume *= 100
    b = location_factors(p, 3)
    for c in ["daily_location", "volume_location21", "equal_location21"]:
        np.testing.assert_allclose(a[c], b[c], atol=1e-12, equal_nan=True)


def test_missing_day_invalidates_full_window_and_preserves_no_view():
    p = prices()
    p.loc[3, "volume"] = np.nan
    f = location_factors(p, 3)
    assert not f.location_available.iloc[3:6].any() and f.location_available.iloc[6]
    check = LocationGate(f, "volume_location21", "加权")(4, None, 1, 100)
    assert not check["entry_allowed"] and check["location_gate_value"] is None
    assert check["location_status"].startswith("NO_VIEW")


def test_flat_range_neutral_but_bad_range_and_zero_total_volume_invalid():
    p = prices()
    p.loc[0, ["high", "low", "close"]] = 11.
    assert location_factors(p, 3).daily_location.iloc[0] == 0
    p.loc[1, "close"] = 15.
    assert not location_factors(p, 3).location_available.iloc[2]
    p = prices()
    p.loc[:2, "volume"] = 0.
    assert not location_factors(p, 3).location_available.iloc[2]


def test_future_rows_cannot_change_previous_factors():
    p = prices()
    original = location_factors(p, 3)
    p.loc[5:, "close"] = 12.
    p.loc[5:, "volume"] = 999.
    assert_frame_equal(original.iloc[:5], location_factors(p, 3).iloc[:5])


def account_args():
    data = pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=9), "open": 10., "close": 10., "previous_close": 10., "dividend": 0.})
    dividends = pd.DataFrame(columns=["record_date", "ex_date", "payment_date", "cash_dividend_per_share"])
    cfg = {"initial_capital": 100000., "lot": 100, "tick": .001, "limit_fraction": .1}
    cost = {"commission": 0., "minimum": 0., "slippage": 0.}
    rule = {"entry": np.ones(9, int), "exit": {1: np.zeros(9, bool)}}
    spec = {"cooldown": 2, "modes": {1: {"loss": None, "trail": None, "take": None, "days": 2}}}
    return [data, dividends, cfg, cost, str(data.date.iloc[1].date()), rule, spec]


def test_gate_defers_but_does_not_rearm_continuous_original_entry_after_sale():
    args = account_args()
    f = pd.DataFrame({"location_available": True, "volume_location21": [-.1] + [.1] * 8})
    l, d, c = simulate_self_performance_exit(*args, entry_gate=LocationGate(f, "volume_location21", "加权"))
    assert d.iloc[0].entry_rearmed and d.iloc[0].requested_quantity == 0
    assert l[l.filled_quantity > 0].date.to_list() == [args[0].date.iloc[2]]
    assert c.exit_date.to_list() == [args[0].date.iloc[4]]
    assert not d.action.str.contains("自身过去一年").any()


def test_all_positive_gate_recovers_original_account_economics():
    args = account_args()
    f = pd.DataFrame({"location_available": True, "volume_location21": [.1] * 9})
    original = simulate_rearmed_exit(*args)
    actual = simulate_self_performance_exit(*args, entry_gate=LocationGate(f, "volume_location21", "加权"))
    assert_frame_equal(original[0], actual[0])
    assert_frame_equal(original[2], actual[2])
