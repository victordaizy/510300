"""验证新指标的分红尺度、完整窗口、门槛、因果时钟和实际退出。"""
import numpy as np
import pandas as pd
from research.synthetic_fear_recovery_inputs_v1 import factor_frame, trading_rule
from research.simple_price_entry_exit_v1 import simulate_policy


def fixture():
    n = 65
    data = pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=n), "wealth": 1., "open": 10., "close": 10.,
        "low": 9.9, "previous_close": 10., "dividend": 0.})
    data.loc[45, "low"] = 9.
    data.loc[46:, ["wealth", "close"]] = [1.001, 10.01]
    data.loc[46:, "low"] = 9.99
    data.loc[47:, "previous_close"] = 10.01
    return data


def test_formula_uses_22_complete_closes_and_known_dividend_low_scale():
    d = fixture()
    d.loc[44, ["low", "close", "dividend"]] = [8.9, 9., 1.]
    f = factor_frame(d)
    assert f.fear22.iloc[:21].isna().all()
    assert np.isclose(f.fear22.iloc[21], 1.)
    assert np.isclose(f.wealth_low.iloc[44], .99) and np.isclose(f.fear22.iloc[44], 1.)
    assert np.isclose(f.fear22.iloc[45], 10.)


def test_complete_band_and_previous_band_are_required_before_entry():
    f = factor_frame(fixture())
    assert f.fear_upper20.iloc[:40].isna().all()
    assert not f.factor_valid.iloc[40] and f.factor_valid.iloc[41]
    assert trading_rule(f)["entry"].sum() == 1
    assert trading_rule(f)["entry"][46] == 1


def test_previous_above_is_strict_current_boundary_allowed_and_price_must_rise():
    f = pd.DataFrame({"factor_valid": True, "fear22": [3., 5., 2., 1.], "fear_upper20": [3., 3., 2., 2.],
        "wealth": [1., 1.01, 1.02, 1.01], "price_mean20": 1.})
    np.testing.assert_array_equal(trading_rule(f)["entry"], [0, 0, 1, 0])
    f.loc[2, "wealth"] = 1.
    assert trading_rule(f)["entry"][2] == 0


def test_future_extremes_cannot_change_past_factors_or_signal():
    data = fixture()
    original = factor_frame(data)
    data.loc[50:, ["wealth", "low", "close"]] *= 10
    changed = factor_frame(data)
    pd.testing.assert_frame_equal(original.iloc[:50], changed.iloc[:50])
    pd.testing.assert_frame_equal(original.iloc[:50], factor_frame(data.iloc[:50]))


def test_missing_low_blocks_new_entry_but_known_price_exit_remains():
    data = fixture()
    data.loc[46, "low"] = np.nan
    f = factor_frame(data)
    assert not f.factor_valid.iloc[46] and trading_rule(f)["entry"][46] == 0
    assert trading_rule(f)["exit"][1][46]
    assert f.fear_upper20.iloc[46:65].isna().all()


def test_signal_is_filled_next_open_and_price_exit_waits_until_next_open():
    data = fixture()
    f = factor_frame(data)
    cfg = {"initial_capital": 200000., "lot": 100, "tick": .001, "limit_fraction": .1}
    cost = {"commission": .0002, "minimum": 5., "slippage": .0005}
    div = pd.DataFrame(columns=["record_date", "ex_date", "payment_date", "cash_dividend_per_share"])
    spec = {"cooldown": 2, "modes": {1: {"loss": .04, "take": .06, "trail": None, "days": 10}}}
    ledger, decisions, cycles = simulate_policy(data, div, cfg, cost, str(data.date.iloc[43].date()), trading_rule(f), spec)
    assert ledger.loc[ledger.filled_quantity.gt(0), "date"].tolist() == [data.date.iloc[47]]
    assert ledger.loc[ledger.filled_quantity.lt(0), "date"].tolist() == [data.date.iloc[48]]
    assert len(cycles) == 1 and ledger.accounting_error.abs().max() < 1e-6
    assert ledger.commission.sum() > 0 and ledger.slippage_cost.sum() > 0 and ledger.shares.iloc[-1] == 0
