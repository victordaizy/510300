"""检验价格曲折度、新状态时钟及两模式真实退出。"""
import numpy as np
import pandas as pd
from research.choppiness_state_entry_inputs_v1 import factor_frame, classify_choppiness, rules, attach_factor_context
from research.simple_price_entry_exit_v1 import simulate_policy, specifications


def market(n=80):
    close = 10.+np.arange(n)*.02
    return pd.DataFrame({"date": pd.bdate_range("2020-01-02", periods=n), "open": close, "close": close, "high": close+.005, "low": close-.005,
                         "previous_close": np.r_[close[0], close[:-1]], "wealth": close/10., "dividend": 0., "sma120": .1, "z20": 1., "feature_valid": True})


def test_true_range_ratio_matches_direct_calculation_and_state_is_not_direction():
    d = market()
    f = factor_frame(d)
    t = 30
    high, low, close = d.high.to_numpy()/10., d.low.to_numpy()/10., d.wealth.to_numpy()
    tr = [max(high[j]-low[j], abs(high[j]-close[j-1]), abs(low[j]-close[j-1])) for j in range(t-13, t+1)]
    expected = 100*np.log10(sum(tr)/(max(high[t-13:t+1])-min(low[t-13:t+1])))/np.log10(14)
    np.testing.assert_allclose(f.choppiness.iloc[t], expected, atol=1e-10)
    assert f.market_state.iloc[t] == 1 and f.raw_entry.iloc[t] == 1
    d.sma120 = -.1
    assert factor_frame(d).raw_entry.eq(0).all()


def test_hysteresis_boundaries_initial_middle_and_missing_reset():
    values = [50., 38.2, 30., 50., 61.8, 70., 50., np.nan, 50., 70.]
    np.testing.assert_allclose(classify_choppiness(values), [np.nan, np.nan, 1., 1., 1., 2., 2., np.nan, np.nan, 2.], equal_nan=True)


def test_dividend_scaling_and_degenerate_flat_range_do_not_create_artificial_trend():
    d = market()
    d.loc[:, ["open", "close", "high", "low"]] = 10.
    d.loc[30:, ["open", "close", "high", "low"]] = 9.9
    d.loc[30, "dividend"] = .1
    d.wealth = 1.
    f = factor_frame(d)
    np.testing.assert_allclose(f.wealth_high, 1.)
    np.testing.assert_allclose(f.wealth_low, 1.)
    assert f.choppiness.isna().all() and f.raw_entry.eq(0).all()


def test_complete_window_missing_and_future_prefix_invariance():
    d = market()
    original = factor_frame(d)
    pd.testing.assert_frame_equal(original.iloc[:50], factor_frame(d.iloc[:50]))
    d.loc[50:, ["high", "low", "close", "wealth"]] *= 2
    pd.testing.assert_frame_equal(original.iloc[:50], factor_frame(d).iloc[:50])
    d = market()
    d.loc[30, "high"] = np.nan
    f = factor_frame(d)
    assert f.choppiness.iloc[30:44].isna().all() and np.isfinite(f.choppiness.iloc[44])


def test_actual_next_open_mode_protection_and_unknown_hold():
    d = market()
    d.loc[25:, ["open", "close", "high", "low"]] *= .88
    d.loc[25:, "wealth"] *= .88
    d.previous_close = d.close.shift().fillna(d.close.iloc[0])
    d.loc[25, "high"] = np.nan
    f = factor_frame(d)
    cfg = {"initial_capital": 200000., "lot": 100, "tick": .001, "limit_fraction": .1}
    cost = {"commission": .0002, "minimum": 5., "slippage": .0005}
    div = pd.DataFrame(columns=["record_date", "ex_date", "payment_date", "cash_dividend_per_share"])
    ledger, decisions, cycles = simulate_policy(d, div, cfg, cost, str(d.date.iloc[20].date()), rules(f), specifications()["S1_TREND_REBOUND"])
    decisions = attach_factor_context(decisions, f)
    assert ledger.loc[ledger.filled_quantity.gt(0), "date"].iloc[0] == d.date.iloc[21]
    request = decisions[decisions.origin.eq(d.date.iloc[25])].iloc[0]
    assert request.requested_quantity < 0 and "固定止损" in request.exit_reasons
    assert ledger.loc[ledger.date.eq(d.date.iloc[26]), "shares"].iloc[0] == 0
    empty = decisions[decisions.origin.eq(d.date.iloc[27])].iloc[0]
    assert pd.isna(empty.reference_weight) and empty.requested_quantity == 0
    assert ledger.accounting_error.abs().max() < 1e-6
    range_factors = f.copy()
    range_factors["raw_entry"] = 0
    range_factors.loc[20, "raw_entry"] = 2
    range_factors["raw_exit_range"] = False
    range_factors.loc[23, "raw_exit_range"] = True
    other, _, cycles = simulate_policy(d, div, cfg, cost, str(d.date.iloc[20].date()), rules(range_factors), specifications()["S1_TREND_REBOUND"])
    assert cycles.iloc[0]["mode"] == 2 and other.loc[other.filled_quantity.lt(0), "date"].iloc[0] == d.date.iloc[24]
