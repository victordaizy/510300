"""检查窗口端点、重复极值、分红连续性、缺失和真实下一开盘。"""
import numpy as np
import pandas as pd
from research.aroon_time_entry_inputs_v1 import factor_frame, adjusted_window, most_recent_age, rules, attach_factor_context
from research.simple_price_entry_exit_v1 import simulate_policy


def market(n=80):
    close = 10.+np.arange(n)*.1
    return pd.DataFrame({"date": pd.bdate_range("2020-01-02", periods=n), "high": close+.05, "low": close-.05, "close": close, "open": close, "previous_close": np.r_[10., close[:-1]], "dividend": 0.})


def test_period_25_has_26_observations_and_full_endpoint_range():
    f = factor_frame(market())
    assert f.aroon_up.iloc[:25].isna().all()
    assert f.window_observations.iloc[25] == 26
    assert f.aroon_up.iloc[25] == 100 and f.aroon_down.iloc[25] == 0
    assert f.high_age_trading_days.iloc[25] == 0 and f.low_age_trading_days.iloc[25] == 25
    assert f.raw_entry.iloc[25] == 1 and not f.raw_exit.iloc[25]


def test_repeated_extremes_use_latest_day_without_erasing_a_real_tick():
    assert most_recent_age([12., 11., 12., 11.], True, 1e-12) == 1
    assert most_recent_age([10., 11., 10., 11.], False, 1e-12) == 1
    assert most_recent_age([10., 10.001, 10.], True, 1e-12) == 1
    assert most_recent_age([10., 10., 10.+1e-14], False, 1e-12) == 0


def test_cash_distribution_does_not_create_a_false_new_low():
    d = market(40)
    d.loc[:, ["high", "low", "close"]] = 10.
    d.loc[20:, ["high", "low", "close"]] = 9.9
    d.loc[20, "dividend"] = .1
    f = factor_frame(d)
    assert f.aroon_up.iloc[25] == f.aroon_down.iloc[25] == 100
    assert not f.raw_exit.iloc[25] and f.raw_entry.iloc[25] == 0
    high, low = adjusted_window(*d[["high", "low", "close", "dividend"]].to_numpy().T)
    np.testing.assert_allclose(high, 10., atol=1e-12, rtol=0)
    np.testing.assert_allclose(low, 10., atol=1e-12, rtol=0)


def test_future_cannot_change_past_and_missing_recovers_only_after_whole_window():
    d = market()
    original = factor_frame(d)
    changed = d.copy()
    changed.loc[50:, ["high", "low", "close"]] = [25., 20., 22.]
    pd.testing.assert_frame_equal(original.iloc[:50], factor_frame(changed).iloc[:50])
    pd.testing.assert_frame_equal(original.iloc[:50], factor_frame(d.iloc[:50]))
    missing = d.copy()
    missing.loc[30, "high"] = np.nan
    f = factor_frame(missing)
    assert f.aroon_up.iloc[30:56].isna().all()
    assert np.isfinite(f.aroon_up.iloc[56])


def test_missing_aroon_does_not_block_known_loss_exit_and_accounts_use_next_open():
    d = market()
    d.loc[30:, ["open", "close"]] = 11.61
    d["previous_close"] = d.close.shift(1).fillna(d.close.iloc[0])
    d["high"], d["low"] = d.close+.05, d.close-.05
    d.loc[30, "high"] = np.nan
    factors = factor_frame(d)
    cfg = {"initial_capital": 200000., "lot": 100, "tick": .001, "limit_fraction": .1}
    cost = {"commission": .0002, "minimum": 5., "slippage": .0005}
    spec = {"cooldown": 2, "modes": {1: {"loss": .06, "trail": .08, "take": None, "days": 60}}}
    dividends = pd.DataFrame(columns=["record_date", "ex_date", "payment_date", "cash_dividend_per_share"])
    ledger, decisions, cycles = simulate_policy(d, dividends, cfg, cost, str(d.date.iloc[26].date()), rules(factors), spec)
    decisions = attach_factor_context(decisions, factors)
    assert ledger.loc[ledger.filled_quantity.gt(0), "date"].iloc[0] == d.date.iloc[26]
    exit_request = decisions[decisions.origin.eq(d.date.iloc[30])].iloc[0]
    assert exit_request.requested_quantity < 0 and "固定止损" in exit_request.exit_reasons
    assert ledger.loc[ledger.date.eq(d.date.iloc[31]), "shares"].iloc[0] == 0
    assert pd.isna(decisions.loc[decisions.origin.eq(d.date.iloc[31]), "reference_weight"].iloc[0])
    assert len(cycles) == 1 and ledger.accounting_error.abs().max() < 1e-6
