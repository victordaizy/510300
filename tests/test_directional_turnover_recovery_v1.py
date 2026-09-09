"""检查实际成交额、分红方向、缺失和下一开盘经济路径。"""
import numpy as np
import pandas as pd
from research.directional_turnover_recovery_inputs_v1 import factor_frame, rules, attach_factor_context
from research.simple_price_entry_exit_v1 import simulate_policy


def market(n=60):
    close = 10.+np.arange(n)*.01
    return pd.DataFrame({"date": pd.bdate_range("2020-01-02", periods=n), "open": close, "high": close+.002, "low": close-.002,
        "close": close, "previous_close": np.r_[10., close[:-1]], "dividend": 0., "amount": np.full(n, 100.)})


def test_directional_amount_uses_real_amount_and_four_prices_for_three_flows():
    d = market(4)
    d.loc[:, ["high", "low", "close"]] = np.repeat(np.array([10., 11., 10., 11.])[:, None], 3, axis=1)
    d["amount"] = [99., 2., 3., 5.]
    f = factor_frame(d, period=3)
    assert f.up_amount_share.iloc[:3].isna().all()
    assert f.positive_amount_window_cny.iloc[3] == 7. and f.negative_amount_window_cny.iloc[3] == 3.
    assert f.up_amount_share.iloc[3] == 70.
    scaled = d.copy()
    scaled.amount *= 1000
    np.testing.assert_allclose(f.up_amount_share, factor_frame(scaled, period=3).up_amount_share, equal_nan=True)


def test_cash_dividend_flat_price_is_neutral_and_empty_flow_is_no_view():
    d = market()
    d.loc[:, ["high", "low", "close"]] = 10.
    d.loc[20:, ["high", "low", "close"]] = 9.9
    d.loc[20, "dividend"] = .1
    f = factor_frame(d)
    assert f.adjusted_price_direction.iloc[20:22].eq(0).all()
    assert f.up_amount_share.isna().all() and f.raw_entry.eq(0).all()
    assert f.factor_status.iloc[14:].eq("NO_VIEW_NO_DIRECTIONAL_TURNOVER").all()


def test_missing_amount_retains_full_window_and_known_zero_is_distinct():
    d = market()
    d.loc[30, "amount"] = np.nan
    f = factor_frame(d)
    assert f.up_amount_share.iloc[30:44].isna().all() and f.up_amount_share.iloc[44] == 100.
    d.loc[:, "amount"] = 0.
    empty = factor_frame(d)
    assert empty.factor_status.iloc[14:].eq("NO_VIEW_NO_DIRECTIONAL_TURNOVER").all()


def test_future_does_not_change_past_and_recovery_requires_crossing_below_balance():
    d = market()
    original = factor_frame(d)
    modified = d.copy()
    modified.loc[35:, "amount"] = 1.e12
    modified.loc[35:, ["high", "low", "close"]] = 20.
    pd.testing.assert_frame_equal(original.iloc[:35], factor_frame(modified).iloc[:35])
    pd.testing.assert_frame_equal(original.iloc[:35], factor_frame(d.iloc[:35]))
    assert original.raw_entry.eq(0).all()


def test_actual_next_open_and_known_stop_survive_missing_flow():
    d = market()
    d.loc[:14, "close"] = 10.-np.arange(15)*.01
    d.loc[15:16, "close"] = 9.9
    d.loc[17:, "close"] = 9.4
    d["open"] = d.close
    d["high"], d["low"] = d.close+.002, d.close-.002
    d["previous_close"] = d.close.shift(1).fillna(d.close.iloc[0])
    d.loc[15, "amount"] = 600.
    d.loc[17, "amount"] = np.nan
    f = factor_frame(d)
    assert f.raw_entry.iloc[15] == 1 and 20. < f.up_amount_share.iloc[15] < 50.
    cfg = {"initial_capital": 200000., "lot": 100, "tick": .001, "limit_fraction": .1}
    cost = {"commission": .0002, "minimum": 5., "slippage": .0005}
    spec = {"cooldown": 2, "modes": {1: {"loss": .04, "trail": None, "take": .06, "days": 10}}}
    div = pd.DataFrame(columns=["record_date", "ex_date", "payment_date", "cash_dividend_per_share"])
    ledger, decisions, cycles = simulate_policy(d, div, cfg, cost, str(d.date.iloc[15].date()), rules(f), spec)
    decisions = attach_factor_context(decisions, f)
    assert ledger.loc[ledger.filled_quantity.gt(0), "date"].iloc[0] == d.date.iloc[16]
    request = decisions[decisions.origin.eq(d.date.iloc[17])].iloc[0]
    assert request.requested_quantity < 0 and "固定止损" in request.exit_reasons
    assert ledger.loc[ledger.date.eq(d.date.iloc[18]), "shares"].iloc[0] == 0
    assert pd.isna(decisions.loc[decisions.origin.eq(d.date.iloc[18]), "reference_weight"].iloc[0])
    assert len(cycles) == 1 and ledger.accounting_error.abs().max() < 1e-6
