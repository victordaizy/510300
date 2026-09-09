"""核对平方风险分母、分红、缺失、时点及两个真实资金路径。"""
import numpy as np
import pandas as pd
import pytest
from research.downside_reference_risk_inputs_v1 import moment_risk_frame, moment_reference_targets, PRIMARY, CONTROL, MODELS
from research.event_clock_account_v1 import simulate_event_account

CFG = {"risk_window": 20, "downside_symmetric_scale": 2., "annual_days": 242, "target_volatility": .10,
    "weight_band": .10, "initial_capital": 200000., "lot": 100, "tick": .001, "limit_fraction": .1}


def prices(returns):
    close = np.r_[100., 100.*np.cumprod(1.+np.asarray(returns))]
    previous = np.r_[np.nan, close[:-1]]
    return pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=len(close)), "close": close,
        "previous_close": previous, "open": np.r_[100., close[:-1]], "dividend": 0., "variance60": .0004})


def references(data, first=22):
    indices = np.arange(first-1, len(data)-1)
    refs = {cost: pd.DataFrame({"origin": data.date.iloc[indices].to_numpy(), "execution_date": data.date.iloc[indices+1].to_numpy(),
        "origin_index": indices, "reference_weight": 1.}) for cost in ["BASE", "STRESS"]}
    return refs, data.date.iloc[first]


def test_complete_denominator_symmetric_scaling_and_direction():
    symmetric = moment_risk_frame(prices([-.02, .02]*10), CFG)
    assert symmetric[f"{PRIMARY}_risk"].iloc[:20].isna().all()
    assert np.isclose(symmetric.downside_second_moment20.iloc[-1], .0002)
    assert np.isclose(symmetric.full_second_moment20.iloc[-1], .0004)
    assert np.isclose(symmetric[f"{PRIMARY}_risk"].iloc[-1], .02*np.sqrt(242))
    assert np.isclose(symmetric[f"{PRIMARY}_risk"].iloc[-1], symmetric[f"{CONTROL}_risk"].iloc[-1])
    negative = moment_risk_frame(prices([-.02]*20), CFG)
    assert np.isclose(negative[f"{PRIMARY}_risk"].iloc[-1], np.sqrt(2)*negative[f"{CONTROL}_risk"].iloc[-1])
    positive = moment_risk_frame(prices([.02]*20), CFG)
    assert positive[f"{PRIMARY}_risk"].iloc[-1] == 0.
    assert positive[f"{PRIMARY}_multiplier"].iloc[-1] == 1.


def test_economic_return_includes_dividend_without_ex_date_false_drop():
    data = prices([0.]*30)
    data.loc[21, "close"] -= 2.
    data.loc[21, "dividend"] = 2.
    risk = moment_risk_frame(data, CFG)
    assert risk.economic_simple_return.eq(0.).iloc[1:].all()
    assert risk[f"{PRIMARY}_risk"].eq(0.).iloc[20:].all()


def test_missing_window_stays_unknown_and_zero_risk_carries_prior_multiplier():
    data = prices([-.4, .1]*20+[0.]*25)
    data.loc[25, "dividend"] = np.nan
    risk = moment_risk_frame(data, CFG)
    for model in MODELS:
        assert risk[f"{model}_risk"].iloc[25:45].isna().all()
        assert risk[f"{model}_multiplier"].iloc[25:45].eq(risk[f"{model}_multiplier"].iloc[24]).all()
        assert risk[f"{model}_risk"].iloc[60:].eq(0.).all()
        assert risk[f"{model}_multiplier"].iloc[60:].eq(risk[f"{model}_multiplier"].iloc[59]).all()
        assert risk[f"{model}_multiplier"].between(0., 1.).all()
    assert risk.complete_return_count20.iloc[25:45].eq(19.).all()


def test_cost_matching_unknown_reference_and_future_prefix_invariance():
    data = prices([-.02, .01]*40)
    risk = moment_risk_frame(data, CFG)
    refs, start = references(data)
    refs["BASE"].loc[2, "reference_weight"] = np.nan
    refs["STRESS"].loc[2, "reference_weight"] = 0.
    a, summary = moment_reference_targets(data, risk, refs, CFG, start, "BASE", PRIMARY)
    b, _ = moment_reference_targets(data, risk, refs, CFG, start, "STRESS", PRIMARY)
    assert pd.isna(a.target.iloc[23]) and b.target.iloc[23] == 0.
    assert a.target.iloc[:21].isna().all() and pd.isna(a.target.iloc[-1])
    assert summary["unknown_reference_rows"] == 1 and "realized_annual_volatility20" not in a
    changed = data.copy(); changed.loc[45:, "close"] *= .8
    modified = moment_risk_frame(changed, CFG)
    pd.testing.assert_frame_equal(risk.iloc[:45], modified.iloc[:45])
    pd.testing.assert_frame_equal(risk.iloc[:40], moment_risk_frame(data.iloc[:40], CFG))


def test_invalid_data_clock_configuration_and_reference_are_rejected():
    data = prices([-.02, .01]*20)
    for column, value in [("close", 0.), ("previous_close", np.inf), ("dividend", -.1)]:
        changed = data.copy(); changed.loc[25, column] = value
        with pytest.raises(ValueError):
            moment_risk_frame(changed, CFG)
    with pytest.raises(ValueError):
        moment_risk_frame(data, {**CFG, "downside_symmetric_scale": 1.})
    risk = moment_risk_frame(data, CFG)
    refs, start = references(data)
    refs["BASE"].origin += pd.Timedelta(days=1)
    with pytest.raises(ValueError):
        moment_reference_targets(data, risk, refs, CFG, start, "BASE", PRIMARY)
    with pytest.raises(ValueError):
        moment_reference_targets(data, risk, refs, CFG, start, "STRESS", "UNREGISTERED")


def test_both_moment_targets_trade_own_account_with_partial_adjustments_and_dividends():
    data = prices([-.01, .01]*15+[-.08, .07]*20)
    data.loc[26, "dividend"] = .1
    refs, start = references(data)
    refs["BASE"].loc[refs["BASE"].origin_index.eq(40), "reference_weight"] = 0.
    refs["BASE"].loc[refs["BASE"].origin_index.eq(43), "reference_weight"] = np.nan
    risk = moment_risk_frame(data, CFG)
    dividends = pd.DataFrame({"record_date": [data.date.iloc[25]], "ex_date": [data.date.iloc[26]],
        "payment_date": [data.date.iloc[28]], "cash_dividend_per_share": [.1]})
    cost = {"commission": .0002, "minimum": 5., "slippage": .0005}
    paths = []
    for model in MODELS:
        f, _ = moment_reference_targets(data, risk, refs, CFG, start, "BASE", model)
        ledger, decisions = simulate_event_account(data, dividends, CFG, cost, start, model,
            targets=f.target.to_numpy(), event_mask=np.ones(len(data), bool))
        assert pd.DatetimeIndex(ledger.date).equals(pd.DatetimeIndex(data.date.iloc[22:]))
        assert ledger.filled_quantity.gt(0).sum() >= 2 and ledger.filled_quantity.lt(0).sum() >= 3
        assert ledger.loc[ledger.date.eq(data.date.iloc[41]), "shares"].iloc[0] == 0
        assert ledger.loc[ledger.date.eq(data.date.iloc[44]), "requested_quantity"].iloc[0] == 0
        assert ledger.dividend_recognized.sum() > 0 and np.isclose(ledger.dividend_paid.sum(), ledger.dividend_recognized.sum())
        assert ledger.shares.iloc[-1] == 0 and ledger.mark_clock.iloc[-1] == "OPEN_TERMINAL"
        assert ledger.accounting_error.abs().max() < 1e-6
        np.testing.assert_allclose(ledger.cash+ledger.shares*ledger.mark+ledger.dividend_receivable, ledger.equity, atol=1e-6)
        assert pd.DatetimeIndex(decisions.execution_date).equals(pd.DatetimeIndex(ledger.date))
        paths.append(ledger.shares.to_numpy())
    assert not np.array_equal(*paths)
