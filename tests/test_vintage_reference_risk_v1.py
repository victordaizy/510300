"""组合仅用对应费用的已知参考；检查缺失、时钟和真实增减仓。"""
import numpy as np
import pandas as pd
import pytest
from research.vintage_reference_risk_inputs_v1 import vintage_risk_targets, MULTIPLIER, VOLATILITY
from research.event_clock_account_v1 import simulate_event_account
from research.adaptive_allocation_v1 import Account, target_request

CFG = {"target_volatility": .10, "weight_band": .10, "initial_capital": 200000., "lot": 100, "tick": .001, "limit_fraction": .1}


def fixture(n=8, first=2):
    data = pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=n), "open": 10., "close": 10.,
        "previous_close": 10., "dividend": 0., "variance60": .0004})
    indices = np.arange(first-1, n-1)
    common = {"origin": data.date.iloc[indices].to_numpy(), "execution_date": data.date.iloc[indices+1].to_numpy(), "origin_index": indices}
    refs = {cost: pd.DataFrame({**common, "reference_weight": np.ones(len(indices))}) for cost in ["BASE", "STRESS"]}
    risk = pd.DataFrame({"date": data.date, MULTIPLIER: .5, VOLATILITY: .2})
    return data, risk, refs, data.date.iloc[first]


def test_matching_cost_and_close_indices_do_not_take_next_day_holdings():
    data, risk, refs, start = fixture()
    refs["BASE"].reference_weight = [0., 1., 1., 0., 1., 0.]
    refs["STRESS"].reference_weight = [0., 1., 0., 0., 1., 0.]
    a, _ = vintage_risk_targets(data, risk, refs, CFG, start, "BASE")
    b, _ = vintage_risk_targets(data, risk, refs, CFG, start, "STRESS")
    np.testing.assert_allclose(a.target, [np.nan, 0., .5, .5, 0., .5, 0., np.nan], equal_nan=True)
    assert a.target.iloc[3] == .5 and b.target.iloc[3] == 0.
    assert a.reference_cost.eq("BASE").all() and b.reference_cost.eq("STRESS").all()


def test_missing_reference_is_not_zero_and_outside_range_is_unknown():
    data, risk, refs, start = fixture()
    refs["BASE"].loc[1, "reference_weight"] = np.nan
    refs["BASE"].loc[2, "reference_weight"] = 0.
    f, summary = vintage_risk_targets(data, risk, refs, CFG, start, "BASE")
    assert pd.isna(f.target.iloc[2]) and f.target_status.iloc[2] == "NO_VIEW_REFERENCE_INTENT"
    assert f.target.iloc[3] == 0. and f.target_status.iloc[3] == "REFERENCE_AND_RISK_TARGET_AVAILABLE"
    assert f.target.iloc[[0, -1]].isna().all() and summary["unknown_reference_rows"] == 1


def test_zero_or_missing_volatility_keeps_saved_explicit_multiplier_and_cap():
    data, risk, refs, start = fixture()
    risk.loc[2, VOLATILITY] = 0.; risk.loc[3, VOLATILITY] = np.nan
    f, _ = vintage_risk_targets(data, risk, refs, CFG, start, "BASE")
    assert f.target.iloc[2] == .5 and f.target.iloc[3] == .5
    assert f.risk_view_status.iloc[2:4].eq("NO_NEW_VOLATILITY_VIEW_CARRIED_MULTIPLIER").all()
    risk.loc[4, MULTIPLIER] = 1.1
    with pytest.raises(ValueError):
        vintage_risk_targets(data, risk, refs, CFG, start, "BASE")


def test_shifted_clock_missing_cost_and_nonbinary_reference_are_rejected():
    data, risk, refs, start = fixture()
    shifted = {k: v.copy() for k, v in refs.items()}
    shifted["BASE"].origin += pd.Timedelta(days=1)
    with pytest.raises(ValueError):
        vintage_risk_targets(data, risk, shifted, CFG, start, "BASE")
    with pytest.raises(ValueError):
        vintage_risk_targets(data, risk, {"BASE": refs["BASE"]}, CFG, start, "STRESS")
    refs["BASE"].loc[0, "reference_weight"] = .5
    with pytest.raises(ValueError):
        vintage_risk_targets(data, risk, refs, CFG, start, "BASE")


def test_future_reference_changes_and_short_prefix_preserve_prior_targets():
    data, risk, refs, start = fixture(12)
    original, _ = vintage_risk_targets(data, risk, refs, CFG, start, "BASE")
    changed_refs = {k: v.copy() for k, v in refs.items()}
    changed_refs["BASE"].loc[changed_refs["BASE"].origin_index.ge(7), "reference_weight"] = 0.
    changed_risk = risk.copy(); changed_risk.loc[7:, MULTIPLIER] = .2
    changed, _ = vintage_risk_targets(data, changed_risk, changed_refs, CFG, start, "BASE")
    pd.testing.assert_frame_equal(original.iloc[:7], changed.iloc[:7])
    prefix_refs = {k: v[v.origin_index.lt(6)].copy() for k, v in refs.items()}
    prefix, _ = vintage_risk_targets(data.iloc[:7], risk.iloc[:7], prefix_refs, CFG, start, "BASE")
    pd.testing.assert_frame_equal(original.iloc[:6], prefix.iloc[:6])
    assert pd.isna(prefix.target.iloc[-1])


def test_existing_band_uses_own_nav_and_zero_target_exits_all():
    account = Account(100000.); account.shares = 10000
    assert target_request(account, 10., .59, CFG)["requested_quantity"] == 0
    assert target_request(account, 10., .61, CFG)["requested_quantity"] == 2200
    assert target_request(account, 10., 0., CFG)["requested_quantity"] == -10000
    empty = Account(200000.)
    assert target_request(empty, 10., .05, CFG)["requested_quantity"] == 1000


def test_real_account_partial_change_latest_target_retry_dividend_and_terminal():
    data, risk, refs, start = fixture(12)
    refs["BASE"].reference_weight = [1., 1., 1., 1., 0., 1., 0., 1., 1., 1.]
    risk[MULTIPLIER] = [1., .5, .55, .2, .2, .2, .2, .2, .6, .6, .6, .6]
    risk[VOLATILITY] = .1/risk[MULTIPLIER]
    data.loc[3, "dividend"] = .1; data.loc[6, "open"] = 9.
    div = pd.DataFrame({"record_date": [data.date.iloc[2]], "ex_date": [data.date.iloc[3]], "payment_date": [data.date.iloc[4]], "cash_dividend_per_share": [.1]})
    f, _ = vintage_risk_targets(data, risk, refs, CFG, start, "BASE")
    cost = {"commission": .0002, "minimum": 5., "slippage": .0005}
    ledger, decisions = simulate_event_account(data, div, CFG, cost, start, "VINTAGE_REFERENCE_RISK", targets=f.target.to_numpy(), event_mask=np.ones(len(data), bool))
    assert ledger.loc[ledger.filled_quantity.gt(0), "date"].tolist() == [data.date.iloc[2], data.date.iloc[9]]
    assert ledger.loc[ledger.filled_quantity.lt(0), "date"].tolist() == [data.date.iloc[4], data.date.iloc[8], data.date.iloc[11]]
    assert ledger[ledger.date.eq(data.date.iloc[3])].iloc[0].requested_quantity == 0
    assert ledger[ledger.date.eq(data.date.iloc[4])].iloc[0].shares > 0
    assert ledger[ledger.date.eq(data.date.iloc[6])].iloc[0].status == "UNFILLED_DIRECTIONAL_LIMIT"
    assert ledger[ledger.date.eq(data.date.iloc[7])].iloc[0].requested_quantity == 0
    assert np.isclose(ledger.dividend_recognized.sum(), ledger[ledger.date.eq(data.date.iloc[2])].shares.iloc[0]*.1)
    assert ledger.shares.iloc[-1] == 0 and ledger.accounting_error.abs().max() < 1e-6
    assert pd.DatetimeIndex(decisions.execution_date).equals(pd.DatetimeIndex(ledger.date))
