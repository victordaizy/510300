"""验证最小方差适配、费用分工、缺失处理及实际部分买卖。"""
import numpy as np
import pandas as pd
import pytest
from research.covariance_reference_pair_inputs_v1 import covariance_reference_frames, MODELS, PRIMARY
from research.event_clock_account_v1 import simulate_event_account

CFG = {"risk_window": 242, "annual_days": 242, "weight_band": .10, "initial_capital": 200000., "lot": 100, "tick": .001,
    "limit_fraction": .1, "costs": {"BASE": {"commission": .0002, "minimum": 5., "slippage": .0005}, "STRESS": {"commission": .0004, "minimum": 5., "slippage": .001}}}


def fixture(n=540):
    data = pd.DataFrame({"date": pd.bdate_range("2019-01-01", periods=n), "open": 10., "close": 10.,
        "previous_close": 10., "dividend": 0., "variance60": .0004})
    first = 2
    patterns = [np.array([-.02, -.01, .01, .02]), np.array([-.01, .02, -.02, .01])*.4]
    ledgers = {model: pd.DataFrame({"date": data.date.iloc[first:].to_numpy(), "net_return": patterns[j][np.arange(n-first) % 4]+.001,
        "mark_clock": ["CLOSE"]*(n-first-1)+["OPEN_TERMINAL"], "source_cost": "BASE", "source_model": model}) for j, model in enumerate(MODELS)}
    indices = np.arange(first-1, n-1)
    parents = {cost: {model: pd.DataFrame({"origin": data.date.iloc[indices].to_numpy(), "execution_date": data.date.iloc[indices+1].to_numpy(),
        "origin_index": indices, "reference_weight": (.8 if j == 0 else .2) if cost == "BASE" else (.6 if j == 0 else .1),
        "source_cost": cost, "source_model": model}) for j, model in enumerate(MODELS)} for cost in CFG["costs"]}
    return data, ledgers, parents, data.date.iloc[first]


def test_analytic_budget_and_cost_matched_targets_share_base_risk():
    data, ledgers, parents, start = fixture()
    frames, summary = covariance_reference_frames(data, ledgers, parents, CFG, start)
    base, stress = frames["BASE"], frames["STRESS"]
    assert base.target.iloc[1] == .5 and stress.target.iloc[1] == .35
    mature = base[base.risk_update_scheduled & base.risk_status.eq("MIN_VARIANCE_BUDGET_AVAILABLE")]
    assert len(mature) > 0 and summary[0]["budget_changes"] > 0
    for row in mature.itertuples():
        inputs = np.column_stack([ledger[ledger.date.between(row.risk_window_start, row.date)].net_return for ledger in ledgers.values()])
        cov = np.cov(inputs.T, ddof=1)
        weight = np.clip((cov[1, 1]-cov[0, 1])/(cov[0, 0]+cov[1, 1]-2*cov[0, 1]), 0., 1.)
        assert np.isclose(row.downside_budget, weight) and np.isclose(row.target, weight*.8+(1-weight)*.2)
        assert np.isclose(stress.target.iloc[row.origin_index], weight*.6+(1-weight)*.1)
    for column in ["downside_budget", "continuous_budget", "reference_covariance", "risk_status"]:
        pd.testing.assert_series_equal(base[column], stress[column])


def test_unknown_target_never_reallocates_and_both_zero_exit():
    data, ledgers, parents, start = fixture()
    for cost in parents:
        parents[cost][MODELS[0]].loc[parents[cost][MODELS[0]].origin_index.eq(400), "reference_weight"] = np.nan
        for parent in parents[cost].values():
            parent.loc[parent.origin_index.eq(410), "reference_weight"] = 0.
    frames, _ = covariance_reference_frames(data, ledgers, parents, CFG, start)
    for frame in frames.values():
        assert pd.isna(frame.target.iloc[400]) and frame.target.iloc[410] == 0.


def test_zero_missing_or_identical_return_risk_keeps_previous_budget():
    data, ledgers, parents, start = fixture()
    ledgers[MODELS[1]]["net_return"] = ledgers[MODELS[0]].net_return
    identical, _ = covariance_reference_frames(data, ledgers, parents, CFG, start)
    assert identical["BASE"].downside_budget.iloc[1:-1].eq(.5).all()
    assert "NO_VIEW_DEGENERATE_DIFFERENCE_KEEP_BUDGET" in set(identical["BASE"].risk_status)
    ledgers[MODELS[0]]["net_return"] = 0.
    zero, _ = covariance_reference_frames(data, ledgers, parents, CFG, start)
    assert zero["BASE"].downside_budget.iloc[1:-1].eq(.5).all()
    assert "NO_VIEW_ZERO_OR_INVALID_RISK_KEEP_BUDGET" in set(zero["BASE"].risk_status)
    ledgers[MODELS[0]]["net_return"] = np.nan
    missing, _ = covariance_reference_frames(data, ledgers, parents, CFG, start)
    assert missing["BASE"].downside_budget.iloc[1:-1].eq(.5).all()
    assert "NO_VIEW_INCOMPLETE_WINDOW_KEEP_BUDGET" in set(missing["BASE"].risk_status)


def test_future_prefix_and_terminal_open_boundaries():
    data, ledgers, parents, start = fixture()
    original, _ = covariance_reference_frames(data, ledgers, parents, CFG, start)
    altered = {k: v.copy() for k, v in ledgers.items()}
    for ledger in altered.values():
        ledger.loc[ledger.date.ge(data.date.iloc[430]), "net_return"] = -.9
    changed, _ = covariance_reference_frames(data, altered, parents, CFG, start)
    pd.testing.assert_frame_equal(original["BASE"].iloc[:430], changed["BASE"].iloc[:430])
    altered = {k: v.copy() for k, v in ledgers.items()}
    for ledger in altered.values():
        ledger.loc[len(ledger)-1, "net_return"] = 100.
    terminal_changed, _ = covariance_reference_frames(data, altered, parents, CFG, start)
    pd.testing.assert_frame_equal(original["BASE"], terminal_changed["BASE"])
    shorter_ledgers = {k: v[v.date.le(data.date.iloc[399])].copy().reset_index(drop=True) for k, v in ledgers.items()}
    for ledger in shorter_ledgers.values():
        ledger.loc[len(ledger)-1, "mark_clock"] = "OPEN_TERMINAL"
    shorter_parents = {cost: {k: v[v.origin_index.lt(399)].copy() for k, v in group.items()} for cost, group in parents.items()}
    shorter, _ = covariance_reference_frames(data.iloc[:400], shorter_ledgers, shorter_parents, CFG, start)
    pd.testing.assert_frame_equal(original["BASE"].iloc[:399], shorter["BASE"].iloc[:399])
    assert shorter["BASE"].target.iloc[[0, -1]].isna().all()


def test_wrong_window_cost_or_target_identity_is_rejected():
    data, ledgers, parents, start = fixture()
    with pytest.raises(ValueError):
        covariance_reference_frames(data, ledgers, parents, {**CFG, "risk_window": 20}, start)
    ledgers[MODELS[0]]["source_cost"] = "STRESS"
    with pytest.raises(ValueError):
        covariance_reference_frames(data, ledgers, parents, CFG, start)
    ledgers[MODELS[0]]["source_cost"] = "BASE"
    parents["STRESS"][MODELS[0]]["source_model"] = MODELS[1]
    with pytest.raises(ValueError):
        covariance_reference_frames(data, ledgers, parents, CFG, start)


def test_real_partial_rebalance_full_exit_reentry_dividend_and_terminal():
    data, ledgers, parents, start = fixture()
    for cost in parents:
        for parent in parents[cost].values():
            parent.loc[parent.origin_index.between(350, 370), "reference_weight"] = 0.
    data.loc[20, "dividend"] = .1
    dividends = pd.DataFrame({"record_date": [data.date.iloc[19]], "ex_date": [data.date.iloc[20]],
        "payment_date": [data.date.iloc[22]], "cash_dividend_per_share": [.1]})
    frames, _ = covariance_reference_frames(data, ledgers, parents, CFG, start)
    ledger, decisions = simulate_event_account(data, dividends, CFG, CFG["costs"]["BASE"], start, PRIMARY,
        targets=frames["BASE"].target.to_numpy(float), event_mask=np.ones(len(data), bool))
    assert ((ledger.filled_quantity < 0) & (ledger.shares > 0)).any()
    assert ledger[ledger.date.eq(data.date.iloc[351])].shares.iloc[0] == 0
    assert ledger[ledger.date.eq(data.date.iloc[372])].filled_quantity.iloc[0] > 0
    assert ledger.shares.iloc[-1] == 0 and ledger.mark_clock.iloc[-1] == "OPEN_TERMINAL"
    assert np.isclose(ledger.dividend_recognized.sum(), ledger[ledger.date.eq(data.date.iloc[19])].shares.iloc[0]*.1)
    assert np.isclose(ledger.dividend_recognized.sum(), ledger.dividend_paid.sum())
    assert ledger.accounting_error.abs().max() < 1e-6
    assert pd.DatetimeIndex(decisions.execution_date).equals(pd.DatetimeIndex(ledger.date))
