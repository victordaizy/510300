"""检查旧控制器复用、评分成本、目标费用、未知与未来时点边界。"""
import numpy as np
import pandas as pd
import pytest
from research.recent_reference_selection_inputs_v1 import selected_reference_frames, aligned_reference_inputs, MODELS, PRIMARY
from research.event_clock_account_v1 import simulate_event_account

CFG = {"selection_window": 242, "annual_days": 242, "weight_band": .10, "initial_capital": 200000., "lot": 100, "tick": .001,
    "limit_fraction": .1, "costs": {"BASE": {"commission": .0002, "minimum": 5., "slippage": .0005}, "STRESS": {"commission": .0004, "minimum": 5., "slippage": .001}}}


def fixture(n=540):
    data = pd.DataFrame({"date": pd.bdate_range("2019-01-01", periods=n), "open": 10., "close": 10.,
        "previous_close": 10., "dividend": 0., "variance60": .0004})
    first = 2
    returns = np.arange(n-first) % 2*.02-.01
    ledgers = {model: pd.DataFrame({"date": data.date.iloc[first:].to_numpy(), "net_return": returns+(j+1)*.001,
        "mark_clock": ["CLOSE"]*(len(returns)-1)+["OPEN_TERMINAL"], "source_cost": "BASE", "source_model": model}) for j, model in enumerate(MODELS)}
    indices = np.arange(first-1, n-1)
    parents = {cost: {model: pd.DataFrame({"origin": data.date.iloc[indices].to_numpy(), "execution_date": data.date.iloc[indices+1].to_numpy(),
        "origin_index": indices, "reference_weight": (.8 if j == 0 else .2) if cost == "BASE" else (.6 if j == 0 else .1),
        "source_cost": cost, "source_model": model}) for j, model in enumerate(MODELS)} for cost in CFG["costs"]}
    return data, ledgers, parents, data.date.iloc[first]


def test_shared_base_scores_and_matched_target_cost_after_mature_selection():
    data, ledgers, parents, start = fixture()
    frames, summaries = selected_reference_frames(data, ledgers, parents, CFG, start)
    a, b = frames["BASE"], frames["STRESS"]
    assert a.selected_parent.iloc[1] == MODELS[0] and a.target.iloc[1] == .8 and b.target.iloc[1] == .6
    mature = a.selection_status.eq("POSITIVE_PAST_SHARPE_PARENT_SELECTED")
    assert mature.any() and a.loc[mature, "selected_parent"].eq(MODELS[1]).all()
    assert a.loc[mature, "target"].eq(.2).all() and b.loc[mature, "target"].eq(.1).all()
    for column in ["selected_parent", "downside_past_sharpe", "continuous_past_sharpe", "selection_changed"]:
        pd.testing.assert_series_equal(a[column], b[column])
    assert summaries[0]["selection_changes"] == 1
    attempts = a[a.selection_update_scheduled]
    assert all(row.date.month != data.date.iloc[row.origin_index-1].month for row in attempts.itertuples())


def test_missing_selected_target_is_unknown_without_switching_to_other():
    data, ledgers, parents, start = fixture()
    for cost in parents:
        parents[cost][MODELS[1]].loc[parents[cost][MODELS[1]].origin_index.eq(400), "reference_weight"] = np.nan
    frames, _ = selected_reference_frames(data, ledgers, parents, CFG, start)
    assert frames["BASE"].selected_parent.iloc[400] == MODELS[1]
    assert pd.isna(frames["BASE"].target.iloc[400]) and pd.isna(frames["STRESS"].target.iloc[400])


def test_nonpositive_scores_choose_cash_but_zero_or_missing_risk_keeps_initial_choice():
    data, ledgers, parents, start = fixture()
    for ledger in ledgers.values():
        ledger["net_return"] = np.arange(len(ledger)) % 2*.02-.03
    frames, _ = selected_reference_frames(data, ledgers, parents, CFG, start)
    assert frames["BASE"].selected_parent.iloc[400] == "CASH" and frames["BASE"].target.iloc[400] == 0.
    for ledger in ledgers.values():
        ledger["net_return"] = 0.
    zero, _ = selected_reference_frames(data, ledgers, parents, CFG, start)
    assert zero["BASE"].selected_parent.iloc[400] == MODELS[0]
    assert "NO_VIEW_ZERO_OR_INVALID_VOLATILITY_KEEP_SELECTION" in set(zero["BASE"].selection_status)
    ledgers[MODELS[0]].loc[250:500, "net_return"] = np.nan
    missing, _ = selected_reference_frames(data, ledgers, parents, CFG, start)
    assert missing["BASE"].selected_parent.iloc[400] == MODELS[0] and "NO_VIEW_INCOMPLETE_WINDOW_KEEP_SELECTION" in set(missing["BASE"].selection_status)


def test_future_changes_and_terminal_open_return_cannot_change_prior_selection():
    data, ledgers, parents, start = fixture()
    original, _ = selected_reference_frames(data, ledgers, parents, CFG, start)
    altered = {k: v.copy() for k, v in ledgers.items()}
    for ledger in altered.values():
        ledger.loc[ledger.date.ge(data.date.iloc[430]), "net_return"] = -.9
    changed, _ = selected_reference_frames(data, altered, parents, CFG, start)
    pd.testing.assert_frame_equal(original["BASE"].iloc[:430], changed["BASE"].iloc[:430])
    for model in MODELS:
        altered[model] = ledgers[model].copy(); altered[model].loc[len(ledgers[model])-1, "net_return"] = 100.
    final_changed, _ = selected_reference_frames(data, altered, parents, CFG, start)
    pd.testing.assert_frame_equal(original["BASE"], final_changed["BASE"])
    assert original["BASE"].downside_scoring_return.iloc[[0, 1, -1]].isna().all()
    assert original["BASE"].target.iloc[[0, -1]].isna().all()
    shorter_ledgers = {k: v[v.date.le(data.date.iloc[399])].copy().reset_index(drop=True) for k, v in ledgers.items()}
    for ledger in shorter_ledgers.values():
        ledger.loc[len(ledger)-1, "mark_clock"] = "OPEN_TERMINAL"
    shorter_parents = {cost: {k: v[v.origin_index.lt(399)].copy() for k, v in group.items()} for cost, group in parents.items()}
    shorter, _ = selected_reference_frames(data.iloc[:400], shorter_ledgers, shorter_parents, CFG, start)
    pd.testing.assert_frame_equal(original["BASE"].iloc[:399], shorter["BASE"].iloc[:399])
    assert pd.isna(shorter["BASE"].target.iloc[-1])


def test_invalid_source_cost_calendar_return_and_window_are_rejected():
    data, ledgers, parents, start = fixture()
    bad = {k: v.copy() for k, v in ledgers.items()}; bad[MODELS[0]].source_cost = "STRESS"
    with pytest.raises(ValueError):
        aligned_reference_inputs(data, bad, parents, CFG, start)
    bad = {k: v.copy() for k, v in ledgers.items()}; bad[MODELS[0]].loc[3, "net_return"] = -1.
    with pytest.raises(ValueError):
        aligned_reference_inputs(data, bad, parents, CFG, start)
    bad = {k: v.copy() for k, v in ledgers.items()}; bad[MODELS[0]].date += pd.Timedelta(days=1)
    with pytest.raises(ValueError):
        aligned_reference_inputs(data, bad, parents, CFG, start)
    with pytest.raises(ValueError):
        selected_reference_frames(data, ledgers, parents, {**CFG, "selection_window": 20}, start)


def test_actual_account_rebalances_on_new_selection_and_keeps_dividend_and_terminal():
    data, ledgers, parents, start = fixture()
    data.loc[20, "dividend"] = .1
    dividends = pd.DataFrame({"record_date": [data.date.iloc[19]], "ex_date": [data.date.iloc[20]],
        "payment_date": [data.date.iloc[22]], "cash_dividend_per_share": [.1]})
    frames, _ = selected_reference_frames(data, ledgers, parents, CFG, start)
    factors = frames["BASE"]
    ledger, decisions = simulate_event_account(data, dividends, CFG, CFG["costs"]["BASE"], start, PRIMARY,
        targets=factors.target.to_numpy(float), event_mask=np.ones(len(data), bool))
    changed_at = int(np.flatnonzero(factors.selection_changed)[0])
    execution = ledger[ledger.date.eq(data.date.iloc[changed_at+1])].iloc[0]
    assert execution.filled_quantity < 0 and execution.shares > 0
    assert ledger.shares.iloc[-1] == 0 and ledger.mark_clock.iloc[-1] == "OPEN_TERMINAL"
    assert np.isclose(ledger.dividend_recognized.sum(), ledger[ledger.date.eq(data.date.iloc[19])].shares.iloc[0]*.1)
    assert np.isclose(ledger.dividend_recognized.sum(), ledger.dividend_paid.sum())
    assert ledger.accounting_error.abs().max() < 1e-6
    assert pd.DatetimeIndex(decisions.execution_date).equals(pd.DatetimeIndex(ledger.date))
