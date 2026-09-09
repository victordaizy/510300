"""验证最弱子阶段目标、月度时钟、缺失语义及真实账户。"""
import numpy as np
import pandas as pd
import pytest
from research.robust_block_growth_inputs_v1 import estimate_budget, block_growth, robust_block_growth_frames, PRIMARY, MODELS
from research.event_clock_account_v1 import simulate_event_account
from tests.test_trend_reference_router_v1 import CFG as OLD_CFG

CFG = {**OLD_CFG, "combination": "MONTHLY_MAXIMUM_WORST_THREE_BLOCK_LOG_GROWTH", "initial_131_budget": 0,
    "training_rows": 242, "block_lengths": [81, 81, 80], "optimizer_iterations": 80, "numeric_tolerance": 1e-12}


def fixture(n=540):
    data = pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=n), "open": 10., "close": 10., "previous_close": 10.,
        "dividend": 0., "variance60": .0004})
    indices = np.arange(24, n-1)
    parents = {cost: {model: pd.DataFrame({"origin": data.date.iloc[indices].to_numpy(), "origin_index": indices,
        "execution_date": data.date.iloc[indices+1].to_numpy(), "reference_weight": ((.8 if j == 0 else .2) if cost == "BASE" else (.6 if j == 0 else .1)),
        "source_cost": cost, "source_model": model}) for j, model in enumerate(MODELS)} for cost in CFG["costs"]}
    ledgers = {model: pd.DataFrame({"date": data.date.iloc[25:].to_numpy(), "net_return": .001 if j == 0 else .0001,
        "mark_clock": "CLOSE", "source_cost": "BASE", "source_model": model}) for j, model in enumerate(MODELS)}
    return data, parents, ledgers, data.date.iloc[25]


def test_worst_block_complementarity_endpoints_and_identical_tie():
    returns = np.vstack([np.tile([.02, -.02], (81, 1)), np.tile([-.02, .02], (81, 1)), np.tile([.001, .001], (80, 1))])
    weight, details = estimate_budget(returns, 0, CFG)
    assert np.isclose(weight, .5, atol=1e-10) and details["chosen_worst_block_growth"] >= details["previous_worst_block_growth"]
    assert block_growth(returns, weight).min() > block_growth(returns, 0).min()
    weight, _ = estimate_budget(np.tile([.001, 0.], (242, 1)), 0, CFG)
    assert weight == 1
    weight, _ = estimate_budget(np.tile([.001, .001], (242, 1)), .37, CFG)
    assert weight == .37
    with pytest.raises(ValueError):
        estimate_budget(np.tile([-1., 0.], (242, 1)), 0, CFG)


def test_warmup_monthly_training_and_shared_base_budget():
    data, parents, ledgers, start = fixture()
    frames, summaries, estimates = robust_block_growth_frames(data, parents, ledgers, CFG, start)
    first_fit = next(row for row in estimates if row["status"] == "BUDGET_OPTIMIZATION_COMPLETE")
    t = first_fit["origin_index"]
    f = frames["BASE"]
    assert f.reference_budget131.iloc[24:t].eq(0).all() and f.reference_budget131.iloc[t:-1].eq(1).all()
    assert t >= 25+241 and data.date.iloc[t].month != data.date.iloc[t-1].month
    assert first_fit["training_end"] == str(data.date.iloc[t].date()) and first_fit["complete_pair_rows"] == 242
    pd.testing.assert_series_equal(f.reference_budget131, frames["STRESS"].reference_budget131)
    assert f.target.iloc[24] == .2 and frames["STRESS"].target.iloc[24] == .1
    assert summaries[0]["shared_base_budget_optimizations"] > 0


def test_missing_reference_data_is_not_warmup_or_cash():
    data, parents, ledgers, start = fixture()
    _, _, estimates = robust_block_growth_frames(data, parents, ledgers, CFG, start)
    t = next(row["origin_index"] for row in estimates if row["status"] == "BUDGET_OPTIMIZATION_COMPLETE")
    ledger = ledgers[MODELS[0]]
    ledger.loc[ledger.date.eq(data.date.iloc[t]), "net_return"] = np.nan
    parents["BASE"][MODELS[0]].loc[0, "reference_weight"] = np.nan
    frames, _, altered = robust_block_growth_frames(data, parents, ledgers, CFG, start)
    assert pd.isna(frames["BASE"].target.iloc[24])
    assert pd.isna(frames["BASE"].reference_budget131.iloc[t]) and pd.isna(frames["BASE"].target.iloc[t])
    record = next(row for row in altered if row["origin_index"] == t)
    assert record["status"] == "NO_VIEW_INCOMPLETE_REFERENCE_RETURNS" and record["complete_pair_rows"] == 241


def test_wrong_reference_cost_or_clock_rejected():
    data, parents, ledgers, start = fixture()
    ledgers[MODELS[0]]["source_cost"] = "STRESS"
    with pytest.raises(ValueError):
        robust_block_growth_frames(data, parents, ledgers, CFG, start)
    data, parents, ledgers, start = fixture()
    parents["STRESS"][MODELS[0]]["execution_date"] = parents["STRESS"][MODELS[0]].origin
    with pytest.raises(ValueError):
        robust_block_growth_frames(data, parents, ledgers, CFG, start)


def test_future_returns_targets_and_prefix_do_not_change_previous_budget():
    data, parents, ledgers, start = fixture()
    original, _, _ = robust_block_growth_frames(data, parents, ledgers, CFG, start)
    changed_ledgers = {model: ledger.copy() for model, ledger in ledgers.items()}
    for ledger in changed_ledgers.values():
        ledger.loc[ledger.date.ge(data.date.iloc[400]), "net_return"] = -.03
    altered = {cost: {model: parent.copy() for model, parent in group.items()} for cost, group in parents.items()}
    for group in altered.values():
        for parent in group.values():
            parent.loc[parent.origin_index.ge(400), "reference_weight"] = 0
    future, _, _ = robust_block_growth_frames(data, altered, changed_ledgers, CFG, start)
    pd.testing.assert_frame_equal(original["BASE"].iloc[:400], future["BASE"].iloc[:400])
    pp = {cost: {model: parent[parent.origin_index.lt(399)].copy() for model, parent in group.items()} for cost, group in parents.items()}
    ll = {model: ledger[ledger.date.lt(data.date.iloc[400])].copy() for model, ledger in ledgers.items()}
    prefix, _, _ = robust_block_growth_frames(data.iloc[:400], pp, ll, CFG, start)
    pd.testing.assert_frame_equal(original["BASE"].iloc[:399], prefix["BASE"].iloc[:399])


def test_actual_initial_parent_exit_reentry_and_dividend():
    data, parents, ledgers, start = fixture(80)
    parent = parents["BASE"][MODELS[1]]
    parent["reference_weight"] = .8
    parent.loc[parent.origin_index.eq(30), "reference_weight"] = 0
    parent.loc[parent.origin_index.eq(35), "reference_weight"] = .3
    data.loc[28, "dividend"] = .1
    dividends = pd.DataFrame({"record_date": [data.date.iloc[27]], "ex_date": [data.date.iloc[28]], "payment_date": [data.date.iloc[29]], "cash_dividend_per_share": [.1]})
    frames, _, _ = robust_block_growth_frames(data, parents, ledgers, CFG, start)
    ledger, decisions = simulate_event_account(data, dividends, CFG, CFG["costs"]["BASE"], start, PRIMARY,
        targets=frames["BASE"].target.to_numpy(float), event_mask=np.ones(len(data), bool))
    by_date = ledger.set_index("date")
    assert by_date.loc[data.date.iloc[31], "shares"] == 0 and by_date.loc[data.date.iloc[32], "filled_quantity"] > 0
    assert by_date.loc[data.date.iloc[36], "filled_quantity"] < 0 and by_date.loc[data.date.iloc[36], "shares"] > 0
    assert ledger.shares.iloc[-1] == 0 and ledger.mark_clock.iloc[-1] == "OPEN_TERMINAL"
    assert np.isclose(ledger.dividend_recognized.sum(), by_date.loc[data.date.iloc[27], "shares"]*.1)
    assert np.isclose(ledger.dividend_paid.sum(), ledger.dividend_recognized.sum()) and ledger.accounting_error.abs().max() < 1e-6
    assert pd.DatetimeIndex(decisions.execution_date).equals(pd.DatetimeIndex(ledger.date))
