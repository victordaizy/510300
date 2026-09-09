"""验证15:05真实可用边界、训练支持条件、费用及实际切换账户。"""
import numpy as np
import pandas as pd
import pytest
from research.model_support_reference_router_inputs_v1 import support_routed_frames, validate_support_records, MODELS, PRIMARY, INSUFFICIENT
from research.event_clock_account_v1 import simulate_event_account

CFG = {"decision_clock": "15:05:00", "minimum_training_cycles": 10, "minimum_training_rows": 100, "weight_band": .1,
    "initial_capital": 200000., "lot": 100, "tick": .001, "limit_fraction": .1,
    "costs": {"BASE": {"commission": .0002, "minimum": 5., "slippage": .0005}, "STRESS": {"commission": .0004, "minimum": 5., "slippage": .001}}}


def record(data, index, mature):
    count = 10 if mature else 1
    return {"fit_index": index, "fit_origin": str(data.date.iloc[index].date()),
        "fit_time": str(data.date.iloc[index].date())+"T15:05:00", "status": "FIT_COMPLETE" if mature else INSUFFICIENT,
        "eligible_for_fit": mature, "training_cycle_count": count, "training_cycles": list(range(count)), "training_rows": 100 if mature else 50,
        "latest_exit_index": index-1, "latest_exit_date": str(data.date.iloc[index-1].date()), "missing_feature_rows": 0,
        "failure": None, "model": {"saved_model_identity": "合成训练记录"} if mature else None}


def fixture(n=12):
    data = pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=n), "open": 10., "close": 10., "previous_close": 10., "dividend": 0., "variance60": .0004})
    records = [record(data, 1, False), record(data, 4, True)]
    indices = np.arange(1, n-1)
    parents = {cost: {model: pd.DataFrame({"origin": data.date.iloc[indices].to_numpy(), "execution_date": data.date.iloc[indices+1].to_numpy(),
        "origin_index": indices, "reference_weight": (.8 if j == 0 else .2) if cost == "BASE" else (.6 if j == 0 else .1),
        "source_cost": cost, "source_model": model}) for j, model in enumerate(MODELS)} for cost in CFG["costs"]}
    return data, records, parents, data.date.iloc[2]


def test_known_support_chooses_source_and_latest_insufficient_record_does_not_use_older_fit():
    data, records, parents, start = fixture()
    records.append(record(data, 8, False))
    frames, summary = support_routed_frames(data, records, parents, CFG, start)
    frame = frames["BASE"]
    assert frame.target.iloc[1] == .8 and frame.target.iloc[4] == .2 and frame.target.iloc[8] == .8
    assert summary[0]["selection_changes"] == 2 and frame.target.iloc[[0, -1]].isna().all()
    empty, _ = support_routed_frames(data, [], parents, CFG, start)
    assert empty["BASE"].selected_parent.iloc[1:-1].eq(MODELS[0]).all()


def test_training_at_1506_cannot_be_used_for_same_day_1505_decision():
    data, records, parents, start = fixture()
    records[1]["fit_time"] = str(data.date.iloc[4].date())+"T15:06:00"
    frames, _ = support_routed_frames(data, records, parents, CFG, start)
    assert frames["BASE"].selected_parent.iloc[4] == MODELS[0]
    assert frames["BASE"].selected_parent.iloc[5] == MODELS[1]
    known = frames["BASE"].dropna(subset=["support_fit_time"])
    assert (known.support_fit_time <= known.decision_time).all()


def test_bad_support_future_exit_and_unknown_status_are_rejected():
    data, records, parents, start = fixture()
    for changes in [{"training_rows": 99}, {"missing_feature_rows": 1}, {"status": "UNKNOWN"},
        {"latest_exit_index": 5}, {"fit_time": str(data.date.iloc[4].date())+"T14:59:00"}]:
        changed = [dict(r) for r in records]
        changed[1].update(changes)
        with pytest.raises(ValueError):
            validate_support_records(data, changed)
    with pytest.raises(ValueError):
        support_routed_frames(data, records, parents, {**CFG, "minimum_training_rows": 99}, start)


def test_support_is_shared_but_selected_unknown_target_never_falls_back():
    data, records, parents, start = fixture()
    frames, _ = support_routed_frames(data, records, parents, CFG, start)
    pd.testing.assert_series_equal(frames["BASE"].selected_parent, frames["STRESS"].selected_parent)
    assert frames["BASE"].target.iloc[4] == .2 and frames["STRESS"].target.iloc[4] == .1
    parents["BASE"][MODELS[1]].loc[parents["BASE"][MODELS[1]].origin_index.eq(5), "reference_weight"] = np.nan
    unknown, _ = support_routed_frames(data, records, parents, CFG, start)
    assert unknown["BASE"].selected_parent.iloc[5] == MODELS[1] and pd.isna(unknown["BASE"].target.iloc[5])
    parents["STRESS"][MODELS[0]]["source_cost"] = "BASE"
    with pytest.raises(ValueError):
        support_routed_frames(data, records, parents, CFG, start)


def test_future_records_targets_and_shorter_prefix_keep_prior_choices():
    data, records, parents, start = fixture()
    original, _ = support_routed_frames(data, records, parents, CFG, start)
    altered_records = records+[record(data, 8, False)]
    altered_parents = {cost: {k: v.copy() for k, v in group.items()} for cost, group in parents.items()}
    for group in altered_parents.values():
        for parent in group.values():
            parent.loc[parent.origin_index.ge(8), "reference_weight"] = 0.
    future, _ = support_routed_frames(data, altered_records, altered_parents, CFG, start)
    pd.testing.assert_frame_equal(original["BASE"].iloc[:8], future["BASE"].iloc[:8])
    prefix = {cost: {k: v[v.origin_index.lt(7)].copy() for k, v in group.items()} for cost, group in parents.items()}
    short, _ = support_routed_frames(data.iloc[:8], altered_records, prefix, CFG, start)
    pd.testing.assert_frame_equal(original["BASE"].iloc[:7], short["BASE"].iloc[:7])


def test_real_account_rebalances_exits_reenters_and_receives_dividend():
    data, records, parents, start = fixture()
    for group in parents.values():
        group[MODELS[1]].loc[group[MODELS[1]].origin_index.eq(6), "reference_weight"] = 0.
    data.loc[3, "dividend"] = .1
    dividends = pd.DataFrame({"record_date": [data.date.iloc[2]], "ex_date": [data.date.iloc[3]],
        "payment_date": [data.date.iloc[4]], "cash_dividend_per_share": [.1]})
    frames, _ = support_routed_frames(data, records, parents, CFG, start)
    ledger, decisions = simulate_event_account(data, dividends, CFG, CFG["costs"]["BASE"], start, PRIMARY,
        targets=frames["BASE"].target.to_numpy(float), event_mask=np.ones(len(data), bool))
    by_date = ledger.set_index("date")
    assert by_date.loc[data.date.iloc[5], "filled_quantity"] < 0 and by_date.loc[data.date.iloc[5], "shares"] > 0
    assert by_date.loc[data.date.iloc[7], "shares"] == 0 and by_date.loc[data.date.iloc[8], "filled_quantity"] > 0
    assert ledger.shares.iloc[-1] == 0 and ledger.mark_clock.iloc[-1] == "OPEN_TERMINAL"
    assert np.isclose(ledger.dividend_recognized.sum(), by_date.loc[data.date.iloc[2], "shares"]*.1)
    assert np.isclose(ledger.dividend_paid.sum(), ledger.dividend_recognized.sum())
    assert ledger.accounting_error.abs().max() < 1e-6
    assert pd.DatetimeIndex(decisions.execution_date).equals(pd.DatetimeIndex(ledger.date))
