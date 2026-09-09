"""验证共同目标的缺失语义、费用来源、未来边界和真实账户。"""
import numpy as np
import pandas as pd
import pytest
from research.consensus_reference_target_inputs_v1 import consensus_target_frames, PRIMARY, MODELS
from research.event_clock_account_v1 import simulate_event_account
from tests.test_trend_reference_router_v1 import fixture, CFG as PARENT_CFG

CFG = {**PARENT_CFG, "combination": "MINIMUM_OF_TWO_KNOWN_PARENT_TARGETS"}


def assign(parents, cost, model, origin, value):
    frame = parents[cost][model]
    frame.loc[frame.origin_index.eq(origin), "reference_weight"] = value


def test_positive_intersection_zero_and_unknown_are_distinct():
    data, parents, start = fixture()
    assign(parents, "BASE", MODELS[0], 120, 0)
    assign(parents, "BASE", MODELS[1], 121, np.nan)
    assign(parents, "BASE", MODELS[0], 121, 0)
    assign(parents, "BASE", MODELS[0], 122, .05)
    frames, summary = consensus_target_frames(data, parents, CFG, start)
    frame = frames["BASE"]
    assert frame.target.iloc[119] == .2 and frame.target.iloc[120] == 0
    assert pd.isna(frame.target.iloc[121]) and frame.target.iloc[122] == .05
    assert frame.target.iloc[:119].isna().all() and pd.isna(frame.target.iloc[-1])
    assert summary[0]["unknown_target_origins"] == 1


def test_cost_identity_clock_and_invalid_weight_are_checked():
    data, parents, start = fixture()
    frames, _ = consensus_target_frames(data, parents, CFG, start)
    assert frames["BASE"].target.iloc[119] == .2 and frames["STRESS"].target.iloc[119] == .1
    for field, value in [("source_cost", "BASE"), ("source_model", "OTHER"), ("reference_weight", 1.1)]:
        altered = {cost: {model: frame.copy() for model, frame in group.items()} for cost, group in parents.items()}
        altered["STRESS"][MODELS[0]][field] = value
        with pytest.raises(ValueError):
            consensus_target_frames(data, altered, CFG, start)
    parents["BASE"][MODELS[0]]["execution_date"] = parents["BASE"][MODELS[0]].origin
    with pytest.raises(ValueError):
        consensus_target_frames(data, parents, CFG, start)


def test_future_parent_targets_and_shorter_prefix_do_not_change_past():
    data, parents, start = fixture()
    original, _ = consensus_target_frames(data, parents, CFG, start)
    altered = {cost: {model: frame.copy() for model, frame in group.items()} for cost, group in parents.items()}
    for group in altered.values():
        for frame in group.values():
            frame.loc[frame.origin_index.ge(130), "reference_weight"] = 0
    future, _ = consensus_target_frames(data, altered, CFG, start)
    pd.testing.assert_frame_equal(original["BASE"].iloc[:130], future["BASE"].iloc[:130])
    prefix = {cost: {model: frame[frame.origin_index.lt(129)].copy() for model, frame in group.items()} for cost, group in parents.items()}
    short, _ = consensus_target_frames(data.iloc[:130], prefix, CFG, start)
    pd.testing.assert_frame_equal(original["BASE"].iloc[:129], short["BASE"].iloc[:129])


def test_actual_zero_exit_reentry_partial_adjustment_and_dividend():
    data, parents, start = fixture()
    for model in MODELS:
        parents["BASE"][model]["reference_weight"] = .8
    assign(parents, "BASE", MODELS[0], 124, 0)
    assign(parents, "BASE", MODELS[1], 127, .3)
    data.loc[123, "dividend"] = .1
    dividends = pd.DataFrame({"record_date": [data.date.iloc[122]], "ex_date": [data.date.iloc[123]],
        "payment_date": [data.date.iloc[124]], "cash_dividend_per_share": [.1]})
    frames, _ = consensus_target_frames(data, parents, CFG, start)
    ledger, decisions = simulate_event_account(data, dividends, CFG, CFG["costs"]["BASE"], start, PRIMARY,
        targets=frames["BASE"].target.to_numpy(float), event_mask=np.ones(len(data), bool))
    by_date = ledger.set_index("date")
    assert by_date.loc[data.date.iloc[125], "shares"] == 0 and by_date.loc[data.date.iloc[126], "filled_quantity"] > 0
    assert by_date.loc[data.date.iloc[128], "filled_quantity"] < 0 and by_date.loc[data.date.iloc[128], "shares"] > 0
    assert ledger.shares.iloc[-1] == 0 and ledger.mark_clock.iloc[-1] == "OPEN_TERMINAL"
    assert np.isclose(ledger.dividend_recognized.sum(), by_date.loc[data.date.iloc[122], "shares"]*.1)
    assert np.isclose(ledger.dividend_paid.sum(), ledger.dividend_recognized.sum())
    assert ledger.accounting_error.abs().max() < 1e-6
    assert pd.DatetimeIndex(decisions.execution_date).equals(pd.DatetimeIndex(ledger.date))
