"""覆盖双方向、零重置、缺失保留、时点隔离及实际退出再进入。"""
import numpy as np
import pandas as pd
import pytest
from research.monotone_episode_budget_inputs_v1 import monotone_episode_frames, PRIMARY, CANDIDATES, MODELS
from research.event_clock_account_v1 import simulate_event_account
from tests.test_episode_trend_admission_v1 import fixture, CFG as OLD_CFG

CFG = {**OLD_CFG, "combination": "BOTH_MONOTONE_PARENT_EPISODE_BUDGETS", "candidate_models": list(CANDIDATES)}


def sample():
    data, parents, start = fixture()
    for cost, group in parents.items():
        group[MODELS[0]]["reference_weight"] = 0.
        group[MODELS[0]].loc[:7, "reference_weight"] = [0., .8, .4, np.nan, .7, 0., .3, .6]
    return data, parents, start


def test_both_directions_unknown_and_zero_reset():
    data, parents, start = sample()
    frames, summaries = monotone_episode_frames(data, parents, CFG, start)
    f = frames["BASE"]
    np.testing.assert_allclose(f[PRIMARY+"_target"].iloc[224:233], [0, .8, .4, np.nan, .4, 0, .3, .3, 0], equal_nan=True)
    np.testing.assert_allclose(f[list(CANDIDATES)[1]+"_target"].iloc[224:233], [0, .8, .8, np.nan, .8, 0, .3, .6, 0], equal_nan=True)
    assert f.parent_episode_start.iloc[228] == data.date.iloc[225]
    assert f.parent_episode_start.iloc[230] == data.date.iloc[230]
    assert all(row["parent_signal_episodes"] == 2 and row["ended_signal_episodes"] == 2 for row in summaries)


def test_initial_unknown_and_cost_specific_parent_episodes():
    data, parents, start = sample()
    parents["BASE"][MODELS[0]].loc[0, "reference_weight"] = np.nan
    parents["STRESS"][MODELS[0]].loc[1, "reference_weight"] = 0
    frames, _ = monotone_episode_frames(data, parents, CFG, start)
    assert pd.isna(frames["BASE"][PRIMARY+"_target"].iloc[224])
    assert frames["BASE"][PRIMARY+"_target"].iloc[225] == .8
    assert frames["STRESS"][PRIMARY+"_target"].iloc[225] == 0
    assert frames["STRESS"][list(CANDIDATES)[1]+"_target"].iloc[226] == .4


def test_invalid_parent_clock_and_bound_rejected():
    data, parents, start = sample()
    parents["BASE"][MODELS[0]].loc[1, "reference_weight"] = 1.1
    with pytest.raises(ValueError):
        monotone_episode_frames(data, parents, CFG, start)
    parents["BASE"][MODELS[0]].loc[1, "reference_weight"] = .8
    parents["BASE"][MODELS[0]]["execution_date"] = parents["BASE"][MODELS[0]].origin
    with pytest.raises(ValueError):
        monotone_episode_frames(data, parents, CFG, start)


def test_future_and_prefix_isolation():
    data, parents, start = sample()
    original, _ = monotone_episode_frames(data, parents, CFG, start)
    changed = {cost: {model: f.copy() for model, f in group.items()} for cost, group in parents.items()}
    for group in changed.values():
        group[MODELS[0]].loc[group[MODELS[0]].origin_index.ge(260), "reference_weight"] = .9
    future, _ = monotone_episode_frames(data, changed, CFG, start)
    pd.testing.assert_frame_equal(original["BASE"].iloc[:260], future["BASE"].iloc[:260])
    prefix = {cost: {model: f[f.origin_index.lt(259)].copy() for model, f in group.items()} for cost, group in parents.items()}
    short, _ = monotone_episode_frames(data.iloc[:260], prefix, CFG, start)
    pd.testing.assert_frame_equal(original["BASE"].iloc[:259], short["BASE"].iloc[:259])


def test_both_actual_accounts_exit_reentry_and_dividend():
    data, parents, start = sample()
    data.loc[228, "dividend"] = .1
    dividends = pd.DataFrame({"record_date": [data.date.iloc[227]], "ex_date": [data.date.iloc[228]],
        "payment_date": [data.date.iloc[229]], "cash_dividend_per_share": [.1]})
    frames, _ = monotone_episode_frames(data, parents, CFG, start)
    for model in CANDIDATES:
        ledger, decisions = simulate_event_account(data, dividends, CFG, CFG["costs"]["BASE"], start, model,
            targets=frames["BASE"][model+"_target"].to_numpy(float), event_mask=np.ones(len(data), bool))
        by_date = ledger.set_index("date")
        assert by_date.loc[data.date.iloc[226], "filled_quantity"] > 0
        assert by_date.loc[data.date.iloc[230], "shares"] == 0
        assert by_date.loc[data.date.iloc[231], "filled_quantity"] > 0
        assert by_date.loc[data.date.iloc[228], "requested_quantity"] == 0
        assert ledger.shares.iloc[-1] == 0 and ledger.mark_clock.iloc[-1] == "OPEN_TERMINAL"
        assert np.isclose(ledger.dividend_recognized.sum(), by_date.loc[data.date.iloc[227], "shares"]*.1)
        assert np.isclose(ledger.dividend_paid.sum(), ledger.dividend_recognized.sum())
        assert ledger.accounting_error.abs().max() < 1e-6
        assert pd.DatetimeIndex(decisions.execution_date).equals(pd.DatetimeIndex(ledger.date))
