"""检验首次资格固定、零退出、未知状态与真实账户再进入。"""
import numpy as np
import pandas as pd
import pytest
from research.episode_trend_admission_inputs_v1 import episode_trend_frames, checked_long_trend, PRIMARY, MODELS
from research.event_clock_account_v1 import simulate_event_account
from tests.test_trend_reference_router_v1 import CFG as OLD_CFG

CFG = {**OLD_CFG, "entry_trend_window": 200, "entry_trend_threshold": 0,
    "combination": "FIRST_POSITIVE_PARENT_EPISODE_LONG_TREND_ADMISSION"}


def factors(data):
    data["sma200"] = data.wealth/data.wealth.rolling(200, min_periods=200).mean()-1


def fixture():
    data = pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=280), "open": 10., "close": 10., "previous_close": 10.,
        "dividend": 0., "variance60": .0004, "wealth": 1.})
    data.loc[224, "wealth"] = .9
    data.loc[225:231, "wealth"] = 1.2
    data.loc[232:235, "wealth"] = .7
    data.loc[236:, "wealth"] = 1.1
    factors(data)
    indices = np.arange(224, 279)
    parents = {cost: {MODELS[0]: pd.DataFrame({"origin": data.date.iloc[indices].to_numpy(), "origin_index": indices,
        "execution_date": data.date.iloc[indices+1].to_numpy(), "reference_weight": .8 if cost == "BASE" else .6,
        "source_cost": cost, "source_model": MODELS[0]})} for cost in CFG["costs"]}
    for group in parents.values():
        frame = group[MODELS[0]]
        frame.loc[frame.origin_index.isin([229, 235]), "reference_weight"] = 0
    return data, parents, data.date.iloc[225]


def test_first_rejection_admission_fixed_through_opposite_trend():
    data, parents, start = fixture()
    frames, summary = episode_trend_frames(data, parents, CFG, start)
    f = frames["BASE"]
    assert f.target.iloc[224:230].eq(0).all()
    assert f.episode_gate.iloc[228] == "REJECTED" and f.long_trend_deviation200.iloc[228] > 0
    assert f.target.iloc[230:235].eq(.8).all()
    assert f.long_trend_deviation200.iloc[232] < 0 and f.episode_gate.iloc[232] == "ADMITTED"
    assert f.target.iloc[235] == 0 and f.target.iloc[236] == .8
    assert summary[0]["admission_counts"] == {"ADMITTED": 2, "REJECTED": 1, "NO_VIEW_ENTRY_TREND": 0}
    assert summary[0]["ended_signal_episodes"] == 2 and summary[0]["last_signal_episode_still_active"]


def test_unknown_entry_whole_episode_parent_missing_preserves_state():
    data, parents, start = fixture()
    data.loc[220, "wealth"] = np.nan
    factors(data)
    frames, _ = episode_trend_frames(data, parents, CFG, start)
    assert frames["BASE"].target.iloc[224:229].isna().all() and frames["BASE"].target.iloc[229] == 0
    assert frames["BASE"].episode_gate.iloc[224] == "NO_VIEW_ENTRY_TREND"
    data, parents, start = fixture()
    parents["BASE"][MODELS[0]].loc[parents["BASE"][MODELS[0]].origin_index.eq(226), "reference_weight"] = np.nan
    frames, summary = episode_trend_frames(data, parents, CFG, start)
    assert pd.isna(frames["BASE"].target.iloc[226]) and frames["BASE"].target.iloc[227] == 0
    assert frames["BASE"].episode_entry_origin.iloc[227] == data.date.iloc[224] and summary[0]["parent_signal_episodes"] == 3


def test_separate_cost_episodes_and_wrong_clock_or_factor_rejected():
    data, parents, start = fixture()
    parents["STRESS"][MODELS[0]].loc[0, "reference_weight"] = 0
    frames, _ = episode_trend_frames(data, parents, CFG, start)
    assert frames["BASE"].target.iloc[225] == 0 and frames["STRESS"].target.iloc[225] == .6
    changed = data.copy()
    changed.loc[224, "sma200"] = 1
    with pytest.raises(AssertionError):
        checked_long_trend(changed, CFG)
    parents["BASE"][MODELS[0]]["execution_date"] = parents["BASE"][MODELS[0]].origin
    with pytest.raises(ValueError):
        episode_trend_frames(data, parents, CFG, start)


def test_future_market_and_target_changes_and_prefix_isolation():
    data, parents, start = fixture()
    original, _ = episode_trend_frames(data, parents, CFG, start)
    changed = data.copy()
    changed.loc[260:, "wealth"] *= 4
    factors(changed)
    altered = {cost: {model: frame.copy() for model, frame in group.items()} for cost, group in parents.items()}
    for group in altered.values():
        for frame in group.values():
            frame.loc[frame.origin_index.ge(260), "reference_weight"] = 0
    future, _ = episode_trend_frames(changed, altered, CFG, start)
    pd.testing.assert_frame_equal(original["BASE"].iloc[:260], future["BASE"].iloc[:260])
    prefix = {cost: {model: frame[frame.origin_index.lt(259)].copy() for model, frame in group.items()} for cost, group in parents.items()}
    short, _ = episode_trend_frames(data.iloc[:260], prefix, CFG, start)
    pd.testing.assert_frame_equal(original["BASE"].iloc[:259], short["BASE"].iloc[:259])


def test_actual_rejected_episode_entry_exit_reentry_and_dividend():
    data, parents, start = fixture()
    data.loc[233, "dividend"] = .1
    dividends = pd.DataFrame({"record_date": [data.date.iloc[232]], "ex_date": [data.date.iloc[233]],
        "payment_date": [data.date.iloc[234]], "cash_dividend_per_share": [.1]})
    frames, _ = episode_trend_frames(data, parents, CFG, start)
    ledger, decisions = simulate_event_account(data, dividends, CFG, CFG["costs"]["BASE"], start, PRIMARY,
        targets=frames["BASE"].target.to_numpy(float), event_mask=np.ones(len(data), bool))
    by_date = ledger.set_index("date")
    assert ledger.shares.iloc[:6].eq(0).all()
    assert by_date.loc[data.date.iloc[231], "filled_quantity"] > 0
    assert by_date.loc[data.date.iloc[236], "shares"] == 0 and by_date.loc[data.date.iloc[237], "filled_quantity"] > 0
    assert ledger.shares.iloc[-1] == 0 and ledger.mark_clock.iloc[-1] == "OPEN_TERMINAL"
    assert np.isclose(ledger.dividend_recognized.sum(), by_date.loc[data.date.iloc[232], "shares"]*.1)
    assert np.isclose(ledger.dividend_paid.sum(), ledger.dividend_recognized.sum()) and ledger.accounting_error.abs().max() < 1e-6
    assert pd.DatetimeIndex(decisions.execution_date).equals(pd.DatetimeIndex(ledger.date))
