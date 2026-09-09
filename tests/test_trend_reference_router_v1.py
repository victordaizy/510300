"""检查完整趋势窗口、未知状态、来源费用、未来边界和实际交易时钟。"""
import numpy as np
import pandas as pd
import pytest
from research.trend_reference_router_inputs_v1 import checked_trend, trend_routed_frames, MODELS, PRIMARY
from research.event_clock_account_v1 import simulate_event_account

CFG = {"decision_clock": "15:05:00", "trend_window": 120, "trend_threshold": 0, "weight_band": .1,
    "initial_capital": 200000., "lot": 100, "tick": .001, "limit_fraction": .1,
    "costs": {"BASE": {"commission": .0002, "minimum": 5., "slippage": .0005}, "STRESS": {"commission": .0004, "minimum": 5., "slippage": .001}}}


def fixture():
    data = pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=140), "open": 10., "close": 10., "previous_close": 10.,
        "dividend": 0., "variance60": .0004, "wealth": 1.})
    data.loc[121:123, "wealth"] = 1.2
    data.loc[124:126, "wealth"] = .9
    data.loc[127:, "wealth"] = 1.1
    data["sma120"] = data.wealth / data.wealth.rolling(120).mean()-1
    indices = np.arange(119, 139)
    parents = {cost: {model: pd.DataFrame({"origin": data.date.iloc[indices].to_numpy(), "execution_date": data.date.iloc[indices+1].to_numpy(),
        "origin_index": indices, "reference_weight": (.8 if j == 0 else .2) if cost == "BASE" else (.6 if j == 0 else .1),
        "source_cost": cost, "source_model": model}) for j, model in enumerate(MODELS)} for cost in CFG["costs"]}
    return data, parents, data.date.iloc[120]


def test_complete_window_equality_and_two_directions():
    data, parents, start = fixture()
    frames, summary = trend_routed_frames(data, parents, CFG, start)
    frame = frames["BASE"]
    assert frame.target.iloc[119] == .2 and frame.trend_deviation120.iloc[119] == 0
    assert frame.target.iloc[121] == .8 and frame.target.iloc[124] == .2 and frame.target.iloc[127] == .8
    assert frame.target.iloc[:119].isna().all() and pd.isna(frame.target.iloc[-1])
    assert summary[0]["selection_changes"] == 3
    assert frame.trend_deviation120.iloc[:119].isna().all()


def test_missing_window_or_selected_target_never_falls_back():
    data, parents, start = fixture()
    data.loc[124, "wealth"] = np.nan
    data["sma120"] = data.wealth/data.wealth.rolling(120).mean()-1
    frames, _ = trend_routed_frames(data, parents, CFG, start)
    assert frames["BASE"].target.iloc[124:-1].isna().all()
    data, parents, start = fixture()
    parents["BASE"][MODELS[0]].loc[parents["BASE"][MODELS[0]].origin_index.eq(121), "reference_weight"] = np.nan
    frames, _ = trend_routed_frames(data, parents, CFG, start)
    assert pd.isna(frames["BASE"].target.iloc[121]) and frames["BASE"].selected_parent.iloc[121] == MODELS[0]


def test_shared_trend_separate_cost_and_wrong_source_rejected():
    data, parents, start = fixture()
    frames, _ = trend_routed_frames(data, parents, CFG, start)
    pd.testing.assert_series_equal(frames["BASE"].selected_parent, frames["STRESS"].selected_parent)
    assert frames["BASE"].target.iloc[121] == .8 and frames["STRESS"].target.iloc[121] == .6
    parents["STRESS"][MODELS[0]]["source_cost"] = "BASE"
    with pytest.raises(ValueError):
        trend_routed_frames(data, parents, CFG, start)


def test_future_data_and_shorter_prefix_do_not_change_past():
    data, parents, start = fixture()
    original, _ = trend_routed_frames(data, parents, CFG, start)
    changed = data.copy()
    changed.loc[130:, "wealth"] *= 2
    changed["sma120"] = changed.wealth/changed.wealth.rolling(120).mean()-1
    altered = {cost: {k: v.copy() for k, v in group.items()} for cost, group in parents.items()}
    for group in altered.values():
        for parent in group.values():
            parent.loc[parent.origin_index.ge(130), "reference_weight"] = 0
    future, _ = trend_routed_frames(changed, altered, CFG, start)
    pd.testing.assert_frame_equal(original["BASE"].iloc[:130], future["BASE"].iloc[:130])
    prefix = {cost: {k: v[v.origin_index.lt(129)].copy() for k, v in group.items()} for cost, group in parents.items()}
    short, _ = trend_routed_frames(data.iloc[:130], prefix, CFG, start)
    pd.testing.assert_frame_equal(original["BASE"].iloc[:129], short["BASE"].iloc[:129])


def test_stale_factor_wrong_clock_and_shifted_parent_rejected():
    data, parents, start = fixture()
    changed = data.copy()
    changed.loc[121, "sma120"] = 0
    with pytest.raises(AssertionError):
        checked_trend(changed)
    with pytest.raises(ValueError):
        trend_routed_frames(data, parents, {**CFG, "decision_clock": "15:00:00"}, start)
    parents["BASE"][MODELS[0]]["execution_date"] = parents["BASE"][MODELS[0]].origin
    with pytest.raises(ValueError):
        trend_routed_frames(data, parents, CFG, start)


def test_actual_switch_adjustment_exit_reentry_and_dividend():
    data, parents, start = fixture()
    for group in parents.values():
        group[MODELS[1]].loc[group[MODELS[1]].origin_index.eq(125), "reference_weight"] = 0
    data.loc[123, "dividend"] = .1
    dividends = pd.DataFrame({"record_date": [data.date.iloc[122]], "ex_date": [data.date.iloc[123]],
        "payment_date": [data.date.iloc[124]], "cash_dividend_per_share": [.1]})
    frames, _ = trend_routed_frames(data, parents, CFG, start)
    ledger, decisions = simulate_event_account(data, dividends, CFG, CFG["costs"]["BASE"], start, PRIMARY,
        targets=frames["BASE"].target.to_numpy(float), event_mask=np.ones(len(data), bool))
    by_date = ledger.set_index("date")
    assert by_date.loc[data.date.iloc[122], "filled_quantity"] > 0
    assert by_date.loc[data.date.iloc[125], "filled_quantity"] < 0 and by_date.loc[data.date.iloc[125], "shares"] > 0
    assert by_date.loc[data.date.iloc[126], "shares"] == 0 and by_date.loc[data.date.iloc[127], "filled_quantity"] > 0
    assert ledger.shares.iloc[-1] == 0 and ledger.mark_clock.iloc[-1] == "OPEN_TERMINAL"
    assert np.isclose(ledger.dividend_recognized.sum(), by_date.loc[data.date.iloc[122], "shares"]*.1)
    assert np.isclose(ledger.dividend_paid.sum(), ledger.dividend_recognized.sum())
    assert ledger.accounting_error.abs().max() < 1e-6
    assert pd.DatetimeIndex(decisions.execution_date).equals(pd.DatetimeIndex(ledger.date))
