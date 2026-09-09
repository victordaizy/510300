"""验证预算边界、缺失语义、费用时钟、未来隔离及实际账户。"""
import numpy as np
import pandas as pd
import pytest
from research.trend_noise_reference_blend_inputs_v1 import trend_noise_frames, market_amplitudes, PRIMARY, MODELS
from research.event_clock_account_v1 import simulate_event_account
from tests.test_trend_reference_router_v1 import CFG as PARENT_CFG

CFG = {**PARENT_CFG, "noise_window": 20, "annual_days": 242, "combination": "POSITIVE_TREND_DIVIDED_BY_TREND_PLUS_NOISE"}


def factors(data):
    data["sma120"] = data.wealth/data.wealth.rolling(120).mean()-1
    data["total_simple"] = data.wealth.pct_change(fill_method=None)
    data["vol20"] = data.total_simple.rolling(20).std(ddof=1)*np.sqrt(242)
    return data


def fixture():
    data = pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=180), "open": 10., "close": 10., "previous_close": 10.,
        "dividend": 0., "variance60": .0004, "wealth": 1.})
    data.loc[120:, "wealth"] = 1.1
    factors(data)
    indices = np.arange(119, 179)
    parents = {cost: {model: pd.DataFrame({"origin": data.date.iloc[indices].to_numpy(), "execution_date": data.date.iloc[indices+1].to_numpy(),
        "origin_index": indices, "reference_weight": (.8 if j == 0 else .2) if cost == "BASE" else (.6 if j == 0 else .1),
        "source_cost": cost, "source_model": model}) for j, model in enumerate(MODELS)} for cost in CFG["costs"]}
    return data, parents, data.date.iloc[120]


def test_known_zero_noise_and_continuous_budget_boundaries():
    data, parents, start = fixture()
    frames, _ = trend_noise_frames(data, parents, CFG, start)
    frame = frames["BASE"]
    assert frame.reference_budget131.iloc[119] == 0 and frame.target.iloc[119] == .2
    assert 0 < frame.reference_budget131.iloc[120] < 1 and .2 < frame.target.iloc[120] < .8
    assert frame.reference_budget131.iloc[145] == 1 and frame.target.iloc[145] == .8
    assert frame.target.iloc[:119].isna().all() and pd.isna(frame.target.iloc[-1])
    finite = frame.reference_budget131.dropna()
    assert finite.between(0, 1).all()


def test_missing_factor_or_zero_budget_parent_is_not_hidden():
    data, parents, start = fixture()
    parents["BASE"][MODELS[0]].loc[parents["BASE"][MODELS[0]].origin_index.eq(119), "reference_weight"] = np.nan
    data.loc[125, "total_simple"] = np.nan
    data["vol20"] = data.total_simple.rolling(20).std(ddof=1)*np.sqrt(242)
    frames, _ = trend_noise_frames(data, parents, CFG, start)
    frame = frames["BASE"]
    assert frame.reference_budget131.iloc[119] == 0 and pd.isna(frame.target.iloc[119])
    assert frame.target.iloc[125:145].isna().all()


def test_cost_mapping_factor_definition_and_execution_clock():
    data, parents, start = fixture()
    frames, _ = trend_noise_frames(data, parents, CFG, start)
    pd.testing.assert_series_equal(frames["BASE"].reference_budget131, frames["STRESS"].reference_budget131)
    assert frames["BASE"].target.iloc[120] != frames["STRESS"].target.iloc[120]
    changed = data.copy()
    changed.loc[120, "vol20"] *= 2
    with pytest.raises(AssertionError):
        market_amplitudes(changed, CFG)
    parents["STRESS"][MODELS[0]]["execution_date"] = parents["STRESS"][MODELS[0]].origin
    with pytest.raises(ValueError):
        trend_noise_frames(data, parents, CFG, start)


def test_future_market_and_target_changes_keep_previous_budgets():
    data, parents, start = fixture()
    original, _ = trend_noise_frames(data, parents, CFG, start)
    changed = data.copy()
    changed.loc[150:, "wealth"] *= 2
    factors(changed)
    altered = {cost: {model: frame.copy() for model, frame in group.items()} for cost, group in parents.items()}
    for group in altered.values():
        for frame in group.values():
            frame.loc[frame.origin_index.ge(150), "reference_weight"] = 0
    future, _ = trend_noise_frames(changed, altered, CFG, start)
    pd.testing.assert_frame_equal(original["BASE"].iloc[:150], future["BASE"].iloc[:150])
    prefix = {cost: {model: frame[frame.origin_index.lt(149)].copy() for model, frame in group.items()} for cost, group in parents.items()}
    short, _ = trend_noise_frames(data.iloc[:150], prefix, CFG, start)
    pd.testing.assert_frame_equal(original["BASE"].iloc[:149], short["BASE"].iloc[:149])


def test_actual_positive_blend_zero_exit_reentry_and_dividend():
    data, parents, start = fixture()
    for model in MODELS:
        frame = parents["BASE"][model]
        frame["reference_weight"] = .8
        frame.loc[frame.origin_index.eq(124), "reference_weight"] = 0
        frame.loc[frame.origin_index.eq(128), "reference_weight"] = .3
    data.loc[123, "dividend"] = .1
    dividends = pd.DataFrame({"record_date": [data.date.iloc[122]], "ex_date": [data.date.iloc[123]],
        "payment_date": [data.date.iloc[124]], "cash_dividend_per_share": [.1]})
    frames, _ = trend_noise_frames(data, parents, CFG, start)
    ledger, decisions = simulate_event_account(data, dividends, CFG, CFG["costs"]["BASE"], start, PRIMARY,
        targets=frames["BASE"].target.to_numpy(float), event_mask=np.ones(len(data), bool))
    by_date = ledger.set_index("date")
    assert by_date.loc[data.date.iloc[125], "shares"] == 0 and by_date.loc[data.date.iloc[126], "filled_quantity"] > 0
    assert by_date.loc[data.date.iloc[129], "filled_quantity"] < 0 and by_date.loc[data.date.iloc[129], "shares"] > 0
    assert ledger.shares.iloc[-1] == 0 and ledger.mark_clock.iloc[-1] == "OPEN_TERMINAL"
    assert np.isclose(ledger.dividend_recognized.sum(), by_date.loc[data.date.iloc[122], "shares"]*.1)
    assert np.isclose(ledger.dividend_paid.sum(), ledger.dividend_recognized.sum())
    assert ledger.accounting_error.abs().max() < 1e-6
    assert pd.DatetimeIndex(decisions.execution_date).equals(pd.DatetimeIndex(ledger.date))
