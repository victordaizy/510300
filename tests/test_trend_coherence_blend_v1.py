"""验证上升连贯程度、缺失、费用时钟、未来隔离及完整账户。"""
import numpy as np
import pandas as pd
import pytest
from research.trend_coherence_blend_inputs_v1 import rolling_trend_coherence, trend_coherence_frames, MODELS, PRIMARY
from research.event_clock_account_v1 import simulate_event_account
from tests.test_trend_noise_reference_blend_v1 import fixture as old_fixture, CFG as OLD_CFG

CFG = {**OLD_CFG, "combination": "POSITIVE_LOG_TREND_R_SQUARED_BUDGET"}


def fixture():
    data, old, start = old_fixture()
    parents = {cost: {new: group[old_model].copy().assign(source_model=new) for new, old_model in zip(MODELS, group)} for cost, group in old.items()}
    return data, parents, start


def test_flat_straight_rise_fall_and_noisy_upward_windows():
    data, parents, start = fixture()
    data["wealth"] = 1.
    flat, _ = trend_coherence_frames(data, parents, CFG, start)
    assert flat["BASE"].reference_budget131.iloc[119] == 0 and flat["BASE"].target.iloc[119] == .2
    for direction in [1, -1]:
        data["wealth"] = np.exp(direction*.001*np.arange(len(data)))
        frames, _ = trend_coherence_frames(data, parents, CFG, start)
        f = frames["BASE"]
        assert np.allclose(f.trend_r_squared120.iloc[119:], 1.)
        assert np.allclose(f.log_wealth_slope120.iloc[119:], direction*.001)
        assert np.allclose(f.reference_budget131.iloc[119:], 1 if direction == 1 else 0)
    data["wealth"] = np.exp(.001*np.arange(len(data))+.01*np.sin(np.arange(len(data))))
    slopes, quality = rolling_trend_coherence(data, CFG)
    assert (slopes[119:] > 0).all() and ((quality[119:] > 0) & (quality[119:] < 1)).all()


def test_missing_full_window_and_zero_budget_parent_are_not_hidden():
    data, parents, start = fixture()
    parents["BASE"][MODELS[0]].loc[0, "reference_weight"] = np.nan
    data.loc[125, "wealth"] = np.nan
    frames, _ = trend_coherence_frames(data, parents, CFG, start)
    assert pd.isna(frames["BASE"].target.iloc[119])
    assert frames["BASE"].target.iloc[125:].isna().all()
    data.loc[125, "wealth"] = 0
    with pytest.raises(ValueError):
        rolling_trend_coherence(data, CFG)


def test_shared_market_budget_separate_cost_and_wrong_clock_rejected():
    data, parents, start = fixture()
    frames, _ = trend_coherence_frames(data, parents, CFG, start)
    pd.testing.assert_series_equal(frames["BASE"].reference_budget131, frames["STRESS"].reference_budget131)
    assert frames["BASE"].target.iloc[120] != frames["STRESS"].target.iloc[120]
    parents["BASE"][MODELS[0]]["execution_date"] = parents["BASE"][MODELS[0]].origin
    with pytest.raises(ValueError):
        trend_coherence_frames(data, parents, CFG, start)


def test_future_wealth_and_target_changes_and_prefix_isolation():
    data, parents, start = fixture()
    original, _ = trend_coherence_frames(data, parents, CFG, start)
    changed = data.copy()
    changed.loc[150:, "wealth"] *= 3
    altered = {cost: {model: frame.copy() for model, frame in group.items()} for cost, group in parents.items()}
    for group in altered.values():
        for frame in group.values():
            frame.loc[frame.origin_index.ge(150), "reference_weight"] = 0
    future, _ = trend_coherence_frames(changed, altered, CFG, start)
    pd.testing.assert_frame_equal(original["BASE"].iloc[:150], future["BASE"].iloc[:150])
    prefix = {cost: {model: frame[frame.origin_index.lt(149)].copy() for model, frame in group.items()} for cost, group in parents.items()}
    short, _ = trend_coherence_frames(data.iloc[:150], prefix, CFG, start)
    pd.testing.assert_frame_equal(original["BASE"].iloc[:149], short["BASE"].iloc[:149])


def test_actual_exit_partial_adjustment_reentry_and_dividend():
    data, parents, start = fixture()
    for model in MODELS:
        frame = parents["BASE"][model]
        frame["reference_weight"] = .8
        frame.loc[frame.origin_index.eq(124), "reference_weight"] = 0
        frame.loc[frame.origin_index.eq(128), "reference_weight"] = .3
    data.loc[123, "dividend"] = .1
    dividends = pd.DataFrame({"record_date": [data.date.iloc[122]], "ex_date": [data.date.iloc[123]],
        "payment_date": [data.date.iloc[124]], "cash_dividend_per_share": [.1]})
    frames, _ = trend_coherence_frames(data, parents, CFG, start)
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
