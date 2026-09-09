"""验证信号来源固定、零目标释放、未知与实际交易的时点边界。"""
import numpy as np
import pandas as pd
import pytest
from research.episode_trend_reference_inputs_v1 import episode_routed_frames, MODELS, PRIMARY
from research.event_clock_account_v1 import simulate_event_account
from tests.test_trend_reference_router_v1 import fixture, CFG as PARENT_CFG

CFG = {**PARENT_CFG, "source_commitment": "UNTIL_SELECTED_PARENT_EXPLICIT_ZERO"}


def set_target(parents, cost, model, origin, value):
    frame = parents[cost][model]
    frame.loc[frame.origin_index.eq(origin), "reference_weight"] = value


def test_lock_ignores_later_trend_and_zero_releases_only_for_next_origin():
    data, parents, start = fixture()
    for cost in parents:
        for t in [124, 125, 126]:
            set_target(parents, cost, MODELS[1], t, 0)
    frames, summary = episode_routed_frames(data, parents, CFG, start)
    frame = frames["BASE"]
    assert frame.selected_parent.iloc[121] == MODELS[1] and frame.daily_trend_choice.iloc[121] == MODELS[0]
    assert frame.target.iloc[124] == 0 and frame.episode_released.iloc[124] and pd.isna(frame.locked_parent_after.iloc[124])
    assert frame.episode_started.iloc[127] and frame.selected_parent.iloc[127] == MODELS[0]
    assert summary[0]["signal_episodes_started"] == 2 and summary[0]["signal_episodes_released_by_zero"] == 1
    assert frame.target.iloc[:119].isna().all() and pd.isna(frame.target.iloc[-1])


def test_zero_and_unknown_do_not_start_an_episode():
    data, parents, start = fixture()
    for model, t, value in [(MODELS[1], 119, np.nan), (MODELS[1], 120, 0), (MODELS[0], 121, 0), (MODELS[0], 122, np.nan)]:
        set_target(parents, "BASE", model, t, value)
    frames, _ = episode_routed_frames(data, parents, CFG, start)
    frame = frames["BASE"]
    assert frame.locked_parent_after.iloc[119:123].isna().all()
    assert pd.isna(frame.target.iloc[119]) and pd.isna(frame.target.iloc[122])
    assert frame.episode_started.iloc[123] and frame.selected_parent.iloc[123] == MODELS[0]


def test_locked_unknown_parent_is_preserved_and_trend_unknown_only_blocks_new_selection():
    data, parents, start = fixture()
    data.loc[124, "wealth"] = np.nan
    data["sma120"] = data.wealth/data.wealth.rolling(120).mean()-1
    set_target(parents, "BASE", MODELS[1], 125, np.nan)
    set_target(parents, "BASE", MODELS[1], 127, 0)
    frames, _ = episode_routed_frames(data, parents, CFG, start)
    frame = frames["BASE"]
    assert pd.isna(frame.trend_deviation120.iloc[124]) and frame.target.iloc[124] == .2
    assert pd.isna(frame.target.iloc[125]) and frame.locked_parent_after.iloc[125] == MODELS[1]
    assert frame.target.iloc[126] == .2 and frame.episode_released.iloc[127]
    assert frame.target.iloc[128:-1].isna().all() and frame.selected_parent.iloc[128:-1].isna().all()


def test_cost_specific_parent_exits_can_create_different_locks():
    data, parents, start = fixture()
    set_target(parents, "BASE", MODELS[1], 126, 0)
    frames, _ = episode_routed_frames(data, parents, CFG, start)
    assert frames["BASE"].selected_parent.iloc[127] == MODELS[0]
    assert frames["STRESS"].selected_parent.iloc[127] == MODELS[1]
    assert frames["BASE"].target.iloc[127] == .8 and frames["STRESS"].target.iloc[127] == .1
    parents["STRESS"][MODELS[0]]["source_cost"] = "BASE"
    with pytest.raises(ValueError):
        episode_routed_frames(data, parents, CFG, start)


def test_future_changes_and_shorter_prefix_keep_prior_episode_decisions():
    data, parents, start = fixture()
    original, _ = episode_routed_frames(data, parents, CFG, start)
    changed = data.copy()
    changed.loc[130:, "wealth"] *= 2
    changed["sma120"] = changed.wealth/changed.wealth.rolling(120).mean()-1
    altered = {cost: {model: frame.copy() for model, frame in group.items()} for cost, group in parents.items()}
    for group in altered.values():
        for frame in group.values():
            frame.loc[frame.origin_index.ge(130), "reference_weight"] = 0
    future, _ = episode_routed_frames(changed, altered, CFG, start)
    pd.testing.assert_frame_equal(original["BASE"].iloc[:130], future["BASE"].iloc[:130])
    prefix = {cost: {model: frame[frame.origin_index.lt(129)].copy() for model, frame in group.items()} for cost, group in parents.items()}
    short, _ = episode_routed_frames(data.iloc[:130], prefix, CFG, start)
    pd.testing.assert_frame_equal(original["BASE"].iloc[:129], short["BASE"].iloc[:129])


def test_actual_exit_reentry_partial_adjustment_and_dividend():
    data, parents, start = fixture()
    for t in [124, 127]:
        set_target(parents, "BASE", MODELS[1], t, 0)
    set_target(parents, "BASE", MODELS[0], 131, .5)
    data.loc[123, "dividend"] = .1
    dividends = pd.DataFrame({"record_date": [data.date.iloc[122]], "ex_date": [data.date.iloc[123]],
        "payment_date": [data.date.iloc[124]], "cash_dividend_per_share": [.1]})
    frames, _ = episode_routed_frames(data, parents, CFG, start)
    ledger, decisions = simulate_event_account(data, dividends, CFG, CFG["costs"]["BASE"], start, PRIMARY,
        targets=frames["BASE"].target.to_numpy(float), event_mask=np.ones(len(data), bool))
    by_date = ledger.set_index("date")
    assert by_date.loc[data.date.iloc[125], "shares"] == 0 and by_date.loc[data.date.iloc[126], "filled_quantity"] > 0
    assert by_date.loc[data.date.iloc[128], "shares"] == 0 and by_date.loc[data.date.iloc[129], "filled_quantity"] > 0
    assert by_date.loc[data.date.iloc[132], "filled_quantity"] < 0 and by_date.loc[data.date.iloc[132], "shares"] > 0
    assert ledger.shares.iloc[-1] == 0 and ledger.mark_clock.iloc[-1] == "OPEN_TERMINAL"
    assert np.isclose(ledger.dividend_recognized.sum(), by_date.loc[data.date.iloc[122], "shares"]*.1)
    assert np.isclose(ledger.dividend_paid.sum(), ledger.dividend_recognized.sum())
    assert ledger.accounting_error.abs().max() < 1e-6
    assert pd.DatetimeIndex(decisions.execution_date).equals(pd.DatetimeIndex(ledger.date))
