"""核对三个固定组合的条件语义、来源时点和各自实际进退场。"""
import numpy as np
import pandas as pd
import pytest
from research.return_confirmation_auxiliary_batch_inputs_v1 import (
    PRIMARY, MODELS, CANDIDATES, FORMULAS, return_confirmation_auxiliary_batch_frames)
from research.event_clock_account_v1 import simulate_event_account
from tests.test_heikin_price_state_v1 import prices
from tests.test_runs_reference_blend_v1 import fixture as two_parents
from tests.test_trend_reference_router_v1 import CFG as OLD_CFG

CFG = {**OLD_CFG, 'candidate_models': list(CANDIDATES), 'parent_models': MODELS,
    'combination': 'FIXED_THREE_DIRECTION_GATES_CORE_PLUS_AUXILIARY_CAPPED_AT_ONE', 'direction_formulas': FORMULAS}


def fixture():
    data, parents, start = two_parents()
    lag = [1., 0., 1., 0., 1., 1., 1., 1., 0., 1., 1., 0., 1., 0., 1.]
    sign = [1., 1., 0., 1., 0., 0., 1., 0., 1., 1., 0., 1., 0., 1., 1.]
    for group in parents.values():
        for model, direction in zip(MODELS[2:], [lag, sign]):
            group[model] = group[MODELS[0]].copy()
            group[model]['source_model'] = model
            group[model]['positive_direction'] = direction
            group[model]['reference_weight'] = np.array(direction)*.25
    return data, parents, start


def test_three_hand_targets_core_retention_auxiliary_and_cap():
    data, parents, start = fixture()
    frames, summary = return_confirmation_auxiliary_batch_frames(data, parents, CFG, start)
    expected = [
        [.8, .8, 0., 0., 0., .6, 1., .6, 0., .8, .4, .4, 0., 0., .8],
        [.8, .8, .4, 0., 0., .6, 1., .6, 0., .8, .4, .4, 0., 0., .8],
        [.8, 1., .4, 0., 0., .6, 1., .6, 0., .8, .4, .4, 0., 0., .8]]
    for model, target in zip(CANDIDATES, expected):
        np.testing.assert_allclose(frames['BASE'][model+'_target'].iloc[2:-1], target, atol=1e-15)
    assert [row['full_weight_origins'] for row in summary[:3]] == [1, 1, 2]
    with pytest.raises(ValueError):
        return_confirmation_auxiliary_batch_frames(data, parents, {**CFG, 'direction_formulas': {}}, start)


def test_required_unknowns_no_boolean_shortcut_and_unused_sign_ignored():
    data, parents, start = fixture()
    group = parents['BASE']
    group[MODELS[1]].loc[group[MODELS[1]].origin_index.eq(5), 'reference_weight'] = np.nan
    group[MODELS[2]].loc[group[MODELS[2]].origin_index.eq(6), 'positive_direction'] = np.nan
    group[MODELS[3]].loc[group[MODELS[3]].origin_index.eq(7), 'positive_direction'] = np.nan
    frames, summary = return_confirmation_auxiliary_batch_frames(data, parents, CFG, start)
    for model in CANDIDATES:
        assert frames['BASE'][model+'_target'].iloc[5:7].isna().all()
    assert frames['BASE']['LAG_CONFIRMED_RUNS_AUXILIARY_target'].iloc[7] == .6
    assert pd.isna(frames['BASE'][PRIMARY+'_target'].iloc[7])
    assert pd.isna(frames['BASE']['EITHER_CONFIRMED_RUNS_AUXILIARY_target'].iloc[7])
    group[MODELS[3]].loc[0, 'positive_direction'] = .5
    with pytest.raises(ValueError):
        return_confirmation_auxiliary_batch_frames(data, parents, CFG, start)


def test_market_directions_do_not_reuse_standalone_risk_targets():
    data, parents, start = fixture()
    original, _ = return_confirmation_auxiliary_batch_frames(data, parents, CFG, start)
    for model in MODELS[2:]:
        parents['BASE'][model]['reference_weight'] = np.nan
    changed, _ = return_confirmation_auxiliary_batch_frames(data, parents, CFG, start)
    for model in CANDIDATES:
        pd.testing.assert_series_equal(original['BASE'][model+'_target'], changed['BASE'][model+'_target'])


def test_cost_identity_next_open_and_target_range():
    data, parents, start = fixture()
    parents['STRESS'][MODELS[0]]['reference_weight'] /= 2.
    frames, _ = return_confirmation_auxiliary_batch_frames(data, parents, CFG, start)
    for model in CANDIDATES:
        assert frames['BASE'][model+'_target'].iloc[2] == .8
        assert frames['STRESS'][model+'_target'].iloc[2] == .4
    parents['STRESS'][MODELS[2]]['source_cost'] = 'BASE'
    with pytest.raises(ValueError):
        return_confirmation_auxiliary_batch_frames(data, parents, CFG, start)
    data, parents, start = fixture()
    parents['BASE'][MODELS[3]]['execution_date'] = parents['BASE'][MODELS[3]].origin
    with pytest.raises(ValueError):
        return_confirmation_auxiliary_batch_frames(data, parents, CFG, start)
    data, parents, start = fixture()
    parents['BASE'][MODELS[0]].loc[0, 'reference_weight'] = 1.01
    with pytest.raises(ValueError):
        return_confirmation_auxiliary_batch_frames(data, parents, CFG, start)


def test_future_prefix_and_decision_clock():
    data, parents, start = fixture()
    original, _ = return_confirmation_auxiliary_batch_frames(data, parents, CFG, start)
    changed = {cost: {model: f.copy() for model, f in group.items()} for cost, group in parents.items()}
    for group in changed.values():
        for model, f in group.items():
            f.loc[f.origin_index.ge(12), 'reference_weight'] = 0.
            if model in MODELS[2:]:
                f.loc[f.origin_index.ge(12), 'positive_direction'] = 0.
    future, _ = return_confirmation_auxiliary_batch_frames(data, changed, CFG, start)
    pd.testing.assert_frame_equal(original['BASE'].iloc[:12], future['BASE'].iloc[:12])
    prefix = {cost: {model: f[f.origin_index.lt(11)].copy() for model, f in group.items()} for cost, group in parents.items()}
    short, _ = return_confirmation_auxiliary_batch_frames(data.iloc[:12], prefix, CFG, start)
    pd.testing.assert_frame_equal(original['BASE'].iloc[:11], short['BASE'].iloc[:11])
    expected = pd.DatetimeIndex(data.date.iloc[2:-1])+pd.Timedelta(hours=15, minutes=5)
    assert pd.DatetimeIndex(original['BASE'].decision_time.iloc[2:-1]).equals(expected)
    for model in CANDIDATES:
        assert original['BASE'][model+'_target'].iloc[:2].isna().all()
        assert pd.isna(original['BASE'][model+'_target'].iloc[-1])


def test_each_account_withdraws_auxiliary_retains_core_then_exits_and_reenters():
    data, parents, start = fixture()
    group = parents['BASE']
    group[MODELS[1]].loc[group[MODELS[1]].origin_index.eq(9), 'reference_weight'] = .4
    for model in MODELS[2:]:
        group[model].loc[group[model].origin_index.eq(9), 'positive_direction'] = 0.
    frames, _ = return_confirmation_auxiliary_batch_frames(data, parents, CFG, start)
    rights = pd.DataFrame(columns=['record_date', 'ex_date', 'payment_date', 'cash_dividend_per_share'])
    for model in CANDIDATES:
        target = frames['BASE'][model+'_target'].to_numpy(float)
        ledger, decisions = simulate_event_account(data, rights, CFG, CFG['costs']['BASE'], start, model,
            targets=target, event_mask=np.ones(len(data), bool))
        by_date = ledger.set_index('date')
        assert 0 < by_date.loc[data.date.iloc[10], 'shares'] < by_date.loc[data.date.iloc[9], 'shares']
        assert target[9] == .6 and target[10] == 0.
        assert by_date.loc[data.date.iloc[11], 'shares'] == 0
        assert by_date.loc[data.date.iloc[12], 'shares'] > 0
        assert pd.DatetimeIndex(decisions.execution_date).equals(pd.DatetimeIndex(ledger.date))
        assert ledger.accounting_error.abs().max() < 1e-6 and ledger.shares.iloc[-1] == 0


def test_each_account_unknown_hold_dividend_rights_payment_and_terminal_open():
    original, parents, start = fixture()
    raw = original.close.to_numpy().copy()
    raw[9:] -= .1
    cash = np.zeros(len(raw))
    cash[9] = .1
    data = prices(raw, cash)
    data.loc[9, 'open'] = raw[8]-.1
    parents['BASE'][MODELS[2]].loc[parents['BASE'][MODELS[2]].origin_index.eq(8), 'positive_direction'] = np.nan
    frames, _ = return_confirmation_auxiliary_batch_frames(data, parents, CFG, start)
    rights = pd.DataFrame({'record_date': [data.date.iloc[8]], 'ex_date': [data.date.iloc[9]],
        'payment_date': [data.date.iloc[11]], 'cash_dividend_per_share': [.1]})
    for model in CANDIDATES:
        target = frames['BASE'][model+'_target'].to_numpy(float)
        ledger, decisions = simulate_event_account(data, rights, CFG, CFG['costs']['BASE'], start, model,
            targets=target, event_mask=np.ones(len(data), bool))
        by_date = ledger.set_index('date')
        assert by_date.loc[data.date.iloc[8], 'shares'] > 0
        assert by_date.loc[data.date.iloc[9], 'shares'] == by_date.loc[data.date.iloc[8], 'shares']
        assert decisions.loc[decisions.origin_index.eq(8), 'requested_quantity'].iloc[0] == 0
        earned = by_date.loc[data.date.iloc[8], 'shares']*.1
        assert by_date.loc[data.date.iloc[9], 'dividend_recognized'] == pytest.approx(earned)
        assert by_date.loc[data.date.iloc[11], 'dividend_paid'] == pytest.approx(earned)
        assert ledger.shares.iloc[-1] == 0 and ledger.mark.iloc[-1] == data.open.iloc[-1]
        assert ledger.accounting_error.abs().max() < 1e-6
        assert ledger.commission.sum() > 0 and ledger.slippage_cost.sum() > 0
