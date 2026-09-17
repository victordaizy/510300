"""核对核心保留、辅助方向、未知与费用时点以及实际资金进退场。"""
import numpy as np
import pandas as pd
import pytest
from research.sign_confirmed_runs_auxiliary_inputs_v1 import PRIMARY, MODELS, CANDIDATES, sign_confirmed_runs_auxiliary_frames
from research.event_clock_account_v1 import simulate_event_account
from research.adaptive_allocation_v1 import Account, target_request
from tests.test_heikin_price_state_v1 import prices
from tests.test_runs_reference_blend_v1 import fixture as two_parents
from tests.test_trend_reference_router_v1 import CFG as OLD_CFG

CFG = {**OLD_CFG, 'candidate_models': list(CANDIDATES), 'parent_models': MODELS,
    'combination': 'CORE_PLUS_SIGN_ALLOWED_AUXILIARY_CAPPED_AT_ONE', 'direction_column': 'positive_direction'}


def fixture():
    data, parents, start = two_parents()
    direction = [1., 1., 0., 1., 0., 0., 1., 0., 1., 1., 0., 1., 0., 1., 1.]
    for cost, group in parents.items():
        group[MODELS[2]] = group[MODELS[0]].copy()
        group[MODELS[2]]['source_model'] = MODELS[2]
        group[MODELS[2]]['positive_direction'] = direction
        group[MODELS[2]]['reference_weight'] = np.array(direction)*.25
    return data, parents, start


def test_hand_core_auxiliary_direction_and_cap():
    data, parents, start = fixture()
    frames, summary = sign_confirmed_runs_auxiliary_frames(data, parents, CFG, start)
    expected = [.8, 1., 0., 0., 0., .6, 1., .6, 0., .8, .4, .4, 0., 0., .8]
    np.testing.assert_allclose(frames['BASE'][PRIMARY+'_target'].iloc[2:-1], expected, atol=1e-15)
    assert summary[0]['auxiliary_rejected_positive_origins'] == 1
    assert summary[0]['core_kept_while_direction_disallowed_origins'] == 3
    assert summary[0]['admitted_auxiliary_only_positive_origins'] == 1
    assert summary[0]['full_weight_origins'] == 2
    with pytest.raises(ValueError):
        sign_confirmed_runs_auxiliary_frames(data, parents, {**CFG, 'combination': 'UNCONDITIONAL_SUM'}, start)


def test_unknown_unused_auxiliary_and_direction_never_hidden():
    data, parents, start = fixture()
    group = parents['BASE']
    group[MODELS[1]].loc[group[MODELS[1]].origin_index.eq(5), 'reference_weight'] = np.nan
    group[MODELS[2]].loc[group[MODELS[2]].origin_index.eq(5), 'positive_direction'] = 0.
    group[MODELS[0]].loc[group[MODELS[0]].origin_index.eq(6), 'reference_weight'] = 1.
    group[MODELS[1]].loc[group[MODELS[1]].origin_index.eq(6), 'reference_weight'] = np.nan
    group[MODELS[2]].loc[group[MODELS[2]].origin_index.eq(7), 'positive_direction'] = np.nan
    frames, summary = sign_confirmed_runs_auxiliary_frames(data, parents, CFG, start)
    assert frames['BASE'][PRIMARY+'_target'].iloc[5:8].isna().all()
    assert summary[0]['unknown_target_origins'] == 3
    group[MODELS[2]].loc[0, 'positive_direction'] = .5
    with pytest.raises(ValueError):
        sign_confirmed_runs_auxiliary_frames(data, parents, CFG, start)


def test_direction_independent_of_standalone_volatility_target():
    data, parents, start = fixture()
    original, _ = sign_confirmed_runs_auxiliary_frames(data, parents, CFG, start)
    parents['BASE'][MODELS[2]]['reference_weight'] = np.nan
    changed, _ = sign_confirmed_runs_auxiliary_frames(data, parents, CFG, start)
    pd.testing.assert_series_equal(original['BASE'][PRIMARY+'_target'], changed['BASE'][PRIMARY+'_target'])
    assert changed['BASE']['auxiliary_direction'].iloc[3] == 1.
    assert changed['BASE'][PRIMARY+'_target'].iloc[3] == 1.


def test_cost_identity_next_open_and_invalid_target_rejected():
    data, parents, start = fixture()
    parents['STRESS'][MODELS[0]]['reference_weight'] /= 2.
    frames, _ = sign_confirmed_runs_auxiliary_frames(data, parents, CFG, start)
    assert frames['BASE'][PRIMARY+'_target'].iloc[2] == .8
    assert frames['STRESS'][PRIMARY+'_target'].iloc[2] == .4
    parents['STRESS'][MODELS[2]]['source_cost'] = 'BASE'
    with pytest.raises(ValueError):
        sign_confirmed_runs_auxiliary_frames(data, parents, CFG, start)
    data, parents, start = fixture()
    parents['BASE'][MODELS[2]]['execution_date'] = parents['BASE'][MODELS[2]].origin
    with pytest.raises(ValueError):
        sign_confirmed_runs_auxiliary_frames(data, parents, CFG, start)
    data, parents, start = fixture()
    parents['BASE'][MODELS[0]].loc[0, 'reference_weight'] = 1.01
    with pytest.raises(ValueError):
        sign_confirmed_runs_auxiliary_frames(data, parents, CFG, start)


def test_future_prefix_and_full_decision_clock():
    data, parents, start = fixture()
    original, _ = sign_confirmed_runs_auxiliary_frames(data, parents, CFG, start)
    changed = {cost: {model: f.copy() for model, f in group.items()} for cost, group in parents.items()}
    for group in changed.values():
        for model, f in group.items():
            f.loc[f.origin_index.ge(12), 'reference_weight'] = 0.
            if model == MODELS[2]:
                f.loc[f.origin_index.ge(12), 'positive_direction'] = 0.
    future, _ = sign_confirmed_runs_auxiliary_frames(data, changed, CFG, start)
    pd.testing.assert_frame_equal(original['BASE'].iloc[:12], future['BASE'].iloc[:12])
    prefix = {cost: {model: f[f.origin_index.lt(11)].copy() for model, f in group.items()} for cost, group in parents.items()}
    short, _ = sign_confirmed_runs_auxiliary_frames(data.iloc[:12], prefix, CFG, start)
    pd.testing.assert_frame_equal(original['BASE'].iloc[:11], short['BASE'].iloc[:11])
    expected = pd.DatetimeIndex(data.date.iloc[2:-1])+pd.Timedelta(hours=15, minutes=5)
    assert pd.DatetimeIndex(original['BASE'].decision_time.iloc[2:-1]).equals(expected)
    assert original['BASE'][PRIMARY+'_target'].iloc[:2].isna().all()
    assert pd.isna(original['BASE'][PRIMARY+'_target'].iloc[-1])


def test_own_funds_band_and_positive_rounding_exit():
    account = Account(200000.)
    assert target_request(account, 10., .05, CFG)['requested_quantity'] == 1000
    assert target_request(account, 10., .0001, CFG)['requested_quantity'] == 0
    account.cash, account.shares = 100000., 10000
    assert target_request(account, 10., .55, CFG)['requested_quantity'] == 0
    assert target_request(account, 10., .7, CFG)['requested_quantity'] == 4000
    assert target_request(account, 10., .0001, CFG)['requested_quantity'] == -10000
    account.cash, account.shares = 199000., 100
    assert target_request(account, 10., .0001, CFG)['requested_quantity'] == 0
    assert target_request(account, 10., 0., CFG)['requested_quantity'] == -100


def test_actual_core_retained_auxiliary_removed_unknown_dividend_and_terminal():
    original, parents, start = fixture()
    raw = original.close.to_numpy().copy()
    raw[9:] -= .1
    cash = np.zeros(len(raw))
    cash[9] = .1
    data = prices(raw, cash)
    data.loc[9, 'open'] = raw[8]-.1
    parents['BASE'][MODELS[2]].loc[parents['BASE'][MODELS[2]].origin_index.eq(8), 'positive_direction'] = np.nan
    frames, _ = sign_confirmed_runs_auxiliary_frames(data, parents, CFG, start)
    rights = pd.DataFrame({'record_date': [data.date.iloc[8]], 'ex_date': [data.date.iloc[9]],
        'payment_date': [data.date.iloc[11]], 'cash_dividend_per_share': [.1]})
    target = frames['BASE'][PRIMARY+'_target'].to_numpy(float)
    ledger, decisions = simulate_event_account(data, rights, CFG, CFG['costs']['BASE'], start, PRIMARY,
        targets=target, event_mask=np.ones(len(data), bool))
    by_date = ledger.set_index('date')
    assert by_date.loc[data.date.iloc[3], 'shares'] > 0
    assert by_date.loc[data.date.iloc[5], 'shares'] == 0
    assert by_date.loc[data.date.iloc[8], 'shares'] > 0
    assert by_date.loc[data.date.iloc[9], 'shares'] == by_date.loc[data.date.iloc[8], 'shares']
    assert decisions.loc[decisions.origin_index.eq(8), 'requested_quantity'].iloc[0] == 0
    assert by_date.loc[data.date.iloc[10], 'shares'] > 0
    assert by_date.loc[data.date.iloc[11], 'shares'] == 0
    assert by_date.loc[data.date.iloc[12], 'shares'] > 0
    earned = by_date.loc[data.date.iloc[8], 'shares']*.1
    assert earned > 0 and by_date.loc[data.date.iloc[9], 'dividend_recognized'] == pytest.approx(earned)
    assert by_date.loc[data.date.iloc[11], 'dividend_paid'] == pytest.approx(earned)
    zero = np.flatnonzero((target == 0.) & np.r_[False, target[:-1] > 0.])
    assert all(by_date.loc[data.date.iloc[t+1], 'shares'] == 0 for t in zero)
    assert ledger.shares.iloc[-1] == 0 and ledger.mark.iloc[-1] == data.open.iloc[-1]
    assert ledger.accounting_error.abs().max() < 1e-6
    assert ledger.commission.sum() > 0 and ledger.slippage_cost.sum() > 0
    assert pd.DatetimeIndex(decisions.execution_date).equals(pd.DatetimeIndex(ledger.date))
