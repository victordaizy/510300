"""核对父目标等权合成、未知语义、时钟和独立实际资金账户。"""
import numpy as np
import pandas as pd
import pytest
from research.runs_reference_blend_inputs_v1 import PRIMARY, MODELS, CANDIDATES, runs_reference_blend_frames
from research.event_clock_account_v1 import simulate_event_account
from research.adaptive_allocation_v1 import Account, target_request
from tests.test_heikin_price_state_v1 import prices
from tests.test_trend_reference_router_v1 import CFG as OLD_CFG

CFG = {**OLD_CFG, 'candidate_models': list(CANDIDATES), 'parent_models': MODELS, 'blend_weights': {model: .5 for model in MODELS}}


def fixture():
    data = prices(np.full(18, 10.))
    indices = np.arange(2, len(data)-1)
    weights = [[.8, .8, 0., 0., 0., .6, .6, .6, 0., 0., .4, .4, 0., 0., .8],
        [0., .4, .4, 0., 0., 0., .4, 0., 0., .8, 0., 0., 0., 0., 0.]]
    parents = {cost: {model: pd.DataFrame({'origin': data.date.iloc[indices].to_numpy(),
        'execution_date': data.date.iloc[indices+1].to_numpy(), 'origin_index': indices,
        'reference_weight': weights[j], 'source_cost': cost, 'source_model': model})
        for j, model in enumerate(MODELS)} for cost in CFG['costs']}
    return data, parents, data.date.iloc[3]


def test_exact_half_sum_no_idle_reallocation_and_fixed_weights():
    data, parents, start = fixture()
    frames, summary = runs_reference_blend_frames(data, parents, CFG, start)
    expected = [.4, .6, .2, 0., 0., .3, .5, .3, 0., .4, .2, .2, 0., 0., .4]
    np.testing.assert_allclose(frames['BASE'][PRIMARY+'_target'].iloc[2:-1], expected, atol=1e-15)
    assert summary[0]['both_parent_positive_origins'] == 2 and summary[0]['only_runs_positive_origins'] == 2
    with pytest.raises(ValueError):
        runs_reference_blend_frames(data, parents, {**CFG, 'blend_weights': {MODELS[0]: .6, MODELS[1]: .4}}, start)


def test_any_unknown_parent_remains_unknown_even_with_other_zero():
    data, parents, start = fixture()
    parents['BASE'][MODELS[1]].loc[parents['BASE'][MODELS[1]].origin_index.eq(5), 'reference_weight'] = np.nan
    frames, summary = runs_reference_blend_frames(data, parents, CFG, start)
    assert pd.isna(frames['BASE'][PRIMARY+'_target'].iloc[5])
    assert frames['STRESS'][PRIMARY+'_target'].iloc[5] == 0.
    assert summary[0]['unknown_target_origins'] == 1
    parents['BASE'][MODELS[0]].loc[0, 'reference_weight'] = 1.01
    with pytest.raises(ValueError):
        runs_reference_blend_frames(data, parents, CFG, start)


def test_cost_sources_remain_separate_and_identity_clock_rejected():
    data, parents, start = fixture()
    parents['STRESS'][MODELS[0]]['reference_weight'] /= 2.
    frames, _ = runs_reference_blend_frames(data, parents, CFG, start)
    assert frames['BASE'][PRIMARY+'_target'].iloc[2] == .4 and frames['STRESS'][PRIMARY+'_target'].iloc[2] == .2
    parents['STRESS'][MODELS[0]]['source_cost'] = 'BASE'
    with pytest.raises(ValueError):
        runs_reference_blend_frames(data, parents, CFG, start)
    data, parents, start = fixture()
    parents['BASE'][MODELS[0]]['execution_date'] = parents['BASE'][MODELS[0]].origin
    with pytest.raises(ValueError):
        runs_reference_blend_frames(data, parents, CFG, start)


def test_future_prefix_and_known_target_clock():
    data, parents, start = fixture()
    original, _ = runs_reference_blend_frames(data, parents, CFG, start)
    changed = {cost: {model: frame.copy() for model, frame in group.items()} for cost, group in parents.items()}
    for group in changed.values():
        for frame in group.values():
            frame.loc[frame.origin_index.ge(12), 'reference_weight'] = 0.
    future, _ = runs_reference_blend_frames(data, changed, CFG, start)
    pd.testing.assert_frame_equal(original['BASE'].iloc[:12], future['BASE'].iloc[:12])
    prefix = {cost: {model: frame[frame.origin_index.lt(11)].copy() for model, frame in group.items()} for cost, group in parents.items()}
    short, _ = runs_reference_blend_frames(data.iloc[:12], prefix, CFG, start)
    pd.testing.assert_frame_equal(original['BASE'].iloc[:11], short['BASE'].iloc[:11])
    expected = pd.DatetimeIndex(data.date.iloc[2:-1])+pd.Timedelta(hours=15, minutes=5)
    assert pd.DatetimeIndex(original['BASE'].decision_time.iloc[2:-1]).equals(expected)
    assert original['BASE'][PRIMARY+'_target'].iloc[:2].isna().all() and pd.isna(original['BASE'][PRIMARY+'_target'].iloc[-1])


def test_own_account_band_initial_entry_and_known_zero_exit():
    account = Account(200000.)
    assert target_request(account, 10., .05, CFG)['requested_quantity'] == 1000
    account.cash, account.shares = 100000., 10000
    assert target_request(account, 10., .55, CFG)['requested_quantity'] == 0
    assert target_request(account, 10., .7, CFG)['requested_quantity'] == 4000
    assert target_request(account, 10., 0., CFG)['requested_quantity'] == -10000
    smaller = Account(100000.)
    assert target_request(smaller, 10., .05, CFG)['requested_quantity'] == 500


def test_actual_next_open_one_parent_exit_unknown_hold_reentry_dividend():
    original, parents, start = fixture()
    raw = original.close.to_numpy().copy(); raw[9:] -= .1
    cash = np.zeros(len(raw)); cash[9] = .1
    data = prices(raw, cash); data.loc[9, 'open'] = raw[8]-.1
    parents['BASE'][MODELS[0]].loc[parents['BASE'][MODELS[0]].origin_index.eq(8), 'reference_weight'] = np.nan
    frames, _ = runs_reference_blend_frames(data, parents, CFG, start)
    target = frames['BASE'][PRIMARY+'_target'].to_numpy(float)
    rights = pd.DataFrame({'record_date': [data.date.iloc[8]], 'ex_date': [data.date.iloc[9]],
        'payment_date': [data.date.iloc[11]], 'cash_dividend_per_share': [.1]})
    ledger, decisions = simulate_event_account(data, rights, CFG, CFG['costs']['BASE'], start, PRIMARY,
        targets=target, event_mask=np.ones(len(data), bool))
    by_date = ledger.set_index('date')
    assert by_date.loc[data.date.iloc[5], 'shares'] > 0 and by_date.loc[data.date.iloc[6], 'shares'] == 0
    assert by_date.loc[data.date.iloc[8], 'shares'] > 0 and by_date.loc[data.date.iloc[9], 'shares'] == by_date.loc[data.date.iloc[8], 'shares']
    assert decisions.loc[decisions.origin_index.eq(8), 'requested_quantity'].iloc[0] == 0
    earned = by_date.loc[data.date.iloc[8], 'shares']*.1
    assert earned > 0 and by_date.loc[data.date.iloc[9], 'dividend_recognized'] == pytest.approx(earned)
    assert by_date.loc[data.date.iloc[11], 'dividend_paid'] == pytest.approx(earned)
    zero = np.flatnonzero((target == 0.) & np.r_[False, target[:-1] > 0.])
    assert len(zero) > 0 and all(by_date.loc[data.date.iloc[t+1], 'shares'] == 0 for t in zero)
    assert ledger.shares.iloc[-1] == 0 and ledger.mark.iloc[-1] == data.open.iloc[-1]
    assert ledger.accounting_error.abs().max() < 1e-6 and ledger.commission.sum() > 0 and ledger.slippage_cost.sum() > 0
    assert pd.DatetimeIndex(decisions.execution_date).equals(pd.DatetimeIndex(ledger.date))
