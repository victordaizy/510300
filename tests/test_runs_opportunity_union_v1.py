"""核对两个固定进场合并目标、时点来源及两个真实资金账户。"""
import numpy as np
import pandas as pd
import pytest
from research.runs_opportunity_union_inputs_v1 import PRIMARY, MODELS, CANDIDATES, runs_opportunity_union_frames
from research.event_clock_account_v1 import simulate_event_account
from research.adaptive_allocation_v1 import Account, target_request
from tests.test_heikin_price_state_v1 import prices
from tests.test_runs_reference_blend_v1 import fixture
from tests.test_trend_reference_router_v1 import CFG as OLD_CFG

CFG = {**OLD_CFG, 'candidate_models': list(CANDIDATES), 'parent_models': MODELS,
    'combination': 'MAXIMUM_AND_CAPPED_SUM_OF_SAVED_PARENT_TARGETS'}


def test_hand_maximum_capped_sum_single_source_and_rule_identity():
    data, parents, start = fixture()
    frames, summary = runs_opportunity_union_frames(data, parents, CFG, start)
    expected = [.8, .8, .4, 0., 0., .6, .6, .6, 0., .8, .4, .4, 0., 0., .8]
    capped = [.8, 1., .4, 0., 0., .6, 1., .6, 0., .8, .4, .4, 0., 0., .8]
    np.testing.assert_allclose(frames['BASE'][PRIMARY+'_target'].iloc[2:-1], expected, atol=1e-15)
    np.testing.assert_allclose(frames['BASE']['RUNS_OPPORTUNITY_CAPPED_SUM_target'].iloc[2:-1], capped, atol=1e-15)
    assert summary[0]['both_parent_positive_origins'] == 2 and summary[1]['full_weight_origins'] == 2
    with pytest.raises(ValueError):
        runs_opportunity_union_frames(data, parents, {**CFG, 'combination': 'FIXED_HALF'}, start)


def test_unknown_never_hidden_even_with_other_full_weight_or_zero():
    data, parents, start = fixture()
    for origin, other in [(5, 0.), (6, 1.)]:
        parents['BASE'][MODELS[0]].loc[parents['BASE'][MODELS[0]].origin_index.eq(origin), 'reference_weight'] = other
        parents['BASE'][MODELS[1]].loc[parents['BASE'][MODELS[1]].origin_index.eq(origin), 'reference_weight'] = np.nan
    frames, summary = runs_opportunity_union_frames(data, parents, CFG, start)
    for model in CANDIDATES:
        assert frames['BASE'][model+'_target'].iloc[5:7].isna().all()
    assert all(r['unknown_target_origins'] == 2 for r in summary if r['cost'] == 'BASE')
    parents['BASE'][MODELS[0]].loc[0, 'reference_weight'] = 1.01
    with pytest.raises(ValueError):
        runs_opportunity_union_frames(data, parents, CFG, start)


def test_cost_sources_identity_and_execution_date():
    data, parents, start = fixture()
    parents['STRESS'][MODELS[0]]['reference_weight'] /= 2.
    frames, _ = runs_opportunity_union_frames(data, parents, CFG, start)
    for model in CANDIDATES:
        assert frames['BASE'][model+'_target'].iloc[2] == .8 and frames['STRESS'][model+'_target'].iloc[2] == .4
    parents['STRESS'][MODELS[0]]['source_cost'] = 'BASE'
    with pytest.raises(ValueError):
        runs_opportunity_union_frames(data, parents, CFG, start)
    data, parents, start = fixture()
    parents['BASE'][MODELS[0]]['execution_date'] = parents['BASE'][MODELS[0]].origin
    with pytest.raises(ValueError):
        runs_opportunity_union_frames(data, parents, CFG, start)


def test_future_prefix_and_decision_clock():
    data, parents, start = fixture()
    original, _ = runs_opportunity_union_frames(data, parents, CFG, start)
    changed = {cost: {model: f.copy() for model, f in group.items()} for cost, group in parents.items()}
    for group in changed.values():
        for f in group.values():
            f.loc[f.origin_index.ge(12), 'reference_weight'] = 0.
    future, _ = runs_opportunity_union_frames(data, changed, CFG, start)
    pd.testing.assert_frame_equal(original['BASE'].iloc[:12], future['BASE'].iloc[:12])
    prefix = {cost: {model: f[f.origin_index.lt(11)].copy() for model, f in group.items()} for cost, group in parents.items()}
    short, _ = runs_opportunity_union_frames(data.iloc[:12], prefix, CFG, start)
    pd.testing.assert_frame_equal(original['BASE'].iloc[:11], short['BASE'].iloc[:11])
    expected = pd.DatetimeIndex(data.date.iloc[2:-1])+pd.Timedelta(hours=15, minutes=5)
    assert pd.DatetimeIndex(original['BASE'].decision_time.iloc[2:-1]).equals(expected)
    for model in CANDIDATES:
        assert original['BASE'][model+'_target'].iloc[:2].isna().all() and pd.isna(original['BASE'][model+'_target'].iloc[-1])


def test_own_funds_band_initial_entry_and_positive_rounding_exit():
    account = Account(200000.)
    assert target_request(account, 10., .05, CFG)['requested_quantity'] == 1000
    assert target_request(account, 10., .0001, CFG)['requested_quantity'] == 0
    account.cash, account.shares = 100000., 10000
    assert target_request(account, 10., .55, CFG)['requested_quantity'] == 0
    assert target_request(account, 10., .7, CFG)['requested_quantity'] == 4000
    assert target_request(account, 10., .0001, CFG)['requested_quantity'] == -10000
    assert target_request(account, 10., 0., CFG)['requested_quantity'] == -10000
    account.cash, account.shares = 199000., 100
    assert target_request(account, 10., .0001, CFG)['requested_quantity'] == 0
    assert target_request(account, 10., 0., CFG)['requested_quantity'] == -100


def test_both_actual_accounts_next_open_unknown_reentry_dividend_and_terminal():
    original, parents, start = fixture()
    raw = original.close.to_numpy().copy(); raw[9:] -= .1
    cash = np.zeros(len(raw)); cash[9] = .1
    data = prices(raw, cash); data.loc[9, 'open'] = raw[8]-.1
    parents['BASE'][MODELS[0]].loc[parents['BASE'][MODELS[0]].origin_index.eq(8), 'reference_weight'] = np.nan
    frames, _ = runs_opportunity_union_frames(data, parents, CFG, start)
    rights = pd.DataFrame({'record_date': [data.date.iloc[8]], 'ex_date': [data.date.iloc[9]],
        'payment_date': [data.date.iloc[11]], 'cash_dividend_per_share': [.1]})
    for model in CANDIDATES:
        target = frames['BASE'][model+'_target'].to_numpy(float)
        ledger, decisions = simulate_event_account(data, rights, CFG, CFG['costs']['BASE'], start, model,
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
