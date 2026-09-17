"""集中检查门槛边界与两类退出，不重测旧因素生成。"""
import copy

import numpy as np
import pandas as pd
import pytest

from research.adaptive_allocation_v1 import Account
from research.event_clock_account_v1 import simulate_event_account
from research.finite_rebalance_band_batch_inputs_v1 import CANDIDATES, MODELS, SETTINGS, band_frames, band_request, simulate_band_account
from tests.test_two_close_zero_exit_v1 import fixture


def settings():
    cfg = fixture()[2]
    cfg.update(candidate_models=list(CANDIDATES), candidate_settings=copy.deepcopy(SETTINGS), decision_clock='15:05:00',
        costs={'BASE': fixture()[3], 'STRESS': {'commission': .0004, 'minimum': 5., 'slippage': .001}})
    return cfg


def sources():
    data = fixture()[0]
    parents = {}
    for cost in ['BASE', 'STRESS']:
        parents[cost] = {}
        for j, model in enumerate(MODELS):
            values = np.array([.3, 0., 0., .4, np.nan, .5, .2, 0., 0.])+j*.01
            parents[cost][model] = pd.DataFrame({'origin': data.date.iloc[:-1].to_numpy(),
                'execution_date': data.date.iloc[1:].to_numpy(), 'origin_index': np.arange(len(data)-1),
                'reference_weight': values, 'source_cost': cost, 'source_model': model})
    return data, parents


def test_fixed_eight_settings_parent_identity_and_cost_alignment():
    data, parents = sources()
    cfg = settings()
    frames, summaries = band_frames(data, parents, cfg, str(data.date.iloc[1].date()))
    assert len(SETTINGS) == 8 and len(summaries) == 16
    for model, setting in SETTINGS.items():
        assert setting['band'] != .1
        for cost in cfg['costs']:
            np.testing.assert_allclose(frames[cost][model+'_target'].iloc[:-1],
                parents[cost][setting['parent']].reference_weight, atol=0, rtol=0, equal_nan=True)
    bad = copy.deepcopy(parents)
    bad['BASE'][MODELS[0]]['source_cost'] = 'STRESS'
    with pytest.raises(ValueError, match='费用'):
        band_frames(data, bad, cfg, str(data.date.iloc[1].date()))
    cfg['candidate_settings']['INTENT_MIX_BAND_15']['band'] = .16
    with pytest.raises(ValueError, match='门槛'):
        band_frames(data, parents, cfg, str(data.date.iloc[1].date()))


def test_band_boundaries_are_absolute_and_zero_band_rebalances():
    cfg = settings()
    for model, setting in SETTINGS.items():
        band = setting['band']
        if band:
            account = Account(100000.-2*band*100000.)
            account.shares = int(2*band*100000/10)
            assert band_request(account, 10., 1.5*band, cfg, model, 0)['requested_quantity'] == 0
            actual = account.shares*10/account.value(10.)
            assert actual-band == band
            assert band_request(account, 10., band, cfg, model, 0)['requested_quantity'] < 0
        else:
            account = Account(70000.)
            account.shares = 3000
            assert band_request(account, 10., .31, cfg, model, 0)['requested_quantity'] == 100
            assert band_request(account, 10., .3, cfg, model, 0)['requested_quantity'] == 0


def test_initial_entry_bypasses_band_and_two_exit_semantics_differ():
    cfg = settings()
    for model, setting in SETTINGS.items():
        account = Account(100000.)
        assert band_request(account, 10., .05, cfg, model, 0)['requested_quantity'] == 500
        account.cash, account.shares = 70000., 3000
        first = band_request(account, 10., 0., cfg, model, 1)
        assert first['requested_quantity'] == (0 if setting['zero_confirmations'] == 2 else -3000)
        assert band_request(account, 10., 0., cfg, model, 2)['requested_quantity'] == -3000


def test_unknown_breaks_confirmation_and_new_positive_can_reenter():
    data, div, _, cost = fixture()
    cfg = settings()
    targets = np.array([.3, 0., np.nan, 0., 0., .4, .4, 0., 0., 0.])
    for model in ['ZERO_CONFIRM_BAND_20', 'INTENT_MIX_BAND_20']:
        ledger, decisions = simulate_band_account(data, div, cfg, cost, str(data.date.iloc[1].date()), model,
            targets=targets, event_mask=np.ones(len(data), bool))
        if model.startswith('ZERO_CONFIRM'):
            assert ledger.shares.iloc[:4].eq(3000).all() and ledger.shares.iloc[4] == 0
            assert decisions.used_zero_count.iloc[3] == 1
        else:
            assert ledger.shares.iloc[1:5].eq(0).all()
        assert ledger.shares.iloc[5] > 0 and ledger.shares.iloc[-1] == 0


def test_immediate_exit_wrapper_matches_normal_engine_at_its_own_band():
    data, div, _, cost = fixture()
    cfg = settings()
    targets = np.array([.3, .38, .55, .55, 0., .8, .7, np.nan, .2, 0.])
    model = 'INTENT_MIX_BAND_15'
    kwargs = {'targets': targets, 'event_mask': np.ones(len(data), bool)}
    ledger, decisions = simulate_band_account(data, div, cfg, cost, str(data.date.iloc[1].date()), model, **kwargs)
    ordinary, original = simulate_event_account(data, div, {**cfg, 'weight_band': .15}, cost, str(data.date.iloc[1].date()), model, **kwargs)
    pd.testing.assert_frame_equal(ledger, ordinary)
    pd.testing.assert_frame_equal(decisions[original.columns], original)


def test_future_source_and_market_changes_leave_prior_accounts_unchanged():
    data, div, _, cost = fixture()
    cfg = settings()
    targets = np.array([.3, 0., np.nan, .4, .3, .4, .7, .2, 0., 0.])
    for model in ['ZERO_CONFIRM_BAND_05', 'INTENT_MIX_BAND_05']:
        old, decisions = simulate_band_account(data, div, cfg, cost, str(data.date.iloc[1].date()), model,
            targets=targets, event_mask=np.ones(len(data), bool))
        modified, values = data.copy(), targets.copy()
        modified.loc[6:, ['open', 'close', 'previous_close']] = 10.2
        values[5:] = 0.
        newer, changed = simulate_band_account(modified, div, cfg, cost, str(data.date.iloc[1].date()), model,
            targets=values, event_mask=np.ones(len(data), bool))
        pd.testing.assert_frame_equal(old.iloc[:5], newer.iloc[:5])
        pd.testing.assert_frame_equal(decisions.iloc[:5], changed.iloc[:5])
