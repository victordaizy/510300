"""验证初次入场、真实加仓、卖出和两方向边界分别处理。"""
import copy

import numpy as np
import pandas as pd
import pytest

from research.adaptive_allocation_v1 import Account
from research.addition_only_return_gate_batch_inputs_v1 import CANDIDATES, MODELS, SETTINGS, allowance_column, gate_frames, gate_request, market_factors, simulate_gate_account
from tests.test_two_close_zero_exit_v1 import fixture


def settings():
    cfg = fixture()[2]
    cfg.update(candidate_models=list(CANDIDATES), candidate_settings=copy.deepcopy(SETTINGS), decision_clock='15:05:00',
        costs={'BASE': fixture()[3], 'STRESS': {'commission': .0004, 'minimum': 5., 'slippage': .001}})
    return cfg


def run(targets, frame=None, model='ADD_GATE_BAND00_LOWER01'):
    data, div, _, cost = fixture()
    data = data if frame is None else frame
    return simulate_gate_account(data, div, settings(), cost, str(data.date.iloc[1].date()), model,
        targets=np.asarray(targets, float), event_mask=np.ones(len(data), bool))


def test_both_directions_include_exact_zero_and_one_percent_and_dividend_neutrality():
    data = pd.DataFrame({'close': [.1, 101., 101.001, 100., 99., 100.],
        'previous_close': [.3, 100., 100., 100., 100., np.nan], 'dividend': [.2, 0., 0., 0., 0., 0.]})
    f = market_factors(data)
    np.testing.assert_array_equal(f.buy_allowed_upper_r00, [True, False, False, True, True, False])
    np.testing.assert_array_equal(f.buy_allowed_upper_r01, [True, True, False, True, True, False])
    np.testing.assert_array_equal(f.buy_allowed_lower_r00, [True, True, True, True, False, False])
    np.testing.assert_array_equal(f.buy_allowed_lower_r01, [False, True, True, False, False, False])


def test_initial_entry_is_unfiltered_addition_can_be_blocked_and_sells_are_kept():
    cfg = settings()
    for model in CANDIDATES:
        account = Account(100000.)
        for observed in [.02, -.02, np.nan]:
            entry = gate_request(account, 10., .5, cfg, model, False, observed)
            assert entry['requested_quantity'] == 5000 and entry['addition_suppressed'] == 0
        account.cash, account.shares = 70000., 3000
        blocked = gate_request(account, 10., .8, cfg, model, False, np.nan)
        assert blocked['requested_quantity'] == 0 and blocked['normal_requested_quantity'] == 5000 and blocked['reference_weight'] == .8
        assert gate_request(account, 10., .8, cfg, model, True, 0.)['requested_quantity'] > 0
        assert gate_request(account, 10., .05, cfg, model, False, np.nan)['requested_quantity'] < 0
        assert gate_request(account, 10., 0., cfg, model, False, np.nan)['requested_quantity'] == -3000


def test_failed_entry_is_not_treated_as_holding_and_new_entry_bypasses_gate():
    data = fixture()[0]
    data.loc[1, 'open'] = 11.
    targets = [.5, .5, .8, np.nan, .2, 0., .5, .8, .8, .8]
    ledger, decisions = run(targets, data)
    assert ledger.iloc[0].status == 'UNFILLED_DIRECTIONAL_LIMIT' and ledger.shares.iloc[0] == 0
    assert decisions.iloc[1].own_close_shares == 0 and decisions.iloc[1].requested_quantity > 0
    assert ledger.shares.iloc[1] > 0
    assert decisions.iloc[2].normal_requested_quantity > 0 and decisions.iloc[2].requested_quantity == 0
    assert ledger.shares.iloc[3] == ledger.shares.iloc[2]
    assert ledger.shares.iloc[5] == 0 and ledger.shares.iloc[6] > 0 and ledger.shares.iloc[-1] == 0
    suppressed = decisions.addition_suppressed.eq(1)
    assert decisions.loc[suppressed, 'own_close_shares'].gt(0).all()


def test_eight_mappings_cost_identity_and_market_factor_selection():
    data, cfg, parents = fixture()[0], settings(), {}
    for cost in cfg['costs']:
        parents[cost] = {m: pd.DataFrame({'origin': data.date.iloc[:-1].to_numpy(),
            'execution_date': data.date.iloc[1:].to_numpy(), 'origin_index': np.arange(len(data)-1),
            'reference_weight': .5, 'source_cost': cost, 'source_model': m}) for m in MODELS}
    frames, summaries = gate_frames(data, parents, cfg, str(data.date.iloc[1].date()))
    assert len(SETTINGS) == 8 and len(summaries) == 16
    for model, setting in SETTINGS.items():
        assert allowance_column(setting) in frames['BASE']
        np.testing.assert_allclose(frames['BASE'][model+'_target'].iloc[:-1], .5)
    bad = copy.deepcopy(parents)
    bad['STRESS'][MODELS[0]]['source_cost'] = 'BASE'
    with pytest.raises(ValueError, match='费用'):
        gate_frames(data, bad, cfg, str(data.date.iloc[1].date()))
    cfg['candidate_settings']['ADD_GATE_BAND00_LOWER00']['direction'] = 'UPPER'
    with pytest.raises(ValueError, match='设置'):
        gate_frames(data, parents, cfg, str(data.date.iloc[1].date()))


def test_future_prices_targets_and_direction_flags_do_not_change_past_accounts():
    data = fixture()[0]
    targets = np.array([.5, .8, np.nan, .6, .3, .5, .7, .2, 0., .3])
    for model in ['ADD_GATE_BAND00_LOWER01', 'ADD_GATE_BAND20_UPPER00']:
        old, decisions = run(targets, model=model)
        changed, later = data.copy(), targets.copy()
        changed.loc[6:, ['open', 'close', 'previous_close']] = 10.4
        later[5:] = [0., .8, .8, .8, .8]
        newer, updated = run(later, changed, model)
        pd.testing.assert_frame_equal(old.iloc[:5], newer.iloc[:5])
        pd.testing.assert_frame_equal(decisions.iloc[:5], updated.iloc[:5])
