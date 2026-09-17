"""检验正目标减仓四种方式与明确零退出、实际持仓和时点隔离。"""
import copy
import numpy as np
import pandas as pd
import pytest
from research.adaptive_allocation_v1 import Account
from research.positive_target_reduction_batch_inputs_v1 import CANDIDATES, MODELS, SETTINGS, gate_frames, gate_request, market_factors, reduction_request, simulate_gate_account
from tests.test_two_close_zero_exit_v1 import fixture


def settings():
    cfg = fixture()[2]
    cfg.update(candidate_models=list(CANDIDATES), candidate_settings=copy.deepcopy(SETTINGS), decision_clock='15:05:00',
        costs={'BASE': fixture()[3], 'STRESS': {'commission': .0004, 'minimum': 5., 'slippage': .001}})
    return cfg


def run(targets, frame=None, model='REDUCE_105_HALF'):
    data, div, _, cost = fixture()
    return simulate_gate_account(data if frame is None else frame, div, settings(), cost, str(data.date.iloc[1].date()), model,
        targets=np.asarray(targets, float), event_mask=np.ones(len(data), bool))


def test_decimal_zero_and_one_percent_and_ex_dividend_boundaries():
    data = pd.DataFrame({'close': [.1, 101., 101.001, 100., 99., 100.],
        'previous_close': [.3, 100., 100., 100., 100., np.nan], 'dividend': [.2, 0., 0., 0., 0., 0.]})
    f = market_factors(data)
    np.testing.assert_array_equal(f.buy_allowed_upper_r00, [True, False, False, True, True, False])
    np.testing.assert_array_equal(f.buy_allowed_lower_r00, [True, True, True, True, False, False])
    np.testing.assert_array_equal(f.buy_allowed_lower_r01, [False, True, True, False, False, False])


def test_all_entries_bypass_gates_additions_require_permission_and_zero_exit_is_full():
    cfg = settings()
    for model in CANDIDATES:
        account = Account(100000.)
        for value in [-.02, .02, np.nan]:
            r = gate_request(account, 10., .5, cfg, model, False, value)
            assert r['requested_quantity'] == 5000 and r['addition_suppressed'] == 0
        account.cash, account.shares = 70000., 3000
        r = gate_request(account, 10., .8, cfg, model, False, np.nan)
        assert r['requested_quantity'] == 0 and r['addition_suppressed'] == 1
        assert gate_request(account, 10., .8, cfg, model, True, .01)['requested_quantity'] == 5000
        assert gate_request(account, 10., 0., cfg, model, False, np.nan)['requested_quantity'] == -3000


def test_reduction_modes_handle_sign_unknown_half_lots_and_positive_target_full_sale():
    cfg = settings()
    account = Account(50000.)
    account.shares = 5000
    for model, setting in SETTINGS.items():
        for upper, lower in [(True, False), (False, True), (True, True), (False, False)]:
            r = gate_request(account, 10., .1, cfg, model, False, np.nan, upper, lower)
            mode = setting['reduction_mode']
            expected = -2000 if mode == 'HALF' else -4000 if (mode == 'UPPER00' and upper) or (mode == 'LOWER00' and lower) else 0
            assert r['normal_requested_quantity'] == -4000 and r['requested_quantity'] == expected
            assert r['reference_weight'] == .1 and r['ordinary_reduction'] == 1
            full = gate_request(account, 10., .00001, cfg, model, False, np.nan, upper, lower)
            expected_full = -2500 if mode == 'HALF' else -5000 if (mode == 'UPPER00' and upper) or (mode == 'LOWER00' and lower) else 0
            assert full['requested_quantity'] == expected_full
    assert reduction_request(-300, .1, 100, 'HALF', False, False) == -100
    assert reduction_request(-100, .1, 100, 'HALF', False, False) == 0
    assert reduction_request(-300, 0., 100, 'HALF', False, False) == -300


def test_eight_mappings_caps_unknown_and_matching_cost_sources():
    data, cfg, parents = fixture()[0], settings(), {}
    raw = np.array([.5, .96, 1., 0., np.nan, .2, .8, .9, .3])
    for cost in cfg['costs']:
        parents[cost] = {m: pd.DataFrame({'origin': data.date.iloc[:-1].to_numpy(),
            'execution_date': data.date.iloc[1:].to_numpy(), 'origin_index': np.arange(len(data)-1),
            'reference_weight': raw if cost == 'BASE' else raw*.8, 'source_cost': cost, 'source_model': m}) for m in MODELS}
    frames, summaries = gate_frames(data, parents, cfg, str(data.date.iloc[1].date()))
    assert len(CANDIDATES) == 8 and len(summaries) == 16
    for cost, frame in frames.items():
        for model, setting in SETTINGS.items():
            expected = np.minimum(1., (raw if cost == 'BASE' else raw*.8)*setting['exposure_multiplier'])
            np.testing.assert_allclose(frame[model+'_target'].iloc[:-1], expected, equal_nan=True)
    bad = copy.deepcopy(parents)
    bad['STRESS'][MODELS[0]]['source_cost'] = 'BASE'
    with pytest.raises(ValueError, match='费用'):
        gate_frames(data, bad, cfg, str(data.date.iloc[1].date()))


def test_unknown_target_keeps_actual_shares_and_terminal_exit_overrides_hold():
    targets = [.5, .8, np.nan, .1, .3, .5, .7, .2, .2, .3]
    for model in CANDIDATES:
        ledger, decisions = run(targets, model=model)
        assert ledger.shares.iloc[2] == ledger.shares.iloc[1]
        assert ledger.shares.iloc[-1] == 0
        if SETTINGS[model]['reduction_mode'] == 'HOLD':
            ordinary = decisions.reference_weight.gt(0)&decisions.normal_requested_quantity.lt(0)
            assert decisions.loc[ordinary, 'requested_quantity'].eq(0).all()


def test_unfilled_entry_stays_flat_and_reentry_ignores_gate():
    data = fixture()[0]
    data.loc[1, 'open'] = 11.
    ledger, decisions = run([.5, .5, .8, np.nan, .2, 0., .5, .8, .8, .8], data, 'REDUCE_100_HOLD')
    assert ledger.iloc[0].status == 'UNFILLED_DIRECTIONAL_LIMIT' and ledger.shares.iloc[0] == 0
    assert decisions.iloc[1].own_close_shares == 0 and decisions.iloc[1].requested_quantity > 0
    assert ledger.shares.iloc[1] > 0 and ledger.shares.iloc[5] == 0 and ledger.shares.iloc[6] > 0


def test_future_changes_do_not_alter_prior_decisions_or_equity():
    targets = np.array([.5, .8, np.nan, .6, .3, .5, .7, .2, 0., .3])
    for model in CANDIDATES:
        data = fixture()[0]
        old, decisions = run(targets, data, model)
        changed, later = data.copy(), targets.copy()
        changed.loc[6:, ['open', 'close', 'previous_close']] = 10.4
        later[5:] = [0., .8, .8, .8, .8]
        newer, updated = run(later, changed, model)
        pd.testing.assert_frame_equal(old.iloc[:5], newer.iloc[:5])
        pd.testing.assert_frame_equal(decisions.iloc[:5], updated.iloc[:5])
