"""检验延后退出语义、单账户净额、来源约束和未来隔离。"""
import copy

import numpy as np
import pandas as pd
import pytest

from research.adaptive_allocation_v1 import Account, target_request
from research.event_clock_account_v1 import simulate_event_account
from research.three_source_order_intent_mix_inputs_v1 import CANDIDATES, MODELS, PRIMARY, SOURCES, intent_frames, source_plan, validate_weights
from research.two_close_zero_exit_account_v1 import simulate_confirmed_zero
from tests.test_two_close_zero_exit_v1 import fixture


def inputs():
    frame, div, cfg, base = fixture()
    cfg.update(costs={'BASE': base, 'STRESS': {'commission': .0004, 'minimum': 5., 'slippage': .001}},
        candidate_models=list(CANDIDATES), source_folders=SOURCES.copy(), decision_clock='15:05:00',
        evaluation_start=str(frame.date.iloc[1].date()), source_weights={PRIMARY: dict(zip(MODELS, [.8, .15, .05])),
        'INTENT_MIX_DIAGNOSTIC_WEIGHTS': dict(zip(MODELS, [.77, .15, .08]))})
    sources, ledgers = {}, {}
    targets = np.array([.3, 0., 0., .4, np.nan, .5, .2, 0., 0., 0.])
    for cost, costs in cfg['costs'].items():
        sources[cost] = {}
        for j, model in enumerate(MODELS):
            engine = simulate_event_account if j == 2 else simulate_confirmed_zero
            ledger, decisions = engine(frame, div, cfg, costs, cfg['evaluation_start'],
                'TWO_CLOSE_ZERO_EXIT' if j < 2 else model, targets=targets, event_mask=np.ones(len(frame), bool))
            sources[cost][model] = decisions.assign(source_model=model, source_cost=cost,
                decision_time=decisions.origin+pd.Timedelta(hours=15, minutes=5))
            ledgers[(cost, model)] = ledger
    return frame, div, cfg, sources, ledgers


def test_first_zero_retains_plan_second_zero_exits_and_unknown_propagates():
    frame, _, cfg, sources, ledgers = inputs()
    source = sources['BASE'][MODELS[0]]
    plan = source_plan(frame, source, ledgers['BASE', MODELS[0]], cfg, 1)
    assert source.reference_weight.iloc[1] == 0 and plan.planned_shares.iloc[1] == 3000
    assert plan.planned_weight.iloc[1] > 0 and plan.planned_weight.iloc[2] == 0
    assert pd.isna(plan.planned_weight.iloc[4])
    frames, _ = intent_frames(frame, sources, cfg, cfg['evaluation_start'], lambda p, c, m: ledgers[c, m])
    assert frames['BASE'][PRIMARY+'_target'].iloc[1] > 0
    assert frames['BASE'][PRIMARY+'_target'].iloc[2] == 0
    assert pd.isna(frames['BASE'][PRIMARY+'_target'].iloc[4])


def test_source_cost_date_and_weight_identities_cannot_be_mixed():
    frame, _, cfg, sources, ledgers = inputs()
    loader = lambda p, c, m: ledgers[c, m]
    bad = copy.deepcopy(sources)
    bad['STRESS'][MODELS[0]]['source_cost'] = 'BASE'
    with pytest.raises(ValueError, match='费用'):
        intent_frames(frame, bad, cfg, cfg['evaluation_start'], loader)
    bad = copy.deepcopy(sources)
    bad['BASE'][MODELS[1]].loc[0, 'execution_date'] = frame.date.iloc[2]
    with pytest.raises(ValueError, match='下一开盘'):
        intent_frames(frame, bad, cfg, cfg['evaluation_start'], loader)
    bad_cfg = copy.deepcopy(cfg)
    bad_cfg['source_weights'][PRIMARY][MODELS[0]] = .81
    with pytest.raises(ValueError, match='权重'):
        validate_weights(bad_cfg)


def test_negative_plan_or_leverage_is_rejected_without_clipping():
    frame, _, cfg, sources, ledgers = inputs()
    for qty in [-100, 20000]:
        source = sources['BASE'][MODELS[0]].copy()
        source.loc[0, 'requested_quantity'] = qty
        with pytest.raises(ValueError, match='计划'):
            source_plan(frame, source, ledgers['BASE', MODELS[0]], cfg, 1)


def test_source_weights_hand_calculation_and_one_net_request():
    frame, div, cfg, sources, ledgers = inputs()
    frames, _ = intent_frames(frame, sources, cfg, cfg['evaluation_start'], lambda p, c, m: ledgers[c, m])
    f = frames['BASE']
    for i in range(len(frame)-1):
        expected = .8*f.iloc[i][MODELS[0]+'_planned_weight']+.15*f.iloc[i][MODELS[1]+'_planned_weight']+.05*f.iloc[i][MODELS[2]+'_planned_weight']
        np.testing.assert_allclose(f.iloc[i][PRIMARY+'_target'], expected, atol=0, rtol=0, equal_nan=True)
    account = Account(50000.)
    account.shares = 5000
    # 两个等额来源一增一减，合并目标仍为一半，只形成零差额。
    assert target_request(account, 10., .5*.8+.5*.2, cfg)['requested_quantity'] == 0
    ledger, decisions = simulate_event_account(frame, div, cfg, cfg['costs']['BASE'], cfg['evaluation_start'], PRIMARY,
        targets=f[PRIMARY+'_target'].to_numpy(), event_mask=np.ones(len(frame), bool))
    assert ledger.shares.iloc[1] > 0 and ledger.shares.iloc[2] == 0
    assert decisions.requested_quantity.iloc[2] < 0 and ledger.shares.iloc[-1] == 0


def test_future_fills_nav_and_next_open_do_not_change_prior_plan():
    frame, _, cfg, sources, ledgers = inputs()
    source, ledger = sources['BASE'][MODELS[0]], ledgers['BASE', MODELS[0]]
    old = source_plan(frame, source, ledger, cfg, 1)
    changed, modified, later = ledger.copy(), frame.copy(), source.copy()
    changed.loc[5:, 'equity'] *= 1.2
    changed.loc[5:, 'shares'] = 0
    changed['filled_quantity'] = 0
    modified.loc[5:, 'open'] = 11.
    later.loc[6:, 'requested_quantity'] = 0
    new = source_plan(modified, later, changed, cfg, 1)
    pd.testing.assert_frame_equal(old.iloc[:6], new.iloc[:6])
