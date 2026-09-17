"""集中检查十七套映射与共用退出行为，不重复测试每套预算的旧算法。"""
import copy

import numpy as np
import pandas as pd
import pytest

from research.confirmed_exit_existing_budget_batch_inputs_v1 import CANDIDATES, MAPPING, MODELS, SOURCES, mapped_frames, simulate_mapped_confirmation
from research.two_close_zero_exit_account_v1 import simulate_confirmed_zero
from research.two_close_zero_exit_inputs_v1 import PRIMARY as OLD_PRIMARY
from tests.test_two_close_zero_exit_v1 import fixture


def settings():
    cfg = fixture()[2]
    cfg.update(candidate_models=list(CANDIDATES), candidate_sources=copy.deepcopy(SOURCES), candidate_mapping=MAPPING.copy(),
        decision_clock='15:05:00', costs={'BASE': fixture()[3], 'STRESS': {'commission': .0004, 'minimum': 5., 'slippage': .001}})
    return cfg


def sources():
    frame = fixture()[0]
    result = {}
    for cost_no, cost in enumerate(['BASE', 'STRESS']):
        result[cost] = {}
        for j, model in enumerate(MODELS):
            values = np.full(len(frame)-1, .05+.03*j+.005*cost_no)
            values[1:6] = [0., 0., np.nan, 0., .3]
            result[cost][model] = pd.DataFrame({'origin': frame.date.iloc[:-1].to_numpy(),
                'execution_date': frame.date.iloc[1:].to_numpy(), 'origin_index': np.arange(len(frame)-1),
                'reference_weight': values, 'source_cost': cost, 'source_model': model})
    return frame, result


def test_seventeen_sources_exclude_already_completed_198_and_reject_changes():
    assert len(SOURCES) == len(CANDIDATES) == 17
    assert 'ACCOUNT_VOLATILITY_EXPOSURE' not in SOURCES
    cfg = settings()
    cfg['candidate_sources']['ACCOUNT_RISK_12']['budget_percent'] = 13
    frame, parents = sources()
    with pytest.raises(ValueError, match='来源'):
        mapped_frames(frame, parents, cfg, str(frame.date.iloc[1].date()))


def test_every_candidate_uses_its_own_source_cost_and_zero_sequence():
    frame, parents = sources()
    frames, summaries = mapped_frames(frame, parents, settings(), str(frame.date.iloc[1].date()))
    assert len(summaries) == 34
    for cost in ['BASE', 'STRESS']:
        for model, parent in MAPPING.items():
            np.testing.assert_allclose(frames[cost][model+'_target'].iloc[:-1], parents[cost][parent].reference_weight,
                                       atol=0, rtol=0, equal_nan=True)
            np.testing.assert_array_equal(frames[cost][model+'_zero_streak'].iloc[:6], [0, 1, 2, 0, 1, 0])
            assert pd.isna(frames[cost][model+'_target'].iloc[-1])


def test_wrong_source_cost_or_execution_date_is_rejected():
    frame, parents = sources()
    first = MODELS[0]
    bad = copy.deepcopy(parents)
    bad['STRESS'][first]['source_cost'] = 'BASE'
    with pytest.raises(ValueError, match='费用'):
        mapped_frames(frame, bad, settings(), str(frame.date.iloc[1].date()))
    bad = copy.deepcopy(parents)
    bad['BASE'][first].loc[0, 'execution_date'] = frame.date.iloc[2]
    with pytest.raises(ValueError, match='下一开盘'):
        mapped_frames(frame, bad, settings(), str(frame.date.iloc[1].date()))


def test_new_identity_reproduces_198_confirmation_account_for_same_targets():
    frame, div, _, cost = fixture()
    cfg = settings()
    targets = np.array([.3, 0., np.nan, 0., 0., .8, .4, 0., 0., 0.])
    common = (frame, div, cfg, cost, str(frame.date.iloc[1].date()))
    kwargs = {'targets': targets, 'event_mask': np.ones(len(frame), bool)}
    original_ledger, original_decisions = simulate_confirmed_zero(*common, OLD_PRIMARY, **kwargs)
    for model in [list(CANDIDATES)[0], list(CANDIDATES)[-1]]:
        ledger, decisions = simulate_mapped_confirmation(*common, model, **kwargs)
        pd.testing.assert_frame_equal(ledger, original_ledger)
        pd.testing.assert_frame_equal(decisions[original_decisions.columns], original_decisions)
        assert decisions.loc[decisions.reference_weight.notna(), 'budget_source'].eq(MAPPING[model]).all()


def test_future_source_or_price_changes_do_not_change_past_decisions():
    frame, div, _, cost = fixture()
    cfg, model = settings(), list(CANDIDATES)[0]
    targets = np.array([.3, 0., .4, .8, .2, .4, .8, .6, .3, .3])
    kwargs = {'targets': targets, 'event_mask': np.ones(len(frame), bool)}
    old_ledger, old_decisions = simulate_mapped_confirmation(frame, div, cfg, cost, str(frame.date.iloc[1].date()), model, **kwargs)
    changed = frame.copy()
    changed.loc[6:, ['open', 'close', 'previous_close']] = 10.2
    targets[5:] = [0., np.nan, 0., 0., .4]
    ledger, decisions = simulate_mapped_confirmation(changed, div, cfg, cost, str(frame.date.iloc[1].date()), model, **kwargs)
    pd.testing.assert_frame_equal(ledger.iloc[:5], old_ledger.iloc[:5])
    pd.testing.assert_frame_equal(decisions.iloc[:5], old_decisions.iloc[:5])
