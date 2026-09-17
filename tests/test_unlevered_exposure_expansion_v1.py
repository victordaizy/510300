import numpy as np
import pandas as pd
import pytest

from research.unlevered_exposure_expansion_inputs_v1 import CANDIDATES, MULTIPLIERS, PARENT, expanded_target, expansion_frames


def fixture():
    data = pd.DataFrame({'date': pd.bdate_range('2020-01-01', periods=8)})
    idx = np.arange(1, 7)
    parents = {}
    for cost in ['BASE', 'STRESS']:
        parents[cost] = {PARENT: pd.DataFrame({
            'origin_index': idx, 'origin': data.date.iloc[idx].to_numpy(),
            'execution_date': data.date.iloc[idx + 1].to_numpy(),
            'source_cost': cost, 'source_model': PARENT,
            'reference_weight': [0.0, .2, .6, np.nan, 1.0, 0.0],
        })}
    cfg = {'candidate_models': list(CANDIDATES), 'multipliers': MULTIPLIERS, 'decision_clock': '15:05:00',
           'weight_band': .1, 'costs': {'BASE': {}, 'STRESS': {}}}
    return data, parents, cfg, str(data.date.iloc[2].date())


def test_hand_computed_cash_cap_and_unknown():
    v = np.array([0, .2, .6, 1, np.nan])
    np.testing.assert_allclose(expanded_target(v, 1.5), [0, .3, .9, 1, np.nan], equal_nan=True)
    np.testing.assert_allclose(expanded_target(v, 2), [0, .4, 1, 1, np.nan], equal_nan=True)
    with pytest.raises(ValueError):
        expanded_target([1.01], 1.5)


def test_source_identity_and_next_open_are_required():
    d, p, c, s = fixture()
    p['BASE'][PARENT].loc[0, 'execution_date'] = d.date.iloc[1]
    with pytest.raises(ValueError):
        expansion_frames(d, p, c, s)


def test_future_source_changes_do_not_change_earlier_targets():
    d, p, c, s = fixture()
    before, _ = expansion_frames(d, p, c, s)
    p['BASE'][PARENT].loc[5, 'reference_weight'] = .8
    after, _ = expansion_frames(d, p, c, s)
    for model in CANDIDATES:
        np.testing.assert_allclose(before['BASE'][model + '_target'].iloc[:6], after['BASE'][model + '_target'].iloc[:6], equal_nan=True)
        np.testing.assert_allclose(before['STRESS'][model + '_target'], after['STRESS'][model + '_target'], equal_nan=True)


def test_warmup_terminal_and_unknown_are_not_cash():
    d, p, c, s = fixture()
    frames, _ = expansion_frames(d, p, c, s)
    for f in frames.values():
        for model in CANDIDATES:
            assert f[model + '_target'].iloc[[0, 4, 7]].isna().all()
            assert f[model + '_target'].iloc[[1, 6]].eq(0).all()
