"""验证联合作用的可识别性、成熟缓存、固定版本及受阻后的退出。"""
import copy
from itertools import product

import numpy as np
import pandas as pd
import pytest

from research import holding_market_coupling_exit_inputs_v1 as module
from research.learned_cycle_exit_v1 import FEATURES
from research.rearmed_cycle_exit_account_v1 import simulate_rearmed_exit
from tests.test_single_component_exit_v1 import monthly_fixture
from tests.test_median_continuation_v1 import fixture

CFG = {'feature_clip': 5., 'ridge_alpha': 1., 'interaction_pairs': [list(p) for p in module.PAIRS],
       'recent_cycles': 20, 'minimum_cycles': 4, 'minimum_rows': 40}


def sample():
    base = np.array(list(product([-1., 1.], repeat=7)))
    base = np.insert(base, 3, 1., axis=1)
    x = np.tile(base, (4, 1))
    rows = pd.DataFrame(x, columns=FEATURES)
    rows['cycle_id'] = np.repeat(np.arange(1, 5), 128)
    rows['origin_index'] = np.arange(512)
    rows['exit_index'] = np.repeat([128, 256, 384, 512], 128)
    rows['sample_weight'] = 1/128
    rows['target'] = .05*x[:, 1]*x[:, 4]+rows.cycle_id*.1
    return rows


def test_identifiable_interaction_reverses_profit_effect_across_market_states():
    fitted = module.fit_coupling_cycle(sample(), CFG)
    expected = np.zeros(20)
    expected[12] = .04
    np.testing.assert_allclose(fitted['coefficients'], expected, atol=1e-12, rtol=0)
    assert fitted['intercept'] == pytest.approx(.25)
    x = np.zeros(8)
    x[1], x[4] = 1., 1.
    positive = module.coupling_prediction(fitted, x)
    x[4] = -1.
    negative = module.coupling_prediction(fitted, x)
    assert positive == pytest.approx(.29) and negative == pytest.approx(.21)


def test_two_stage_scale_and_full_normal_equation_with_unequal_cycle_sizes():
    rows = sample()
    rows = rows[~(rows.cycle_id.eq(4) & rows.origin_index.ge(450))].copy()
    rows['sample_weight'] = 1/rows.groupby('cycle_id').origin_index.transform('count')
    rows.loc[rows.cycle_id.eq(1), FEATURES[4]] *= 2.
    fitted = module.fit_coupling_cycle(rows, CFG)
    w = rows.sample_weight.to_numpy()
    x = rows[FEATURES].to_numpy()
    z = np.clip((x-np.average(x, axis=0, weights=w))/fitted['scale'], -5., 5.)
    assert len(module.PAIRS) == 12 and len(set(module.PAIRS)) == 12
    products = np.column_stack([z[:, i]*z[:, j] for i in range(3) for j in range(4, 8)])
    mu = np.average(products, axis=0, weights=w)
    sd = np.sqrt(np.average((products-mu)**2, axis=0, weights=w))
    sd[sd <= 1e-12] = 1.
    np.testing.assert_allclose(fitted['product_mean'], mu, atol=1e-12)
    np.testing.assert_allclose(fitted['product_scale'], sd, atol=1e-12)
    design = np.column_stack([z, np.clip((products-mu)/sd, -5., 5.)])
    target = rows.target.to_numpy().copy()
    for cycle in rows.cycle_id.unique():
        mask = rows.cycle_id.eq(cycle).to_numpy()
        design[mask] -= np.average(design[mask], axis=0, weights=w[mask])
        target[mask] -= np.average(target[mask], weights=w[mask])
    beta = np.linalg.solve(design.T@(w[:, None]*design)+np.eye(20), design.T@(w*target))
    np.testing.assert_allclose(fitted['coefficients'], beta, atol=1e-12)
    assert fitted['scale'][3] == 1. and abs(fitted['coefficients'][3]) < 1e-12


def test_missing_inputs_are_rejected_and_flat_columns_have_finite_scale():
    rows = sample()
    rows[FEATURES] = 0.
    fitted = module.fit_coupling_cycle(rows, CFG)
    assert fitted['scale'] == [1.]*8 and fitted['product_scale'] == [1.]*12
    assert module.coupling_prediction(fitted, np.zeros(8)) == pytest.approx(.25)
    rows.loc[0, FEATURES[2]] = np.nan
    with pytest.raises(ValueError, match='禁止删行'):
        module.fit_coupling_cycle(rows, CFG)
    with pytest.raises(ValueError, match='完整八项'):
        module.coupling_prediction(fitted, np.full(8, np.nan))


def test_past_cache_only_fits_identical_input_once_and_preserves_failure(monkeypatch):
    rows, records = monthly_fixture()
    future = rows[rows.cycle_id.eq(1)].copy()
    future['cycle_id'], future['origin_index'], future['exit_index'] = 5, np.arange(150, 166), 200
    future['target'] = 999.
    extended = pd.concat([rows, future], ignore_index=True)
    full, _, _, counts = module.build_monthly_models(extended, records, CFG)
    prefix, _, _, _ = module.build_monthly_models(rows, records[:2], CFG)
    assert full[:2] == prefix and full[0]['model'] is None
    assert counts['new_model_fits'] == 1 and counts['reused_monthly_fits'] == 2
    assert [r['parameter_first_fit_index'] for r in full] == [None, 100, 100, 100]
    calls = []
    def fail(*args):
        calls.append(1)
        raise ValueError('合成失败')
    monkeypatch.setattr(module, 'fit_coupling_cycle', fail)
    failed, _, _, counts = module.build_monthly_models(rows, records, CFG)
    assert len(calls) == 1 and counts['failed_fits'] == 1
    assert all(r['model'] is None for r in failed)


def constant(value, t=0, latest=None):
    return {'fit_index': t, 'parameter_first_fit_index': t, 'latest_exit_index': t if latest is None else latest,
        'status': 'FIT_COMPLETE', 'model': {'kind': module.KIND, 'features': FEATURES, 'mean': [0.]*8,
        'scale': [1.]*8, 'coefficients': [0.]*20, 'feature_clip': 5., 'intercept': value,
        'interaction_pairs': [list(p) for p in module.PAIRS], 'product_mean': [0.]*12, 'product_scale': [1.]*12}}


def test_entry_model_is_fixed_missing_state_resets_count_and_future_is_rejected():
    data = fixture()[0]
    data.loc[2, 'mom5'] = np.nan
    cycle = {'cycle_id': 1, 'entry_index': 1, 'entry_cost_cny': 100000., 'mode': 1}
    controller = module.CouplingExitController(data, [constant(-.01), constant(.05, 2)])
    got = [controller(t, cycle, 100000., 100000.) for t in range(1, 5)]
    assert [r['negative_confirmation_count'] for r in got] == [1, 0, 1, 2]
    assert len({r['fixed_prediction_identity'] for r in got}) == 1
    cycle.update(cycle_id=2, entry_index=5)
    assert controller(5, cycle, 100000., 100000.)['continuation_prediction'] == .05
    with pytest.raises(ValueError, match='未来周期'):
        module.CouplingExitController(data, [constant(-.1, 0, 8)])(5, cycle, 100000., 100000.)
    with pytest.raises(ValueError, match='首次持仓收盘'):
        module.CouplingExitController(data, [constant(-.1)])(6, cycle, 100000., 100000.)


def test_blocked_next_open_exit_dividend_reentry_and_no_model_natural_exit():
    args = fixture()
    data = args[0]
    data.loc[3:, ['open', 'close']] = 9.9
    data.loc[4:, 'previous_close'] = 9.9
    data.loc[3, ['open', 'dividend']] = [8.91, .1]
    args[1] = pd.DataFrame({'record_date': [data.date.iloc[2]], 'ex_date': [data.date.iloc[3]],
        'payment_date': [data.date.iloc[5]], 'cash_dividend_per_share': [.1]})
    args[5]['entry'][5] = 0
    ledger, decisions, cycles = simulate_rearmed_exit(*args, module.CouplingExitController(data, [constant(-.01), constant(.03, 3)]))
    assert ledger.iloc[2].status == 'UNFILLED_DIRECTIONAL_LIMIT'
    assert ledger.loc[ledger.filled_quantity.lt(0), 'date'].tolist() == [data.date.iloc[4], data.date.iloc[9]]
    assert ledger.loc[ledger.filled_quantity.gt(0), 'date'].tolist() == [data.date.iloc[1], data.date.iloc[7]]
    by_date = ledger.set_index('date')
    amount = by_date.loc[data.date.iloc[2], 'shares']*.1
    assert by_date.loc[data.date.iloc[3], 'dividend_recognized'] == pytest.approx(amount)
    assert by_date.loc[data.date.iloc[5], 'dividend_paid'] == pytest.approx(amount)
    assert ledger.accounting_error.abs().max() < 1e-6 and not ledger.terminal_unliquidated.iloc[-1]
    args = fixture()
    args[5]['exit'][1][2] = True
    ledger, decisions, cycles = simulate_rearmed_exit(*args, module.CouplingExitController(args[0], [constant(-.5, 2)]))
    assert decisions.continuation_prediction.dropna().empty and '价格退出条件' in cycles.iloc[0].exit_reasons
