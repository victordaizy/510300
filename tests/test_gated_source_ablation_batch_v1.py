"""检验来源删减、计划等待、未知传播、入场退出与未来隔离。"""
import copy
import numpy as np
import pandas as pd
import pytest
from research.adaptive_allocation_v1 import Account
from research.gated_source_ablation_batch_inputs_v1 import CANDIDATES, MODELS, SETTINGS, SOURCES, gate_frames, gate_request, simulate_gate_account
from tests.test_three_source_order_intent_mix_v1 import inputs


def fixture():
    data, div, cfg, sources, ledgers = inputs()
    cfg.update(candidate_models=list(CANDIDATES), candidate_settings=copy.deepcopy(SETTINGS), source_folders=SOURCES.copy())
    return data, div, cfg, sources, ledgers


def test_six_planned_mixes_hand_values_preserve_first_zero_wait_and_second_zero_exit():
    data, _, cfg, sources, ledgers = fixture()
    frames, summaries = gate_frames(data, sources, cfg, cfg['evaluation_start'], lambda p,c,m: ledgers[c,m])
    assert len(CANDIDATES) == 6 and len(summaries) == 12
    for cost, frame in frames.items():
        for model, setting in SETTINGS.items():
            expected = np.zeros(len(data))
            for parent, weight in setting['source_weights'].items():
                if weight:
                    expected += weight*frame[parent+'_planned_weight'].to_numpy(float)
            expected = np.minimum(1., expected*setting['exposure_multiplier'])
            np.testing.assert_allclose(frame[model+'_target'], expected, atol=0, rtol=0, equal_nan=True)
            assert frame[model+'_target'].iloc[1] > 0 and frame[model+'_target'].iloc[2] == 0
            assert pd.isna(frame[model+'_target'].iloc[4])
            assert abs(sum(setting['source_weights'].values())-1) < 1e-12


def test_removed_unknown_source_cannot_pollute_retained_known_sources():
    data, _, cfg, sources, ledgers = fixture()
    sources['BASE'][MODELS[1]].loc[3, 'reference_weight'] = np.nan
    frames, _ = gate_frames(data, sources, cfg, cfg['evaluation_start'], lambda p,c,m: ledgers[c,m])
    f = frames['BASE']
    for percent in [100,105]:
        assert np.isfinite(f[f'ABLATE_{percent}_NO_EPISODE_target'].iloc[3])
        assert np.isfinite(f[f'ABLATE_{percent}_CORE_ONLY_target'].iloc[3])
        assert np.isnan(f[f'ABLATE_{percent}_NO_REBOUND_target'].iloc[3])


def test_cost_identity_and_changed_candidate_weights_are_rejected():
    data, _, cfg, sources, ledgers = fixture()
    bad = copy.deepcopy(sources)
    bad['STRESS'][MODELS[0]]['source_cost'] = 'BASE'
    with pytest.raises(ValueError, match='费用'):
        gate_frames(data, bad, cfg, cfg['evaluation_start'], lambda p,c,m: ledgers[c,m])
    cfg['candidate_settings']['ABLATE_100_CORE_ONLY']['source_weights'][MODELS[0]] = .9
    with pytest.raises(ValueError, match='设置'):
        gate_frames(data, sources, cfg, cfg['evaluation_start'], lambda p,c,m: ledgers[c,m])


def test_initial_entry_addition_permission_ordinary_reduction_and_zero_exit():
    _, _, cfg, _, _ = fixture()
    for model in CANDIDATES:
        a = Account(100000.)
        assert gate_request(a,10.,.5,cfg,model,False,np.nan)['requested_quantity'] == 5000
        a.cash, a.shares = 50000.,5000
        assert gate_request(a,10.,.9,cfg,model,False,np.nan)['requested_quantity'] == 0
        assert gate_request(a,10.,.9,cfg,model,True,.01)['requested_quantity'] == 4000
        assert gate_request(a,10.,.1,cfg,model,False,np.nan)['requested_quantity'] == -4000
        assert gate_request(a,10.,0.,cfg,model,False,np.nan)['requested_quantity'] == -5000


def test_future_fills_and_source_equity_do_not_change_past_mixed_targets():
    data, _, cfg, sources, ledgers = fixture()
    old, _ = gate_frames(data,sources,cfg,cfg['evaluation_start'],lambda p,c,m:ledgers[c,m])
    changed, later, altered = data.copy(), copy.deepcopy(sources), copy.deepcopy(ledgers)
    changed.loc[6:,['open','close','previous_close']] = 10.4
    for table in later.values():
        for source in table.values():
            source.loc[6:,'requested_quantity'] = 0
    for ledger in altered.values():
        ledger.loc[6:,'equity'] *= 1.2
        ledger['filled_quantity'] = 0
    newer, _ = gate_frames(changed,later,cfg,cfg['evaluation_start'],lambda p,c,m:altered[c,m])
    for cost in old:
        for model in CANDIDATES:
            pd.testing.assert_series_equal(old[cost][model+'_target'].iloc[:6],newer[cost][model+'_target'].iloc[:6])


def test_simulated_unknown_holds_terminal_exits_and_future_prices_are_isolated():
    data, div, cfg, _, _ = fixture()
    targets = np.array([.5,.8,np.nan,.1,.3,.5,.7,.2,0.,.3])
    for model in CANDIDATES:
        def run(frame, values):
            return simulate_gate_account(frame,div,cfg,cfg['costs']['BASE'],cfg['evaluation_start'],model,
                targets=values,event_mask=np.ones(len(frame),bool))
        old, decisions = run(data,targets)
        assert old.shares.iloc[2] == old.shares.iloc[1] and old.shares.iloc[-1] == 0
        changed, later = data.copy(),targets.copy()
        changed.loc[6:,['open','close','previous_close']] = 10.4
        later[5:] = [0.,.8,.8,.8,.8]
        new, updated = run(changed,later)
        pd.testing.assert_frame_equal(old.iloc[:5],new.iloc[:5])
        pd.testing.assert_frame_equal(decisions.iloc[:5],updated.iloc[:5])
