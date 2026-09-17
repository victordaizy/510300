"""验证筛选权重、五个计划来源、外层普通调仓和源账户未来隔离。"""
import copy
import numpy as np
import pandas as pd
import pytest
from research.incremental_selected_intent_mix_inputs_v1 import CANDIDATES, MODELS, SETTINGS, SOURCES, gate_frames, simulate_gate_account
from tests.test_three_source_order_intent_mix_v1 import inputs


def fixture():
    data,div,cfg,old,books = inputs()
    old_models = list(old['BASE'])
    sources,ledgers = {},{}
    for cost,rows in old.items():
        sources[cost] = {}
        for j,model in enumerate(MODELS):
            source = old_models[j%3]
            sources[cost][model] = rows[source].copy().assign(source_model=model)
            ledgers[cost,model] = books[cost,source].copy()
    cfg.update(candidate_models=list(CANDIDATES),candidate_settings=copy.deepcopy(SETTINGS),source_folders=SOURCES.copy())
    return data,div,cfg,sources,ledgers


def test_weights_two_bands_hand_mix_and_source_zero_wait():
    data,_,cfg,sources,ledgers = fixture()
    frames,summaries = gate_frames(data,sources,cfg,cfg['evaluation_start'],lambda p,c,m:ledgers[c,m])
    assert len(SETTINGS)==8 and len(summaries)==16
    for frame in frames.values():
        for model,s in SETTINGS.items():
            assert abs(sum(s['source_weights'].values())-1)<1e-12
            expected=np.zeros(len(data))
            for m,w in s['source_weights'].items():
                if w>0:
                    expected+=w*frame[m+'_planned_weight'].to_numpy(float)
            np.testing.assert_allclose(frame[model+'_target'],np.minimum(1.,expected),atol=0,rtol=0,equal_nan=True)
            assert frame[model+'_target'].iloc[1]>0 and frame[model+'_target'].iloc[2]==0
    assert list(SETTINGS['SELECTED_MIX_BAND00_SIMPLE3']['source_weights'].values())==[.7,.15,.15,0.,0.]
    assert list(SETTINGS['SELECTED_MIX_BAND10_SIMPLE2']['source_weights'].values())==[.85,0.,.15,0.,0.]


def test_zero_weight_unknown_is_ignored_positive_weight_unknown_is_preserved():
    data,_,cfg,sources,ledgers=fixture()
    sources['BASE'][MODELS[1]].loc[3,'reference_weight']=np.nan
    frames,_=gate_frames(data,sources,cfg,cfg['evaluation_start'],lambda p,c,m:ledgers[c,m])
    for band in [0,10]:
        assert np.isfinite(frames['BASE'][f'SELECTED_MIX_BAND{band:02d}_SIMPLE2_target'].iloc[3])
        assert np.isnan(frames['BASE'][f'SELECTED_MIX_BAND{band:02d}_SIMPLE3_target'].iloc[3])


def test_source_cost_and_frozen_weight_mismatch_are_rejected():
    data,_,cfg,sources,ledgers=fixture()
    bad=copy.deepcopy(sources)
    bad['STRESS'][MODELS[0]]['source_cost']='BASE'
    with pytest.raises(ValueError,match='费用'):
        gate_frames(data,bad,cfg,cfg['evaluation_start'],lambda p,c,m:ledgers[c,m])
    cfg['candidate_settings']['SELECTED_MIX_BAND00_SIMPLE2']['source_weights'][MODELS[0]]=.86
    with pytest.raises(ValueError,match='设置'):
        gate_frames(data,sources,cfg,cfg['evaluation_start'],lambda p,c,m:ledgers[c,m])


def test_master_does_not_repeat_addition_filter_and_sells_normally():
    data,div,cfg,_,_=fixture()
    targets=np.array([.3,.8,.1,np.nan,.5,0.,.4,.8,.8,.8])
    for model in CANDIDATES:
        ledger,decisions=simulate_gate_account(data,div,cfg,cfg['costs']['BASE'],cfg['evaluation_start'],model,
            targets=targets,event_mask=np.ones(len(data),bool))
        assert decisions.iloc[1].observed_daily_return==0 and decisions.iloc[1].requested_quantity>0
        assert decisions.iloc[2].requested_quantity<0
        assert ledger.shares.iloc[3]==ledger.shares.iloc[2]
        assert ledger.shares.iloc[5]==0 and ledger.shares.iloc[6]>0 and ledger.shares.iloc[-1]==0
        assert decisions.addition_suppressed.dropna().eq(0).all()


def test_future_source_fills_and_equity_cannot_change_old_plan():
    data,_,cfg,sources,ledgers=fixture()
    old,_=gate_frames(data,sources,cfg,cfg['evaluation_start'],lambda p,c,m:ledgers[c,m])
    changed,later,books=data.copy(),copy.deepcopy(sources),copy.deepcopy(ledgers)
    changed.loc[6:,['open','close','previous_close']]=10.4
    for row in later.values():
        for d in row.values():
            d.loc[6:,'requested_quantity']=0
    for ledger in books.values():
        ledger.loc[6:,'equity']*=1.2
        ledger['filled_quantity']=0
    newer,_=gate_frames(changed,later,cfg,cfg['evaluation_start'],lambda p,c,m:books[c,m])
    for cost in old:
        for model in CANDIDATES:
            pd.testing.assert_series_equal(old[cost][model+'_target'].iloc[:6],newer[cost][model+'_target'].iloc[:6])


def test_future_prices_targets_do_not_change_old_account_decisions():
    data,div,cfg,_,_=fixture()
    targets=np.array([.5,.8,np.nan,.6,.3,.5,.7,.2,0.,.3])
    for model in CANDIDATES:
        def run(frame,values):
            return simulate_gate_account(frame,div,cfg,cfg['costs']['BASE'],cfg['evaluation_start'],model,
                targets=values,event_mask=np.ones(len(frame),bool))
        old,decisions=run(data,targets)
        changed,later=data.copy(),targets.copy()
        changed.loc[6:,['open','close','previous_close']]=10.4
        later[5:]=[0.,.8,.8,.8,.8]
        newer,updated=run(changed,later)
        pd.testing.assert_frame_equal(old.iloc[:5],newer.iloc[:5])
        pd.testing.assert_frame_equal(decisions.iloc[:5],updated.iloc[:5])
