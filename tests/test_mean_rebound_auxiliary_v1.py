"""验证两档组合与原辅助生命周期，避免日期、未知或等待规则变化。"""
import copy
import numpy as np
import pandas as pd
import pytest

from research.mean_rebound_auxiliary_inputs_v1 import CORE,AUX,MODELS,CANDIDATES,WEIGHTS,combined_targets,rebound_frames
from research.simple_price_entry_exit_v1 import signals,simulate_policy,specifications

CFG={'candidate_models':list(CANDIDATES),'auxiliary_weights':WEIGHTS.copy()}


def test_two_fixed_allocations_keep_core_and_cap_total_at_one():
    got=combined_targets([0.,.2,.9,0.],[1.,1.,1.,0.],CFG)
    np.testing.assert_allclose(got['MEAN_REBOUND_AUX_25'],[.25,.45,1.,0.])
    np.testing.assert_allclose(got['MEAN_REBOUND_AUX_50'],[.5,.7,1.,0.])
    for target in combined_targets([.2,.8,0.],[0.,0.,0.],CFG).values():
        np.testing.assert_array_equal(target,[.2,.8,0.])


def test_unknown_is_not_filled_even_if_other_source_is_full():
    for target in combined_targets([np.nan,1.,0.],[1.,np.nan,np.nan],CFG).values():
        assert np.isnan(target).all()
    bad=copy.deepcopy(CFG)
    bad['auxiliary_weights']['MEAN_REBOUND_AUX_50']=.75
    with pytest.raises(ValueError,match='固定两档'):
        combined_targets([.2],[.3],bad)


def test_both_sources_match_cost_and_next_open_without_future_changes():
    dates=pd.bdate_range('2020-01-01',periods=5)
    data=pd.DataFrame({'date':dates})
    cfg={**CFG,'costs':{'BASE':{},'STRESS':{}},'decision_clock':'15:05:00','weight_band':.1}
    parents={cost:{model:pd.DataFrame({'origin_index':range(4),'origin':dates[:-1],'execution_date':dates[1:],
        'reference_weight':[.2,0.,.3,.4] if model==CORE else [0.,1.,1.,0.],
        'source_cost':cost,'source_model':model}) for model in MODELS} for cost in cfg['costs']}
    frames,_=rebound_frames(data,parents,cfg,dates[1])
    np.testing.assert_allclose(frames['BASE']['MEAN_REBOUND_AUX_25_target'],[.2,.25,.55,.4,np.nan],equal_nan=True)
    changed=copy.deepcopy(parents)
    changed['BASE'][AUX].loc[3,'reference_weight']=1.
    other,_=rebound_frames(data,changed,cfg,dates[1])
    pd.testing.assert_frame_equal(frames['BASE'].iloc[:3],other['BASE'].iloc[:3])
    changed['BASE'][AUX]['source_cost']='STRESS'
    with pytest.raises(ValueError,match='费用或身份'):
        rebound_frames(data,changed,cfg,dates[1])


def fixture():
    dates=pd.bdate_range('2020-01-01',periods=14)
    frame=pd.DataFrame({'date':dates,'open':10.,'close':10.,'previous_close':10.,'dividend':0.,
        'wealth':np.arange(100.,114.),'sma120':0.,'efficiency20':0.,'sma60':0.,'sma20':0.,'z20':-2.,
        'mom5':0.,'mom20':0.,'rsi2':50.,'dd20':0.,'close_location':.5,'feature_valid':True})
    dividends=pd.DataFrame(columns=['record_date','ex_date','payment_date','cash_dividend_per_share'])
    cfg={'initial_capital':100000.,'lot':100,'tick':.001,'limit_fraction':.1}
    cost={'commission':.0002,'minimum':5.,'slippage':.0005}
    return frame,dividends,cfg,cost


def test_original_rebound_entry_is_strict_deviation_and_first_recovery():
    frame=fixture()[0]
    frame.loc[2,'z20']=-1.5
    frame.loc[3,'wealth']=frame.loc[2,'wealth']-1
    frame.loc[4,'feature_valid']=False
    frame.loc[5,'z20']=0.
    rule=signals(frame)[AUX]
    assert rule['entry'][1]==1
    assert rule['entry'][0]==rule['entry'][2]==rule['entry'][3]==rule['entry'][4]==0
    assert rule['exit'][1][5] and not rule['exit'][1][1]


def test_reference_mean_exit_loss_time_limit_and_one_day_wait():
    frame,div,cfg,cost=fixture()
    rule={'entry':np.ones(len(frame),int),'exit':{1:np.zeros(len(frame),bool)}}
    rule['exit'][1][2]=True
    spec=specifications()[AUX]
    ledger,decisions,cycles=simulate_policy(frame,div,cfg,cost,str(frame.date.iloc[1].date()),rule,spec)
    assert cycles.iloc[0].exit_date==frame.date.iloc[3]
    assert decisions[decisions.origin_index.eq(3)].iloc[0].requested_quantity==0
    assert ledger[ledger.filled_quantity.gt(0)].date.iloc[1]==frame.date.iloc[5]
    frame,div,cfg,cost=fixture()
    frame.loc[2,'close']=9.4
    frame.loc[3,'previous_close']=9.4
    rule['exit'][1][:]=False
    ledger,decisions,cycles=simulate_policy(frame,div,cfg,cost,str(frame.date.iloc[1].date()),rule,spec)
    assert cycles.iloc[0].exit_date==frame.date.iloc[3] and '固定止损' in cycles.iloc[0].exit_reasons
    frame,div,cfg,cost=fixture()
    ledger,decisions,cycles=simulate_policy(frame,div,cfg,cost,str(frame.date.iloc[1].date()),rule,spec)
    assert cycles.iloc[0].exit_date==frame.date.iloc[11] and '最长持有' in cycles.iloc[0].exit_reasons


def test_reference_blocked_exit_stays_locked_after_condition_recovers():
    frame,div,cfg,cost=fixture()
    frame.loc[3,'open']=9.
    rule={'entry':np.ones(len(frame),int),'exit':{1:np.zeros(len(frame),bool)}}
    rule['exit'][1][2]=True
    ledger,decisions,cycles=simulate_policy(frame,div,cfg,cost,str(frame.date.iloc[1].date()),rule,specifications()[AUX])
    assert ledger.iloc[2].status=='UNFILLED_DIRECTIONAL_LIMIT'
    assert decisions[decisions.origin_index.eq(3)].iloc[0].requested_quantity<0
    assert cycles.iloc[0].exit_date==frame.date.iloc[4]
    assert (cycles.holding_intervals>=1).all() and ledger.iloc[-1].shares==0
