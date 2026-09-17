"""验证记录型训练基准、成熟时点和新模型出现后的持仓版本锁定。"""
import json
import numpy as np
import pandas as pd

from tests.test_post_selection_continuous_accounts_v1 import fixture
from research.september_monthly_training_v1 import reference_samples
from research.learned_cycle_exit_account_v1 import simulate_learned_exit
from research.learned_cycle_exit_v1 import FEATURES,state_values,training_rows
from research.entry_vintage_exit_inputs_v1 import EntryVintageExitController
from research.post_selection_continuous_accounts_v1 import pack,unpack


def test_recording_reference_matches_original_normal_account_and_features():
    data,div,cfg,cost,rule,spec=fixture()
    for c in ['mom5','mom20','sma120']:data[c]=.01
    data['vol20']=.12
    cfg.update(costs={'BASE':cost},reference_start=str(data.date.iloc[1].date()),candidate_specs={'D60_INTRA':spec})
    def recorder(t,cycle,current,peak):
        return {'learning_cycle_id':cycle['cycle_id'],'learned_exit_requested':False,**dict(zip(FEATURES,state_values(data,t,cycle,current,peak)))}
    old=simulate_learned_exit(data,div,cfg,cost,cfg['reference_start'],rule,spec,recorder)
    new=reference_samples(data,div,cfg,rule,str(data.date.iloc[-1].date()),'2026-08-19')
    pd.testing.assert_frame_equal(old[0].iloc[:-1],new[1].iloc[:-1],check_exact=True)
    left,right=old[1],new[2].iloc[:-1]
    np.testing.assert_allclose(right[FEATURES+['learning_cycle_id']].to_numpy(float),left[FEATURES+['learning_cycle_id']].to_numpy(float),rtol=0,atol=0,equal_nan=True)
    assert (new[-1].exit_index<=len(data)-1).all() and (new[-1].early_exit_index<new[-1].exit_index).all()


def test_training_excludes_later_finished_cycle():
    samples=pd.DataFrame({'cycle_id':[1,1,2,3],'origin_index':[2,3,5,6],'exit_index':[10,10,20,21]})
    rows,ids=training_rows(samples,20,{'recent_cycles':20})
    assert ids==[1,2] and 3 not in rows.cycle_id.tolist()
    np.testing.assert_array_equal(rows.groupby('cycle_id').sample_weight.sum(),[1.,1.])


def test_restored_open_cycle_keeps_entry_model_after_new_month_record():
    dates=pd.bdate_range('2026-08-27',periods=6)
    data=pd.DataFrame({'date':dates,'mom5':.01,'mom20':.02,'sma120':.03,'vol20':.1})
    def record(index,intercept):
        return {'fit_index':index,'latest_exit_index':index,'status':'FIT_COMPLETE','model':{'kind':'WITHIN_CYCLE_FIXED_INTERCEPT_RIDGE',
            'features':FEATURES.copy(),'mean':[0.]*8,'scale':[1.]*8,'coefficients':[0.]*8,'intercept':intercept,'feature_clip':5.}}
    old=record(0,.01);new=record(3,-.02)
    controller=EntryVintageExitController(data,[old],2)
    cycle={'cycle_id':1,'entry_index':1,'entry_cost_cny':100.,'mode':1}
    first=controller(1,cycle,101.,102.)
    fields=pack({k:v for k,v in controller.__dict__.items() if k not in ['data','models','indexes']})
    restored=EntryVintageExitController(data,[old,new],2)
    restored.__dict__.update(unpack(json.loads(json.dumps(fields))))
    after=restored(4,cycle,103.,104.)
    assert after['fixed_prediction_identity']==first['fixed_prediction_identity'] and after['continuation_prediction']==.01
    later=restored(5,{'cycle_id':2,'entry_index':5,'entry_cost_cny':100.,'mode':1},100.,100.)
    assert later['continuation_prediction']==-.02 and later['fixed_prediction_identity']!=first['fixed_prediction_identity']
