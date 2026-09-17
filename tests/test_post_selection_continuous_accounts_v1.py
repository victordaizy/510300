"""验证原结算兼容、持仓中断续接、分红权利及内部退出计数。"""
import json
import numpy as np
import pandas as pd
import pytest

from research.adaptive_allocation_v1 import target_request
from research.event_account_indexed_request_v1 import simulate_indexed_request_account as old_target
from research.simple_price_entry_exit_v1 import simulate_policy as old_price
from research.rearmed_cycle_exit_account_v1 import simulate_rearmed_exit as old_rearmed
from research.post_selection_continuous_accounts_v1 import simulate_indexed_request_account,simulate_policy,simulate_rearmed_exit,unpack


class CountingExit:
    def __init__(self):
        self.count=0

    def __call__(self,t,cycle,current_value,peak_value):
        self.count+=1
        return {'negative_confirmation_count':self.count,'learned_exit_requested':self.count>=5}


def fixture():
    dates=pd.bdate_range('2026-08-03',periods=12)
    close=np.array([4.,4.01,4.02,3.92,3.94,3.96,3.98,4.,4.02,4.03,4.04,4.05])
    data=pd.DataFrame({'date':dates,'open':close,'close':close,'previous_close':np.r_[4.,close[:-1]],
        'dividend':np.r_[0.,0.,0.,.1,np.zeros(8)],'variance60':.0001})
    div=pd.DataFrame({'record_date':[dates[2]],'ex_date':[dates[3]],'payment_date':[dates[6]],'cash_dividend_per_share':[.1]})
    cfg={'initial_capital':200000.,'lot':100,'tick':.001,'limit_fraction':.1,'weight_band':.1}
    cost={'commission':.0004,'minimum':5.,'slippage':.001}
    rule={'entry':np.ones(len(data),int),'exit':{1:np.arange(len(data))>=8}}
    spec={'cooldown':2,'modes':{1:{'loss':.06,'trail':.08,'take':None,'days':60}}}
    return data,div,cfg,cost,rule,spec


def call(kind,old=False,**kwargs):
    data,div,cfg,cost,rule,spec=fixture()
    common=(data,div,cfg,cost,str(data.date.iloc[1].date()))
    if kind=='target':
        fn=old_target if old else simulate_indexed_request_account
        targets=np.r_[np.ones(7)*.75,np.nan,0.,np.ones(3)*.6]
        return fn(*common,'TEST_CONTINUOUS',targets=targets,event_mask=np.ones(len(data),bool),
            request_policy=lambda account,price,value,settings,model,t:target_request(account,price,value,settings),**kwargs)
    fn=(old_price if old else simulate_policy) if kind=='price' else (old_rearmed if old else simulate_rearmed_exit)
    return fn(*common,rule,spec,*([CountingExit()] if kind=='rearmed' else []),**kwargs)


@pytest.mark.parametrize('kind',['target','price','rearmed'])
def test_original_terminal_mode_matches_saved_engine(kind):
    expected=call(kind,old=True)
    actual=call(kind,terminal_liquidation=True)
    for left,right in zip(expected,actual):
        pd.testing.assert_frame_equal(left,right,check_exact=True)


@pytest.mark.parametrize('kind',['target','price','rearmed'])
def test_split_resume_matches_continuous_with_receivable_and_holdings(kind):
    full=call(kind,next_execution_date='2026-08-19')
    first=call(kind,stop_index=3)
    snapshot=json.loads(json.dumps(first[-1],ensure_ascii=False,allow_nan=False))
    state=unpack(snapshot['state'])
    assert state['account']['shares']>0 and sum(state['account']['receivables'].values())>0
    if kind=='rearmed':
        assert snapshot['controller']['fields']['count']==3
    resumed=call(kind,resume=snapshot,next_execution_date='2026-08-19')
    pd.testing.assert_frame_equal(pd.concat([first[0],resumed[0]],ignore_index=True),full[0],check_exact=True)
    pd.testing.assert_frame_equal(pd.concat([first[1],resumed[1]],ignore_index=True),full[1],check_exact=True)
    assert resumed[-1]==full[-1]
    assert full[0].mark_clock.eq('CLOSE').all()
    assert full[1].execution_date.iloc[-1]==pd.Timestamp('2026-08-19')


def test_terminal_snapshot_and_missing_next_date_rejected():
    snapshot=call('target',terminal_liquidation=True)[-1]
    with pytest.raises(ValueError,match='结算方式'):
        call('target',resume=snapshot,next_execution_date='2026-08-19')
    with pytest.raises(ValueError,match='下一交易日'):
        call('target')
