"""验证限价的事前约束、到期、除息撤单及完整账户。"""
import numpy as np
import pandas as pd
import pytest
from research.limit_entry_account_v1 import simulate_limit_policy


@pytest.fixture
def base():
    n=8
    data=pd.DataFrame({'date':pd.bdate_range('2020-01-01',periods=n),'open':100.,'close':100.,'low':100.,'previous_close':100.,'dividend':0.})
    div=pd.DataFrame(columns=['record_date','ex_date','payment_date','cash_dividend_per_share'])
    cfg={'initial_capital':200000.,'lot':100,'tick':.001,'limit_fraction':.1}
    cost={'commission':.0002,'minimum':5.,'slippage':0.}
    rule={'entry':np.array([1,0,0,0,0,0,0,0]),'exit':{1:np.zeros(n,bool)}}
    spec={'cooldown':2,'modes':{1:{'loss':None,'trail':None,'take':None,'days':None}}}
    return data,div,cfg,cost,rule,spec


def run(base,assumption='PENETRATION',days=1):
    data,div,cfg,cost,rule,spec=base
    return simulate_limit_policy(data,div,cfg,cost,str(data.date.iloc[1].date()),rule,spec,{'discount':.01,'valid_days':days},assumption)


def test_touch_alone_does_not_fill(base):
    base[0].loc[1,'low']=99.
    ledger,_,_,orders=run(base)
    assert ledger.filled_quantity.sum()==0
    assert orders.final_status.tolist()==['EXPIRED_WITHOUT_FILL']


def test_intraday_fill_at_predefined_limit_and_next_day_exit(base):
    base[0].loc[1,'low']=98.997
    base[4]['exit'][1][1]=True
    ledger,_,cycles,orders=run(base)
    assert ledger.loc[ledger.filled_quantity>0,'fill_price'].tolist()==[99.]
    assert cycles.entry_date.iloc[0]==base[0].date.iloc[1]
    assert cycles.exit_date.iloc[0]==base[0].date.iloc[2]
    assert orders.quantity.iloc[0]==2000
    assert ledger.equity.iloc[-1]==pytest.approx(201920.4)
    assert ledger.accounting_error.abs().max()<1e-7


def test_open_only_cannot_use_intraday_low(base):
    base[0].loc[1,'low']=98.
    ledger,_,_,_=run(base,assumption='OPEN_ONLY')
    assert (ledger.filled_quantity==0).all()


def test_gap_open_can_fill_below_predefined_limit(base):
    base[0].loc[1,['open','low']]=98.
    ledger,_,_,orders=run(base,assumption='OPEN_ONLY')
    buy=ledger.loc[ledger.filled_quantity>0].iloc[0]
    assert buy.fill_price==98.
    assert buy.filled_quantity==2000
    assert orders.limit_price.iloc[0]==99.


def test_three_day_expiry_prevents_later_fill(base):
    base[0].loc[4,'low']=98.
    ledger,_,_,orders=run(base,days=3)
    assert (ledger.filled_quantity==0).all()
    assert orders.end_index.iloc[0]==3


def test_ex_dividend_cancels_waiting_order(base):
    base[0].loc[2,['low','dividend']]=[98.,1.]
    ledger,_,_,orders=run(base,days=3)
    assert (ledger.filled_quantity==0).all()
    assert orders.final_status.iloc[0]=='CANCELLED_ON_EX_DIVIDEND_DATE'


def test_blocked_exit_remains_after_condition_disappears(base):
    base[0].loc[1,'low']=98.997
    base[0].loc[2,['open','low']]=90.
    base[4]['exit'][1][1]=True
    ledger,_,cycles,_=run(base)
    assert ledger.loc[ledger.date==base[0].date.iloc[2],'status'].iloc[0]=='UNFILLED_DIRECTIONAL_LIMIT'
    assert cycles.exit_date.iloc[0]==base[0].date.iloc[3]


def test_slippage_is_charged_without_exceeding_limit(base):
    base[3]['slippage']=.001
    base[0].loc[1,'low']=98.89
    ledger,_,_,_=run(base)
    buy=ledger.loc[ledger.filled_quantity>0].iloc[0]
    assert buy.fill_price<=99.
    assert buy.slippage_cost>0
    assert ledger.accounting_error.abs().max()<1e-7
