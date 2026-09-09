"""检验真实下一开盘退出、除息不误报、到期及受限退出继续请求。"""
import numpy as np
import pandas as pd
from research.explicit_entry_exit_account_v1 import simulate_entry_exit_account,exit_reasons


def setup(opening,closing,maximum=60):
    n=len(opening);dates=pd.bdate_range('2025-01-01',periods=n)
    frame=pd.DataFrame({'date':dates,'open':opening,'close':closing,'previous_close':[closing[0],*closing[:-1]],'dividend':np.zeros(n),'sma120':np.repeat(.1,n)})
    config={'initial_capital':200000.,'lot':100,'tick':.001,'limit_fraction':.1,'loss_stop_fraction':.08,
            'trailing_stop_fraction':.12,'maximum_holding_trading_days':maximum,'reentry_cooldown_trading_days':5}
    cost={'commission':0.,'minimum':0.,'slippage':0.}
    dividends=pd.DataFrame(columns=['record_date','ex_date','payment_date','cash_dividend_per_share'])
    signals=np.full(n,np.nan);signals[0]=1.;mask=np.zeros(n,bool);mask[0]=True
    return frame,dividends,config,cost,signals,mask


def run(parts):
    frame,dividends,config,cost,signals,mask=parts
    return simulate_entry_exit_account(frame,dividends,config,cost,str(frame.date.iloc[1].date()),signals,mask)


def test_stop_triggers_at_close_but_gap_realizes_worse_loss_next_open():
    parts=setup([10,10,8.6,8.6,8.6],[10,9.1,8.6,8.6,8.6])
    ledger,decisions,cycles=run(parts)
    assert ledger.iloc[0].shares==20000 and ledger.iloc[1].shares==0
    assert decisions.loc[decisions.new_exit_trigger,'origin_index'].tolist()==[1]
    assert np.isclose(cycles.iloc[0].cycle_net_return,-.14)


def test_limit_blocked_exit_remains_pending_after_recovery():
    parts=setup([10,10,8.19,9.8,9.8],[10,9.1,9.8,9.8,9.8])
    ledger,decisions,cycles=run(parts)
    assert ledger.iloc[1].status=='UNFILLED_DIRECTIONAL_LIMIT'
    assert ledger.iloc[2].filled_quantity==-20000
    assert decisions.loc[decisions.origin_index.eq(2),'signal_state'].iloc[0]=='EXPLICIT_EXIT_POLICY_WAITING_FOR_FILL'


def test_cash_dividend_entitlement_prevents_false_ex_dividend_stop():
    parts=list(setup([10,10,9,9,9],[10,10,9,9,9]))
    frame=parts[0];frame.loc[2,'dividend']=1.
    parts[1]=pd.DataFrame({'record_date':[frame.date.iloc[1]],'ex_date':[frame.date.iloc[2]],'payment_date':[frame.date.iloc[3]],'cash_dividend_per_share':[1.]})
    ledger,decisions,cycles=run(parts)
    assert not decisions.new_exit_trigger.any()
    assert np.isclose(cycles.iloc[0].cycle_net_return,0)
    assert ledger.dividend_recognized.sum()==20000


def test_holding_expiry_uses_trading_intervals_and_still_exits_with_missing_eps():
    parts=setup([10]*6,[10]*6,maximum=2)
    ledger,decisions,cycles=run(parts)
    assert cycles.iloc[0].holding_open_to_open_trading_intervals==2
    assert '最长持有期到期' in cycles.iloc[0].exit_reasons


def test_reentry_requires_cooldown_and_new_month_end_signal():
    parts=list(setup([10]*12,[10]*12,maximum=2))
    parts[4][4]=1.;parts[5][4]=True
    parts[4][8]=1.;parts[5][8]=True
    ledger,decisions,cycles=run(parts)
    assert decisions.loc[decisions.origin_index.eq(4),'requested_quantity'].iloc[0]==0
    assert decisions.loc[decisions.origin_index.eq(8),'requested_quantity'].iloc[0]>0


def test_trailing_exit_does_not_require_negative_total_trade_return():
    config=setup([10]*5,[10]*5)[2]
    reasons=exit_reasons(110,100,130,20,False,False,config)
    assert reasons==['持仓价值从最高值回落触及退出线']
