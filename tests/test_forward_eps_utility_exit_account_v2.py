"""检验趋势进入与退出一致，以及已触发退出不被新预测或趋势恢复撤回。"""
import numpy as np
import pandas as pd
from research.forward_eps_utility_exit_account_v1 import simulate_utility_exit_account as original
from research.forward_eps_utility_exit_account_v2 import simulate_utility_exit_account as coherent


def inputs():
    dates=pd.bdate_range('2025-01-01',periods=12)
    data=pd.DataFrame({'date':dates,'open':10.,'close':10.,'previous_close':10.,'dividend':0.,'sma120':.1,'variance60':.0001})
    dividends=pd.DataFrame(columns=['record_date','ex_date','payment_date','cash_dividend_per_share'])
    config={'evaluation_start':str(dates[1].date()),'initial_capital':200000.,'lot':100,'tick':.001,'limit_fraction':.1,
            'gamma':4.,'weights':[0.,.25,.5,.75,1.],'horizon':60,'forecast_validity_trading_days':60}
    cost={'commission':0.,'minimum':0.,'slippage':0.}
    prediction=np.full(12,np.nan);prediction[[0,4,8]]=.1
    mask=np.isfinite(prediction)
    return data,dividends,config,cost,prediction,mask


def test_weak_known_trend_blocks_initial_entry_and_reentry_until_recovered():
    args=list(inputs());args[0].loc[:3,'sma120']=-.1
    args[0].loc[6:7,'sma120']=-.1
    ledger,decisions=coherent(*args,expiry_enabled=True,trend_enabled=True,entry_trend_required=True)
    assert decisions.loc[decisions.origin_index.eq(0),'requested_quantity'].iloc[0]==0
    assert decisions.loc[decisions.requested_quantity.gt(0),'origin_index'].tolist()==[4,8]
    assert decisions.loc[decisions.new_exit_trigger,'origin_index'].tolist()==[7]
    assert ledger.loc[ledger.date.eq(args[0].date.iloc[8]),'shares'].iloc[0]==0


def test_disabling_entry_gate_reproduces_frozen_previous_engine():
    args=list(inputs());args[0].loc[[0,1],'sma120']=-.1
    old,old_decisions=original(*args,expiry_enabled=True,trend_enabled=True)
    new,new_decisions=coherent(*args,expiry_enabled=True,trend_enabled=True,entry_trend_required=False)
    for field in ['equity','cash','shares','filled_quantity','net_return']:
        np.testing.assert_array_equal(old[field],new[field])
    np.testing.assert_array_equal(old_decisions.requested_quantity,new_decisions.requested_quantity)


def test_latched_exit_survives_new_positive_forecast_and_recovered_trend():
    args=list(inputs());args[0].loc[1:2,'sma120']=-.1
    args[0].loc[3,'open']=9.
    args[4][3]=.1;args[5][3]=True
    ledger,decisions=coherent(*args,expiry_enabled=True,trend_enabled=True,entry_trend_required=True)
    assert ledger.loc[ledger.date.eq(args[0].date.iloc[3]),'status'].iloc[0]=='UNFILLED_DIRECTIONAL_LIMIT'
    assert decisions.loc[decisions.origin_index.eq(3),'signal_state'].iloc[0]=='EXIT_PENDING_UNTIL_FILLED'
    assert ledger.loc[ledger.date.eq(args[0].date.iloc[4]),'shares'].iloc[0]==0
