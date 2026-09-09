"""检验无有效入场时保留真实空账户与明确零行交易循环。"""
import numpy as np
import pandas as pd
from research.forward_eps_corrected_source_pipeline_v2 import empty_cycle_schema


def test_no_entry_has_zero_cycles_and_unchanged_account():
    dates=pd.bdate_range('2025-01-01',periods=5)
    data=pd.DataFrame({'date':dates,'open':[10.]*5,'close':[10.]*5,'previous_close':[10.]*5,'dividend':[0.]*5,'sma120':[.1]*5})
    dividends=pd.DataFrame(columns=['record_date','ex_date','payment_date','cash_dividend_per_share'])
    config={'initial_capital':200000.,'lot':100,'tick':.001,'limit_fraction':.1,'loss_stop_fraction':.08,
            'trailing_stop_fraction':.12,'maximum_holding_trading_days':60,'reentry_cooldown_trading_days':5}
    cost={'commission':.0002,'minimum':5.,'slippage':.0005}
    ledger,decisions,cycles=empty_cycle_schema(data,dividends,config,cost,str(dates[1].date()),np.full(5,np.nan),np.array([True,False,False,False,False]))
    assert cycles.empty and {'exit_date','cycle_net_profit_cny'}<=set(cycles.columns)
    assert ledger.equity.eq(200000.).all() and ledger.shares.eq(0).all()
    assert decisions.requested_quantity.eq(0).all()
