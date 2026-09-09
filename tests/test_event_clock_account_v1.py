"""检查公告时钟执行和无判断状态不会被解释成现金目标。"""
import numpy as np
import pandas as pd
from research.event_clock_account_v1 import simulate_event_account


def fixture():
    data=pd.DataFrame({"date":pd.bdate_range("2020-01-01",periods=5),"open":10.,"close":10.,"previous_close":10.,"dividend":0.,"variance60":.0001})
    dividends=pd.DataFrame(columns=["record_date","ex_date","payment_date","cash_dividend_per_share"])
    config={"initial_capital":20000.,"lot":100,"tick":.001,"limit_fraction":.1,"weight_band":.1}
    cost={"commission":0.,"minimum":0.,"slippage":0.}
    events=np.array([False,False,True,False,False])
    return data,dividends,config,cost,events


def test_only_announced_event_changes_next_open_shares():
    data,dividends,config,cost,events=fixture()
    ledger,decisions=simulate_event_account(data,dividends,config,cost,"2020-01-02","规则",
        targets=np.array([1.,0.,0.,1.,1.]),event_mask=events)
    assert ledger.filled_quantity.tolist()==[2000,0,-2000,0]
    assert decisions.origin.iloc[2]==pd.Timestamp("2020-01-03")
    assert decisions.execution_date.iloc[2]==pd.Timestamp("2020-01-06")


def test_no_view_keeps_existing_shares_and_unknown_model_target():
    data,dividends,config,cost,events=fixture()
    ledger,decisions=simulate_event_account(data,dividends,config,cost,"2020-01-02","规则",
        targets=np.array([1.,0.,np.nan,0.,0.]),event_mask=events)
    assert ledger.shares.tolist()==[2000,2000,2000,0]
    assert decisions.signal_state.iloc[2]=="NO_VIEW_KEEP_EXISTING_SHARES"
    assert pd.isna(decisions.reference_weight.iloc[2])
