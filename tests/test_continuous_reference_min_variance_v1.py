"""连续参考历史与新资金账户相互独立的必要检验。"""
import copy
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from research.continuous_reference_min_variance_v1 import reference_factors
from research.learned_cycle_exit_v1 import ExitController
from research.event_clock_account_v1 import simulate_event_account


def fixture():
    dates=pd.bdate_range("2020-01-27",periods=65);t=np.arange(len(dates))
    data=pd.DataFrame({"date":dates,"open":10.,"close":10.,"previous_close":10.,"dividend":0.,"variance60":.001,"mom5":0.,"mom20":0.,"sma120":0.,"vol20":.2})
    returns=np.column_stack([.001+.002*np.sin(t),.002+.004*np.cos(t)])
    references={}
    for column,model in enumerate(["PANIC_ONLY","REARM_RIDGE"]):
        ledger=pd.DataFrame({"date":dates[1:],"net_return":returns[1:,column],"equity":9000000.,"shares":900000})
        decisions=pd.DataFrame({"origin":dates[:-1],"execution_date":dates[1:],"reference_weight":1.})
        references[model]=(ledger,decisions)
    return data,references,{"risk_window":5}


def test_continuous_budget_has_prior_information_when_new_fund_starts():
    data,references,cfg=fixture();f=reference_factors(data,references,1,cfg)
    month=data.index[data.date.eq("2020-02-03")][0]
    assert f.risk_status.iloc[month]=="MIN_VARIANCE_BUDGET_AVAILABLE"
    assert f.risk_window_start.iloc[month]==data.date.iloc[1]
    start=12
    assert f.last_successful_risk_origin.iloc[start-1]==data.date.iloc[month]
    assert f.panic_budget.iloc[start-1]==f.panic_budget.iloc[month]
    assert f.risk_window_observations.iloc[start-1]==5


def test_prefix_and_future_returns_do_not_change_past_budget_or_reference_intent():
    data,references,cfg=fixture();full=reference_factors(data,references,1,cfg)
    prefix=reference_factors(data.iloc[:31],references,1,cfg)
    assert_frame_equal(full.iloc[:30],prefix.iloc[:30])
    changed=copy.deepcopy(references)
    for ledger,decisions in changed.values():
        ledger.loc[ledger.date.ge(data.date.iloc[30]),"net_return"]=-.9
        decisions.loc[decisions.origin.ge(data.date.iloc[30]),"reference_weight"]=0.
    f=reference_factors(data,changed,1,cfg)
    assert_frame_equal(full.iloc[:30],f.iloc[:30])
    assert pd.isna(prefix.target.iloc[-1]) and not prefix.risk_update_scheduled.iloc[-1]


def test_terminal_reference_close_cannot_enter_earlier_account_signals():
    data,references,cfg=fixture();prefix=reference_factors(data.iloc[:31],references,1,cfg)
    changed=copy.deepcopy(references)
    for ledger,decisions in changed.values():ledger.loc[ledger.date.eq(data.date.iloc[30]),"net_return"]=100.
    other=reference_factors(data.iloc[:31],changed,1,cfg)
    columns=[c for c in prefix if not c.endswith("reference_return")]
    assert_frame_equal(prefix[columns],other[columns])


def test_missing_intent_stays_no_view_and_missing_calendar_is_rejected():
    data,references,cfg=fixture();references["PANIC_ONLY"][1].loc[12,"reference_weight"]=np.nan
    f=reference_factors(data,references,1,cfg);assert pd.isna(f.target.iloc[12])
    ledger,decisions=references["PANIC_ONLY"];references["PANIC_ONLY"]=(ledger.drop(index=5),decisions)
    with pytest.raises(ValueError,match="完整日历"):reference_factors(data,references,1,cfg)


def test_saved_learning_model_is_unavailable_before_fit_and_requires_mature_cycles():
    data,references,cfg=fixture()
    model={"kind":"RIDGE","mean":[0.]*8,"scale":[1.]*8,"coefficients":[0.]*8,"intercept":-.01,"feature_clip":5.}
    stored=[{"fit_index":8,"latest_exit_index":8,"status":"FIT_COMPLETE","model":model}]
    cycle={"cycle_id":1,"entry_index":2,"entry_cost_cny":100.,"mode":1}
    controller=ExitController(data,stored,2)
    assert controller(5,cycle,100.,100.)["learning_status"]=="NO_VIEW_NO_MATURE_MODEL"
    assert controller(8,cycle,100.,100.)["negative_confirmation_count"]==1
    assert controller(9,cycle,100.,100.)["learned_exit_requested"]
    invalid=copy.deepcopy(stored);invalid[0]["latest_exit_index"]=9
    with pytest.raises(ValueError,match="未来周期"):ExitController(data,invalid,2)(8,cycle,100.,100.)


def test_new_account_uses_new_cash_without_inheriting_reference_shares_or_dividends():
    data,references,cfg=fixture();f=reference_factors(data,references,1,cfg)
    dates=data.date;data.loc[8:,["open","close","previous_close"]]=9.9;data.loc[8,["previous_close","dividend"]]=[10.,.1]
    div=pd.DataFrame({"record_date":[dates.iloc[7]],"ex_date":[dates.iloc[8]],"payment_date":[dates.iloc[15]],"cash_dividend_per_share":[.1]})
    account={"initial_capital":200000.,"lot":100,"tick":.001,"limit_fraction":.1,"weight_band":.1}
    cost={"commission":.0002,"minimum":5.,"slippage":.0005};start=12
    ledger,decisions=simulate_event_account(data,div,account,cost,str(dates.iloc[start].date()),"NEW_CASH_CONTINUOUS_REFERENCE",targets=f.target.to_numpy(),event_mask=np.ones(len(data),bool))
    assert ledger.shares_before.iloc[0]==0 and 0<ledger.shares.iloc[0]<21000
    assert ledger.equity.iloc[0]<200000. and ledger.dividend_recognized.sum()==0.
    assert decisions.origin.iloc[0]==dates.iloc[start-1] and decisions.execution_date.iloc[0]==dates.iloc[start]
    assert ledger.shares.iloc[-1]==0 and not ledger.terminal_unliquidated.iloc[-1]
    assert ledger.accounting_error.abs().max()<1e-6
