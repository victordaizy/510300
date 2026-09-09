"""检验整周期本金损失、分红成熟和每月预算。"""
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from research.cycle_adverse_risk_inputs_v1 import extract_cycle_risks,cycle_risk_budget_frame


def account_fixture(ex_index=3):
    dates=pd.bdate_range("2020-01-02",periods=8);price=np.array([10.,10.,9.9,9.5,10.,10.,10.,10.])
    qty=np.array([0,100,0,0,-100,0,0,0]);shares=np.cumsum(qty);fee=np.where(qty!=0,5.,0.)
    paid=np.zeros(8);paid[ex_index]=10.;trade=-qty*price-fee
    ledger=pd.DataFrame({"date":dates,"origin":dates,"filled_quantity":qty,"shares_before":np.r_[0,shares[:-1]],"shares":shares,
        "mark_clock":"CLOSE","mark":price,"open_price":price,"fill_price":price,"commission":fee,"slippage_cost":0.,
        "dividend_recognized":paid,"equity":2000.+np.cumsum(trade)+shares*price+np.cumsum(paid)})
    record=2 if ex_index==3 else 3
    div=pd.DataFrame({"record_date":[dates[record]],"ex_date":[dates[ex_index]],"payment_date":[dates[7]],"cash_dividend_per_share":[.1]})
    return ledger,div,{"initial_capital":2000.}


def budget_fixture():
    dates=pd.bdate_range("2020-01-27",periods=65)
    cycles=pd.DataFrame({"model":["PANIC_ONLY","REARM_RIDGE"],"cycle":[1,1],"exit_date":[dates[5],dates[6]],
        "maturity_date":[dates[5],dates[6]],"terminal_exit":[False,False],"status":["COMPLETE_CYCLE_RISK"]*2,"maximum_capital_loss":[.08,.02]})
    cfg={"prior_loss_fractions":{"PANIC_ONLY":.04,"REARM_RIDGE":.06}}
    states=np.tile([1.,0.],(len(dates),1))
    return dates,cycles,states,cfg


def test_cycle_loss_uses_actual_entry_debit_close_marks_costs_and_dividend_once():
    ledger,div,cfg=account_fixture();c=extract_cycle_risks(ledger,div,cfg,"PANIC_ONLY").iloc[0]
    assert c.entry_capital==1005. and c.cycle_net_profit==0. and c.dividend_recognized==10.
    assert np.isclose(c.maximum_capital_loss,45./1005.)
    assert c.worst_loss_date==ledger.date.iloc[3] and c.maturity_date==ledger.date.iloc[4]


def test_post_exit_dividend_waits_for_recognition_and_terminal_cycle_is_excluded():
    ledger,div,cfg=account_fixture(5);c=extract_cycle_risks(ledger,div,cfg,"PANIC_ONLY").iloc[0]
    assert c.maturity_date==ledger.date.iloc[5] and c.exit_date==ledger.date.iloc[4]
    assert np.isclose(c.maximum_capital_loss,55./1005.) and c.cycle_net_profit==0.
    ledger,div,cfg=account_fixture();short=ledger.iloc[:5].copy();short.loc[short.index[-1],"mark_clock"]="OPEN_TERMINAL"
    c=extract_cycle_risks(short,div,cfg,"PANIC_ONLY").iloc[0]
    assert c.status=="NO_VIEW_TERMINAL_CYCLE" and pd.isna(c.maximum_capital_loss)


def test_prior_is_one_original_stop_observation_and_only_mature_cycles_update_it():
    dates,cycles,states,cfg=budget_fixture();f=cycle_risk_budget_frame(dates,cycles,states,1,cfg)
    assert np.isclose(f.panic_budget.iloc[0],.6)
    month=5;rp=np.sqrt((.04**2+.08**2)/2);rl=.06
    assert f.risk_update_scheduled.iloc[month] and f.panic_mature_cycles.iloc[month]==1 and f.learned_mature_cycles.iloc[month]==0
    assert np.isclose(f.panic_budget.iloc[month],rl/(rp+rl))
    assert f.learned_mature_cycles.iloc[6]==0 and not f.risk_update_scheduled.iloc[6]
    march=dates.get_loc(pd.Timestamp("2020-03-02"));rl=np.sqrt((.06**2+.02**2)/2)
    assert f.learned_mature_cycles.iloc[march]==1 and np.isclose(f.panic_budget.iloc[march],rl/(rp+rl))


def test_zero_loss_completed_cycles_are_kept_and_terminal_cycle_cannot_change_budget():
    dates,cycles,states,cfg=budget_fixture();cycles.loc[:,"maximum_capital_loss"]=0.
    f=cycle_risk_budget_frame(dates,cycles,states,1,cfg)
    assert f.panic_mature_cycles.iloc[5]==1 and np.isclose(f.panic_cycle_risk.iloc[5],.04/np.sqrt(2))
    altered=pd.concat([cycles,pd.DataFrame({"model":["PANIC_ONLY"],"cycle":[2],"exit_date":[dates[3]],"maturity_date":[dates[3]],"terminal_exit":[True],"status":["NO_VIEW_TERMINAL_CYCLE"],"maximum_capital_loss":[np.nan]})],ignore_index=True)
    assert_frame_equal(f,cycle_risk_budget_frame(dates,altered,states,1,cfg))


def test_unknown_mature_risk_keeps_budget_and_missing_intent_is_not_cash():
    dates,cycles,states,cfg=budget_fixture();cycles.loc[0,"maximum_capital_loss"]=np.nan;states[7,1]=np.nan
    f=cycle_risk_budget_frame(dates,cycles,states,1,cfg)
    assert f.risk_status.iloc[5]=="NO_VIEW_INCOMPLETE_MATURE_CYCLE_KEEP_BUDGET" and np.isclose(f.panic_budget.iloc[5],.6)
    assert f.panic_mature_cycles.iloc[5]==1 and pd.isna(f.target.iloc[7])


def test_future_cycle_outcomes_and_later_data_do_not_change_prior_budgets():
    dates,cycles,states,cfg=budget_fixture();f=cycle_risk_budget_frame(dates,cycles,states,1,cfg)
    late=cycles.copy();late.loc[1,"maximum_capital_loss"]=.9
    changed=cycle_risk_budget_frame(dates,late,states,1,cfg)
    assert_frame_equal(f.iloc[:6],changed.iloc[:6])
    prefix=cycle_risk_budget_frame(dates[:31],cycles,states[:31],1,cfg)
    assert_frame_equal(f.iloc[:30],prefix.iloc[:30])
    assert pd.isna(prefix.target.iloc[-1]) and not prefix.risk_update_scheduled.iloc[-1]
