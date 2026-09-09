"""过去净夏普选择的时间、现金、缺失与实际成交检验。"""
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from research.past_sharpe_parent_selector_inputs_v1 import PARENTS,selection_frame
from research.event_clock_account_v1 import simulate_event_account


def fixture():
    dates=pd.bdate_range("2020-01-27",periods=75)
    sign=np.where(np.arange(75)%2,1.,-1.)
    returns=np.column_stack([.001+.004*sign,.003+.003*sign])
    targets=np.tile([.2,.8],(75,1))
    return dates,returns,targets


def test_initial_parent_monthly_selection_and_complete_window_score():
    dates,returns,targets=fixture();f=selection_frame(dates,returns,targets,1,5)
    month=dates.get_loc(pd.Timestamp("2020-02-03"))
    assert f.selected_parent.iloc[:month].eq(PARENTS[0]).all()
    assert f.selected_parent.iloc[month]==PARENTS[1] and f.selection_changed.iloc[month]
    scores=returns[1:month+1].mean(axis=0)/returns[1:month+1].std(axis=0,ddof=1)*np.sqrt(242)
    np.testing.assert_allclose(f.iloc[month][["two_past_sharpe","three_past_sharpe"]].to_numpy(float),scores,atol=1e-12,rtol=0)
    assert f.selection_window_start.iloc[month]==dates[1] and f.selection_window_observations.iloc[month]==5
    changed=returns.copy();changed[month+1]=[10.,-.8]
    same=selection_frame(dates,changed,targets,1,5)
    assert same.selected_parent.iloc[month+1]==PARENTS[1] and not same.selection_update_scheduled.iloc[month+1]


def test_negative_scores_choose_cash_and_later_positive_scores_allow_reentry():
    dates,returns,targets=fixture();march=dates.get_loc(pd.Timestamp("2020-03-02"));april=dates.get_loc(pd.Timestamp("2020-04-01"))
    returns[march-4:march+1]-=.04
    f=selection_frame(dates,returns,targets,1,5)
    assert f.selected_parent.iloc[march]=="CASH" and f.target.iloc[march]==0.
    assert f.selection_status.iloc[march]=="EXPLICIT_CASH_NONPOSITIVE_PAST_SHARPE"
    assert f.selected_parent.iloc[april]==PARENTS[1] and f.target.iloc[april]==.8


def test_equal_positive_scores_preserve_parent_and_use_two_after_cash():
    dates,returns,targets=fixture();march=dates.get_loc(pd.Timestamp("2020-03-02"));april=dates.get_loc(pd.Timestamp("2020-04-01"))
    returns[march-4:march+1,0]=returns[march-4:march+1,1]
    f=selection_frame(dates,returns,targets,1,5);assert f.selected_parent.iloc[march]==PARENTS[1]
    returns[march-4:march+1]-=.04;returns[april-4:april+1,0]=returns[april-4:april+1,1]
    f=selection_frame(dates,returns,targets,1,5)
    assert f.selected_parent.iloc[march]=="CASH" and f.selected_parent.iloc[april]==PARENTS[0]


def test_warmup_missing_and_zero_variance_do_not_become_cash_or_skip_days():
    dates,returns,targets=fixture();long=selection_frame(dates,returns,targets,1,242)
    assert long.selected_parent.iloc[:-1].eq(PARENTS[0]).all() and long.last_successful_selection_origin.isna().all()
    march=dates.get_loc(pd.Timestamp("2020-03-02"))
    for value,status in [(np.nan,"NO_VIEW_INCOMPLETE_WINDOW_KEEP_SELECTION"),(0.,"NO_VIEW_ZERO_OR_INVALID_VOLATILITY_KEEP_SELECTION")]:
        changed=returns.copy();changed[march-4:march+1,0]=value
        f=selection_frame(dates,changed,targets,1,5)
        assert f.selection_status.iloc[march]==status and f.selected_parent.iloc[march]==PARENTS[1]
        assert f.selection_window_observations.iloc[march]==5 and f.target.iloc[march]==.8


def test_missing_selected_target_keeps_no_view_unselected_missing_does_not_override():
    dates,returns,targets=fixture();targets[10,1]=np.nan;targets[11,0]=np.nan
    f=selection_frame(dates,returns,targets,1,5)
    assert pd.isna(f.target.iloc[10]) and f.target.iloc[11]==.8
    march=dates.get_loc(pd.Timestamp("2020-03-02"));returns[march-4:march+1]-=.04;targets[march]=np.nan
    f=selection_frame(dates,returns,targets,1,5);assert f.target.iloc[march]==0. and f.selected_parent.iloc[march]=="CASH"


def test_future_and_terminal_returns_cannot_change_past_selection():
    dates,returns,targets=fixture();original=selection_frame(dates,returns,targets,1,5)
    changed_r=returns.copy();changed_t=targets.copy();changed_r[25:]=[10.,-.5];changed_t[25:]=[1.,0.]
    changed=selection_frame(dates,changed_r,changed_t,1,5)
    assert_frame_equal(original.iloc[:25],changed.iloc[:25])
    prefix=selection_frame(dates[:26],returns[:26],targets[:26],1,5);assert_frame_equal(original.iloc[:25],prefix.iloc[:25])
    returns[-1]=100.;assert_frame_equal(original,selection_frame(dates,returns,targets,1,5))


def test_new_account_adjusts_own_shares_at_next_open_and_cash_exit_preserves_dividend():
    dates,returns,targets=fixture();march=dates.get_loc(pd.Timestamp("2020-03-02"));returns[march-4:march+1]-=.04
    data=pd.DataFrame({"date":dates,"open":10.,"close":10.,"previous_close":10.,"dividend":0.,"variance60":.001})
    cfg={"initial_capital":200000.,"lot":100,"tick":.001,"limit_fraction":.1,"weight_band":.1}
    cost={"commission":.0002,"minimum":5.,"slippage":.0005}
    div=pd.DataFrame({"record_date":[dates[7]],"ex_date":[dates[8]],"payment_date":[dates[9]],"cash_dividend_per_share":[.1]})
    data.loc[8:,["open","close","previous_close"]]=9.9;data.loc[8,["previous_close","dividend"]]=[10.,.1]
    f=selection_frame(dates,returns,targets,1,5)
    ledger,decisions=simulate_event_account(data,div,cfg,cost,str(dates[1].date()),"SYNTHETIC_PARENT_SELECTOR",targets=f.target.to_numpy(),event_mask=np.ones(len(dates),bool))
    month=dates.get_loc(pd.Timestamp("2020-02-03"));request=decisions[decisions.origin.eq(dates[month])].iloc[0]
    assert request.requested_quantity>0 and ledger[ledger.date.eq(dates[month+1])].filled_quantity.iloc[0]>0
    exit_request=decisions[decisions.origin.eq(dates[march])].iloc[0]
    assert exit_request.requested_quantity<0 and ledger[ledger.date.eq(dates[march+1])].shares.iloc[0]==0
    held=ledger[ledger.date.eq(dates[7])].shares.iloc[0];assert np.isclose(ledger.dividend_recognized.sum(),held*.1)
    assert ledger.shares.iloc[-1]==0 and ledger.accounting_error.abs().max()<1e-6
    assert ledger.commission.sum()>0 and ledger.slippage_cost.sum()>0
