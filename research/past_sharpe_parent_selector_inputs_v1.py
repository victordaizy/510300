"""按当时已实现的完整账户净夏普选择两套原组合或显式现金。"""
from __future__ import annotations
import numpy as np
import pandas as pd
from research.intraday_overnight_increment_v1 import require

PARENTS=["TWO_POLICY_MIN_VARIANCE","THREE_POLICY_MIN_VARIANCE"]


def selection_frame(dates,reference_returns,parent_targets,first,window=242,annual_days=242):
    dates=pd.DatetimeIndex(dates);returns=np.asarray(reference_returns,dtype=float);targets=np.asarray(parent_targets,dtype=float)
    require(dates.is_monotonic_increasing and not dates.has_duplicates,"父组合日期重复或未递增")
    require(returns.shape==targets.shape==(len(dates),2),"父组合收益或目标维度不符")
    require(1<=first<len(dates)-1 and isinstance(window,int) and window>=2,"选择起点或窗口不合法")
    require((np.isnan(returns)|(np.isfinite(returns)&(returns>-1))).all(),"父组合完整收益存在非法值")
    require((np.isnan(targets)|(np.isfinite(targets)&(targets>=0)&(targets<=1))).all(),"父组合目标超出无杠杆范围")
    selected=0;status="INITIAL_TWO_POLICY_PARENT";scores=np.full(2,np.nan);count=0
    attempted=successful=start=pd.NaT;rows=[]
    for t in range(len(dates)):
        outside=t<first-1 or t==len(dates)-1;scheduled=False;changed=False
        if not outside and t>=first and dates[t].to_period("M")!=dates[t-1].to_period("M"):
            scheduled=True;attempted=dates[t];count=min(window,t-first+1);start=dates[t-count+1]
            values=returns[t-count+1:t+1];scores=np.full(2,np.nan);prior=selected
            if count<window:
                status="NO_VIEW_WARMUP_KEEP_SELECTION"
            elif not np.isfinite(values).all():
                status="NO_VIEW_INCOMPLETE_WINDOW_KEEP_SELECTION"
            else:
                deviation=np.std(values,axis=0,ddof=1)
                if not np.isfinite(deviation).all() or (deviation<=0).any():
                    status="NO_VIEW_ZERO_OR_INVALID_VOLATILITY_KEEP_SELECTION"
                else:
                    scores=np.mean(values,axis=0)/deviation*np.sqrt(annual_days)
                    require(np.isfinite(scores).all(),"历史净夏普出现非法数")
                    if max(scores)<=0:
                        selected=-1;status="EXPLICIT_CASH_NONPOSITIVE_PAST_SHARPE"
                    elif scores[0]==scores[1]:
                        selected=prior if prior>=0 else 0;status="EQUAL_POSITIVE_SCORE_KEEP_PARENT_OR_TWO_AFTER_CASH"
                    else:
                        selected=int(np.argmax(scores));status="POSITIVE_PAST_SHARPE_PARENT_SELECTED"
                    successful=dates[t]
            changed=selected!=prior
        name="CASH" if selected<0 else PARENTS[selected]
        target=np.nan if outside else 0. if selected<0 else float(targets[t,selected])
        rows.append({"date":dates[t],"selected_parent":None if outside else name,"target":target,
            "two_parent_target":targets[t,0],"three_parent_target":targets[t,1],
            "two_past_sharpe":np.nan if outside else scores[0],"three_past_sharpe":np.nan if outside else scores[1],
            "selection_status":"NO_VIEW_OUTSIDE_DECISION_PERIOD" if outside else status,"selection_update_scheduled":scheduled,"selection_changed":changed,
            "selection_attempt_origin":pd.NaT if outside else attempted,"last_successful_selection_origin":pd.NaT if outside else successful,
            "selection_window_start":pd.NaT if outside else start,"selection_window_observations":0 if outside else count})
    return pd.DataFrame(rows)
