"""三条既有策略的单一非负最小方差预算，无收益参数搜索。"""
from __future__ import annotations
from itertools import combinations
import numpy as np
import pandas as pd
from research.intraday_overnight_increment_v1 import require

NAMES=["panic","learned","breakout"]


def minimum_variance(covariance):
    covariance=np.asarray(covariance,dtype=float)
    require(covariance.shape==(3,3) and np.isfinite(covariance).all(),"三维协方差输入不完整")
    require(np.allclose(covariance,covariance.T,atol=1e-15,rtol=0),"协方差不对称")
    scale=float(np.max(np.diag(covariance)))
    if scale<=0:
        return {"status":"NO_VIEW_ZERO_OR_INVALID_RISK_KEEP_BUDGET","weights":None,"minimum_scaled_eigenvalue":None}
    matrix=covariance/scale
    eigenvalues=np.linalg.eigvalsh(matrix)
    if eigenvalues[0]<=np.finfo(float).eps*3*max(float(eigenvalues[-1]),1.):
        return {"status":"NO_VIEW_SINGULAR_COVARIANCE_KEEP_BUDGET","weights":None,"minimum_scaled_eigenvalue":float(eigenvalues[0])}
    # 三个单策略端点、三个双策略边界和三策略内部，覆盖同一个凸目标的全部可能最优位置。
    candidates=[]
    for count in [1,2,3]:
        for active in combinations(range(3),count):
            part=matrix[np.ix_(active,active)]
            inverse_one=np.linalg.solve(part,np.ones(count))
            local=inverse_one/inverse_one.sum()
            if (local>=-1e-12).all():
                local=np.maximum(local,0.);local/=local.sum()
                weights=np.zeros(3);weights[list(active)]=local
                candidates.append((float(weights@matrix@weights),weights,active))
    objective,weights,active=min(candidates,key=lambda x:x[0])
    marginal=matrix@weights
    require(np.min(marginal-objective)>=-1e-10 and np.max(np.abs((marginal-objective)*weights))<1e-10,"三维预算不满足全局最优条件")
    return {"status":"MIN_VARIANCE_BUDGET_AVAILABLE","weights":weights,"portfolio_variance":objective*scale,
        "active_strategies":"+".join(NAMES[i] for i in active if weights[i]>0),"minimum_scaled_eigenvalue":float(eigenvalues[0])}


def budget_frame(dates,reference_returns,expert_states,first,window=242):
    dates=pd.DatetimeIndex(dates); returns=np.asarray(reference_returns,dtype=float);states=np.asarray(expert_states,dtype=float)
    require(dates.is_monotonic_increasing and not dates.has_duplicates,"三策略日期必须完整递增")
    require(returns.shape==states.shape==(len(dates),3),"三条参考策略的形状或日历不符")
    require(1<=first<len(dates)-1 and isinstance(window,int) and window>=4,"评价起点或三维风险窗口不合法")
    require((np.isnan(returns)|(np.isfinite(returns)&(returns>-1))).all(),"参考净收益非法")
    require((np.isnan(states)|np.isin(states,[0.,1.])).all(),"参考交易意向必须为零、一或缺失")
    weights=np.full(3,1/3); covariance=np.full((3,3),np.nan);sd=np.full(3,np.nan)
    status="INITIAL_EQUAL_BUDGET_NO_RISK_ESTIMATE";attempted=successful=start=pd.NaT;count=0
    variance=eigenvalue=np.nan;active=None;rows=[]
    for t in range(len(dates)):
        outside=t<first-1 or t==len(dates)-1;scheduled=False
        if not outside and t>=first and dates[t].to_period("M")!=dates[t-1].to_period("M"):
            scheduled=True;attempted=dates[t];count=min(window,t-first+1);start=dates[t-count+1]
            values=returns[t-count+1:t+1];sd=np.full(3,np.nan);covariance=np.full((3,3),np.nan);variance=eigenvalue=np.nan;active=None
            if count<window:
                status="NO_VIEW_WARMUP_KEEP_BUDGET"
            elif not np.isfinite(values).all():
                status="NO_VIEW_INCOMPLETE_WINDOW_KEEP_BUDGET"
            else:
                sd=np.std(values,axis=0,ddof=1)
                if not np.isfinite(sd).all() or (sd<=0).any():
                    status="NO_VIEW_ZERO_OR_INVALID_RISK_KEEP_BUDGET"
                else:
                    centered=values-values.mean(axis=0);covariance=centered.T@centered/(window-1)
                    try:
                        result=minimum_variance(covariance);status=result["status"]
                        eigenvalue=result["minimum_scaled_eigenvalue"]
                        if result["weights"] is not None:
                            weights=result["weights"];variance=result["portfolio_variance"];active=result["active_strategies"];successful=dates[t]
                    except np.linalg.LinAlgError:
                        status="NO_VIEW_OPTIMIZER_FAILED_KEEP_BUDGET"
        require(np.isfinite(weights).all() and (weights>=0).all() and np.isclose(weights.sum(),1.),"三预算超出无杠杆资金范围")
        target=float(states[t]@weights) if not outside and np.isfinite(states[t]).all() else np.nan
        row={"date":dates[t],"risk_status":"NO_VIEW_OUTSIDE_DECISION_PERIOD" if outside else status,"risk_update_scheduled":scheduled,
            "risk_attempt_origin":pd.NaT if outside else attempted,"last_successful_risk_origin":pd.NaT if outside else successful,
            "risk_window_start":pd.NaT if outside else start,"risk_window_observations":0 if outside else count,
            "estimated_portfolio_variance":np.nan if outside else variance,"minimum_scaled_eigenvalue":np.nan if outside else eigenvalue,
            "active_strategies":None if outside else active,"target":target}
        for i,name in enumerate(NAMES):
            row[name+"_budget"]=np.nan if outside else weights[i]
            row[name+"_sd"]=np.nan if outside else sd[i]
            row[name+"_state"]=states[t,i]
        for i,j in [(0,1),(0,2),(1,2)]:row[NAMES[i]+"_"+NAMES[j]+"_covariance"]=np.nan if outside else covariance[i,j]
        rows.append(row)
    return pd.DataFrame(rows)
