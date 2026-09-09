"""从真实完整持仓周期提取本金损失，只在权益成熟后更新风险预算。"""
from __future__ import annotations
import numpy as np
import pandas as pd
from research.intraday_overnight_increment_v1 import require
from scripts.review_round74_saved import saved_cycles

MODELS=["PANIC_ONLY","REARM_RIDGE"]


def extract_cycle_risks(ledger,dividends,cfg,model):
    require(model in MODELS,"未知周期风险策略")
    require(ledger.date.is_monotonic_increasing and not ledger.date.duplicated().any(),"周期风险账户日历不完整")
    complete=saved_cycles(ledger,dividends,cfg);out=[];by_date=ledger.set_index("date")
    credited=pd.Series(0.,index=by_date.index)
    for c in complete:
        require(c["buy_trades"]==1 and c["sell_trades"]==1,"本轮只允许原参考一次买入及全部退出的完整周期")
        entry,exit_date=pd.Timestamp(c["entry_date"]),pd.Timestamp(c["exit_date"])
        events=dividends[dividends.record_date.ge(entry)&dividends.record_date.lt(exit_date)].copy()
        income={};maturity=exit_date;all_observed=True
        for event in events.itertuples():
            require(event.record_date in by_date.index,"周期分红登记日缺失")
            quantity=int(by_date.loc[event.record_date,"shares"])
            require(quantity>0,"周期分红登记日没有实际份额")
            amount=quantity*event.cash_dividend_per_share
            maturity=max(maturity,pd.Timestamp(event.ex_date))
            if event.ex_date not in by_date.index:all_observed=False
            else:
                income[event.ex_date]=income.get(event.ex_date,0.)+amount
                credited.loc[event.ex_date]+=amount
        rows=ledger[ledger.date.ge(entry)&ledger.date.le(maturity)]
        trade_cash=0.;dividend_value=0.;pnl_path=[]
        for row in rows.itertuples():
            if row.date<=exit_date and row.filled_quantity!=0:
                trade_cash-=row.filled_quantity*row.fill_price+row.commission
            dividend_value+=income.get(row.date,0.)
            stock=row.shares*row.mark if row.date<exit_date else 0.
            pnl_path.append((row.date,trade_cash+dividend_value+stock))
        require(len(pnl_path)>0 and c["buy_debit"]>0,"周期本金或估值路径无效")
        terminal_pnl=pnl_path[-1][1]
        require(abs(terminal_pnl-c["net_profit"])<1e-6,"周期风险路径和完整含分红净利润不符")
        worst=min(pnl_path,key=lambda x:x[1]);loss=max(0.,-worst[1]/c["buy_debit"])
        status="COMPLETE_CYCLE_RISK" if all_observed and not c["terminal_exit"] else "NO_VIEW_TERMINAL_CYCLE" if c["terminal_exit"] else "NO_VIEW_INCOMPLETE_DIVIDEND_RECOGNITION"
        out.append({"model":model,"cycle":c["cycle"],"entry_date":entry,"exit_date":exit_date,"maturity_date":maturity,
            "terminal_exit":c["terminal_exit"],"status":status,"entry_capital":c["buy_debit"],"held_closes":c["held_closes"],
            "cycle_net_profit":c["net_profit"],"cycle_net_return":c["cycle_net_return"],"dividend_recognized":c["dividend_recognized"],
            "maximum_capital_loss":loss if all_observed and not c["terminal_exit"] else np.nan,
            "observed_maximum_capital_loss":loss,"worst_loss_date":worst[0]})
    np.testing.assert_allclose(credited.to_numpy(),ledger.dividend_recognized.to_numpy(),rtol=0,atol=1e-6)
    return pd.DataFrame(out)


def cycle_risk_budget_frame(dates,cycles,states,first,cfg):
    dates=pd.DatetimeIndex(dates);states=np.asarray(states,float)
    require(dates.is_monotonic_increasing and not dates.has_duplicates and 1<=first<len(dates)-1,"周期风险日期或起点错误")
    require(states.shape==(len(dates),2) and (np.isnan(states)|np.isin(states,[0.,1.])).all(),"周期风险持有意向无效")
    require(set(cycles.model.unique()).issubset(MODELS),"周期风险表含额外策略")
    require(not cycles.duplicated(["model","cycle"]).any() and cycles.maturity_date.notna().all(),"周期风险重复或成熟日期未知")
    require((cycles.maturity_date>=cycles.exit_date).all(),"周期风险早于退出就成熟")
    priors=np.array([cfg["prior_loss_fractions"][m] for m in MODELS],float)
    require(np.isfinite(priors).all() and (priors>0).all(),"原止损风险先验必须为正")
    risks=priors.copy();weights=(1/risks)/(1/risks).sum();counts=np.zeros(2,int);last=pd.NaT;status="EXPLICIT_ORIGINAL_STOP_RISK_PRIOR"
    observations=np.full(2,np.nan);latest=[pd.NaT,pd.NaT];rows=[]
    for t,date in enumerate(dates):
        outside=t<first-1 or t==len(dates)-1;scheduled=False
        if not outside and t>=first and date.to_period("M")!=dates[t-1].to_period("M"):
            scheduled=True;values=[];counts=[];latest=[];observations=[];valid=True
            for model,prior in zip(MODELS,priors,strict=True):
                chosen=cycles[cycles.model.eq(model)&~cycles.terminal_exit&cycles.maturity_date.le(date)]
                counts.append(len(chosen));latest.append(chosen.maturity_date.max() if len(chosen) else pd.NaT)
                a=chosen.maximum_capital_loss.to_numpy(float)
                okay=np.isfinite(a).all() and (a>=0).all() and chosen.status.eq("COMPLETE_CYCLE_RISK").all()
                valid=valid and okay
                observations.append(float(np.square(a).sum()) if okay else np.nan)
                values.append(float(np.sqrt((prior**2+np.square(a).sum())/(1+len(a)))) if okay else np.nan)
            if valid:
                risks=np.array(values);weights=(1/risks)/(1/risks).sum();last=date
                status="MATURE_CYCLE_RISK_BUDGET_AVAILABLE" if sum(counts) else "EXPLICIT_ORIGINAL_STOP_RISK_PRIOR"
            else:status="NO_VIEW_INCOMPLETE_MATURE_CYCLE_KEEP_BUDGET"
        require(np.isfinite(weights).all() and (weights>0).all() and np.isclose(weights.sum(),1.),"周期预算无效")
        target=np.nan if outside or not np.isfinite(states[t]).all() else float(states[t]@weights)
        rows.append({"date":date,"panic_state":states[t,0],"learned_state":states[t,1],"target":target,
            "panic_budget":np.nan if outside else weights[0],"learned_budget":np.nan if outside else weights[1],
            "panic_cycle_risk":np.nan if outside else risks[0],"learned_cycle_risk":np.nan if outside else risks[1],
            "panic_mature_cycles":counts[0],"learned_mature_cycles":counts[1],"panic_squared_loss_sum":observations[0],"learned_squared_loss_sum":observations[1],
            "panic_latest_maturity":latest[0],"learned_latest_maturity":latest[1],"risk_update_scheduled":scheduled,
            "last_successful_risk_origin":pd.NaT if outside else last,"risk_status":"NO_VIEW_OUTSIDE_DECISION_PERIOD" if outside else status})
    return pd.DataFrame(rows)
