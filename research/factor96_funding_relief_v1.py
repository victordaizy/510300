"""T14资金压力缓和：原样固定阈值，分开进入和退出的完整账户检验。"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

from research import factor96_margin_repair_v1 as core

ROOT=core.ROOT
OUT=ROOT/"reports/research/510300_factor96_funding_relief_v1"
PRIOR=core.OUT/"completed_run"
DR=ROOT/"data/curated/510300_asymmetric_stress_hazard_v1_source_remediation_v1_0_2/dr007_daily_20150105_20260814.parquet"
POLICY_ROOT=ROOT/"data/curated/510300_asymmetric_stress_hazard_v1_source_remediation_v1_0_1"
POLICY=POLICY_ROOT/"pboc_7d_reverse_repo_rate_changes_anchor_20150105_20260814.parquet"
NODES=ROOT/"reports/research/510300_policy_information_clock_v1"
POLICIES={"PRICE_ONLY":(False,False),"ENTRY_ONLY":(True,False),"EXIT_ONLY":(False,True),"FULL":(True,True)}
PRIMARY="FULL"
save,now,digest=core.save,core.now,core.digest


def policy_ledger(raw,nodes):
    """公开操作利率水平与宣布中的目标分开；只补已核实的实施事实。"""
    q=raw.copy().sort_values("notice_date").reset_index(drop=True)
    q["effective_date"]=pd.to_datetime(q.notice_date)
    q["known_at"]=pd.to_datetime(q.published_at).dt.tz_localize("Asia/Shanghai")
    q["rate"]=q.seven_day_rate_percent.astype(float)
    q["record_kind"]="PUBLISHED_OPERATION_RATE"
    row=next(r for r in nodes if r["node_id"]=="R02")
    patch={"notice_date":pd.Timestamp(row["economic_event_date"]),
        "published_at":pd.Timestamp(row["source_available_upper"]).tz_localize(None),
        "seven_day_rate_percent":row["amount"],"previous_rate_percent":1.7,
        "source_url":row["source_url"],"raw_path":str(NODES/row["source_path"]),"raw_sha256":row["source_sha256"],
        "effective_date":pd.Timestamp(row["underlying_execution_date"]),
        "known_at":pd.Timestamp(row["source_available_upper"]).tz_convert("Asia/Shanghai"),"rate":float(row["amount"]),
        "record_kind":"OFFICIAL_EFFECTIVE_DATE_WITH_END_OF_DAY_AVAILABILITY"}
    q=pd.concat([q,pd.DataFrame([patch])],ignore_index=True).sort_values("known_at").reset_index(drop=True)
    assert q.known_at.is_monotonic_increasing
    assert (q.effective_date <= q.known_at.dt.tz_localize(None).dt.normalize()).all()
    return q


def funding_features(market,dr,policy,lag=1):
    """DR007至少下一A股交易日开盘才可用；决定后再下一开盘执行。"""
    dates=pd.DatetimeIndex(market.date)
    src=dr[["date","dr007"]].copy().sort_values("date").reset_index(drop=True)
    end=src.date.dt.tz_localize("Asia/Shanghai")+pd.Timedelta(hours=23,minutes=59,seconds=59)
    src["stat_end"]=end
    src=pd.merge_asof(src.sort_values("stat_end"),policy[["known_at","effective_date","rate"]].sort_values("known_at"),
        left_on="stat_end",right_on="known_at",direction="backward")
    assert (src.loc[src.rate.notna(),"effective_date"] <= src.loc[src.rate.notna(),"date"]).all()
    ids=np.searchsorted(dates.values,src.date.to_numpy(),side="right")
    src=src.loc[ids<len(dates)].copy()
    ids=ids[ids<len(dates)]
    src["available_at"]=dates[ids].tz_localize("Asia/Shanghai")+pd.Timedelta(hours=9,minutes=30)
    src["gap_pp"]=src.dr007-src.rate
    src=src.rename(columns={"date":"fund_stat_date","known_at":"policy_known_at"})
    columns=["fund_stat_date","available_at","policy_known_at","effective_date","dr007","rate","gap_pp"]
    decision=pd.DataFrame({"date":dates,"decision_time":dates.tz_localize("Asia/Shanghai")+pd.Timedelta(hours=16)})
    merged=pd.merge_asof(decision,src[columns].sort_values(["available_at","fund_stat_date"]),
        left_on="decision_time",right_on="available_at",direction="backward")
    merged["source_age_days"]=(merged.date-merged.fund_stat_date).dt.days
    good=merged.source_age_days.between(1,10)&merged.gap_pp.notna()
    merged.loc[~good,"gap_pp"]=np.nan
    merged["K05"]=merged.gap_pp.rolling(5,min_periods=5).mean()
    merged["K05_q90"]=merged.K05.shift().rolling(252,min_periods=120).quantile(.9)
    merged["K05_q50"]=merged.K05.shift().rolling(252,min_periods=120).quantile(.5)
    merged["K05_change5"]=merged.K05-merged.K05.shift(5)
    if lag==2:
        keep=merged.columns.difference(["date","decision_time"])
        merged.loc[:,keep]=merged.loc[:,keep].shift()
        merged["source_age_days"]=(merged.date-merged.fund_stat_date).dt.days
    elif lag!=1:
        raise ValueError("仅登记原时钟和额外一个交易日延迟")
    merged["fund_known"]=np.isfinite(merged[["K05","K05_q90","K05_q50"]]).all(axis=1)&merged.source_age_days.between(1,10)
    merged["fund_high"]=merged.fund_known & merged.K05.gt(merged.K05_q90)
    high_previous=merged.fund_high.shift(fill_value=False).rolling(5,min_periods=5).max().eq(1)
    merged["fund_relief"]=merged.fund_known & high_previous & merged.K05.lt(merged.K05_q50)
    assert (merged.loc[merged.fund_known,"available_at"] <= merged.loc[merged.fund_known,"decision_time"]).all()
    assert (merged.loc[merged.fund_known,"fund_stat_date"] < merged.loc[merged.fund_known,"date"]).all()
    return merged,src


def price_events(price_state):
    """冲击日之后首个收涨；观察窗10日，冲击事件起点也以10日去重。"""
    x=price_state.copy().reset_index(drop=True)
    x["shock"]=x.r.lt(-2*x.rv20.shift())
    x["price_confirmation"]=False
    x["shock_origin"]=-1
    x["shock_low"]=np.nan
    x["post_shock_rv3_ratio"]=np.nan
    short=x.r.rolling(3).std(ddof=1)
    active,last=None,-1000
    for i in range(len(x)):
        if active is not None:
            if i-active>10:
                active=None
            elif i>active and x.r.iloc[i]>0:
                e=active
                x.loc[i,["price_confirmation","shock_origin","shock_low","post_shock_rv3_ratio"]]=[True,e,
                    float(x.low_w.iloc[e]),float(short.iloc[i]/x.rv20.iloc[e-1])]
                active=None
        if bool(x.shock.iloc[i]) and i-last>=10:
            active,last=i,i
    return x


def load_engine():
    spec=importlib.util.spec_from_file_location("factor96_funding_engine",OUT/"code/account_engine.py")
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module


def simulate(d,x,dividends,policy,capital,cost_name,start,end,e):
    entry_filter,funding_exit=POLICIES[policy]
    ids=np.flatnonzero(d.date.between(start,end))
    first,last=int(ids[0]),int(ids[-1])
    account=e.Account(float(capital))
    cost,cfg=core.COSTS[cost_name],{"lot":100,"tick":.001,"limit_fraction":.1}
    previous_equity,previous_mark,previous_reserve,peak=capital,float(d.close.iloc[first-1]),0.,capital
    entry_idx,stop_level=None,np.nan
    stopped,exit_pending=False,False
    events=dividends.to_dict("records")
    records,decisions=[],[]
    for i in ids:
        row,f=d.iloc[i],x.iloc[i-1]
        day,op,close,old=row.date,float(row.open),float(row.close),account.shares
        recognized,paid=0.,0.
        for k,event in enumerate(events):
            if event["ex_date"]==day:
                value=account.entitlements.get(k,0)*event["cash_dividend_per_share"]
                account.receivables[k]=value
                recognized+=value
            if event["payment_date"]<day and k in account.receivables:
                value=account.receivables.pop(k)
                account.cash+=value
                paid+=value
        ref=float(d.close.iloc[i-1]-row.dividend)
        q,reason=0,"无新的合格价格确认"
        signal=bool(f.price_confirmation and (not entry_filter or f.fund_relief))
        trigger=""
        if old:
            if stopped:
                trigger="回撤停机"
            elif float(f.wealth)<stop_level:
                trigger="收盘失守冲击日低点"
            elif i>=entry_idx+5:
                trigger="五日到期"
            elif funding_exit and bool(f.fund_high):
                trigger="资金压力重返90分位"
            elif funding_exit and not bool(f.fund_known):
                trigger="资金数据失效退出"
            exit_pending|=bool(trigger)
            if exit_pending:
                q,reason=-old,trigger or "先前退出请求继续等待可成交"
            else:
                target=min(old,core.target_quantity(e,account,ref,peak,float(f.es95)))
                q,reason=target-old,"已有份额仅按风险预算削减"
        elif not stopped and signal and i<last:
            q=core.target_quantity(e,account,ref,peak,float(f.es95))
            reason="确认后下一开盘申请"
        before=q
        if q>0:
            q=min(q,core.target_quantity(e,account,op,peak,float(f.es95)))
        sellable=account.sellable(i)
        trade=e.execute_order(account,q,op,float(row.previous_close),float(row.dividend),int(i),cost,cfg)
        if trade["filled_quantity"]>0 and old==0:
            entry_idx,stop_level=int(i),float(f.shock_low)
            exit_pending=False
        for k,event in enumerate(events):
            if event["payment_date"]==day and k in account.receivables:
                value=account.receivables.pop(k)
                account.cash+=value
                paid+=value
            if event["record_date"]==day:
                account.entitlements[k]=account.shares
        reserve=0.
        if i==last and account.shares:
            px=e.fill_price(close,-1,core.COSTS["STRESS"],.001)
            reserve=account.shares*(close-px)+e.commission(account.shares,px,core.COSTS["STRESS"])
        equity=account.value(close)-reserve
        price_pnl=old*(op-previous_mark)+account.shares*(close-op)
        error=equity-previous_equity-price_pnl-recognized+trade["commission"]+trade["slippage_cost"]+reserve-previous_reserve
        assert abs(error)<1e-6
        account.assert_valid()
        peak=max(peak,equity)
        drawdown=1-equity/peak
        stopped|=drawdown>=.1
        records.append({"date":day,"idx":i,"policy":policy,"open":op,"mark":close,"cash":account.cash,
            "shares":account.shares,"dividend_receivable":account.receivable(),"terminal_exit_reserve":reserve,
            "equity":equity,"net_return":equity/previous_equity-1,"price_pnl":price_pnl,
            "dividend_recognized":recognized,"dividend_paid":paid,"exposure":account.shares*close/equity,
            "accounting_error":error,"drawdown":drawdown,"risk_stopped":stopped,"sellable_before":sellable,
            "terminal_unliquidated":bool(i==last and account.shares),**trade})
        decisions.append({"date":day,"origin":d.date.iloc[i-1],"origin_idx":i-1,"policy":policy,
            "active_signal":signal,"price_confirmation":f.price_confirmation,"fund_relief":f.fund_relief,
            "fund_known":f.fund_known,"fund_high":f.fund_high,"fund_stat_date":f.fund_stat_date,
            "fund_available_at":f.available_at,"policy_known_at":f.policy_known_at,"K05":f.K05,
            "K05_q90":f.K05_q90,"K05_q50":f.K05_q50,"shock_origin":f.shock_origin,
            "es95_5d":f.es95,"reason":reason,"pre_open_request":before,"requested_quantity":q,
            "filled_quantity":trade["filled_quantity"],"entry_idx":entry_idx,"frozen_stop":stop_level})
        if account.shares==0:
            entry_idx,exit_pending=None,False
        previous_equity,previous_mark,previous_reserve=equity,close,reserve
    return pd.DataFrame(records),pd.DataFrame(decisions)


def prepare():
    if (OUT/"source_receipt.json").exists():
        raise RuntimeError("来源准备已完成，不原地覆盖。")
    for name in ["inputs","raw","code"]:
        (OUT/name).mkdir(parents=True,exist_ok=True)
    paths={"market.parquet":PRIOR/"inputs/market.parquet","dividends.csv":PRIOR/"inputs/dividends.csv",
        "dr007.parquet":DR,"operation_changes.parquet":POLICY,
        "published_operation_rates.parquet":POLICY_ROOT/"pboc_7d_reverse_repo_published_rates_anchor_20150105_20260814.parquet",
        "pboc_acquisition_manifest.json":POLICY_ROOT/"pboc_source_acquisition_manifest.json",
        "dr007_manifest.json":DR.parent/"dr007_tushare_source_acquisition_manifest.json",
        "policy_nodes.json":NODES/"inputs/policy_nodes.json",
        "mature_risk_labels.parquet":PRIOR/"mature_risk_labels.parquet",
        "risk_training_records.json":PRIOR/"risk_training_records.json",
        "prior_protocol.json":PRIOR/"protocol.json",
        "current_mandate.json":ROOT/"config/510300_existing_data_training_mandate_v1.json"}
    receipts=[]
    for name,path in paths.items():
        shutil.copy2(path,OUT/"inputs"/name)
        receipts.append({"original":path.relative_to(ROOT).as_posix(),"snapshot":"inputs/"+name,"sha256":digest(path)})
    all_features=pd.read_parquet(PRIOR/"daily_features_lag1.parquet")
    cols=["date","wealth","high_w","low_w","r","rv20","atr20","ma20",*core.STATE,"es95"]
    all_features[cols].to_parquet(OUT/"inputs/price_state.parquet",index=False)
    raw_rates=pd.read_parquet(OUT/"inputs/operation_changes.parquet")
    nodes=json.loads((OUT/"inputs/policy_nodes.json").read_text(encoding="utf-8"))
    ledger=policy_ledger(raw_rates,nodes)
    for record in ledger.itertuples():
        path=Path(record.raw_path)
        if not path.is_absolute():
            path=ROOT/path
        assert digest(path)==record.raw_sha256
        shutil.copy2(path,OUT/"raw"/path.name)
    ledger.to_parquet(OUT/"inputs/policy_rate_ledger.parquet",index=False)
    raw_dr=ROOT/"data/raw/510300_asymmetric_stress_hazard_v1_dr007_tushare_v1_0_2/repo_daily"
    pieces=[]
    for path in sorted(raw_dr.glob("*.json")):
        j=json.loads(path.read_text(encoding="utf-8"))
        f=pd.DataFrame(j["data"]["items"],columns=j["data"]["fields"])
        assert f.ts_code.eq("DR007.IB").all() and f.repo_maturity.eq("DR007").all()
        pieces.append(f)
        shutil.copy2(path,OUT/"raw"/path.name)
    combined=pd.concat(pieces).sort_values("trade_date")
    combined["date"]=pd.to_datetime(combined.trade_date,format="%Y%m%d")
    decoded=pd.read_parquet(OUT/"inputs/dr007.parquet").sort_values("date")
    assert combined.date.tolist()==decoded.date.tolist()
    np.testing.assert_allclose(combined.weight.astype(float),decoded.dr007,atol=1e-12,rtol=0)
    save(OUT/"source_receipt.json",{"prepared_at":now(),"sources":receipts,"dr007_rows":len(decoded),
        "dr007_raw_recomputed":True,"policy_original_changes":len(raw_rates),"policy_ledger_rows":len(ledger),
        "known_patch":"2024-09-27实施1.5%以原文日期上界23:59:59可得；2024-09-24宣布目标不替代实际生效。",
        "first_version_authenticated":False,"admission":"HISTORICAL_DISCOVERY_WITH_CONSERVATIVE_CLOCK",
        "new_price_return_or_strategy_evaluations":0},True)
    print("T14来源已准备：DR007原始响应逐值复核，25条操作变化和1条已知实施事实分列。",flush=True)


def freeze():
    if (OUT/"freeze.json").exists():
        raise RuntimeError("T14已冻结，不覆盖。")
    assert (OUT/"source_receipt.json").exists()
    protocol={"study_id":"510300_FACTOR96_FUNDING_RELIEF_V1","registered_at":now(),"primary":"FULL",
        "new_candidate_definitions":1,"seed_source":"用户18策略中的T14/K05/B04",
        "periods":core.PERIODS,"capital_cny":[200000,20000],"annual_days":242,"cash_rate":0.,"rf":0.,
        "costs":core.COSTS,"primary_source_lag":1,"sensitivity_extra_session_lag":2,
        "rate_clock":"DR007统计日结束后至少下一A股交易日09:30可用；收盘16:00取最近可得的一条，保留银行周末工作日；不使用本日DR007。",
        "policy_clock":"DR007日结束时已公开的最新7天操作利率或实施事实；宣布中的未来目标单列不用作生效利率；缺失不是0。",
        "K05":"最近5个A股决策日已可见的DR007减相应已公开操作利率的均值；单位百分点。源年龄超过10自然日缺失。",
        "quantiles":"前252个A股决策日K05、至少120有效值；当前值排除；额外延迟时整体位移1交易日。",
        "fund_relief":"前5个决策日曾严格高于各自当时90分位，当前严格低于当时50分位。",
        "price_event":"对数总回报<-2倍此前20日波动，冲击起点10日去重；之后第1至10日首次收涨，不另加不创新低要求。",
        "entry":"价格确认与资金缓和在同一收盘成立，下一开盘申请。",
        "exit":"收盘失守冻结冲击日低点、资金压力重返90分位、五个开盘间隔到期或公共风险退出；次开盘卖。",
        "factorial":POLICIES,"factorial_meaning":"固定进入开关×退出开关；FULL为唯一主方案，其他三项仅分解作用，失败后不晋升最好对照。",
        "funding_missing":"FULL/EXIT_ONLY既有持仓遇资金数据失效退出；缺失不伪装缓和。",
        "risk":"完全复用前轮已冻结的两年成熟状态五日ES95、最大目标50%、2.5%ES预算、5%跳空损失预算和10%回撤停止；无新风险拟合或参数搜索。",
        "execution":"原始开盘价、100份、0.001价位、T+1、最低费、方向涨跌停保守不成交；末日持仓扣压力退出成本准备但不假造成交。",
        "comparisons":"FULL减PRICE_ONLY是主要增量；进入和退出拆分仅描述。主期20万元压力日收益差，20日区块4000次；随机种子20260928。",
        "outer_selection":"本轮1项，加前轮2项，候选库累计3项；家族统计不能清除96候选及旧研究选择偏差。",
        "acceptance":{"sharpe":1.2,"cagr":.1,"max_drawdown":.1,"both_capitals_stress_required":True,
            "independent_validation":"NOT_ESTABLISHED_ALREADY_OBSERVED_HISTORY"},
        "stop":"固定主方案未达标则保留；不调分位、观察窗、退出或移用较好的对照。",
        "orders_authorized":False,"goal_achieved":False,"goal_status":"active"}
    for name,path in {"account_engine.py":ROOT/"research/intraday_overnight_increment_v1.py",
        "factor96_funding_relief_v1.py":Path(__file__),"factor96_margin_repair_v1.py":Path(core.__file__),
        "test_factor96_funding_relief_v1.py":ROOT/"tests/test_factor96_funding_relief_v1.py"}.items():
        shutil.copy2(path,OUT/"code"/name)
    save(OUT/"protocol.json",protocol,True)
    files=list((OUT/"inputs").iterdir())+list((OUT/"raw").iterdir())+list((OUT/"code").iterdir())+[OUT/"protocol.json",OUT/"source_receipt.json"]
    save(OUT/"freeze.json",{"at":now(),"before_new_strategy_returns":True,
        "files":[{"path":p.relative_to(OUT).as_posix(),"sha256":digest(p)} for p in files]},True)
    print("T14固定合同已冻结：1个主候选、3个作用分解对照、两种披露时钟。",flush=True)


def run():
    frozen=json.loads((OUT/"freeze.json").read_text(encoding="utf-8"))
    for r in frozen["files"]:
        assert digest(OUT/r["path"])==r["sha256"],r["path"]
    assert digest(Path(__file__))==digest(OUT/"code/factor96_funding_relief_v1.py")
    assert digest(Path(core.__file__))==digest(OUT/"code/factor96_margin_repair_v1.py")
    save(OUT/"run_started.json",{"at":now(),"freeze_sha256":digest(OUT/"freeze.json")},True)
    d=pd.read_parquet(OUT/"inputs/market.parquet")
    d=d[d.date.le("2025-12-31")].reset_index(drop=True)
    x=price_events(pd.read_parquet(OUT/"inputs/price_state.parquet"))
    assert x.date.tolist()==d.date.tolist()
    dr=pd.read_parquet(OUT/"inputs/dr007.parquet")
    policy=pd.read_parquet(OUT/"inputs/policy_rate_ledger.parquet")
    div=core.normalize_dividends(pd.read_csv(OUT/"inputs/dividends.csv"))
    assert pd.DatetimeIndex(d.loc[d.date.ge("2015-01-05"),"date"]).difference(dr.date).empty
    e=load_engine()
    metrics,annual,stages,accounts=[],[],[],{}
    for lag in [1,2]:
        funding,source=funding_features(d,dr,policy,lag)
        f=pd.concat([x,funding.drop(columns=["date"])],axis=1)
        f.to_parquet(OUT/f"daily_features_lag{lag}.parquet",index=False)
        if lag==1:
            source.to_parquet(OUT/"aligned_bank_source.parquet",index=False)
        policies=list(POLICIES) if lag==1 else ["ENTRY_ONLY","EXIT_ONLY","FULL"]
        for period,(start,end) in core.PERIODS.items():
            slice_=f[f.date.between(start,end)]
            stages.append({"period":period,"lag":lag,"days":len(slice_),"fund_known_days":int(slice_.fund_known.sum()),
                "price_confirmations":int(slice_.price_confirmation.sum()),"fund_relief_days":int(slice_.fund_relief.sum()),
                "full_entry_signals":int((slice_.price_confirmation & slice_.fund_relief).sum())})
            for capital in [200000,20000]:
                for cost in core.COSTS:
                    for model in policies:
                        ledger,decisions=simulate(d,f,div,model,capital,cost,start,end,e)
                        folder=OUT/f"accounts/{period}/{capital}/{cost}/LAG{lag}/{model}"
                        folder.mkdir(parents=True,exist_ok=True)
                        ledger.to_parquet(folder/"ledger.parquet",index=False)
                        decisions.to_parquet(folder/"decisions.parquet",index=False)
                        cc=core.cycle_records(ledger,capital)
                        cc.to_csv(folder/"cycles.csv",index=False,encoding="utf-8-sig")
                        closed=cc[cc.closed.astype(bool)]
                        key={"period":period,"capital":capital,"cost":cost,"lag":lag,"policy":model}
                        m=core.metrics(ledger,capital)
                        metrics.append({**key,**m,"closed_cycles":len(closed),"win_rate":closed.profit.gt(0).mean() if len(closed) else None,
                            "point_pass":m["net_sharpe"] is not None and m["net_sharpe"]>=1.2 and m["cagr"]>=.1 and m["max_drawdown"]<=.1})
                        prev=capital
                        for year,group in ledger.groupby(ledger.date.dt.year):
                            annual.append({**key,"year":int(year),**core.metrics(group,prev)})
                            prev=float(group.equity.iloc[-1])
                        accounts[period,capital,cost,lag,model]=ledger
                        if period=="MAIN" and capital==200000 and cost=="STRESS":
                            print(f"T14 {model}/延迟{lag}：夏普{m['net_sharpe']}，年化{m['cagr']:.3%}，完整周期{len(closed)}。",flush=True)
    pd.DataFrame(metrics).to_csv(OUT/"metrics.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(annual).to_csv(OUT/"annual_metrics.csv",index=False,encoding="utf-8-sig")
    save(OUT/"signal_stage_counts.json",stages)
    primary=accounts["MAIN",200000,"STRESS",1,"FULL"]
    price=accounts["MAIN",200000,"STRESS",1,"PRICE_ONLY"]
    n,draws,block=len(primary),4000,20
    rng=np.random.default_rng(20260928)
    starts=rng.integers(0,n,size=(draws,int(np.ceil(n/block))))
    indices=((starts[:,:,None]+np.arange(block))%n).reshape(draws,-1)[:,:n]
    np.savez_compressed(OUT/"bootstrap_indices.npz",indices=indices)
    diff=primary.net_return.to_numpy()-price.net_return.to_numpy()
    boot=diff[indices].mean(axis=1)*242
    mean=diff.mean()*242
    increment={"comparison":"FULL_MINUS_PRICE_ONLY","annual_arithmetic_increment":mean,
        "ci95_low":np.quantile(boot,.025),"ci95_high":np.quantile(boot,.975),
        "one_sided_p":(1+int(np.sum(boot-mean>=mean)))/(draws+1),"global_selection_adjusted":False}
    save(OUT/"paired_increment.json",increment)
    frame=pd.DataFrame(metrics)
    selected=frame[(frame.period=="MAIN")&(frame.cost=="STRESS")&(frame.lag==1)&(frame.policy=="FULL")]
    save(OUT/"result.json",{"study_id":"510300_FACTOR96_FUNDING_RELIEF_V1","completed_at":now(),
        "new_candidates":1,"new_accounts":len(frame),"primary_rows":selected.to_dict("records"),
        "historical_joint_point_pass":bool(selected.point_pass.all()),"increment":increment,
        "status":"COMPLETE_FIXED_T14_RESEARCH","goal_achieved":False,"goal_status":"active",
        "independent_forward_observations":0,"external_review":"NOT_PERFORMED","orders_authorized":False})
    print(f"T14完成{len(frame)}个新账户，主方案共同目标通过：{bool(selected.point_pass.all())}。",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=["prepare","freeze","run"])
    args=parser.parse_args()
    {"prepare":prepare,"freeze":freeze,"run":run}[args.action]()
