"""T02：冻结两次试低，比较同一批点时成员的新低覆盖，不寻找事后双底。"""
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
OUT=ROOT/"reports/research/510300_factor96_internal_reclaim_v1_0_1"
FIRST=core.OUT/"completed_run"
MEMROOT=ROOT/"data/curated/510300_csi300_pit_membership_weights_source_remediation_v1"
CLASSROOT=ROOT/"data/curated/510300_stress_transmission_hazard_v2_g1_historical_remediation_v1"
RAWROOT=ROOT/"data/raw/510300_stress_transmission_hazard_v2_four_state_ledger_execution_v1_0_1"
LEGACY=ROOT/"data/curated/510300_asymmetric_stress_hazard_v1_original_route_g2/constituent_history_20141101_20200228.parquet"
POLICIES=["PRICE_ALL","PRICE_COMMON","FULL"]
save,now,digest=core.save,core.now,core.digest


def normalize_raw(legacy,fresh):
    cols=["date","symbol","open","high","low","close"]
    a=legacy.rename(columns={"con_code":"symbol",**{"raw_"+k:k for k in ["open","high","low","close"]}})[cols].copy()
    b=fresh.rename(columns={"ts_code":"symbol","trade_date":"date"})[cols].copy()
    b["date"]=pd.to_datetime(b.date,format="%Y%m%d")
    a["date"]=pd.to_datetime(a.date)
    assert not a.duplicated(["date","symbol"]).any() and not b.duplicated(["date","symbol"]).any()
    a["priority"],b["priority"]=1,2
    return pd.concat([a,b],ignore_index=True).sort_values(["date","symbol","priority"]).drop_duplicates(["date","symbol"],keep="last").drop(columns="priority")


def member_new_lows(market,classified,raw,membership):
    """仅在完整21日可比低价窗口内判断新低；缺失处的计算坐标不用于比较。"""
    dates=pd.DatetimeIndex(market.date)
    members=membership.rename(columns={"membership_date":"date"}).copy()
    members["date"]=pd.to_datetime(members.date)
    assert not members.duplicated(["date","symbol"]).any()
    assert members.groupby("date").symbol.nunique().eq(300).all()
    symbols=pd.Index(sorted(members.symbol.unique()))
    q=classified.merge(raw,on=["date","symbol"],how="left",validate="one_to_one")
    suspended=q.constituent_return_state.eq("OFFICIAL_SUSPENSION") & q.return_is_usable
    assert q.loc[suspended,"corporate_action_status"].eq("NONE_CONFIRMED").all()
    assert q.loc[suspended,"previous_unadjusted_close"].gt(0).all()
    q.loc[suspended,"unadjusted_close"]=q.loc[suspended,"previous_unadjusted_close"]
    for name in ["open","high","low","close"]:
        q.loc[suspended,name]=q.loc[suspended,"unadjusted_close"]
    close_match=np.isclose(q.close,q.unadjusted_close,atol=1e-9,rtol=0)
    good=(q.return_is_usable & q.constituent_return_state.isin(["TRADED_VALID","OFFICIAL_SUSPENSION"])
        & close_match & q.low.gt(0) & q.low.le(q.close) & q.low.le(q.open) & q.high.ge(q.close)
        & q.high.ge(q.open) & q.high.ge(q.low) & np.isfinite(q.daily_total_shareholder_return)
        & q.daily_total_shareholder_return.gt(-1))
    def wide(series):
        return q.assign(value=series).pivot(index="date",columns="symbol",values="value").reindex(index=dates,columns=symbols)
    returns=wide(q.daily_total_shareholder_return.where(good))
    cash=q.cash_distribution_per_pre_event_share-q.subscription_cash_outflow_per_pre_event_share
    ratio=(q.low*q.post_to_pre_share_ratio+cash)/(q.close*q.post_to_pre_share_ratio+cash)
    scale=wide(ratio.where(good))
    # 未知行只用中性坐标使后续独立窗口可重新起算；有效窗口门禁止跨未知日比较。
    wealth=(1+returns.fillna(0.)).cumprod()
    lows=wealth*scale
    valid=returns.notna()&lows.notna()&lows.gt(0)
    complete=valid.rolling(21,min_periods=21).sum().eq(21)
    previous=lows.shift().rolling(20,min_periods=20).min()
    flag=lows.lt(previous*(1-1e-12))&complete
    mask=members.assign(member=True).pivot(index="date",columns="symbol",values="member").reindex(index=dates,columns=symbols).eq(True)
    counts=mask.sum(axis=1)
    covered=(complete&mask).sum(axis=1)
    coverage=pd.DataFrame({"date":dates,"members":counts.to_numpy(),"complete_low20_members":covered.to_numpy(),
        "coverage":(covered/counts.replace(0,np.nan)).to_numpy(),"new_low_members":(flag&mask).sum(axis=1).to_numpy()})
    coverage["known"]=coverage.members.eq(300)&coverage.complete_low20_members.ge(294)
    coverage["new_low_fraction"]=(coverage.new_low_members/coverage.complete_low20_members).where(coverage.known)
    diagnostic={"classified_rows":len(classified),"source_raw_close_mismatch_usable":int((q.return_is_usable&~close_match&~suspended).sum()),
        "official_suspension_rows":int(suspended.sum()),"usable_low_rows":int(good.sum()),"symbols":len(symbols)}
    return flag,complete,mask,coverage,diagnostic


def price_reclaims(market,state):
    """前20日低点在二次试低日冻结；首次低点只在两根更高低价结束后确认。"""
    f=state.copy().reset_index(drop=True)
    n=len(f)
    prior=f.low_w.shift().rolling(20,min_periods=20).min()
    raw_break=f.low_w.lt(prior*(1-1e-12))
    f["raw_break20"]=raw_break
    for col,value in {"price_signal":False,"event_id":-1,"first_test_idx":-1,"second_test_idx":-1,
        "first_confirm_idx":-1,"reclaim_idx":-1,"frozen_low":np.nan,"frozen_stop":np.nan}.items():
        f[col]=value
    events=[]
    confirmed=[]
    active=None
    used_first=set()
    for i in range(n):
        a=i-2
        if a>=20 and raw_break.iloc[a] and f.low_w.iloc[a+1]>f.low_w.iloc[a] and f.low_w.iloc[a+2]>f.low_w.iloc[a]:
            confirmed.append(a)
        if active is not None and i>active["deadline_idx"]:
            active["outcome"]="EXPIRED_WITHOUT_RECLAIM"
            active=None
        if active is None and raw_break.iloc[i]:
            eligible=[a for a in confirmed if 3<=i-a<=10 and a not in used_first
                and abs(f.low_w.iloc[a]-prior.iloc[i])<1e-12
                and abs(f.low_w.iloc[i]-f.low_w.iloc[a])<.5*f.atr20.iloc[a-1]]
            if eligible:
                a=max(eligible)
                active={"event_id":len(events),"first_test_idx":a,"first_confirm_idx":a+2,"second_test_idx":i,
                    "first_test_date":f.date.iloc[a],"first_confirm_date":f.date.iloc[a+2],"second_test_date":f.date.iloc[i],
                    "frozen_low":float(prior.iloc[i]),"frozen_stop":float(prior.iloc[i]-.5*f.atr20.iloc[i-1]),
                    "deadline_idx":i+10,"reclaim_idx":None,"reclaim_date":None,"outcome":"CENSORED_AT_END",
                    "B02_amount_ratio":float(market.amount.iloc[i]/market.amount.iloc[a-2:a+3].mean()),
                    "B02_first_log_return":float(f.r.iloc[a]),"B02_second_log_return":float(f.r.iloc[i]),
                    "B02_first_clv":float((f.wealth.iloc[a]-f.low_w.iloc[a])/(f.high_w.iloc[a]-f.low_w.iloc[a])) if f.high_w.iloc[a]>f.low_w.iloc[a] else None,
                    "B02_second_clv":float((f.wealth.iloc[i]-f.low_w.iloc[i])/(f.high_w.iloc[i]-f.low_w.iloc[i])) if f.high_w.iloc[i]>f.low_w.iloc[i] else None}
                events.append(active)
                used_first.add(a)
        if active is not None and f.wealth.iloc[i]>active["frozen_low"]:
            active["reclaim_idx"],active["reclaim_date"],active["outcome"]=i,f.date.iloc[i],"RECLAIM_CONFIRMED"
            for key in ["event_id","first_test_idx","second_test_idx","first_confirm_idx","reclaim_idx","frozen_low","frozen_stop"]:
                f.loc[i,key]=active[key]
            f.loc[i,"price_signal"]=True
            active=None
    return f,pd.DataFrame(events)


def attach_internal(f,events,flags,valid,members,daily_coverage):
    """两个测试日取同成员且两边都完整的股票；比较分母完全相同。"""
    rows=[]
    e=events.copy()
    for event in e.to_dict("records"):
        a,b=event["first_test_idx"],event["second_test_idx"]
        common=members.iloc[a]&members.iloc[b]
        usable=common&valid.iloc[a]&valid.iloc[b]
        n=int(usable.sum())
        source_gate=bool(daily_coverage.known.iloc[a] and daily_coverage.known.iloc[b])
        known=bool(n>=294 and source_gate)
        count_a=int(flags.iloc[a][usable].sum())
        count_b=int(flags.iloc[b][usable].sum())
        fraction_a=count_a/n if n else np.nan
        fraction_b=count_b/n if n else np.nan
        event.update({"common_members":int(common.sum()),"common_usable_members":n,"internal_known":known,
            "first_new_low_count":count_a,"second_new_low_count":count_b,
            "first_new_low_fraction":fraction_a,"second_new_low_fraction":fraction_b,
            "E02_change":fraction_b-fraction_a,"internal_divergence":bool(known and count_b<count_a),
            "usable_symbols":"|".join(members.columns[usable])})
        rows.append(event)
    e=pd.DataFrame(rows)
    f=f.copy()
    f["internal_known"],f["internal_divergence"]=False,False
    f["E02_change"],f["common_usable_members"]=np.nan,0
    for event in rows:
        if event["reclaim_idx"] is None or pd.isna(event["reclaim_idx"]):
            continue
        i=int(event["reclaim_idx"])
        for k in ["internal_known","internal_divergence","E02_change","common_usable_members"]:
            f.loc[i,k]=event[k]
    return f,e


def lag_features(frame,extra_delay):
    if extra_delay==0:
        return frame.copy()
    assert extra_delay==1
    f=frame.copy()
    cols=["price_signal","event_id","first_test_idx","second_test_idx","first_confirm_idx","reclaim_idx",
        "frozen_low","frozen_stop","internal_known","internal_divergence","E02_change","common_usable_members"]
    for col in cols:
        if frame[col].dtype==bool:
            f[col]=frame[col].shift(fill_value=False)
        else:
            f[col]=frame[col].shift()
    f["price_signal"]=f.price_signal&f.wealth.gt(f.frozen_low)
    return f


def load_engine():
    spec=importlib.util.spec_from_file_location("factor96_internal_engine",OUT/"code/account_engine.py")
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module


def simulate(d,x,dividends,policy,capital,cost_name,start,end,engine):
    ids=np.flatnonzero(d.date.between(start,end))
    first,last=int(ids[0]),int(ids[-1])
    account=engine.Account(float(capital))
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
        q,reason=0,"无新的合格二次试低收复"
        signal=bool(f.price_signal and (policy=="PRICE_ALL" or f.internal_known)
            and (policy!="FULL" or f.internal_divergence))
        trigger=""
        if old:
            if stopped:
                trigger="回撤停机"
            elif float(f.wealth)<stop_level:
                trigger="收盘失守冻结前低下方0.5ATR"
            elif i>=entry_idx+5:
                trigger="五日到期"
            exit_pending|=bool(trigger)
            if exit_pending:
                q,reason=-old,trigger or "先前退出继续等待可成交"
            else:
                q,reason=min(old,core.target_quantity(engine,account,ref,peak,float(f.es95)))-old,"已有份额只按风险预算削减"
        elif not stopped and signal and i<last:
            q=core.target_quantity(engine,account,ref,peak,float(f.es95))
            reason="二次试低收复后下一开盘申请"
        before=q
        if q>0:
            q=min(q,core.target_quantity(engine,account,op,peak,float(f.es95)))
        sellable=account.sellable(i)
        trade=engine.execute_order(account,q,op,float(row.previous_close),float(row.dividend),int(i),cost,cfg)
        if trade["filled_quantity"]>0 and old==0:
            entry_idx,stop_level=int(i),float(f.frozen_stop)
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
            px=engine.fill_price(close,-1,core.COSTS["STRESS"],.001)
            reserve=account.shares*(close-px)+engine.commission(account.shares,px,core.COSTS["STRESS"])
        equity=account.value(close)-reserve
        price_pnl=old*(op-previous_mark)+account.shares*(close-op)
        error=equity-previous_equity-price_pnl-recognized+trade["commission"]+trade["slippage_cost"]+reserve-previous_reserve
        assert abs(error)<1e-6
        account.assert_valid()
        peak=max(peak,equity)
        drawdown=1-equity/peak
        stopped|=drawdown>=.1
        records.append({"date":day,"idx":i,"policy":policy,"open":op,"mark":close,"cash":account.cash,"shares":account.shares,
            "dividend_receivable":account.receivable(),"terminal_exit_reserve":reserve,"equity":equity,
            "net_return":equity/previous_equity-1,"price_pnl":price_pnl,"dividend_recognized":recognized,
            "dividend_paid":paid,"exposure":account.shares*close/equity,"accounting_error":error,"drawdown":drawdown,
            "risk_stopped":stopped,"sellable_before":sellable,"terminal_unliquidated":bool(i==last and account.shares),**trade})
        decisions.append({"date":day,"origin":d.date.iloc[i-1],"origin_idx":i-1,"policy":policy,"active_signal":signal,
            "price_signal":f.price_signal,"internal_known":f.internal_known,"internal_divergence":f.internal_divergence,
            "event_id":f.event_id,"first_test_idx":f.first_test_idx,"second_test_idx":f.second_test_idx,
            "first_confirm_idx":f.first_confirm_idx,"reclaim_idx":f.reclaim_idx,"E02_change":f.E02_change,
            "common_usable_members":f.common_usable_members,"es95_5d":f.es95,"reason":reason,"pre_open_request":before,
            "requested_quantity":q,"filled_quantity":trade["filled_quantity"],"entry_idx":entry_idx,"frozen_stop":stop_level})
        if account.shares==0:
            entry_idx,exit_pending=None,False
        previous_equity,previous_mark,previous_reserve=equity,close,reserve
    return pd.DataFrame(records),pd.DataFrame(decisions)


def prepare():
    assert not (OUT/"source_receipt.json").exists(),"来源已准备，不能覆盖"
    for name in ["inputs","raw","code","source_evidence"]:
        (OUT/name).mkdir(parents=True,exist_ok=True)
    paths={"inputs/market.parquet":FIRST/"inputs/market.parquet","inputs/dividends.csv":FIRST/"inputs/dividends.csv",
        "inputs/price_features.parquet":FIRST/"daily_features_lag1.parquet",
        "inputs/risk_training_records.json":FIRST/"risk_training_records.json","inputs/mature_risk_labels.parquet":FIRST/"mature_risk_labels.parquet",
        "inputs/membership.parquet":MEMROOT/"000300_daily_pit_membership_20150101_20260814.parquet",
        "inputs/classified.parquet":CLASSROOT/"classified_constituent_returns_remediated.parquet",
        "inputs/daily_coverage.parquet":CLASSROOT/"four_state_daily_coverage.parquet",
        "raw/fresh_daily.parquet":RAWROOT/"fresh_daily_observations.parquet","raw/legacy_daily.parquet":LEGACY,
        "raw/dividend_actions.parquet":RAWROOT/"dividend_action_candidates.parquet",
        "source_evidence/membership_admission.json":MEMROOT/"source_remediation_admission_manifest.json",
        "source_evidence/transitions.parquet":MEMROOT/"official_membership_transition_ledger.parquet",
        "source_evidence/clock_correction.json":ROOT/"config/510300_csi300_pit_membership_weights_source_remediation_v1_0_1_clock_correction.json",
        "source_evidence/membership_2015_admission.json":MEMROOT/"official_2015_extension_admission_manifest.json",
        "source_evidence/return_build_receipt.json":ROOT/"reports/audit/510300_stress_transmission_hazard_v2_g1_historical_remediation_build_v1.json",
        "source_evidence/suspension_evidence.parquet":CLASSROOT/"all_missing_member_day_official_evidence.parquet",
        "source_evidence/acquisition_manifest.json":RAWROOT/"acquisition_manifest.json",
        "source_evidence/current_mandate.json":ROOT/"config/510300_existing_data_training_mandate_v1.json"}
    receipts=[]
    for name,path in paths.items():
        shutil.copy2(path,OUT/name)
        receipts.append({"original":path.relative_to(ROOT).as_posix(),"snapshot":name,"sha256":digest(path)})
    build=json.loads((OUT/"source_evidence/return_build_receipt.json").read_text(encoding="utf-8"))
    assert digest(OUT/"inputs/classified.parquet")==build["outputs"]["remediated_classified_returns"]["sha256"]
    assert digest(OUT/"inputs/daily_coverage.parquet")==build["outputs"]["four_state_daily_coverage"]["sha256"]
    admission=json.loads((OUT/"source_evidence/membership_admission.json").read_text(encoding="utf-8"))
    assert digest(OUT/"inputs/membership.parquet")==admission["conditional_2015_extension"]["combined_daily_membership_sha256"]
    save(OUT/"source_receipt.json",{"at":now(),"sources":receipts,"returns_and_membership_match_prior_receipts":True,
        "pit_weights_used":False,"current_members_backfilled":False,"new_network_data_requests":0,
        "first_public_version_authenticated":False,"new_strategy_returns_evaluated":False},True)
    print("T02来源快照已准备：原始低价、公司行为四态收益、官方点时成员分别固定。",flush=True)


def freeze():
    assert (OUT/"source_receipt.json").exists() and not (OUT/"freeze.json").exists()
    protocol={"study_id":"510300_FACTOR96_INTERNAL_RECLAIM_V1_0_1","at":now(),"primary":"FULL","new_candidates":1,
        "implementation_correction":"仅将官方全日停牌且无公司行为的未观测当日价映射为已知前收盘平值；原V1运行及失败保留，不改任何策略参数。",
        "periods":core.PERIODS,"capital":[200000,20000],"costs":core.COSTS,"annual_days":242,
        "member_low":"原始最低价加可解释现金分红/股数转换形成财富低价；前20个A股交易日最低值不含当前；要求21个连续有效低价和收益记录。",
        "missing":"只承认官方全日停牌的零收益；其他缺失/未解释公司行为不填零。计算坐标中的中性缺口不通过完整窗口门。",
        "member_identity":"两次测试日均为当时成员，且两个低价窗口完整；共同可比数至少294/300，两日各自覆盖也必须>=98%。同一批股票同一分母比较。",
        "clock":"当日成员日线保守18:00可用；两日延迟确认首个20日新低；全部条件当时已知，下一交易日开盘执行。",
        "first_test":"a为严格跌破此前20日财富低点；a+1及a+2低价均严格高于a时，a+2收盘才确认。",
        "second_test":"a+3至a+10中出现严格破此前20日低点，a仍为前20日最低，且与a低价距离严格小于0.5ATR[a-1]；同a仅认领一次二测，未结束事件不重叠。",
        "reclaim":"二測日b冻结L0=min低[b-20:b-1]；首次财富收盘>L0确认，可为b当日；最多观察至b+10，未收复事件保留。",
        "internal_divergence":"在二测b固定同成员新低比例；b严格小于a才为真，收复时不重新挑一个更好的内部日期。",
        "B02":"记录二测成交额/首次低点前后各2日已结束窗口均额、两日回报与CLV，不额外筛选。",
        "policies":POLICIES,"price_control":"PRICE_ALL只用相同价格事件；PRICE_COMMON还要求同样内部数据覆盖。主增量FULL减PRICE_COMMON，不混入缺失筛选收益。",
        "exit":"财富收盘<L0-0.5ATR[b-1]、5个开盘间隔到期或公共风险退出；冻结止损不移动。",
        "extra_delay":"额外延迟1交易日同时延迟事件和内部状态，并要求最新财富收盘仍>L0；仅FULL/PRICE_COMMON，不事后替换主时钟。",
        "account":"沿用100份、0.001价位、T+1、方向涨跌停、最低费用、分红权益、期末压力成本准备及每天完整现金账户。",
        "risk":"复用已冻结两年成熟五日ES95，每日估计；最大目标50%、2.5%ES预算、5%跳空预算、回撤空间折半、10%回撤停机不恢复。",
        "acceptance":{"net_sharpe":1.2,"cagr":.1,"max_drawdown":.1,"both_capitals_stress_required":True},
        "bootstrap":{"comparison":"FULL_MINUS_PRICE_COMMON","period":"MAIN","capital":200000,"cost":"STRESS","block":20,"draws":4000,"seed":20260929},
        "selection":"候选库累计第4项；旧价格/广度研究已失败，不能因换组合清除选择偏差。",
        "independent_validation":"NOT_ESTABLISHED_ALREADY_OBSERVED_HISTORY","stop":"固定主方案失败就保留；不改同成员门、观察窗、阈值、退出或对照晋升。",
        "orders_authorized":False,"goal_achieved":False,"goal_status":"active"}
    for name,path in {"factor96_internal_reclaim_v1_0_1.py":Path(__file__),"factor96_margin_repair_v1.py":Path(core.__file__),
        "account_engine.py":ROOT/"research/intraday_overnight_increment_v1.py",
        "test_factor96_internal_reclaim_v1_0_1.py":ROOT/"tests/test_factor96_internal_reclaim_v1_0_1.py"}.items():
        shutil.copy2(path,OUT/"code"/name)
    save(OUT/"protocol.json",protocol,True)
    files=[p for name in ["inputs","raw","code","source_evidence"] for p in (OUT/name).rglob("*") if p.is_file()]
    files.extend([OUT/"protocol.json",OUT/"source_receipt.json"])
    save(OUT/"freeze.json",{"at":now(),"before_new_features_and_strategy_returns":True,
        "files":[{"path":p.relative_to(OUT).as_posix(),"sha256":digest(p)} for p in files]},True)
    print("T02主候选与两个对照已冻结；尚未计算新低覆盖或新账户。",flush=True)


def run():
    frozen=json.loads((OUT/"freeze.json").read_text(encoding="utf-8"))
    for row in frozen["files"]:
        assert digest(OUT/row["path"])==row["sha256"],row["path"]
    assert digest(Path(__file__))==digest(OUT/"code/factor96_internal_reclaim_v1_0_1.py")
    assert digest(Path(core.__file__))==digest(OUT/"code/factor96_margin_repair_v1.py")
    save(OUT/"run_started.json",{"at":now(),"freeze_sha256":digest(OUT/"freeze.json")},True)
    market=pd.read_parquet(OUT/"inputs/market.parquet")
    market=market[market.date.le("2025-12-31")].reset_index(drop=True)
    original=pd.read_parquet(OUT/"inputs/price_features.parquet")
    keep=["date","wealth","low_w","high_w","r","rv20","atr20","es95",*core.STATE]
    f,events=price_reclaims(market,original[keep])
    raw=normalize_raw(pd.read_parquet(OUT/"raw/legacy_daily.parquet"),pd.read_parquet(OUT/"raw/fresh_daily.parquet"))
    flags,valid,members,coverage,diagnostic=member_new_lows(market,pd.read_parquet(OUT/"inputs/classified.parquet"),raw,pd.read_parquet(OUT/"inputs/membership.parquet"))
    raw.to_parquet(OUT/"normalized_raw_prices.parquet",index=False)
    for name,frame in [("member_new_low_flags",flags),("member_low_valid",valid),("membership_mask",members)]:
        frame.to_parquet(OUT/(name+".parquet"))
    coverage.to_parquet(OUT/"member_daily_coverage.parquet",index=False)
    f,events=attach_internal(f,events,flags,valid,members,coverage)
    events.to_parquet(OUT/"event_registry.parquet",index=False)
    events.drop(columns="usable_symbols").to_csv(OUT/"event_registry.csv",index=False,encoding="utf-8-sig")
    save(OUT/"source_calculation_diagnostics.json",diagnostic)
    div=core.normalize_dividends(pd.read_csv(OUT/"inputs/dividends.csv"))
    engine=load_engine()
    metrics,annual,stages,accounts=[],[],[],{}
    for lag in [0,1]:
        x=lag_features(f,lag)
        x.to_parquet(OUT/f"daily_features_lag{lag}.parquet",index=False)
        for period,(start,end) in core.PERIODS.items():
            v=x[x.date.between(start,end)]
            stages.append({"period":period,"extra_delay":lag,"days":len(v),"price_confirmations":int(v.price_signal.sum()),
                "common_coverage_confirmations":int((v.price_signal&v.internal_known).sum()),
                "full_signals":int((v.price_signal&v.internal_known&v.internal_divergence).sum())})
            for capital in [200000,20000]:
                for cost in core.COSTS:
                    for model in (POLICIES if lag==0 else ["PRICE_COMMON","FULL"]):
                        ledger,decisions=simulate(market,x,div,model,capital,cost,start,end,engine)
                        folder=OUT/f"accounts/{period}/{capital}/{cost}/LAG{lag}/{model}"
                        folder.mkdir(parents=True,exist_ok=True)
                        ledger.to_parquet(folder/"ledger.parquet",index=False)
                        decisions.to_parquet(folder/"decisions.parquet",index=False)
                        cycles=core.cycle_records(ledger,capital)
                        cycles.to_csv(folder/"cycles.csv",index=False,encoding="utf-8-sig")
                        closed=cycles[cycles.closed.astype(bool)]
                        key={"period":period,"capital":capital,"cost":cost,"lag":lag,"policy":model}
                        m=core.metrics(ledger,capital)
                        metrics.append({**key,**m,"closed_cycles":len(closed),"win_rate":closed.profit.gt(0).mean() if len(closed) else None,
                            "point_pass":m["net_sharpe"] is not None and m["net_sharpe"]>=1.2 and m["cagr"]>=.1 and m["max_drawdown"]<=.1})
                        previous=capital
                        for year,part in ledger.groupby(ledger.date.dt.year):
                            annual.append({**key,"year":int(year),**core.metrics(part,previous)})
                            previous=float(part.equity.iloc[-1])
                        accounts[period,capital,cost,lag,model]=ledger
                        if period=="MAIN" and capital==200000 and cost=="STRESS":
                            print(f"T02 {model}/额外延迟{lag}：夏普{m['net_sharpe']}，年化{m['cagr']:.3%}，完整周期{len(closed)}。",flush=True)
    frame=pd.DataFrame(metrics)
    frame.to_csv(OUT/"metrics.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(annual).to_csv(OUT/"annual_metrics.csv",index=False,encoding="utf-8-sig")
    save(OUT/"signal_stage_counts.json",stages)
    selected=frame[(frame.period=="MAIN")&(frame.cost=="STRESS")&(frame.lag==0)&(frame.policy=="FULL")]
    a,b=accounts["MAIN",200000,"STRESS",0,"FULL"],accounts["MAIN",200000,"STRESS",0,"PRICE_COMMON"]
    n=len(a)
    starts=np.random.default_rng(20260929).integers(0,n,size=(4000,int(np.ceil(n/20))))
    indices=((starts[:,:,None]+np.arange(20))%n).reshape(4000,-1)[:,:n]
    np.savez_compressed(OUT/"bootstrap_indices.npz",indices=indices)
    diff=a.net_return.to_numpy()-b.net_return.to_numpy()
    draws=diff[indices].mean(axis=1)*242
    increment={"comparison":"FULL_MINUS_PRICE_COMMON","annual_arithmetic_increment":diff.mean()*242,
        "ci95_low":np.quantile(draws,.025),"ci95_high":np.quantile(draws,.975),"global_selection_adjusted":False}
    save(OUT/"paired_increment.json",increment)
    save(OUT/"result.json",{"study_id":"510300_FACTOR96_INTERNAL_RECLAIM_V1_0_1","at":now(),"new_accounts":len(frame),
        "new_candidates":1,"primary_rows":selected.to_dict("records"),"historical_joint_point_pass":bool(selected.point_pass.all()),
        "increment":increment,"events_total":len(events),"events_reclaimed":int(events.outcome.eq("RECLAIM_CONFIRMED").sum()),
        "goal_achieved":False,"goal_status":"active","independent_forward_observations":0,"external_review":"NOT_PERFORMED","orders_authorized":False})
    print(f"T02固定检验完成{len(frame)}个账户，共同目标通过：{bool(selected.point_pass.all())}。",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=["prepare","freeze","run"])
    args=parser.parse_args()
    {"prepare":prepare,"freeze":freeze,"run":run}[args.action]()
