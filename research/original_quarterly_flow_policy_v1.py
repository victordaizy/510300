"""第十轮：原始公告时钟下的季度申购赎回与价格增量研究。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from research.adaptive_allocation_v1 import ROOT, summarize, save_account
from research.event_clock_account_v1 import simulate_event_account
from research.intraday_overnight_increment_v1 import (
    now, require, write_json, normalize_dividends, holding_total_return,
    return_metrics, block_indices, interval,
)

STUDY="510300_ORIGINAL_QUARTERLY_FLOW_POLICY_V1"
OUT=ROOT/"reports/research/510300_original_quarterly_flow_policy_v1"
SOURCE=ROOT/"reports/research/510300_original_fund_subscription_reports_v1"
PARENT=ROOT/"reports/research/510300_adaptive_allocation_v1"
CONFIG=ROOT/"config/510300_original_quarterly_flow_policy_v1.json"
MANIFEST=ROOT/"config/510300_original_quarterly_flow_policy_v1_manifest.json"
PRICE=["mom60","sma120","vol60"]
FLOW=["net_units_ratio","net_units_change","gross_activity_ratio"]
FEATURES=PRICE+FLOW
MODELS={"M1_PRICE_FLOW":FEATURES,"M2_PRICE_ONLY":PRICE,"M3_FLOW_ONLY":FLOW}
RULES=["R1_POSITIVE_FLOW","R2_NEGATIVE_FLOW","R3_QUARTER_TREND","R4_FLOW_AND_TREND"]
NAMES={"M1_PRICE_FLOW":"主方案：价格加季度申赎", "M2_PRICE_ONLY":"匹配对照：仅价格",
       "M3_FLOW_ONLY":"匹配对照：仅季度申赎", "R1_POSITIVE_FLOW":"净申购顺向",
       "R2_NEGATIVE_FLOW":"净赎回反向", "R3_QUARTER_TREND":"同公告节奏的趋势",
       "R4_FLOW_AND_TREND":"需求与趋势共同确认", "BUY_HOLD":"买入持有"}


def physical(path: Path) -> Path:
    if path.exists():
        return path.resolve()
    relative=path.relative_to(ROOT)
    return Path(r"E:\ResearchData\New project 8")/relative if relative.parts[0]=="data" else path


def sha(path: Path) -> str:
    return hashlib.sha256(physical(path).read_bytes()).hexdigest()


def ident(path: Path) -> dict:
    return {"path":path.relative_to(ROOT).as_posix(),"bytes":physical(path).stat().st_size,"sha256":sha(path)}


def source_event_date(row: dict, calendar: pd.DatetimeIndex):
    if pd.isna(row.get("publication_date")):
        return pd.NaT
    dates=[pd.Timestamp(row["publication_date"])]
    if pd.notna(row.get("reported_send_date")):
        dates.append(pd.Timestamp(row["reported_send_date"]))
    later=calendar[calendar>max(dates)]
    return later[0] if len(later) else pd.NaT


def flow_values(current: dict, previous: dict | None, origin: pd.Timestamp) -> tuple[dict,str]:
    empty={key:float("nan") for key in FLOW}
    if not current.get("source_usable",False):
        return empty,"NO_VIEW_CURRENT_SOURCE_UNUSABLE"
    if previous is None or pd.Period(previous["period"],freq="Q")+1!=pd.Period(current["period"],freq="Q"):
        return empty,"NO_VIEW_PREVIOUS_QUARTER_MISSING"
    if not previous.get("source_usable",False):
        return empty,"NO_VIEW_PREVIOUS_SOURCE_UNUSABLE"
    for row in [current,previous]:
        if pd.isna(row.get("feature_available_session")) or pd.Timestamp(row["feature_available_session"])>origin:
            return empty,"NO_VIEW_SOURCE_NOT_YET_PUBLIC"
    if abs(current["beginning_units"]-previous["ending_units"])>.01:
        return empty,"NO_VIEW_QUARTER_TRANSITION_UNEXPLAINED"
    for row in [current,previous]:
        values=[row[k] for k in ["beginning_units","gross_subscription_units","gross_redemption_units","ending_units","split_delta_units"]]
        if not np.isfinite(values).all() or min(values[:4])<0 or values[0]<=0 or abs(values[0]+values[1]-values[2]+values[4]-values[3])>.01:
            return empty,"NO_VIEW_SHARE_FLOW_VALUES_UNPROVEN"
    net=(current["gross_subscription_units"]-current["gross_redemption_units"])/current["beginning_units"]
    prior=(previous["gross_subscription_units"]-previous["gross_redemption_units"])/previous["beginning_units"]
    return {"net_units_ratio":net,"net_units_change":net-prior,
            "gross_activity_ratio":(current["gross_subscription_units"]+current["gross_redemption_units"])/current["beginning_units"]},"PASS_ORIGINAL_QUARTER_FLOW_INPUTS"


def feature_row(current,previous,origin,data,is_quarter=True):
    dates=pd.DatetimeIndex(data.date)
    t=int(dates.get_indexer([origin])[0]) if pd.notna(origin) else -1
    values,state=flow_values(current,previous,origin) if pd.notna(origin) else ({k:float("nan") for k in FLOW},"NO_VIEW_PUBLICATION_CLOCK_MISSING")
    prices={key:float(data[key].iloc[t]) if t>=0 else float("nan") for key in PRICE}
    valid=state.startswith("PASS_") and np.isfinite(list(prices.values())).all() and t>=0
    if state.startswith("PASS_") and not valid:
        state="NO_VIEW_PRICE_WARMUP_OR_CALENDAR_MISSING"
    return {"period":current["period"],"origin":origin,"origin_index":t,"is_quarter_training_origin":is_quarter,
            "source_sha256":current["source_sha256"],"previous_source_sha256":previous["source_sha256"] if previous else None,
            "current_available_session":current.get("feature_available_session"),
            "previous_available_session":previous.get("feature_available_session") if previous else None,
            "feature_state":state,"all_features_valid":bool(valid),**prices,**values}


def prepare():
    require(not (OUT/"source_receipt.json").exists(),"第十轮输入事件已经保存")
    source_result=json.loads((SOURCE/"quarterly_facts_v1_1_result.json").read_text(encoding="utf-8"))
    require(not source_result["unexplained_quarter_transitions"],"来源相邻期衔接仍有未解释问题")
    data=pd.read_parquet(PARENT/"features.parquet")
    facts=pd.read_parquet(SOURCE/"quarterly_share_flow_facts_v1_1.parquet").sort_values("period")
    require(len(facts)==56 and facts.period.is_unique,"季度来源数量或唯一性异常")
    calendar=pd.read_parquet(physical(ROOT/"data/reference/a_share_hs_trading_calendar_2010_2026_v1.parquet"))
    days=pd.DatetimeIndex(calendar.loc[calendar.is_open,"trade_date"])
    prior=None
    rows=[]
    fact_rows={r["period"]:r for r in facts.to_dict("records")}
    for row in fact_rows.values():
        origin=source_event_date(row,days)
        rows.append(feature_row(row,prior,origin,data))
        prior=row
    frame=pd.DataFrame(rows)
    known=frame.loc[frame.origin_index>=0].sort_values("origin_index")
    require(known.origin_index.is_unique,"不同季度报告发生相同可用交易日，需明确新版本协议")
    old=json.loads((ROOT/"config/510300_adaptive_allocation_v1.json").read_text(encoding="utf-8"))
    anchor=int(np.flatnonzero(data.date>=pd.Timestamp(old["evaluation_start"]))[0])-1
    latest=known.loc[known.origin_index<=anchor].iloc[-1]
    current=fact_rows[latest.period]
    prior=fact_rows.get(str(pd.Period(latest.period,freq="Q")-1))
    init=feature_row(current,prior,data.date.iloc[anchor],data,is_quarter=False)
    schedule=pd.concat([known.loc[known.origin_index>=anchor],pd.DataFrame([init])],ignore_index=True).sort_values("origin_index")
    require(schedule.origin_index.is_unique,"初始化与季度事件重合，需要明确合并规则")
    OUT.mkdir(parents=True,exist_ok=True)
    frame.to_parquet(OUT/"quarter_feature_origins.parquet",index=False)
    schedule.to_parquet(OUT/"evaluation_event_features.parquet",index=False)
    frame.to_csv(OUT/"每季度唯一训练起点与中文状态.csv",index=False,encoding="utf-8-sig")
    schedule.to_csv(OUT/"完整账户公告决策时点.csv",index=False,encoding="utf-8-sig")
    receipt={"prepared_at":now(),"source_reports":len(facts),"official_clock_events":len(known),
             "valid_quarter_training_origins":int(frame.all_features_valid.sum()),
             "evaluation_decision_events_including_initialization":len(schedule),
             "valid_evaluation_event_features":int(schedule.all_features_valid.sum()),
             "initialization_origin":str(data.date.iloc[anchor].date()),"initialization_latest_report":latest.period,
             "new_labels_computed":False,"new_strategy_returns_read":False,
             "feature_missing_rows":frame.loc[~frame.all_features_valid,["period","feature_state"]].to_dict("records")}
    write_json(OUT/"source_receipt.json",receipt,exclusive=True)
    print(json.dumps(receipt,ensure_ascii=False),flush=True)


def eligible_training(origins:pd.DataFrame,origin:pd.Timestamp) -> pd.DataFrame:
    return origins.loc[origins.is_quarter_training_origin & origins.all_features_valid &
                       (origins.origin<origin) & origins.label_exit_date.notna() &
                       (origins.label_exit_date<=origin) & np.isfinite(origins.Y60)]


def fixed_fit(train:pd.DataFrame,row:pd.Series,columns:list[str],alpha:float):
    model=make_pipeline(StandardScaler(),Ridge(alpha=alpha))
    model.fit(train[columns],train.Y60)
    value=float(model.predict(pd.DataFrame([row[columns].to_dict()],columns=columns))[0])
    return model,value


def freeze():
    require(not CONFIG.exists() and not MANIFEST.exists(),"第十轮已冻结")
    old=json.loads((ROOT/"config/510300_adaptive_allocation_v1.json").read_text(encoding="utf-8"))
    config={k:v for k,v in old.items() if k not in ["models","rule_names","meta_names","shadow_start"]}
    config.update({"study_id":STUDY,"primary":"M1_PRICE_FLOW","models":MODELS,"rules":RULES,"names":NAMES,
                   "ridge_alpha":10.0,"horizon":60,"minimum_train_samples":12,
                   "training_unit":"ONE_ORIGINAL_QUARTER_REPORT_ONE_ORIGIN","only_announcement_events_and_initialization":True,
                   "flow_is_cash_amount":False,"source_uses_original_fund_reports":True})
    write_json(CONFIG,config,exclusive=True)
    paths=[CONFIG,Path(__file__),ROOT/"research/event_clock_account_v1.py",
           ROOT/"research/adaptive_allocation_v1.py",ROOT/"research/intraday_overnight_increment_v1.py",
           ROOT/"tests/test_original_quarterly_flow_policy_v1.py",ROOT/"tests/test_event_clock_account_v1.py",
           ROOT/"docs/510300_ORIGINAL_QUARTERLY_FLOW_POLICY_V1_PROTOCOL.md",ROOT/"config/510300_research_authority_v6.json",
           ROOT/"config/510300_adaptive_allocation_v1.json",PARENT/"features.parquet",PARENT/"input_receipt.json",
           ROOT/config["inputs"]["dividends"],ROOT/config["inputs"]["calendar"],
           SOURCE/"quarterly_facts_v1_1_result.json",SOURCE/"quarterly_share_flow_facts_v1_1.parquet",
           SOURCE/"quarterly_download_v1_1_result.json",OUT/"quarter_feature_origins.parquet",
           OUT/"evaluation_event_features.parquet",OUT/"source_receipt.json"]
    paths+=sorted((SOURCE/"quarterly_fact_records_v1_1").glob("*.json"))
    manifests=["config/510300_original_fund_subscription_source_v1_manifest.json",
               "config/510300_original_fund_quarterly_facts_v1_manifest.json",
               "config/510300_original_fund_quarterly_facts_v1_1_manifest.json"]
    for name in manifests:
        manifest=ROOT/name
        paths.append(manifest)
        for row in json.loads(manifest.read_text(encoding="utf-8"))["files"]:
            require(sha(ROOT/row["path"])==row["sha256"],"来源冻结文件变化："+row["path"])
            paths.append(ROOT/row["path"])
    amendment=SOURCE/"catalog_title_and_missing_source_amendment_v1_1.json"
    paths.extend([amendment,ROOT/"scripts/reconcile_original_fund_quarterly_catalog_20260906.py"])
    for row in json.loads((SOURCE/"quarterly_download_v1_1_result.json").read_text(encoding="utf-8"))["rows"]:
        require(sha(ROOT/row["raw_path"])==row["sha256"],"季度PDF与下载收据不一致")
        paths.append(ROOT/row["raw_path"])
    unique={p.relative_to(ROOT).as_posix():p for p in paths}
    write_json(MANIFEST,{"study_id":STUDY,"frozen_at":now(),"files":[ident(unique[k]) for k in sorted(unique)],
                        "new_strategy_returns_read_before_freeze":False,"history_previously_observed":True},exclusive=True)
    print(json.dumps({"状态":"第十轮来源、因子及方法已冻结","清单哈希":sha(MANIFEST)},ensure_ascii=False),flush=True)


def run(expected):
    require(sha(MANIFEST)==expected,"第十轮冻结清单哈希不一致")
    for row in json.loads(MANIFEST.read_text(encoding="utf-8"))["files"]:
        require(sha(ROOT/row["path"])==row["sha256"],"第十轮冻结文件变化："+row["path"])
    config=json.loads(CONFIG.read_text(encoding="utf-8"))
    write_json(OUT/"RUN_STARTED.json",{"started_at":now(),"manifest_sha256":expected},exclusive=True)
    data=pd.read_parquet(PARENT/"features.parquet")
    dividends=normalize_dividends(pd.read_csv(physical(ROOT/config["inputs"]["dividends"])))
    origins=pd.read_parquet(OUT/"quarter_feature_origins.parquet")
    events=pd.read_parquet(OUT/"evaluation_event_features.parquet")
    origins["Y60"]=np.nan
    origins["label_exit_date"]=pd.NaT
    for i,row in origins.iterrows():
        t=int(row.origin_index)
        if row.all_features_valid and t>=0 and t+61<len(data):
            origins.loc[i,"Y60"]=holding_total_return(data,dividends,t+1,t+61)[0]
            origins.loc[i,"label_exit_date"]=data.date.iloc[t+61]
    origins.to_parquet(OUT/"quarterly_labels.parquet",index=False)
    predictions={k:np.full(len(data),np.nan) for k in MODELS}
    targets={k:np.full(len(data),np.nan) for k in RULES}
    mask=np.zeros(len(data),dtype=bool)
    receipts=[]
    for _,row in events.iterrows():
        t=int(row.origin_index)
        mask[t]=True
        if row.all_features_valid:
            positive,negative,trend=row.net_units_ratio>0,row.net_units_ratio<0,row.sma120>0
            for key,value in zip(RULES,[positive,negative,trend,positive and trend]):
                targets[key][t]=float(value)
        train=eligible_training(origins,row.origin)
        eligible=bool(row.all_features_valid) and len(train)>=config["minimum_train_samples"]
        for key,columns in MODELS.items():
            receipt={"model":key,"origin":str(row.origin.date()),"period":row.period,
                     "origin_is_initialization":not row.is_quarter_training_origin,"feature_state":row.feature_state,
                     "training_samples":len(train),"training_periods":train.period.tolist(),
                     "latest_label_exit_date":str(train.label_exit_date.max().date()) if len(train) else None,
                     "status":"TRAINED_POINT_IN_TIME_QUARTER_MODEL" if eligible else "NO_VIEW_FEATURES_OR_MATURE_QUARTERS_INSUFFICIENT"}
            if eligible:
                model,value=fixed_fit(train,row,columns,config["ridge_alpha"])
                predictions[key][t]=value
                path=OUT/"fitted_models"/f"{key}_{row.origin:%Y%m%d}.joblib"
                path.parent.mkdir(parents=True,exist_ok=True)
                joblib.dump(model,path)
                receipt.update({"prediction":value,"columns":columns,"model_path":path.relative_to(ROOT).as_posix(),"model_sha256":sha(path)})
            else:
                receipt["prediction"]=None
            receipts.append(receipt)
        print(f"公告起点拟合：{row.origin:%Y-%m-%d}，来源 {row.period}，已成熟季度 {len(train)}，{'完成' if eligible else '无判断'}",flush=True)
    write_json(OUT/"training_receipts.json",{"rows":receipts},exclusive=True)
    pd.DataFrame({"date":data.date,"event_mask":mask,**predictions,**targets}).to_parquet(OUT/"event_signals.parquet",index=False)
    metrics,yearly,eras,uncertainty,checks=[],[],[],{},[]
    for costid,cost in config["costs"].items():
        accounts={}
        for key in list(MODELS)+RULES+["BUY_HOLD"]:
            ledger,decisions=simulate_event_account(data,dividends,config,cost,config["evaluation_start"],key,
                    prediction=predictions.get(key),targets=targets.get(key),horizon=config["horizon"],event_mask=mask)
            allowed=decisions.origin_index.isin(events.origin_index)
            require(decisions.loc[~allowed,"requested_quantity"].eq(0).all(),"公告以外发出主动换仓请求")
            no_view=decisions.signal_state.eq("NO_VIEW_KEEP_EXISTING_SHARES")
            require(decisions.loc[no_view,"requested_quantity"].eq(0).all(),"无判断被转换为交易")
            if key=="BUY_HOLD":
                old=pd.read_parquet(PARENT/"evaluation"/costid/"BUY_HOLD_ledger.parquet")
                for col in ["equity","shares","cash","net_return","commission","slippage_cost","dividend_receivable"]:
                    np.testing.assert_array_equal(ledger[col],old[col])
            checks.append({"cost":costid,"model":key,"scheduled_decisions":int(allowed.sum()),"no_view_daily_decisions":int(no_view.sum()),
                           "requests_outside_events":int(decisions.loc[~allowed,"requested_quantity"].ne(0).sum()),
                           "buy_hold_parity":key=="BUY_HOLD"})
            accounts[key]=ledger
            save_account(OUT/"evaluation"/costid,key,ledger,decisions)
        baseline=summarize(accounts["BUY_HOLD"],config)
        for key,ledger in accounts.items():
            row={"cost":costid,"model":key,**summarize(ledger,config)}
            row["annualized_return_excess_vs_buy_hold"]=row["annualized_return"]-baseline["annualized_return"]
            row["meets_point_target"]=row["net_sharpe"] is not None and row["net_sharpe"]>=config["high_sharpe_target"]
            metrics.append(row)
            for year,part in ledger.groupby(ledger.date.dt.year):
                yearly.append({"cost":costid,"model":key,"year":int(year),**summarize(part,config)})
            for name,left,right in [("2020—2021","2020-01-01","2021-12-31"),("2022—2023","2022-01-01","2023-12-31"),("2024—终点","2024-01-01",config["data_cutoff"])]:
                eras.append({"cost":costid,"model":key,"era":name,**summarize(ledger.loc[(ledger.date>=left)&(ledger.date<=right)],config)})
        rr=pd.DataFrame({"date":accounts["BUY_HOLD"].date,**{k:v.net_return.to_numpy() for k,v in accounts.items()}})
        rr.to_parquet(OUT/f"{costid}_all_evaluation_returns.parquet",index=False)
        rng=np.random.default_rng(config["random_seed"])
        samples={"primary_sharpe":[],"increment_vs_price_only":[],"increment_vs_quarter_trend":[]}
        for _ in range(config["bootstrap_repetitions"]):
            take=block_indices(rng,len(rr),config["bootstrap_day_block"])
            r=rr[config["primary"]].to_numpy()[take]
            samples["primary_sharpe"].append(return_metrics(r,config["annual_days"])["net_sharpe"])
            for name,key in [("price_only","M2_PRICE_ONLY"),("quarter_trend","R3_QUARTER_TREND")]:
                samples[f"increment_vs_{name}"].append(float((r-rr[key].to_numpy()[take]).mean()*config["annual_days"]))
        pd.DataFrame(samples).to_parquet(OUT/f"{costid}_saved_bootstrap_statistics.parquet",index=False)
        uncertainty[costid]={k+"_95_interval":interval(v) for k,v in samples.items()}
        print(f"第十轮 {costid} 的8条完整账户及共同区块区间已完成",flush=True)
    frame=pd.DataFrame(metrics)
    frame.to_csv(OUT/"metrics.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(yearly).to_csv(OUT/"yearly_metrics.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(eras).to_csv(OUT/"era_metrics.csv",index=False,encoding="utf-8-sig")
    write_json(OUT/"uncertainty.json",uncertainty,exclusive=True)
    write_json(OUT/"execution_and_baseline_checks.json",{"rows":checks},exclusive=True)
    primary=frame.loc[frame.model==config["primary"]]
    best=frame.loc[(frame.cost=="BASE")&(frame.model!="BUY_HOLD")].sort_values("net_sharpe",ascending=False).iloc[0]
    reached=bool(primary.loc[primary.cost=="BASE","meets_point_target"].iloc[0])
    result={"study_id":STUDY,"completed_at":now(),"manifest_sha256":expected,
            "status":"HISTORICAL_PRIMARY_POINT_TARGET_MET_VALIDATION_PENDING" if reached else "COMPLETED_PRIMARY_TARGET_NOT_MET",
            "primary":primary.to_dict("records"),"post_selected_best_base":best.to_dict(),
            "number_of_candidates":7,"evaluation_accounts":16,"trained_model_count":sum(r["status"].startswith("TRAINED_") for r in receipts),
            "fit_opportunities":len(receipts),"evaluation_events":len(events),"uncertainty":uncertainty,
            "independent_validation":"NOT_ESTABLISHED_ALREADY_OBSERVED_HISTORY","multiple_search_adjusted":False,
            "fund_flow_scope":"510300_FUND_UNITS_NOT_ALL_PUBLIC_FUNDS_OR_CNY_CASH","position_impact":0}
    write_json(OUT/"result.json",result,exclusive=True)
    print(json.dumps(result,ensure_ascii=False,default=str),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description="第十轮原始公告季度申购赎回研究")
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare",action="store_true")
    group.add_argument("--freeze",action="store_true")
    group.add_argument("--run",action="store_true")
    parser.add_argument("--expected-manifest-sha256")
    args=parser.parse_args()
    if args.prepare:
        prepare()
    elif args.freeze:
        freeze()
    else:
        require(bool(args.expected_manifest_sha256),"必须提供预期冻结清单哈希")
        run(args.expected_manifest_sha256)
