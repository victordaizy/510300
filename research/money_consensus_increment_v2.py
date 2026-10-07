"""冻结补齐的共识资料，按既定五日时钟检验消息相对价格的增量。"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from collect_bualuang_money_consensus_v1 import now

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STUDY = ROOT / "reports/research/510300_money_consensus_increment_v2"
SOURCES = ROOT / "reports/research/510300_money_consensus_source_extension_v1"
PARENT = ROOT / "reports/research/510300_post_information_capital_adjustment_v1"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.15g")


def freeze() -> None:
    study = DEFAULT_STUDY
    if study.exists():
        raise RuntimeError("新研究目录已存在，禁止覆盖冻结资料。")
    chosen, duplicates, source_results = {}, [], []
    roots = [("Bualuang", SOURCES), ("OCBC", SOURCES/"ocbc"), ("NBG_FIVE_DAY", SOURCES/"nbg"), ("NBG_WEEKLY", SOURCES/"nbg_weekly")]
    for family, directory in roots:
        source_results.append({"source_family":family,"result":json.loads((directory/"result.json").read_text(encoding="utf-8"))})
        for record in json.loads((directory/"results/admitted_pairs.json").read_text(encoding="utf-8")):
            assert sha(directory/"raw"/record["filename"]) == record["sha256"]
            item = dict(record, source_family=family)
            if item["stat_month"] in chosen:
                duplicates.append({"stat_month":item["stat_month"],"selected_source":chosen[item["stat_month"]]["source_family"],
                    "alternative_source":family,"selected_M1":chosen[item["stat_month"]]["M1"]["forecast_pp"],
                    "alternative_M1":item["M1"]["forecast_pp"],"selected_M2":chosen[item["stat_month"]]["M2"]["forecast_pp"],
                    "alternative_M2":item["M2"]["forecast_pp"],"alternative_url":item["url"]})
            else:
                chosen[item["stat_month"]] = item
    for folder in ("inputs", "sources", "results", "figures", "code", "evidence"):
        (study/folder).mkdir(parents=True, exist_ok=True)
    official = pd.read_csv(SOURCES/"inputs/official_releases.csv")
    rows = []
    for raw in official.to_dict("records"):
        item = chosen.get(raw["stat_month"])
        row = {k:raw[k] for k in ["stat_month","published_at","available_at_upper_bound","publication_time_precision","training_regime","definition_version","m1_yoy_pp","m2_yoy_pp","source_url"]}
        row["consensus_admitted"] = item is not None
        if item:
            row.update(forecast_date=item["report_date"], forecast_source=item["source_family"], forecast_url=item["url"],
                forecast_sha256=item["sha256"], forecast_file=item["filename"], forecast_age_days=item["offset_days"],
                expected_m1_pp=item["M1"]["forecast_pp"], expected_m2_pp=item["M2"]["forecast_pp"],
                expected_spread_proxy_pp=item["expected_spread_proxy_pp"],
                spread_surprise_proxy_pp=raw["m1_yoy_pp"]-raw["m2_yoy_pp"]-item["expected_spread_proxy_pp"],
                m1_surprise_pp=raw["m1_yoy_pp"]-item["M1"]["forecast_pp"],
                m2_surprise_pp=raw["m2_yoy_pp"]-item["M2"]["forecast_pp"])
        rows.append(row)
    csv(pd.DataFrame(rows), study/"inputs/104个月共识选择.csv")
    csv(pd.DataFrame(duplicates),study/"sources/重复来源对照.csv")
    save(study/"sources/selected_numeric_sources.json",list(chosen.values()))
    save(study/"sources/source_collection_results.json",source_results)
    for filename in ("market_daily.parquet","market_daily.csv","official_releases.csv","policy_context_events.json"):
        shutil.copy2(PARENT/"inputs"/filename,study/"inputs"/filename)
    shutil.copy2(PARENT/"protocol.json",study/"evidence/parent_protocol.json")
    shutil.copy2(SOURCES/"fund_clock/result.json",study/"evidence/fund_clock_scope.json")
    shutil.copy2(PARENT/"evidence/fund_share_clock_closure.json",study/"evidence/fund_share_clock_closure.json")
    protocol = {
        "study_id":"510300_MONEY_CONSENSUS_INCREMENT_V2","frozen_at":now(),
        "scope":"补齐来源后，同口径事件的A价格基准与B增加剪刀差预期差的历史增量；C/D缺资金时点保持NOT_RUN。不是完成四组共同样本检验。",
        "source_priority":"Bualuang、OCBC、NBG原五日检索、NBG自然周报依次填缺；选择在新收益计算前完成，记录全部来源扩展历史。",
        "source_counts_before_returns":{"total":len(chosen),"old":sum(m<'2025-01' for m in chosen),"new":sum(m>='2025-01' for m in chosen)},
        "parent_protocol":"evidence/parent_protocol.json，模型、五交易日标签、36训练加24评价、成本及门槛不变",
        "clock":"公布后第一个完整交易日收盘观察，下一交易日开盘进入，第五交易日预定收盘结束；日期精度不足按公布日23:59:59。",
        "features_A":["price_pre20_return","initial_response","price_log_sigma20"],
        "features_B":["price_pre20_return","initial_response","price_log_sigma20","spread_surprise_proxy_pp"],
        "fund_status":"目前没有足量事件对应的五日份额变化及历史公开上界证据；公众号/原报告新线索没有自动转成完整变量。",
        "maturity":"训练退出时刻严格早于当前观察时刻，重叠入场至退出区间按事件群计算；行情截止以后标签保留PENDING_MARKET_DATA。",
        "regime":"M1旧口径和2025新口径分开；原M2数字人民币口径标记保留，不增加结果驱动的分组。",
        "minimum":{"train_clusters":36,"evaluation_clusters":24},
        "ridge":"训练内均值/总体标准差标准化；平均MSE加1倍斜率平方和，截距不罚；扩展训练，无参数搜索。",
        "bootstrap":{"cluster_block_length":4,"draws":10000,"seed":5103005,"type":"不循环的连续移动块，有放回抽块后截断到原样本数","one_sided_lower_quantile":0.1},
        "MSE_gate":"B相对A的平方误差改善均值>0、90%单侧下界>0、评价前后两半均>0，三条件都报告。",
        "entry_gate_after_MSE_only":"如MSE门通过，B预测须高于以观察日收盘估计的20万满预算压力往返成本；单笔收益用实际次日开盘及固定第五日收盘扣费用估算，至少12群、均值下界>0、单一正贡献<=30%。这不是主D账户。",
        "capital":{"main":200000,"comparison":20000},"commission":0.0002,"minimum_commission":5,"lot":100,"stress_slippage":0.001,
        "dividend":"标签包含入场之后除息、退出之前或当日除息所获得的现金权利，不再投资；入场日除息不享有。无账户现金到账或登记日模拟，未把应收现金当作已可花现金。",
        "independent_forward_events":0,"all_returns_historical_development":True,
        "exits_and_accounts":"只有D主门通过才可主账户及固定/回撤退出比较；当前C/D未准入，相关阶段保持NOT_RUN。",
        "no_rescue":"看到本轮结果后不改窗口、方向、来源优先级、ridge、成本或样本门。失败不改名为独立未见样本。"}
    save(study/"protocol.json",protocol)
    files = [*sorted((study/"inputs").glob("*")),study/"protocol.json",study/"sources/selected_numeric_sources.json",study/"evidence/parent_protocol.json"]
    save(study/"freeze_receipt.json",{"frozen_at":now(),"new_market_outcomes_read":False,"files":[{"path":str(p.relative_to(study)).replace('\\','/'),"sha256":sha(p)} for p in files]})
    shutil.copy2(Path(__file__),study/"code"/Path(__file__).name)
    print(json.dumps(protocol["source_counts_before_returns"],ensure_ascii=False),flush=True)


def ridge_predict(train: pd.DataFrame, test: pd.Series, columns: list[str]) -> tuple[float, dict]:
    x = train[columns].to_numpy(float)
    y = train.residual_5d_gross_return.to_numpy(float)
    mean, scale = x.mean(axis=0), x.std(axis=0,ddof=0)
    scale = np.where(scale>0,scale,1.)
    z=(x-mean)/scale
    beta=np.linalg.solve(z.T@z/len(x)+np.eye(len(columns)),z.T@(y-y.mean())/len(x))
    prediction=float(y.mean()+((test[columns].to_numpy(float)-mean)/scale)@beta)
    return prediction,{"columns":columns,"mean":mean.tolist(),"scale":scale.tolist(),"beta":beta.tolist(),"intercept":float(y.mean())}


def lower_bound(values: np.ndarray, cfg: dict) -> float:
    n=len(values)
    block=min(cfg["cluster_block_length"],n)
    rng=np.random.default_rng(cfg["seed"])
    starts=rng.integers(0,n-block+1,size=(cfg["draws"],math.ceil(n/block)))
    indices=(starts[:,:,None]+np.arange(block)).reshape(cfg["draws"],-1)[:,:n]
    return float(np.quantile(values[indices].mean(axis=1),cfg["one_sided_lower_quantile"]))


def round_trip(entry: float, exit_price: float, dividend: float, capital: float, protocol: dict) -> dict:
    slip=protocol["stress_slippage"]
    buy,sell=entry*(1+slip),exit_price*(1-slip)
    quantity=int(capital//(buy*protocol["lot"]))*protocol["lot"]
    fee=lambda value:max(protocol["minimum_commission"],protocol["commission"]*value)
    while quantity>0 and quantity*buy+fee(quantity*buy)>capital:
        quantity-=protocol["lot"]
    if quantity<=0:
        raise RuntimeError("单笔成本预算不足一手。")
    net=quantity*(sell-buy+dividend)-fee(quantity*buy)-fee(quantity*sell)
    return {"quantity":quantity,"net_return_on_capital":float(net/capital)}


def analyse(study: Path, output: Path) -> None:
    protocol=json.loads((study/"protocol.json").read_text(encoding="utf-8"))
    frozen=json.loads((study/"freeze_receipt.json").read_text(encoding="utf-8"))
    for record in frozen["files"]:
        assert sha(study/record["path"])==record["sha256"],record["path"]
    output.mkdir(parents=True,exist_ok=True)
    consensus=pd.read_csv(study/"inputs/104个月共识选择.csv")
    market=pd.read_parquet(study/"inputs/market_daily.parquet").sort_values("date").reset_index(drop=True)
    market["date"]=pd.to_datetime(market.date).dt.normalize()
    dates=pd.DatetimeIndex(market.date)
    assert dates.is_unique and dates.is_monotonic_increasing
    opens=dates.tz_localize("Asia/Shanghai")+pd.Timedelta(hours=9,minutes=30)
    closes=dates.tz_localize("Asia/Shanghai")+pd.Timedelta(hours=15)
    cases,pending=[],[]
    for row in consensus[consensus.consensus_admitted].to_dict("records"):
        published=pd.Timestamp(row["available_at_upper_bound"])
        obs=int(opens.searchsorted(published,side="right"))
        entry,end=obs+1,obs+5
        if end>=len(market):
            pending.append(dict(row,label_status="PENDING_MARKET_DATA"))
            continue
        pre=int(closes.searchsorted(published,side="left"))-1
        assert pre>=20 and pre<obs
        p,o,e,z=[market.iloc[k] for k in (pre,obs,entry,end)]
        dividend=float(market.iloc[entry+1:end+1].dividend.sum())
        sigma=float(market.iloc[obs-19:obs+1].total_simple.std(ddof=1))
        initial=float((o.close+market.iloc[pre+1:obs+1].dividend.sum())/p.close-1)
        cases.append(dict(row,event_id="MONEY_"+row["stat_month"],pre_date=str(p.date.date()),
            observation_date=str(o.date.date()),observation_at=closes[obs].isoformat(),
            entry_date=str(e.date.date()),entry_at=opens[entry].isoformat(),exit_date=str(z.date.date()),exit_at=closes[end].isoformat(),
            pre_close=float(p.close),observation_close=float(o.close),entry_open=float(e.open),exit_close=float(z.close),
            earned_dividend_per_share=dividend,initial_response=initial,residual_5d_gross_return=float((z.close+dividend)/e.open-1),
            price_pre20_return=float(p.wealth/market.iloc[pre-20].wealth-1),price_log_sigma20=math.log(sigma),
            label_status="MATURE_HISTORICAL_LABEL"))
    data=pd.DataFrame(cases).sort_values("entry_at").reset_index(drop=True)
    cluster,previous_end=0,pd.Timestamp("1900-01-01",tz="Asia/Shanghai")
    for i,row in data.iterrows():
        start,end=pd.Timestamp(row.entry_at),pd.Timestamp(row.exit_at)
        if start>previous_end:
            cluster+=1
        data.loc[i,"event_cluster"]=cluster
        previous_end=max(previous_end,end)
    data.event_cluster=data.event_cluster.astype(int)
    csv(data,output/"全部准入事件_固定五日.csv")
    csv(pd.DataFrame(pending),output/"已准入但行情未成熟.csv")
    predictions,models,adjudications,trade_rows=[],[],[],[]
    fits=0
    for regime in consensus.training_regime.drop_duplicates():
        sample=data[data.training_regime==regime].copy()
        clusters=sample.event_cluster.nunique()
        if clusters<sum(protocol["minimum"].values()):
            adjudications.append({"regime":regime,"status":"NOT_RUN_SAMPLE_GATE","mature_events":len(sample),"independent_clusters":clusters,
                "required_clusters":sum(protocol["minimum"].values()),"evaluation_events":0,"MSE_A":None,"MSE_B":None})
            continue
        current=[]
        for index,row in sample.iterrows():
            train=sample[(sample.exit_at<row.observation_at)&(sample.event_cluster!=row.event_cluster)]
            if train.event_cluster.nunique()<protocol["minimum"]["train_clusters"]:
                continue
            values={}
            for arm in ("A","B"):
                pred,fit=ridge_predict(train,row,protocol["features_"+arm])
                values["prediction_"+arm]=pred
                models.append({"event_id":row.event_id,"regime":regime,"arm":arm,"training_events":train.event_id.tolist(),
                    "latest_label_mature_at":train.exit_at.max(),"decision_at":row.observation_at,**fit})
                fits+=1
            y=float(row.residual_5d_gross_return)
            record={"event_id":row.event_id,"stat_month":row.stat_month,"regime":regime,"event_cluster":int(row.event_cluster),
                "observation_at":row.observation_at,"entry_date":row.entry_date,"exit_date":row.exit_date,"training_events":len(train),
                "actual_5d_return":y,**values}
            record.update(squared_error_A=(y-values["prediction_A"])**2,squared_error_B=(y-values["prediction_B"])**2)
            record["MSE_improvement_A_minus_B"]=record["squared_error_A"]-record["squared_error_B"]
            current.append(record)
        frame=pd.DataFrame(current)
        assert frame.event_cluster.nunique()>=protocol["minimum"]["evaluation_clusters"]
        grouped=frame.groupby("event_cluster",sort=True).MSE_improvement_A_minus_B.mean().to_numpy()
        lb=lower_bound(grouped,protocol["bootstrap"])
        half=len(grouped)//2
        halves=[float(grouped[:half].mean()),float(grouped[half:].mean())]
        mse_a,mse_b=float(frame.squared_error_A.mean()),float(frame.squared_error_B.mean())
        passed=bool(grouped.mean()>0 and lb>0 and all(v>0 for v in halves))
        answer={"regime":regime,"status":"PASS_MSE_GATE" if passed else "REJECTED_FROZEN_NO_RELIABLE_MESSAGE_INCREMENT",
            "mature_events":len(sample),"independent_clusters":clusters,"evaluation_events":len(frame),
            "evaluation_start":frame.entry_date.min(),"evaluation_end":frame.exit_date.max(),"MSE_A":mse_a,"MSE_B":mse_b,
            "relative_MSE_improvement":1-mse_b/mse_a,"MSE_improvement_mean":float(grouped.mean()),
            "MSE_improvement_one_sided_90pct_lower":lb,"evaluation_half_improvements":halves,"MSE_gate_pass":passed,
            "entry_gate_status":"NOT_RUN_MSE_GATE"}
        if passed:
            eligible=[]
            for pred in current:
                row=sample[sample.event_id==pred["event_id"]].iloc[0]
                threshold=-round_trip(row.observation_close,row.observation_close,0,protocol["capital"]["main"],protocol)["net_return_on_capital"]
                if pred["prediction_B"]<=threshold:
                    continue
                main=None
                for capital in protocol["capital"].values():
                    result=round_trip(row.entry_open,row.exit_close,row.earned_dividend_per_share,capital,protocol)
                    trade_rows.append({"event_id":row.event_id,"event_cluster":int(row.event_cluster),"capital":capital,
                        "estimated_threshold_at_observation":threshold,"prediction_B":pred["prediction_B"],**result,"role":"ISOLATED_EVENT_COST_GATE_NOT_ACCOUNT"})
                    if capital==protocol["capital"]["main"]:
                        main=result["net_return_on_capital"]
                eligible.append({"cluster":int(row.event_cluster),"net":main})
            if eligible:
                nets=pd.DataFrame(eligible).groupby("cluster").net.mean().to_numpy()
                net_lb=lower_bound(nets,protocol["bootstrap"])
                positive=nets[nets>0]
                concentration=float(positive.max()/positive.sum()) if len(positive) else None
                entry_pass=bool(len(nets)>=12 and net_lb>0 and concentration is not None and concentration<=.3)
                answer.update(entry_gate_status="PASS_EVENT_COST_GATE" if entry_pass else "REJECTED_EVENT_COST_GATE",
                    independent_positive_forecast_entries=len(nets),mean_net_event_return=float(nets.mean()),net_mean_lower=net_lb,
                    largest_positive_contribution_fraction=concentration)
            else:
                answer.update(entry_gate_status="REJECTED_NO_POSITIVE_COST_ADJUSTED_ENTRIES",independent_positive_forecast_entries=0)
        adjudications.append(answer)
        predictions.extend(current)
    csv(pd.DataFrame(predictions),output/"A_B全部逐期预测.csv")
    csv(pd.DataFrame(trade_rows),output/"仅当预测门通过的逐事件成本检验.csv")
    save(output/"全部模型训练记录.json",models)
    save(output/"分口径裁决.json",adjudications)
    summaries=[]
    for regime,part in data.groupby("training_regime",sort=False):
        surprise=part.spread_surprise_proxy_pp.mask(part.spread_surprise_proxy_pp.abs()<1e-10,0.)
        positive=surprise>0
        negative=surprise<0
        summaries.append({"regime":regime,"n":len(part),"positive_surprises":int(positive.sum()),"negative_surprises":int(negative.sum()),
            "zero_surprises":int((surprise==0).sum()),
            "initial_same_sign":int((surprise*part.initial_response>0).sum()),
            "residual_same_sign":int((surprise*part.residual_5d_gross_return>0).sum()),
            "positive_surprise_mean_residual":float(part.loc[positive,'residual_5d_gross_return'].mean()) if positive.any() else None,
            "negative_surprise_mean_residual":float(part.loc[negative,'residual_5d_gross_return'].mean()) if negative.any() else None,
            "role":"DESCRIPTIVE_NOT_CAUSAL_NOT_TRADING_WIN_RATE"})
    save(output/"描述统计.json",summaries)
    status={"study_id":protocol["study_id"],"status":"FROZEN_HISTORICAL_MESSAGE_INCREMENT_ADJUDICATED",
        "universe_months":len(consensus),"admitted_consensus_months":int(consensus.consensus_admitted.sum()),
        "mature_event_labels":len(data),"pending_labels":len(pending),"new_model_fits":fits,"adjudications":adjudications,
        "C":"NOT_RUN_FUND_CLOCK_AND_COVERAGE","D":"NOT_RUN_FUND_CLOCK_AND_COVERAGE",
        "account_runs":0,"account_CAGR":None,"account_net_Sharpe":None,"retracement_exit_comparison":"NOT_RUN_ENTRY_NOT_VALIDATED",
        "strict_forward_events":0,"account_target_achieved":False,"market_data_end":str(dates.max().date()),
        "full_information_complete":False,"DSR":"NOT_COMPUTED_NO_ACCOUNT_AND_INCOMPLETE_UPSTREAM_SELECTION_DISTRIBUTION"}
    save(output/"summary.json",status)
    print(json.dumps(status,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description="补齐共识后的固定五日增量检验")
    parser.add_argument("action",choices=["freeze","analyse"])
    parser.add_argument("--study-dir",type=Path,default=DEFAULT_STUDY)
    parser.add_argument("--output-dir",type=Path)
    args=parser.parse_args()
    if args.action=="freeze":
        freeze()
    else:
        analyse(args.study_dir.resolve(),args.output_dir or args.study_dir/"results")
