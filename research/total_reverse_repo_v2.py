"""各期限逆回购公开信息重建及完整账户研究；不触发真实交易。"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from threadpoolctl import threadpool_limits
from research.adaptive_allocation_v1 import ROOT, model_for, eligible_training, summarize, save_account
from research.intraday_overnight_increment_v1 import now, write_json, require, normalize_dividends, holding_total_return, block_indices, return_metrics, interval
from research.policy_liquidity_quantity_v1 import FUNDING, DR007, NOTICE, asof_join, simulate_policy

STUDY = "510300_TOTAL_REVERSE_REPO_V2"
OUT = ROOT / "reports/research/510300_total_reverse_repo_v2"
PARENT = ROOT / "reports/research/510300_adaptive_allocation_v1"
CONFIG = ROOT / "config/510300_total_reverse_repo_v2.json"
MANIFEST = ROOT / "config/510300_total_reverse_repo_v2_manifest.json"
QCOLS = ["disclosed_log5", "disclosed_log20", "disclosed_surprise20", "disclosed_short_long", "disclosed_age", "disclosed_funding_interaction"]
BCOLS = ["buyout_actual_month_report_log20", "buyout_planned_log20", "all_disclosed_log20", "buyout_disclosed_share20", "regular_long_tenor_share20", "regular_weighted_tenor20"]
GROUPS = ["PRICE", "FUNDING", "SEVEN", "REGULAR", "ALL"]
ENSEMBLES = {"T1_PRIMARY_ALL": "ALL", "T2_REGULAR_ALL_TENOR": "REGULAR", "T3_SEVEN_DISCLOSURE": "SEVEN", "T4_FUNDING": "FUNDING", "T5_PRICE": "PRICE"}
NAMES = {"T1_PRIMARY_ALL":"主方案：各期限与买断式公开量", "T2_REGULAR_ALL_TENOR":"常规逆回购全期限", "T3_SEVEN_DISCLOSURE":"七天公告流量对照", "T4_FUNDING":"价格与资金利率对照", "T5_PRICE":"价格对照", "BUY_HOLD":"买入持有"}

def physical(path: Path) -> Path:
    p = path.relative_to(ROOT).as_posix()
    return Path(r"E:\ResearchData\New project 8") / p if p.startswith("data/") else path

def sha(path: Path) -> str:
    return hashlib.sha256(physical(path).read_bytes()).hexdigest()

def ident(path: Path) -> dict:
    return {"path":path.relative_to(ROOT).as_posix(),"bytes":physical(path).stat().st_size,"sha256":sha(path)}

def compact(value: str) -> str:
    return re.sub(r"\s+", "", value).replace(",", "").replace("，", "")

def parse_amount(value: str) -> float:
    m = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)(万亿元|亿元|万元|元)", compact(value))
    require(m is not None, f"金额单位或数值无法裁定：{value}")
    return float(m.group(1)) * {"万亿元":10000,"亿元":1,"万元":0.0001,"元":0.00000001}[m.group(2)]

def term_days(value: str) -> int:
    v = compact(value)
    exact = re.search(r"[（(](\d+)天[）)]", v)
    if exact:
        return int(exact.group(1))
    exact = re.fullmatch(r"(\d+)天(?:期)?", v)
    require(exact is not None, f"期限没有明确自然日数：{value}")
    return int(exact.group(1))

def amount_from_row(cells: list[str], header: list[str]) -> float:
    indices = [i for i,x in enumerate(header) if "中标量" in x]
    if not indices:
        indices = [i for i,x in enumerate(header) if x in ("操作量", "交易量")]
    require(len(indices) == 1 and indices[0] < len(cells), f"逆回购金额表头不明确：{header} / {cells}")
    return parse_amount(cells[indices[0]])

def inverse_only_rows(payload: bytes) -> list[dict]:
    """按表格前的业务标题区分逆回购与同一公告内的中期借贷便利。"""
    s=BeautifulSoup(payload.decode("utf-8"),"lxml")
    zoom=s.select_one("#zoom")
    require(zoom is not None,"公告缺正文")
    full=compact(zoom.get_text(" ",strip=True))
    output=[]
    for table in zoom.find_all("table"):
        if table.find("table") is not None:
            continue
        txt=compact(table.get_text(" ",strip=True))
        before=full[:full.find(txt)]
        tags={k:before.rfind(k) for k in ("逆回购操作情况","MLF操作情况","中期借贷便利操作情况","正回购操作情况","央行票据")}
        if tags["逆回购操作情况"]<0 or max(tags,key=tags.get)!="逆回购操作情况":
            continue
        rows=[[compact(td.get_text(" ",strip=True)) for td in tr.find_all(["td","th"],recursive=False)] for tr in table.find_all("tr")]
        for cells in rows:
            if cells and re.match(r"^\d+(?:天|个月|年)",cells[0]):
                output.append({"cells":cells,"header":rows[0]})
    return output

def prepare_sources() -> None:
    require(not MANIFEST.exists(), "冻结后不能更改本轮来源")
    inventory = json.loads((OUT / "ordinary_structure_inventory.json").read_text(encoding="utf-8"))
    notices = pd.read_parquet(physical(NOTICE))
    lookup = {r.raw_path:r for r in notices.itertuples()}
    ops, status, unresolved = [], [], []
    for item in inventory:
        old = lookup[item["path"]]
        body = compact(item["text"])
        rows = item["rows"]
        if "MLF" in body or "中期借贷便利" in body:
            rows=inverse_only_rows(physical(ROOT/item["path"]).read_bytes())
        kind = "UNKNOWN"
        parsed = []
        if rows and "逆回购" in body:
            for rr in rows:
                amount = amount_from_row(rr["cells"], rr["header"])
                term = term_days(rr["cells"][0])
                parsed.append((term,amount))
            require(len(set(parsed)) == len(parsed), "同公告逆回购期限重复")
            kind = "ORDINARY_ACTUAL"
            total = sum(a for _,a in parsed)
            explicit = re.findall(r"开展了?([0-9.]+(?:万亿元|亿元))逆回购",body)
            if len(explicit) == 1:
                require(abs(parse_amount(explicit[0])-total)<1e-6, f"公告正文和分期限总量不同：{item['path']}")
        elif not rows and any(z in body for z in ("不开展","暂停","无逆回购操作","逆回购操作量为零")) and ("公开市场" in body or "逆回购" in body):
            kind = "EXPLICIT_NO_ORDINARY_OPERATION"
            parsed = [(0,0.0)]
        elif "央行票据" in body:
            kind = "CENTRAL_BANK_BILL_EXCLUDED"
        elif "正回购" in body and "逆回购" not in body:
            kind = "REPO_DRAIN_EXCLUDED"
        elif "现券买断" in body and "特别国债" in body:
            kind = "OUTRIGHT_GOVERNMENT_BOND_PURCHASE_EXCLUDED"
        elif not rows and ("MLF" in body or "中期借贷便利" in body):
            kind = "MLF_ONLY_EXCLUDED"
        if kind == "UNKNOWN":
            unresolved.append(item)
        status.append({"date":old.notice_date,"published_at":old.published_at,"source_url":old.source_url,"raw_path":item["path"],"classification":kind})
        for term, amount in parsed:
            ops.append({"operation_date":old.notice_date,"published_at":old.published_at,"tenor_days":term,
                        "amount_100m":amount,"seven_amount_100m":amount if term==7 else 0.0,
                        "kind":kind,"source_url":old.source_url,"raw_path":item["path"],"raw_sha256":old.raw_sha256})
    write_json(OUT / "unresolved_ordinary_notices.json",unresolved)
    pd.DataFrame(status).to_csv(OUT / "ordinary_notice_classification.csv",index=False,encoding="utf-8-sig")
    require(not unresolved, f"还有 {len(unresolved)} 篇公告类型未裁定")
    regular=pd.DataFrame(ops)
    regular=regular.loc[regular.operation_date>="2015-01-01"].reset_index(drop=True)
    require(not regular.duplicated(["source_url","tenor_days"]).any(), "重复的常规逆回购事件")
    regular.to_parquet(OUT / "ordinary_all_tenor_operations.parquet",index=False)
    buyout=[]
    for r in json.loads((OUT / "buyout_source_records.json").read_text(encoding="utf-8")):
        if not r["included_before_cutoff"]:
            continue
        body=compact(r["body"])
        planned = "将以" in body or "将开展" in body
        kind = "BUYOUT_PLANNED" if planned else "BUYOUT_MONTHLY_ACTUAL_DISCLOSURE"
        stamp=pd.Timestamp(r["published_at"])
        period=re.search(r"(\d{4})年(\d+)月(?:(\d+)日)?",body)
        require(period is not None,"买断式公告无操作期间")
        operation_date=pd.Timestamp(f"{period.group(1)}-{int(period.group(2)):02d}-{int(period.group(3)):02d}") if period.group(3) else pd.NaT
        month=f"{period.group(1)}-{int(period.group(2)):02d}"
        s=BeautifulSoup(physical(ROOT/r["raw_path"]).read_bytes().decode("utf-8"),"lxml")
        pairs=[]
        for tr in s.select("#zoom tr"):
            cells=[compact(td.get_text(" ",strip=True)) for td in tr.find_all(["td","th"],recursive=False)]
            if cells and re.match(r"^\d+个月",cells[0]):
                pairs.append((term_days(cells[0]),parse_amount(cells[1])))
        totalmatch=re.search(r"开展了?([0-9.]+(?:万亿元|亿元))买断式逆回购",body)
        require(totalmatch is not None,"买断式公告总量未解析")
        total=parse_amount(totalmatch.group(1))
        if not pairs:
            tm=re.search(r"期限为(\d+个月[（(]\d+天[）)])",body)
            require(tm is not None,"买断式无明确期限")
            pairs=[(term_days(tm.group(1)),total)]
        require(abs(sum(a for _,a in pairs)-total)<1e-6,"买断式总额与期限表不符")
        due=re.search(r"到期日为(\d{4})年(\d+)月(\d+)日",body)
        maturity=pd.Timestamp(f"{due.group(1)}-{int(due.group(2)):02d}-{int(due.group(3)):02d}") if due else pd.NaT
        for term, amount in pairs:
            buyout.append({"operation_month":month,"operation_date":operation_date,"published_at":stamp,
                           "tenor_days":term,"amount_100m":amount,"kind":kind,"explicit_maturity_date":maturity,
                           "source_url":r["source_url"],"raw_path":r["raw_path"],"raw_sha256":r["sha256"]})
    b=pd.DataFrame(buyout)
    require(not b.duplicated(["operation_month","operation_date","tenor_days","kind"]).any(),"买断式重复记录")
    # 当前目录实际月报到2025年5月，随后是预告，避免同一月份两种披露重复合并。
    overlap=set(b.loc[b.kind=="BUYOUT_PLANNED","operation_month"]) & set(b.loc[b.kind!="BUYOUT_PLANNED","operation_month"])
    require(not overlap,"同月同时存在月报和预告，须先建立逐笔对应关系")
    b.to_parquet(OUT / "buyout_tenor_disclosures.parquet",index=False)
    year=regular.operation_date.dt.year
    annual=regular.assign(year=year).groupby(["year","tenor_days"],as_index=False).amount_100m.sum()
    annual.to_csv(OUT/"annual_ordinary_tenor_totals.csv",index=False,encoding="utf-8-sig")
    y2016=annual.loc[annual.year==2016].set_index("tenor_days").amount_100m
    reference={7:179000.0,14:39000.0,28:30000.0}
    differences={str(k):float(y2016.get(k,0)-v) for k,v in reference.items()}
    # 官方年报为0.1万亿元的四舍五入，容许误差为500亿元。
    require(all(abs(v)<=500 for v in differences.values()) and abs(y2016.sum()-248000)<=500,
            "2016年分期限汇总无法对上央行年报")
    write_json(OUT/"source_reconstruction_receipt.json",{
        "status":"PASS_ALL_ORDINARY_TENORS_AND_BUYOUT_DISCLOSURES_SEPARATED","created_at":now(),
        "ordinary_notice_count":len(notices),"classifications":pd.Series([r['classification'] for r in status]).value_counts().to_dict(),
        "ordinary_operation_rows":len(regular),"ordinary_tenors":sorted(int(x) for x in regular.tenor_days.unique() if x),
        "ordinary_total_100m":float(regular.amount_100m.sum()),"seven_day_total_100m":float(regular.seven_amount_100m.sum()),
        "other_tenor_total_100m":float((regular.amount_100m-regular.seven_amount_100m).sum()),
        "buyout_notice_count":b.source_url.nunique(),"buyout_rows":len(b),"buyout_kind_rows":b.kind.value_counts().to_dict(),
        "official_2016_rounding_difference_100m":differences,"all_actual_daily_net_injection":"NOT_RECONSTRUCTED_MISSING_ACTUAL_DAY_AND_MATURITY_EVIDENCE",
        "source_gap_zero_imputation":False,"planned_recategorized_as_actual":False,
        "reference_2016_report":"https://www.pbc.gov.cn/zhengcehuobisi/125207/125227/125957/3066656/9f140965cb464e3bbd1904a84e5d7cde/2017021719463365852.pdf"})
    print("各期限与买断式来源重建完成，年度总量已核对。",flush=True)

def disclosed_features(data: pd.DataFrame, regular: pd.DataFrame, buyout: pd.DataFrame) -> pd.DataFrame:
    result=data.copy()
    stamps=pd.to_datetime(regular.published_at)
    bt=pd.to_datetime(buyout.published_at)
    matrix=[]
    for date in result.date:
        t=date+pd.Timedelta(hours=15)
        known=stamps<=t
        r20=known & (stamps>t-pd.Timedelta(days=20))
        r5=known & (stamps>t-pd.Timedelta(days=5))
        age=(t-stamps[known].max()).total_seconds()/86400 if known.any() else np.nan
        row={"repo_available_age":age,"repo_last_published_at":stamps[known].max() if known.any() else pd.NaT}
        for prefix,amountcol in (("seven","seven_amount_100m"),("regular","amount_100m")):
            s5=float(regular.loc[r5,amountcol].sum())
            s20=float(regular.loc[r20,amountcol].sum())
            row.update({f"{prefix}_amount5":s5,f"{prefix}_amount20":s20,f"{prefix}_disclosed_log5":np.log1p(s5),
                        f"{prefix}_disclosed_log20":np.log1p(s20),f"{prefix}_disclosed_short_long":np.log1p(s5/5)-np.log1p(s20/20),
                        f"{prefix}_disclosed_age":age})
        bk=(bt<=t)&(bt>t-pd.Timedelta(days=20))
        actual=float(buyout.loc[bk&(buyout.kind=="BUYOUT_MONTHLY_ACTUAL_DISCLOSURE"),"amount_100m"].sum())
        plan=float(buyout.loc[bk&(buyout.kind=="BUYOUT_PLANNED"),"amount_100m"].sum())
        total=row["regular_amount20"]+actual+plan
        den=row["regular_amount20"]
        row.update({"buyout_actual_month_report_log20":np.log1p(actual),"buyout_planned_log20":np.log1p(plan),
                    "all_disclosed_log20":np.log1p(total),"buyout_disclosed_share20":(actual+plan)/total if total else 0.0,
                    "regular_long_tenor_share20":float(regular.loc[r20&(regular.tenor_days>7),"amount_100m"].sum())/den if den else 0.0,
                    "regular_weighted_tenor20":float((regular.loc[r20,"amount_100m"]*regular.loc[r20,"tenor_days"]).sum())/den if den else 0.0})
        matrix.append(row)
    for col in matrix[0]:
        result[col]=[r[col] for r in matrix]
    for prefix in ("seven","regular"):
        val=result[f"{prefix}_disclosed_log20"]
        mu=val.shift(1).rolling(252,min_periods=60).mean()
        sd=val.shift(1).rolling(252,min_periods=60).std(ddof=1)
        result[f"{prefix}_disclosed_surprise20"]=((val-mu)/sd.replace(0,np.nan)).where(sd!=0,0)
    return result

def features() -> tuple[pd.DataFrame,list[str]]:
    data=pd.read_parquet(PARENT/"features.parquet")
    pc=json.loads((PARENT/"input_receipt.json").read_text(encoding="utf-8"))["features"]
    regular=pd.read_parquet(OUT/"ordinary_all_tenor_operations.parquet")
    buyout=pd.read_parquet(OUT/"buyout_tenor_disclosures.parquet")
    result=disclosed_features(data,regular,buyout)
    result["origin_time"]=result.date+pd.Timedelta(hours=15)
    dates=pd.DatetimeIndex(result.date)
    dr=pd.read_parquet(physical(DR007)).sort_values("date")
    ix=dates.searchsorted(pd.to_datetime(dr.date),side="right")
    dr=dr.loc[ix<len(dates)].copy()
    dr["dr_available_at"]=dates[ix[ix<len(dates)]].to_numpy()+pd.Timedelta(hours=9,minutes=30)
    dr=dr.rename(columns={"date":"dr_date"}).sort_values(["dr_available_at","dr_date"]).drop_duplicates("dr_available_at",keep="last")
    result=asof_join(result,dr[["dr_date","dr_available_at","dr007"]],"dr_available_at")
    notices=pd.read_parquet(physical(NOTICE))
    rates=notices.loc[notices.seven_day_rate_percent.notna(),["published_at","seven_day_rate_percent"]].rename(columns={"published_at":"rate_published_at"})
    result=asof_join(result,rates,"rate_published_at")
    result["dr007_known"]=result.dr007/100
    result["policy_rate_known"]=result.seven_day_rate_percent/100
    result["funding_gap"]=result.dr007_known-result.policy_rate_known
    result["dr007_change5"]=result.dr007_known.diff(5)
    result["dr007_change20"]=result.dr007_known.diff(20)
    result["funding_gap_mean20"]=result.funding_gap.rolling(20).mean()
    for prefix in ("seven","regular"):
        result[f"{prefix}_disclosed_funding_interaction"]=result[f"{prefix}_disclosed_surprise20"]*result.funding_gap
    result["dr_age_days"]=(result.date-result.dr_date).dt.total_seconds()/86400
    cols=pc+FUNDING+[p+"_"+c for p in ("seven","regular") for c in QCOLS]+BCOLS
    result["feature_valid"] &= np.isfinite(result[cols]).all(axis=1)&(result.repo_available_age<=10)&(result.dr_age_days<=7)&(result.date>="2015-03-01")
    for c in ("repo_last_published_at","dr_available_at","rate_published_at"):
        mask=result[c].notna()
        require((result.loc[mask,c]<=result.loc[mask,"origin_time"]).all(),"修正因子发生信息时钟穿越")
    return result,pc

def columns(group: str, pc: list[str]) -> list[str]:
    cols=list(pc)
    if group!="PRICE":cols+=FUNDING
    if group in ("SEVEN","REGULAR","ALL"):
        prefix="seven" if group=="SEVEN" else "regular"
        cols += [prefix+"_"+c for c in QCOLS]
    if group=="ALL":cols+=BCOLS
    return cols

def freeze() -> None:
    require(not CONFIG.exists() and not MANIFEST.exists(),"本轮已经冻结")
    receipt=json.loads((OUT/"source_reconstruction_receipt.json").read_text(encoding="utf-8"))
    require(receipt["status"].startswith("PASS_"),"来源未完成")
    config=json.loads((ROOT/"config/510300_adaptive_allocation_v1.json").read_text(encoding="utf-8"))
    config.update({"study_id":STUDY,"primary":"T1_PRIMARY_ALL","models":[{"id":f"{g}_{k}_H20","group":g,"kind":k,"horizon":20} for g in GROUPS for k in ("RIDGE","ET")],
                   "information_clock":"PUBLICATION_TIME_AT_OR_BEFORE_15_00","quantity_measure":"DISCLOSED_AMOUNT_NOT_ACTUAL_DAILY_NET_CASH",
                   "source_budget":0,"historical_vendor_first_delivery_proven":False,"common_feature_support":True})
    write_json(CONFIG,config,exclusive=True)
    paths=[CONFIG,Path(__file__),ROOT/"docs/510300_TOTAL_REVERSE_REPO_V2_PROTOCOL.md",ROOT/"docs/510300_FACTOR_SCOPE_CORRECTION_20260906.md",
           ROOT/"tests/test_total_reverse_repo_v2.py",ROOT/"config/510300_research_authority_v6.json",ROOT/"research/adaptive_allocation_v1.py",
           ROOT/"research/policy_liquidity_quantity_v1.py",ROOT/"research/intraday_overnight_increment_v1.py",PARENT/"features.parquet",PARENT/"input_receipt.json",
           ROOT/config["inputs"]["dividends"],NOTICE,DR007]
    paths += [OUT/x for x in ("ordinary_all_tenor_operations.parquet","buyout_tenor_disclosures.parquet","source_reconstruction_receipt.json","ordinary_structure_inventory.json","buyout_source_records.json")]
    raw=pd.read_parquet(OUT/"ordinary_all_tenor_operations.parquet").raw_path.tolist()+pd.read_parquet(OUT/"buyout_tenor_disclosures.parquet").raw_path.tolist()
    paths += [ROOT/x for x in set(raw)]
    entries=[]
    for n,p in enumerate(sorted(set(paths)),1):
        entries.append(ident(p))
        if n%500==0:print(f"已固定输入 {n}/{len(set(paths))}",flush=True)
    write_json(MANIFEST,{"study_id":STUDY,"frozen_at":now(),"files":entries,"own_returns_read_before_freeze":False,"underlying_history_previously_observed":True},exclusive=True)
    print(json.dumps({"状态":"第七轮已冻结","清单哈希":sha(MANIFEST)},ensure_ascii=False),flush=True)

def train(data:pd.DataFrame,pc:list[str],dividends:pd.DataFrame,config:dict) -> dict:
    first=int(np.flatnonzero(data.date>=config["evaluation_start"])[0])-1
    quarters=data.date.dt.to_period("Q").astype(str).to_numpy()
    cuts=[first]+[t for t in range(first+1,len(data)-1) if quarters[t]!=quarters[t-1]]+[len(data)-1]
    y=np.full(len(data),np.nan)
    for t in range(len(data)-21):y[t]=holding_total_return(data,dividends,t+1,t+21)[0]
    predictions,receipts={},[]
    (OUT/"final_models").mkdir(exist_ok=True)
    for item in config["models"]:
        cols=columns(item["group"],pc)
        x=data[cols].to_numpy(float)
        pred=np.full(len(data),np.nan)
        for t,end in zip(cuts[:-1],cuts[1:]):
            ix=eligible_training(data,t,20,None)
            require(len(ix)>=config["minimum_train_samples"],"共同有效成熟训练样本不足")
            seed=config["random_seed"]+int(data.date.iloc[t].strftime("%Y%m%d"))
            model=model_for(item["kind"],seed)
            eligible=np.arange(t,end)[data.feature_valid.iloc[t:end].to_numpy()]
            with threadpool_limits(limits=1):
                model.fit(x[ix],y[ix])
                if len(eligible):pred[eligible]=model.predict(x[eligible])
            receipts.append({"model":item["id"],"fit_origin":data.date.iloc[t],"train_rows":len(ix),"first_train_origin":data.date.iloc[ix[0]],
                             "last_train_origin":data.date.iloc[ix[-1]],"last_label_exit":data.date.iloc[ix[-1]+21],
                             "training_array_sha256":hashlib.sha256(x[ix].tobytes()+y[ix].tobytes()).hexdigest(),"feature_columns":cols})
        predictions[item["id"]]=pred
        joblib.dump({"model":model,"columns":cols,"specification":item,"last_fit_receipt":receipts[-1]},OUT/"final_models"/(item["id"]+".joblib"),compress=3)
        print(f"第七轮模型已完成：{item['id']}",flush=True)
    for key,group in ENSEMBLES.items():predictions[key]=np.mean(np.column_stack([predictions[f"{group}_{kind}_H20"] for kind in ("RIDGE","ET")]),axis=1)
    pd.DataFrame(receipts).to_json(OUT/"training_receipts.json",orient="records",date_format="iso",force_ascii=False,indent=2)
    pd.DataFrame({"date":data.date,**predictions}).to_parquet(OUT/"predictions.parquet",index=False)
    pd.DataFrame({"date":data.date,"Y20":y}).to_parquet(OUT/"labels.parquet",index=False)
    return predictions

def run(expected:str) -> None:
    require(sha(MANIFEST)==expected,"第七轮清单哈希不符")
    items=json.loads(MANIFEST.read_text(encoding="utf-8"))["files"]
    for n,item in enumerate(items,1):
        require(sha(ROOT/item["path"])==item["sha256"],f"冻结文件变化：{item['path']}")
        if n%700==0:print(f"冻结输入已核对 {n}/{len(items)}",flush=True)
    config=json.loads(CONFIG.read_text(encoding="utf-8"))
    write_json(OUT/"RUN_STARTED.json",{"started_at":now(),"manifest_sha256":expected},exclusive=True)
    data,pc=features()
    data.to_parquet(OUT/"features.parquet",index=False)
    write_json(OUT/"feature_receipt.json",{"groups":{g:columns(g,pc) for g in GROUPS},"valid_origins":int(data.feature_valid.sum()),
               "no_view_evaluation_origins":data.loc[(data.date>="2019-12-31")&(data.date<config['data_cutoff'])&~data.feature_valid,"date"].tolist(),
               "clock_rule":"公开时间不晚于当日15点","disclosed_amount_is_daily_actual_cash":False})
    dividends=normalize_dividends(pd.read_csv(physical(ROOT/config["inputs"]["dividends"])))
    predictions=train(data,pc,dividends,config)
    metrics,yearly,eras,uncertainty=[],[],[],{}
    for costid,cost in config["costs"].items():
        accounts={}
        for key in [*predictions,"BUY_HOLD"]:
            ledger,decisions=simulate_policy(data,dividends,config,cost,key,predictions.get(key),20)
            accounts[key]=ledger
            save_account(OUT/"evaluation"/costid,key,ledger,decisions)
        baseline=summarize(accounts["BUY_HOLD"],config)
        for key,ledger in accounts.items():
            m={"cost":costid,"model":key,**summarize(ledger,config)}
            m["annualized_return_excess_vs_buy_hold"]=m["annualized_return"]-baseline["annualized_return"]
            m["meets_point_target"]=m["net_sharpe"] is not None and m["net_sharpe"]>=1.2
            metrics.append(m)
            for year,part in ledger.groupby(ledger.date.dt.year):yearly.append({"cost":costid,"model":key,"year":int(year),**summarize(part,config)})
            for era,start,end in (("2020—2021","2020-01-01","2021-12-31"),("2022—2023","2022-01-01","2023-12-31"),("2024—终点","2024-01-01",config["data_cutoff"])):
                part=ledger.loc[(ledger.date>=start)&(ledger.date<=end)]
                eras.append({"cost":costid,"model":key,"era":era,**summarize(part,config)})
        rr=pd.DataFrame({"date":accounts["BUY_HOLD"].date,**{k:v.net_return.to_numpy() for k,v in accounts.items()}})
        rr.to_parquet(OUT/f"{costid}_all_evaluation_returns.parquet",index=False)
        rng=np.random.default_rng(config["random_seed"])
        samples={"primary_sharpe":[],"increment_vs_regular":[],"increment_vs_seven":[],"increment_vs_funding":[]}
        for _ in range(config["bootstrap_repetitions"]):
            ix=block_indices(rng,len(rr),config["bootstrap_day_block"])
            r=rr[config["primary"]].to_numpy()[ix]
            samples["primary_sharpe"].append(return_metrics(r,242)["net_sharpe"])
            for name,key in (("regular","T2_REGULAR_ALL_TENOR"),("seven","T3_SEVEN_DISCLOSURE"),("funding","T4_FUNDING")):
                samples[f"increment_vs_{name}"].append(float((r-rr[key].to_numpy()[ix]).mean()*242))
        uncertainty[costid]={k+"_95_interval":interval(v) for k,v in samples.items()}
        print(f"第七轮 {costid} 的16个完整账户已完成",flush=True)
    frame=pd.DataFrame(metrics)
    frame.to_csv(OUT/"metrics.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(yearly).to_csv(OUT/"yearly_metrics.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(eras).to_csv(OUT/"era_metrics.csv",index=False,encoding="utf-8-sig")
    best=frame.loc[(frame.cost=="BASE")&(frame.model!="BUY_HOLD")].sort_values("net_sharpe",ascending=False).iloc[0]
    primary=frame.loc[frame.model==config["primary"]]
    reached=bool(primary.loc[primary.cost=="BASE","meets_point_target"].iloc[0])
    result={"study_id":STUDY,"completed_at":now(),"manifest_sha256":expected,"status":"HISTORICAL_PRIMARY_POINT_TARGET_MET_VALIDATION_PENDING" if reached else "COMPLETED_PRIMARY_TARGET_NOT_MET",
            "primary":primary.to_dict("records"),"post_selected_best_base":best.to_dict(),"number_of_candidates":15,"evaluation_accounts":32,"trained_model_count":10,
            "uncertainty":uncertainty,"independent_validation":"NOT_ESTABLISHED_ALREADY_OBSERVED_HISTORY","position_impact":0,
            "quantity_scope":"各期限常规实际操作公告量，加分别记录的买断式月报与预告公开量；不等同于实际每日净投放"}
    write_json(OUT/"result.json",result,exclusive=True)
    write_json(OUT/"uncertainty.json",uncertainty)
    print(json.dumps(result,ensure_ascii=False,default=str),flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser(description="重建总逆回购公开口径，冻结并执行完整账户")
    parser.add_argument("--prepare-sources",action="store_true")
    parser.add_argument("--freeze",action="store_true")
    parser.add_argument("--run",action="store_true")
    parser.add_argument("--expected-manifest-sha256")
    args=parser.parse_args()
    require(sum([args.prepare_sources,args.freeze,args.run])==1,"请选择来源重建、冻结、运行之一")
    if args.prepare_sources:prepare_sources()
    elif args.freeze:freeze()
    else:
        require(bool(args.expected_manifest_sha256),"缺少清单哈希")
        run(args.expected_manifest_sha256)
