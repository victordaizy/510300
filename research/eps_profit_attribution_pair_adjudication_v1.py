"""核实两组固定研报的归母利润语义，计算其条件覆盖影响。"""
from __future__ import annotations
import argparse
from datetime import datetime
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import pdfplumber

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/"config/510300_eps_profit_attribution_pair_adjudication_v1.json"
PROTOCOL=ROOT/"docs/510300_EPS_PROFIT_ATTRIBUTION_PAIR_ADJUDICATION_V1.md"
OUT=ROOT/"reports/research/510300_eps_profit_attribution_pair_adjudication_v1"


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8") as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write("\n")


def identity(path):
    return {"path":path.relative_to(ROOT).as_posix(),"size_bytes":path.stat().st_size,
            "sha256":hashlib.sha256(path.read_bytes()).hexdigest()}


def compact(text):
    return re.sub(r"[\s,]","",unicodedata.normalize("NFKC",text))


def detail_row(text,label,count):
    found=[]
    for line in unicodedata.normalize("NFKC",text).splitlines():
        match=re.search(r"(?:^|\s)"+re.escape(label)+r"\s+((?:[\d,]+(?:\.\d+)?\s*)+)",line)
        if match:
            values=re.findall(r"\d[\d,]*(?:\.\d+)?",match[1])
            if len(values)==count:
                found.append({"source_line":line,"values":[str(Decimal(x.replace(",",""))) for x in values]})
    if len(found)!=1:
        raise ValueError(f"{label}详细行不是唯一且完整的{count}列：{len(found)}")
    return found[0]


def source_paths(config):
    paths=[CONFIG,PROTOCOL,Path(__file__)]
    previous=ROOT/config["previous_diagnostic"]
    paths += [previous/"result.json",previous/"holdings_and_eps_evidence.parquet",previous/"selected_institution_company_evidence.parquet"]
    for pair in config["pairs"]:
        for field in ["current_report_id","prior_report_id"]:
            aid=pair[field]
            record=ROOT/config["original_record_root"]/f"{aid}.json"
            facts=ROOT/config["original_fact_root"]/f"{aid}.json"
            meta=read(record)
            paths += [record,facts,ROOT/meta["source"]["raw_pdf_path"],ROOT/meta["source"]["raw_html_path"],ROOT/meta["directory_record"]["source_raw_path"]]
    return sorted(set(paths))


def adjudicate_document(pair,role,config,extract):
    aid=pair[role+"_report_id"]
    record=read(ROOT/config["original_record_root"]/f"{aid}.json")
    facts=read(ROOT/config["original_fact_root"]/f"{aid}.json")
    raw=ROOT/record["source"]["raw_pdf_path"]
    if identity(raw)["sha256"]!=record["source"]["pdf_sha256"]:
        raise ValueError("研报原件哈希变化")
    if record["ts_code"]!=pair["symbol"] or str(record["provider_metadata"]["company_code"])!="80000007":
        raise ValueError("公司或机构身份不符")
    with pdfplumber.open(raw) as pdf:
        pages=[{"page":i+1,"text":p.extract_text() or ""} for i,p in enumerate(pdf.pages)]
        page_record={"report_id":aid,"source":identity(raw),"metadata":pdf.metadata,"pages":pages}
    if extract:
        save(OUT/"pages"/f"{aid}.json",page_record)
    elif extract is False and page_record!=read(OUT/"pages"/f"{aid}.json"):
        raise ValueError("原PDF页面重新提取不一致")
    front=pages[0]["text"]
    if pair["symbol"] not in compact(front) or pair["name"] not in front:
        raise ValueError("原PDF首页公司身份不符")
    target=config["target_fiscal_year"]
    fact=next(x for x in facts["facts"] if x["target_fiscal_year"]==target)
    years=[int(x[0]) for x in fact["header"]]
    if compact(fact["header_raw"]) not in compact(front) or compact(fact["net_profit_source_raw"]) not in compact(front):
        raise ValueError("首页精确预测行或年度顺序不符")
    summary=detail_row(front,fact["net_profit_source_label"]+"(百万元)",len(years))
    index=years.index(target)
    if Decimal(summary["values"][index])!=Decimal(fact["net_profit_value_exact"]):
        raise ValueError("首页目标年预测与旧事实不一致")
    detail_text=pages[pair["detail_page"]-1]["text"]
    if compact("利润表("+pair["detail_unit"]+")") not in compact(detail_text):
        raise ValueError("利润表单位不符")
    detailed=detail_row(detail_text,pair["detail_label"],len(years))
    scale=Decimal(1 if pair["detail_unit"]=="百万元" else 1000)
    comparisons=[]
    for year,amount,shown in zip(years,summary["values"],detailed["values"]):
        exact=Decimal(amount)/scale
        delta=abs(exact-Decimal(shown))
        if (scale==1 and delta!=0) or (scale==1000 and delta>=Decimal("0.5")):
            raise ValueError("首页与详细归母利润不能按披露精度对应")
        comparisons.append({"year":year,"front_million":amount,"detail_displayed":shown,"detail_unit":pair["detail_unit"],
                            "absolute_difference_in_detail_unit":str(delta),"same_parent_profit_row":True})
    negative_control=None
    if pair["symbol"]=="601166.SH":
        total=detail_row(detail_text,"净利润",len(years))
        exact=Decimal(fact["net_profit_value_exact"])/1000
        total_value=Decimal(total["values"][index])
        if abs(exact-total_value)<Decimal("0.5"):
            raise ValueError("目标年归母和总净利润无法排除混淆")
        negative_control={"target_year":target,"wrong_total_net_profit_displayed":str(total_value),
                          "correct_parent_profit_displayed":detailed["values"][index],"unit":"十亿元",
                          "front_million":fact["net_profit_value_exact"],"wrong_row_excluded":True,"source_line":total["source_line"]}
    clocks=[record["directory_record"]["publishDate"][:10],record["provider_metadata"]["notice_date"][:10],record["provider_metadata"]["eitime"][:10],facts["report_internal_date"],facts["conservative_information_date"]]
    available=max(clocks)
    origin=pd.Timestamp(config["origin"])
    cutoff=origin if role=="current" else origin-pd.Timedelta(days=90)
    if not pd.Timestamp(available)<cutoff or (cutoff-pd.Timestamp(available)).days>180:
        raise ValueError("报告可用日期不满足原90日和180日规则")
    anchor=str(config["same_historical_anchor_year"])
    return {"report_id":aid,"symbol":pair["symbol"],"role":role,"institution":"国信证券",
            "status":"PASS_DOCUMENT_SPECIFIC_PARENT_PROFIT_SEMANTICS_WITH_CURRENCY_LIMIT",
            "original_pdf":identity(raw),"original_url":record["source"]["pdf_url"],"front_label":fact["net_profit_source_label"],
            "canonical_definition":"PROFIT_ATTRIBUTABLE_TO_PARENT_COMPANY_SHAREHOLDERS",
            "summary_row":summary,"detailed_row":detailed,"detail_page":pair["detail_page"],"comparisons":comparisons,
            "target_year":target,"target_value_reported_million":fact["net_profit_value_exact"],
            "common_historical_anchor":{"year":int(anchor),"reported_million":summary["values"][years.index(int(anchor))]},
            "negative_control":negative_control,"conservative_information_date":available,"comparison_cutoff":str(cutoff.date()),
            "reported_monetary_units_reconciled":True,"iso_currency_independently_proven":False,
            "historical_immutable_snapshot_proven":False,"global_profit_label_alias_admitted":False}


def compute(extract=False):
    config=read(CONFIG)
    docs=[];pairs=[]
    for pair in config["pairs"]:
        current=adjudicate_document(pair,"current",config,extract)
        prior=adjudicate_document(pair,"prior",config,extract)
        docs.extend([current,prior])
        if current["common_historical_anchor"]!=prior["common_historical_anchor"]:
            raise ValueError("同一历史年度归母利润不相等")
        a,b=Decimal(current["target_value_reported_million"]),Decimal(prior["target_value_reported_million"])
        pairs.append({"symbol":pair["symbol"],"name":pair["name"],"target_fiscal_year":config["target_fiscal_year"],
                      "current_report_id":current["report_id"],"prior_report_id":prior["report_id"],
                      "current_reported_million":str(a),"prior_reported_million":str(b),
                      "relative_change":float(a/b-1),"symmetric_revision":float(2*(a-b)/(abs(a)+abs(b))),
                      "status":"PAIR_SPECIFIC_PARENT_PROFIT_REVISION_CANDIDATE_WITH_REPORTED_CURRENCY_LIMIT",
                      "old_label_mismatch_preserved":True,"original_factor_modified":False})
    previous=ROOT/config["previous_diagnostic"]
    frame=pd.read_parquet(previous/"holdings_and_eps_evidence.parquet")
    frame=frame.loc[frame.origin.eq(pd.Timestamp(config["origin"]))].copy()
    original=frame.copy(deep=True)
    def summary(data):
        known=np.isfinite(data.profit_revision)
        cover=float(data.loc[known,"stock_weight"].sum())
        unknown=float(data.loc[~known,"stock_weight"].sum())
        signed=float((data.loc[known,"stock_weight"]*np.sign(data.loc[known,"profit_revision"])).sum())
        return {"covered_company_count":int(known.sum()),"covered_stock_weight":cover,"unknown_stock_weight":unknown,
                "known_signed_weight":signed,"lower_bound":signed-unknown,"upper_bound":signed+unknown,
                "bounds_include_zero":bool(signed-unknown<=0<=signed+unknown)}
    before=summary(frame)
    added_weight=0.0
    for pair in pairs:
        mask=frame.ts_code.eq(pair["symbol"])
        if int(mask.sum())!=1 or np.isfinite(frame.loc[mask,"profit_revision"]).any():
            raise ValueError("条件修正目标不是唯一缺失公司")
        weight=float(frame.loc[mask,"stock_weight"].iloc[0])
        pair["previously_disclosed_stock_weight"]=weight
        added_weight+=weight
        frame.loc[mask,"profit_revision"]=pair["symmetric_revision"]
    after=summary(frame)
    best_case_lower=before["known_signed_weight"]+added_weight-(before["unknown_stock_weight"]-added_weight)
    changed=np.isfinite(frame.profit_revision).astype(int)-np.isfinite(original.profit_revision).astype(int)
    if int(changed.sum())!=2:
        raise ValueError("条件诊断改变了两家之外的覆盖")
    unchanged=[x for x in frame.columns if x!="profit_revision"]
    pd.testing.assert_frame_equal(frame[unchanged],original[unchanged],check_exact=True)
    result={"study_id":config["study_id"],"status":"PAIR_SEMANTICS_ADJUDICATED_CONDITIONAL_COVERAGE_STILL_INSUFFICIENT",
            "documents":docs,"pairs":pairs,"origin":config["origin"],"before":before,"conditional_after":after,
            "added_stock_weight":added_weight,"both_recovered_revisions_positive_best_case_lower_bound":best_case_lower,
            "even_both_positive_cannot_determine_full_stock_positive_revision":best_case_lower<=0,
            "currency_iso_independent_proof_added":False,"global_label_alias_admitted":False,"original_monthly_factors_modified":False,
            "original_accounts_modified":False,"new_return_labels":0,"new_model_fits":0,"new_accounts_generated":0,
            "new_network_requests":0,"new_random_draws":0,"large_source_queues_resumed":False,"goal_achieved":False}
    return frame,result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",choices=["run","verify"],required=True)
    parser.add_argument("--receipt")
    args=parser.parse_args()
    if args.mode=="run":
        OUT.mkdir(parents=True,exist_ok=True)
        save(OUT/"claim.json",{"started_at":now(),"existing_texts_and_old_strategy_outcomes_seen":True,
                              "new_strategy_preregistration":False,"files":[identity(p) for p in source_paths(read(CONFIG))]})
    else:
        for item in read(OUT/"claim.json")["files"]:
            if identity(ROOT/item["path"])!=item:
                raise ValueError("冻结来源或代码改变")
    frame,result=compute(extract=args.mode=="run")
    if args.mode=="run":
        frame.to_parquet(OUT/"conditional_holdings_evidence.parquet",index=False)
        frame.to_csv(OUT/"条件补充后的股票覆盖.csv",index=False,encoding="utf-8-sig")
        save(OUT/"result.json",{"completed_at":now(),**result})
    else:
        saved=read(OUT/"result.json")
        if result!={k:v for k,v in saved.items() if k!="completed_at"}:
            raise ValueError("保存结果重算不相等")
        existing=pd.read_parquet(OUT/"conditional_holdings_evidence.parquet")
        frame["origin"]=frame.origin.astype("datetime64[ns]")
        existing["origin"]=existing.origin.astype("datetime64[ns]")
        pd.testing.assert_frame_equal(frame,existing,check_exact=True)
        receipt={"status":"PASS_FOUR_ORIGINAL_PDFS_TWO_SEMANTIC_PAIRS_AND_CONDITIONAL_BOUNDS_RECOMPUTED",
                 "completed_at":now(),"original_pdfs":4,"source_pages_reextracted":20,"pairs":2,
                 "conditional_holdings_rows":len(frame),"new_network_requests":0,"new_model_fits":0,
                 "new_accounts":0,"goal_achieved":False}
        if args.receipt:
            save(Path(args.receipt),receipt)
        print(json.dumps(receipt,ensure_ascii=False),flush=True)
    print(json.dumps({k:result[k] for k in ["status","pairs","before","conditional_after","added_stock_weight","both_recovered_revisions_positive_best_case_lower_bound"]},ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
