"""补齐固定调查字段的公开来源；来源不足时不读取市场标签。"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
from zoneinfo import ZoneInfo

from bs4 import BeautifulSoup
import requests


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_private_manager_intent_source_extension_v2"
PARENT = ROOT / "reports/research/510300_private_manager_intent_v1"
URLS = [
    "https://www.chinanews.com/stock/2014/05-06/6139535.shtml",
    "https://www.nbd.com.cn/articles/2016-04-08/996688.html",
    "https://epaper.mrjjxw.com/shtml/mrjjxw/20190102/157438.shtml",
    "https://fund.eastmoney.com/a/1590%2C201912051313977856.html",
    "https://www.myfund.com/article/674.html",
    "https://ronghesz.com/xinwenzixun/touzixinwen/24.html",
    "https://www.yicai.com/news/101071398.html",
    "https://www.sohu.com/a/573494132_639898",
    "https://www.stcn.com/article/detail/697639.html",
    "https://epaper.cs.com.cn/zgzqb/html/2023-11/03/nw.D110000zgzqb_20231103_2-A03.htm",
    "https://finance.caijing.com.cn/20251020/5119888.shtml",
    "https://finance.eastmoney.com/a/202510203538086560.html",
    "https://www.simuwang.com/news/236398.html",
    "https://www.simuwang.com/news/232010.html",
    "https://paper.cnstock.com/html/2023-05/06/content_1762812.htm",
    "https://www.nbd.com.cn/articles/2020-02-04/1405412.html",
]

# 只记录同期原文明示的数值，不由环比变化或其他指数倒推。
ADDITIONS = [
    ("2014-05", "2014-05-06", 114.29, None, "011ee5b02b511878", "中国证券报报道的中新网转载；文内其他仓位为公募测算，不混入私募调查"),
    ("2016-04", "2016-04-08", 114.29, None, "0bdfefba81b51976", "每经网直接报道；未给同口径调查平均仓位"),
    ("2019-01", "2019-01-02", 113.48, 57.0, "2789642298118c3e", "每日经济新闻同期报纸；月份、计划与调查仓位明确"),
    ("2019-12", "2019-12-05", 107.58, 67.04, "a8d118a7fbde05cb", "保留该转载页面日期；不混入后段公募仓位89.07%"),
    ("2020-01", "2020-01-06", 108.74, None, "0cce4110510dac32", "金融机构公开文章；文内88.99%为股票型基金仓位，未用于本调查"),
    ("2021-06", "2021-06-03", 106.34, 82.0, "eeeb4c010d068d7d", "第一财经发布的完整调查；不用发布方8月迟到摘要回填6月"),
    ("2022-08", "2022-08-01", 104.0, 78.0, "f85ca254c3952b0e", "按原文计划指数记录；分项占比19%与70%等合计并非100%，保留内部疑点，不反推或修正指数"),
    ("2022-10", "2022-10-11", 105.5, 72.0, "6e95e40df37e6d9d", "上海证券报署名报道的证券时报网转载"),
    ("2023-11", "2023-11-03", 116.67, 80.0, "77969021d7a37a5d", "中国证券报同期报纸；股票主观多头调查平均仓位"),
    ("2025-10", "2025-10-20", 111.76, 78.0, "9a01980f822bf3c2", "上海证券报报道的东方财富转载；采用页面日期，不提前到当前PDF封面日"),
]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def fetch_one(url: str) -> dict:
    key = hashlib.sha256(url.encode()).hexdigest()[:16]
    receipt_path = OUT / "receipts" / f"{key}.json"
    if receipt_path.exists():
        return load(receipt_path)
    receipt = {"key": key, "url": url, "retrieved_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()}
    try:
        response = requests.get(url, timeout=(8, 20), headers={"User-Agent": "Mozilla/5.0"})
        receipt.update(http_status=response.status_code, final_url=response.url)
        if response.status_code != 200:
            receipt["status"] = "ACCESS_FAILED"
        else:
            raw_path = OUT / "raw_local_only" / f"{key}.html"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_bytes(response.content)
            soup = BeautifulSoup(response.content, "html.parser", from_encoding=response.apparent_encoding)
            receipt["title"] = soup.title.get_text(" ", strip=True) if soup.title else ""
            metas = []
            for meta in soup.find_all("meta"):
                label = meta.get("name", meta.get("property", ""))
                if any(k in label.lower() for k in ("publish", "date", "time")):
                    metas.append({"label": label, "content": meta.get("content", "")})
            receipt["date_metadata"] = metas
            for item in soup(["script", "style", "nav", "footer"]):
                item.decompose()
            text = soup.get_text(" ", strip=True)
            text_path = OUT / "raw_local_only" / f"{key}.txt"
            text_path.write_text(text, encoding="utf-8")
            compact = re.sub(r"\s+", "", text)
            receipt.update(
                status="SAVED", bytes=len(response.content),
                sha256=hashlib.sha256(response.content).hexdigest(),
                raw_path=raw_path.relative_to(OUT).as_posix(),
                text_path=text_path.relative_to(OUT).as_posix(),
                date_candidates=list(dict.fromkeys(re.findall(r"20\d{2}[年/-]\d{1,2}[月/-]\d{1,2}(?:日)?", text[:6000])))[:15],
                plan_evidence=re.findall(r".{0,35}仓位增减.{0,95}", compact)[:6],
                exposure_evidence=re.findall(r".{0,40}(?:平均仓位|整体仓位).{0,65}", compact)[:5],
                source_links=[a.get("href") for a in soup.find_all("a",href=True) if "来源" in a.get_text()][:8],
            )
    except requests.RequestException as exc:
        receipt.update(status="ACCESS_FAILED", error=str(exc))
    write_json(receipt_path, receipt)
    return receipt


def fetch() -> None:
    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(fetch_one, URLS))
    write_json(OUT / "download_summary.json", records)
    for row in records:
        print(json.dumps({k:row.get(k) for k in ("key", "status", "title", "date_candidates", "date_metadata", "plan_evidence", "exposure_evidence")}, ensure_ascii=False))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        fields=list(dict.fromkeys(k for r in rows for k in r))
        writer=csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def admit() -> None:
    records={r["key"]:r for r in load(OUT/"download_summary.json")}
    old=load(PARENT/"admitted_sources.json")
    old_admitted=[r for r in old if r["status"]=="ADMITTED_HISTORICAL_RECONSTRUCTION"]
    old_months={r["month"] for r in old_admitted}
    added=[]
    for month,date,plan,exposure,key,note in ADDITIONS:
        rec=records[key]
        if rec["status"]!="SAVED" or month in old_months or date[:7]>month:
            raise ValueError(f"来源或月份资格失败：{month}")
        raw=(OUT/rec["raw_path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=rec["sha256"]:
            raise ValueError(f"来源文件身份改变：{month}")
        compact=re.sub(r"\s+","",(OUT/rec["text_path"]).read_text(encoding="utf-8"))
        if not any(abs(float(x)-plan)<1e-8 for x in re.findall(r"仓位增减.{0,18}?(?:为|为：)(\d+(?:\.\d+)?)",compact)):
            raise ValueError(f"原文未匹配明示的计划指数：{month}")
        if exposure is not None and not any(abs(float(x)-exposure)<1e-8 for x in re.findall(r"(\d+(?:\.\d+)?)%",compact)):
            raise ValueError(f"原文未匹配人工核对的调查仓位：{month}")
        date_forms=(date, date.replace("-","/"), f"{date[:4]}年{date[5:7]}月{date[8:]}日", f"{int(date[:4])}年{int(date[5:7])}月{int(date[8:])}日")
        if not any(x in compact for x in date_forms):
            raise ValueError(f"页面日期未核对：{month}")
        added.append({
            "month":month,"available_date":date,"plan_index":plan,"survey_exposure_pct":exposure,
            "key":key,"url":rec["url"],"source_sha256":rec["sha256"],
            "clock":"DATED_ARTICLE","status":"ADMITTED_HISTORICAL_RECONSTRUCTION",
            "note":note,"archived_first_version":False,"actual_cash_flow":False,
            "extension_version":"V2","numeric_and_date_present_in_saved_source":True,
            "survey_components_internally_questioned":month=="2022-08",
        })
    combined=sorted(old_admitted+added,key=lambda x:x["month"])
    if len({r["month"] for r in combined})!=len(combined):
        raise ValueError("合并月份重复。")
    covered={r["month"]:r for r in combined}
    months=[]
    for year in range(2014,2027):
        for month in range(1,13):
            key=f"{year:04d}-{month:02d}"
            if key>"2026-08":continue
            row=covered.get(key)
            months.append({"month":key,"available_date":row["available_date"] if row else None,
                           "plan_index":row["plan_index"] if row else None,
                           "survey_exposure_pct":row["survey_exposure_pct"] if row else None,
                           "status":row["status"] if row else "MISSING_OR_ORIGINAL_EXCLUSION_PRESERVED"})
    both=[r for r in combined if r.get("survey_exposure_pct") is not None]
    protocol=load(PARENT/"protocol.json")
    min_train=int(protocol["estimator"]["minimum_mature_training_events"])
    min_eval=int(protocol["promotion_gate"]["minimum_evaluation_events"])
    upper_b=max(0,len(combined)-min_train)
    upper_c=max(0,len(both)-min_train)
    result={
        "study_id":"510300_PRIVATE_MANAGER_INTENT_SOURCE_EXTENSION_V2",
        "completed_at":datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "status":"SOURCE_EXTENSION_COMPLETE_NO_VIEW_INSUFFICIENT_PREDICTIVE_SAMPLE",
        "parent_admitted_events":len(old_admitted),"new_admitted_events":len(added),
        "combined_admitted_events":len(combined),"combined_intent_and_exposure_events":len(both),
        "full_month_universe":len(months),"missing_or_original_excluded_months":len(months)-len(combined),
        "new_source_urls_attempted":len(records),"new_sources_saved":sum(r["status"]=="SAVED" for r in records.values()),
        "new_sources_failed":sum(r["status"]!="SAVED" for r in records.values()),
        "minimum_mature_training_events":min_train,"minimum_evaluation_events":min_eval,
        "upper_bound_evaluation_B":upper_b,"upper_bound_evaluation_C":upper_c,
        "bound_note":"乐观计数上界N减24，尚未扣标签未成熟或缺少交易日等；不是实际评价次数",
        "sample_gate_B":upper_b>=min_eval,"sample_gate_C":upper_c>=min_eval,
        "plan_above100":sum(r["plan_index"]>100 for r in combined),
        "plan_equal100":sum(r["plan_index"]==100 for r in combined),
        "plan_below100":sum(r["plan_index"]<100 for r in combined),
        "questioned_internal_components_months":[r["month"] for r in added if r["survey_components_internally_questioned"]],
        "parent_records_unchanged":all(r in combined for r in old_admitted),
        "new_market_outcome_reads":0,"new_model_fits":0,"new_bootstraps":0,"new_accounts":0,
        "account_stage":"NOT_RUN_PARENT_PREDICTIVE_SAMPLE_GATE",
        "net_sharpe":"NOT_COMPUTED","annualized_return":"NOT_COMPUTED","max_drawdown":"NOT_COMPUTED",
        "strict_forward_events":0,"whole_goal_achieved":False,"goal_status":"ACTIVE","position_impact":0,
    }
    if result["sample_gate_B"] or result["sample_gate_C"]:
        raise RuntimeError("样本计数已允许继续，请先另行冻结收益读取，不能在来源脚本自动运行模型。")
    write_json(OUT/"new_admitted_sources.json",added)
    write_json(OUT/"combined_admitted_sources.json",combined)
    write_json(OUT/"result.json",result)
    write_csv(OUT/"新增月份与来源.csv",added)
    write_csv(OUT/"全部月份覆盖与缺口.csv",months)
    public_receipts=[{k:r.get(k) for k in ("key","url","retrieved_at","http_status","status","title","sha256","bytes","date_candidates","date_metadata","error")} for r in records.values()]
    write_json(OUT/"来源下载回执_无正文.json",public_receipts)
    for name in ("protocol.json","freeze_receipt.json","summary.json","admitted_sources.json","source_plan.json"):
        destination=OUT/"parent_snapshot"/name
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes((PARENT/name).read_bytes())
    write_json(OUT/"source_freeze_receipt.json",{
        "frozen_at":datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "source_set":[{"key":r["key"],"source_sha256":r["source_sha256"]} for r in added],
        "frozen_files":{n:hashlib.sha256((OUT/n).read_bytes()).hexdigest() for n in ("new_admitted_sources.json","combined_admitted_sources.json","source_plan.json","parent_snapshot/protocol.json","parent_snapshot/admitted_sources.json")},
        "new_market_outcome_reads":0,"new_model_fits":0,"new_accounts":0,
    })
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description="扩展私募意向来源，不读取未来价格")
    parser.add_argument("action", choices=["fetch", "admit"])
    args=parser.parse_args()
    if args.action == "fetch":
        fetch()
    else:
        admit()
