"""补齐首批金融盈利锚点所需的上一年年报与上年同期原件。"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pypdfium2 as pdfium
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.financial_original_facts_v1_1 import BASE, NAMES, OUT as FACTS, compact, now, write_json

OUT = ROOT / "reports/research/510300_financial_ttm_dependencies_v1"
RAW = BASE / "data/raw/510300_financial_ttm_dependencies_v1"
QUEUE = ROOT / "reports/research/510300_original_earnings_source_completion_v1/missing_original_reports_queue.parquet"
MANIFEST = ROOT / "config/510300_financial_ttm_dependencies_v1_manifest.json"


def freeze():
    facts = pd.read_parquet(FACTS / "current_original_financial_facts.parquet")
    docs = facts[["ts_code", "sec_name", "report_period", "announcement_id"]].drop_duplicates()
    docs["report_period"] = pd.to_datetime(docs.report_period, format="mixed")
    anchors = docs.sort_values("report_period").groupby("ts_code", as_index=False).tail(1)
    queue = pd.read_parquet(QUEUE)
    existing = set(zip(docs.ts_code, docs.report_period))
    requested, dependencies = [], []
    for a in anchors.to_dict("records"):
        year = a["report_period"].year
        for typ, period in [("上一年全年", pd.Timestamp(year=year-1,month=12,day=31)), ("上年同期累计", a["report_period"]-pd.DateOffset(years=1))]:
            owned = (a["ts_code"], period) in existing
            match = queue.loc[queue.ts_code.eq(a["ts_code"]) & queue.report_period.eq(period)]
            if not owned and len(match) != 1:
                raise ValueError(f"必要原始报告目录不唯一：{a['ts_code']} {period}")
            dependency = {"anchor_announcement_id": a["announcement_id"], "ts_code": a["ts_code"], "sec_name": a["sec_name"],
                          "anchor_period": str(a["report_period"].date()), "dependency_type": typ, "needed_period": str(period.date()),
                          "already_has_current_original_facts": owned}
            if not owned:
                record = match.iloc[0].to_dict()
                record["reason"] = typ
                requested.append(record)
                dependency["requested_announcement_id"] = record["announcement_id"]
            dependencies.append(dependency)
    selected = pd.DataFrame(requested).drop_duplicates("announcement_id").sort_values(["ts_code", "report_period"])
    OUT.mkdir(parents=True, exist_ok=True)
    if MANIFEST.exists():
        raise RuntimeError("滚动盈利原始依赖已登记")
    selected.to_parquet(OUT / "selected_before_download.parquet", index=False)
    write_json(OUT / "dependency_map.json", {"anchors": anchors.to_dict("records"), "dependencies": dependencies}, exclusive=True)
    paths = [Path(__file__), ROOT / "docs/510300_FINANCIAL_TTM_DEPENDENCIES_V1.md", FACTS / "current_original_financial_facts.parquet", QUEUE,
             ROOT / "research/financial_original_facts_v1_1.py", OUT / "selected_before_download.parquet", OUT / "dependency_map.json"]
    write_json(MANIFEST, {"registered_at": now(), "purpose": "ORIGINAL_TTM_COMPONENT_COLLECTION_NO_RETURN_READ", "budget_cny": 0,
                         "anchors": len(anchors), "requested_documents": len(selected),
                         "files": [{"path": p.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]}, exclusive=True)
    print(f"滚动盈利依赖已登记：{len(anchors)}个锚点，需新增{len(selected)}份原件", flush=True)


def check_subject(pages, source):
    front = compact("\n".join(pages[:4]))
    code_front = compact("\n".join(pages[:12]))
    year = int(str(source["report_period"])[:4])
    month = pd.Timestamp(source["report_period"]).month
    terms = {12: ["年度报告", "年报"], 9: ["第三季度", "三季度"], 6: ["半年度报告", "中期报告"], 3: ["第一季度", "一季度"]}[month]
    passed = NAMES[source["ts_code"]] in front and source["ts_code"].split(".")[0] in code_front and str(year) in front and any(x in front for x in terms)
    return {"status": "PASS_FRONT_SUBJECT_CODE_REPORT_PERIOD" if passed else "NO_VIEW_FRONT_SUBJECT_OR_PERIOD_NOT_PROVEN",
            "financial_facts_admitted": False, "expected_full_name": NAMES[source["ts_code"]]}


def run():
    for item in json.loads(MANIFEST.read_text("utf-8"))["files"]:
        if hashlib.sha256((ROOT/item["path"]).read_bytes()).hexdigest() != item["sha256"]:
            raise RuntimeError("滚动盈利依赖输入变化")
    if (OUT / "result.json").exists():
        raise RuntimeError("本批原件收集已有完成结果")
    RAW.mkdir(parents=True, exist_ok=True)
    records, stopped = [], False
    selected = pd.read_parquet(OUT / "selected_before_download.parquet")
    for row in selected.to_dict("records"):
        aid = str(row["announcement_id"])
        receipt = OUT / "records" / (aid + ".json")
        cached = OUT / "page_texts" / (aid + ".json")
        if receipt.exists() and cached.exists():
            result = json.loads(receipt.read_text("utf-8"))
            records.append(result)
            continue
        source = {k: row[k] for k in ["announcement_id", "ts_code", "sec_name", "report_period", "event_publication_date", "official_pdf_url", "announcement_title", "reason"]}
        source.update({"retrieved_at": now(), "facts_used_by_a_strategy": False})
        try:
            if stopped:
                raise RuntimeError("本批来源已明确限流或拒绝，后续未发请求")
            path = RAW / (aid + ".pdf")
            if not path.exists():
                response = requests.get(row["official_pdf_url"], headers={"User-Agent": "Mozilla/5.0", "Referer": "https://www.cninfo.com.cn/"}, timeout=(12, 35))
                if response.status_code in (403, 429):
                    stopped = True
                    raise RuntimeError(f"来源拒绝或限流：{response.status_code}")
                response.raise_for_status()
                if not response.content.startswith(b"%PDF-"):
                    raise ValueError("官方地址未返回PDF")
                path.write_bytes(response.content)
            content = path.read_bytes()
            pdf = pdfium.PdfDocument(content)
            pages = []
            try:
                for i in range(len(pdf)):
                    page = pdf[i]
                    textpage = page.get_textpage()
                    try:
                        pages.append(textpage.get_text_range())
                    finally:
                        textpage.close()
                        page.close()
            finally:
                pdf.close()
            source.update({"raw_path": path.relative_to(BASE).as_posix(), "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content), "page_count": len(pages),
                           "status": "ORIGINAL_ARCHIVED_FULL_TEXT_PENDING_FINANCIAL_EXTRACTION", "subject": check_subject(pages, source)})
            write_json(cached, {"source": source, "pages": pages})
        except Exception as error:
            source.update({"status": "SOURCE_COLLECTION_INCOMPLETE", "error": str(error)})
        write_json(receipt, source)
        records.append(source)
        write_json(OUT / "progress.json", {"updated_at": now(), "processed": len(records), "requested": len(selected), "last": source})
        print(f"滚动盈利原件 {len(records)}/{len(selected)}：{source['sec_name']} {str(source['report_period'])[:10]}，{source['status']}", flush=True)
    write_json(OUT / "result.json", {"completed_at": now(), "status": "TTM_COMPONENT_SOURCES_ARCHIVED_FINANCIAL_SCOPE_EXTRACTION_NEXT", "requested": len(selected),
                                   "archived": sum(r["status"].startswith("ORIGINAL_ARCHIVED") for r in records),
                                   "front_subject_passed": sum(r.get("subject", {}).get("status", "").startswith("PASS_") for r in records),
                                   "new_financial_facts_admitted": 0, "new_portfolio_evaluation": False, "rows": records}, exclusive=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="补全原始年报与同期报表，为滚动盈利重建准备证据")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", action="store_true")
    group.add_argument("--run", action="store_true")
    args = parser.parse_args()
    freeze() if args.freeze else run()
