"""从实际财报缺口生成补齐清单，下载首批原始报告，不读取策略收益。"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd
import pdfplumber
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.intraday_overnight_increment_v1 import now, require, write_json
from research.original_earnings_breadth_v1 import METRICS

PARENT = ROOT / "reports/research/510300_original_earnings_breadth_v1"
OUT = ROOT / "reports/research/510300_original_earnings_source_completion_v1"
DATA_ROOT = Path(r"E:\ResearchData\New project 8")
RAW = DATA_ROOT / "data/raw/510300_original_earnings_source_completion_v1"


def queue():
    events = pd.read_parquet(PARENT / "original_event_subset.parquet")
    facts = pd.read_parquet(PARENT / "selected_verified_facts.parquet")
    wide = facts.pivot(index="announcement_id", columns="metric_id", values="verified_value")
    ready = set(wide.index[np.isfinite(wide[list(METRICS)]).all(axis=1)])
    panel = pd.read_parquet(PARENT / "daily_member_facts.parquet", columns=["date", "ts_code", "announcement_id", "prior_announcement_id", "report_age_days", "company_valid"])
    focus = panel.loc[(panel.date >= "2017-01-01") & panel.report_age_days.between(0, 200) & ~panel.company_valid]
    ids = pd.concat([focus.announcement_id, focus.prior_announcement_id]).dropna()
    counts = ids[~ids.isin(ready)].value_counts().rename_axis("announcement_id").rename("affected_member_days").reset_index()
    result = counts.merge(events, on="announcement_id", how="left", validate="one_to_one")
    require(result.official_pdf_url.notna().all(), "缺口公告未对应官方原始地址")
    result = result.sort_values(["affected_member_days", "ts_code", "report_period", "announcement_id"], ascending=[False, True, True, True])
    result.to_parquet(OUT / "missing_original_reports_queue.parquet", index=False)
    result[["announcement_id", "ts_code", "sec_name", "report_period", "event_publication_date", "affected_member_days", "official_pdf_url"]].to_csv(OUT / "原始财报待补齐清单.csv", index=False, encoding="utf-8-sig")
    rank = result.groupby("ts_code").affected_member_days.sum().sort_values(ascending=False)
    selected = pd.concat([result.loc[result.ts_code == symbol].head(2) for symbol in rank.head(12).index]).reset_index(drop=True)
    selected.to_parquet(OUT / "batch_01_selected_before_download.parquet", index=False)
    write_json(OUT / "batch_01_protocol.json", {"registered_at": now(), "study_id": "510300_ORIGINAL_EARNINGS_SOURCE_COMPLETION_V1",
               "missing_original_documents": len(result), "affected_symbols": int(result.ts_code.nunique()),
               "selection": "按2017年至截止日缺失原始事实影响的成分股日数排序，取前十二家公司，每家公司取影响最多的两份报告；不读取策略收益。",
               "batch_documents": len(selected), "source_budget": 0, "maximum_workers": 4,
               "account_training_or_return_evaluation": "NOT_RUN_SOURCE_COLLECTION_ONLY",
               "financial_sector_note": "银行与非银金融需另行定义盈利信息，不能直接把企业经营现金流质量含义套给银行。"}, exclusive=True)
    return selected


def acquire(record):
    identifier = str(record["announcement_id"])
    path = RAW / f"{identifier}.pdf"
    out = {"announcement_id": identifier, "ts_code": record["ts_code"], "sec_name": record["sec_name"],
           "report_period": str(record["report_period"]), "event_publication_date": str(record["event_publication_date"]),
           "official_pdf_url": record["official_pdf_url"], "retrieved_at": now()}
    try:
        if not path.exists():
            response = requests.get(record["official_pdf_url"], timeout=(12, 35), headers={"User-Agent": "Mozilla/5.0", "Referer": "https://www.cninfo.com.cn/"})
            response.raise_for_status()
            require(response.content.startswith(b"%PDF-"), "官方地址没有返回PDF")
            path.write_bytes(response.content)
        content = path.read_bytes()
        summaries = []
        with pdfplumber.open(path) as pdf:
            pages = len(pdf.pages)
            for number, page in enumerate(pdf.pages[:12], 1):
                text = page.extract_text() or ""
                summaries.append({"page": number, "text": text})
            metadata = {str(k): str(v) for k, v in pdf.metadata.items()}
        out.update({"status": "PDF_ARCHIVED_FIRST_TWELVE_PAGES_EXTRACTED_PENDING_FACT_VALIDATION", "raw_path": path.relative_to(DATA_ROOT).as_posix(),
                    "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content), "page_count": pages,
                    "pdf_metadata": metadata, "extracted_pages": summaries,
                    "individually_verified_financial_facts": False, "used_in_round_eight": False})
    except Exception as exc:
        out.update({"status": "SOURCE_FETCH_OR_TEXT_EXTRACTION_FAILED", "error": str(exc)})
    write_json(OUT / "batch_01_records" / f"{identifier}.json", out)
    return {k: v for k, v in out.items() if k not in ("extracted_pages", "pdf_metadata")}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    require(not (OUT / "batch_01_result.json").exists(), "首批补齐来源结果已存在")
    selected = pd.read_parquet(OUT / "batch_01_selected_before_download.parquet") if (OUT / "batch_01_protocol.json").exists() else queue()
    rows = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(acquire, r) for r in selected.to_dict("records")]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(f"原始财报补齐 {len(rows)}/{len(selected)}：{row['sec_name']}，{row['status']}", flush=True)
    write_json(OUT / "batch_01_result.json", {"completed_at": now(), "status": "SOURCE_COLLECTION_CONTINUES_FINANCIAL_FACTS_NOT_YET_ADMITTED",
               "archived": sum(r["status"].startswith("PDF_ARCHIVED") for r in rows), "requested": len(rows),
               "financial_fact_extraction_complete": False, "portfolio_evaluation_run": False, "rows": sorted(rows, key=lambda r: r["announcement_id"])}, exclusive=True)


if __name__ == "__main__":
    main()
