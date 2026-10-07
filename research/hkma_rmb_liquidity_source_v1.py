"""取得香港金管局人民币流动资金安排使用量，不把设施借款当作股票净流入。"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
import pdfplumber

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research.selected_mix_reappraisal_v1 import read, save, now, digest

OUT = ROOT / "reports/research/510300_hkma_rmb_liquidity_source_v1"
API = "https://api.hkma.gov.hk/public/market-data-and-statistics/daily-monetary-statistics/usage-rmb-liquidity-fac"
DOCS = {
    "api_parameters.html": "https://apidocs.hkma.gov.hk/documentation/",
    "api_fields.html": "https://apidocs.hkma.gov.hk/documentation/market-data-and-statistics/daily-monetary-statistics/usage-rmb-liquidity-fac/",
    "terms_20250926.pdf": "https://brdr.hkma.gov.hk/eng/doc-ldg/docId/getPdf/20250926-4-EN/20250926-4-EN.pdf",
    "terms_20220722.pdf": "https://brdr.hkma.gov.hk/eng/doc-ldg/docId/getPdf/20220722-4-EN/20220722-4-EN.pdf",
    "archive_20161101.html": "https://www.hkma.gov.hk/eng/data-publications-and-research/data-and-statistics/daily-monetary-statistics/2016/11/ms-20161101/",
    "archive_20230414.html": "https://www.hkma.gov.hk/eng/data-publications-and-research/data-and-statistics/daily-monetary-statistics/2023/04/ms-20230414/",
    "archive_20251009.html": "https://www.hkma.gov.hk/eng/data-publications-and-research/data-and-statistics/daily-monetary-statistics/2025/10/ms-20251009/",
    "archive_20260924.html": "https://www.hkma.gov.hk/eng/data-publications-and-research/data-and-statistics/daily-monetary-statistics/2026/09/ms-20260924/",
}


def get(name, url, params=None):
    target = OUT / "raw" / name
    receipt_path = OUT / "raw" / (name+".receipt.json")
    if receipt_path.exists():
        return read(receipt_path)
    receipt = {"started_at": now(), "url": url, "params": params}
    try:
        response = requests.get(url, params=params, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        target.write_bytes(response.content)
        receipt.update(status_code=response.status_code, completed_at=now(), response_url=response.url,
                       path=target.relative_to(ROOT).as_posix(), sha256=digest(target))
        response.raise_for_status()
        receipt["success"] = True
    except Exception as exc:
        receipt.update(success=False, error=f"{type(exc).__name__}: {exc}")
    save(receipt_path, receipt, True)
    return receipt


def run():
    save(OUT / "collection_plan.json", {"at": now(), "earlier_probe_preserved": True,
         "api_query": "按官方pagesize=1000、offset分页，2016-11-01至2026-09-24，日期升序；最多7个新增页面，含原探测共不超过8页。",
         "explanatory_sources": DOCS,
         "next_candidate_information_fixed_before_price_labels": "只研究日间回购已用金额在14:00与16:00间的净变化；变换为(16时金额-14时金额)/(1+16时金额+14时金额)，金额单位为百万元，双方均零时值为零。",
         "reason": "同日两时点可避免把2025年10月额度从200亿元变300亿元的机械变化当成逐日借款冲击；不能排除价格、抵押品规则或支付需求变化。",
         "availability": "按每日发布日期翌日00:00保守视为可用，再合并到A股09:00；历史实际首次送达未认证。",
         "excluded_interpretations": ["股市净流入", "设施总使用量", "纯粹非银融资需求", "直接外汇贬值压力"],
         "returns_read": False, "new_models": 0, "new_accounts": 0}, True)
    receipts, records = [], []
    completed = False
    for offset in range(0, 7000, 1000):
        params = {"pagesize": 1000, "offset": offset, "choose": "end_of_date", "from": "2016-11-01", "to": "2026-09-24",
                  "sortby": "end_of_date", "sortorder": "asc"}
        receipt = get(f"full_page_{offset:04d}.json", API, params)
        receipts.append(receipt)
        if not receipt["success"]:
            break
        response = read(ROOT / receipt["path"])
        if not response.get("header", {}).get("success"):
            break
        rows = response["result"]["records"]
        for row in rows:
            records.append({**row, "raw_path": receipt["path"], "raw_sha256": receipt["sha256"]})
        print(f"人民币设施使用量已取{len(records)}行。", flush=True)
        if len(rows) < 1000:
            completed = True
            break
    with ThreadPoolExecutor(max_workers=3) as pool:
        docs = list(pool.map(lambda item: get(item[0], item[1]), DOCS.items()))
    save(OUT / "collection_receipts.json", {"api": receipts, "documents": docs}, True)
    if not completed:
        save(OUT / "result.json", {"at": now(), "status": "SOURCE_INCOMPLETE_API_PAGINATION", "rows_retrieved": len(records),
             "new_models": 0, "new_accounts": 0, "goal_achieved": False}, True)
        return
    frame = pd.DataFrame(records).rename(columns={"end_of_date": "source_date"})
    frame["source_date"] = pd.to_datetime(frame.source_date).astype("datetime64[ns]")
    assert not frame.source_date.duplicated().any()
    assert frame.source_date.between("2016-11-01", "2026-09-24").all()
    for name in ["intraday_repo_at_1400", "intraday_repo_at_1600"]:
        frame[name] = pd.to_numeric(frame[name], errors="raise")
        assert (frame[name].dropna() >= 0).all()
    a, b = frame.intraday_repo_at_1400, frame.intraday_repo_at_1600
    frame["afternoon_usage_change"] = (b-a)/(1+b+a)
    frame["available_at"] = frame.source_date.dt.tz_localize("Asia/Shanghai") + pd.Timedelta(days=1)
    assert frame.afternoon_usage_change.dropna().between(-1, 1).all()
    frame = frame.sort_values("source_date").reset_index(drop=True)
    frame.to_parquet(OUT / "rmb_liquidity_usage.parquet", index=False)
    extracts = {}
    for receipt in docs:
        if not receipt["success"]:
            continue
        path = ROOT / receipt["path"]
        if path.suffix == ".pdf":
            with pdfplumber.open(path) as document:
                value = "\n".join(page.extract_text() or "" for page in document.pages)
        else:
            value = BeautifulSoup(path.read_bytes(), "html.parser").get_text(" ", strip=True)
        extracts[path.name] = value
    save(OUT / "official_method_text.json", extracts, True)
    changes = {"announcement_date": "2025-09-26", "effective_date": "2025-10-09",
               "intraday_quota_RMB_billion_before": 20, "intraday_quota_RMB_billion_after": 30,
               "overnight_quota_RMB_billion_before": 20, "overnight_quota_RMB_billion_after": 10,
               "new_T1_tenors": ["two-week", "one-month"], "source": DOCS["terms_20250926.pdf"],
               "identification_limit": "API三类使用量字段不覆盖所有期限的全部设施，不能加总为完整总量。"}
    save(OUT / "known_methodology_change.json", changes, True)
    result = {"at": now(), "study_id": "510300_HKMA_RMB_LIQUIDITY_SOURCE_V1",
              "status": "SOURCE_READY_INTRADAY_USAGE_CHANGE_ONLY" if "terms_20250926.pdf" in extracts else "SOURCE_PENDING_METHOD_TERMS",
              "rows": len(frame), "first_date": frame.source_date.min(), "last_date": frame.source_date.max(),
              "complete_afternoon_pairs": int(frame.afternoon_usage_change.notna().sum()),
              "positive_changes": int(frame.afternoon_usage_change.gt(0).sum()),
              "zero_changes": int(frame.afternoon_usage_change.eq(0).sum()),
              "negative_changes": int(frame.afternoon_usage_change.lt(0).sum()),
              "api_pages": len(receipts), "new_document_requests": len(docs),
              "failed_documents": [r["url"] for r in docs if not r["success"]],
              "historical_first_vintage_verified": False, "raw_latest_probe_date_20260925_used_as_historical_feature": False,
              "new_models": 0, "new_accounts": 0, "new_independent_forward_observations": 0,
              "goal_achieved": False, "orders_authorized": False}
    save(OUT / "result.json", result, True)
    print(f"来源已保存：{len(frame)}日，完整下午时点{result['complete_afternoon_pairs']}日，待检验的一项变化量已固定。", flush=True)


if __name__ == "__main__":
    run()
