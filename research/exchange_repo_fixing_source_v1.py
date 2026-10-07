"""取得带证券代码的交易所回购定盘原文，保留字段和历史时钟边界。"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from urllib.parse import urlencode

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.exchange_repo_source_probe_v1 import OUT as PROBE, fetch
from research.selected_mix_reappraisal_v1 import read, save, now, digest

OUT = ROOT / "reports/research/510300_exchange_repo_fixing_source_v1"
STUDY = "510300_EXCHANGE_REPO_FIXING_SOURCE_V1"
METHOD = "https://www.sse.com.cn/lawandrules/regulations/csrcannoun/c/10787043/files/6e68fb5e2cf54a3fb3461692ae169297.pdf"


def initial():
    if (OUT / "protocol.json").exists():
        raise RuntimeError("本次来源计划已存在，禁止重新采集覆盖。")
    for folder in ["code", "results"]:
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    windows = []
    for year in range(2017, 2027):
        begin = "2017-05-22" if year == 2017 else f"{year}-01-01"
        end = "2026-09-25" if year == 2026 else f"{year + 1}-01-01"
        params = {"isPagination": "false", "BEGIN_DATE": begin, "END_DATE": end,
                  "CODE": "204007", "sqlId": "COMMON_SSE_PL_ZQXX_ZQZYSHGDPLL_L"}
        windows.append({"name": f"gc007_year_{year}", "begin": begin, "end": end,
                        "url": "https://query.sse.com.cn/sseQuery/commonQuery.do?" + urlencode(params)})
    save(OUT / "protocol.json", {"at": now(), "study_id": STUDY, "requests": windows,
         "method_url": METHOD, "maximum_requests": 11,
         "field_origin": "当前上交所披露页面实际引用脚本；CODE=204007、REPU_NAME=GC007、RATE为定盘百分数、BIZ_DATE为经济日期。每个查询不超过一年。",
         "excluded_old_curve": "旧曲线RATE_1DAY与现代GC007一致，AVG_RATE_1DAY与现代GC001一致；RATE_7DAY对应前者五日平均。保存矛盾，不修改原字段、不用旧曲线入模。",
         "method_boundary": "2025年统计指标标准指引BD-II-4说明2017-05-22起定盘为全天成交量加权平均，不能称为收盘报价或七天无风险利率。",
         "clock_boundary": "BIZ_DATE不是发布或接收时间。2011通知记载次日发布但已列失效，无法据此认证2026及全部历史首版时钟。后续只能在明确延迟假设下做开发研究，不能由历史序列得到当前可用状态。",
         "admission": "代码、简称、日期、非重复和数值范围检查；与另一个已保存官方查询重叠值一致；如有缺日不填补。403/429停止，不绕过。",
         "new_strategy_returns": 0, "new_accounts": 0, "independent_forward_observations": 0,
         "goal_achieved": False}, True)
    (OUT / "code" / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    receipts = []
    for item in [{"name": "methodology_2025_pdf", "url": METHOD}, *windows]:
        receipt = fetch(item["url"], item["name"])
        receipts.append({"name": item["name"], **receipt})
        print(f"已保存{item['name']}：{receipt['status']}，{receipt.get('bytes', 0)}字节。", flush=True)
        if receipt.get("stop_new_requests"):
            break
    save(OUT / "requests_result.json", {"at": now(), "requests": receipts}, True)


def build():
    if (OUT / "result.json").exists():
        raise RuntimeError("来源解析已有终态，不覆盖。")
    plan = read(OUT / "protocol.json")
    records, failures = [], []
    for window in plan["requests"]:
        receipt_path = PROBE / "requests" / f"{window['name']}.json"
        if not receipt_path.exists():
            failures.append({"name": window["name"], "status": "NOT_REQUESTED"})
            continue
        receipt = read(receipt_path)
        if receipt["status"] != "HTTP_OK_UNPARSED":
            failures.append({"name": window["name"], "status": receipt["status"]})
            continue
        raw = ROOT / receipt["raw_path"]
        assert digest(raw) == receipt["sha256"]
        payload = read(raw)
        assert not payload.get("actionErrors") and not payload.get("fieldErrors")
        rows = payload["result"]
        for entry in rows:
            assert entry["CODE"] == "204007" and entry["REPU_NAME"] == "GC007"
            date = pd.Timestamp(entry["BIZ_DATE"])
            assert pd.Timestamp(window["begin"]) <= date <= pd.Timestamp(window["end"])
            rate = float(entry["RATE"])
            assert np.isfinite(rate) and 0 <= rate <= 100
            records.append({"date": date, "gc007_percent": rate, "code": entry["CODE"],
                            "raw_rate_text": entry["RATE"], "source_url": receipt["url"],
                            "source_hash": receipt["sha256"], "retrieved_at": receipt["received_at"],
                            "historical_first_publication_verified": False})
    if not records:
        save(OUT / "result.json", {"at": now(), "status": "NO_ADMITTED_SERIES", "failures": failures}, True)
        return
    frame = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    duplicated = frame[frame.date.duplicated(keep=False)]
    if not duplicated.empty:
        assert duplicated.groupby("date").gc007_percent.nunique().eq(1).all(), "同日不同数值须隔离来源。"
        assert duplicated.groupby("date").source_hash.nunique().eq(1).all(), "跨响应重复须另行判断。"
        save(OUT / "results/exact_source_duplicates.json", {
            "at": now(), "reason": "初次解析因重复检查失败；原响应内2017-12-29有两条完全相同记录。保留两条原始记录，规范序列中仅计一个经济日期。",
            "records": duplicated.to_dict("records"), "same_response_same_date_same_value_only": True}, True)
        frame = frame.drop_duplicates().reset_index(drop=True)
    assert frame.date.is_unique
    (OUT / "code" / "source_build_completion.py").write_bytes(Path(__file__).read_bytes())
    check = read(PROBE / "raw/modern_fixing_recent_json.bin")["result"]
    lookup = frame.set_index("date").gc007_percent
    for row in check:
        assert lookup.loc[pd.Timestamp(row["BIZ_DATE"])] == float(row["RATE"])
    old = read(PROBE / "raw/official_recent_json.bin")["result"]
    overlap = []
    for row in old:
        day = pd.Timestamp(row["TRADE_DATE"])
        if day in lookup:
            overlap.append({"date": day, "modern_gc007": lookup.loc[day],
                            "old_rate_1day": float(row["RATE_1DAY"]), "old_rate_7day": float(row["RATE_7DAY"])})
    market = pd.read_parquet(ROOT / "reports/research/510300_repo_segmentation_daily_v1/inputs/market.parquet", columns=["date"])
    expected = pd.to_datetime(market.date)
    expected = expected[expected.between(frame.date.min(), frame.date.max())]
    missing = expected[~expected.isin(frame.date)]
    frame.to_parquet(OUT / "results/gc007_fixing.parquet", index=False)
    save(OUT / "results/field_comparison.json", overlap, True)
    finish()


def finish():
    if (OUT / "result.json").exists():
        raise RuntimeError("来源已有终态。")
    import pdfplumber
    frame = pd.read_parquet(OUT / "results/gc007_fixing.parquet")
    market = pd.read_parquet(ROOT / "reports/research/510300_repo_segmentation_daily_v1/inputs/market.parquet", columns=["date"])
    expected = pd.to_datetime(market.date)
    expected = expected[expected.between(frame.date.min(), frame.date.max())]
    missing = expected[~expected.isin(frame.date)]
    excerpts = []
    with pdfplumber.open(PROBE / "raw/methodology_2025_pdf.bin") as document:
        for number in [145, 146]:
            text = document.pages[number].extract_text()
            assert "定盘利率" in text and "2017" in text
            excerpts.append({"pdf_page_one_based": number + 1, "text": text})
    save(OUT / "results/methodology_excerpts.json", excerpts, True)
    check = read(PROBE / "raw/modern_fixing_recent_json.bin")["result"]
    overlap = read(OUT / "results/field_comparison.json")
    failures = [r for r in read(OUT / "requests_result.json")["requests"] if r["status"] != "HTTP_OK_UNPARSED"]
    (OUT / "code/source_completion_pdfplumber.py").write_bytes(Path(__file__).read_bytes())
    save(OUT / "result.json", {"at": now(), "study_id": STUDY,
         "status": "OFFICIAL_CODED_FIXING_SERIES_CLOCK_ASSUMPTION_REQUIRED",
         "rows": len(frame), "first_date": frame.date.min(), "last_date": frame.date.max(),
         "missing_market_dates": missing.dt.strftime("%Y-%m-%d").tolist(),
         "cross_query_matching_rows": len(check), "old_curve_comparison_rows": len(overlap),
         "rate_min_percent": float(frame.gc007_percent.min()), "rate_max_percent": float(frame.gc007_percent.max()),
         "failed_windows": failures, "methodology_excerpts": len(excerpts),
         "historical_first_publication_verified": False, "current_market_view": "NO_VIEW",
         "new_strategy_returns": 0, "new_accounts": 0, "goal_achieved": False}, True)
    print(f"取得{len(frame)}条带代码GC007定盘，重叠核对{len(check)}条；缺失A股交易日{len(missing)}，历史首版时钟未认证。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GC007官方定盘有限历史来源")
    parser.add_argument("command", choices=["initial", "build", "finish"])
    args = parser.parse_args()
    globals()[args.command]()
