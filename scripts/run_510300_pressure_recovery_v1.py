"""保存本轮输入、核查来源能力，并从真实本地档案生成四表；不下载行情或下单。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.pressure_recovery_v1 import commission, expectancy, next_exit_request
from scripts.analyze_510300_close_concession_sources_v1 import analyze, snapshot, verify_source_receipts

CN = timezone(timedelta(hours=8))
OLD = ROOT / "reports/research/510300_close_concession_source_feasibility_v1"
DEFAULT = ROOT / "reports/research/510300_pressure_recovery_v1"
INPUTS = [
    "data/raw/market/510300_intraday_tencent_snapshots.parquet",
    "data/raw/primary_market/510300_iopv_snapshots.parquet",
    "data/raw/primary_market_v1_2/510300_iopv_snapshots_v1_2.parquet",
]
URLS = {
    "sse_rules": "https://www.sse.com.cn/lawandrules/sselawsrules2025/trade/universal/c/c_20260424_10816492.shtml",
    "sse_calendar": "https://www.sse.com.cn/disclosure/dealinstruc/closed/c/c_20251222_10802510.shtml",
    "sse_level1": "https://www.sseinfo.com/services/assortment/level1/",
    "sse_history_spec": "https://bsp.sseinfo.com/admin/static/public/2024-06-07/0017f8fa1c1a47cb997d433c6d05fc11/上海证券交易所历史数据接口说明书.pdf",
    "sse_md102_spec": "https://www.sse.com.cn/services/tradingtech/development/c/10817287/files/b0b6c624e6ce48e3a05dc75ccf189920.pdf",
    "sse_t1": "https://www.sse.com.cn/assortment/fund/etf/question/c/c_20240118_5734755.shtml",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_write(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def clean(value):
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value


def csv_write(path: Path, rows: list[dict], columns: list[str] | None = None) -> None:
    fields = columns or list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def probe_documents(out: Path) -> list[dict]:
    destination = out / "official_sources"
    destination.mkdir(exist_ok=True)
    json_write(destination / "request_manifest.json", {"saved_before_requests": datetime.now(CN).isoformat(), "urls": URLS})

    def fetch(item):
        identity, url = item
        started = datetime.now(CN).isoformat()
        receipt = {"id": identity, "url": url, "request_started_at": started, "market_observation": False}
        try:
            response = requests.get(url, timeout=(8, 20), headers={"User-Agent": "Mozilla/5.0"})
            receipt.update(received_at=datetime.now(CN).isoformat(), http_status=response.status_code)
            suffix = ".pdf" if response.content.startswith(b"%PDF") else ".html"
            path = destination / (identity + suffix)
            path.write_bytes(response.content)
            receipt.update(path=str(path.relative_to(out)), sha256=sha(path), bytes=path.stat().st_size,
                           status="DOCUMENT_RECEIVED" if response.ok else "HTTP_ERROR_BODY_RETAINED")
            if suffix == ".pdf":
                import pdfplumber
                with pdfplumber.open(path) as pdf:
                    texts = [f"PDF物理页 {i+1}\n{page.extract_text() or ''}" for i, page in enumerate(pdf.pages)]
                    receipt["pdf_pages"] = len(texts)
                    (destination / (identity + ".txt")).write_text("\n\n".join(texts), encoding="utf-8")
        except Exception as exc:
            receipt.update(status="SOURCE_REQUEST_OR_PARSE_FAILED", error=f"{type(exc).__name__}: {exc}", received_at=datetime.now(CN).isoformat())
        return receipt

    with ThreadPoolExecutor(max_workers=6) as pool:
        receipts = list(pool.map(fetch, URLS.items()))
    json_write(destination / "receipts.json", receipts)
    return receipts


def normalized_archive(out: Path) -> tuple[list[dict], list[dict]]:
    observations, inventory = [], []
    for relative in INPUTS:
        path = ROOT / relative
        frame = pd.read_parquet(path)
        identity = sha(path)
        saved = out / "input_snapshot" / relative
        saved.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, saved)
        inventory.append({"path": relative, "sha256": identity, "rows": len(frame), "columns": list(frame.columns),
                          "date_count": int(frame.trade_date.nunique()), "first_date": str(frame.trade_date.min()), "last_date": str(frame.trade_date.max()),
                          "orderbook_available": False, "reference_economic_time_available": False,
                          "after_hours_pending_qty_available": False, "actual_order_receipts_available": False})
        for i, raw in enumerate(frame.to_dict("records")):
            r = {k: clean(v) for k, v in raw.items()}
            day = str(r["trade_date"])[:10]
            is_minute = "timestamp" in r
            time_raw = r.get("timestamp", r.get("exchange_timestamp"))
            received = r.get("retrieved_at")
            flags = ["NO_SYNCHRONIZED_REFERENCE", "NO_VALUATION_ERROR_BOUND", "NO_ORDERBOOK", "NO_AFTER_HOURS_QUEUE"]
            if is_minute:
                flags.append("MINUTE_PRICE_IS_NOT_EXECUTABLE_ASK")
            else:
                flags.append("ENVELOPE_IS_NOT_PRICE_OR_IOPV_ECONOMIC_TIME")
            if received and str(received)[:19] > day + "T15:06:00":
                flags.append("RECEIVED_AFTER_M1_DEADLINE")
            observations.append({
                "observation_id": f"{path.stem}_{i:06d}", "trade_date": day,
                "symbol": "510300.SH", "source_timestamp_raw": time_raw,
                "price_economic_at": None, "reference_economic_at": None,
                "received_at": received, "price_observed": r.get("price", r.get("last_price")),
                "iopv_observed_unqualified": r.get("iopv"), "reference_low": None, "reference_high": None,
                "bid1": None, "ask1": None, "spread_bps": None, "bid_depth_cny": None,
                "obi": None, "aggressor_trade_imbalance": None,
                "cumulative_volume_raw": r.get("cumulative_volume", r.get("volume_shares")),
                "source_volume_unit": r.get("volume_unit", "share"),
                "post_close_volume_shares": r.get("post_close_volume_shares"),
                "post_close_pending_buy_qty": None, "post_close_pending_sell_qty": None,
                "market_phase_raw": r.get("session_phase", r.get("trade_phase_raw")),
                "quality_status": "NO_VIEW", "quality_reasons": ";".join(flags),
                "raw_source_path": relative, "raw_source_sha256": identity,
            })
    # 只读取分钟档案的模式与日期覆盖，不读取其收益，不扩展旧用途豁免。
    minute = ROOT / "data/raw/market/510300_1m_tushare_raw.parquet"
    minute_metadata = json.loads(minute.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    inventory.append({"path": str(minute.relative_to(ROOT)), "rows": pq.read_metadata(minute).num_rows,
                      "columns": pq.read_schema(minute).names,
                      "first_date": minute_metadata["actual_first_trade_time"], "last_date": minute_metadata["actual_last_trade_time"],
                      "price_values_read": False, "admission": "NO_BOOK_NO_IOPV_NO_AFTER_HOURS_QUEUE"})
    return observations, inventory


def build(out: Path, fetch_documents: bool) -> dict:
    if (out / "status.json").exists():
        raise ValueError("已有本轮状态，复算请指定新的输出目录，保留原结果")
    out.mkdir(parents=True, exist_ok=True)
    inputs = out / "input_snapshot"
    inputs.mkdir(exist_ok=True)
    for relative in ("config/510300_pressure_recovery_v1.json", "research/pressure_recovery_v1.py", "scripts/run_510300_pressure_recovery_v1.py", "config/510300_research_authority_v6.json", "data/reference/sse_trade_calendar_2026.csv"):
        target = inputs / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    user_text = Path(r"E:\CodexData\.codex\attachments\37d8f6c5-c201-444d-b02b-23de0a99ab57\pasted-text-1.txt")
    shutil.copy2(user_text, inputs / "用户任务原文.txt")
    config = json.loads((ROOT / "config/510300_pressure_recovery_v1.json").read_text(encoding="utf-8"))
    json_write(out / "registration_receipt.json", {
        "registered_before_evaluation_at": datetime.now(CN).isoformat(),
        "config_sha256": sha(ROOT / "config/510300_pressure_recovery_v1.json"),
        "engine_sha256": sha(ROOT / "research/pressure_recovery_v1.py"),
        "prior_return_views_this_study": 0, "protocol_is_seed_not_optimum": True,
        "account_parameters_are_research_assumptions": True,
    })
    old_result, _, _ = analyze(OLD)
    # 旧分析器仅离线解析，原目录不写入。
    json_write(out / "legacy_recomputed.json", json.loads(json.dumps(old_result, default=str)))
    _, receipts = verify_source_receipts(OLD)
    observations, inventory = normalized_archive(out)
    for identity in ("sse_snapshot_existing_fields", "sse_snapshot_second_lunch"):
        receipt = receipts[identity]
        source = OLD / receipt["raw_path"]
        r = snapshot(source.read_bytes())
        target = inputs / "legacy_raw" / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        observations.append({"observation_id": identity, "trade_date": "2026-09-07", "symbol": "510300.SH",
                             "source_timestamp_raw": str(r["source_envelope_time"]),
                             "received_at": receipt["received_at"], "price_observed": float(r["last"]),
                             "iopv_observed_unqualified": float(r["iopv"]),
                             "post_close_volume_shares": r["fp_volume"], "market_phase_raw": r["tradephase"],
                             "quality_status": "NO_VIEW", "quality_reasons": "LUNCH_SAMPLE;UNCALIBRATED_CLOCK;NO_SYNCHRONIZED_REFERENCE;NO_AFTER_HOURS_QUEUE",
                             "raw_source_path": str(source.relative_to(ROOT)), "raw_source_sha256": sha(source)})
    calendar = pd.read_csv(ROOT / "data/reference/sse_trade_calendar_2026.csv").trade_date.tolist()
    dates = sorted({r["trade_date"] for r in observations})
    coverage = []
    for day in dates:
        rows = [r for r in observations if r["trade_date"] == day]
        coverage.append({"trade_date": day, "raw_rows": len(rows), "qualified_m1_rows": 0, "qualified_m2_rows": 0,
                         "m1_status": "NO_VIEW_NO_SYNCHRONIZED_CLOSE_OR_QUEUE", "m2_status": "NO_VIEW_NO_SPREAD_DEPTH_BASELINE",
                         "expected_exit_request_at": next_exit_request(day, calendar),
                         "is_event": False, "missing_day_is_no_event": False})
    event_columns = ["event_id", "mechanism", "trade_date", "shock_at", "observation_end_at", "recovery_state", "opposing_evidence", "source_observation_ids", "entry_eligible"]
    order_columns = ["order_id", "event_id", "mechanism", "evidence_type", "side", "requested_at", "limit_price", "requested_quantity", "confirmed_at", "filled_quantity", "fill_vwap", "cancel_requested_at", "cancel_confirmed_at", "state", "evidence_path"]
    results = []
    for variant in (*config["m1"]["variants"], config["m2"]["variant"]):
        for account in (config["account_cny"], config["sensitivity_account_cny"]):
            results.append({"variant": variant, "account_cny": account, "status": "NOT_COMPUTED_NO_ADMISSIBLE_EVENTS_OR_FILLS",
                            "qualifying_event_count": 0, "submitted_orders_this_study": 0, "confirmed_filled_trade_count": 0,
                            "fill_rate": None, "net_expectancy": None, "win_rate": None, "payoff_ratio": None,
                            "tail_loss": None, "capital_occupation": None, "net_sharpe": None, "max_drawdown": None,
                            "full_account_ledger": "NOT_RUN", "actual_broker_commission": "UNKNOWN",
                            "reason": "同步估值、价差深度、盘后队列和成交证据未齐"})
    costs = []
    for amount in config["cost_scenarios"]["nominal_sizes_cny"]:
        fees = 2 * commission(amount)
        costs.append({"notional_cny": amount, "commission_roundtrip_cny": fees,
                      "commission_roundtrip_bps": fees / amount * 1e4,
                      "illustrative_two_sided_friction_bps": 10,
                      "illustrative_total_bps": fees / amount * 1e4 + 10,
                      "status": "ASSUMPTION_NOT_MEASURED_COST"})
    csv_write(out / "01_同步行情表.csv", observations)
    csv_write(out / "02_事件表.csv", [], event_columns)
    csv_write(out / "03_订单表.csv", [], order_columns)
    csv_write(out / "04_结果表.csv", results)
    csv_write(out / "每日覆盖与缺失.csv", coverage)
    csv_write(out / "费用情景.csv", costs)
    json_write(out / "source_inventory.json", inventory)
    source_receipts = probe_documents(out) if fetch_documents else []
    status = {
        "study_id": config["study_id"], "generated_at": datetime.now(CN).isoformat(),
        "status": "IMPLEMENTED_MEASUREMENT_LOCAL_DATA_NOT_ADMITTED",
        "goal_achieved": False, "net_positive_expectancy_established": False,
        "raw_observations": len(observations), "distinct_raw_days": len(dates), "raw_dates": dates,
        "qualified_close_observations": 0, "qualified_m2_events": 0,
        "event_table_rows": 0, "order_table_rows": 0, "result_status_rows": len(results),
        "known_no_event_days": 0, "unknown_event_days": len(dates),
        "confirmed_fills": 0, "fill_probability": None,
        "net_expectancy": None, "net_sharpe": None, "full_account_status": "NOT_RUN_NO_ADMISSIBLE_TRADES",
        "legacy_pcf_fields_match": old_result["pcf"]["mismatch_count"] == 0,
        "required_component_prices_in_legacy_sample": old_result["pcf"]["valuation_input_recipe"]["required_positive_quantity_price_count"],
        "broker_connections": 0, "orders_submitted": 0, "automatic_collection_started": False,
        "position_impact": 0, "model_action": "ABSTAIN", "model_position_target": "UNSET",
        "new_live_market_samples": 0, "official_document_requests": len(source_receipts),
        "historical_tick_data_downloaded": False, "security_audit_performed": False,
        "independent_validation_performed": False, "external_review_performed": False,
        "zero_events_means": "not identifiable from retained sources; not evidence of no market events",
        "research_mode": "historical_evidence_and_protocol_implementation_only",
    }
    json_write(out / "status.json", status)
    json_write(out / "workbook_data.json", {"status": status, "coverage": coverage, "observations": observations,
                                            "results": results, "costs": costs,
                                            "event_columns": event_columns, "order_columns": order_columns,
                                            "sources": URLS})
    entries = [{"path": str(p.relative_to(out)), "bytes": p.stat().st_size, "sha256": sha(p)}
               for p in sorted(out.rglob("*")) if p.is_file() and p.name != "FILE_INDEX.csv"]
    csv_write(out / "FILE_INDEX.csv", entries)
    return status


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行510300压力恢复四表测量与历史来源核查")
    parser.add_argument("--output", type=Path, default=DEFAULT)
    parser.add_argument("--fetch-official-documents", action="store_true")
    arguments = parser.parse_args()
    print(json.dumps(build(arguments.output.resolve(), arguments.fetch_official_documents), ensure_ascii=False, indent=2))
