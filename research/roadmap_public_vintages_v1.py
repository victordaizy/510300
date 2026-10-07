"""免费公开源的有限版本留存；只记录收到什么，不生成交易观点。"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from research.roadmap_execution_v1 import ROOT, CONFIG, write_json


def clock() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def source_requests(stat_date: str) -> list[dict]:
    compact = stat_date.replace("-", "")
    return [
        {"source_id": "SSE_MARGIN", "url": "https://query.sse.com.cn/marketdata/tradedata/queryMargin.do",
         "referer": "https://www.sse.com.cn/", "params": {"isPagination": "true", "beginDate": compact,
          "endDate": compact, "tabType": "", "stockCode": "", "pageHelp.pageSize": "20", "pageHelp.pageNo": "1",
          "pageHelp.beginPage": "1", "pageHelp.cacheSize": "1", "pageHelp.endPage": "1"}},
        {"source_id": "SZSE_MARGIN", "url": "https://www.szse.cn/api/report/ShowReport/data",
         "referer": "https://www.szse.cn/disclosure/margin/margin/index.html",
         "params": {"SHOWTYPE": "JSON", "CATALOGID": "1837_xxpl", "txtDate": stat_date, "tab1PAGENO": "1"}},
        {"source_id": "SSE_ETF_SHARES", "url": "https://query.sse.com.cn/commonQuery.do",
         "referer": "https://www.sse.com.cn/", "params": {"isPagination": "true", "pageHelp.pageSize": "10000",
          "pageHelp.pageNo": "1", "pageHelp.beginPage": "1", "pageHelp.cacheSize": "1", "pageHelp.endPage": "1",
          "sqlId": "COMMON_SSE_ZQPZ_ETFZL_XXPL_ETFGM_SEARCH_L", "STAT_DATE": stat_date}},
    ]


def payload_rows(value: object) -> list[dict]:
    if isinstance(value, dict) and isinstance(value.get("result"), list):
        return [row for row in value["result"] if isinstance(row, dict)]
    if isinstance(value, list) and value and isinstance(value[0], dict) and isinstance(value[0].get("data"), list):
        return [row for row in value[0]["data"] if isinstance(row, dict)]
    return []


def collect_source(source: dict, destination: Path, cfg: dict, request=requests.get) -> dict:
    started = clock()
    raw_path = destination / (source["source_id"] + ".response.bin")
    result = {"source_id": source["source_id"], "url": source["url"], "params": source["params"],
              "request_started_at": started, "received_at": None, "status": "REQUEST_FAILED", "rows": 0,
              "historical_first_publication_proven": False, "prediction_generated": False}
    raw = b""
    try:
        with request(source["url"], params=source["params"], headers={"Referer": source["referer"],
                     "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
                     timeout=cfg["request_timeout_seconds"], stream=True, verify=True) as response:
            result["http_status"] = response.status_code
            for chunk in response.iter_content(65536):
                if len(raw) + len(chunk) > cfg["max_bytes_per_response"]:
                    result["status"] = "RESPONSE_SIZE_LIMIT"
                    raise ValueError("公开源响应超过单次空间上限")
                raw += chunk
            result["received_at"] = clock()
            raw_path.write_bytes(raw)
            result.update(raw_path=raw_path.name, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
            response.raise_for_status()
            value = json.loads(raw.decode("utf-8-sig"))
            rows = payload_rows(value)
            result.update(status="PUBLIC_ROWS_RECEIVED" if rows else "NO_ROWS_OR_UNRECOGNIZED_SCHEMA", rows=len(rows),
                          observed_fields=sorted({key for row in rows for key in row}),
                          economic_date="REQUESTED_DATE_ONLY_NOT_YET_FIELD_ADMITTED")
            if rows:
                write_json(destination / (source["source_id"] + ".observed_rows.json"), rows)
    except Exception as exc:
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)[:1200]
        result["finished_at"] = clock()
        if raw and not raw_path.exists():
            raw_path.write_bytes(raw)
            result.update(raw_path=raw_path.name, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(), raw_is_truncated=True)
    write_json(destination / (source["source_id"] + ".receipt.json"), result)
    return result


def collect() -> dict:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    settings = cfg["raw_vintage_collector"]
    output = ROOT / cfg["output"]
    stamp = datetime.now(ZoneInfo("Asia/Shanghai"))
    observation_day = stamp.date().isoformat()
    root = output / "raw_vintages"
    root.mkdir(parents=True, exist_ok=True)
    total = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    if total + settings["max_requests_per_invocation"] * settings["max_bytes_per_response"] > settings["max_total_storage_bytes"]:
        result = {"status": "NO_COLLECTION_STORAGE_LIMIT", "recorded_at": clock(), "bytes": total}
        write_json(output / "collector_current_status.json", result)
        return result
    calendar_path = ROOT / "data/reference/sse_trade_calendar_2026.csv"
    calendar_meta = json.loads((ROOT / "data/reference/sse_trade_calendar_2026.metadata.json").read_text(encoding="utf-8"))
    if hashlib.sha256(calendar_path.read_bytes()).hexdigest() != calendar_meta["sha256"]:
        raise ValueError("已核对的官方交易日历身份变化")
    calendar = pd.read_csv(calendar_path)
    dates = pd.to_datetime(calendar["trade_date"])
    if "is_open" in calendar:
        dates = dates.loc[calendar.is_open.astype(int).eq(1)]
    if dates.max().date() < stamp.date():
        result = {"status": "NO_COLLECTION_CALENDAR_EXPIRED", "recorded_at": clock()}
        write_json(output / "collector_current_status.json", result)
        return result
    historical = dates.loc[dates.dt.date < stamp.date()]
    if historical.empty:
        raise ValueError("没有上一交易日，不能猜统计日期")
    stat_date = historical.max().date().isoformat()
    destination = root / observation_day
    destination.mkdir(exist_ok=True)
    claim = destination / "run_claim.json"
    try:
        with claim.open("x", encoding="utf-8") as stream:
            json.dump({"started_at": clock(), "requested_stat_date": stat_date}, stream, ensure_ascii=False)
    except FileExistsError:
        return {"status": "ALREADY_RECORDED_THIS_OBSERVATION_DAY", "observation_day": observation_day}
    source_specs = source_requests(stat_date)
    if len(source_specs) > settings["max_requests_per_invocation"]:
        raise ValueError("请求超过登记范围")
    results = [collect_source(source, destination, settings) for source in source_specs]
    result = {"status": "PUBLIC_VINTAGE_OBSERVATION_RECORDED", "completed_at": clock(), "observation_day": observation_day,
        "requested_stat_date": stat_date, "requests": len(results), "sources_with_rows": sum(r["rows"] > 0 for r in results),
        "raw_bytes": sum(r.get("bytes", 0) for r in results), "sources": results,
        "retrospective_retrieval_is_not_first_publication": True, "new_admitted_historical_days": 0,
        "new_forward_strategy_observations": 0, "new_model_fits": 0, "new_orders": 0, "fees_cny": 0}
    write_json(destination / "summary.json", result)
    write_json(output / "collector_current_status.json", result)
    return result


def main() -> None:
    argparse.ArgumentParser(description="每日一次记录两市融资及沪市ETF公开响应，不形成信号").parse_args()
    result = collect()
    print(json.dumps({key: result.get(key) for key in ["status", "requests", "sources_with_rows", "raw_bytes", "new_forward_strategy_observations"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
