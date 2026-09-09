"""下载上交所官方510300研究期起点与月末基金份额截面。"""

from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = PROJECT_ROOT / "config" / "settings.yaml"
DAILY_FILE = PROJECT_ROOT / "data" / "raw" / "market" / "510300_daily_raw.parquet"
OUTPUT_FILE = PROJECT_ROOT / "data" / "raw" / "fund" / "510300_monthly_shares_sse.parquet"
METADATA_FILE = OUTPUT_FILE.with_suffix(".metadata.json")
REPORT_FILE = PROJECT_ROOT / "reports" / "data_quality" / "510300_monthly_shares_sse_quality.json"
URL = "https://query.sse.com.cn/commonQuery.do"
SQL_ID = "COMMON_SSE_ZQPZ_ETFZL_XXPL_ETFGM_SEARCH_L"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_parquet(data: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    data.to_parquet(temporary, index=False, engine="pyarrow")
    temporary.replace(path)


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def fetch_date(date: pd.Timestamp, maximum_retries: int = 4) -> dict:
    params = {
        "isPagination": "true",
        "pageHelp.pageSize": "10000",
        "pageHelp.pageNo": "1",
        "pageHelp.beginPage": "1",
        "pageHelp.cacheSize": "1",
        "pageHelp.endPage": "1",
        "sqlId": SQL_ID,
        "STAT_DATE": date.strftime("%Y-%m-%d"),
    }
    headers = {"Referer": "https://www.sse.com.cn/", "User-Agent": "Mozilla/5.0"}
    final_error: Exception | None = None
    for attempt in range(1, maximum_retries + 1):
        try:
            response = requests.get(URL, params=params, headers=headers, timeout=45)
            response.raise_for_status()
            payload = response.json()
            rows = [row for row in payload.get("result", []) if row.get("SEC_CODE") == "510300"]
            if len(rows) != 1:
                raise RuntimeError(f"目标记录数不是1：{len(rows)}")
            row = rows[0]
            return {
                "date": pd.to_datetime(row["STAT_DATE"], errors="raise"),
                "symbol": "510300.SH",
                "fund_name": row.get("SEC_NAME"),
                "share_10k_provider": float(row["TOT_VOL"]),
                "shares": float(row["TOT_VOL"]) * 10_000.0,
                "source_result_row_count": len(payload.get("result", [])),
            }
        except Exception as exc:
            final_error = exc
            if attempt < maximum_retries:
                time.sleep(attempt)
    raise RuntimeError(f"{date.date()}上交所ETF份额请求失败：{final_error}")


def main() -> int:
    config = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    timezone = ZoneInfo(config["project"]["timezone"])
    retrieved_at = datetime.now(timezone).isoformat()
    daily = pd.read_parquet(DAILY_FILE)
    daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
    dates = daily.groupby(daily["date"].dt.to_period("M"))["date"].max().tolist()
    first_date = daily["date"].min()
    dates = sorted(set([first_date, *dates]))

    records: dict[pd.Timestamp, dict] = {}
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(fetch_date, date): date for date in dates}
        for future in as_completed(futures):
            date = futures[future]
            try:
                records[date] = future.result()
            except Exception as exc:
                failures.append(str(exc))
    if failures:
        raise RuntimeError("月末份额下载不完整：" + " | ".join(failures[:10]))

    output = pd.DataFrame([records[date] for date in dates]).sort_values("date").reset_index(drop=True)
    output["share_change"] = output["shares"].diff()
    output["share_change_pct"] = output["shares"].pct_change()
    output["sampling_role"] = "research_start_or_month_end_trading_day"
    output["source"] = "sse.COMMON_SSE_ZQPZ_ETFZL_XXPL_ETFGM_SEARCH_L"
    output["retrieved_at"] = retrieved_at
    errors: list[str] = []
    if len(output) != len(dates):
        errors.append("记录数与请求日期数不一致")
    if output["date"].duplicated().any():
        errors.append("存在重复日期")
    if (output["shares"] <= 0).any():
        errors.append("基金份额存在非正值")
    if not output["date"].is_monotonic_increasing:
        errors.append("日期未严格升序")
    if set(output["date"]) != set(dates):
        errors.append("请求日期与返回日期不一致")

    atomic_parquet(output, OUTPUT_FILE)
    checksum = sha256_file(OUTPUT_FILE)
    report = {
        "status": "PASS" if not errors else "FAIL",
        "checked_at": retrieved_at,
        "row_count": int(len(output)),
        "first_date": output["date"].min().date().isoformat(),
        "last_date": output["date"].max().date().isoformat(),
        "minimum_shares": float(output["shares"].min()),
        "maximum_shares": float(output["shares"].max()),
        "latest_shares": float(output.iloc[-1]["shares"]),
        "source_result_row_count_range": [
            int(output["source_result_row_count"].min()),
            int(output["source_result_row_count"].max()),
        ],
        "errors": errors,
        "limitations": [
            "仅保存研究期起点和每月最后交易日，不是日频份额序列。",
            "变化量只代表相邻快照间份额差，不是已对账的日频净申购或资金流。",
            "官方接口每次返回当日全部ETF，为避免对公共接口发起上千次大响应，未逐日抓取。",
        ],
    }
    atomic_json(report, REPORT_FILE)
    metadata = {
        "status": report["status"],
        "file": OUTPUT_FILE.relative_to(PROJECT_ROOT).as_posix(),
        "sha256": checksum,
        "row_count": int(len(output)),
        "sampling": "research_start_plus_month_end_trading_days",
        "unit": "share",
        "source": report.get("source", "sse.COMMON_SSE_ZQPZ_ETFZL_XXPL_ETFGM_SEARCH_L"),
        "retrieved_at": retrieved_at,
    }
    atomic_json(metadata, METADATA_FILE)
    print(json.dumps({"metadata": metadata, "quality": report}, ensure_ascii=False, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
