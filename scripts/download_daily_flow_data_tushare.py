"""下载510300日频份额、融资融券与北向资金历史。"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import numpy as np
import tushare as ts
import yaml
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
SETTINGS_FILE = ROOT / "config" / "settings.yaml"
OUTPUT_DIR = ROOT / "data" / "raw" / "flow"
FUND_SHARE_FILE = OUTPUT_DIR / "510300_fund_share_daily_tushare.parquet"
MARGIN_FILE = OUTPUT_DIR / "510300_margin_detail_daily_tushare.parquet"
NORTHBOUND_FILE = OUTPUT_DIR / "northbound_money_daily_tushare.parquet"
REPORT_FILE = ROOT / "reports" / "data_quality" / "daily_flow_data_tushare_status.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_token() -> str:
    load_dotenv(ROOT / ".env")
    token = os.getenv("TUSHARE_TOKEN") or os.getenv("TS_TOKEN")
    if not token:
        raise RuntimeError("未设置TUSHARE_TOKEN或TS_TOKEN")
    return token


def normalize_fund_share(data: pd.DataFrame, retrieved_at: datetime) -> pd.DataFrame:
    required = {"ts_code", "trade_date", "fd_share"}
    if missing := required - set(data.columns):
        raise ValueError(f"fund_share缺少字段：{sorted(missing)}")
    result = data.copy()
    result["date"] = pd.to_datetime(result["trade_date"], errors="coerce")
    result["fund_share_10k"] = pd.to_numeric(result["fd_share"], errors="coerce")
    result["fund_shares"] = result["fund_share_10k"] * 10000.0
    result = result.dropna(subset=["date", "fund_shares"])
    result = result.sort_values(["date", "ts_code"]).drop_duplicates("date", keep="last")
    result["source"] = "tushare.fund_share"
    result["retrieved_at"] = retrieved_at
    return result[
        ["date", "ts_code", "fund_share_10k", "fund_shares", "fund_type", "market", "source", "retrieved_at"]
    ].reset_index(drop=True)


def normalize_margin(data: pd.DataFrame, retrieved_at: datetime) -> pd.DataFrame:
    required = {"trade_date", "ts_code", "rzye", "rqye", "rzmre", "rzche", "rzrqye"}
    if missing := required - set(data.columns):
        raise ValueError(f"margin_detail缺少字段：{sorted(missing)}")
    result = data.copy()
    result["date"] = pd.to_datetime(result["trade_date"], errors="coerce")
    numeric = ["rzye", "rqye", "rzmre", "rqyl", "rzche", "rqchl", "rqmcl", "rzrqye"]
    result[numeric] = result[numeric].apply(pd.to_numeric, errors="coerce")
    result = result.dropna(subset=["date", "rzye", "rzmre", "rzche"])
    result = result.sort_values("date").drop_duplicates("date", keep="last")
    result["financing_net_buy_cny"] = result["rzmre"] - result["rzche"]
    result["source"] = "tushare.margin_detail"
    result["retrieved_at"] = retrieved_at
    return result[
        ["date", "ts_code", *numeric, "financing_net_buy_cny", "source", "retrieved_at"]
    ].reset_index(drop=True)


def normalize_northbound(data: pd.DataFrame, retrieved_at: datetime) -> pd.DataFrame:
    required = {"trade_date", "hgt", "sgt", "north_money"}
    if missing := required - set(data.columns):
        raise ValueError(f"moneyflow_hsgt缺少字段：{sorted(missing)}")
    result = data.copy()
    result["date"] = pd.to_datetime(result["trade_date"], errors="coerce")
    numeric = ["ggt_ss", "ggt_sz", "hgt", "sgt", "north_money", "south_money"]
    result[numeric] = result[numeric].apply(pd.to_numeric, errors="coerce")
    result = result.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last")
    disclosure_change = pd.Timestamp("2024-08-19")
    result["northbound_semantics"] = np.where(
        result["date"].lt(disclosure_change),
        "LEGACY_DAILY_NET_FLOW",
        "NONCOMPARABLE_AFTER_2024_08_19_DISCLOSURE_CHANGE",
    )
    result["usable_as_net_flow"] = result["date"].lt(disclosure_change)
    result["source"] = "tushare.moneyflow_hsgt"
    result["retrieved_at"] = retrieved_at
    return result[
        ["date", *numeric, "northbound_semantics", "usable_as_net_flow", "source", "retrieved_at"]
    ].reset_index(drop=True)


def _year_chunks(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[str, str]]:
    chunks: list[tuple[str, str]] = []
    cursor = start
    while cursor <= end:
        chunk_end = min(pd.Timestamp(cursor.year, 12, 31), end)
        chunks.append((cursor.strftime("%Y%m%d"), chunk_end.strftime("%Y%m%d")))
        cursor = chunk_end + pd.Timedelta(days=1)
    return chunks


def main() -> int:
    token = ""
    try:
        token = get_token()
        settings = yaml.safe_load(SETTINGS_FILE.read_text(encoding="utf-8"))
        start = pd.Timestamp(settings["project"]["start_date"])
        end = pd.Timestamp(settings["project"]["end_date"])
        retrieved_at = datetime.now(ZoneInfo(settings["project"]["timezone"]))
        api_url = os.getenv("TUSHARE_API_URL", "https://api.tushare.pro").strip()
        interval = max(float(os.getenv("TUSHARE_REQUEST_INTERVAL_SECONDS", "0.8")), 0.5)
        pro = ts.pro_api(token)
        pro._DataApi__http_url = api_url
        fund_share_raw = pro.query(
            "fund_share", ts_code="510300.SH", start_date=start.strftime("%Y%m%d"), end_date=end.strftime("%Y%m%d")
        )
        time.sleep(interval)
        margin_raw = pro.query(
            "margin_detail", ts_code="510300.SH", start_date=start.strftime("%Y%m%d"), end_date=end.strftime("%Y%m%d")
        )
        northbound_parts: list[pd.DataFrame] = []
        for chunk_start, chunk_end in _year_chunks(start, end):
            time.sleep(interval)
            part = pro.query("moneyflow_hsgt", start_date=chunk_start, end_date=chunk_end)
            if part is not None and not part.empty:
                northbound_parts.append(part)
        if not northbound_parts:
            raise ValueError("moneyflow_hsgt没有返回任何数据")
        fund_share = normalize_fund_share(fund_share_raw, retrieved_at)
        margin = normalize_margin(margin_raw, retrieved_at)
        northbound = normalize_northbound(pd.concat(northbound_parts, ignore_index=True), retrieved_at)
        expected_trading_days = pd.read_parquet(
            ROOT / "data" / "raw" / "market" / "510300_daily_raw.parquet",
            columns=["date"],
        )
        expected_trading_days["date"] = pd.to_datetime(expected_trading_days["date"])
        expected = expected_trading_days.loc[
            expected_trading_days["date"].between(start, end), "date"
        ].nunique()
        if fund_share["date"].nunique() < expected * 0.95:
            raise ValueError("510300日频基金份额覆盖率不足95%")
        if margin["date"].nunique() < expected * 0.95:
            raise ValueError("510300融资融券覆盖率不足95%")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fund_share.to_parquet(FUND_SHARE_FILE, index=False)
        margin.to_parquet(MARGIN_FILE, index=False)
        northbound.to_parquet(NORTHBOUND_FILE, index=False)
        northbound_nonzero = northbound.loc[northbound["north_money"].fillna(0).ne(0), "date"]
        legacy_net_flow = northbound.loc[northbound["usable_as_net_flow"], "date"]
        report = {
            "status": "PASS",
            "checked_at": retrieved_at.isoformat(),
            "api_host": api_url.split("//", 1)[-1].split("/", 1)[0],
            "credential_note": "报告不保存或回显Token。",
            "expected_trading_day_count": int(expected),
            "datasets": {
                "fund_share": {
                    "rows": int(len(fund_share)),
                    "first_date": str(fund_share["date"].min().date()),
                    "last_date": str(fund_share["date"].max().date()),
                    "trading_day_coverage": float(fund_share["date"].nunique() / expected),
                    "file": FUND_SHARE_FILE.relative_to(ROOT).as_posix(),
                    "sha256": _sha256(FUND_SHARE_FILE),
                },
                "margin_detail": {
                    "rows": int(len(margin)),
                    "first_date": str(margin["date"].min().date()),
                    "last_date": str(margin["date"].max().date()),
                    "trading_day_coverage": float(margin["date"].nunique() / expected),
                    "file": MARGIN_FILE.relative_to(ROOT).as_posix(),
                    "sha256": _sha256(MARGIN_FILE),
                },
                "northbound": {
                    "rows": int(len(northbound)),
                    "first_date": str(northbound["date"].min().date()),
                    "last_date": str(northbound["date"].max().date()),
                    "last_nonzero_north_money_date": (
                        str(northbound_nonzero.max().date()) if not northbound_nonzero.empty else None
                    ),
                    "last_date_usable_as_net_flow": (
                        str(legacy_net_flow.max().date()) if not legacy_net_flow.empty else None
                    ),
                    "semantic_warning": "2024-08-19起互联互通披露机制变化，接口数值不再与此前日净流入同口径，禁止拼接为连续净流量因子。",
                    "file": NORTHBOUND_FILE.relative_to(ROOT).as_posix(),
                    "sha256": _sha256(NORTHBOUND_FILE),
                },
            },
        }
        REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
        REPORT_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except Exception as error:
        message = str(error).replace(token, "[REDACTED]") if token else str(error)
        print(f"日频资金流下载失败：{type(error).__name__}: {message}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
