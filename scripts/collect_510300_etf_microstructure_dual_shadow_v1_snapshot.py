"""采集510300微观结构双影子V1的冻结后只读快照。

本程序不覆盖任何冻结历史输入。每次运行都写入独立时间戳目录，并保存
来源、日期覆盖、缺失日期和文件SHA-256，供固定规则留出/前瞻重放使用。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import akshare as ak
import numpy as np
import pandas as pd
import tushare as ts
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.download_510300_daily import fetch_from_sina
from scripts.download_510300_nav import fetch_eastmoney, fetch_sina
from scripts.download_daily_flow_data_tushare import (
    normalize_fund_share,
    normalize_margin,
)
from scripts.download_h00300_total_return import normalize_columns


PROJECT_ID = "510300_ETF_MICROSTRUCTURE_DUAL_SHADOW_V1"
TIMEZONE = ZoneInfo("Asia/Shanghai")
DEFAULT_START = pd.Timestamp("2026-08-13")
SNAPSHOT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "forward"
    / "510300_etf_microstructure_dual_shadow_v1"
    / "snapshots"
)
LOCAL_FALLBACKS = {
    "etf": PROJECT_ROOT / "data" / "raw" / "market" / "510300_daily_raw.parquet",
    "benchmark": (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "market"
        / "H00300_total_return_daily_raw.parquet"
    ),
    "fund_share": (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "flow"
        / "510300_fund_share_extension_20260818_v2.parquet"
    ),
    "margin": (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "flow"
        / "510300_margin_extension_20260818_v2.parquet"
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, engine="pyarrow")
    temporary.replace(path)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _normalize_dates(frame: pd.DataFrame, *, name: str) -> pd.DataFrame:
    if "date" not in frame.columns:
        raise ValueError(f"{name}缺少date字段")
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise").dt.normalize()
    result.sort_values("date", inplace=True)
    result.drop_duplicates("date", keep="last", inplace=True)
    result.reset_index(drop=True, inplace=True)
    return result


def _slice_dates(
    frame: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    name: str,
) -> pd.DataFrame:
    result = _normalize_dates(frame, name=name)
    result = result.loc[result["date"].between(start, end)].copy()
    result.reset_index(drop=True, inplace=True)
    return result


def _append_fallback(
    remote: pd.DataFrame,
    fallback_path: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    name: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    audit: dict[str, Any] = {
        "path": str(fallback_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "exists": fallback_path.exists(),
        "rows_used": 0,
        "sha256": _sha256(fallback_path) if fallback_path.exists() else None,
    }
    frames = [_slice_dates(remote, start, end, name=f"{name}_remote")]
    if fallback_path.exists():
        fallback = _slice_dates(
            pd.read_parquet(fallback_path), start, end, name=f"{name}_fallback"
        )
        if not fallback.empty:
            fallback = fallback.copy()
            if "source" not in fallback.columns:
                fallback["source"] = f"local_fallback:{fallback_path.name}"
            audit["rows_used"] = int(len(fallback))
            frames.insert(0, fallback)
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined.sort_values("date", inplace=True)
    combined.drop_duplicates("date", keep="last", inplace=True)
    combined.reset_index(drop=True, inplace=True)
    return combined, audit


def _get_tushare_client() -> tuple[Any, str]:
    load_dotenv(PROJECT_ROOT / ".env")
    token = os.getenv("TUSHARE_TOKEN") or os.getenv("TS_TOKEN")
    if not token:
        raise RuntimeError("未设置TUSHARE_TOKEN或TS_TOKEN")
    api_url = os.getenv("TUSHARE_API_URL", "https://api.tushare.pro").strip()
    client = ts.pro_api(token)
    client._DataApi__http_url = api_url
    api_host = api_url.split("//", 1)[-1].split("/", 1)[0]
    return client, api_host


def _fetch_flows(
    start: pd.Timestamp,
    end: pd.Timestamp,
    retrieved_at: datetime,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    remote_share = pd.DataFrame(columns=["date"])
    remote_margin = pd.DataFrame(columns=["date"])
    errors: list[str] = []
    api_host: str | None = None
    try:
        client, api_host = _get_tushare_client()
        start_text = start.strftime("%Y%m%d")
        end_text = end.strftime("%Y%m%d")
        share_raw = client.query(
            "fund_share",
            ts_code="510300.SH",
            start_date=start_text,
            end_date=end_text,
        )
        time.sleep(max(float(os.getenv("TUSHARE_REQUEST_INTERVAL_SECONDS", "0.8")), 0.5))
        margin_raw = client.query(
            "margin_detail",
            ts_code="510300.SH",
            start_date=start_text,
            end_date=end_text,
        )
        if share_raw is not None and not share_raw.empty:
            remote_share = normalize_fund_share(share_raw, retrieved_at)
        else:
            errors.append("tushare.fund_share返回空数据")
        if margin_raw is not None and not margin_raw.empty:
            remote_margin = normalize_margin(margin_raw, retrieved_at)
        else:
            errors.append("tushare.margin_detail返回空数据")
    except Exception as exc:
        errors.append(f"Tushare采集失败：{type(exc).__name__}: {exc}")

    shares, share_fallback = _append_fallback(
        remote_share,
        LOCAL_FALLBACKS["fund_share"],
        start,
        end,
        name="fund_share",
    )
    margin, margin_fallback = _append_fallback(
        remote_margin,
        LOCAL_FALLBACKS["margin"],
        start,
        end,
        name="margin",
    )
    audit = {
        "api_host": api_host,
        "credential_note": "未保存或回显Token。",
        "errors": errors,
        "fund_share_remote_rows": int(len(remote_share)),
        "margin_remote_rows": int(len(remote_margin)),
        "fund_share_fallback": share_fallback,
        "margin_fallback": margin_fallback,
    }
    return shares, margin, audit


def _fetch_nav(
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    start_text = start.date().isoformat()
    end_text = end.date().isoformat()
    errors: list[str] = []
    eastmoney = pd.DataFrame(columns=["date", "nav_eastmoney", "acc_nav_eastmoney"])
    sina = pd.DataFrame(columns=["date", "nav_sina", "acc_nav_sina"])
    try:
        eastmoney = fetch_eastmoney(start_text, end_text)
    except Exception as exc:
        errors.append(f"东方财富净值失败：{type(exc).__name__}: {exc}")
    try:
        sina = fetch_sina(start_text, end_text)
    except Exception as exc:
        errors.append(f"新浪净值失败：{type(exc).__name__}: {exc}")
    merged = eastmoney.merge(sina, on="date", how="outer", indicator=True)
    merged = _normalize_dates(merged, name="nav")
    matched = merged.loc[merged["_merge"].eq("both")].copy()
    matched["unit_nav"] = pd.to_numeric(matched["nav_eastmoney"], errors="coerce")
    matched["accumulated_nav"] = pd.to_numeric(
        matched["acc_nav_eastmoney"], errors="coerce"
    )
    unit_difference = (
        pd.to_numeric(matched["nav_eastmoney"], errors="coerce")
        - pd.to_numeric(matched["nav_sina"], errors="coerce")
    ).abs()
    accumulated_difference = (
        pd.to_numeric(matched["acc_nav_eastmoney"], errors="coerce")
        - pd.to_numeric(matched["acc_nav_sina"], errors="coerce")
    ).abs()
    if (unit_difference > 0.00005).any() or (accumulated_difference > 0.00005).any():
        raise ValueError("冻结后两来源净值差异超过四位小数精度容差")
    matched["source_primary"] = "eastmoney.f10.lsjz"
    matched["source_secondary"] = "sina.CaihuiFundInfoService.getNav"
    unmatched = merged.loc[~merged["_merge"].eq("both"), "date"]
    audit = {
        "errors": errors,
        "eastmoney_rows": int(len(eastmoney)),
        "sina_rows": int(len(sina)),
        "matched_rows": int(len(matched)),
        "unmatched_dates": [date.date().isoformat() for date in unmatched],
        "maximum_unit_nav_absolute_difference": (
            float(unit_difference.max()) if not unit_difference.empty else None
        ),
        "maximum_accumulated_nav_absolute_difference": (
            float(accumulated_difference.max())
            if not accumulated_difference.empty
            else None
        ),
    }
    return matched.reset_index(drop=True), audit


def _fetch_market(
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    errors: list[str] = []
    remote_etf = pd.DataFrame(columns=["date"])
    remote_benchmark = pd.DataFrame(columns=["date"])
    try:
        remote_etf = fetch_from_sina(
            "510300.SH", start.date().isoformat(), end.date().isoformat()
        )
        remote_etf["source"] = "akshare.fund_etf_hist_sina"
    except Exception as exc:
        errors.append(f"510300行情采集失败：{type(exc).__name__}: {exc}")
    try:
        raw_total = ak.stock_zh_index_hist_csindex(
            symbol="H00300",
            start_date=start.strftime("%Y%m%d"),
            end_date=end.strftime("%Y%m%d"),
        )
        remote_benchmark = normalize_columns(raw_total)
        remote_benchmark["source"] = "akshare.stock_zh_index_hist_csindex:H00300"
    except Exception as exc:
        errors.append(f"H00300行情采集失败：{type(exc).__name__}: {exc}")
    etf, etf_fallback = _append_fallback(
        remote_etf, LOCAL_FALLBACKS["etf"], start, end, name="etf"
    )
    benchmark, benchmark_fallback = _append_fallback(
        remote_benchmark,
        LOCAL_FALLBACKS["benchmark"],
        start,
        end,
        name="benchmark",
    )
    for name, frame, columns in [
        ("510300", etf, ["open", "high", "low", "close"]),
        ("H00300", benchmark, ["close"]),
    ]:
        missing = set(columns).difference(frame.columns)
        if missing:
            raise ValueError(f"{name}缺少行情字段：{sorted(missing)}")
        frame[columns] = frame[columns].apply(pd.to_numeric, errors="coerce")
        if frame[columns].isna().any(axis=None) or (frame[columns] <= 0).any(axis=None):
            raise ValueError(f"{name}行情存在空值或非正价格")
    audit = {
        "errors": errors,
        "etf_remote_rows": int(len(remote_etf)),
        "benchmark_remote_rows": int(len(remote_benchmark)),
        "etf_fallback": etf_fallback,
        "benchmark_fallback": benchmark_fallback,
    }
    return etf, benchmark, audit


def _dataset_summary(frame: pd.DataFrame, path: Path) -> dict[str, Any]:
    dates = pd.to_datetime(frame["date"], errors="coerce") if "date" in frame else pd.Series(dtype="datetime64[ns]")
    return {
        "file": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "rows": int(len(frame)),
        "first_date": dates.min().date().isoformat() if not frame.empty else None,
        "last_date": dates.max().date().isoformat() if not frame.empty else None,
        "columns": list(frame.columns),
        "sha256": _sha256(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="采集双影子V1冻结后数据快照")
    parser.add_argument("--start", default=DEFAULT_START.date().isoformat())
    parser.add_argument("--end", default=None, help="默认上海本地当天")
    parser.add_argument("--run-id", default=None, help="可选的只追加快照编号")
    arguments = parser.parse_args()
    retrieved_at = datetime.now(TIMEZONE)
    start = pd.Timestamp(arguments.start).normalize()
    end = pd.Timestamp(arguments.end or retrieved_at.date()).normalize()
    if start < DEFAULT_START:
        raise ValueError("冻结后快照起点不得早于2026-08-13")
    if end < start:
        raise ValueError("结束日期早于起始日期")
    run_id = arguments.run_id or retrieved_at.strftime("%Y%m%dT%H%M%S%z")
    snapshot_dir = SNAPSHOT_ROOT / run_id
    if snapshot_dir.exists():
        raise FileExistsError(f"快照目录已存在，拒绝覆盖：{snapshot_dir}")
    snapshot_dir.mkdir(parents=True, exist_ok=False)

    etf, benchmark, market_audit = _fetch_market(start, end)
    nav, nav_audit = _fetch_nav(start, end)
    shares, margin, flow_audit = _fetch_flows(start, end, retrieved_at)

    nav = nav.merge(etf[["date", "close"]], on="date", how="left", validate="one_to_one")
    nav["close_premium_to_nav"] = nav["close"] / nav["unit_nav"] - 1.0
    nav["close_premium_bps"] = nav["close_premium_to_nav"] * 10_000.0
    nav["retrieved_at"] = retrieved_at.isoformat()
    shares["retrieved_at"] = retrieved_at.isoformat()
    margin["retrieved_at"] = retrieved_at.isoformat()
    etf["retrieved_at"] = retrieved_at.isoformat()
    benchmark["retrieved_at"] = retrieved_at.isoformat()

    required_signal_columns = {
        "nav": (nav, ["close_premium_to_nav"]),
        "fund_share": (shares, ["fund_shares"]),
        "margin": (margin, ["rzye", "rqye", "rqyl", "rzrqye"]),
    }
    signal_calendar = pd.DatetimeIndex(etf["date"].sort_values().unique())
    missing_by_source: dict[str, list[str]] = {}
    complete_dates = signal_calendar
    for name, (frame, columns) in required_signal_columns.items():
        absent = set(columns).difference(frame.columns)
        if absent:
            raise ValueError(f"{name}缺少信号字段：{sorted(absent)}")
        valid = frame.loc[frame[columns].notna().all(axis=1), "date"]
        valid_dates = pd.DatetimeIndex(valid.unique())
        missing = signal_calendar.difference(valid_dates)
        missing_by_source[name] = [date.date().isoformat() for date in missing]
        complete_dates = complete_dates.intersection(valid_dates)
    benchmark_dates = pd.DatetimeIndex(benchmark["date"].unique())
    missing_benchmark = signal_calendar.difference(benchmark_dates)

    paths = {
        "etf": snapshot_dir / "510300_daily.parquet",
        "benchmark": snapshot_dir / "H00300_total_return_daily.parquet",
        "nav": snapshot_dir / "510300_nav_daily.parquet",
        "fund_share": snapshot_dir / "510300_fund_share_daily.parquet",
        "margin": snapshot_dir / "510300_margin_detail_daily.parquet",
    }
    frames = {
        "etf": etf,
        "benchmark": benchmark,
        "nav": nav,
        "fund_share": shares,
        "margin": margin,
    }
    for name, path in paths.items():
        _atomic_parquet(frames[name], path)

    has_signal = len(complete_dates) > 0
    has_market_pair = len(signal_calendar.intersection(benchmark_dates)) > 0
    if has_signal and has_market_pair:
        status = "PASS_COMPLETE_ROWS_AVAILABLE"
    elif has_signal or has_market_pair:
        status = "PARTIAL_SUCCESS"
    else:
        status = "FAILED_NO_USABLE_POST_CUTOFF_ROWS"
    payload = {
        "project_id": PROJECT_ID,
        "status": status,
        "run_id": run_id,
        "retrieved_at_asia_shanghai": retrieved_at.isoformat(),
        "requested_start": start.date().isoformat(),
        "requested_end": end.date().isoformat(),
        "frozen_files_untouched": True,
        "signal_calendar": {
            "etf_trading_dates": [date.date().isoformat() for date in signal_calendar],
            "complete_signal_dates": [date.date().isoformat() for date in complete_dates],
            "missing_by_source": missing_by_source,
        },
        "execution_market": {
            "missing_h00300_dates_vs_etf": [
                date.date().isoformat() for date in missing_benchmark
            ]
        },
        "collection_audit": {
            "market": market_audit,
            "nav": nav_audit,
            "flows": flow_audit,
        },
        "datasets": {
            name: _dataset_summary(frames[name], paths[name]) for name in paths
        },
        "boundaries": {
            "research_snapshot_only": True,
            "position_mapping": "DISABLED",
            "order_generation": "DISABLED",
            "broker_connection": "DISABLED",
            "live_trading_authorized": False,
        },
    }
    metadata_path = snapshot_dir / "snapshot_manifest.json"
    _atomic_json(payload, metadata_path)
    print(json.dumps({**payload, "snapshot_manifest": str(metadata_path)}, ensure_ascii=False, indent=2))
    return 0 if status != "FAILED_NO_USABLE_POST_CUTOFF_ROWS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
