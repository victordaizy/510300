from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import akshare as ak
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "market_margin_leverage_v0"
ETF_CALENDAR_FILE = PROJECT_ROOT / "data" / "raw" / "r6" / "510300_daily.parquet"
START_DATE = pd.Timestamp("2015-01-01")
DEVELOPMENT_CUTOFF = pd.Timestamp("2020-12-31")
SZSE_SAMPLE_DATES = ["20151231", "20161230", "20171229", "20181228", "20191231", "20201231"]

SSE_PAGE = "https://www.sse.com.cn/market/othersdata/margin/sum/"
SZSE_PAGE = "https://www.szse.cn/disclosure/margin/margin/index.html"
JIN10_SH_PAGE = "https://datacenter.jin10.com/reportType/dc_market_margin_sse"
JIN10_SZ_PAGE = "https://datacenter.jin10.com/reportType/dc_market_margin_sz"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".parquet", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _normalize_jin10(frame: pd.DataFrame, exchange: str) -> pd.DataFrame:
    if frame.shape[1] != 7:
        raise ValueError(f"{exchange} Jin10 两融汇总字段数量漂移：{frame.shape[1]}")
    result = frame.copy()
    result.columns = ["date", "rzmre", "rzye", "rqmcl", "rqyl", "rqye", "rzrqye"]
    result["date"] = pd.to_datetime(result["date"], errors="raise").dt.normalize()
    numeric = ["rzmre", "rzye", "rqmcl", "rqyl", "rqye", "rzrqye"]
    result[numeric] = result[numeric].apply(pd.to_numeric, errors="raise")
    result = result[result["date"].between(START_DATE, DEVELOPMENT_CUTOFF)].copy()
    result = result.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    result["exchange"] = exchange
    result["transport_source"] = "jin10.cdn.history"
    result["underlying_source"] = f"{exchange}.exchange.margin.summary"
    return result[["date", "exchange", *numeric, "transport_source", "underlying_source"]]


def _normalize_sse_official(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.shape[1] != 7:
        raise ValueError(f"上交所两融汇总字段数量漂移：{frame.shape[1]}")
    result = frame.copy()
    result.columns = ["date", "rzye", "rzmre", "rqyl", "rqye", "rqmcl", "rzrqye"]
    result["date"] = pd.to_datetime(result["date"], format="%Y%m%d", errors="raise").dt.normalize()
    numeric = ["rzye", "rzmre", "rqyl", "rqye", "rqmcl", "rzrqye"]
    result[numeric] = result[numeric].apply(pd.to_numeric, errors="raise")
    result = result.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    result["exchange"] = "SSE"
    result["source"] = "sse.official.queryMargin"
    return result[["date", "exchange", *numeric, "source"]]


def _normalize_szse_official(date: str, frame: pd.DataFrame) -> pd.DataFrame:
    if frame.shape != (1, 6):
        raise ValueError(f"深交所样本 {date} 返回结构异常：{frame.shape}")
    result = frame.copy()
    result.columns = ["rzmre", "rzye", "rqmcl", "rqyl", "rqye", "rzrqye"]
    numeric = ["rzmre", "rzye", "rqmcl", "rqyl", "rqye", "rzrqye"]
    result[numeric] = result[numeric].apply(pd.to_numeric, errors="raise") * 100_000_000.0
    result.insert(0, "date", pd.Timestamp(date))
    result.insert(1, "exchange", "SZSE")
    result["source"] = "szse.official.ShowReport"
    return result[["date", "exchange", *numeric, "source"]]


def _fetch_szse_date(date: pd.Timestamp, maximum_attempts: int = 4) -> pd.DataFrame:
    date_text = date.strftime("%Y%m%d")
    last_error: Exception | None = None
    for attempt in range(1, maximum_attempts + 1):
        try:
            return _normalize_szse_official(date_text, ak.stock_margin_szse(date_text))
        except Exception as exc:
            last_error = exc
            if attempt < maximum_attempts:
                time.sleep(0.5 * attempt)
    raise RuntimeError(f"深交所两融汇总下载失败：{date_text}") from last_error


def _comparison(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    columns: list[str],
    label: str,
    tolerance: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    merged = reference[["date", *columns]].merge(
        candidate[["date", *columns]],
        on="date",
        how="inner",
        suffixes=("_official", "_secondary"),
        validate="one_to_one",
    )
    if merged.empty:
        raise ValueError(f"{label} 没有可交叉核对日期")
    relative_columns: list[str] = []
    for column in columns:
        denominator = merged[f"{column}_official"].abs().clip(lower=1.0)
        relative_column = f"{column}_relative_error"
        merged[relative_column] = (
            merged[f"{column}_secondary"] - merged[f"{column}_official"]
        ).abs() / denominator
        relative_columns.append(relative_column)
    maximum = float(merged[relative_columns].max().max())
    summary = {
        "label": label,
        "matched_dates": int(len(merged)),
        "first_date": merged["date"].min().date().isoformat(),
        "last_date": merged["date"].max().date().isoformat(),
        "maximum_relative_error": maximum,
        "tolerance": tolerance,
        "status": "PASS" if maximum <= tolerance else "FAIL",
    }
    return merged, summary


def _build_combined(sh: pd.DataFrame, sz: pd.DataFrame) -> pd.DataFrame:
    numeric = ["rzmre", "rzye", "rqmcl", "rqyl", "rqye", "rzrqye"]
    combined = sh[["date", *numeric]].merge(
        sz[["date", *numeric]],
        on="date",
        how="inner",
        suffixes=("_sse", "_szse"),
        validate="one_to_one",
    )
    for column in numeric:
        combined[f"market_{column}"] = combined[f"{column}_sse"] + combined[f"{column}_szse"]
    combined["market_rzye_change"] = combined["market_rzye"].diff()
    combined["market_financing_balance_identity_residual"] = (
        combined["market_rzye_change"] - combined["market_rzmre"]
    )
    combined["publication_rule"] = "交易日T汇总于T+1交易日上午发布；T日收盘决策只能使用严格早于T的记录"
    combined["source"] = "sse_official_plus_jin10_szse_crosschecked"
    return combined.sort_values("date").reset_index(drop=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    retrieved_at = datetime.now(timezone.utc).isoformat()

    sh_secondary = _normalize_jin10(ak.macro_china_market_margin_sh(), "SSE")
    sz_secondary = _normalize_jin10(ak.macro_china_market_margin_sz(), "SZSE")
    sse_official = _normalize_sse_official(
        ak.stock_margin_sse(START_DATE.strftime("%Y%m%d"), DEVELOPMENT_CUTOFF.strftime("%Y%m%d"))
    )
    calendar = pd.read_parquet(
        ETF_CALENDAR_FILE,
        columns=["date"],
        filters=[
            ("date", ">=", START_DATE.to_pydatetime()),
            ("date", "<=", DEVELOPMENT_CUTOFF.to_pydatetime()),
        ],
    )
    calendar["date"] = pd.to_datetime(calendar["date"], errors="raise").dt.normalize()
    calendar = calendar.sort_values("date").drop_duplicates("date", keep="last")
    szse_official = pd.concat(
        [_fetch_szse_date(pd.Timestamp(date), maximum_attempts=8) for date in SZSE_SAMPLE_DATES],
        ignore_index=True,
    )

    sse_comparison, sse_summary = _comparison(
        sse_official,
        sh_secondary,
        ["rzye", "rzmre", "rqyl", "rqmcl", "rzrqye"],
        "SSE_OFFICIAL_VS_JIN10_HISTORY",
        1e-6,
    )
    szse_comparison, szse_summary = _comparison(
        szse_official,
        sz_secondary,
        ["rzye", "rzmre", "rzrqye"],
        "SZSE_OFFICIAL_VS_JIN10_HISTORY",
        2e-4,
    )
    if sse_summary["status"] != "PASS" or szse_summary["status"] != "PASS":
        raise ValueError(
            f"两融历史交叉核对失败：SSE={sse_summary['maximum_relative_error']}, "
            f"SZSE={szse_summary['maximum_relative_error']}"
        )

    combined = _build_combined(sse_official, sz_secondary)
    expected_dates = pd.DatetimeIndex(calendar["date"].sort_values().drop_duplicates())
    actual_dates = pd.DatetimeIndex(combined["date"].sort_values().drop_duplicates())
    coverage_by_year: dict[str, Any] = {}
    for year in range(START_DATE.year, DEVELOPMENT_CUTOFF.year + 1):
        expected_year = expected_dates[expected_dates.year == year]
        actual_year = actual_dates[actual_dates.year == year]
        missing_year = expected_year.difference(actual_year)
        coverage_by_year[str(year)] = {
            "expected_trading_days": int(len(expected_year)),
            "observed_common_days": int(len(expected_year.intersection(actual_year))),
            "coverage": float(len(expected_year.intersection(actual_year)) / len(expected_year)),
            "missing_dates": [date.date().isoformat() for date in missing_year],
        }
    selection_missing = expected_dates[
        (expected_dates >= pd.Timestamp("2019-01-01")) & (expected_dates <= DEVELOPMENT_CUTOFF)
    ].difference(actual_dates)
    if len(selection_missing):
        raise ValueError(f"2019-2020 选择段两融汇总存在缺口：{selection_missing.tolist()}")
    files = {
        "sse_jin10_history": OUTPUT_DIR / "sse_margin_history_jin10.parquet",
        "szse_jin10_history": OUTPUT_DIR / "szse_margin_history_jin10.parquet",
        "sse_official_history": OUTPUT_DIR / "sse_margin_history_official.parquet",
        "szse_official_samples": OUTPUT_DIR / "szse_margin_samples_official.parquet",
        "sse_comparison": OUTPUT_DIR / "sse_crosscheck.parquet",
        "szse_comparison": OUTPUT_DIR / "szse_crosscheck.parquet",
        "combined": OUTPUT_DIR / "market_margin_sh_sz_daily.parquet",
    }
    frames = {
        "sse_jin10_history": sh_secondary,
        "szse_jin10_history": sz_secondary,
        "sse_official_history": sse_official,
        "szse_official_samples": szse_official,
        "sse_comparison": sse_comparison,
        "szse_comparison": szse_comparison,
        "combined": combined,
    }
    for key, path in files.items():
        _atomic_parquet(frames[key], path)

    metadata = {
        "schema_version": "0.1.0",
        "dataset_id": "MARKET_MARGIN_LEVERAGE_V0",
        "status": "DISCOVERY_INPUT_SECONDARY_SZSE_OFFICIAL_SAMPLE_CROSSCHECK_PASS",
        "retrieved_at_utc": retrieved_at,
        "development_cutoff": DEVELOPMENT_CUTOFF.date().isoformat(),
        "history_transport": {
            "provider": "Jin10 CDN through AkShare",
            "sse_page": JIN10_SH_PAGE,
            "szse_page": JIN10_SZ_PAGE,
            "role": "上交所模型输入使用官方历史；深交所使用批量历史并以每年固定样本和官方接口逐字段核对。",
        },
        "official_crosscheck": {
            "sse_page": SSE_PAGE,
            "szse_page": SZSE_PAGE,
            "sse": sse_summary,
            "szse": szse_summary,
            "szse_fixed_sample_dates": SZSE_SAMPLE_DATES,
        },
        "combined": {
            "rows": int(len(combined)),
            "first_date": combined["date"].min().date().isoformat(),
            "last_date": combined["date"].max().date().isoformat(),
            "coverage_by_year": coverage_by_year,
            "publication_rule": "交易日T汇总于T+1交易日上午发布；收盘信号严格滞后一交易日。",
        },
        "artifacts": {
            key: {
                "file": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "rows": int(len(frames[key])),
                "sha256": _sha256(path),
            }
            for key, path in files.items()
        },
    }
    metadata_path = OUTPUT_DIR / "metadata.json"
    _atomic_json(metadata, metadata_path)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
