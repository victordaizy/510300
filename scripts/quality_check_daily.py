"""对 510300 五年原始日线执行可重复的数据质量审计。"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = PROJECT_ROOT / "config" / "settings.yaml"
DATA_FILE = PROJECT_ROOT / "data" / "raw" / "market" / "510300_daily_raw.parquet"
METADATA_FILE = PROJECT_ROOT / "data" / "raw" / "market" / "510300_daily_raw.metadata.json"
DIVIDEND_FILE = PROJECT_ROOT / "data" / "reference" / "510300_dividends.csv"
CROSS_CHECK_FILE = PROJECT_ROOT / "reports" / "data_quality" / "510300_daily_cross_source.json"
REPORT_DIR = PROJECT_ROOT / "reports" / "data_quality"
REPORT_FILE = REPORT_DIR / "510300_daily_quality.json"


def load_config() -> dict:
    with CONFIG_FILE.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_daily_metadata(
    data: pd.DataFrame,
    data_file: Path,
    *,
    requested_start: str,
    requested_end: str,
    evaluation_start: str,
    historical_evaluation_end: str,
) -> dict:
    """根据已经落盘的日线文件构造可审计元数据。"""

    dates = pd.to_datetime(data["date"], errors="raise")
    source_segments = []
    for source, group in data.groupby("source", dropna=False, sort=True):
        group_dates = pd.to_datetime(group["date"], errors="raise")
        source_segments.append(
            {
                "source": str(source),
                "row_count": int(len(group)),
                "first_date": group_dates.min().date().isoformat(),
                "last_date": group_dates.max().date().isoformat(),
            }
        )
    retrieved_values = sorted(data["retrieved_at"].dropna().astype(str).unique().tolist())
    try:
        file_name = data_file.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        file_name = str(data_file)
    return {
        "schema_version": 2,
        "symbol": str(data["symbol"].iloc[0]),
        "coverage_role": "FEATURE_WARMUP_AND_FORWARD_REFRESH",
        "start_date_requested": requested_start,
        "end_date_requested": requested_end,
        "evaluation_start_date": evaluation_start,
        "historical_evaluation_end_date": historical_evaluation_end,
        "actual_first_date": dates.min().date().isoformat(),
        "actual_last_date": dates.max().date().isoformat(),
        "row_count": int(len(data)),
        "source_segments": source_segments,
        "retrieved_at_values": retrieved_values,
        "volume_unit": "share",
        "amount_unit": "CNY",
        "file": file_name,
        "sha256": sha256_file(data_file),
    }


def write_daily_metadata(
    data: pd.DataFrame,
    data_file: Path,
    metadata_file: Path,
    *,
    requested_start: str,
    requested_end: str,
    evaluation_start: str,
    historical_evaluation_end: str,
) -> dict:
    """原子写入日线元数据，避免行情文件与元数据只更新一半。"""

    metadata = build_daily_metadata(
        data,
        data_file,
        requested_start=requested_start,
        requested_end=requested_end,
        evaluation_start=evaluation_start,
        historical_evaluation_end=historical_evaluation_end,
    )
    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = metadata_file.with_suffix(metadata_file.suffix + ".tmp")
    temporary.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(metadata_file)
    return metadata


def check_metadata_contract(
    data: pd.DataFrame, metadata: dict, data_hash: str
) -> list[dict]:
    """检查元数据是否逐项描述当前Parquet，而不只比较哈希。"""

    errors: list[dict] = []
    dates = pd.to_datetime(data["date"], errors="coerce")
    expected = {
        "sha256": data_hash,
        "row_count": int(len(data)),
        "actual_first_date": dates.min().date().isoformat(),
        "actual_last_date": dates.max().date().isoformat(),
    }
    for field, actual in expected.items():
        if metadata.get(field) != actual:
            errors.append(
                {
                    "code": f"METADATA_{field.upper()}_MISMATCH",
                    "expected": actual,
                    "actual": metadata.get(field),
                }
            )

    declared_sources = {
        str(item.get("source")): int(item.get("row_count", -1))
        for item in metadata.get("source_segments", [])
    }
    actual_sources = {
        str(source): int(count)
        for source, count in data.groupby("source", dropna=False).size().items()
    }
    if declared_sources != actual_sources:
        errors.append(
            {
                "code": "METADATA_SOURCE_SEGMENTS_MISMATCH",
                "expected": actual_sources,
                "actual": declared_sources,
            }
        )
    return errors


def check_daily_data(
    data: pd.DataFrame, config: dict, metadata: dict | None = None
) -> tuple[list[dict], list[dict]]:
    """返回阻断回测的错误，以及需要人工关注但不阻断的警告。"""
    errors: list[dict] = []
    warnings: list[dict] = []
    required_columns = [
        "symbol", "date", "open", "high", "low", "close", "volume", "amount",
        "source", "retrieved_at", "volume_unit", "amount_unit",
    ]
    missing = [column for column in required_columns if column not in data.columns]
    if missing:
        return [{"code": "MISSING_COLUMNS", "columns": missing}], warnings
    if data.empty:
        return [{"code": "EMPTY_DATA"}], warnings

    dates = pd.to_datetime(data["date"], errors="coerce")
    if dates.isna().any():
        errors.append({"code": "INVALID_DATE", "count": int(dates.isna().sum())})
    duplicate_count = int(dates.duplicated().sum())
    if duplicate_count:
        errors.append({"code": "DUPLICATE_DATE", "count": duplicate_count})
    if not dates.is_monotonic_increasing:
        errors.append({"code": "DATE_NOT_SORTED"})
    weekend_count = int((dates.dt.weekday >= 5).sum())
    if weekend_count:
        errors.append({"code": "WEEKEND_DATA", "count": weekend_count})

    numeric_columns = ["open", "high", "low", "close", "volume", "amount"]
    numeric = data[numeric_columns].apply(pd.to_numeric, errors="coerce")
    invalid = numeric.isna().sum()
    for column, count in invalid[invalid > 0].items():
        errors.append({"code": "INVALID_NUMERIC", "column": column, "count": int(count)})
    if errors:
        return errors, warnings

    prices = numeric[["open", "high", "low", "close"]]
    non_positive = int((prices <= 0).any(axis=1).sum())
    if non_positive:
        errors.append({"code": "NON_POSITIVE_PRICE", "count": non_positive})
    negative_flow = int((numeric[["volume", "amount"]] < 0).any(axis=1).sum())
    if negative_flow:
        errors.append({"code": "NEGATIVE_VOLUME_OR_AMOUNT", "count": negative_flow})
    zero_flow = int((numeric[["volume", "amount"]] == 0).any(axis=1).sum())
    if zero_flow:
        errors.append({"code": "ZERO_VOLUME_OR_AMOUNT", "count": zero_flow})

    high_bad = numeric["high"] < numeric[["open", "close", "low"]].max(axis=1)
    low_bad = numeric["low"] > numeric[["open", "close", "high"]].min(axis=1)
    if high_bad.any() or low_bad.any():
        errors.append({"code": "OHLC_INCONSISTENT", "count": int((high_bad | low_bad).sum())})

    price_tick = float(config.get("quality", {}).get("price_tick", 0.001))
    tick_scaled = prices / price_tick
    invalid_tick = (~np.isclose(tick_scaled, np.round(tick_scaled), atol=1e-7)).any(axis=1)
    if invalid_tick.any():
        errors.append({"code": "INVALID_PRICE_TICK", "count": int(invalid_tick.sum())})

    implied_vwap = numeric["amount"] / numeric["volume"]
    vwap_bad = (implied_vwap < numeric["low"] - price_tick) | (implied_vwap > numeric["high"] + price_tick)
    if vwap_bad.any():
        errors.append({"code": "IMPLIED_VWAP_OUTSIDE_RANGE", "count": int(vwap_bad.sum())})

    expected_symbol = config["symbols"]["etf"]
    symbols = data["symbol"].dropna().astype(str).unique().tolist()
    if symbols != [expected_symbol]:
        errors.append({"code": "UNEXPECTED_SYMBOL", "expected": expected_symbol, "actual": symbols})
    if data["volume_unit"].dropna().unique().tolist() != ["share"]:
        errors.append({"code": "INVALID_VOLUME_UNIT"})
    if data["amount_unit"].dropna().unique().tolist() != ["CNY"]:
        errors.append({"code": "INVALID_AMOUNT_UNIT"})

    metadata = metadata or {}
    expected_start = pd.Timestamp(
        metadata.get(
            "start_date_requested",
            config["project"].get("feature_warmup_start", config["project"]["start_date"]),
        )
    )
    expected_end = pd.Timestamp(
        metadata.get("end_date_requested", config["project"]["end_date"])
    )
    actual_start, actual_end = dates.min(), dates.max()
    if actual_start != expected_start:
        errors.append({"code": "START_DATE_MISMATCH", "expected": str(expected_start.date()), "actual": str(actual_start.date())})
    if actual_end != expected_end:
        errors.append({"code": "END_DATE_MISMATCH", "expected": str(expected_end.date()), "actual": str(actual_end.date())})

    source_values = data["source"].dropna().unique().tolist()
    allowed_sources = {
        "akshare.fund_etf_hist_sina",
        "sina.hq.batch_quote",
    }
    unexpected_sources = sorted(set(source_values).difference(allowed_sources))
    if unexpected_sources:
        errors.append({"code": "UNEXPECTED_PRIMARY_SOURCE", "values": unexpected_sources})
    large_jump = numeric["close"].pct_change().abs() > 0.10
    if large_jump.any():
        warnings.append({"code": "PRICE_JUMP_OVER_10_PERCENT", "dates": dates[large_jump].dt.strftime("%Y-%m-%d").tolist()})
    return errors, warnings


def check_supporting_artifacts(data_hash: str, dates: pd.Series) -> tuple[list[dict], list[dict], dict]:
    errors: list[dict] = []
    warnings: list[dict] = []
    evidence: dict = {
        "minute_check": "NOT_APPLICABLE_DAILY_BACKTEST",
        "overfitting_check": "NOT_APPLICABLE_DATA_AUDIT_REQUIRES_STRATEGY",
    }

    if not DIVIDEND_FILE.exists():
        errors.append({"code": "MISSING_DIVIDEND_REFERENCE"})
    else:
        dividends = pd.read_csv(DIVIDEND_FILE, parse_dates=["record_date", "ex_date", "payment_date"])
        invalid_cash = dividends["cash_dividend_per_share"].isna() | (dividends["cash_dividend_per_share"] <= 0)
        if invalid_cash.any():
            errors.append({"code": "INVALID_DIVIDEND", "count": int(invalid_cash.sum())})
        in_range = dividends["ex_date"].between(dates.min(), dates.max())
        missing_ex_dates = dividends.loc[in_range & ~dividends["ex_date"].isin(dates), "ex_date"]
        if not missing_ex_dates.empty:
            errors.append({"code": "DIVIDEND_DATE_MISSING_FROM_PRICE_DATA", "dates": missing_ex_dates.dt.strftime("%Y-%m-%d").tolist()})
        evidence["dividend_events_in_range"] = int(in_range.sum())

    if not CROSS_CHECK_FILE.exists():
        errors.append({"code": "MISSING_CROSS_SOURCE_REPORT"})
    else:
        cross = json.loads(CROSS_CHECK_FILE.read_text(encoding="utf-8"))
        evidence["cross_source"] = cross
        if cross.get("primary_sha256") != data_hash:
            errors.append({"code": "STALE_CROSS_SOURCE_REPORT"})
        elif cross.get("status") != "PASS":
            errors.append({"code": "CROSS_SOURCE_CHECK_FAILED", "status": cross.get("status")})
    return errors, warnings, evidence


def main() -> int:
    config = load_config()
    if not DATA_FILE.exists():
        print(f"找不到数据文件：{DATA_FILE}", file=sys.stderr)
        return 1
    data = pd.read_parquet(DATA_FILE)
    data_hash = sha256_file(DATA_FILE)
    metadata = json.loads(METADATA_FILE.read_text(encoding="utf-8")) if METADATA_FILE.exists() else {}
    errors, warnings = check_daily_data(data, config, metadata)
    support_errors, support_warnings, evidence = check_supporting_artifacts(data_hash, pd.to_datetime(data["date"]))
    errors.extend(support_errors)
    warnings.extend(support_warnings)

    errors.extend(check_metadata_contract(data, metadata, data_hash))
    status = "FAIL" if errors else ("WARN" if warnings else "PASS")
    report = {
        "status": status,
        "scope": "510300 日频原始覆盖与回测契约",
        "checked_at": datetime.now(ZoneInfo(config["project"]["timezone"])).isoformat(),
        "file": DATA_FILE.relative_to(PROJECT_ROOT).as_posix(),
        "sha256": data_hash,
        "row_count": int(len(data)),
        "actual_first_date": pd.to_datetime(data["date"]).min().date().isoformat(),
        "actual_last_date": pd.to_datetime(data["date"]).max().date().isoformat(),
        "errors": errors,
        "warnings": warnings,
        "evidence": evidence,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if status == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
