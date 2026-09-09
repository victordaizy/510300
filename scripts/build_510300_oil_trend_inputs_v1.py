"""获取并审计510300原油趋势候选的冻结输入。

本脚本只保存外生原油价格并核对既有行情输入；不计算未来收益、
策略收益、仓位路径或夏普率。协议冻结后禁止再次覆盖输入快照。
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "510300_oil_trend_monthly_timing_v1.yaml"
MANIFEST_PATH = ROOT / "config" / "510300_oil_trend_monthly_timing_v1_manifest.json"
RAW_PATH = ROOT / "data" / "raw" / "macro" / "eia_brent_spot_daily_oil_trend_v1_raw.json"
PARQUET_PATH = ROOT / "data" / "raw" / "macro" / "eia_brent_spot_daily_oil_trend_v1.parquet"
REPORT_PATH = ROOT / "reports" / "data_quality" / "510300_oil_trend_inputs_v1.json"
PRICE_PATH = ROOT / "data" / "raw" / "market" / "510300_daily_downside_risk_v1.parquet"
PRICE_AUDIT_PATH = ROOT / "reports" / "data_quality" / "510300_downside_risk_inputs_v1.json"
DIVIDEND_PATH = ROOT / "data" / "reference" / "510300_dividends.csv"
BENCHMARK_PATH = ROOT / "data" / "raw" / "r6" / "H00300_total_return_daily.parquet"


class InputAuditError(RuntimeError):
    """输入不满足冻结要求。"""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_bytes(
        path,
        (json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode(
            "utf-8"
        ),
    )


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, engine="pyarrow")
    os.replace(temporary, path)


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def _download_eia(config: dict[str, Any]) -> tuple[bytes, str, str]:
    contract = config["data_contract"]
    api_key = os.environ.get("EIA_API_KEY", "DEMO_KEY")
    parameters: list[tuple[str, str | int]] = [
        ("api_key", api_key),
        ("frequency", "daily"),
        ("data[0]", "value"),
        ("facets[series][]", contract["oil_series"]),
        ("start", contract["oil_request_start"]),
        ("end", contract["oil_request_end"]),
        ("sort[0][column]", "period"),
        ("sort[0][direction]", "asc"),
        ("offset", 0),
        ("length", 5000),
    ]
    endpoint = contract["oil_api_route"]
    response = requests.get(endpoint, params=parameters, timeout=180)
    response.raise_for_status()
    redacted_parameters = [
        (key, "REDACTED_PUBLIC_OR_ENV_KEY" if key == "api_key" else value)
        for key, value in parameters
    ]
    return response.content, response.url, endpoint + "?" + urlencode(redacted_parameters)


def _parse_eia(payload: bytes, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    document = json.loads(payload.decode("utf-8"))
    response = document.get("response", {})
    rows = response.get("data", [])
    if not rows:
        raise InputAuditError("EIA响应没有数据行")
    frame = pd.DataFrame(rows)
    required = {"period", "value", "series", "series-description", "units"}
    if missing := required.difference(frame.columns):
        raise InputAuditError(f"EIA响应缺少字段：{sorted(missing)}")
    frame = frame.rename(
        columns={
            "period": "date",
            "series-description": "series_description",
            "product-name": "product_name",
            "process-name": "process_name",
            "area-name": "area_name",
        }
    )
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.sort_values("date").reset_index(drop=True)
    availability_lag = int(config["oil_factor"]["availability_lag_us_business_days"])
    robust_lag = int(
        config["oil_factor"]["robustness_availability_lag_us_business_days"]
    )
    frame["availability_date"] = frame["date"] + pd.offsets.BDay(availability_lag)
    frame["robust_availability_date"] = frame["date"] + pd.offsets.BDay(robust_lag)
    total = int(response.get("total", len(frame)))
    metadata = {
        "api_total": total,
        "response_rows": int(len(frame)),
        "description": response.get("description"),
        "warnings": response.get("warnings", []),
    }
    return frame, metadata


def _audit_existing_inputs(config: dict[str, Any]) -> dict[str, Any]:
    required_paths = [PRICE_PATH, PRICE_AUDIT_PATH, DIVIDEND_PATH, BENCHMARK_PATH]
    missing = [path.relative_to(ROOT).as_posix() for path in required_paths if not path.exists()]
    if missing:
        raise InputAuditError(f"缺少既有输入：{missing}")
    prior_audit = json.loads(PRICE_AUDIT_PATH.read_text(encoding="utf-8"))
    if prior_audit.get("status") != "PASS":
        raise InputAuditError("510300既有行情审计不是PASS")
    if prior_audit.get("candidate_outcomes_computed") is not False:
        raise InputAuditError("510300既有行情审计边界异常")
    market = pd.read_parquet(PRICE_PATH)
    required_columns = set(config["inputs"]["etf_daily"]["required_columns"])
    if missing_columns := required_columns.difference(market.columns):
        raise InputAuditError(f"510300行情缺少字段：{sorted(missing_columns)}")
    market["date"] = pd.to_datetime(market["date"], errors="coerce")
    numeric = market[["open", "high", "low", "close"]].apply(
        pd.to_numeric, errors="coerce"
    )
    return {
        "rows": int(len(market)),
        "first_date": market["date"].min().date().isoformat(),
        "last_date": market["date"].max().date().isoformat(),
        "duplicate_dates": int(market["date"].duplicated().sum()),
        "missing_or_nonpositive_ohlc_rows": int(
            (numeric.isna().any(axis=1) | (numeric <= 0).any(axis=1)).sum()
        ),
        "prior_audit_path": PRICE_AUDIT_PATH.relative_to(ROOT).as_posix(),
        "prior_audit_sha256": sha256_file(PRICE_AUDIT_PATH),
    }


def build() -> dict[str, Any]:
    if MANIFEST_PATH.exists():
        raise InputAuditError(f"协议已经冻结，禁止覆盖输入：{MANIFEST_PATH}")
    config = load_config()
    payload, resolved_url, redacted_url = _download_eia(config)
    frame, api_metadata = _parse_eia(payload, config)
    market = _audit_existing_inputs(config)
    contract = config["data_contract"]

    series_values = sorted(frame["series"].dropna().astype(str).unique().tolist())
    units_values = sorted(frame["units"].dropna().astype(str).unique().tolist())
    date_gap_days = frame["date"].diff().dt.days.dropna()
    checks = {
        "api_total_matches_rows": api_metadata["api_total"] == len(frame),
        "oil_minimum_rows": len(frame) >= int(contract["oil_minimum_rows"]),
        "oil_dates_exact": (
            frame["date"].min() == pd.Timestamp(contract["oil_expected_first_observation"])
            and frame["date"].max()
            == pd.Timestamp(contract["oil_expected_last_observation"])
        ),
        "oil_series_exact": series_values == [contract["oil_series"]],
        "oil_units_exact": units_values == [contract["oil_units"]],
        "oil_dates_unique": not frame["date"].duplicated().any(),
        "oil_values_complete_positive": (
            not frame["value"].isna().any() and bool((frame["value"] > 0).all())
        ),
        "oil_dates_monotonic": bool(frame["date"].is_monotonic_increasing),
        "availability_strictly_after_observation": bool(
            (frame["availability_date"] > frame["date"]).all()
        ),
        "robust_availability_not_earlier": bool(
            (frame["robust_availability_date"] >= frame["availability_date"]).all()
        ),
        "etf_rows_exact": market["rows"] == int(contract["expected_etf_rows"]),
        "etf_dates_exact": (
            market["first_date"] == contract["expected_etf_first_date"]
            and market["last_date"] == contract["expected_etf_last_date"]
        ),
        "etf_unique_complete_ohlc": (
            market["duplicate_dates"] == 0
            and market["missing_or_nonpositive_ohlc_rows"] == 0
        ),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    report = {
        "schema_version": "1.0.0",
        "report_id": "510300_OIL_TREND_INPUTS_V1",
        "status": status,
        "checked_at_asia_shanghai": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "candidate_outcomes_computed": False,
        "portfolio_returns_computed": False,
        "source": {
            "owner": contract["oil_source_owner"],
            "api_route": contract["oil_api_route"],
            "resolved_url_with_key_omitted": redacted_url,
            "http_resolved_url_was_received": bool(resolved_url),
            "series": series_values,
            "units": units_values,
            "request_start": contract["oil_request_start"],
            "request_end": contract["oil_request_end"],
            "raw_response_sha256": sha256_bytes(payload),
            "raw_response_bytes": len(payload),
            **api_metadata,
        },
        "oil_daily": {
            "rows": int(len(frame)),
            "first_date": frame["date"].min().date().isoformat(),
            "last_date": frame["date"].max().date().isoformat(),
            "duplicate_dates": int(frame["date"].duplicated().sum()),
            "missing_values": int(frame["value"].isna().sum()),
            "nonpositive_values": int((frame["value"] <= 0).sum()),
            "minimum_value": float(frame["value"].min()),
            "maximum_value": float(frame["value"].max()),
            "maximum_calendar_gap_days": int(date_gap_days.max()),
            "availability_lag_us_business_days": int(
                config["oil_factor"]["availability_lag_us_business_days"]
            ),
            "robustness_availability_lag_us_business_days": int(
                config["oil_factor"]["robustness_availability_lag_us_business_days"]
            ),
        },
        "etf_daily": market,
        "checks": checks,
        "boundaries": {
            "oil_factor_computed": False,
            "future_510300_returns_read": False,
            "candidate_outcomes_computed": False,
            "strategy_returns_read": False,
            "parameter_selection_performed": False,
            "live_trading_authorized": False,
        },
        "bound_input_hashes_before_outputs": {
            path.relative_to(ROOT).as_posix(): sha256_file(path)
            for path in [PRICE_PATH, PRICE_AUDIT_PATH, DIVIDEND_PATH, BENCHMARK_PATH]
        },
    }
    if status != "PASS":
        raise InputAuditError(json.dumps(report, ensure_ascii=False, allow_nan=False))

    _atomic_bytes(RAW_PATH, payload)
    _atomic_parquet(PARQUET_PATH, frame)
    report["output_files"] = {
        RAW_PATH.relative_to(ROOT).as_posix(): {
            "sha256": sha256_file(RAW_PATH),
            "bytes": RAW_PATH.stat().st_size,
        },
        PARQUET_PATH.relative_to(ROOT).as_posix(): {
            "sha256": sha256_file(PARQUET_PATH),
            "bytes": PARQUET_PATH.stat().st_size,
        },
    }
    _atomic_json(REPORT_PATH, report)
    return report


def main() -> int:
    report = build()
    print(
        json.dumps(
            {
                "status": report["status"],
                "oil_rows": report["oil_daily"]["rows"],
                "oil_first_date": report["oil_daily"]["first_date"],
                "oil_last_date": report["oil_daily"]["last_date"],
                "raw_response_sha256": report["source"]["raw_response_sha256"],
                "candidate_outcomes_computed": False,
                "portfolio_returns_computed": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
