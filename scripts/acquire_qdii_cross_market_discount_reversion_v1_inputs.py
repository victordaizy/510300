"""补齐并轻量核对QDII跨市场折价回归V1所需的纳斯达克100指数。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "qdii_cross_market_discount_reversion_v1.yaml"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) CodexResearch/1.0"
YAHOO_BASE = "https://query1.finance.yahoo.com/v8/finance/chart"
SYMBOL = "^NDX"
START_DATE = "2000-01-01"
CUTOFF_DATE = "2026-08-18"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def request_json(url: str, retries: int) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = requests.get(
                url,
                timeout=30,
                headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
            )
            response.raise_for_status()
            payload = response.json()
            chart_error = payload.get("chart", {}).get("error")
            if chart_error:
                raise RuntimeError(f"Yahoo图表接口返回错误：{chart_error}")
            return payload
        except Exception as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(min(2**attempt, 10))
    assert last_error is not None
    raise last_error


def fetch_ndx(retries: int = 6) -> tuple[pd.DataFrame, dict[str, Any]]:
    start_epoch = int(pd.Timestamp(START_DATE, tz="UTC").timestamp())
    end_epoch = int((pd.Timestamp(CUTOFF_DATE, tz="UTC") + pd.Timedelta(days=1)).timestamp())
    url = (
        f"{YAHOO_BASE}/{quote(SYMBOL, safe='')}?period1={start_epoch}"
        f"&period2={end_epoch}&interval=1d&events=div%2Csplits"
    )
    payload = request_json(url, retries)
    result = payload["chart"]["result"][0]
    timestamps = result.get("timestamp") or []
    quote_block = result["indicators"]["quote"][0]
    adjusted_block = result["indicators"].get("adjclose", [{}])[0]
    timezone_name = result.get("meta", {}).get("exchangeTimezoneName", "UTC")
    dates = pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(timezone_name).date
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "open": quote_block.get("open"),
            "high": quote_block.get("high"),
            "low": quote_block.get("low"),
            "close": quote_block.get("close"),
            "adj_close": adjusted_block.get("adjclose", quote_block.get("close")),
            "volume": quote_block.get("volume"),
            "symbol": SYMBOL,
            "source": "yahoo.finance.chart.v8",
        }
    )
    frame = frame.dropna(subset=["date", "open", "high", "low", "close", "adj_close"])
    frame = frame.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    numeric = ["open", "high", "low", "close", "adj_close"]
    checks = {
        "minimum_5000_rows": len(frame) >= 5000,
        "date_unique": frame["date"].is_unique,
        "date_increasing": frame["date"].is_monotonic_increasing,
        "prices_finite": bool(np.isfinite(frame[numeric].to_numpy(float)).all()),
        "prices_positive": bool(frame[numeric].gt(0.0).all().all()),
        "high_valid": bool(frame["high"].ge(frame[["open", "close"]].max(axis=1) - 1e-9).all()),
        "low_valid": bool(frame["low"].le(frame[["open", "close"]].min(axis=1) + 1e-9).all()),
        "cutoff_not_exceeded": bool(frame["date"].max() <= pd.Timestamp(CUTOFF_DATE)),
    }
    if not all(checks.values()):
        raise RuntimeError(f"纳斯达克100输入核对失败：{checks}")
    metadata = {
        "symbol": SYMBOL,
        "url": url,
        "exchange_timezone": timezone_name,
        "currency": result.get("meta", {}).get("currency"),
        "exchange_name": result.get("meta", {}).get("exchangeName"),
        "retrieved_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "rows": int(len(frame)),
        "first_date": frame["date"].min().date().isoformat(),
        "last_date": frame["date"].max().date().isoformat(),
        "checks": checks,
    }
    return frame, metadata


def audit_existing_inputs(contract: dict[str, Any]) -> dict[str, Any]:
    required = {
        "etf_total_return_panel": ROOT / contract["inputs"]["etf_total_return_panel"],
        "fund_master": ROOT / contract["inputs"]["fund_master"],
        "benchmark_total_return": ROOT / contract["inputs"]["benchmark_total_return"],
        "benchmark_price_ohlc": ROOT / contract["inputs"]["benchmark_price_ohlc"],
    }
    for group in contract["tracked_groups"]:
        required[f"index_{group['group_id']}"] = (
            ROOT / contract["inputs"]["global_index_directory"] / group["index_file"]
        )
    fx_files = sorted(
        (ROOT / contract["inputs"]["fx_directory"]).glob(
            contract["inputs"]["fx_file_pattern"]
        )
    )
    if not fx_files:
        raise FileNotFoundError("未找到人民币中间价JSON")
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"缺少QDII输入：{missing}")
    return {
        "required_files": {
            name: {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
            for name, path in required.items()
        },
        "fx_files": [
            {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
            for path in fx_files
        ],
    }


def acquire(retries: int) -> dict[str, Any]:
    contract = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    output_directory = ROOT / contract["inputs"]["global_index_directory"]
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / "NDX_daily.parquet"
    metadata_path = output_directory / "NDX_daily.metadata.json"
    frame, metadata = fetch_ndx(retries=retries)
    temporary = output_path.with_suffix(".parquet.tmp")
    frame.to_parquet(temporary, index=False, engine="pyarrow")
    os.replace(temporary, output_path)
    metadata["path"] = output_path.relative_to(ROOT).as_posix()
    metadata["sha256"] = sha256_file(output_path)
    atomic_json(metadata_path, metadata)
    inventory = audit_existing_inputs(contract)
    report = {
        "schema_version": "1.0.0",
        "candidate_id": contract["protocol"]["candidate_id"],
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "status": "PASS_INPUT_COVERAGE_ONLY_NO_PERFORMANCE_VIEW",
        "performance_or_future_return_computed": False,
        "ndx": metadata,
        "inventory": inventory,
    }
    audit_path = ROOT / contract["outputs"]["input_audit_json"]
    atomic_json(audit_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="补齐QDII折价回归V1输入")
    parser.add_argument("--retries", type=int, default=6)
    args = parser.parse_args()
    report = acquire(max(1, args.retries))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
