"""下载美国上市中国ETF、SPY与VIX日线，仅形成输入，不计算510300收益。"""

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


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_CONTRACT = (
    ROOT / "config" / "510300_us_china_overnight_binary_screen_v1_candidates.yaml"
)
OUTPUT_DIRECTORY = ROOT / "data" / "raw" / "us_china_overnight_v1"
AUDIT_PATH = ROOT / "reports" / "data_quality" / "us_china_overnight_inputs_v1.json"
YAHOO_BASE = "https://query1.finance.yahoo.com/v8/finance/chart"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) CodexResearch/1.0"
START_DATE = "2013-01-01"
CUTOFF_DATE = "2026-08-25"
SYMBOLS = ["ASHR", "FXI", "MCHI", "KWEB", "SPY", "^VIX"]
FILE_NAMES = {
    "ASHR": "ASHR_daily.parquet",
    "FXI": "FXI_daily.parquet",
    "MCHI": "MCHI_daily.parquet",
    "KWEB": "KWEB_daily.parquet",
    "SPY": "SPY_daily.parquet",
    "^VIX": "VIX_daily.parquet",
}


def sha256_file(path: Path) -> str:
    """流式计算文件SHA-256。"""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    """原子写入JSON。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def request_json(url: str, retries: int) -> dict[str, Any]:
    """带有限重试读取公开图表接口。"""

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


def fetch_symbol(symbol: str, retries: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    """读取一个标的并完成基础OHLC与日期核对。"""

    start_epoch = int(pd.Timestamp(START_DATE, tz="UTC").timestamp())
    end_epoch = int(
        (pd.Timestamp(CUTOFF_DATE, tz="UTC") + pd.Timedelta(days=1)).timestamp()
    )
    url = (
        f"{YAHOO_BASE}/{quote(symbol, safe='')}?period1={start_epoch}"
        f"&period2={end_epoch}&interval=1d&events=div%2Csplits"
    )
    payload = request_json(url, retries)
    result = payload["chart"]["result"][0]
    timestamps = result.get("timestamp") or []
    quote_block = result["indicators"]["quote"][0]
    adjusted_block = result["indicators"].get("adjclose", [{}])[0]
    timezone_name = result.get("meta", {}).get(
        "exchangeTimezoneName", "America/New_York"
    )
    local_dates = pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(
        timezone_name
    ).date
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(local_dates),
            "open": quote_block.get("open"),
            "high": quote_block.get("high"),
            "low": quote_block.get("low"),
            "close": quote_block.get("close"),
            "adj_close": adjusted_block.get("adjclose", quote_block.get("close")),
            "volume": quote_block.get("volume"),
            "symbol": symbol,
            "source": "yahoo.finance.chart.v8",
        }
    )
    frame.dropna(
        subset=["date", "open", "high", "low", "close", "adj_close"],
        inplace=True,
    )
    frame.sort_values("date", kind="mergesort", inplace=True)
    frame.drop_duplicates("date", keep="last", inplace=True)
    frame.reset_index(drop=True, inplace=True)
    numeric = ["open", "high", "low", "close", "adj_close"]
    minimum_rows = 2500 if symbol in {"ASHR", "KWEB"} else 3000
    timezone_valid = (
        timezone_name == "America/New_York"
        if symbol != "^VIX"
        else timezone_name in {"America/Chicago", "America/New_York"}
    )
    checks = {
        "minimum_rows": bool(len(frame) >= minimum_rows),
        "date_unique": bool(frame["date"].is_unique),
        "date_increasing": bool(frame["date"].is_monotonic_increasing),
        "prices_finite": bool(np.isfinite(frame[numeric].to_numpy(float)).all()),
        "prices_positive": bool(frame[numeric].gt(0.0).all().all()),
        "high_valid": bool(
            frame["high"].ge(frame[["open", "close"]].max(axis=1) - 1e-9).all()
        ),
        "low_valid": bool(
            frame["low"].le(frame[["open", "close"]].min(axis=1) + 1e-9).all()
        ),
        "cutoff_not_exceeded": bool(
            frame["date"].max() <= pd.Timestamp(CUTOFF_DATE)
        ),
        "exchange_timezone_valid": bool(timezone_valid),
    }
    if not all(checks.values()):
        raise RuntimeError(f"{symbol}输入核对失败：{checks}")
    metadata = {
        "symbol": symbol,
        "url": url,
        "exchange_timezone": timezone_name,
        "currency": result.get("meta", {}).get("currency"),
        "exchange_name": result.get("meta", {}).get("exchangeName"),
        "instrument_type": result.get("meta", {}).get("instrumentType"),
        "rows": int(len(frame)),
        "first_date": frame["date"].min().date().isoformat(),
        "last_date": frame["date"].max().date().isoformat(),
        "checks": checks,
    }
    return frame, metadata


def acquire(retries: int) -> dict[str, Any]:
    """下载并原子保存六个固定输入。"""

    if not CANDIDATE_CONTRACT.exists():
        raise FileNotFoundError("候选规则合同不存在")
    candidate_sha256 = sha256_file(CANDIDATE_CONTRACT)
    if candidate_sha256 != "836f662330d489007bbd35c700629643d5f3e5f5777dc9777a138ac587dff160":
        raise ValueError("候选规则合同在数据下载前发生漂移")
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, Any] = {}
    for symbol in SYMBOLS:
        frame, metadata = fetch_symbol(symbol, retries)
        output_path = OUTPUT_DIRECTORY / FILE_NAMES[symbol]
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, output_path)
        relative = output_path.relative_to(ROOT).as_posix()
        artifacts[symbol] = {
            **metadata,
            "file": relative,
            "sha256": sha256_file(output_path),
        }
    report = {
        "status": "PASS_INPUT_ACQUISITION_ONLY_NO_510300_PERFORMANCE_VIEW",
        "study_id": "510300_US_CHINA_OVERNIGHT_BINARY_SCREEN_V1",
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "candidate_contract_sha256": candidate_sha256,
        "start_date": START_DATE,
        "cutoff_date": CUTOFF_DATE,
        "symbol_count": len(SYMBOLS),
        "artifacts": artifacts,
        "attempt_history": [
            {
                "attempt": 1,
                "status": "PROGRAM_FAILED",
                "failure": "VIX_EXCHANGE_TIMEZONE_GUARD_TOO_NARROW",
                "observed_timezone": "America/Chicago",
                "performance_computed": False,
            },
            {
                "attempt": 2,
                "status": "PASS_INPUT_ACQUISITION_ONLY_NO_510300_PERFORMANCE_VIEW",
                "performance_computed": False,
            },
        ],
        "target_510300_or_benchmark_read": False,
        "future_return_or_performance_computed": False,
    }
    atomic_json(report, AUDIT_PATH)
    return report


def main() -> int:
    """命令行入口。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retries", type=int, default=6)
    args = parser.parse_args()
    report = acquire(max(1, args.retries))
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
