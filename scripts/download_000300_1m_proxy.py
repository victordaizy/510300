"""通过已授权通用分钟入口采集000300一分钟对照数据并做日线交叉审计。"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.download_510300_1m_tushare import (
    DownloadConfig,
    RateLimiter,
    atomic_parquet,
    iter_chunks,
    request_chunk_with_retry,
)


ENV_FILE = PROJECT_ROOT / ".env"
SETTINGS_FILE = PROJECT_ROOT / "config" / "settings.yaml"
DAILY_FILE = PROJECT_ROOT / "data" / "raw" / "market" / "000300_daily_raw.parquet"
OUTPUT_FILE = PROJECT_ROOT / "data" / "raw" / "market" / "000300_1m_tushare_raw.parquet"
CACHE_DIR = PROJECT_ROOT / "data" / "raw" / "market" / ".000300_1m_proxy_chunks"
METADATA_FILE = OUTPUT_FILE.with_suffix(".metadata.json")
REPORT_FILE = PROJECT_ROOT / "reports" / "data_quality" / "000300_1m_proxy_quality.json"
TIMEZONE = "Asia/Shanghai"


def _atomic_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _audit(data: pd.DataFrame, daily: pd.DataFrame) -> dict[str, object]:
    errors: list[dict[str, object]] = []
    warnings: list[dict[str, object]] = [
        {
            "code": "GENERIC_PROXY_ENDPOINT",
            "detail": "数据由代理stk_mins返回000300.SH，不等同于已获得官方idx_mins权限。",
        },
        {
            "code": "TIMESTAMP_INTERVAL_SEMANTICS_PENDING",
            "detail": "标签集合通过不等同于bar起止语义已人工确认。",
        },
    ]
    if data.empty:
        return {"status": "FAIL", "errors": [{"code": "EMPTY_DATA"}], "warnings": warnings}
    if data["trade_time"].duplicated().any():
        errors.append({"code": "DUPLICATE_TIMESTAMPS"})
    counts = data.groupby(data["trade_time"].dt.normalize()).size()
    bad_counts = counts[counts != 241]
    if not bad_counts.empty:
        errors.append(
            {
                "code": "UNEXPECTED_BARS_PER_DAY",
                "day_count": int(len(bad_counts)),
                "examples": {str(day.date()): int(value) for day, value in bad_counts.head(10).items()},
            }
        )
    reference = daily.copy()
    reference["date"] = pd.to_datetime(reference["date"]).dt.normalize()
    actual_dates = pd.DatetimeIndex(data["trade_time"].dt.normalize().drop_duplicates())
    expected_dates = pd.DatetimeIndex(reference["date"].drop_duplicates())
    missing_dates = expected_dates.difference(actual_dates)
    extra_dates = actual_dates.difference(expected_dates)
    if len(missing_dates):
        errors.append(
            {
                "code": "MISSING_DAILY_REFERENCE_DATES",
                "day_count": int(len(missing_dates)),
                "examples": [str(day.date()) for day in missing_dates[:10]],
            }
        )
    if len(extra_dates):
        errors.append({"code": "EXTRA_TRADING_DATES", "day_count": int(len(extra_dates))})
    minute_daily = data.assign(date=data["trade_time"].dt.normalize()).groupby("date", as_index=False).agg(
        minute_open=("open", "first"),
        minute_high=("high", "max"),
        minute_low=("low", "min"),
        minute_close=("close", "last"),
    )
    comparison = minute_daily.merge(
        reference[["date", "open", "high", "low", "close"]],
        on="date", how="inner", validate="one_to_one",
    )
    invalid_close = ~np.isfinite(data["close"]) | (data["close"] <= 0)
    if invalid_close.any():
        errors.append({"code": "INVALID_RESEARCH_CLOSE", "row_count": int(invalid_close.sum())})
    close_mismatch = ~np.isclose(
        comparison["minute_close"], comparison["close"], atol=0.002, rtol=0.0
    )
    if close_mismatch.any():
        errors.append(
            {
                "code": "DAILY_CLOSE_MISMATCH_OVER_0_002",
                "day_count": int(close_mismatch.sum()),
                "examples": [
                    str(day.date()) for day in comparison.loc[close_mismatch, "date"].head(10)
                ],
            }
        )
    non_close_mismatch = pd.Series(False, index=comparison.index)
    for minute_column, daily_column in [
        ("minute_open", "open"), ("minute_high", "high"), ("minute_low", "low")
    ]:
        non_close_mismatch |= ~np.isclose(
            comparison[minute_column], comparison[daily_column], atol=0.002, rtol=0.0
        )
    nonpositive_non_close = (data[["open", "high", "low"]] <= 0).any(axis=1)
    if non_close_mismatch.any() or nonpositive_non_close.any():
        warnings.append(
            {
                "code": "NON_RESEARCH_OHLC_ANOMALY_PRESERVED",
                "daily_mismatch_day_count": int(non_close_mismatch.sum()),
                "nonpositive_row_count": int(nonpositive_non_close.sum()),
                "impact": "本研究指数合同只使用close；原始异常不修改。",
            }
        )
    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "warnings": warnings,
        "row_count": int(len(data)),
        "trading_day_count": int(len(actual_dates)),
        "first_trade_time": data["trade_time"].min().isoformat(),
        "last_trade_time": data["trade_time"].max().isoformat(),
        "daily_ohlc_compared_days": int(len(comparison)),
    }


def main() -> int:
    load_dotenv(ENV_FILE)
    token = os.environ.get("TUSHARE_PROXY_TOKEN", "").strip()
    if not token:
        raise ValueError("缺少TUSHARE_PROXY_TOKEN")
    endpoint = os.environ.get("TUSHARE_PROXY_URL", "https://tt.xiaodefa.cn").rstrip("/")
    with SETTINGS_FILE.open("r", encoding="utf-8") as file:
        settings = yaml.safe_load(file)
    start = pd.Timestamp(settings["project"]["start_date"])
    end_exclusive = pd.Timestamp(settings["project"]["end_date"]) + pd.Timedelta(days=1)
    config = DownloadConfig(
        symbol="000300.SH",
        start=start,
        end_exclusive=end_exclusive,
        endpoint=endpoint,
        api_name="stk_mins",
        frequency="1min",
        chunk_days=31,
        minimum_interval_seconds=0.55,
        timeout_seconds=90.0,
        maximum_retries=6,
        force=False,
    )
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    limiter = RateLimiter(config.minimum_interval_seconds)
    session = requests.Session()
    frames: list[pd.DataFrame] = []
    downloaded_chunks = 0
    reused_chunks = 0
    for left, right in iter_chunks(start, end_exclusive, config.chunk_days):
        chunk_path = CACHE_DIR / f"{left:%Y%m%d}__{right:%Y%m%d}.parquet"
        if chunk_path.exists():
            frame = pd.read_parquet(chunk_path)
            reused_chunks += 1
        else:
            frame = request_chunk_with_retry(session, token, config, limiter, left, right)
            atomic_parquet(frame, chunk_path)
            downloaded_chunks += 1
        frames.append(frame)
        print(f"000300 {left.date()} 至 {(right - pd.Timedelta(days=1)).date()}：{len(frame)}行", flush=True)
    data = pd.concat(frames, ignore_index=True).sort_values("trade_time")
    data = data.drop_duplicates("trade_time", keep="last").reset_index(drop=True)
    data["trade_time"] = pd.to_datetime(data["trade_time"])
    daily = pd.read_parquet(DAILY_FILE)
    daily = daily.loc[pd.to_datetime(daily["date"]).between(start, end_exclusive, inclusive="left")]
    report = _audit(data, daily)
    _atomic_json(report, REPORT_FILE)
    if report["status"] != "PASS":
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 2
    atomic_parquet(data, OUTPUT_FILE)
    metadata = {
        "status": "PASS",
        "symbol": "000300.SH",
        "source": "Tushare-compatible proxy stk_mins generic endpoint",
        "official_idx_mins_permission": False,
        "retrieved_at": datetime.now(ZoneInfo(TIMEZONE)).isoformat(),
        "downloaded_chunks": downloaded_chunks,
        "reused_chunks": reused_chunks,
        "row_count": int(len(data)),
        "sha256": _sha256(OUTPUT_FILE),
    }
    _atomic_json(metadata, METADATA_FILE)
    print(json.dumps({**metadata, "quality_report": str(REPORT_FILE)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
