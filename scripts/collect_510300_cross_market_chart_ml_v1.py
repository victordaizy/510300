"""采集并审计跨市场图形机器学习V1的海外宽基日线。"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

import akshare as ak
import numpy as np
import pandas as pd
import requests
import yaml
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = ROOT / "config" / "510300_cross_market_chart_ml_v1.yaml"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) CodexResearch/1.0"


def _load_config() -> dict[str, Any]:
    config = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("跨市场配置必须是YAML对象")
    return config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _request_json(url: str, retries: int = 7) -> dict[str, Any]:
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
                raise RuntimeError(f"Yahoo返回错误：{chart_error}")
            return payload
        except Exception as exc:  # 网络失败必须保留最后一次明确错误
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(min(2**attempt, 10))
    assert last_error is not None
    raise last_error


def _epoch(date_text: str) -> int:
    timestamp = pd.Timestamp(date_text, tz="UTC")
    return int(timestamp.timestamp())


def _fetch_yahoo(symbol: str, contract: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    base = str(contract["yahoo_chart_base_url"]).rstrip("/")
    period1 = _epoch(str(contract["start_date"]))
    period2 = _epoch(str(pd.Timestamp(contract["cutoff_date"]) + pd.Timedelta(days=1)))
    url = (
        f"{base}/{quote(symbol, safe='')}?period1={period1}&period2={period2}"
        f"&interval={contract['interval']}&events=div%2Csplits"
    )
    payload = _request_json(url)
    result = payload["chart"]["result"][0]
    timestamps = result.get("timestamp") or []
    quote_block = result["indicators"]["quote"][0]
    adjusted_block = result["indicators"].get("adjclose", [{}])[0]
    timezone_name = result.get("meta", {}).get("exchangeTimezoneName", "UTC")
    dates = (
        pd.to_datetime(timestamps, unit="s", utc=True)
        .tz_convert(timezone_name)
        .date
    )
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "open": quote_block.get("open"),
            "high": quote_block.get("high"),
            "low": quote_block.get("low"),
            "close": quote_block.get("close"),
            "adj_close": adjusted_block.get("adjclose", quote_block.get("close")),
            "volume": quote_block.get("volume"),
        }
    )
    frame["symbol"] = symbol
    frame["source"] = "yahoo.finance.chart.v8"
    frame = frame.dropna(subset=["date", "open", "high", "low", "close", "adj_close"])
    frame = frame.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    metadata = {
        "symbol": symbol,
        "url": url,
        "exchange_timezone": timezone_name,
        "currency": result.get("meta", {}).get("currency"),
        "exchange_name": result.get("meta", {}).get("exchangeName"),
        "instrument_type": result.get("meta", {}).get("instrumentType"),
        "retrieved_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
    }
    return frame, metadata


def _validate_market(frame: pd.DataFrame, minimum_rows: int) -> dict[str, Any]:
    numeric = ["open", "high", "low", "close", "adj_close"]
    checks = {
        "minimum_rows": bool(len(frame) >= minimum_rows),
        "date_unique": bool(frame["date"].is_unique),
        "date_increasing": bool(frame["date"].is_monotonic_increasing),
        "required_prices_finite": bool(np.isfinite(frame[numeric].to_numpy(float)).all()),
        "required_prices_positive": bool(frame[numeric].gt(0.0).all().all()),
        "high_not_below_open_or_close": bool(
            frame["high"].ge(frame[["open", "close"]].max(axis=1) - 1e-9).all()
        ),
        "low_not_above_open_or_close": bool(
            frame["low"].le(frame[["open", "close"]].min(axis=1) + 1e-9).all()
        ),
    }
    return {
        "rows": int(len(frame)),
        "start_date": str(frame["date"].min().date()) if not frame.empty else None,
        "end_date": str(frame["date"].max().date()) if not frame.empty else None,
        "checks": checks,
        "passed": bool(all(checks.values())),
    }


def _fetch_sina(name: str, retries: int = 4) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            frame = ak.index_global_hist_sina(symbol=name).copy()
            frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
            frame["close"] = pd.to_numeric(frame["close"], errors="raise")
            return frame[["date", "close"]].dropna().drop_duplicates("date", keep="last")
        except Exception as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(min(2**attempt, 8))
    assert last_error is not None
    raise last_error


def _cross_check(
    yahoo: pd.DataFrame,
    sina_name: str,
    contract: dict[str, Any],
) -> dict[str, Any]:
    try:
        sina = _fetch_sina(sina_name).rename(columns={"close": "sina_close"})
        merged = yahoo[["date", "close"]].rename(columns={"close": "yahoo_close"}).merge(
            sina, on="date", how="inner"
        )
        relative_error = (merged["yahoo_close"] / merged["sina_close"] - 1.0).abs()
        common_dates = int(len(merged))
        median_error = float(relative_error.median()) if common_dates else np.nan
        p99_error = float(relative_error.quantile(0.99)) if common_dates else np.nan
        passed = bool(
            common_dates >= int(contract["cross_check_minimum_common_dates"])
            and median_error <= float(contract["cross_check_median_relative_error_maximum"])
            and p99_error <= float(contract["cross_check_p99_relative_error_maximum"])
        )
        return {
            "sina_name": sina_name,
            "status": "PASS" if passed else "FAIL",
            "common_dates": common_dates,
            "median_relative_error": median_error,
            "p99_relative_error": p99_error,
            "passed": passed,
        }
    except Exception as exc:
        return {
            "sina_name": sina_name,
            "status": "ERROR",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "passed": False,
        }


def collect(*, workers: int = 4) -> dict[str, Any]:
    config = _load_config()
    contract = config["data_contracts"]
    output_dir = ROOT / contract["raw_directory"]
    output_dir.mkdir(parents=True, exist_ok=True)
    symbols = contract["training_symbols"] + contract["external_gate_symbols"]
    frames: dict[str, pd.DataFrame] = {}
    market_audits: dict[str, Any] = {}

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_fetch_yahoo, item["symbol"], contract): item
            for item in symbols
        }
        for future in as_completed(futures):
            item = futures[future]
            symbol = item["symbol"]
            try:
                frame, metadata = future.result()
                path = output_dir / f"{symbol.replace('^', '')}_daily.parquet"
                frame.to_parquet(path, index=False)
                audit = _validate_market(frame, int(config["sample_design"]["minimum_rows_per_market"]))
                audit.update(metadata)
                audit["name"] = item["name"]
                audit["file"] = str(path.relative_to(ROOT)).replace("\\", "/")
                audit["sha256"] = _sha256(path)
                market_audits[symbol] = audit
                frames[symbol] = frame
                print(f"已采集 {symbol}：{len(frame)} 行，{audit['start_date']} 至 {audit['end_date']}")
            except Exception as exc:
                market_audits[symbol] = {
                    "name": item["name"],
                    "passed": False,
                    "status": "ERROR",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                print(f"采集失败 {symbol}：{type(exc).__name__}: {exc}")

    cross_checks: dict[str, Any] = {}
    for item in contract["independent_cross_checks"]:
        symbol = item["symbol"]
        if symbol not in frames:
            cross_checks[symbol] = {
                "sina_name": item["sina_name"],
                "status": "BLOCKED_MISSING_YAHOO_DATA",
                "passed": False,
            }
            continue
        cross_checks[symbol] = _cross_check(frames[symbol], item["sina_name"], contract)
        print(f"独立核对 {symbol}：{cross_checks[symbol]['status']}")

    successful_cross_checks = sum(bool(item.get("passed")) for item in cross_checks.values())
    all_markets_pass = len(market_audits) == len(symbols) and all(
        bool(item.get("passed")) for item in market_audits.values()
    )
    overall_pass = bool(
        all_markets_pass
        and successful_cross_checks >= int(contract["minimum_successful_cross_checks"])
    )
    report = {
        "project_id": config["protocol"]["project_id"],
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "status": "PASS" if overall_pass else "NO_VIEW_DATA_GATE_FAILED",
        "overall_pass": overall_pass,
        "market_count_expected": len(symbols),
        "market_count_passed": sum(bool(item.get("passed")) for item in market_audits.values()),
        "successful_cross_checks": successful_cross_checks,
        "minimum_successful_cross_checks": int(contract["minimum_successful_cross_checks"]),
        "markets": market_audits,
        "independent_cross_checks": cross_checks,
    }
    audit_path = ROOT / contract["audit_file"]
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"数据闸门：{report['status']}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="采集跨市场图形机器学习V1数据")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    report = collect(workers=max(1, args.workers))
    return 0 if report["overall_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
