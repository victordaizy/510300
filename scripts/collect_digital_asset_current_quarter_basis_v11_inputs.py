"""采集V11现货与当季期货官方输入，不计算基差、方向或绩效。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data" / "raw" / "digital_asset_current_quarter_basis_v11"
RAW_RESPONSE_DIR = OUTPUT_DIR / "source_payloads"
STATUS_PATH = OUTPUT_DIR / "collection_status.json"
SYMBOLS = ("BTCUSDT", "ETHUSDT")
SOURCE_START = pd.Timestamp("2021-08-25 00:00:00")
VISIBLE_START = pd.Timestamp("2021-09-01 00:00:00")
VISIBLE_END = pd.Timestamp("2026-08-14 23:00:00")
HOUR_MILLISECONDS = 60 * 60 * 1000
SPOT_URL = "https://api.binance.com/api/v3/klines"
FUTURES_URL = "https://fapi.binance.com/fapi/v1/continuousKlines"
HEADERS = {"User-Agent": "research-only-codex/1.0"}
CONTEXT_SOURCES = {
    "fx_visible": ROOT
    / "data/raw/digital_asset_spot_vol_scaled_trend_v1/visible_fx_through_2022.parquet",
    "fx_2023_2026": ROOT
    / "data/raw/digital_asset_spot_vol_scaled_trend_v1/sealed_fx_2023_2026.parquet",
    "benchmark_visible": ROOT
    / "data/raw/digital_asset_spot_vol_scaled_trend_v1/visible_H00300_through_2022.parquet",
    "benchmark_2023_2026": ROOT
    / "data/raw/digital_asset_spot_vol_scaled_trend_v1/sealed_H00300_2023_2026.parquet",
}


def _milliseconds(value: pd.Timestamp) -> int:
    return int(value.tz_localize("UTC").timestamp() * 1000)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _get_json(
    session: requests.Session,
    url: str,
    params: dict[str, Any],
    *,
    attempts: int = 5,
) -> Any:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = session.get(url, params=params, timeout=45)
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, list):
                raise ValueError(f"官方接口未返回数组：{payload}")
            return payload
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(0.75 * (attempt + 1))
    raise RuntimeError(f"Binance官方接口请求失败：{url} {params}") from last_error


def _fetch_klines(
    session: requests.Session,
    symbol: str,
    market: str,
) -> list[list[Any]]:
    cursor = _milliseconds(SOURCE_START)
    end = _milliseconds(VISIBLE_END)
    rows: list[list[Any]] = []
    url = SPOT_URL if market == "spot" else FUTURES_URL
    while cursor <= end:
        params: dict[str, Any] = {
            "interval": "1h",
            "startTime": cursor,
            "endTime": end + HOUR_MILLISECONDS - 1,
            "limit": 1000 if market == "spot" else 1500,
        }
        if market == "spot":
            params["symbol"] = symbol
        else:
            params["pair"] = symbol
            params["contractType"] = "CURRENT_QUARTER"
        payload = _get_json(session, url, params)
        if not payload:
            break
        rows.extend(payload)
        last = int(payload[-1][0])
        if last < cursor:
            raise RuntimeError(f"{symbol} {market}小时K线分页游标未前进")
        cursor = last + HOUR_MILLISECONDS
        time.sleep(0.01)
    unique = {int(item[0]): item for item in rows}
    return [unique[key] for key in sorted(unique) if key <= end]


def _load_or_fetch(
    path: Path,
    session: requests.Session,
    symbol: str,
    market: str,
) -> list[list[Any]]:
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise RuntimeError(f"V11缓存不是数组：{path}")
        return payload
    payload = _fetch_klines(session, symbol, market)
    _atomic_json(path, payload)
    return payload


def _kline_frame(
    raw_by_symbol: dict[str, list[list[Any]]],
    *,
    market: str,
) -> pd.DataFrame:
    source = SPOT_URL if market == "spot" else FUTURES_URL
    rows: list[dict[str, Any]] = []
    for symbol, payload in raw_by_symbol.items():
        for item in payload:
            rows.append(
                {
                    "symbol": symbol,
                    "open_time": pd.to_datetime(int(item[0]), unit="ms", utc=True).tz_localize(None),
                    "open": float(item[1]),
                    "high": float(item[2]),
                    "low": float(item[3]),
                    "close": float(item[4]),
                    "volume": float(item[5]),
                    "close_time": pd.to_datetime(int(item[6]), unit="ms", utc=True).tz_localize(None),
                    "quote_volume": float(item[7]),
                    "trade_count": int(item[8]),
                    "taker_buy_base_volume": float(item[9]),
                    "taker_buy_quote_volume": float(item[10]),
                    "market": market,
                    "source": source,
                }
            )
    return pd.DataFrame(rows).sort_values(["open_time", "symbol"]).reset_index(drop=True)


def _validate_grid(
    frame: pd.DataFrame,
    expected: pd.DatetimeIndex,
    *,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    coverage: dict[str, Any] = {}
    missing_by_symbol: dict[str, pd.DatetimeIndex] = {}
    for symbol in SYMBOLS:
        subset = frame.loc[frame["symbol"].eq(symbol), "open_time"]
        observed = pd.DatetimeIndex(subset).sort_values().unique()
        missing_by_symbol[symbol] = expected.difference(observed)
        coverage[symbol] = {
            "row_count": int(len(observed)),
            "first_time": observed.min().isoformat() if len(observed) else None,
            "last_time": observed.max().isoformat() if len(observed) else None,
            "duplicate_time_count": int(subset.duplicated().sum()),
        }
    common_missing = missing_by_symbol[SYMBOLS[0]]
    synchronized = all(
        missing.equals(common_missing) for missing in missing_by_symbol.values()
    )
    missing_decision = [
        timestamp for timestamp in common_missing if timestamp.hour in (0, 23)
    ]
    if (
        not synchronized
        or missing_decision
        or any(item["duplicate_time_count"] for item in coverage.values())
        or any(item["first_time"] != SOURCE_START.isoformat() for item in coverage.values())
        or any(item["last_time"] != VISIBLE_END.isoformat() for item in coverage.values())
    ):
        raise RuntimeError(f"V11 {label}存在非同步缺口、关键时点缺口、重复或端点缺失")
    gap = {
        "all_symbols_share_identical_observed_grid": synchronized,
        "common_missing_hour_count": int(len(common_missing)),
        "common_missing_hours": [timestamp.isoformat() for timestamp in common_missing],
        "missing_signal_23utc_or_execution_00utc_count": int(len(missing_decision)),
        "missing_values_backfilled": False,
    }
    return coverage, gap


def _validate_numeric(frame: pd.DataFrame, *, label: str) -> None:
    numeric_columns = ["open", "high", "low", "close", "volume", "quote_volume"]
    numeric = frame[numeric_columns].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise RuntimeError(f"V11 {label}小时K线含非有限值")
    if not (frame[["open", "high", "low", "close"]] > 0.0).all().all():
        raise RuntimeError(f"V11 {label}小时K线含非正价格")
    if not (frame[["volume", "quote_volume"]] >= 0.0).all().all():
        raise RuntimeError(f"V11 {label}小时K线含负成交量")


def _context_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    for name, path in CONTEXT_SOURCES.items():
        if not path.is_file():
            raise FileNotFoundError(f"V11缺少上下文输入：{name}={path}")
    fx = pd.concat(
        [
            pd.read_parquet(CONTEXT_SOURCES["fx_visible"]),
            pd.read_parquet(CONTEXT_SOURCES["fx_2023_2026"]),
        ],
        ignore_index=True,
    )
    benchmark = pd.concat(
        [
            pd.read_parquet(CONTEXT_SOURCES["benchmark_visible"]),
            pd.read_parquet(CONTEXT_SOURCES["benchmark_2023_2026"]),
        ],
        ignore_index=True,
    )
    for frame in (fx, benchmark):
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").astype(
            "datetime64[ns]"
        )
        frame.sort_values("date", inplace=True)
        frame.drop_duplicates("date", keep="last", inplace=True)
        frame.reset_index(drop=True, inplace=True)
    fx = fx.loc[fx["date"].between(SOURCE_START.normalize() - pd.Timedelta(days=7), VISIBLE_END.normalize())].copy()
    benchmark = benchmark.loc[
        benchmark["date"].between(SOURCE_START.normalize() - pd.Timedelta(days=7), VISIBLE_END.normalize())
    ].copy()
    if fx.empty or benchmark.empty:
        raise RuntimeError("V11汇率或H00300上下文为空")
    if fx["date"].max() != VISIBLE_END.normalize() or benchmark["date"].max() != VISIBLE_END.normalize():
        raise RuntimeError("V11汇率或H00300未覆盖可见期末日")
    fx_numeric = pd.to_numeric(fx["cny_per_usd"], errors="coerce")
    benchmark_numeric = pd.to_numeric(benchmark["close"], errors="coerce")
    if fx_numeric.isna().any() or benchmark_numeric.isna().any():
        raise RuntimeError("V11汇率或H00300含缺失数值")
    if not (fx_numeric > 0.0).all() or not (benchmark_numeric > 0.0).all():
        raise RuntimeError("V11汇率或H00300含非正数值")
    source_receipts = {
        name: {
            "path": path.relative_to(ROOT).as_posix(),
            "bytes": int(path.stat().st_size),
            "sha256": _sha256(path),
        }
        for name, path in CONTEXT_SOURCES.items()
    }
    return fx, benchmark, source_receipts


def _output_receipt(path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": int(path.stat().st_size),
        "sha256": _sha256(path),
    }


def collect() -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_RESPONSE_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update(HEADERS)
    spot_raw: dict[str, list[list[Any]]] = {}
    futures_raw: dict[str, list[list[Any]]] = {}
    for index, symbol in enumerate(SYMBOLS, start=1):
        print(f"采集V11现货 {index}/{len(SYMBOLS)}：{symbol}", flush=True)
        spot_raw[symbol] = _load_or_fetch(
            RAW_RESPONSE_DIR / f"{symbol}_spot_1h.json",
            session,
            symbol,
            "spot",
        )
        print(f"采集V11当季期货 {index}/{len(SYMBOLS)}：{symbol}", flush=True)
        futures_raw[symbol] = _load_or_fetch(
            RAW_RESPONSE_DIR / f"{symbol}_current_quarter_1h.json",
            session,
            symbol,
            "current_quarter",
        )
    spot = _kline_frame(spot_raw, market="spot")
    futures = _kline_frame(futures_raw, market="current_quarter")
    expected = pd.date_range(SOURCE_START, VISIBLE_END, freq="1h")
    spot_coverage, spot_gap = _validate_grid(spot, expected, label="现货")
    futures_coverage, futures_gap = _validate_grid(
        futures, expected, label="当季期货"
    )
    _validate_numeric(spot, label="现货")
    _validate_numeric(futures, label="当季期货")
    fx, benchmark, context_receipts = _context_inputs()
    paths = {
        "spot_1h": OUTPUT_DIR / "spot_1h_2021_2026.parquet",
        "current_quarter_1h": OUTPUT_DIR / "current_quarter_1h_2021_2026.parquet",
        "fx": OUTPUT_DIR / "fx_2021_2026.parquet",
        "benchmark": OUTPUT_DIR / "H00300_2021_2026.parquet",
    }
    _atomic_parquet(paths["spot_1h"], spot)
    _atomic_parquet(paths["current_quarter_1h"], futures)
    _atomic_parquet(paths["fx"], fx)
    _atomic_parquet(paths["benchmark"], benchmark)
    raw_paths = sorted(RAW_RESPONSE_DIR.glob("*.json"))
    status = {
        "schema_version": "1.0.0",
        "status": "PASS_INPUT_ACQUISITION_ONLY",
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "candidate_id": "DIGITAL_ASSET_CURRENT_QUARTER_BASIS_FACTOR_V11",
        "source_start": SOURCE_START.isoformat(),
        "visible_start": VISIBLE_START.isoformat(),
        "visible_end": VISIBLE_END.isoformat(),
        "symbols": list(SYMBOLS),
        "official_endpoints": {"spot": SPOT_URL, "current_quarter": FUTURES_URL},
        "coverage": {
            "expected_hourly_rows_per_symbol": int(len(expected)),
            "spot_hourly": spot_coverage,
            "spot_gap_audit": spot_gap,
            "current_quarter_hourly": futures_coverage,
            "current_quarter_gap_audit": futures_gap,
            "fx_rows": int(len(fx)),
            "fx_first_date": pd.Timestamp(fx["date"].min()).isoformat(),
            "fx_last_date": pd.Timestamp(fx["date"].max()).isoformat(),
            "benchmark_rows": int(len(benchmark)),
            "benchmark_first_date": pd.Timestamp(benchmark["date"].min()).isoformat(),
            "benchmark_last_date": pd.Timestamp(benchmark["date"].max()).isoformat(),
        },
        "outputs": {key: _output_receipt(path) for key, path in paths.items()},
        "context_source_receipts": context_receipts,
        "source_payload_count": int(len(raw_paths)),
        "source_payload_total_bytes": int(sum(path.stat().st_size for path in raw_paths)),
        "source_payload_hashes": {path.name: _sha256(path) for path in raw_paths},
        "basis_value_or_rank_computed": False,
        "long_or_short_direction_computed": False,
        "signal_or_target_computed": False,
        "strategy_or_benchmark_return_computed": False,
        "account_or_authenticated_endpoint_used": False,
        "post_2023_context_values_read_after_parent_lane_freeze": True,
    }
    _atomic_json(STATUS_PATH, status)
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description="采集V11官方输入，不计算基差排序或绩效")
    parser.parse_args()
    payload = collect()
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
