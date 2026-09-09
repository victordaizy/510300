"""采集V13永续合约与实际资金费率输入，不计算基差、方向或绩效。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = (
    ROOT / "data" / "raw" / "digital_asset_current_quarter_signal_perpetual_v13"
)
RAW_RESPONSE_DIR = OUTPUT_DIR / "source_payloads"
STATUS_PATH = OUTPUT_DIR / "collection_status.json"
V11_DIR = ROOT / "data" / "raw" / "digital_asset_current_quarter_basis_v11"
V11_STATUS_PATH = V11_DIR / "collection_status.json"
SYMBOLS = ("BTCUSDT", "ETHUSDT")
SOURCE_START = pd.Timestamp("2021-08-25 00:00:00")
VISIBLE_START = pd.Timestamp("2021-09-01 00:00:00")
VISIBLE_END = pd.Timestamp("2026-08-14 23:00:00")
FUNDING_END = pd.Timestamp("2026-08-14 16:00:00")
HOUR_MILLISECONDS = 60 * 60 * 1000
PERPETUAL_URL = "https://fapi.binance.com/fapi/v1/klines"
FUNDING_URL = "https://fapi.binance.com/fapi/v1/fundingRate"
HEADERS = {"User-Agent": "research-only-codex/1.0"}


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
) -> list[Any]:
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


def _fetch_perpetual(session: requests.Session, symbol: str) -> list[list[Any]]:
    cursor = _milliseconds(SOURCE_START)
    end = _milliseconds(VISIBLE_END)
    rows: list[list[Any]] = []
    while cursor <= end:
        payload = _get_json(
            session,
            PERPETUAL_URL,
            {
                "symbol": symbol,
                "interval": "1h",
                "startTime": cursor,
                "endTime": end + HOUR_MILLISECONDS - 1,
                "limit": 1500,
            },
        )
        if not payload:
            break
        rows.extend(payload)
        last = int(payload[-1][0])
        if last < cursor:
            raise RuntimeError(f"{symbol}永续小时K线分页游标未前进")
        cursor = last + HOUR_MILLISECONDS
        time.sleep(0.01)
    unique = {int(item[0]): item for item in rows}
    return [unique[key] for key in sorted(unique) if key <= end]


def _fetch_funding(
    session: requests.Session,
    symbol: str,
) -> list[dict[str, Any]]:
    cursor = _milliseconds(SOURCE_START)
    end = _milliseconds(FUNDING_END)
    request_end = end + 1000
    rows: list[dict[str, Any]] = []
    while cursor <= request_end:
        payload = _get_json(
            session,
            FUNDING_URL,
            {
                "symbol": symbol,
                "startTime": cursor,
                "endTime": request_end,
                "limit": 1000,
            },
        )
        if not payload:
            break
        rows.extend(payload)
        last = int(payload[-1]["fundingTime"])
        if last < cursor:
            raise RuntimeError(f"{symbol}资金费率分页游标未前进")
        cursor = last + 1
        time.sleep(0.01)
    unique = {int(item["fundingTime"]): item for item in rows}
    return [unique[key] for key in sorted(unique) if key <= request_end]


def _load_or_fetch(
    path: Path,
    fetcher: Callable[[requests.Session, str], list[Any]],
    session: requests.Session,
    symbol: str,
) -> list[Any]:
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise RuntimeError(f"V13缓存不是数组：{path}")
        return payload
    payload = fetcher(session, symbol)
    _atomic_json(path, payload)
    return payload


def _load_or_refresh_funding(
    path: Path,
    session: requests.Session,
    symbol: str,
) -> list[dict[str, Any]]:
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise RuntimeError(f"V13资金费率缓存不是数组：{path}")
        if payload:
            last_time = pd.to_datetime(
                int(payload[-1]["fundingTime"]), unit="ms", utc=True
            ).tz_localize(None)
            if last_time.round("8h") == FUNDING_END:
                return payload
    payload = _fetch_funding(session, symbol)
    _atomic_json(path, payload)
    return payload


def _perpetual_frame(raw_by_symbol: dict[str, list[list[Any]]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for symbol, payload in raw_by_symbol.items():
        for item in payload:
            rows.append(
                {
                    "symbol": symbol,
                    "open_time": pd.to_datetime(
                        int(item[0]), unit="ms", utc=True
                    ).tz_localize(None),
                    "open": float(item[1]),
                    "high": float(item[2]),
                    "low": float(item[3]),
                    "close": float(item[4]),
                    "volume": float(item[5]),
                    "close_time": pd.to_datetime(
                        int(item[6]), unit="ms", utc=True
                    ).tz_localize(None),
                    "quote_volume": float(item[7]),
                    "trade_count": int(item[8]),
                    "taker_buy_base_volume": float(item[9]),
                    "taker_buy_quote_volume": float(item[10]),
                    "market": "usds_margined_perpetual",
                    "source": PERPETUAL_URL,
                }
            )
    return (
        pd.DataFrame(rows)
        .sort_values(["open_time", "symbol"])
        .reset_index(drop=True)
    )


def _funding_frame(
    raw_by_symbol: dict[str, list[dict[str, Any]]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for symbol, payload in raw_by_symbol.items():
        for item in payload:
            source_time = pd.to_datetime(
                int(item["fundingTime"]), unit="ms", utc=True
            ).tz_localize(None)
            scheduled_time = source_time.round("8h")
            mark_price_text = str(item.get("markPrice", ""))
            rows.append(
                {
                    "symbol": symbol,
                    "source_funding_time": source_time,
                    "funding_time": scheduled_time,
                    "funding_time_offset_milliseconds": float(
                        (source_time - scheduled_time).total_seconds() * 1000.0
                    ),
                    "funding_rate": float(item["fundingRate"]),
                    "mark_price": (
                        float(mark_price_text) if mark_price_text else np.nan
                    ),
                    "source": FUNDING_URL,
                }
            )
    return (
        pd.DataFrame(rows)
        .sort_values(["funding_time", "symbol"])
        .reset_index(drop=True)
    )


def _validate_hourly(frame: pd.DataFrame) -> dict[str, Any]:
    expected = pd.date_range(SOURCE_START, VISIBLE_END, freq="1h")
    coverage: dict[str, Any] = {}
    for symbol in SYMBOLS:
        subset = frame.loc[frame["symbol"].eq(symbol), "open_time"]
        observed = pd.DatetimeIndex(subset).sort_values().unique()
        missing = expected.difference(observed)
        extra = observed.difference(expected)
        duplicate_count = int(subset.duplicated().sum())
        coverage[symbol] = {
            "row_count": int(len(observed)),
            "first_time": observed.min().isoformat() if len(observed) else None,
            "last_time": observed.max().isoformat() if len(observed) else None,
            "missing_hour_count": int(len(missing)),
            "extra_hour_count": int(len(extra)),
            "duplicate_time_count": duplicate_count,
            "exact_expected_grid": bool(observed.equals(expected)),
        }
        if not observed.equals(expected) or duplicate_count:
            raise RuntimeError(
                f"V13 {symbol}永续小时网格不完整："
                f"missing={len(missing)} extra={len(extra)} duplicate={duplicate_count}"
            )
    numeric_columns = ["open", "high", "low", "close", "volume", "quote_volume"]
    numeric = frame[numeric_columns].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise RuntimeError("V13永续小时K线含非有限值")
    if not (frame[["open", "high", "low", "close"]] > 0.0).all().all():
        raise RuntimeError("V13永续小时K线含非正价格")
    if not (frame[["volume", "quote_volume"]] >= 0.0).all().all():
        raise RuntimeError("V13永续小时K线含负成交量")
    return {
        "expected_hourly_rows_per_symbol": int(len(expected)),
        "by_symbol": coverage,
    }


def _validate_funding(frame: pd.DataFrame) -> dict[str, Any]:
    expected = pd.date_range(SOURCE_START, FUNDING_END, freq="8h")
    coverage: dict[str, Any] = {}
    for symbol in SYMBOLS:
        subset = frame.loc[frame["symbol"].eq(symbol)]
        observed = pd.DatetimeIndex(subset["funding_time"]).sort_values().unique()
        missing = expected.difference(observed)
        extra = observed.difference(expected)
        duplicate_count = int(subset["funding_time"].duplicated().sum())
        maximum_offset = float(
            subset["funding_time_offset_milliseconds"].abs().max()
        )
        coverage[symbol] = {
            "row_count": int(len(observed)),
            "first_time": observed.min().isoformat() if len(observed) else None,
            "last_time": observed.max().isoformat() if len(observed) else None,
            "missing_settlement_count": int(len(missing)),
            "extra_settlement_count": int(len(extra)),
            "duplicate_time_count": duplicate_count,
            "maximum_absolute_time_offset_milliseconds": maximum_offset,
            "exact_expected_grid": bool(observed.equals(expected)),
        }
        if (
            not observed.equals(expected)
            or duplicate_count
            or maximum_offset > 1000.0
        ):
            raise RuntimeError(
                f"V13 {symbol}资金费率结算网格不完整："
                f"missing={len(missing)} extra={len(extra)} "
                f"duplicate={duplicate_count} offset_ms={maximum_offset}"
            )
    rates = frame["funding_rate"].to_numpy(dtype=float)
    if not np.isfinite(rates).all():
        raise RuntimeError("V13资金费率含非有限值")
    return {
        "expected_funding_rows_per_symbol": int(len(expected)),
        "by_symbol": coverage,
    }


def _reused_v11_inputs() -> tuple[dict[str, Path], dict[str, Any]]:
    if not V11_STATUS_PATH.is_file():
        raise FileNotFoundError(f"V13缺少V11采集回执：{V11_STATUS_PATH}")
    status = json.loads(V11_STATUS_PATH.read_text(encoding="utf-8"))
    forbidden = (
        "basis_value_or_rank_computed",
        "long_or_short_direction_computed",
        "signal_or_target_computed",
        "strategy_or_benchmark_return_computed",
        "account_or_authenticated_endpoint_used",
    )
    if (
        status.get("status") != "PASS_INPUT_ACQUISITION_ONLY"
        or status.get("candidate_id")
        != "DIGITAL_ASSET_CURRENT_QUARTER_BASIS_FACTOR_V11"
        or any(bool(status.get(key)) for key in forbidden)
    ):
        raise RuntimeError("V13复用的V11原始输入回执不合格")
    paths = {
        "spot_1h": V11_DIR / "spot_1h_2021_2026.parquet",
        "current_quarter_1h": V11_DIR / "current_quarter_1h_2021_2026.parquet",
        "fx": V11_DIR / "fx_2021_2026.parquet",
        "benchmark": V11_DIR / "H00300_2021_2026.parquet",
    }
    for key, path in paths.items():
        receipt = status.get("outputs", {}).get(key, {})
        if not path.is_file() or receipt.get("sha256") != _sha256(path):
            raise RuntimeError(f"V13复用输入与V11采集回执不一致：{key}")
    return paths, status


def _receipt(path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": int(path.stat().st_size),
        "sha256": _sha256(path),
    }


def collect() -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_RESPONSE_DIR.mkdir(parents=True, exist_ok=True)
    reused_paths, reused_status = _reused_v11_inputs()
    session = requests.Session()
    session.headers.update(HEADERS)
    perpetual_raw: dict[str, list[list[Any]]] = {}
    funding_raw: dict[str, list[dict[str, Any]]] = {}
    for index, symbol in enumerate(SYMBOLS, start=1):
        print(f"采集V13永续K线 {index}/{len(SYMBOLS)}：{symbol}", flush=True)
        perpetual_raw[symbol] = _load_or_fetch(
            RAW_RESPONSE_DIR / f"{symbol}_perpetual_1h.json",
            _fetch_perpetual,
            session,
            symbol,
        )
        print(f"采集V13资金费率 {index}/{len(SYMBOLS)}：{symbol}", flush=True)
        funding_raw[symbol] = _load_or_refresh_funding(
            RAW_RESPONSE_DIR / f"{symbol}_funding.json",
            session,
            symbol,
        )
    perpetual = _perpetual_frame(perpetual_raw)
    funding = _funding_frame(funding_raw)
    hourly_coverage = _validate_hourly(perpetual)
    funding_coverage = _validate_funding(funding)
    output_paths = {
        "perpetual_1h": OUTPUT_DIR / "perpetual_1h_2021_2026.parquet",
        "funding_rates": OUTPUT_DIR / "funding_rates_2021_2026.parquet",
    }
    _atomic_parquet(output_paths["perpetual_1h"], perpetual)
    _atomic_parquet(output_paths["funding_rates"], funding)
    raw_paths = sorted(RAW_RESPONSE_DIR.glob("*.json"))
    outputs = {
        **{key: _receipt(path) for key, path in output_paths.items()},
        **{key: _receipt(path) for key, path in reused_paths.items()},
    }
    status = {
        "schema_version": "1.0.0",
        "status": "PASS_INPUT_ACQUISITION_ONLY",
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "candidate_id": "DIGITAL_ASSET_CURRENT_QUARTER_SIGNAL_PERPETUAL_FACTOR_V13",
        "source_start": SOURCE_START.isoformat(),
        "visible_start": VISIBLE_START.isoformat(),
        "visible_end": VISIBLE_END.isoformat(),
        "funding_end": FUNDING_END.isoformat(),
        "symbols": list(SYMBOLS),
        "official_endpoints": {
            "perpetual_klines": PERPETUAL_URL,
            "funding": FUNDING_URL,
        },
        "coverage": {
            "perpetual_hourly": hourly_coverage,
            "funding": funding_coverage,
            "reused_v11_status": reused_status["status"],
            "reused_v11_status_sha256": _sha256(V11_STATUS_PATH),
        },
        "outputs": outputs,
        "source_payload_count": int(len(raw_paths)),
        "source_payload_total_bytes": int(
            sum(path.stat().st_size for path in raw_paths)
        ),
        "source_payload_hashes": {
            path.name: _sha256(path) for path in raw_paths
        },
        "basis_value_or_rank_computed": False,
        "long_or_short_direction_computed": False,
        "signal_or_target_computed": False,
        "strategy_or_benchmark_return_computed": False,
        "account_or_authenticated_endpoint_used": False,
        "post_2023_values_read_after_v23_lane_freeze": True,
    }
    _atomic_json(STATUS_PATH, status)
    return status


def main() -> int:
    parser = argparse.ArgumentParser(
        description="采集V13官方永续与资金费率输入，不计算基差排序或绩效"
    )
    parser.parse_args()
    payload = collect()
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
