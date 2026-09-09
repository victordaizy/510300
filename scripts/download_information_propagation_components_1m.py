"""可续传下载历史Top50并集成分股的五年1分钟行情。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yaml
from dotenv import load_dotenv

from scripts.download_510300_1m_tushare import (
    DownloadConfig,
    RateLimiter,
    atomic_parquet,
    iter_chunks,
    load_cached_chunk,
    request_chunk_with_retry,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = PROJECT_ROOT / "config" / "information_propagation_alpha.yaml"
ENV_FILE = PROJECT_ROOT / ".env"
UNIVERSE_FILE = (
    PROJECT_ROOT / "data" / "raw" / "constituents" /
    "information_propagation_top50_universe.csv"
)
DAILY_REFERENCE_FILE = (
    PROJECT_ROOT / "data" / "raw" / "constituents" /
    "000300_constituent_daily.parquet"
)
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "constituents" / "minute_1m"
CACHE_DIR = (
    PROJECT_ROOT / "data" / "raw" / "constituents" /
    ".information_propagation_component_1m_chunks"
)
REPORT_FILE = (
    PROJECT_ROOT / "reports" / "data_quality" /
    "information_propagation_component_1m_collection.json"
)
DEFAULT_ENDPOINT = "https://tt.xiaodefa.cn"
TIMEZONE = "Asia/Shanghai"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="下载信息传播研究所需成分股1分钟行情")
    parser.add_argument("--limit", type=int, help="仅处理清单前N只，用于小规模验证")
    parser.add_argument("--symbols", nargs="+", help="仅处理指定股票代码")
    parser.add_argument("--chunk-days", type=int, default=31)
    parser.add_argument("--minimum-interval-seconds", type=float, default=0.55)
    parser.add_argument("--timeout-seconds", type=float, default=90.0)
    parser.add_argument("--maximum-retries", type=int, default=6)
    parser.add_argument("--force", action="store_true", help="忽略缓存与已完成文件")
    return parser.parse_args()


def _safe_symbol(symbol: str) -> str:
    return symbol.replace(".", "_")


def _chunk_path(symbol: str, left: pd.Timestamp, right: pd.Timestamp) -> Path:
    return CACHE_DIR / _safe_symbol(symbol) / f"{left:%Y%m%d}__{right:%Y%m%d}.parquet"


def _output_path(symbol: str) -> Path:
    return OUTPUT_DIR / f"{_safe_symbol(symbol)}.parquet"


def _metadata_path(symbol: str) -> Path:
    return OUTPUT_DIR / f"{_safe_symbol(symbol)}.metadata.json"


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
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


def _load_project_config() -> dict[str, Any]:
    with CONFIG_FILE.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _load_universe(arguments: argparse.Namespace) -> pd.DataFrame:
    if not UNIVERSE_FILE.exists():
        raise FileNotFoundError("缺少历史Top50并集清单，请先运行build_information_propagation_universe")
    universe = pd.read_csv(UNIVERSE_FILE, dtype={"con_code": str})
    if arguments.symbols:
        requested = set(arguments.symbols)
        unavailable = sorted(requested.difference(universe["con_code"]))
        if unavailable:
            raise ValueError(f"指定代码不在历史Top50并集中：{unavailable}")
        universe = universe.loc[universe["con_code"].isin(requested)]
    if arguments.limit is not None:
        if arguments.limit < 1:
            raise ValueError("--limit 必须为正整数")
        universe = universe.head(arguments.limit)
    return universe.reset_index(drop=True)


def _validate_parameters(arguments: argparse.Namespace) -> None:
    if not 1 <= arguments.chunk_days <= 31:
        raise ValueError("--chunk-days 必须位于1至31")
    if arguments.minimum_interval_seconds < 0.5:
        raise ValueError("请求间隔不能低于0.5秒")
    if arguments.timeout_seconds <= 0 or arguments.maximum_retries < 1:
        raise ValueError("超时和重试次数必须为正数")


def _canonicalize(data: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame(
            columns=["con_code", "trade_time", "open", "high", "low", "close", "volume", "amount"]
        )
    output = data.rename(columns={"ts_code": "con_code", "vol": "volume"}).copy()
    columns = ["con_code", "trade_time", "open", "high", "low", "close", "volume", "amount"]
    output = output[columns]
    output["trade_time"] = pd.to_datetime(output["trade_time"], errors="coerce")
    numeric_columns = ["open", "high", "low", "close", "volume", "amount"]
    output[numeric_columns] = output[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if output["trade_time"].isna().any() or output[numeric_columns].isna().any().any():
        raise ValueError(f"{symbol}合并分块后存在无法解析的时间或数值")
    if output["con_code"].drop_duplicates().tolist() != [symbol]:
        raise ValueError(f"{symbol}缓存中出现其他证券代码")
    return output.sort_values("trade_time").drop_duplicates("trade_time", keep="last").reset_index(drop=True)


def audit_symbol(
    data: pd.DataFrame,
    daily_reference: pd.DataFrame,
    symbol: str,
    start: pd.Timestamp,
    end_inclusive: pd.Timestamp,
) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    if data.empty:
        return {"status": "FAIL", "errors": [{"code": "EMPTY_DATA"}], "warnings": []}
    if data["trade_time"].duplicated().any():
        errors.append({"code": "DUPLICATE_TIMESTAMPS"})
    counts = data.groupby(data["trade_time"].dt.normalize()).size()
    bad_counts = counts[counts != 241]
    if not bad_counts.empty:
        errors.append(
            {
                "code": "UNEXPECTED_BARS_PER_DAY",
                "day_count": int(len(bad_counts)),
                "examples": {day.date().isoformat(): int(value) for day, value in bad_counts.head(10).items()},
            }
        )
    reference = daily_reference.loc[
        (daily_reference["con_code"] == symbol)
        & daily_reference["date"].between(start, end_inclusive)
        & ~daily_reference["is_suspended"].fillna(False)
    ].copy()
    expected_dates = pd.DatetimeIndex(reference["date"].drop_duplicates())
    actual_dates = pd.DatetimeIndex(data["trade_time"].dt.normalize().drop_duplicates())
    missing_dates = expected_dates.difference(actual_dates)
    if len(missing_dates):
        errors.append(
            {
                "code": "MISSING_ACTIVE_TRADING_DAYS",
                "day_count": int(len(missing_dates)),
                "examples": [value.date().isoformat() for value in missing_dates[:10]],
            }
        )
    minute_daily = data.assign(date=data["trade_time"].dt.normalize()).groupby("date", as_index=False).agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last")
    )
    comparison = minute_daily.merge(
        reference[["date", "raw_open", "raw_high", "raw_low", "raw_close"]],
        on="date", how="inner", validate="one_to_one",
    )
    mismatch = pd.Series(False, index=comparison.index)
    for minute_column, daily_column in [
        ("open", "raw_open"), ("high", "raw_high"),
        ("low", "raw_low"), ("close", "raw_close"),
    ]:
        mismatch |= ~np.isclose(
            comparison[minute_column], comparison[daily_column],
            atol=0.011, rtol=0.0, equal_nan=False,
        )
    if mismatch.any():
        warnings.append(
            {
                "code": "CROSS_SOURCE_DAILY_OHLC_MISMATCH_OVER_ONE_TICK",
                "day_count": int(mismatch.sum()),
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
        "expected_active_trading_days": int(len(expected_dates)),
    }


def download_symbol(
    symbol: str,
    config: DownloadConfig,
    session: requests.Session,
    token: str,
    limiter: RateLimiter,
    force: bool,
) -> tuple[pd.DataFrame, int, int]:
    frames: list[pd.DataFrame] = []
    downloaded = 0
    reused = 0
    for left, right in iter_chunks(config.start, config.end_exclusive, config.chunk_days):
        path = _chunk_path(symbol, left, right)
        if path.exists() and not force:
            frame = load_cached_chunk(path, symbol, left, right)
            reused += 1
        else:
            frame = request_chunk_with_retry(session, token, config, limiter, left, right)
            atomic_parquet(frame, path)
            downloaded += 1
        frames.append(frame)
    if not frames:
        return _canonicalize(pd.DataFrame(), symbol), downloaded, reused
    combined = pd.concat(frames, ignore_index=True)
    return _canonicalize(combined, symbol), downloaded, reused


def main() -> int:
    arguments = parse_arguments()
    _validate_parameters(arguments)
    load_dotenv(ENV_FILE)
    token = os.environ.get("TUSHARE_PROXY_TOKEN", "").strip()
    if not token:
        raise ValueError("缺少TUSHARE_PROXY_TOKEN")
    endpoint = os.environ.get("TUSHARE_PROXY_URL", DEFAULT_ENDPOINT).rstrip("/")
    project_config = _load_project_config()
    development_start = pd.Timestamp(project_config["research_split"]["development"][0])
    retrospective_end = pd.Timestamp(
        project_config["research_split"]["contaminated_retrospective"][1]
    )
    end_exclusive = retrospective_end + pd.Timedelta(days=1)
    universe = _load_universe(arguments)
    daily_reference = pd.read_parquet(
        DAILY_REFERENCE_FILE,
        columns=["date", "con_code", "is_suspended", "raw_open", "raw_high", "raw_low", "raw_close"],
    )
    daily_reference["date"] = pd.to_datetime(daily_reference["date"]).dt.normalize()
    limiter = RateLimiter(arguments.minimum_interval_seconds)
    session = requests.Session()
    results: list[dict[str, Any]] = []
    started_at = datetime.now(ZoneInfo(TIMEZONE))
    for position, row in universe.iterrows():
        symbol = str(row["con_code"])
        output_path = _output_path(symbol)
        metadata_path = _metadata_path(symbol)
        if output_path.exists() and metadata_path.exists() and not arguments.force:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("status") == "PASS":
                results.append(metadata)
                print(f"[{position + 1}/{len(universe)}] {symbol} 已完成，跳过。", flush=True)
                continue
        print(f"[{position + 1}/{len(universe)}] 正在采集 {symbol}……", flush=True)
        request_config = DownloadConfig(
            symbol=symbol,
            start=development_start,
            end_exclusive=end_exclusive,
            endpoint=endpoint,
            api_name="stk_mins",
            frequency="1min",
            chunk_days=arguments.chunk_days,
            minimum_interval_seconds=arguments.minimum_interval_seconds,
            timeout_seconds=arguments.timeout_seconds,
            maximum_retries=arguments.maximum_retries,
            force=arguments.force,
        )
        data, downloaded, reused = download_symbol(
            symbol, request_config, session, token, limiter, arguments.force
        )
        audit = audit_symbol(
            data, daily_reference, symbol, development_start, retrospective_end
        )
        metadata = {
            "symbol": symbol,
            **audit,
            "frequency": "1min",
            "source": "tushare_proxy.stk_mins",
            "request_policy": {
                "chunk_days": arguments.chunk_days,
                "minimum_interval_seconds": arguments.minimum_interval_seconds,
                "downloaded_chunks": downloaded,
                "reused_chunks": reused,
            },
            "retrieved_at": datetime.now(ZoneInfo(TIMEZONE)).isoformat(),
            "token_stored_in_metadata": False,
        }
        if audit["status"] != "PASS":
            _atomic_json(metadata, metadata_path)
            _atomic_json(
                {
                    "status": "IN_PROGRESS_WITH_FAILURE",
                    "started_at": started_at.isoformat(),
                    "completed_symbols": len(results),
                    "failed_symbol": symbol,
                    "results": results + [metadata],
                },
                REPORT_FILE,
            )
            raise RuntimeError(f"{symbol}质量审计失败：{audit['errors']}")
        atomic_parquet(data, output_path)
        metadata["file"] = {
            "path": str(output_path.relative_to(PROJECT_ROOT)),
            "bytes": output_path.stat().st_size,
            "sha256": _sha256(output_path),
        }
        _atomic_json(metadata, metadata_path)
        results.append(metadata)
        _atomic_json(
            {
                "status": "IN_PROGRESS" if len(results) < len(universe) else "PASS",
                "started_at": started_at.isoformat(),
                "updated_at": datetime.now(ZoneInfo(TIMEZONE)).isoformat(),
                "requested_symbol_count": int(len(universe)),
                "completed_symbol_count": int(len(results)),
                "total_row_count": int(sum(item.get("row_count", 0) for item in results)),
                "results": results,
                "token_stored_in_report": False,
            },
            REPORT_FILE,
        )
        print(
            f"[{position + 1}/{len(universe)}] {symbol} 完成：{len(data):,}行，"
            f"下载分块{downloaded}，复用分块{reused}。",
            flush=True,
        )
    print(f"成分股分钟采集完成：{len(results)}只。", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
