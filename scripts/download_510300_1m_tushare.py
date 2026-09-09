"""通过 TuShare 代理下载 510300 五年 1 分钟 K 线并执行完整性校验。

安全约束：
- token 只从环境变量 ``TUSHARE_PROXY_TOKEN`` 读取，不写入任何文件或日志；
- 默认每次请求至少间隔 0.55 秒；
- 每个日历分块单独落盘，可在中断后续传；
- 仅在全量校验通过后原子替换最终 CSV 与 Parquet 文件。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SETTINGS_FILE = PROJECT_ROOT / "config" / "settings.yaml"
DAILY_REFERENCE_FILE = PROJECT_ROOT / "data" / "raw" / "market" / "510300_daily_raw.parquet"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "market"
CHUNK_DIR = RAW_DIR / ".510300_1m_tushare_chunks"
PARQUET_FILE = RAW_DIR / "510300_1m_tushare_raw.parquet"
CSV_FILE = RAW_DIR / "510300_1m_tushare_raw.csv"
METADATA_FILE = RAW_DIR / "510300_1m_tushare_raw.metadata.json"
REPORT_FILE = PROJECT_ROOT / "reports" / "data_quality" / "510300_1m_tushare_quality.json"

NATIVE_COLUMNS = ["ts_code", "trade_time", "open", "high", "low", "close", "vol", "amount"]
NUMERIC_COLUMNS = ["open", "high", "low", "close", "vol", "amount"]
DEFAULT_ENDPOINT = "https://tt.xiaodefa.cn"
DEFAULT_API_NAME = "stk_mins"
DEFAULT_FREQUENCY = "1min"
TIMEZONE = "Asia/Shanghai"
SOURCE_TUTORIAL = "https://www.yuque.com/aimiao-ke7gj/vosp6x/tsdwcwttw15976iv"
SOURCE_DOCUMENTATION = "https://tushare.pro/document/2?doc_id=387"


class PermanentApiError(RuntimeError):
    """鉴权、权限或参数错误；继续重试没有意义。"""


class TransientApiError(RuntimeError):
    """限速、网络或服务端临时错误；允许退避重试。"""


@dataclass(frozen=True)
class DownloadConfig:
    symbol: str
    start: pd.Timestamp
    end_exclusive: pd.Timestamp
    endpoint: str
    api_name: str
    frequency: str
    chunk_days: int
    minimum_interval_seconds: float
    timeout_seconds: float
    maximum_retries: int
    force: bool


class RateLimiter:
    """确保任意相邻请求的发起时间至少相隔指定秒数。"""

    def __init__(self, minimum_interval_seconds: float) -> None:
        self.minimum_interval_seconds = minimum_interval_seconds
        self._last_request_started: float | None = None

    def wait(self) -> None:
        if self._last_request_started is not None:
            elapsed = time.monotonic() - self._last_request_started
            remaining = self.minimum_interval_seconds - elapsed
            if remaining > 0:
                time.sleep(remaining)
        self._last_request_started = time.monotonic()


def load_settings() -> dict[str, Any]:
    with SETTINGS_FILE.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="下载 510300 五年 1 分钟 K 线")
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("TUSHARE_PROXY_URL", DEFAULT_ENDPOINT),
        help="代理根地址，默认使用 TUSHARE_PROXY_URL 或教程中的 tt 域名",
    )
    parser.add_argument("--start-date", help="开始日期 YYYY-MM-DD；默认读取 config/settings.yaml")
    parser.add_argument("--end-date", help="结束日期 YYYY-MM-DD（含当日）；默认读取配置")
    parser.add_argument("--chunk-days", type=int, default=31, help="每个请求的日历天数")
    parser.add_argument(
        "--minimum-interval-seconds",
        type=float,
        default=0.55,
        help="相邻请求最小间隔，教程要求不低于 0.5 秒",
    )
    parser.add_argument("--timeout-seconds", type=float, default=90.0, help="单次请求超时")
    parser.add_argument("--maximum-retries", type=int, default=6, help="临时错误最大尝试次数")
    parser.add_argument("--force", action="store_true", help="忽略本地分块缓存并重新下载")
    return parser.parse_args()


def build_config(arguments: argparse.Namespace, settings: dict[str, Any]) -> DownloadConfig:
    start_text = arguments.start_date or settings["project"]["start_date"]
    end_text = arguments.end_date or settings["project"]["end_date"]
    start = pd.Timestamp(start_text).normalize()
    end_inclusive = pd.Timestamp(end_text).normalize()
    if start > end_inclusive:
        raise ValueError("开始日期不能晚于结束日期")
    endpoint = arguments.endpoint.rstrip("/")
    parsed = urlparse(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("代理地址必须是有效的 HTTP/HTTPS 根地址")
    if arguments.chunk_days < 1 or arguments.chunk_days > 31:
        raise ValueError("1 分钟行情的 chunk-days 必须位于 1 至 31，避免触发 8000 行截断")
    if arguments.minimum_interval_seconds < 0.5:
        raise ValueError("minimum-interval-seconds 不能低于教程要求的 0.5 秒")
    if arguments.timeout_seconds <= 0:
        raise ValueError("timeout-seconds 必须为正数")
    if arguments.maximum_retries < 1:
        raise ValueError("maximum-retries 必须至少为 1")
    return DownloadConfig(
        symbol=settings["symbols"]["etf"],
        start=start,
        end_exclusive=end_inclusive + pd.Timedelta(days=1),
        endpoint=endpoint,
        api_name=DEFAULT_API_NAME,
        frequency=DEFAULT_FREQUENCY,
        chunk_days=arguments.chunk_days,
        minimum_interval_seconds=arguments.minimum_interval_seconds,
        timeout_seconds=arguments.timeout_seconds,
        maximum_retries=arguments.maximum_retries,
        force=arguments.force,
    )


def iter_chunks(
    start: pd.Timestamp,
    end_exclusive: pd.Timestamp,
    chunk_days: int,
) -> Iterator[tuple[pd.Timestamp, pd.Timestamp]]:
    left = start
    step = pd.Timedelta(days=chunk_days)
    while left < end_exclusive:
        right = min(left + step, end_exclusive)
        yield left, right
        left = right


def format_api_time(value: pd.Timestamp) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S")


def chunk_file(left: pd.Timestamp, right: pd.Timestamp) -> Path:
    return CHUNK_DIR / f"{left:%Y%m%dT%H%M%S}__{right:%Y%m%dT%H%M%S}.parquet"


def atomic_parquet(data: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    data.to_parquet(temporary, index=False, engine="pyarrow")
    temporary.replace(path)


def atomic_csv(data: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    data.to_csv(temporary, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d %H:%M:%S")
    temporary.replace(path)


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_frame(
    data: pd.DataFrame,
    symbol: str,
    left: pd.Timestamp,
    right: pd.Timestamp,
) -> pd.DataFrame:
    missing = [column for column in NATIVE_COLUMNS if column not in data.columns]
    if missing:
        raise PermanentApiError(f"接口返回缺少必要字段：{missing}")
    output = data[NATIVE_COLUMNS].copy()
    output["ts_code"] = output["ts_code"].astype(str)
    output["trade_time"] = pd.to_datetime(output["trade_time"], errors="coerce")
    output[NUMERIC_COLUMNS] = output[NUMERIC_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if output["trade_time"].isna().any() or output[NUMERIC_COLUMNS].isna().any().any():
        raise PermanentApiError("接口返回存在无法解析的时间或数值")
    unexpected_symbols = sorted(set(output["ts_code"]).difference({symbol}))
    if unexpected_symbols:
        raise PermanentApiError(f"接口返回了非目标代码：{unexpected_symbols}")
    outside = ~output["trade_time"].between(left, right, inclusive="left")
    if outside.any():
        first_bad = output.loc[outside, "trade_time"].iloc[0]
        raise PermanentApiError(f"接口返回时间超出请求区间：{first_bad}")
    return (
        output.drop_duplicates("trade_time", keep="last")
        .sort_values("trade_time")
        .reset_index(drop=True)
    )


def classify_api_error(code: Any, message: str) -> type[RuntimeError]:
    normalized = message.lower()
    permanent_markers = ("权限", "token不对", "token 不对", "key不对", "key 不对", "过期", "参数")
    if str(code) in {"40201", "40202", "40203"} or any(
        marker in normalized for marker in permanent_markers
    ):
        return PermanentApiError
    return TransientApiError


def request_chunk_once(
    session: requests.Session,
    token: str,
    config: DownloadConfig,
    left: pd.Timestamp,
    right: pd.Timestamp,
) -> pd.DataFrame:
    payload = {
        "api_name": config.api_name,
        "params": {
            "ts_code": config.symbol,
            "freq": config.frequency,
            "start_date": format_api_time(left),
            "end_date": format_api_time(right),
        },
    }
    response = session.post(
        config.endpoint,
        json=payload,
        headers={"x-api-key": token},
        timeout=config.timeout_seconds,
    )
    response.raise_for_status()
    try:
        body = response.json()
    except requests.exceptions.JSONDecodeError as exc:
        raise TransientApiError("代理返回的内容不是有效 JSON") from exc
    code = body.get("code")
    message = str(body.get("msg") or "")
    if code not in {0, "0", None}:
        error_type = classify_api_error(code, message)
        raise error_type(f"接口返回错误 {code}：{message or '未提供错误说明'}")
    result = body.get("data")
    if not isinstance(result, dict):
        raise TransientApiError("接口响应缺少 data 对象")
    fields = result.get("fields")
    items = result.get("items")
    if not isinstance(fields, list) or items is None:
        raise TransientApiError("接口响应缺少 fields 或 items")
    if not isinstance(items, list):
        raise TransientApiError("接口响应 items 不是列表")
    if len(items) >= 8000:
        raise PermanentApiError("单个分块达到 8000 行上限，请减小 --chunk-days 后重试")
    raw = pd.DataFrame.from_records(items, columns=fields)
    if raw.empty:
        return pd.DataFrame(columns=NATIVE_COLUMNS)
    return normalize_frame(raw, config.symbol, left, right)


def request_chunk_with_retry(
    session: requests.Session,
    token: str,
    config: DownloadConfig,
    limiter: RateLimiter,
    left: pd.Timestamp,
    right: pd.Timestamp,
) -> pd.DataFrame:
    for attempt in range(1, config.maximum_retries + 1):
        limiter.wait()
        try:
            return request_chunk_once(session, token, config, left, right)
        except PermanentApiError:
            raise
        except (TransientApiError, requests.RequestException) as exc:
            if attempt == config.maximum_retries:
                raise RuntimeError(
                    f"分块 {left.date()} 至 {(right - pd.Timedelta(days=1)).date()} "
                    f"连续 {attempt} 次请求失败：{type(exc).__name__}: {exc}"
                ) from exc
            delay = min(2 ** (attempt - 1), 30)
            print(
                f"临时错误，第 {attempt}/{config.maximum_retries} 次失败，"
                f"{delay} 秒后重试：{type(exc).__name__}: {exc}",
                flush=True,
            )
            time.sleep(delay)
    raise AssertionError("重试循环不应执行到此处")


def load_cached_chunk(
    path: Path,
    symbol: str,
    left: pd.Timestamp,
    right: pd.Timestamp,
) -> pd.DataFrame:
    cached = pd.read_parquet(path)
    if cached.empty:
        return pd.DataFrame(columns=NATIVE_COLUMNS)
    return normalize_frame(cached, symbol, left, right)


def expected_timestamps(trade_dates: pd.Series) -> pd.DatetimeIndex:
    pieces: list[pd.DatetimeIndex] = []
    for date in pd.to_datetime(trade_dates).dt.normalize().drop_duplicates().sort_values():
        pieces.append(pd.date_range(date + pd.Timedelta(hours=9, minutes=30), date + pd.Timedelta(hours=11, minutes=30), freq="min"))
        pieces.append(pd.date_range(date + pd.Timedelta(hours=13, minutes=1), date + pd.Timedelta(hours=15), freq="min"))
    if not pieces:
        return pd.DatetimeIndex([])
    combined = pieces[0]
    for piece in pieces[1:]:
        combined = combined.append(piece)
    return combined


def validate_complete_data(
    data: pd.DataFrame,
    config: DownloadConfig,
    daily_reference: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = [
        {
            "code": "UNOFFICIAL_PROXY",
            "impact": "数据经第三方代理获取，不等同于直连TuShare官方主站。",
        },
        {
            "code": "API_NAME_DOCUMENTATION_MISMATCH",
            "impact": "代理请求名为stk_mins，TuShare当前官方ETF历史分钟文档记载的接口名为etf_mins。",
        },
        {
            "code": "TRADE_TIME_SEMANTICS_NOT_PROVIDER_DEFINED",
            "impact": "官方文档仅将trade_time解释为交易时间，不额外断言为bar_start或bar_end。",
        },
        {
            "code": "NO_POST_CLOSE_FIXED_PRICE_SESSION",
            "impact": "序列仅到15:00，不包含2026-07-06以后15:05至15:30盘后固定价格交易。",
        },
    ]
    evidence: dict[str, Any] = {}
    if data.empty:
        return [{"code": "EMPTY_DATA"}], warnings, evidence
    if list(data.columns) != NATIVE_COLUMNS:
        errors.append({"code": "UNEXPECTED_COLUMNS", "actual": data.columns.tolist()})
    duplicate_count = int(data["trade_time"].duplicated().sum())
    if duplicate_count:
        errors.append({"code": "DUPLICATE_TIMESTAMPS", "count": duplicate_count})
    if not data["trade_time"].is_monotonic_increasing:
        errors.append({"code": "TIMESTAMPS_NOT_SORTED"})
    if data["ts_code"].drop_duplicates().tolist() != [config.symbol]:
        errors.append({"code": "UNEXPECTED_SYMBOL", "values": data["ts_code"].unique().tolist()})

    prices = data[["open", "high", "low", "close"]]
    if prices.isna().any().any() or (prices <= 0).any().any():
        errors.append({"code": "INVALID_PRICE"})
    high_bad = data["high"] < data[["open", "close", "low"]].max(axis=1)
    low_bad = data["low"] > data[["open", "close", "high"]].min(axis=1)
    if high_bad.any() or low_bad.any():
        errors.append({"code": "OHLC_INCONSISTENT", "count": int((high_bad | low_bad).sum())})
    negative_flow = (data[["vol", "amount"]] < 0).any(axis=1)
    if negative_flow.any():
        errors.append({"code": "NEGATIVE_VOLUME_OR_AMOUNT", "count": int(negative_flow.sum())})

    reference = daily_reference.copy()
    reference["date"] = pd.to_datetime(reference["date"]).dt.normalize()
    end_inclusive = config.end_exclusive - pd.Timedelta(days=1)
    reference = reference.loc[reference["date"].between(config.start, end_inclusive)].copy()
    expected_index = expected_timestamps(reference["date"])
    actual_index = pd.DatetimeIndex(data["trade_time"])
    missing_timestamps = expected_index.difference(actual_index)
    extra_timestamps = actual_index.difference(expected_index)
    if len(missing_timestamps):
        errors.append(
            {
                "code": "MISSING_TIMESTAMPS",
                "count": int(len(missing_timestamps)),
                "examples": [value.isoformat() for value in missing_timestamps[:20]],
            }
        )
    if len(extra_timestamps):
        errors.append(
            {
                "code": "EXTRA_TIMESTAMPS",
                "count": int(len(extra_timestamps)),
                "examples": [value.isoformat() for value in extra_timestamps[:20]],
            }
        )

    counts = data.groupby(data["trade_time"].dt.normalize()).size()
    invalid_counts = counts[counts != 241]
    if not invalid_counts.empty:
        errors.append(
            {
                "code": "UNEXPECTED_BARS_PER_DAY",
                "count": int(len(invalid_counts)),
                "examples": {
                    day.date().isoformat(): int(count)
                    for day, count in invalid_counts.iloc[:20].items()
                },
            }
        )

    minute_daily = (
        data.assign(date=data["trade_time"].dt.normalize())
        .groupby("date", as_index=False)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            vol=("vol", "sum"),
            amount=("amount", "sum"),
        )
    )
    comparison = minute_daily.merge(
        reference[["date", "open", "high", "low", "close", "volume", "amount"]],
        on="date",
        how="outer",
        suffixes=("_minute", "_daily"),
        indicator=True,
    )
    price_mismatch = pd.Series(False, index=comparison.index)
    for column in ["open", "high", "low", "close"]:
        price_mismatch |= (
            comparison[f"{column}_minute"] - comparison[f"{column}_daily"]
        ).abs() > 0.001
    volume_denominator = comparison["volume"].abs().clip(lower=1)
    amount_denominator = comparison["amount_daily"].abs().clip(lower=1)
    volume_relative_difference = (comparison["vol"] - comparison["volume"]).abs() / volume_denominator
    amount_relative_difference = (
        comparison["amount_minute"] - comparison["amount_daily"]
    ).abs() / amount_denominator
    if price_mismatch.any():
        warnings.append({"code": "DAILY_PRICE_CROSS_SOURCE_MISMATCH", "days": int(price_mismatch.sum())})
    if (volume_relative_difference > 0.01).any():
        warnings.append(
            {
                "code": "DAILY_VOLUME_CROSS_SOURCE_DIFFERENCE_OVER_1_PERCENT",
                "days": int((volume_relative_difference > 0.01).sum()),
                "maximum_relative_difference": float(volume_relative_difference.max()),
            }
        )
    if (amount_relative_difference > 0.01).any():
        warnings.append(
            {
                "code": "DAILY_AMOUNT_CROSS_SOURCE_DIFFERENCE_OVER_1_PERCENT",
                "days": int((amount_relative_difference > 0.01).sum()),
                "maximum_relative_difference": float(amount_relative_difference.max()),
            }
        )
    evidence.update(
        {
            "expected_trading_days": int(reference["date"].nunique()),
            "actual_trading_days": int(counts.size),
            "expected_rows": int(len(expected_index)),
            "actual_rows": int(len(data)),
            "bars_per_day_distribution": {
                str(int(rows)): int(days) for rows, days in counts.value_counts().sort_index().items()
            },
            "daily_cross_source": {
                "reference": DAILY_REFERENCE_FILE.relative_to(PROJECT_ROOT).as_posix(),
                "price_mismatch_days_over_one_tick": int(price_mismatch.sum()),
                "volume_relative_difference_max": float(volume_relative_difference.max()),
                "amount_relative_difference_max": float(amount_relative_difference.max()),
            },
        }
    )
    return errors, warnings, evidence


def main() -> int:
    try:
        settings = load_settings()
        arguments = parse_arguments()
        config = build_config(arguments, settings)
        if not DAILY_REFERENCE_FILE.exists():
            raise FileNotFoundError(f"缺少交易日参考文件：{DAILY_REFERENCE_FILE}")

        chunks = list(iter_chunks(config.start, config.end_exclusive, config.chunk_days))
        requires_download = config.force or any(
            not chunk_file(left, right).exists() for left, right in chunks
        )
        token = os.environ.get("TUSHARE_PROXY_TOKEN", "").strip()
        if requires_download and not token:
            raise ValueError("存在待下载分块，但缺少环境变量 TUSHARE_PROXY_TOKEN")
        if token and len(token) != 56:
            raise ValueError("TUSHARE_PROXY_TOKEN 长度不是教程要求的 56 位")
        print(
            f"开始下载 {config.symbol} {config.frequency}：{config.start.date()} 至 "
            f"{(config.end_exclusive - pd.Timedelta(days=1)).date()}，共 {len(chunks)} 个分块。",
            flush=True,
        )
        session = requests.Session()
        session.headers.update(
            {
                "Accept-Encoding": "gzip",
                "User-Agent": "510300-minute-downloader/1.0",
            }
        )
        limiter = RateLimiter(config.minimum_interval_seconds)
        frames: list[pd.DataFrame] = []
        downloaded_chunks = 0
        reused_chunks = 0
        for index, (left, right) in enumerate(chunks, start=1):
            path = chunk_file(left, right)
            if path.exists() and not config.force:
                frame = load_cached_chunk(path, config.symbol, left, right)
                reused_chunks += 1
                action = "复用缓存"
            else:
                frame = request_chunk_with_retry(session, token, config, limiter, left, right)
                atomic_parquet(frame, path)
                downloaded_chunks += 1
                action = "完成下载"
            frames.append(frame)
            print(
                f"[{index:02d}/{len(chunks):02d}] {left.date()} 至 "
                f"{(right - pd.Timedelta(days=1)).date()}：{action}，{len(frame)} 行。",
                flush=True,
            )

        combined = pd.concat(frames, ignore_index=True)
        combined = normalize_frame(combined, config.symbol, config.start, config.end_exclusive)
        daily_reference = pd.read_parquet(DAILY_REFERENCE_FILE)
        errors, warnings, evidence = validate_complete_data(combined, config, daily_reference)
        checked_at = datetime.now(ZoneInfo(TIMEZONE)).isoformat()
        if errors:
            failed_report = {
                "status": "FAIL",
                "checked_at": checked_at,
                "symbol": config.symbol,
                "frequency": config.frequency,
                "errors": errors,
                "warnings": warnings,
                "evidence": evidence,
            }
            atomic_json(failed_report, REPORT_FILE)
            print(json.dumps(failed_report, ensure_ascii=False, indent=2), file=sys.stderr)
            return 1

        atomic_parquet(combined, PARQUET_FILE)
        atomic_csv(combined, CSV_FILE)
        parquet_hash = sha256_file(PARQUET_FILE)
        csv_hash = sha256_file(CSV_FILE)
        actual_first = combined["trade_time"].min()
        actual_last = combined["trade_time"].max()
        retrieved_at = datetime.now(ZoneInfo(TIMEZONE)).isoformat()
        metadata = {
            "status": "PASS",
            "coverage_status": "COMPLETE",
            "symbol": config.symbol,
            "frequency": config.frequency,
            "api_name": config.api_name,
            "endpoint_host": urlparse(config.endpoint).netloc,
            "start_date_requested": config.start.date().isoformat(),
            "end_date_requested": (config.end_exclusive - pd.Timedelta(days=1)).date().isoformat(),
            "actual_first_trade_time": actual_first.isoformat(),
            "actual_last_trade_time": actual_last.isoformat(),
            "row_count": int(len(combined)),
            "trading_day_count": int(combined["trade_time"].dt.normalize().nunique()),
            "columns": NATIVE_COLUMNS,
            "timestamp_meaning": (
                "提供方字段名为trade_time，TuShare当前官方ETF历史分钟文档仅解释为‘交易时间’；"
                "不在原始数据元数据中进一步断言为bar_start或bar_end。09:30记录包含开盘集合竞价。"
            ),
            "volume_unit": "share",
            "amount_unit": "CNY",
            "price_adjustment": "raw_unadjusted",
            "sort_order": "trade_time_ascending",
            "retrieved_at": retrieved_at,
            "request_policy": {
                "chunk_days": config.chunk_days,
                "minimum_interval_seconds": config.minimum_interval_seconds,
                "maximum_retries": config.maximum_retries,
                "downloaded_chunks": downloaded_chunks,
                "reused_chunks": reused_chunks,
            },
            "files": {
                "parquet": {
                    "path": PARQUET_FILE.relative_to(PROJECT_ROOT).as_posix(),
                    "sha256": parquet_hash,
                    "bytes": PARQUET_FILE.stat().st_size,
                },
                "csv": {
                    "path": CSV_FILE.relative_to(PROJECT_ROOT).as_posix(),
                    "sha256": csv_hash,
                    "bytes": CSV_FILE.stat().st_size,
                    "encoding": "UTF-8 with BOM",
                },
            },
            "source_references": [SOURCE_DOCUMENTATION, SOURCE_TUTORIAL],
            "token_stored": False,
        }
        report = {
            "status": "PASS" if not warnings else "WARN",
            "coverage_status": "PASS",
            "checked_at": checked_at,
            "symbol": config.symbol,
            "frequency": config.frequency,
            "file": PARQUET_FILE.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": parquet_hash,
            "errors": [],
            "warnings": warnings,
            "evidence": evidence,
        }
        atomic_json(metadata, METADATA_FILE)
        atomic_json(report, REPORT_FILE)
        print(
            f"下载完成：{len(combined):,} 行，{metadata['trading_day_count']} 个交易日，"
            f"{actual_first} 至 {actual_last}。",
            flush=True,
        )
        print(f"Parquet：{PARQUET_FILE}", flush=True)
        print(f"CSV：{CSV_FILE}", flush=True)
        print(f"质量报告：{REPORT_FILE}", flush=True)
        print(f"质量状态：{report['status']}；覆盖状态：PASS", flush=True)
        return 0
    except Exception as exc:
        print(f"下载失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
