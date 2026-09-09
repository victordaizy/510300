"""刷新行业预期差冻结原点的可变结果输入。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime, time as clock_time
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import tushare as ts
import yaml
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.industry_expectation_gap_outcome_refresh import (  # noqa: E402
    OutcomeRefreshError,
    build_etf_total_return_extension,
    merge_constituent_extension,
    trading_dates_between,
    validate_close_crosscheck,
)
from scripts.acquire_industry_expectation_gap_etf_model_v2 import (  # noqa: E402
    normalize_constituent_extension,
)
CONFIG_PATH = ROOT / "config" / "industry_expectation_gap_outcome_refresh_v1.yaml"
ENV_PATH = ROOT / ".env"
SINA_QUOTE_URL = "https://hq.sinajs.cn/list={symbols}"
SINA_HEADERS = {
    "Referer": "https://finance.sina.com.cn/",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
}


def _path(relative: str) -> Path:
    root = ROOT.resolve()
    result = (root / relative).resolve()
    try:
        result.relative_to(root)
    except ValueError as exc:
        raise OutcomeRefreshError(f"路径越出工作区：{relative}") from exc
    return result


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_status(config: dict[str, Any], payload: dict[str, Any]) -> Path:
    current = _path(config["outputs"]["current_status"])
    immutable_directory = _path(config["outputs"]["immutable_status_directory"])
    immutable_directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.fromisoformat(str(payload["retrieved_at"])).strftime("%Y%m%d_%H%M%S_%f")
    immutable = immutable_directory / f"outcome_refresh_{stamp}.json"
    _atomic_json(payload, current)
    _atomic_json(payload, immutable)
    return immutable


def _retry(
    call: Callable[[], pd.DataFrame],
    label: str,
    *,
    attempts: int,
) -> pd.DataFrame:
    error: Exception | None = None
    for attempt in range(attempts):
        try:
            result = call()
            if result is None or result.empty:
                raise ValueError("返回空数据")
            return result
        except Exception as exc:
            error = exc
            if attempt + 1 < attempts:
                time.sleep(2.0 * (attempt + 1))
    raise OutcomeRefreshError(f"{label}失败：{type(error).__name__}: {error}")


def _load_proxy_without_persistence(timeout_seconds: int) -> tuple[Any, str]:
    """直接把工作区环境变量传给客户端，不调用会写用户目录的set_token。"""

    load_dotenv(ENV_PATH, override=False)
    token = os.getenv("TUSHARE_PROXY_TOKEN", "").strip()
    if not token:
        raise OutcomeRefreshError("缺少TUSHARE_PROXY_TOKEN；未读取或输出任何令牌内容")
    endpoint = os.getenv("TUSHARE_PROXY_URL", "https://fast.xiaodefa.cn").strip().rstrip("/")
    if endpoint not in {"https://fast.xiaodefa.cn", "https://tt.xiaodefa.cn"}:
        raise OutcomeRefreshError("代理地址不在允许名单内")
    pro = ts.pro_api(token, timeout=timeout_seconds)
    pro._DataApi__http_url = endpoint
    return pro, endpoint


def _sina_symbol(symbol: str) -> str:
    code, exchange = symbol.split(".")
    return f"{exchange.lower()}{code}"


def _fetch_sina_quotes(
    symbols: list[str],
    target: pd.Timestamp,
    *,
    timeout_seconds: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for start in range(0, len(symbols), 100):
        batch = symbols[start : start + 100]
        source_symbols = [_sina_symbol(symbol) for symbol in batch]
        response = requests.get(
            SINA_QUOTE_URL.format(symbols=",".join(source_symbols)),
            headers=SINA_HEADERS,
            timeout=(10, timeout_seconds),
        )
        response.raise_for_status()
        text = response.content.decode("gbk", errors="strict")
        parsed = {
            match.group("symbol"): match.group("values").split(",")
            for match in re.finditer(
                r'var hq_str_(?P<symbol>[a-z]{2}\d{6})="(?P<values>[^"]*)";',
                text,
            )
        }
        for symbol, source_symbol in zip(batch, source_symbols, strict=True):
            values = parsed.get(source_symbol)
            if values is None or len(values) < 32:
                raise OutcomeRefreshError(f"新浪独立快照缺失：{symbol}")
            quote_date = pd.Timestamp(values[30]).normalize()
            if quote_date != target:
                raise OutcomeRefreshError(
                    f"{symbol}新浪快照日期为{quote_date.date()}，不是{target.date()}"
                )
            open_price = pd.to_numeric(values[1], errors="coerce")
            previous_close = pd.to_numeric(values[2], errors="coerce")
            current = pd.to_numeric(values[3], errors="coerce")
            high = pd.to_numeric(values[4], errors="coerce")
            low = pd.to_numeric(values[5], errors="coerce")
            suspended = pd.isna(current) or float(current) <= 0
            close = float(previous_close if suspended else current)
            if not close > 0:
                raise OutcomeRefreshError(f"{symbol}新浪快照没有有效收盘价或昨收价")
            rows.append(
                {
                    "date": target,
                    "con_code": symbol,
                    "raw_open": float(previous_close if suspended else open_price),
                    "raw_high": float(previous_close if suspended else high),
                    "raw_low": float(previous_close if suspended else low),
                    "raw_close": close,
                    "is_suspended": bool(suspended),
                    "source": "sina.hq.batch_quote",
                }
            )
    result = pd.DataFrame(rows)
    if result["con_code"].nunique() != len(symbols):
        raise OutcomeRefreshError("新浪独立快照覆盖不完整")
    return result


def _normalize_fund_daily(
    raw: pd.DataFrame,
    retrieved_at: datetime,
) -> pd.DataFrame:
    required = {"trade_date", "open", "high", "low", "close", "vol", "amount"}
    if missing := sorted(required.difference(raw.columns)):
        raise OutcomeRefreshError(f"ETF主来源日线缺少字段：{missing}")
    frame = raw.copy()
    frame["date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.normalize()
    for column in ("open", "high", "low", "close", "vol", "amount"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[["date", "open", "high", "low", "close", "vol", "amount"]].isna().any(axis=None):
        raise OutcomeRefreshError("ETF主来源日线存在空值")
    frame["volume"] = frame["vol"] * 100.0
    frame["amount"] = frame["amount"] * 1000.0
    frame["symbol"] = "510300.SH"
    frame["source"] = "tushare_proxy.fund_daily"
    frame["volume_unit"] = "share"
    frame["amount_unit"] = "CNY"
    frame["retrieved_at"] = retrieved_at.isoformat()
    columns = [
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "symbol",
        "source",
        "volume_unit",
        "amount_unit",
        "retrieved_at",
    ]
    return frame[columns].sort_values("date").drop_duplicates("date", keep="last")


def _merge_etf_raw(existing: pd.DataFrame, extension: pd.DataFrame) -> pd.DataFrame:
    base = existing.copy()
    new = extension.copy()
    base["date"] = pd.to_datetime(base["date"], errors="coerce").dt.normalize()
    new["date"] = pd.to_datetime(new["date"], errors="coerce").dt.normalize()
    if not new.empty and new["date"].min() <= base["date"].max():
        raise OutcomeRefreshError("ETF原始日线新增部分不得覆盖历史日期")
    combined = pd.concat([base, new.loc[:, base.columns]], ignore_index=True)
    if combined["date"].duplicated().any():
        raise OutcomeRefreshError("ETF原始日线合并后存在重复日期")
    return combined.sort_values("date").reset_index(drop=True)


def _write_parquet_temporary(frame: pd.DataFrame, target: Path) -> Path:
    temporary = target.with_suffix(target.suffix + ".outcome_refresh.tmp")
    if temporary.exists():
        temporary.unlink()
    frame.to_parquet(temporary, index=False, engine="pyarrow")
    return temporary


def _commit_with_backups(
    replacements: dict[Path, pd.DataFrame],
    backup_directory: Path,
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """所有临时文件验证完成后再替换；失败时从明确备份恢复。"""

    before: dict[str, str] = {}
    backups: dict[Path, Path] = {}
    temporary_files: dict[Path, Path] = {}
    try:
        for target, frame in replacements.items():
            if not target.is_file():
                raise OutcomeRefreshError(f"待刷新文件不存在：{target}")
            before[_relative(target)] = _sha256(target)
            temporary_files[target] = _write_parquet_temporary(frame, target)
        backup_directory.mkdir(parents=True, exist_ok=False)
        for target in replacements:
            backup = backup_directory / target.name
            shutil.copy2(target, backup)
            backups[target] = backup
    except Exception:
        for temporary in temporary_files.values():
            if temporary.exists():
                temporary.unlink()
        raise
    committed: list[Path] = []
    try:
        for target, temporary in temporary_files.items():
            os.replace(temporary, target)
            committed.append(target)
    except Exception:
        for target in committed:
            shutil.copy2(backups[target], target)
        raise
    finally:
        for temporary in temporary_files.values():
            if temporary.exists():
                temporary.unlink()
    after = {_relative(target): _sha256(target) for target in replacements}
    backup_paths = {
        _relative(target): _relative(backup) for target, backup in backups.items()
    }
    return before, after, backup_paths


def _skip_status(
    config: dict[str, Any],
    retrieved_at: datetime,
    status: str,
    message: str,
    target: pd.Timestamp | None,
) -> int:
    payload = {
        "schema_version": "1.0.0",
        "refresh_version": config["version"],
        "retrieved_at": retrieved_at.isoformat(),
        "run_status": "SUCCESS",
        "collection_status": status,
        "target_date": target.date().isoformat() if target is not None else None,
        "message": message,
        "files_changed": False,
        "prediction_ledger_write_enabled": False,
        "partial_horizon_return_output_enabled": False,
        "position_mapping_enabled": False,
        "order_generation_enabled": False,
        "broker_connection_enabled": False,
        "live_trading_enabled": False,
    }
    immutable = _write_status(config, payload)
    print(f"结果输入刷新状态：{status}")
    print(f"审计文件：{immutable}")
    return 0


def refresh(config: dict[str, Any], explicit_date: str | None = None) -> int:
    timezone = ZoneInfo(config["timezone"])
    retrieved_at = datetime.now(timezone)
    calendar = pd.read_csv(_path(config["inputs"]["trading_calendar"]))
    date_column = "trade_date" if "trade_date" in calendar.columns else "date"
    trading_dates = pd.to_datetime(calendar[date_column], errors="coerce").dropna().dt.normalize()
    today = pd.Timestamp(retrieved_at.date())
    close_confirmation = clock_time.fromisoformat(
        config["quality"]["market_close_confirmation_time"]
    )
    if explicit_date is None:
        if today not in set(trading_dates):
            return _skip_status(
                config,
                retrieved_at,
                "SKIPPED_NON_TRADING_DAY",
                f"{today.date()}不在官方交易日历。",
                today,
            )
        if retrieved_at.timetz().replace(tzinfo=None) < close_confirmation:
            return _skip_status(
                config,
                retrieved_at,
                "SKIPPED_BEFORE_MARKET_CLOSE",
                "尚未到收盘确认时间，不读取当日结果。",
                today,
            )
        target = today
    else:
        target = pd.Timestamp(explicit_date).normalize()
        if target not in set(trading_dates):
            return _skip_status(
                config,
                retrieved_at,
                "SKIPPED_NON_TRADING_DAY",
                f"{target.date()}不在官方交易日历。",
                target,
            )

    inputs = config["inputs"]
    constituent_path = _path(inputs["constituent_total_return_daily"])
    weights_path = _path(inputs["historical_weights"])
    etf_raw_path = _path(inputs["etf_raw_daily"])
    etf_total_path = _path(inputs["etf_total_return_daily"])
    dividend_path = _path(inputs["etf_dividends"])
    constituent = pd.read_parquet(constituent_path)
    weights = pd.read_parquet(weights_path)
    etf_raw = pd.read_parquet(etf_raw_path)
    etf_total = pd.read_parquet(etf_total_path)
    dividends = pd.read_csv(dividend_path)
    constituent["date"] = pd.to_datetime(constituent["date"], errors="coerce").dt.normalize()
    weights["trade_date"] = pd.to_datetime(weights["trade_date"], errors="coerce").dt.normalize()
    etf_raw["date"] = pd.to_datetime(etf_raw["date"], errors="coerce").dt.normalize()
    etf_total["date"] = pd.to_datetime(etf_total["date"], errors="coerce").dt.normalize()

    constituent_missing = trading_dates_between(
        calendar, constituent["date"].max(), target
    )
    etf_missing = trading_dates_between(calendar, etf_total["date"].max(), target)
    if not constituent_missing and not etf_missing:
        return _skip_status(
            config,
            retrieved_at,
            "ALREADY_CURRENT",
            "成分股与ETF总回报结果输入均已到目标交易日。",
            target,
        )

    latest_weight_date = weights.loc[
        weights["trade_date"].le(target), "trade_date"
    ].max()
    symbols = sorted(
        weights.loc[weights["trade_date"].eq(latest_weight_date), "con_code"]
        .astype(str)
        .unique()
    )
    required_count = int(config["quality"]["required_constituent_count"])
    if len(symbols) != required_count:
        raise OutcomeRefreshError(
            f"目标权重截面证券数为{len(symbols)}，不是{required_count}"
        )

    attempts = int(config["providers"]["request_attempts"])
    pro, endpoint = _load_proxy_without_persistence(
        int(config["providers"]["request_timeout_seconds"])
    )
    daily_by_date: dict[pd.Timestamp, pd.DataFrame] = {}
    factor_by_date: dict[pd.Timestamp, pd.DataFrame] = {}
    for date in constituent_missing:
        date_text = date.strftime("%Y%m%d")
        daily_by_date[date] = _retry(
            lambda date_text=date_text: pro.daily(trade_date=date_text),
            f"{date.date()}全市场日线",
            attempts=attempts,
        )
        factor_by_date[date] = _retry(
            lambda date_text=date_text: pro.adj_factor(trade_date=date_text),
            f"{date.date()}全市场复权因子",
            attempts=attempts,
        )
    constituent_extension = normalize_constituent_extension(
        daily_by_date,
        factor_by_date,
        symbols,
        constituent,
        retrieved_at,
    )
    constituent_extension["retrieved_at"] = retrieved_at.isoformat()
    constituent_combined = merge_constituent_extension(
        constituent,
        constituent_extension,
        required_constituent_count=required_count,
    )

    timeout_seconds = int(config["providers"]["request_timeout_seconds"])
    independent_constituent = _fetch_sina_quotes(
        symbols,
        target,
        timeout_seconds=timeout_seconds,
    )
    constituent_crosscheck = validate_close_crosscheck(
        constituent_extension,
        independent_constituent,
        target_date=target,
        maximum_close_difference=float(
            config["providers"]["maximum_close_difference"]
        ),
        required_constituent_count=required_count,
    )

    missing_raw_dates = [date for date in etf_missing if date not in set(etf_raw["date"])]
    fund_parts: list[pd.DataFrame] = []
    for date in missing_raw_dates:
        date_text = date.strftime("%Y%m%d")
        fund_parts.append(
            _retry(
                lambda date_text=date_text: pro.fund_daily(
                    ts_code="510300.SH", trade_date=date_text
                ),
                f"510300 {date.date()}基金日线",
                attempts=attempts,
            )
        )
    fund_extension = (
        _normalize_fund_daily(pd.concat(fund_parts, ignore_index=True), retrieved_at)
        if fund_parts
        else pd.DataFrame(columns=etf_raw.columns)
    )
    etf_raw_combined = _merge_etf_raw(etf_raw, fund_extension)
    raw_extension = etf_raw_combined.loc[
        etf_raw_combined["date"].isin(etf_missing)
    ].copy()
    if set(raw_extension["date"]) != set(etf_missing):
        raise OutcomeRefreshError("ETF原始日线无法完整覆盖总回报缺失交易日")
    total_extension = build_etf_total_return_extension(
        etf_total,
        raw_extension,
        dividends,
    )
    etf_total_combined = pd.concat(
        [etf_total, total_extension], ignore_index=True
    ).sort_values("date").reset_index(drop=True)
    if etf_total_combined["date"].duplicated().any():
        raise OutcomeRefreshError("ETF总回报合并后存在重复日期")

    etf_independent = _fetch_sina_quotes(
        ["510300.SH"],
        target,
        timeout_seconds=timeout_seconds,
    ).iloc[0]
    etf_primary = etf_raw_combined.loc[etf_raw_combined["date"].eq(target)].iloc[0]
    etf_differences = {
        field: abs(float(etf_primary[field]) - float(etf_independent[f"raw_{field}"]))
        for field in ("open", "high", "low", "close")
    }
    maximum_etf_difference = max(etf_differences.values())
    tolerance = float(config["providers"]["maximum_close_difference"])
    if maximum_etf_difference > tolerance + 1e-12:
        raise OutcomeRefreshError(
            f"ETF独立OHLC最大差异{maximum_etf_difference}超过{tolerance}"
        )

    run_id = retrieved_at.strftime("%Y%m%d_%H%M%S_%f")
    backup_directory = _path(config["outputs"]["backup_directory"]) / run_id
    replacements = {
        constituent_path: constituent_combined,
        etf_raw_path: etf_raw_combined,
        etf_total_path: etf_total_combined,
    }
    before_hashes, after_hashes, backup_paths = _commit_with_backups(
        replacements,
        backup_directory,
    )
    payload = {
        "schema_version": "1.0.0",
        "refresh_version": config["version"],
        "retrieved_at": retrieved_at.isoformat(),
        "run_status": "SUCCESS",
        "collection_status": "REFRESHED_CURRENT_TRADING_DAY",
        "target_date": target.date().isoformat(),
        "weight_snapshot_date": latest_weight_date.date().isoformat(),
        "constituent_missing_dates": [date.date().isoformat() for date in constituent_missing],
        "etf_total_return_missing_dates": [date.date().isoformat() for date in etf_missing],
        "constituent_rows_before": int(len(constituent)),
        "constituent_rows_after": int(len(constituent_combined)),
        "etf_total_return_rows_before": int(len(etf_total)),
        "etf_total_return_rows_after": int(len(etf_total_combined)),
        "constituent_crosscheck": constituent_crosscheck,
        "etf_crosscheck": {
            "status": "PASS",
            "target_date": target.date().isoformat(),
            "absolute_ohlc_differences": etf_differences,
            "maximum_absolute_ohlc_difference": maximum_etf_difference,
        },
        "provider": {
            "primary": config["providers"]["primary_daily"],
            "primary_host": urlsplit(endpoint).netloc or endpoint.split("/", 1)[0],
            "independent": config["providers"]["independent_close_crosscheck"],
            "credential_persisted": False,
        },
        "before_sha256": before_hashes,
        "after_sha256": after_hashes,
        "recoverable_backups": backup_paths,
        "files_changed": True,
        "prediction_ledger_write_enabled": False,
        "partial_horizon_return_output_enabled": False,
        "position_mapping_enabled": False,
        "order_generation_enabled": False,
        "broker_connection_enabled": False,
        "live_trading_enabled": False,
    }
    immutable = _write_status(config, payload)
    print(f"结果输入刷新状态：{payload['collection_status']}")
    print(f"成分股截止：{constituent_combined['date'].max().date()}")
    print(f"ETF总回报截止：{etf_total_combined['date'].max().date()}")
    print(f"审计文件：{immutable}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="刷新行业预期差可变结果输入")
    parser.add_argument("--date", help="目标交易日YYYY-MM-DD，默认上海本地日期")
    args = parser.parse_args()
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        print("结果输入刷新配置顶层必须是对象", file=sys.stderr)
        return 2
    try:
        return refresh(config, args.date)
    except Exception as exc:
        retrieved_at = datetime.now(ZoneInfo(config["timezone"]))
        payload = {
            "schema_version": "1.0.0",
            "refresh_version": config.get("version"),
            "retrieved_at": retrieved_at.isoformat(),
            "run_status": "FAILED",
            "collection_status": "FAILED_OUTCOME_INPUT_REFRESH",
            "target_date": args.date or retrieved_at.date().isoformat(),
            "error": f"{type(exc).__name__}: {exc}",
            "files_changed": False,
            "prediction_ledger_write_enabled": False,
            "partial_horizon_return_output_enabled": False,
            "position_mapping_enabled": False,
            "order_generation_enabled": False,
            "broker_connection_enabled": False,
            "live_trading_enabled": False,
        }
        immutable = _write_status(config, payload)
        print(f"结果输入刷新失败：{payload['error']}", file=sys.stderr)
        print(f"失败审计文件：{immutable}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
