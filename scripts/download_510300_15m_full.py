"""免费下载并构建510300五年完整未复权15分钟行情。"""

from __future__ import annotations

import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from time import sleep
from zoneinfo import ZoneInfo

import pandas as pd
import yaml

from market_data.tdx import (
    TdxMinuteRequest,
    TdxServer,
    connect_tdx,
    fetch_history_transactions,
    fetch_recent_direct_15m,
    reconstruct_15m_from_transactions,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = PROJECT_ROOT / "config" / "full_15m_data.yaml"
DAILY_FILE = PROJECT_ROOT / "data" / "raw" / "market" / "510300_daily_raw.parquet"
DIRECT_FILE = PROJECT_ROOT / "data" / "raw" / "market" / "510300_15m_tdx_direct_raw.parquet"
RECONSTRUCTED_FILE = (
    PROJECT_ROOT / "data" / "raw" / "market" / "510300_15m_tdx_reconstructed_raw.parquet"
)
OUTPUT_FILE = PROJECT_ROOT / "data" / "raw" / "market" / "510300_15m_full_raw.parquet"
METADATA_FILE = OUTPUT_FILE.with_suffix(".metadata.json")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_parquet(data: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    data.to_parquet(temporary, index=False, engine="pyarrow")
    temporary.replace(path)


def _load_config() -> tuple[dict, TdxMinuteRequest, list[TdxServer]]:
    config = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    tdx = config["tdx"]
    request = TdxMinuteRequest(
        symbol=config["symbol"],
        start_date=config["start_date"],
        end_date=config["end_date"],
        page_size=int(tdx["direct_page_size"]),
        transaction_page_size=int(tdx["transaction_page_size"]),
        maximum_pages=int(tdx["maximum_pages"]),
        timeout_seconds=float(tdx["timeout_seconds"]),
        timezone=config["timezone"],
        price_divisor=float(tdx["price_divisor"]),
        volume_multiplier=float(tdx["volume_multiplier"]),
    )
    request.validate()
    servers = [TdxServer(str(item["host"]), int(item["port"])) for item in tdx["servers"]]
    if not servers:
        raise ValueError("至少需要配置一个通达信服务器")
    return config, request, servers


def _connect_with_fallback(
    servers: list[TdxServer],
    preferred_index: int,
    timeout_seconds: float,
):
    errors: list[str] = []
    for offset in range(len(servers)):
        server = servers[(preferred_index + offset) % len(servers)]
        try:
            return connect_tdx(server, timeout_seconds), server
        except Exception as exc:
            errors.append(f"{server.host}:{server.port}={type(exc).__name__}:{exc}")
    raise RuntimeError("所有通达信服务器连接失败：" + " | ".join(errors))


def _download_worker(
    worker_index: int,
    dates: list[pd.Timestamp],
    request: TdxMinuteRequest,
    servers: list[TdxServer],
    maximum_retries: int,
) -> tuple[list[pd.DataFrame], list[dict]]:
    frames: list[pd.DataFrame] = []
    failures: list[dict] = []
    api, selected_server = _connect_with_fallback(
        servers,
        worker_index,
        request.timeout_seconds,
    )
    try:
        for day in dates:
            final_error: Exception | None = None
            for attempt in range(1, maximum_retries + 1):
                try:
                    records = fetch_history_transactions(api, request, day)
                    frame = reconstruct_15m_from_transactions(records, day, request)
                    if len(frame) != 16:
                        raise RuntimeError(f"重建得到{len(frame)}根，预期16根")
                    frame["tdx_server"] = f"{selected_server.host}:{selected_server.port}"
                    frames.append(frame)
                    final_error = None
                    break
                except Exception as exc:
                    final_error = exc
                    if attempt < maximum_retries:
                        sleep(0.25 * attempt)
                        try:
                            api.disconnect()
                        except Exception:
                            pass
                        api, selected_server = _connect_with_fallback(
                            servers,
                            worker_index + attempt,
                            request.timeout_seconds,
                        )
            if final_error is not None:
                failures.append(
                    {
                        "date": day.date().isoformat(),
                        "error": f"{type(final_error).__name__}: {final_error}",
                    }
                )
    finally:
        try:
            api.disconnect()
        except Exception:
            pass
    return frames, failures


def _download_missing_history(
    missing_dates: list[pd.Timestamp],
    existing: pd.DataFrame,
    config: dict,
    request: TdxMinuteRequest,
    servers: list[TdxServer],
) -> tuple[pd.DataFrame, list[dict]]:
    if not missing_dates:
        return existing, []
    worker_count = int(config["tdx"]["worker_count"])
    batch_days = int(config["tdx"]["checkpoint_batch_days"])
    retries = int(config["tdx"]["maximum_retries_per_day"])
    all_failures: list[dict] = []
    completed_before = int(existing["trade_date"].nunique()) if not existing.empty else 0
    for start in range(0, len(missing_dates), batch_days):
        batch = missing_dates[start : start + batch_days]
        chunks = [batch[index::worker_count] for index in range(worker_count)]
        batch_frames: list[pd.DataFrame] = []
        batch_failures: list[dict] = []
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(
                    _download_worker,
                    worker_index,
                    chunk,
                    request,
                    servers,
                    retries,
                ): worker_index
                for worker_index, chunk in enumerate(chunks)
                if chunk
            }
            for future in as_completed(futures):
                frames, failures = future.result()
                batch_frames.extend(frames)
                batch_failures.extend(failures)
        if batch_frames:
            new_data = pd.concat(batch_frames, ignore_index=True)
            existing = pd.concat([existing, new_data], ignore_index=True) if not existing.empty else new_data
            existing = (
                existing.drop_duplicates("bar_end", keep="last")
                .sort_values("bar_end")
                .reset_index(drop=True)
            )
            _atomic_parquet(existing, RECONSTRUCTED_FILE)
        all_failures.extend(batch_failures)
        finished = min(start + len(batch), len(missing_dates))
        print(
            f"历史成交重建进度：{finished}/{len(missing_dates)}个待补交易日，"
            f"累计失败{len(all_failures)}日，检查点已保存。",
            flush=True,
        )
    completed_after = int(existing["trade_date"].nunique()) if not existing.empty else 0
    print(
        f"本次新增重建交易日：{completed_after - completed_before}。",
        flush=True,
    )
    return existing, all_failures


def _validate_canonical(data: pd.DataFrame, expected_dates: set[pd.Timestamp]) -> list[str]:
    errors: list[str] = []
    if data.empty:
        return ["完整15分钟数据为空"]
    if data["bar_end"].duplicated().any():
        errors.append("存在重复bar_end")
    counts = data.groupby("trade_date").size()
    invalid_counts = counts[counts != 16]
    if not invalid_counts.empty:
        errors.append(
            "非16根交易日：" + ", ".join(
                f"{pd.Timestamp(day).date().isoformat()}={int(count)}"
                for day, count in invalid_counts.items()
            )
        )
    actual_dates = set(pd.to_datetime(data["trade_date"]).dt.normalize())
    missing = sorted(expected_dates.difference(actual_dates))
    if missing:
        errors.append("缺少交易日：" + ", ".join(day.date().isoformat() for day in missing))
    prices = data[["open", "high", "low", "close"]]
    if prices.isna().any().any() or (prices <= 0).any().any():
        errors.append("价格存在空值或非正数")
    if (data["high"] < prices.max(axis=1)).any() or (data["low"] > prices.min(axis=1)).any():
        errors.append("OHLC关系错误")
    return errors


def main() -> int:
    if not DAILY_FILE.exists():
        print(f"找不到交易日历和日线参考：{DAILY_FILE}", file=sys.stderr)
        return 1
    config, request, servers = _load_config()
    daily = pd.read_parquet(DAILY_FILE)
    daily_dates = pd.to_datetime(daily["date"]).dt.normalize()
    start_date = pd.Timestamp(request.start_date)
    end_date = pd.Timestamp(request.end_date) if request.end_date else daily_dates.max()
    target_dates = sorted(set(daily_dates.loc[(daily_dates >= start_date) & (daily_dates <= end_date)]))

    direct_api, direct_server = _connect_with_fallback(servers, 0, request.timeout_seconds)
    try:
        direct = fetch_recent_direct_15m(direct_api, request)
    finally:
        direct_api.disconnect()
    direct["tdx_server"] = f"{direct_server.host}:{direct_server.port}"
    _atomic_parquet(direct, DIRECT_FILE)
    direct_first_date = pd.Timestamp(direct["trade_date"].min()).normalize()
    print(
        f"原生15分钟K线：{len(direct)}根，"
        f"{direct_first_date.date().isoformat()}至"
        f"{pd.Timestamp(direct['trade_date'].max()).date().isoformat()}。",
        flush=True,
    )

    reconstructed = pd.read_parquet(RECONSTRUCTED_FILE) if RECONSTRUCTED_FILE.exists() else pd.DataFrame()
    if not reconstructed.empty:
        reconstructed["bar_end"] = pd.to_datetime(reconstructed["bar_end"])
        reconstructed["trade_date"] = pd.to_datetime(reconstructed["trade_date"]).dt.normalize()
    completed_dates = set(reconstructed.groupby("trade_date").size().loc[lambda value: value == 16].index) \
        if not reconstructed.empty else set()
    required_reconstructed_dates = [day for day in target_dates if day < direct_first_date]
    missing_dates = [day for day in required_reconstructed_dates if day not in completed_dates]
    print(
        f"原生窗口以前需重建{len(required_reconstructed_dates)}个交易日，"
        f"本地检查点已完成{len(required_reconstructed_dates) - len(missing_dates)}日，"
        f"本次待补{len(missing_dates)}日。",
        flush=True,
    )
    reconstructed, failures = _download_missing_history(
        missing_dates,
        reconstructed,
        config,
        request,
        servers,
    )

    older = reconstructed.loc[reconstructed["trade_date"] < direct_first_date].copy()
    canonical = pd.concat([older, direct], ignore_index=True)
    canonical = (
        canonical.drop_duplicates("bar_end", keep="last")
        .sort_values("bar_end")
        .reset_index(drop=True)
    )
    canonical["source_regime"] = canonical["construction"].map(
        {
            "transaction_reconstructed_15m": "tdx_transaction_reconstructed",
            "exchange_15m_kline": "tdx_direct_15m",
        }
    )
    expected_dates = set(target_dates)
    validation_errors = _validate_canonical(canonical, expected_dates)
    _atomic_parquet(canonical, OUTPUT_FILE)

    counts = canonical.groupby("trade_date").size()
    failed = bool(failures or validation_errors)
    metadata = {
        "status": "FAIL" if failed else "COVERAGE_COMPLETE_AWAITING_QUALITY_AUDIT",
        "coverage_status": "FAIL" if failed else "COMPLETE",
        "symbol": request.symbol,
        "frequency_minutes": 15,
        "adjustment": "raw",
        "requested_start_date": request.start_date,
        "requested_end_date": request.end_date or "latest_available",
        "actual_first_bar_end": pd.Timestamp(canonical["bar_end"].min()).isoformat(),
        "actual_last_bar_end": pd.Timestamp(canonical["bar_end"].max()).isoformat(),
        "row_count": int(len(canonical)),
        "trading_day_count": int(canonical["trade_date"].nunique()),
        "full_trading_day_count": int((counts == 16).sum()),
        "direct_15m_row_count": int((canonical["construction"] == "exchange_15m_kline").sum()),
        "reconstructed_15m_row_count": int(
            (canonical["construction"] == "transaction_reconstructed_15m").sum()
        ),
        "direct_15m_first_trade_date": direct_first_date.date().isoformat(),
        "reconstruction_note": (
            "原生15分钟保留窗口以前由历史成交记录重建；重建成交额为成交价乘成交量估算，"
            "OHLCV仍需通过日线及重叠窗口审计后才能用于策略因子。"
        ),
        "post_close_included": False,
        "failures": failures,
        "validation_errors": validation_errors,
        "retrieved_at": datetime.now(ZoneInfo(request.timezone)).isoformat(),
        "file": OUTPUT_FILE.relative_to(PROJECT_ROOT).as_posix(),
        "sha256": _sha256_file(OUTPUT_FILE),
    }
    METADATA_FILE.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
