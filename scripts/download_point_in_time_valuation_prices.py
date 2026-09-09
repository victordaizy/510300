"""下载并构建五年点时估值所需的官方权重快照价格表。"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.point_in_time_valuation_price_acquisition import (
    build_snapshot_prices,
    cache_covers_request,
    cross_check_overlap,
    load_config,
    normalize_daily,
    required_missing_symbols,
    sha256_file,
    validate_snapshot_prices,
    write_report,
)
from scripts.download_510300_option_research_data import (
    RateLimiter,
    atomic_parquet,
    load_proxy,
    now,
    retry_call,
    sanitized_error,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="补采点时估值权重快照价格")
    parser.add_argument("--interval", type=float, default=None)
    parser.add_argument("--workers", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config()
    interval = float(args.interval or config["scope"]["minimum_request_interval_seconds"])
    workers = int(args.workers or config["scope"]["maximum_workers"])
    if interval < float(config["scope"]["minimum_request_interval_seconds"]):
        raise ValueError("请求间隔低于冻结下限")
    if workers < 1 or workers > int(config["scope"]["maximum_workers"]):
        raise ValueError("并发数超出冻结范围")
    contracts = config["data_contracts"]
    weights = pd.read_parquet(ROOT / contracts["historical_weights"]["file"])
    current = pd.read_parquet(ROOT / contracts["current_constituent_daily"]["file"])
    prior = pd.read_parquet(ROOT / contracts["prior_snapshot_audit"]["file"])
    symbols, failed_dates = required_missing_symbols(weights, prior, config)
    if len(symbols) != 437 or len(failed_dates) != 40:
        raise RuntimeError(f"补采范围偏离审计：证券{len(symbols)}只、月份{len(failed_dates)}个")
    request_start = pd.Timestamp(config["scope"]["download_start"])
    request_end = pd.Timestamp(config["scope"]["download_end"])
    cache_dir = ROOT / config["artifacts"]["cache_directory"]
    cache_dir.mkdir(parents=True, exist_ok=True)
    pro, token, endpoint = load_proxy()
    limiter = RateLimiter(interval)

    def obtain(symbol: str) -> tuple[pd.DataFrame, bool]:
        path = cache_dir / f"{symbol.replace('.', '_')}.parquet"
        if path.exists():
            cached = pd.read_parquet(path)
            if cache_covers_request(cached, symbol, request_start, request_end):
                return cached, True
        raw = retry_call(
            lambda: pro.daily(
                ts_code=symbol,
                start_date=request_start.strftime("%Y%m%d"),
                end_date=request_end.strftime("%Y%m%d"),
                fields=",".join(config["source"]["fields"]),
            ),
            limiter,
            f"下载{symbol}估值历史日线",
            token,
        )
        normalized = normalize_daily(raw, symbol, request_start, request_end, now())
        atomic_parquet(normalized, path)
        return normalized, False

    histories: dict[str, pd.DataFrame] = {}
    reused = 0
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="估值价格补采") as pool:
        futures = {pool.submit(obtain, symbol): symbol for symbol in symbols}
        for completed, future in enumerate(as_completed(futures), start=1):
            symbol = futures[future]
            try:
                data, was_reused = future.result()
            except Exception as exc:
                raise RuntimeError(f"{symbol}补采失败：{sanitized_error(exc, token)}") from exc
            histories[symbol] = data
            reused += int(was_reused)
            if completed == 1 or completed % 50 == 0 or completed == len(symbols):
                print(
                    f"估值价格进度：{completed}/{len(symbols)}，缓存复用{reused}",
                    flush=True,
                )
    downloaded = pd.concat(histories.values(), ignore_index=True)
    overlap = cross_check_overlap(downloaded, current)
    if overlap["overlap_row_count"] <= 0 or overlap["exact_match_ratio"] < 0.999:
        raise RuntimeError(f"跨源收盘价核验未通过：{overlap}")
    snapshots = build_snapshot_prices(weights, downloaded, current, config)
    missing_symbols = sorted(
        snapshots.loc[snapshots["raw_close"].isna(), "con_code"].astype(str).unique()
    )
    supplemental_count = 0
    if missing_symbols:
        supplemental_start = pd.Timestamp(config["scope"]["supplemental_prehistory_start"])
        supplemental_end = request_start - pd.Timedelta(days=1)
        print(
            f"发现{len(missing_symbols)}只证券在下载起点前已停牌，追加"
            f"{supplemental_start.date()}至{supplemental_end.date()}历史",
            flush=True,
        )
        for symbol in missing_symbols:
            raw = retry_call(
                lambda symbol=symbol: pro.daily(
                    ts_code=symbol,
                    start_date=supplemental_start.strftime("%Y%m%d"),
                    end_date=supplemental_end.strftime("%Y%m%d"),
                    fields=",".join(config["source"]["fields"]),
                ),
                limiter,
                f"追加{symbol}停牌前历史日线",
                token,
            )
            supplement = normalize_daily(
                raw, symbol, supplemental_start, supplemental_end, now()
            )
            existing_symbol = histories[symbol].copy()
            existing_symbol["retrieved_at"] = existing_symbol["retrieved_at"].astype(str)
            supplement["retrieved_at"] = supplement["retrieved_at"].astype(str)
            merged = pd.concat([supplement, existing_symbol], ignore_index=True)
            merged = merged.sort_values("date").drop_duplicates(
                ["date", "con_code"], keep="last"
            )
            merged["request_start"] = supplemental_start
            merged["request_end"] = request_end
            histories[symbol] = merged.reset_index(drop=True)
            cache_path = cache_dir / f"{symbol.replace('.', '_')}.parquet"
            atomic_parquet(histories[symbol], cache_path)
            supplemental_count += 1
        downloaded = pd.concat(histories.values(), ignore_index=True)
        snapshots = build_snapshot_prices(weights, downloaded, current, config)
    validation = validate_snapshot_prices(snapshots, config)
    if validation["status"] != "PASS":
        raise RuntimeError(f"快照价格验收失败：{validation}")
    output = ROOT / config["artifacts"]["snapshot_prices"]
    atomic_parquet(snapshots, output)
    prehistory_symbol_count = sum(
        pd.to_datetime(frame["request_start"], errors="coerce").min() < request_start
        for frame in histories.values()
    )
    report = {
        "status": "PASS_FIVE_YEAR_SNAPSHOT_PRICES",
        "checked_at": now().isoformat(),
        "proxy_host": endpoint.split("//", 1)[-1],
        "token_persisted_in_outputs": False,
        "download_start": str(request_start.date()),
        "download_end": str(request_end.date()),
        "failed_month_count": len(failed_dates),
        "required_symbol_count": len(symbols),
        "downloaded_symbol_count": len(symbols) - reused,
        "reused_cache_symbol_count": reused,
        "supplemental_prehistory_symbol_count": int(prehistory_symbol_count),
        "cache_file_count": len(list(cache_dir.glob("*.parquet"))),
        "snapshot_validation": validation,
        "cross_source_overlap": overlap,
        "inputs": {
            contract["file"]: sha256_file(ROOT / contract["file"])
            for contract in contracts.values()
        },
        "output_file": config["artifacts"]["snapshot_prices"],
        "output_sha256": sha256_file(output),
        "source_documentation": config["source"]["official_documentation"],
        "governance": {
            "current_constituent_file_mutated": False,
            "return_calculation_performed": False,
            "ic_calculation_performed": False,
            "position_mapping_performed": False,
            "order_generation_performed": False,
        },
    }
    write_report(report, config)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"点时估值价格补采失败：{sanitized_error(exc)}", file=sys.stderr, flush=True)
        raise SystemExit(1)
