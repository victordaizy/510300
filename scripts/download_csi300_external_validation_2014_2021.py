"""下载独立早期外部验证所需的沪深300点时成员与行情，不覆盖现有五年数据。"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import akshare as ak
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.download_000300_constituent_daily_sina import (
    AMOUNT_URL,
    HFQ_URL,
    HIST_URL,
    _add_suspension_rows,
    _fetch_text,
    _parse_downloaded_payloads,
    _to_sina_symbol,
)
from scripts.download_h00300_total_return import normalize_columns


MEMBERSHIP_FILE = ROOT / "data" / "raw" / "constituents" / "000300_historical_membership_intervals.parquet"
WEIGHTS_FILE = ROOT / "data" / "raw" / "constituents" / "000300_historical_weights.parquet"
OUTPUT_ROOT = ROOT / "data" / "raw" / "external_validation" / "csi300_2014_2021"
CACHE_DIR = OUTPUT_ROOT / "component_history_cache"
PANEL_FILE = OUTPUT_ROOT / "000300_external_member_panel_2014_2021.parquet"
INDEX_FILE = OUTPUT_ROOT / "000300_price_index_2014_2021.parquet"
BENCHMARK_FILE = OUTPUT_ROOT / "H00300_total_return_2014_2021.parquet"
STATUS_FILE = ROOT / "reports" / "data_quality" / "csi300_external_validation_2014_2021_status.json"
WARMUP_START = pd.Timestamp("2014-07-01")
EVALUATION_START = pd.Timestamp("2015-01-05")
END_DATE = pd.Timestamp("2021-08-11")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint(symbol: str) -> Path:
    return CACHE_DIR / f"{symbol.replace('.', '_')}.parquet"


def _download_symbol(symbol: str) -> pd.DataFrame:
    checkpoint = _checkpoint(symbol)
    if checkpoint.exists():
        cached = pd.read_parquet(checkpoint)
        if (
            not cached.empty
            and pd.Timestamp(cached["date"].min()) <= END_DATE
            and pd.Timestamp(cached["date"].max()) >= min(END_DATE, pd.Timestamp(cached["date"].max()))
        ):
            return cached
    sina_symbol = _to_sina_symbol(symbol)
    error: Exception | None = None
    for attempt in range(3):
        try:
            data = _parse_downloaded_payloads(
                symbol,
                _fetch_text(HIST_URL.format(sina_symbol)),
                _fetch_text(AMOUNT_URL.format(sina_symbol, sina_symbol)),
                _fetch_text(HFQ_URL.format(sina_symbol)),
                str(WARMUP_START.date()),
                str(END_DATE.date()),
            )
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            temporary = checkpoint.with_suffix(".parquet.tmp")
            data.to_parquet(temporary, index=False)
            temporary.replace(checkpoint)
            return data
        except Exception as exc:
            error = exc
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"{symbol}下载失败：{type(error).__name__}: {error}")


def _download_indices() -> tuple[pd.DataFrame, pd.DataFrame]:
    price = ak.stock_zh_index_daily(symbol="sh000300")
    price["date"] = pd.to_datetime(price["date"], errors="coerce")
    price = price.loc[price["date"].between(WARMUP_START, END_DATE)].copy()
    price[["open", "high", "low", "close", "volume"]] = price[
        ["open", "high", "low", "close", "volume"]
    ].apply(pd.to_numeric, errors="coerce")
    price["symbol"] = "000300.SH"
    price["source"] = "akshare.stock_zh_index_daily"
    if price.empty or price[["date", "close"]].isna().any().any():
        raise ValueError("早期沪深300价格指数不可用")

    benchmark = ak.stock_zh_index_hist_csindex(
        symbol="H00300",
        start_date=WARMUP_START.strftime("%Y%m%d"),
        end_date=END_DATE.strftime("%Y%m%d"),
    )
    benchmark = normalize_columns(benchmark)
    benchmark["date"] = pd.to_datetime(benchmark["date"], errors="coerce")
    benchmark["close"] = pd.to_numeric(benchmark["close"], errors="coerce")
    benchmark = benchmark.loc[
        benchmark["date"].between(WARMUP_START, END_DATE), ["date", "symbol", "name", "close"]
    ].copy()
    benchmark["source"] = "akshare.stock_zh_index_hist_csindex"
    benchmark = benchmark.dropna(subset=["date", "close"]).drop_duplicates("date")
    if benchmark.empty or benchmark["close"].le(0).any():
        raise ValueError("早期H00300全收益指数不可用")
    return price.sort_values("date"), benchmark.sort_values("date")


def _build_hybrid_membership(intervals: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    """2016-08前用区间，此后用当时已发布的最近月度权重快照。"""

    snapshots = sorted(
        pd.to_datetime(
            weights.loc[
                pd.to_datetime(weights["trade_date"]).between(WARMUP_START, END_DATE),
                "trade_date",
            ].unique()
        )
    )
    if not snapshots:
        raise ValueError("外部验证区间没有历史权重快照")
    first_snapshot = pd.Timestamp(snapshots[0])
    early = intervals.loc[intervals["opt_in"].lt(first_snapshot)].copy()
    early["opt_out"] = early["opt_out"].where(
        early["opt_out"].notna() & early["opt_out"].lt(first_snapshot),
        first_snapshot,
    )
    early = early.loc[early["opt_out"].gt(WARMUP_START)].copy()
    early["membership_method"] = "THIRD_PARTY_INTERVAL_BEFORE_WEIGHT_HISTORY"

    weight_rows: list[dict] = []
    normalized = weights.copy()
    normalized["trade_date"] = pd.to_datetime(normalized["trade_date"])
    for index, snapshot in enumerate(snapshots):
        next_snapshot = snapshots[index + 1] if index + 1 < len(snapshots) else END_DATE + pd.Timedelta(days=1)
        codes = normalized.loc[normalized["trade_date"].eq(snapshot), "con_code"].astype(str).unique()
        if len(codes) != 300:
            raise ValueError(f"历史权重快照{snapshot.date()}不是300只")
        weight_rows.extend(
            {
                "symbol": code,
                "opt_in": pd.Timestamp(snapshot),
                "opt_out": pd.Timestamp(next_snapshot),
                "source_symbol": code,
                "source": "tushare.index_weight",
                "source_license": "Tushare数据许可",
                "membership_method": "LATEST_PRIOR_MONTHLY_WEIGHT_SNAPSHOT",
            }
            for code in codes
        )
    hybrid = pd.concat([early, pd.DataFrame(weight_rows)], ignore_index=True, sort=False)
    hybrid["opt_in"] = pd.to_datetime(hybrid["opt_in"])
    hybrid["opt_out"] = pd.to_datetime(hybrid["opt_out"])
    return hybrid


def main() -> int:
    intervals = pd.read_parquet(MEMBERSHIP_FILE)
    intervals["opt_in"] = pd.to_datetime(intervals["opt_in"])
    intervals["opt_out"] = pd.to_datetime(intervals["opt_out"])
    weights = pd.read_parquet(WEIGHTS_FILE)
    relevant = _build_hybrid_membership(intervals, weights)
    relevant = relevant.loc[
        relevant["opt_in"].le(END_DATE) & relevant["opt_out"].gt(EVALUATION_START)
    ].copy()
    symbols = sorted(relevant["symbol"].unique())
    print(f"早期外部验证需下载{len(symbols)}只历史成分，支持断点续传。", flush=True)
    results: dict[str, pd.DataFrame] = {}
    failures: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(_download_symbol, symbol): symbol for symbol in symbols}
        for completed, future in enumerate(as_completed(futures), start=1):
            symbol = futures[future]
            try:
                results[symbol] = future.result()
            except Exception as exc:
                failures[symbol] = str(exc)
            if completed % 25 == 0 or completed == len(futures):
                print(
                    f"进度{completed}/{len(futures)}，成功{len(results)}，失败{len(failures)}",
                    flush=True,
                )
    if failures:
        sample = dict(list(failures.items())[:10])
        raise RuntimeError(f"仍有{len(failures)}只下载失败：{sample}")

    price_index, benchmark = _download_indices()
    market_dates = price_index["date"].sort_values().reset_index(drop=True)
    downloaded = pd.concat(results.values(), ignore_index=True)
    panel = _add_suspension_rows(downloaded, relevant, market_dates)
    active = panel.loc[panel["is_index_member"]]
    counts = active.loc[active["date"].between(EVALUATION_START, END_DATE)].groupby("date")["con_code"].nunique()
    if counts.min() != 300 or counts.max() != 300:
        raise ValueError(f"点时成员日数量异常：{counts.min()}至{counts.max()}")
    evaluation = active.loc[active["date"].between(EVALUATION_START, END_DATE)]
    price_columns = ["raw_open", "raw_close", "total_return_open", "total_return_close"]
    missing_prices = int(evaluation[price_columns].isna().any(axis=1).sum())
    if missing_prices:
        sample = evaluation.loc[
            evaluation[price_columns].isna().any(axis=1), ["date", "con_code"]
        ].head(10)
        raise ValueError(f"正式外部验证成员行情缺失{missing_prices}行：{sample.to_dict('records')}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(PANEL_FILE, index=False)
    price_index.to_parquet(INDEX_FILE, index=False)
    benchmark.to_parquet(BENCHMARK_FILE, index=False)
    status = {
        "status": "PASS",
        "checked_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "warmup_start": str(WARMUP_START.date()),
        "evaluation_start": str(EVALUATION_START.date()),
        "end_date": str(END_DATE.date()),
        "symbol_count": len(symbols),
        "component_history_file_count": len(list(CACHE_DIR.glob("*.parquet"))),
        "panel_rows": int(len(panel)),
        "active_member_rows": int(len(active)),
        "minimum_members_per_day": int(counts.min()),
        "maximum_members_per_day": int(counts.max()),
        "evaluation_missing_price_rows": missing_prices,
        "membership_source": sorted(relevant["source"].dropna().astype(str).unique().tolist()),
        "membership_methods": sorted(relevant["membership_method"].dropna().astype(str).unique().tolist()),
        "membership_source_license": sorted(relevant["source_license"].dropna().astype(str).unique().tolist()),
        "component_price_source": "新浪历史日线与后复权因子",
        "benchmark_source": "中证指数H00300历史接口经AkShare",
        "hashes": {
            "membership": _sha256(MEMBERSHIP_FILE),
            "weights": _sha256(WEIGHTS_FILE),
            "panel": _sha256(PANEL_FILE),
            "price_index": _sha256(INDEX_FILE),
            "benchmark": _sha256(BENCHMARK_FILE),
        },
    }
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATUS_FILE.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
