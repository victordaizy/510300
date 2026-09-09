"""下载点时全ETF母表、后复权日线和H00300，支持断点续传。"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import akshare as ak
import pandas as pd
import tushare as ts
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.download_000300_constituent_daily_sina import (  # noqa: E402
    AMOUNT_URL,
    HFQ_URL,
    HIST_URL,
    _fetch_text,
    _parse_downloaded_payloads,
    _to_sina_symbol,
)
from scripts.download_h00300_total_return import normalize_columns  # noqa: E402

CONFIG_FILE = ROOT / "config" / "csi300_all_etf_momentum_alpha_v1.yaml"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def credentials() -> tuple[str, str]:
    load_dotenv(ROOT / ".env")
    value = os.getenv("TUSHARE_TOKEN") or os.getenv("TS_TOKEN")
    endpoint = os.getenv("TUSHARE_API_URL", "https://api.tushare.pro").strip().rstrip("/")
    if not value:
        value = os.getenv("TUSHARE_PROXY_TOKEN")
        endpoint = os.getenv("TUSHARE_PROXY_URL", "https://fast.xiaodefa.cn").strip().rstrip("/")
    if not value:
        raise RuntimeError("未设置TUSHARE_TOKEN、TS_TOKEN或TUSHARE_PROXY_TOKEN")
    if endpoint not in {"https://api.tushare.pro", "https://fast.xiaodefa.cn", "https://tt.xiaodefa.cn"}:
        raise ValueError("Tushare接口地址不在项目已审核允许名单")
    return value, endpoint


def fetch_master(contract: dict) -> pd.DataFrame:
    secret, endpoint = credentials()
    ts.set_token(secret)
    api = ts.pro_api()
    api._DataApi__http_url = endpoint
    fields = "ts_code,name,fund_type,invest_type,type,status,list_date,delist_date,market"
    chunks = []
    for status in contract["universe"]["statuses"]:
        part = api.fund_basic(market=contract["universe"]["exchange_market"], status=status, fields=fields)
        if part is not None and not part.empty:
            chunks.append(part)
        time.sleep(max(float(os.getenv("TUSHARE_REQUEST_INTERVAL_SECONDS", "0.8")), 0.5))
    if not chunks:
        raise RuntimeError("fund_basic未返回任何交易所基金")
    master = pd.concat(chunks, ignore_index=True).drop_duplicates("ts_code", keep="first")
    master["list_date"] = pd.to_datetime(master["list_date"], errors="coerce")
    master["delist_date"] = pd.to_datetime(master["delist_date"], errors="coerce")
    end = pd.Timestamp(contract["periods"]["evaluation_end"])
    start = pd.Timestamp(contract["periods"]["evaluation_start"])
    names = master["name"].fillna("").astype(str)
    exchange = master["ts_code"].astype(str).str.endswith((".SH", ".SZ"))
    etf = names.str.contains(contract["universe"]["name_must_contain_case_insensitive"], case=False, regex=False)
    excluded = names.str.contains(contract["universe"]["risk_name_exclusion_regex"], regex=True)
    date_possible = master["list_date"].notna() & master["list_date"].le(end - pd.Timedelta(days=360))
    overlaps = master["delist_date"].isna() | master["delist_date"].ge(start)
    defensive = str(contract["universe"]["defensive_asset"])
    keep = exchange & overlaps & ((etf & ~excluded & date_possible) | master["ts_code"].eq(defensive))
    master = master.loc[keep].copy().sort_values("ts_code").reset_index(drop=True)
    if defensive not in set(master["ts_code"]):
        raise ValueError("防御ETF不在fund_basic母表")
    return master


def download_one(symbol: str, start: str, end: str, checkpoint_dir: Path) -> pd.DataFrame:
    checkpoint = checkpoint_dir / f"{symbol.replace('.', '_')}.parquet"
    if checkpoint.exists():
        cached = pd.read_parquet(checkpoint)
        if not cached.empty and {"date", "con_code", "raw_open", "total_return_close", "amount"}.issubset(cached.columns):
            return cached
    sina = _to_sina_symbol(symbol)
    error: Exception | None = None
    for attempt in range(3):
        try:
            frame = _parse_downloaded_payloads(
                symbol,
                _fetch_text(HIST_URL.format(sina)),
                _fetch_text(AMOUNT_URL.format(sina, sina)),
                _fetch_text(HFQ_URL.format(sina)),
                start,
                end,
            )
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            temporary = checkpoint.with_suffix(".parquet.tmp")
            frame.to_parquet(temporary, index=False)
            temporary.replace(checkpoint)
            return frame
        except Exception as exc:
            error = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"{type(error).__name__}: {error}")


def main() -> int:
    contract = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    inputs, periods = contract["inputs"], contract["periods"]
    master = fetch_master(contract)
    master_path = ROOT / inputs["master"]
    master_path.parent.mkdir(parents=True, exist_ok=True)
    master.to_parquet(master_path, index=False)
    print(f"点时基金母表筛得 {len(master)} 只ETF（含退市基金与防御ETF）", flush=True)

    checkpoint_dir = ROOT / inputs["checkpoint_directory"]
    results: dict[str, pd.DataFrame] = {}
    failures: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = {
            executor.submit(download_one, code, periods["warmup_start"], periods["evaluation_end"], checkpoint_dir): code
            for code in master["ts_code"].astype(str)
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            code = futures[future]
            try:
                results[code] = future.result()
            except Exception as exc:
                failures[code] = str(exc)
            if completed % 25 == 0 or completed == len(futures):
                print(f"进度 {completed}/{len(futures)}，成功 {len(results)}，失败 {len(failures)}", flush=True)
    if failures:
        failure_path = (ROOT / inputs["directory"] / "download_failures.json")
        failure_path.write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
        raise RuntimeError(f"点时ETF宇宙下载不完整：失败{len(failures)}只，样例={dict(list(failures.items())[:10])}")
    panel = pd.concat(results.values(), ignore_index=True).sort_values(["date", "con_code"])
    if panel[["date", "con_code"]].duplicated().any():
        raise ValueError("ETF面板存在重复键")
    panel_path = ROOT / inputs["panel"]
    panel.to_parquet(panel_path, index=False)

    benchmark = ak.stock_zh_index_hist_csindex(
        symbol="H00300",
        start_date=pd.Timestamp(periods["warmup_start"]).strftime("%Y%m%d"),
        end_date=pd.Timestamp(periods["evaluation_end"]).strftime("%Y%m%d"),
    )
    benchmark = normalize_columns(benchmark)
    benchmark["date"] = pd.to_datetime(benchmark["date"], errors="coerce")
    benchmark["close"] = pd.to_numeric(benchmark["close"], errors="coerce")
    benchmark = benchmark.loc[
        benchmark["date"].between(pd.Timestamp(periods["warmup_start"]), pd.Timestamp(periods["evaluation_end"])),
        ["date", "symbol", "name", "close"],
    ].dropna().drop_duplicates("date").sort_values("date")
    benchmark["source"] = "akshare.stock_zh_index_hist_csindex"
    benchmark_path = ROOT / inputs["benchmark"]
    benchmark.to_parquet(benchmark_path, index=False)

    status = {
        "status": "PASS",
        "checked_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "formula_frozen_before_fund_master_and_return_download": True,
        "master_rows": int(len(master)), "downloaded_symbols": int(panel["con_code"].nunique()),
        "panel_rows": int(len(panel)), "failed_symbols": 0,
        "listed_status_counts": master["status"].value_counts(dropna=False).to_dict(),
        "delisted_fund_count": int(master["delist_date"].notna().sum()),
        "sources": {"master": "tushare.fund_basic", "price": "新浪历史日线", "adjustment": "新浪后复权因子", "benchmark": "H00300经AkShare"},
        "hashes": {"master": sha256(master_path), "panel": sha256(panel_path), "benchmark": sha256(benchmark_path)},
    }
    status_path = ROOT / inputs["data_status"]
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
