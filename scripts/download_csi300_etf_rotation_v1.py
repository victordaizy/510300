"""下载冻结资产池的原始与后复权ETF日线及H00300基准。"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import akshare as ak
import pandas as pd
import yaml

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

CONFIG_FILE = ROOT / "config" / "csi300_etf_rotation_alpha_v1.yaml"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    contract = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    periods, universe, paths = contract["periods"], contract["universe"], contract["inputs"]
    start, end = periods["warmup_start"], periods["evaluation_end"]
    symbols = [*universe["risk_assets"].keys(), *universe["defensive_asset"].keys()]
    frames = []
    for symbol in symbols:
        sina = _to_sina_symbol(symbol)
        frame = _parse_downloaded_payloads(
            symbol,
            _fetch_text(HIST_URL.format(sina)),
            _fetch_text(AMOUNT_URL.format(sina, sina)),
            _fetch_text(HFQ_URL.format(sina)),
            start,
            end,
        )
        frames.append(frame)
        print(f"已下载并校验 {symbol}：{len(frame)} 行", flush=True)
    panel = pd.concat(frames, ignore_index=True).sort_values(["date", "con_code"])
    panel_path = ROOT / paths["etf_panel"]
    panel_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(panel_path, index=False)

    benchmark = ak.stock_zh_index_hist_csindex(
        symbol="H00300",
        start_date=pd.Timestamp(start).strftime("%Y%m%d"),
        end_date=pd.Timestamp(end).strftime("%Y%m%d"),
    )
    benchmark = normalize_columns(benchmark)
    benchmark["date"] = pd.to_datetime(benchmark["date"], errors="coerce")
    benchmark["close"] = pd.to_numeric(benchmark["close"], errors="coerce")
    benchmark = benchmark.loc[
        benchmark["date"].between(pd.Timestamp(start), pd.Timestamp(end)),
        ["date", "symbol", "name", "close"],
    ].dropna().drop_duplicates("date").sort_values("date")
    if benchmark.empty or benchmark["close"].le(0).any():
        raise ValueError("H00300全收益基准不可用")
    benchmark["source"] = "akshare.stock_zh_index_hist_csindex"
    benchmark_path = ROOT / paths["benchmark"]
    benchmark.to_parquet(benchmark_path, index=False)

    first_dates = {symbol: str(pd.Timestamp(frame["date"].min()).date()) for symbol, frame in panel.groupby("con_code")}
    last_dates = {symbol: str(pd.Timestamp(frame["date"].max()).date()) for symbol, frame in panel.groupby("con_code")}
    status = {
        "status": "PASS",
        "checked_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "formula_frozen_before_download": True,
        "symbols": symbols,
        "panel_rows": int(len(panel)),
        "duplicate_symbol_dates": int(panel[["date", "con_code"]].duplicated().sum()),
        "first_dates": first_dates,
        "last_dates": last_dates,
        "benchmark_first": str(pd.Timestamp(benchmark["date"].min()).date()),
        "benchmark_last": str(pd.Timestamp(benchmark["date"].max()).date()),
        "sources": {"etf_price": "新浪历史日线", "etf_adjustment": "新浪后复权因子", "benchmark": "中证指数H00300接口经AkShare"},
        "hashes": {"panel": sha256(panel_path), "benchmark": sha256(benchmark_path)},
    }
    status_path = ROOT / paths["data_status"]
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
