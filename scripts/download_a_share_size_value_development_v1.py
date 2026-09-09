"""下载2015—2026开发期每20个交易日的全A点时估值快照。"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import tushare as ts


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.download_csi300_all_etf_momentum_v1 import credentials  # noqa: E402
from scripts.run_csi300_etf_rotation_alpha_v1 import atomic_parquet, atomic_text  # noqa: E402


OUTPUT_DIR = ROOT / "data/raw/a_share_size_value_development_v1"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
OUTPUT_PANEL = OUTPUT_DIR / "daily_basic_signal_dates.parquet"
OUTPUT_STATUS = ROOT / "reports/data_quality/a_share_size_value_development_v1_status.json"
FIELDS = "ts_code,trade_date,close,turnover_rate,pe_ttm,pb,dv_ttm,total_share,float_share,total_mv,circ_mv"


def signal_dates(path: Path, start: str, end: str) -> list[pd.Timestamp]:
    benchmark = pd.read_parquet(path, columns=["date"])
    calendar = pd.DatetimeIndex(sorted(pd.to_datetime(benchmark["date"]).unique()))
    selected = calendar[(calendar >= pd.Timestamp(start)) & (calendar <= pd.Timestamp(end))][::20]
    return list(selected)


def download_one(api: object, date: pd.Timestamp) -> pd.DataFrame:
    checkpoint = CHECKPOINT_DIR / f"{date:%Y%m%d}.parquet"
    if checkpoint.exists():
        return pd.read_parquet(checkpoint)
    frame = api.daily_basic(trade_date=f"{date:%Y%m%d}", fields=FIELDS)
    if frame is None or frame.empty:
        raise RuntimeError(f"{date:%Y-%m-%d} daily_basic为空")
    required = set(FIELDS.split(","))
    if missing := required.difference(frame.columns):
        raise RuntimeError(f"{date:%Y-%m-%d}缺字段：{sorted(missing)}")
    frame = frame[list(FIELDS.split(","))].copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    atomic_parquet(frame, checkpoint)
    return frame


def main() -> int:
    old_dates = signal_dates(
        ROOT / "data/raw/a_share_hash_holdout_v2/training_H00300.parquet",
        "2015-01-05",
        "2023-11-30",
    )
    recent_dates = signal_dates(
        ROOT / "data/raw/a_share_bucket34_time_holdout_stratified_lowvol_v1/H00300.parquet",
        "2024-01-02",
        "2026-08-14",
    )
    dates = sorted(set(old_dates + recent_dates))
    secret, default_endpoint = credentials()
    ts.set_token(secret)
    api = None
    endpoint_used = None
    for endpoint in dict.fromkeys([default_endpoint, "https://tt.xiaodefa.cn", "https://fast.xiaodefa.cn"]):
        candidate = ts.pro_api()
        candidate._DataApi__http_url = endpoint
        try:
            probe = candidate.daily_basic(trade_date="20240102", fields="ts_code,trade_date,total_mv")
            if probe is not None and not probe.empty:
                api = candidate
                endpoint_used = endpoint
                break
        except Exception:
            continue
    if api is None:
        raise RuntimeError("daily_basic数据节点不可用")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[pd.Timestamp, pd.DataFrame] = {}
    failures: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(download_one, api, date): date for date in dates}
        for completed, future in enumerate(as_completed(futures), start=1):
            date = futures[future]
            try:
                results[date] = future.result()
            except Exception as exc:
                failures[f"{date:%Y-%m-%d}"] = str(exc)
            if completed % 20 == 0 or completed == len(futures):
                print(f"估值快照 {completed}/{len(futures)}，成功 {len(results)}，失败 {len(failures)}", flush=True)
    if failures:
        raise RuntimeError(f"估值快照下载失败：{dict(list(failures.items())[:10])}")
    panel = pd.concat(results.values(), ignore_index=True).sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
    if panel[["trade_date", "ts_code"]].duplicated().any():
        raise RuntimeError("估值快照存在重复证券日期")
    numeric = ["close", "turnover_rate", "pe_ttm", "pb", "dv_ttm", "total_share", "float_share", "total_mv", "circ_mv"]
    panel[numeric] = panel[numeric].apply(pd.to_numeric, errors="coerce")
    if panel[["close", "total_mv", "circ_mv"]].isna().any().any():
        raise RuntimeError("核心价格或市值字段存在缺失")
    if panel[["close", "total_mv", "circ_mv"]].le(0).any().any():
        raise RuntimeError("核心价格或市值字段非正")
    atomic_parquet(panel, OUTPUT_PANEL)
    coverage = panel.groupby("trade_date").agg(
        stocks=("ts_code", "nunique"),
        pe_coverage=("pe_ttm", lambda values: float(values.notna().mean())),
        pb_coverage=("pb", lambda values: float(values.notna().mean())),
        dividend_coverage=("dv_ttm", lambda values: float(values.notna().mean())),
    )
    status = {
        "status": "PASS",
        "checked_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "api_endpoint": endpoint_used,
        "development_only": True,
        "untouched_2005_2013_read": False,
        "snapshot_dates": len(dates),
        "rows": len(panel),
        "date_min": str(panel["trade_date"].min().date()),
        "date_max": str(panel["trade_date"].max().date()),
        "minimum_stock_count": int(coverage["stocks"].min()),
        "minimum_pe_coverage": float(coverage["pe_coverage"].min()),
        "minimum_pb_coverage": float(coverage["pb_coverage"].min()),
        "minimum_dividend_coverage": float(coverage["dividend_coverage"].min()),
    }
    atomic_text(json.dumps(status, ensure_ascii=False, indent=2), OUTPUT_STATUS)
    print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
