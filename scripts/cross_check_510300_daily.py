"""使用腾讯行情对新浪 510300 五年日线做独立来源交叉核验。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import akshare as ak
import numpy as np
import pandas as pd
import yaml

from scripts.quality_check_daily import sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "raw" / "market" / "510300_daily_raw.parquet"
CONFIG_FILE = PROJECT_ROOT / "config" / "settings.yaml"
REPORT_FILE = PROJECT_ROOT / "reports" / "data_quality" / "510300_daily_cross_source.json"


def main() -> int:
    config = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    primary = pd.read_parquet(DATA_FILE).copy()
    primary["date"] = pd.to_datetime(primary["date"])
    start = primary["date"].min().strftime("%Y%m%d")
    end = primary["date"].max().strftime("%Y%m%d")
    secondary = ak.stock_zh_a_hist_tx(symbol="sh510300", start_date=start, end_date=end, adjust="", timeout=30)
    secondary["date"] = pd.to_datetime(secondary["date"])
    merged = primary.merge(secondary, on="date", how="outer", suffixes=("_primary", "_secondary"), indicator=True)

    price_differences: dict[str, dict] = {}
    price_failed = False
    for column in ["open", "high", "low", "close"]:
        difference = (merged[f"{column}_primary"] - merged[f"{column}_secondary"]).abs()
        count = int((difference > 0.0010001).sum())
        price_failed |= count > 0
        price_differences[column] = {"max_absolute": float(difference.max()), "over_one_tick": count}
    flow_differences: dict[str, dict] = {}
    flow_failed = False
    for column in ["volume", "amount"]:
        denominator = merged[f"{column}_primary"].replace(0, np.nan)
        relative = (merged[f"{column}_primary"] - merged[f"{column}_secondary"]).abs() / denominator
        count = int((relative > 0.01).sum())
        flow_failed |= count > 0
        flow_differences[column] = {"max_relative": float(relative.max()), "over_one_percent": count}

    unmatched = merged.loc[merged["_merge"] != "both", "date"].dt.strftime("%Y-%m-%d").tolist()
    status = "FAIL" if unmatched or price_failed or flow_failed else "PASS"
    report = {
        "status": status,
        "checked_at": datetime.now(ZoneInfo(config["project"]["timezone"])).isoformat(),
        "primary_source": "akshare.fund_etf_hist_sina",
        "secondary_source": "akshare.stock_zh_a_hist_tx",
        "primary_sha256": sha256_file(DATA_FILE),
        "overlap_rows": int((merged["_merge"] == "both").sum()),
        "unmatched_dates": unmatched,
        "price_differences": price_differences,
        "flow_differences": flow_differences,
    }
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
