"""审计000300沪深300指数原始日线。"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = PROJECT_ROOT / "config" / "settings.yaml"
DATA_FILE = PROJECT_ROOT / "data" / "raw" / "market" / "000300_daily_raw.parquet"
METADATA_FILE = PROJECT_ROOT / "data" / "raw" / "market" / "000300_daily_raw.metadata.json"
REPORT_DIR = PROJECT_ROOT / "reports" / "data_quality"
REPORT_FILE = REPORT_DIR / "000300_daily_quality.json"


def main() -> int:
    with CONFIG_FILE.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not DATA_FILE.exists():
        print(f"找不到数据文件: {DATA_FILE}", file=sys.stderr)
        return 1

    data = pd.read_parquet(DATA_FILE)
    errors: list[dict] = []
    warnings: list[dict] = []
    required = ["symbol", "date", "open", "high", "low", "close", "volume", "source", "retrieved_at"]
    for column in required:
        if column not in data.columns:
            errors.append({"code": "MISSING_COLUMN", "column": column})
    if not data.empty:
        dates = pd.to_datetime(data["date"], errors="coerce")
        if dates.isna().any():
            errors.append({"code": "INVALID_DATE", "count": int(dates.isna().sum())})
        if data["date"].duplicated().any():
            errors.append({"code": "DUPLICATE_DATE", "count": int(data["date"].duplicated().sum())})
        if not dates.is_monotonic_increasing:
            errors.append({"code": "DATE_NOT_SORTED"})
        prices = ["open", "high", "low", "close"]
        if data[prices].isna().any().any() or (data[prices] <= 0).any().any():
            errors.append({"code": "INVALID_PRICE"})
        if (data["high"] < data[["open", "close", "low"]].max(axis=1)).any():
            errors.append({"code": "OHLC_HIGH_INCONSISTENT"})
        if (data["low"] > data[["open", "close", "high"]].min(axis=1)).any():
            errors.append({"code": "OHLC_LOW_INCONSISTENT"})
        if (data["volume"] < 0).any():
            errors.append({"code": "NEGATIVE_VOLUME"})
        expected = config["symbols"]["index"]
        symbols = data["symbol"].dropna().astype(str).unique().tolist()
        if symbols != [expected]:
            errors.append({"code": "UNEXPECTED_SYMBOL", "expected": expected, "actual": symbols})

    metadata = json.loads(METADATA_FILE.read_text(encoding="utf-8")) if METADATA_FILE.exists() else {}
    status = "FAIL" if errors else ("WARN" if warnings else "PASS")
    report = {
        "status": status,
        "checked_at": datetime.now(ZoneInfo(config["project"]["timezone"])).isoformat(),
        "file": DATA_FILE.as_posix(),
        "row_count": int(len(data)),
        "actual_first_date": pd.to_datetime(data["date"]).min().date().isoformat() if not data.empty else None,
        "actual_last_date": pd.to_datetime(data["date"]).max().date().isoformat() if not data.empty else None,
        "metadata_sha256": metadata.get("sha256"),
        "errors": errors,
        "warnings": warnings,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if status != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
