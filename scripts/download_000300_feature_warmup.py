"""下载沪深300特征预热行情；仅用于计算历史窗口，不扩展回测绩效期。"""

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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = PROJECT_ROOT / "config" / "settings.yaml"
OUTPUT_FILE = PROJECT_ROOT / "data" / "raw" / "market" / "000300_daily_feature_warmup.parquet"
METADATA_FILE = OUTPUT_FILE.with_suffix(".metadata.json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    config = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    start = pd.Timestamp(config["project"]["feature_warmup_start"])
    end = pd.Timestamp(config["project"]["end_date"])
    try:
        data = ak.stock_zh_index_daily(symbol="sh000300")
        required = ["date", "open", "high", "low", "close", "volume"]
        if data is None or data.empty or any(column not in data for column in required):
            raise ValueError("沪深300指数行情为空或字段缺失")
        data = data[required].copy()
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
        data[required[1:]] = data[required[1:]].apply(pd.to_numeric, errors="coerce")
        data = data.loc[data["date"].between(start, end)].sort_values("date").reset_index(drop=True)
        if data[required].isna().any().any() or data["date"].duplicated().any():
            raise ValueError("沪深300预热行情存在空值或重复日期")
        if data["date"].min() > start or (end - data["date"].max()).days > 7:
            raise ValueError("沪深300预热行情范围不足")
        data["symbol"] = "000300.SH"
        data["source"] = "akshare.stock_zh_index_daily"
        data["purpose"] = "feature_warmup_and_backtest"
        retrieved_at = datetime.now(ZoneInfo(config["project"]["timezone"])).isoformat()
        data["retrieved_at"] = retrieved_at
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        temp_file = OUTPUT_FILE.with_suffix(".parquet.tmp")
        data.to_parquet(temp_file, index=False)
        temp_file.replace(OUTPUT_FILE)
        metadata = {
            "symbol": "000300.SH", "file": OUTPUT_FILE.relative_to(PROJECT_ROOT).as_posix(),
            "purpose": "特征预热；绩效回测仍从 project.start_date 开始",
            "row_count": int(len(data)), "actual_first_date": str(data["date"].min().date()),
            "actual_last_date": str(data["date"].max().date()), "sha256": sha256_file(OUTPUT_FILE),
            "source": data["source"].iloc[0], "retrieved_at": retrieved_at,
        }
        METADATA_FILE.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"指数预热行情下载失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
