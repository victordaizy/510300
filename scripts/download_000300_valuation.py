"""下载沪深300历史 PE/PB，并保留足够的回测预热期。"""

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
OUTPUT_FILE = PROJECT_ROOT / "data" / "raw" / "valuation" / "000300_valuation_daily_raw.parquet"
METADATA_FILE = OUTPUT_FILE.with_suffix(".metadata.json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_valuation() -> pd.DataFrame:
    pe = ak.stock_index_pe_lg(symbol="沪深300").rename(
        columns={
            "日期": "date", "指数": "index_close_pe_source", "静态市盈率": "pe_static",
            "滚动市盈率": "pe_ttm", "等权静态市盈率": "pe_static_equal_weight",
            "静态市盈率中位数": "pe_static_median", "等权滚动市盈率": "pe_ttm_equal_weight",
            "滚动市盈率中位数": "pe_ttm_median",
        }
    )
    pb = ak.stock_index_pb_lg(symbol="沪深300").rename(
        columns={
            "日期": "date", "指数": "index_close_pb_source", "市净率": "pb",
            "等权市净率": "pb_equal_weight", "市净率中位数": "pb_median",
        }
    )
    pe["date"] = pd.to_datetime(pe["date"], errors="coerce")
    pb["date"] = pd.to_datetime(pb["date"], errors="coerce")
    data = pe.merge(pb, on="date", how="outer", validate="one_to_one").sort_values("date")
    numeric_columns = [column for column in data.columns if column != "date"]
    data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors="coerce")
    data["symbol"] = "000300.SH"
    data["source"] = "akshare.stock_index_pe_lg+stock_index_pb_lg"
    data["retrieved_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    return data.reset_index(drop=True)


def validate(data: pd.DataFrame, warmup_start: str, end_date: str) -> None:
    required = ["date", "pe_static", "pe_ttm", "pb"]
    missing = [column for column in required if column not in data]
    if missing:
        raise ValueError(f"缺少估值字段：{missing}")
    if data.empty or data[required].isna().any().any():
        raise ValueError("估值数据为空或存在必要字段空值")
    if data["date"].duplicated().any() or not data["date"].is_monotonic_increasing:
        raise ValueError("估值日期重复或未升序")
    if (data[["pe_static", "pe_ttm", "pb"]] <= 0).any().any():
        raise ValueError("估值字段存在非正值")
    if data["date"].min() > pd.Timestamp(warmup_start):
        raise ValueError("估值预热历史不足")
    if (pd.Timestamp(end_date) - data["date"].max()).days > 7:
        raise ValueError("估值数据距回测结束日超过7天")


def main() -> int:
    config = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    warmup_start = config["project"]["feature_warmup_start"]
    end_date = config["project"]["end_date"]
    try:
        data = fetch_valuation()
        data = data.loc[data["date"].between(pd.Timestamp(warmup_start), pd.Timestamp(end_date))].copy()
        validate(data, warmup_start, end_date)
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        temp_file = OUTPUT_FILE.with_suffix(".parquet.tmp")
        data.to_parquet(temp_file, index=False)
        temp_file.replace(OUTPUT_FILE)
        metadata = {
            "symbol": "000300.SH", "file": OUTPUT_FILE.relative_to(PROJECT_ROOT).as_posix(),
            "row_count": int(len(data)), "actual_first_date": str(data["date"].min().date()),
            "actual_last_date": str(data["date"].max().date()), "sha256": sha256_file(OUTPUT_FILE),
            "source": data["source"].iloc[0], "retrieved_at": data["retrieved_at"].iloc[0],
        }
        METADATA_FILE.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"估值下载失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
