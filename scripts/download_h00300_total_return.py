"""从中证指数历史接口下载沪深300全收益指数。"""

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
OUTPUT_FILE = PROJECT_ROOT / "data" / "raw" / "market" / "H00300_total_return_daily_raw.parquet"
METADATA_FILE = OUTPUT_FILE.with_suffix(".metadata.json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repair_text(value: object) -> object:
    """修复供应商响应偶发的 latin1/GBK 乱码。"""
    if not isinstance(value, str):
        return value
    try:
        return value.encode("latin1").decode("gbk")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return value


def normalize_columns(data: pd.DataFrame) -> pd.DataFrame:
    normalized = data.copy()
    normalized.columns = [repair_text(column) for column in normalized.columns]
    return normalized.rename(
        columns={
            "日期": "date",
            "指数代码": "symbol",
            "指数中文全称": "name",
            "收盘": "close",
            "涨跌": "change",
            "涨跌幅": "pct_change",
            "成交量": "volume",
            "成交金额": "amount",
            "样本数量": "constituent_count",
            "滚动市盈率": "pe_ttm",
        }
    )


def main() -> int:
    config = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    start = config["project"]["feature_warmup_start"]
    end = config["project"]["end_date"]
    try:
        data = ak.stock_zh_index_hist_csindex(
            symbol="H00300", start_date=start.replace("-", ""), end_date=end.replace("-", "")
        )
        data = normalize_columns(data)
        required = ["date", "symbol", "name", "close", "pct_change"]
        if missing := set(required).difference(data.columns):
            raise ValueError(f"全收益指数缺少字段：{sorted(missing)}")
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
        numeric_candidates = [
            "close", "change", "pct_change", "volume", "amount", "constituent_count", "pe_ttm"
        ]
        numeric = [column for column in numeric_candidates if column in data]
        data[numeric] = data[numeric].apply(pd.to_numeric, errors="coerce")
        data["name"] = data["name"].map(repair_text)
        data = data[[*required, *[column for column in numeric if column not in required]]]
        data = data.sort_values("date").reset_index(drop=True)
        if data.empty or data[["date", "symbol", "close", "pct_change"]].isna().any().any():
            raise ValueError("全收益指数为空或必要字段缺失")
        if data["date"].duplicated().any():
            raise ValueError("全收益指数日期重复")
        if data["symbol"].astype(str).unique().tolist() != ["H00300"] or (data["close"] <= 0).any():
            raise ValueError("全收益指数代码或收盘点位异常")
        retrieved_at = datetime.now(ZoneInfo(config["project"]["timezone"])).isoformat()
        data["source"] = "akshare.stock_zh_index_hist_csindex"
        data["retrieved_at"] = retrieved_at
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        temp_file = OUTPUT_FILE.with_suffix(".parquet.tmp")
        data.to_parquet(temp_file, index=False)
        temp_file.replace(OUTPUT_FILE)
        metadata = {
            "symbol": "H00300",
            "name": "沪深300全收益指数",
            "file": OUTPUT_FILE.relative_to(PROJECT_ROOT).as_posix(),
            "row_count": int(len(data)),
            "actual_first_date": str(data["date"].min().date()),
            "actual_last_date": str(data["date"].max().date()),
            "sha256": sha256_file(OUTPUT_FILE),
            "source": data["source"].iloc[0],
            "retrieved_at": retrieved_at,
            "price_fields": ["close"],
        }
        METADATA_FILE.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"全收益指数下载失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
