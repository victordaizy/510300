"""下载并验证基于中证官方调样公告整理的沪深300历史成分区间。"""

from __future__ import annotations

import hashlib
import io
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import yaml


ROOT = Path(__file__).resolve().parents[1]
SETTINGS_FILE = ROOT / "config" / "settings.yaml"
MARKET_FILE = ROOT / "data" / "raw" / "market" / "510300_daily_raw.parquet"
CURRENT_FILE = ROOT / "data" / "raw" / "constituents" / "000300_current_weights.parquet"
OUTPUT_FILE = ROOT / "data" / "raw" / "constituents" / "000300_historical_membership_intervals.parquet"
REPORT_FILE = ROOT / "reports" / "data_quality" / "000300_historical_membership_status.json"
SOURCE_URL = "https://raw.githubusercontent.com/unliftedq/index-constitution/main/history/csi300.csv"
SOURCE_REPOSITORY = "https://github.com/unliftedq/index-constitution"
SOURCE_LICENSE = "MIT"


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _to_standard_symbol(value: str) -> str:
    text = str(value).strip().upper()
    if text.startswith("SH"):
        return f"{text[2:]}.SH"
    if text.startswith("SZ"):
        return f"{text[2:]}.SZ"
    raise ValueError(f"无法识别历史成分代码：{value}")


def active_constituents(intervals: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
    """退出日期为不再属于指数的首日，因此区间右端开。"""

    return intervals.loc[
        (intervals["opt_in"] <= date)
        & (intervals["opt_out"].isna() | (date < intervals["opt_out"]))
    ].copy()


def validate_intervals(
    intervals: pd.DataFrame,
    market_dates: pd.Series,
    current: pd.DataFrame,
) -> dict:
    required = {"symbol", "opt_in", "opt_out"}
    missing = sorted(required.difference(intervals.columns))
    if missing:
        raise ValueError(f"历史成分区间缺少字段：{missing}")
    if intervals[["symbol", "opt_in"]].duplicated().any():
        raise ValueError("历史成分存在重复证券加入日期")
    if (intervals.loc[intervals["opt_out"].notna(), "opt_out"] <= intervals.loc[intervals["opt_out"].notna(), "opt_in"]).any():
        raise ValueError("历史成分存在退出日期不晚于加入日期")
    counts: list[int] = []
    duplicate_active_dates: list[str] = []
    for date in pd.to_datetime(market_dates):
        active = active_constituents(intervals, pd.Timestamp(date))
        counts.append(int(len(active)))
        if active["symbol"].duplicated().any():
            duplicate_active_dates.append(str(pd.Timestamp(date).date()))
    if duplicate_active_dates:
        raise ValueError(f"同一证券存在重叠成员区间：{duplicate_active_dates[:5]}")
    if not counts or min(counts) != 300 or max(counts) != 300:
        raise ValueError(f"项目交易日成分数量不恒为300：最少{min(counts)}，最多{max(counts)}")
    current_date = pd.Timestamp(current["effective_date"].max())
    reconstructed = set(active_constituents(intervals, current_date)["symbol"])
    official_current = set(current["symbol"].astype(str))
    only_history = sorted(reconstructed - official_current)
    only_official = sorted(official_current - reconstructed)
    if only_history or only_official:
        raise ValueError(f"历史区间与中证当前快照不一致：历史独有={only_history}，官网独有={only_official}")
    return {
        "interval_count": int(len(intervals)),
        "unique_symbol_count": int(intervals["symbol"].nunique()),
        "project_market_day_count": int(len(counts)),
        "min_active_constituents": int(min(counts)),
        "max_active_constituents": int(max(counts)),
        "official_current_snapshot_date": str(current_date.date()),
        "official_current_set_difference_count": 0,
    }


def main() -> int:
    for path in (SETTINGS_FILE, MARKET_FILE, CURRENT_FILE):
        if not path.exists():
            raise FileNotFoundError(f"历史成分验证缺少输入：{path}")
    settings = yaml.safe_load(SETTINGS_FILE.read_text(encoding="utf-8"))
    response = requests.get(SOURCE_URL, timeout=60)
    response.raise_for_status()
    raw = response.content
    source = pd.read_csv(io.BytesIO(raw))
    source.columns = [str(column).strip().lower().replace("-", "_") for column in source.columns]
    intervals = pd.DataFrame(
        {
            "symbol": source["symbol"].map(_to_standard_symbol),
            "opt_in": pd.to_datetime(source["opt_in"], errors="raise"),
            "opt_out": pd.to_datetime(source["opt_out"], errors="coerce"),
            "source_symbol": source["symbol"].astype(str),
            "source": SOURCE_REPOSITORY,
            "source_license": SOURCE_LICENSE,
        }
    ).sort_values(["symbol", "opt_in"]).reset_index(drop=True)
    market = pd.read_parquet(MARKET_FILE)
    market["date"] = pd.to_datetime(market["date"])
    start = pd.Timestamp(settings["project"]["start_date"])
    end = pd.Timestamp(settings["project"]["end_date"])
    market_dates = market.loc[market["date"].between(start, end), "date"]
    current = pd.read_parquet(CURRENT_FILE)
    evidence = validate_intervals(intervals, market_dates, current)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    intervals.to_parquet(OUTPUT_FILE, index=False)
    report = {
        "status": "PASS_MEMBERSHIP_ONLY_NO_HISTORICAL_WEIGHTS",
        "checked_at": datetime.now(ZoneInfo(settings["project"]["timezone"])).isoformat(),
        "source_url": SOURCE_URL,
        "source_repository": SOURCE_REPOSITORY,
        "source_license": SOURCE_LICENSE,
        "source_method": "中证官方历次调样公告标准化区间；项目另与中证官网当前成分权重快照做集合交叉验证",
        "source_sha256": _sha256_bytes(raw),
        "output_file": OUTPUT_FILE.relative_to(ROOT).as_posix(),
        **evidence,
        "allowed_use": "点时成分等权Breadth与成分集合研究",
        "prohibited_use": "不得称为官方历史权重，不得替代自由流通调整权重",
    }
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
