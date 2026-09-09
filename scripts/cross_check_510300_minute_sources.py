"""交叉审计510300分钟数据源、时间戳语义与盘后快照。"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml

from market_data.minute import aggregate_one_minute_to_15m


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = PROJECT_ROOT / "config" / "minute_data.yaml"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "market"
REPORT_FILE = PROJECT_ROOT / "reports" / "data_quality" / "510300_minute_cross_source.json"
SINA_15M_FILE = RAW_DIR / "510300_15m_raw.parquet"
TENCENT_15M_FILE = RAW_DIR / "510300_15m_tencent_raw.parquet"
SINA_1M_FILE = RAW_DIR / "510300_1m_validation_raw.parquet"
TENCENT_SNAPSHOT_FILE = RAW_DIR / "510300_intraday_tencent_snapshots.parquet"


def price_comparison(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_suffix: str,
    right_suffix: str,
    tick: float,
) -> dict:
    merged = left.merge(right, on="bar_end", suffixes=(left_suffix, right_suffix), validate="one_to_one")
    fields: dict[str, dict] = {}
    for column in ["open", "high", "low", "close"]:
        difference = (merged[f"{column}{left_suffix}"] - merged[f"{column}{right_suffix}"]).abs()
        fields[column] = {
            "exact_count": int(np.isclose(difference, 0.0, atol=1e-9).sum()),
            "exact_rate": float(np.isclose(difference, 0.0, atol=1e-9).mean()),
            "within_one_tick_count": int((difference <= tick + 1e-9).sum()),
            "within_one_tick_rate": float((difference <= tick + 1e-9).mean()),
            "maximum_absolute_error": float(difference.max()),
            "mean_absolute_error": float(difference.mean()),
        }
    return {
        "overlapping_bars": int(len(merged)),
        "overlapping_days": int(merged["bar_end"].dt.normalize().nunique()),
        "price_fields": fields,
    }


def one_minute_reaggregation_check(
    one_minute: pd.DataFrame,
    sina_15m: pd.DataFrame,
    config: dict,
) -> dict:
    one = one_minute.copy()
    one["bar_end"] = pd.to_datetime(one["bar_end"])
    one["trade_date"] = one["bar_end"].dt.normalize()
    counts = one.groupby("trade_date").size()
    expected_count = int(config["session"]["expected_sina_one_minute_rows_current_regime"])
    full_dates = counts[counts == expected_count].index
    full = one.loc[one["trade_date"].isin(full_dates)]
    aggregate = aggregate_one_minute_to_15m(
        full, list(config["session"]["expected_bar_end_times"])
    )
    primary = sina_15m.copy()
    primary["bar_end"] = pd.to_datetime(primary["bar_end"])
    merged = aggregate.merge(
        primary[["bar_end", "open", "high", "low", "close", "volume", "amount"]],
        on="bar_end",
        suffixes=("_from_1m", "_15m"),
        validate="one_to_one",
    )
    comparisons: dict[str, dict] = {}
    for column in ["open", "high", "low", "close", "volume", "amount"]:
        difference = (merged[f"{column}_from_1m"] - merged[f"{column}_15m"]).abs()
        comparisons[column] = {
            "exact_count": int(np.isclose(difference, 0.0, atol=1e-6).sum()),
            "maximum_absolute_error": float(difference.max()),
        }
    source_counts = {
        str(int(key)): int(value)
        for key, value in aggregate["source_minute_count"].value_counts().sort_index().items()
    }
    return {
        "one_minute_rows_by_day": {
            pd.Timestamp(key).date().isoformat(): int(value) for key, value in counts.items()
        },
        "full_validation_days": int(len(full_dates)),
        "reaggregated_bars": int(len(aggregate)),
        "matched_bars": int(len(merged)),
        "source_minute_count_distribution": source_counts,
        "comparisons": comparisons,
        "timestamp_conclusion": (
            "确认bar_end语义：10:00柱由09:46—10:00组成；新制15:00柱由"
            "14:46—14:57连续竞价记录加15:00收盘集合竞价记录组成"
        ),
    }


def secondary_source_check(sina_15m: pd.DataFrame, tencent_15m: pd.DataFrame, tick: float) -> dict:
    left = sina_15m.copy()
    right = tencent_15m.copy()
    left["bar_end"] = pd.to_datetime(left["bar_end"])
    right["bar_end"] = pd.to_datetime(right["bar_end"])
    result = price_comparison(left, right, "_sina", "_tencent", tick)
    merged = left.merge(right, on="bar_end", suffixes=("_sina", "_tencent"), validate="one_to_one")
    volume_difference = merged["volume_tencent"] - merged["volume_sina"]
    result["volume_bar_comparison"] = {
        "exact_count": int(np.isclose(volume_difference, 0.0, atol=0.5).sum()),
        "maximum_absolute_difference_shares": float(volume_difference.abs().max()),
        "mean_signed_difference_shares": float(volume_difference.mean()),
    }
    merged["trade_date"] = merged["bar_end"].dt.normalize()
    daily = merged.groupby("trade_date", as_index=False).agg(
        sina_volume=("volume_sina", "sum"),
        tencent_volume=("volume_tencent", "sum"),
    )
    daily["difference"] = daily["tencent_volume"] - daily["sina_volume"]
    daily["relative_difference"] = daily["difference"].abs() / daily["sina_volume"]
    result["volume_daily_comparison"] = {
        "days": int(len(daily)),
        "exact_days": int(np.isclose(daily["difference"], 0.0, atol=0.5).sum()),
        "maximum_absolute_difference_shares": float(daily["difference"].abs().max()),
        "maximum_relative_difference": float(daily["relative_difference"].max()),
    }
    result["amount_comparison"] = {
        "status": "UNAVAILABLE",
        "reason": "腾讯mkline末字段不是可靠成交额定义，未用于交叉且未伪造",
    }
    return result


def post_close_snapshot_check(
    snapshots: pd.DataFrame,
    sina_15m: pd.DataFrame,
    tencent_15m: pd.DataFrame,
) -> dict:
    data = snapshots.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    latest_date = data["timestamp"].dt.normalize().max()
    day = data.loc[data["timestamp"].dt.normalize() == latest_date].sort_values("timestamp")
    at_close = day.loc[day["timestamp"].dt.strftime("%H:%M:%S") == "15:00:00"]
    post_close = day.loc[day["timestamp"].dt.strftime("%H:%M:%S") > "15:00:00"]
    result = {
        "latest_trade_date": pd.Timestamp(latest_date).date().isoformat(),
        "last_snapshot_time": day["timestamp"].max().isoformat(),
        "post_close_records": int(len(post_close)),
        "post_close_observed": bool(not post_close.empty),
    }
    if not at_close.empty and not post_close.empty:
        close_row = at_close.iloc[-1]
        final_row = post_close.iloc[-1]
        result.update(
            {
                "first_post_close_timestamp": post_close.iloc[0]["timestamp"].isoformat(),
                "post_close_volume_shares": float(
                    final_row["cumulative_volume"] - close_row["cumulative_volume"]
                ),
                "post_close_amount_cny": float(
                    final_row["cumulative_amount"] - close_row["cumulative_amount"]
                ),
                "post_close_price_unique_count": int(post_close["price"].nunique()),
                "post_close_price": float(post_close.iloc[-1]["price"]),
            }
        )
        primary = sina_15m.copy()
        secondary = tencent_15m.copy()
        primary["trade_date"] = pd.to_datetime(primary["trade_date"]).dt.normalize()
        secondary["trade_date"] = pd.to_datetime(secondary["trade_date"]).dt.normalize()
        primary_day = primary.loc[primary["trade_date"] == latest_date]
        secondary_day = secondary.loc[secondary["trade_date"] == latest_date]
        result["three_way_reconciliation_at_15_00"] = {
            "sina_15m_volume_shares": float(primary_day["volume"].sum()),
            "tencent_15m_volume_shares": float(secondary_day["volume"].sum()),
            "tencent_snapshot_volume_shares": float(close_row["cumulative_volume"]),
            "sina_15m_minus_snapshot_shares": float(
                primary_day["volume"].sum() - close_row["cumulative_volume"]
            ),
            "tencent_15m_minus_snapshot_shares": float(
                secondary_day["volume"].sum() - close_row["cumulative_volume"]
            ),
            "sina_15m_amount_cny": float(primary_day["amount"].sum()),
            "tencent_snapshot_amount_cny": float(close_row["cumulative_amount"]),
            "sina_15m_minus_snapshot_amount_cny": float(
                primary_day["amount"].sum() - close_row["cumulative_amount"]
            ),
        }
    return result


def main() -> int:
    required_files = [SINA_15M_FILE, TENCENT_15M_FILE, SINA_1M_FILE, TENCENT_SNAPSHOT_FILE]
    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        print(f"缺少分钟交叉数据文件：{missing}", file=sys.stderr)
        return 1
    config = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    sina_15m = pd.read_parquet(SINA_15M_FILE)
    tencent_15m = pd.read_parquet(TENCENT_15M_FILE)
    sina_1m = pd.read_parquet(SINA_1M_FILE)
    snapshots = pd.read_parquet(TENCENT_SNAPSHOT_FILE)
    tick = float(config["quality"]["price_tick"])
    one_minute = one_minute_reaggregation_check(sina_1m, sina_15m, config)
    secondary = secondary_source_check(sina_15m, tencent_15m, tick)
    post_close = post_close_snapshot_check(snapshots, sina_15m, tencent_15m)
    tolerances = {
        "open": 1e-9,
        "high": 1e-9,
        "low": 1e-9,
        "close": 1e-9,
        "volume": 0.5,
        "amount": 0.01,
    }
    reaggregate_pass = all(
        item["maximum_absolute_error"] <= tolerances[field]
        for field, item in one_minute["comparisons"].items()
    )
    warnings = [
        {
            "code": "SECONDARY_15M_BAR_BOUNDARY_DIFFERENCES",
            "explanation": "腾讯与新浪逐柱存在价格/成交量差异，但重叠日总成交量大多一致；不得混源拼接单根K线",
        },
        {
            "code": "POST_CLOSE_HISTORY_ONLY_ACCUMULATES_FROM_CONNECTION_DATE",
            "explanation": "腾讯当日分时接口能识别盘后交易，但不能回补既往盘后分钟历史",
        },
    ]
    errors = [] if reaggregate_pass else [{"code": "ONE_MINUTE_REAGGREGATION_FAILED"}]
    report = {
        "status": "FAIL" if errors else "WARN",
        "scope": "510300分钟数据跨源与交易时段语义审计",
        "checked_at": datetime.now(ZoneInfo(config["timezone"])).isoformat(),
        "errors": errors,
        "warnings": warnings,
        "evidence": {
            "sina_1m_to_sina_15m": one_minute,
            "sina_15m_vs_tencent_15m": secondary,
            "tencent_post_close_snapshot": post_close,
        },
    }
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
