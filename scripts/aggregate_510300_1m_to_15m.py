"""将510300五年1分钟K线聚合为统一的15分钟研究序列，并进行多层对账。"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = PROJECT_ROOT / "config" / "settings.yaml"
ONE_MINUTE_FILE = PROJECT_ROOT / "data" / "raw" / "market" / "510300_1m_tushare_raw.parquet"
DAILY_FILE = PROJECT_ROOT / "data" / "raw" / "market" / "510300_daily_raw.parquet"
TDX_15M_FILE = PROJECT_ROOT / "data" / "raw" / "market" / "510300_15m_tdx_direct_raw.parquet"
SINA_15M_FILE = PROJECT_ROOT / "data" / "raw" / "market" / "510300_15m_raw.parquet"
OUTPUT_FILE = PROJECT_ROOT / "data" / "raw" / "market" / "510300_15m_from_1m_raw.parquet"
METADATA_FILE = OUTPUT_FILE.with_suffix(".metadata.json")
REPORT_FILE = PROJECT_ROOT / "reports" / "data_quality" / "510300_15m_from_1m_quality.json"
TICK_SIZE = 0.001


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_parquet(data: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    data.to_parquet(temporary, index=False, engine="pyarrow")
    temporary.replace(path)


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def assign_bar_end_label(trade_time: pd.Series) -> pd.Series:
    """映射1分钟交易时间到标准15分钟结束标签。

    09:30开盘集合竞价记录并入首根09:45柱；其余记录向上归入
    09:45、10:00、……、11:30和13:15、……、15:00。这是研究聚合规则，
    不声称提供方trade_time字段本身是bar start或bar end。
    """

    normalized_date = trade_time.dt.normalize()
    minute_of_day = trade_time.dt.hour * 60 + trade_time.dt.minute
    morning = minute_of_day.between(570, 690)
    afternoon = minute_of_day.between(781, 900)
    if not (morning | afternoon).all():
        unexpected = trade_time.loc[~(morning | afternoon)].dt.strftime("%Y-%m-%d %H:%M:%S").head().tolist()
        raise ValueError(f"存在非预期交易时间：{unexpected}")
    morning_end = np.where(
        minute_of_day == 570,
        585,
        ((minute_of_day + 14) // 15) * 15,
    )
    afternoon_end = ((minute_of_day - 780 + 14) // 15) * 15 + 780
    end_minute = np.where(morning, morning_end, afternoon_end)
    return normalized_date + pd.to_timedelta(end_minute, unit="m")


def aggregate_one_minute(data: pd.DataFrame, retrieved_at: str) -> pd.DataFrame:
    required = {"ts_code", "trade_time", "open", "high", "low", "close", "vol", "amount"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"1分钟数据缺少字段：{sorted(missing)}")
    output = data.copy()
    output["trade_time"] = pd.to_datetime(output["trade_time"], errors="raise")
    output = output.sort_values("trade_time").reset_index(drop=True)
    output["bar_end"] = assign_bar_end_label(output["trade_time"])
    result = (
        output.groupby("bar_end", as_index=False, sort=True)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("vol", "sum"),
            amount=("amount", "sum"),
            source_minute_count=("trade_time", "count"),
        )
        .sort_values("bar_end")
        .reset_index(drop=True)
    )
    result["trade_date"] = result["bar_end"].dt.normalize()
    result["symbol"] = "510300.SH"
    result["frequency_minutes"] = 15
    result["adjustment"] = "raw"
    result["source"] = "tushare_proxy.stk_mins.1m_aggregated"
    result["source_grade"] = "cross_validated_unofficial_proxy"
    result["retrieved_at"] = retrieved_at
    return result[
        [
            "symbol",
            "trade_date",
            "bar_end",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
            "source_minute_count",
            "frequency_minutes",
            "adjustment",
            "source",
            "source_grade",
            "retrieved_at",
        ]
    ]


def price_comparison(left: pd.DataFrame, right: pd.DataFrame, suffix: str) -> dict:
    merged = left.merge(right, on="bar_end", how="inner", suffixes=("_derived", f"_{suffix}"))
    result: dict[str, object] = {
        "matched_bars": int(len(merged)),
        "first_bar": merged["bar_end"].min().isoformat() if not merged.empty else None,
        "last_bar": merged["bar_end"].max().isoformat() if not merged.empty else None,
        "price": {},
    }
    for column in ["open", "high", "low", "close"]:
        difference = (merged[f"{column}_derived"] - merged[f"{column}_{suffix}"]).abs()
        result["price"][column] = {
            "exact_rate": float((difference < 1e-12).mean()),
            "within_one_tick_rate": float((difference <= TICK_SIZE + 1e-12).mean()),
            "maximum_absolute_error": float(difference.max()),
        }
    for column in ["volume", "amount"]:
        denominator = merged[f"{column}_{suffix}"].replace(0, np.nan).abs()
        relative = (merged[f"{column}_derived"] - merged[f"{column}_{suffix}"]).abs() / denominator
        result[f"maximum_relative_{column}_difference"] = float(relative.max())
        result[f"median_relative_{column}_difference"] = float(relative.median())
    return result


def daily_reconciliation(data: pd.DataFrame, daily: pd.DataFrame) -> dict:
    aggregated = (
        data.groupby("trade_date", as_index=False)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
            amount=("amount", "sum"),
        )
        .rename(columns={"trade_date": "date"})
    )
    reference = daily[["date", "open", "high", "low", "close", "volume", "amount"]].copy()
    reference["date"] = pd.to_datetime(reference["date"]).dt.normalize()
    merged = aggregated.merge(reference, on="date", how="outer", suffixes=("_15m", "_daily"), indicator=True)
    price = {}
    for column in ["open", "high", "low", "close"]:
        difference = (merged[f"{column}_15m"] - merged[f"{column}_daily"]).abs()
        price[column] = {
            "exact_days": int((difference < 1e-12).sum()),
            "mismatch_over_one_tick_days": int((difference > TICK_SIZE + 1e-12).sum()),
            "maximum_absolute_error": float(difference.max()),
        }
    output = {"matched_dates": int((merged["_merge"] == "both").sum()), "price": price}
    for column in ["volume", "amount"]:
        denominator = merged[f"{column}_daily"].replace(0, np.nan).abs()
        relative = (merged[f"{column}_15m"] - merged[f"{column}_daily"]).abs() / denominator
        output[f"maximum_relative_{column}_difference"] = float(relative.max())
    output["unmatched_dates"] = merged.loc[merged["_merge"] != "both", "date"].dt.strftime("%Y-%m-%d").tolist()
    return output


def main() -> int:
    config = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    retrieved_at = datetime.now(ZoneInfo(config["project"]["timezone"])).isoformat()
    one_minute = pd.read_parquet(ONE_MINUTE_FILE)
    derived = aggregate_one_minute(one_minute, retrieved_at)
    expected_dates = pd.to_datetime(pd.read_parquet(DAILY_FILE)["date"]).dt.normalize()
    actual_dates = derived["trade_date"].drop_duplicates()
    bars_per_day = derived.groupby("trade_date").size()
    minute_counts = derived["source_minute_count"].value_counts().sort_index()
    errors: list[str] = []
    if set(expected_dates) != set(actual_dates):
        errors.append("15分钟聚合交易日与日线不一致")
    if not (bars_per_day == 16).all():
        errors.append("存在非16根15分钟K线的交易日")
    expected_minute_counts = {15: 15 * len(expected_dates), 16: len(expected_dates)}
    actual_minute_counts = {int(key): int(value) for key, value in minute_counts.items()}
    if actual_minute_counts != expected_minute_counts:
        errors.append(f"1分钟对15分钟分组计数异常：{actual_minute_counts}")

    daily = pd.read_parquet(DAILY_FILE)
    daily_check = daily_reconciliation(derived, daily)
    if daily_check["unmatched_dates"]:
        errors.append("15分钟与日线存在日期覆盖差异")
    if any(item["mismatch_over_one_tick_days"] for item in daily_check["price"].values()):
        errors.append("15分钟聚合价格与日线存在超过1个最小价位的差异")

    tdx = pd.read_parquet(TDX_15M_FILE)
    tdx["bar_end"] = pd.to_datetime(tdx["bar_end"])
    sina = pd.read_parquet(SINA_15M_FILE)
    sina["bar_end"] = pd.to_datetime(sina["bar_end"])
    tdx_check = price_comparison(derived, tdx, "tdx")
    sina_check = price_comparison(derived, sina, "sina")
    status = "FAIL" if errors else "PASS_WITH_SOURCE_CAVEATS"
    atomic_parquet(derived, OUTPUT_FILE)
    checksum = sha256_file(OUTPUT_FILE)
    report = {
        "status": status,
        "checked_at": retrieved_at,
        "file": OUTPUT_FILE.relative_to(PROJECT_ROOT).as_posix(),
        "sha256": checksum,
        "coverage": {
            "row_count": int(len(derived)),
            "trading_day_count": int(derived["trade_date"].nunique()),
            "first_bar_end_label": derived["bar_end"].min().isoformat(),
            "last_bar_end_label": derived["bar_end"].max().isoformat(),
            "bars_per_day_distribution": {
                str(key): int(value) for key, value in bars_per_day.value_counts().sort_index().items()
            },
            "source_minute_count_distribution": {
                str(key): int(value) for key, value in minute_counts.items()
            },
        },
        "daily_reconciliation": daily_check,
        "cross_source_15m": {"tdx_direct": tdx_check, "sina_recent": sina_check},
        "errors": errors,
        "warnings": [
            {
                "code": "UNOFFICIAL_PROXY",
                "impact": "1分钟数据经第三方代理获取，不等同于直连官方TuShare主站。",
            },
            {
                "code": "API_NAME_DOCUMENTATION_MISMATCH",
                "impact": "代理请求名为stk_mins，当前TuShare官方ETF文档记载的接口名为etf_mins。",
            },
            {
                "code": "TRADE_TIME_SEMANTICS_NOT_PROVIDER_DEFINED",
                "impact": "官方文档仅称trade_time为交易时间；15分钟分组规则是本项目明确的研究约定。",
            },
            {
                "code": "NO_POST_CLOSE_FIXED_PRICE_SESSION",
                "impact": "序列仅到15:00，不包含2026-07-06以后15:05至15:30盘后固定价格交易。",
            },
        ],
        "research_use": {
            "preferred_five_year_15m_ohlcv": not errors,
            "reason": "五年日期完整，聚合后每日OHLCVA与双源日线对账，并与TDX原生15分钟区间交叉比较。",
            "execution_limit": "仍不得假设能以柱内high/low成交；不包含真实Level-2、PV/LC或排队成交数据。",
        },
    }
    atomic_json(report, REPORT_FILE)
    metadata = {
        "status": status,
        "symbol": "510300.SH",
        "frequency_minutes": 15,
        "adjustment": "raw_unadjusted",
        "row_count": int(len(derived)),
        "trading_day_count": int(derived["trade_date"].nunique()),
        "actual_first_bar_end_label": derived["bar_end"].min().isoformat(),
        "actual_last_bar_end_label": derived["bar_end"].max().isoformat(),
        "source": "tushare_proxy.stk_mins.1m_aggregated",
        "source_file": ONE_MINUTE_FILE.relative_to(PROJECT_ROOT).as_posix(),
        "aggregation_convention": "09:30集合竞价并入09:45柱，其余1分钟记录向上归入刻度15分钟结束标签。",
        "sha256": checksum,
        "retrieved_at": retrieved_at,
    }
    atomic_json(metadata, METADATA_FILE)
    print(json.dumps({"metadata": metadata, "quality": report}, ensure_ascii=False, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
