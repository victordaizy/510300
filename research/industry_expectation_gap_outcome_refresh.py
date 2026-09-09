"""行业预期差冻结原点的可变结果输入构造与质量闸门。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd


class OutcomeRefreshError(ValueError):
    """可变结果输入不满足日期、覆盖或独立交叉核验。"""


def trading_dates_between(
    calendar: pd.DataFrame,
    start_exclusive: Any,
    end_inclusive: Any,
) -> list[pd.Timestamp]:
    """返回两个日期之间严格晚于起点、且不晚于终点的交易日。"""

    column = "trade_date" if "trade_date" in calendar.columns else "date"
    if column not in calendar.columns:
        raise OutcomeRefreshError("交易日历缺少trade_date或date列")
    dates = pd.to_datetime(calendar[column], errors="coerce").dropna().dt.normalize()
    if dates.duplicated().any():
        raise OutcomeRefreshError("交易日历存在重复日期")
    start = pd.Timestamp(start_exclusive).normalize()
    end = pd.Timestamp(end_inclusive).normalize()
    if end < start:
        raise OutcomeRefreshError("目标日期早于现有数据日期")
    return sorted(pd.Timestamp(value) for value in dates.loc[(dates > start) & (dates <= end)])


def build_etf_total_return_extension(
    existing: pd.DataFrame,
    raw_extension: pd.DataFrame,
    dividends: pd.DataFrame,
) -> pd.DataFrame:
    """按冻结口径为新增ETF日线生成分红总回报和隔夜/盘中分解。"""

    required_existing = {
        "date",
        "close",
        "total_return",
        "cash_dividend_per_share",
    }
    required_raw = {
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "symbol",
        "source",
        "volume_unit",
        "amount_unit",
        "retrieved_at",
    }
    if missing := sorted(required_existing.difference(existing.columns)):
        raise OutcomeRefreshError(f"ETF总回报表缺少字段：{missing}")
    if missing := sorted(required_raw.difference(raw_extension.columns)):
        raise OutcomeRefreshError(f"ETF新增原始日线缺少字段：{missing}")
    base = existing.copy()
    base["date"] = pd.to_datetime(base["date"], errors="coerce").dt.normalize()
    raw = raw_extension.copy()
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce").dt.normalize()
    if base["date"].isna().any() or raw["date"].isna().any():
        raise OutcomeRefreshError("ETF日期解析失败")
    if raw.empty:
        return raw.reindex(columns=existing.columns)
    if raw["date"].duplicated().any():
        raise OutcomeRefreshError("ETF新增日线存在重复日期")
    if raw["date"].min() <= base["date"].max():
        raise OutcomeRefreshError("ETF新增日线不得覆盖既有日期")
    price_columns = ["open", "high", "low", "close"]
    raw[price_columns] = raw[price_columns].apply(pd.to_numeric, errors="coerce")
    if raw[price_columns].isna().any(axis=None) or raw[price_columns].le(0).any(axis=None):
        raise OutcomeRefreshError("ETF新增OHLC存在空值、零值或负值")
    if (raw["high"] < raw[["open", "close", "low"]].max(axis=1)).any():
        raise OutcomeRefreshError("ETF新增最高价违反OHLC约束")
    if (raw["low"] > raw[["open", "close", "high"]].min(axis=1)).any():
        raise OutcomeRefreshError("ETF新增最低价违反OHLC约束")

    events = dividends[["ex_date", "cash_dividend_per_share"]].copy()
    events["ex_date"] = pd.to_datetime(events["ex_date"], errors="coerce").dt.normalize()
    events["cash_dividend_per_share"] = pd.to_numeric(
        events["cash_dividend_per_share"], errors="coerce"
    )
    if events.isna().any(axis=None):
        raise OutcomeRefreshError("ETF分红事件存在空值")
    dividend_by_date = events.groupby("ex_date")["cash_dividend_per_share"].sum()

    extension = raw.sort_values("date").reset_index(drop=True)
    extension["cash_dividend_per_share"] = (
        extension["date"].map(dividend_by_date).fillna(0.0)
    )
    prior_closes = pd.concat(
        [
            pd.Series([float(base.sort_values("date").iloc[-1]["close"])]),
            extension["close"].iloc[:-1].reset_index(drop=True),
        ],
        ignore_index=True,
    )
    extension["prev_close"] = prior_closes.to_numpy()
    extension["price_return"] = extension["close"] / extension["prev_close"] - 1.0
    extension["dividend_yield"] = (
        extension["cash_dividend_per_share"] / extension["prev_close"]
    )
    extension["total_return"] = (
        extension["close"] + extension["cash_dividend_per_share"]
    ) / extension["prev_close"] - 1.0
    extension["overnight_price_return"] = extension["open"] / extension["prev_close"] - 1.0
    extension["overnight_total_contribution"] = (
        extension["open"]
        + extension["cash_dividend_per_share"]
        - extension["prev_close"]
    ) / extension["prev_close"]
    extension["intraday_return"] = extension["close"] / extension["open"] - 1.0
    extension["intraday_contribution"] = (
        extension["close"] - extension["open"]
    ) / extension["prev_close"]
    extension["decomposition_error"] = (
        extension["total_return"]
        - extension["overnight_total_contribution"]
        - extension["intraday_contribution"]
    )
    extension["range_return"] = extension["high"] / extension["low"] - 1.0
    extension["year"] = extension["date"].dt.year
    if extension["decomposition_error"].abs().max() > 1e-12:
        raise OutcomeRefreshError("ETF新增总回报分解不守恒")
    missing_output = sorted(set(existing.columns).difference(extension.columns))
    if missing_output:
        raise OutcomeRefreshError(f"ETF新增总回报缺少既有字段：{missing_output}")
    return extension.loc[:, existing.columns].copy()


def merge_constituent_extension(
    existing: pd.DataFrame,
    extension: pd.DataFrame,
    *,
    required_constituent_count: int = 300,
) -> pd.DataFrame:
    """只追加完整的证券日期面板，不允许覆盖历史行。"""

    required = {
        "date",
        "con_code",
        "raw_open",
        "raw_high",
        "raw_low",
        "raw_close",
        "total_return_open",
        "total_return_high",
        "total_return_low",
        "total_return_close",
    }
    if missing := sorted(required.difference(existing.columns)):
        raise OutcomeRefreshError(f"既有成分股面板缺少字段：{missing}")
    if missing := sorted(set(existing.columns).difference(extension.columns)):
        raise OutcomeRefreshError(f"成分股新增面板缺少既有字段：{missing}")
    base = existing.copy()
    new = extension.loc[:, existing.columns].copy()
    base["date"] = pd.to_datetime(base["date"], errors="coerce").dt.normalize()
    new["date"] = pd.to_datetime(new["date"], errors="coerce").dt.normalize()
    if new.empty:
        return base
    if new[["date", "con_code"]].duplicated().any():
        raise OutcomeRefreshError("成分股新增面板存在重复证券日期")
    if new["date"].min() <= base["date"].max():
        raise OutcomeRefreshError("成分股新增面板不得覆盖既有日期")
    counts = new.groupby("date")["con_code"].nunique()
    if not counts.eq(required_constituent_count).all():
        raise OutcomeRefreshError(f"成分股新增面板覆盖异常：{counts.to_dict()}")
    prices = [
        "raw_open",
        "raw_high",
        "raw_low",
        "raw_close",
        "total_return_open",
        "total_return_high",
        "total_return_low",
        "total_return_close",
    ]
    if new[prices].isna().any(axis=None) or new[prices].le(0).any(axis=None):
        raise OutcomeRefreshError("成分股新增价格存在空值、零值或负值")
    combined = pd.concat([base, new], ignore_index=True)
    if combined[["date", "con_code"]].duplicated().any():
        raise OutcomeRefreshError("合并后成分股面板存在重复证券日期")
    return combined.sort_values(["date", "con_code"]).reset_index(drop=True)


def validate_close_crosscheck(
    primary: pd.DataFrame,
    independent: pd.DataFrame,
    *,
    target_date: Any,
    maximum_close_difference: float,
    required_constituent_count: int = 300,
) -> dict[str, Any]:
    """以独立供应商收盘价核对目标日300只证券。"""

    target = pd.Timestamp(target_date).normalize()
    left = primary.copy()
    right = independent.copy()
    left["date"] = pd.to_datetime(left["date"], errors="coerce").dt.normalize()
    right["date"] = pd.to_datetime(right["date"], errors="coerce").dt.normalize()
    left = left.loc[left["date"].eq(target), ["con_code", "raw_close"]]
    right = right.loc[right["date"].eq(target), ["con_code", "raw_close"]]
    if left["con_code"].nunique() != required_constituent_count:
        raise OutcomeRefreshError("主来源目标日证券数不足")
    if right["con_code"].nunique() != required_constituent_count:
        raise OutcomeRefreshError("独立来源目标日证券数不足")
    comparison = left.merge(
        right,
        on="con_code",
        how="inner",
        suffixes=("_primary", "_independent"),
        validate="one_to_one",
    )
    difference = (
        pd.to_numeric(comparison["raw_close_primary"], errors="coerce")
        - pd.to_numeric(comparison["raw_close_independent"], errors="coerce")
    ).abs()
    if difference.isna().any():
        raise OutcomeRefreshError("独立收盘价交叉核验存在空值")
    maximum = float(difference.max())
    if maximum > maximum_close_difference + 1e-12:
        raise OutcomeRefreshError(
            f"独立收盘价最大差异{maximum}超过{maximum_close_difference}"
        )
    return {
        "status": "PASS",
        "target_date": target.date().isoformat(),
        "comparable_count": int(len(comparison)),
        "maximum_absolute_close_difference": maximum,
        "exact_match_ratio": float(np.isclose(difference, 0.0, atol=1e-12).mean()),
    }


__all__ = [
    "OutcomeRefreshError",
    "build_etf_total_return_extension",
    "merge_constituent_extension",
    "trading_dates_between",
    "validate_close_crosscheck",
]
