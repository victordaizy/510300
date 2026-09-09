"""极端缩波后市场/个股状态转换研究 V2 的纯计算函数。

本模块只包含无外部写入的计算逻辑。它不生成交易信号、不连接券商，也不把
历史条件相关解释成已验证 alpha。所有时间序列百分位均只使用当前时点之前的
有效观测；第一次升波只按顺序穿越识别，禁止事后寻找局部最低点。
"""

from __future__ import annotations

from math import sqrt
from typing import Iterable

import numpy as np
import pandas as pd


ANNUALIZATION_DAYS = 252.0


def trailing_valid_midrank_percentile(
    values: np.ndarray | pd.Series,
    *,
    lookback_valid_values: int,
    minimum_valid_history: int,
) -> tuple[np.ndarray, np.ndarray]:
    """计算严格排除当前值的滚动经验中秩百分位。

    窗口按最近的有效值计数，缺失值不消耗有效历史长度。返回百分位和每个位置
    实际可用的历史有效值数量。
    """

    raw = np.asarray(values, dtype=float)
    percentile = np.full(raw.size, np.nan, dtype=float)
    history_count = np.zeros(raw.size, dtype=np.int32)
    valid_positions = np.flatnonzero(np.isfinite(raw))
    if valid_positions.size == 0:
        return percentile, history_count
    compact = pd.Series(raw[valid_positions], dtype=float)
    rolling = compact.rolling(
        lookback_valid_values + 1,
        min_periods=minimum_valid_history + 1,
    )
    rank_with_current = rolling.rank(method="average")
    count_with_current = rolling.count()
    denominator = count_with_current - 1.0
    compact_percentile = (rank_with_current - 1.0) / denominator
    percentile[valid_positions] = compact_percentile.to_numpy(dtype=float)
    history_count[valid_positions] = np.minimum(
        np.arange(valid_positions.size, dtype=np.int32),
        lookback_valid_values,
    )
    percentile[history_count < minimum_valid_history] = np.nan
    return percentile, history_count


def realized_volatility(
    log_returns: np.ndarray | pd.Series,
    *,
    window: int,
    minimum_valid: int | None = None,
) -> np.ndarray:
    """计算年化样本标准差；默认要求窗口内全部观测有效。"""

    if minimum_valid is None:
        minimum_valid = window
    return (
        pd.Series(np.asarray(log_returns, dtype=float))
        .rolling(window, min_periods=minimum_valid)
        .std(ddof=1)
        .to_numpy(dtype=float)
        * sqrt(ANNUALIZATION_DAYS)
    )


def classify_quintile(percentile: float) -> str:
    """按固定历史百分位边界分类，不使用样本内 qcut。"""

    if not np.isfinite(percentile):
        return "MISSING"
    if percentile <= 0.20:
        return "Q1"
    if percentile <= 0.40:
        return "Q2"
    if percentile <= 0.60:
        return "Q3"
    if percentile <= 0.80:
        return "Q4"
    return "Q5"


def compute_market_volatility_state(
    market: pd.DataFrame,
    *,
    percentile_lookback: int = 756,
    percentile_minimum_history: int = 252,
) -> pd.DataFrame:
    """从 H00300 全收益收盘构造点时市场波动水平和方向状态。"""

    required = {"date", "close"}
    missing = sorted(required - set(market.columns))
    if missing:
        raise ValueError(f"市场数据缺少字段：{', '.join(missing)}")
    result = market[["date", "close"]].copy()
    result["date"] = pd.to_datetime(result["date"]).dt.normalize()
    result["close"] = pd.to_numeric(result["close"], errors="coerce")
    result = result.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    close = result["close"].to_numpy(dtype=float)
    log_return = np.full(close.size, np.nan, dtype=float)
    valid = (close[1:] > 0) & (close[:-1] > 0)
    log_return[1:][valid] = np.log(close[1:][valid] / close[:-1][valid])
    rv5 = realized_volatility(log_return, window=5)
    rv20 = realized_volatility(log_return, window=20)
    rv120 = realized_volatility(log_return, window=120)
    ratio = np.full(close.size, np.nan, dtype=float)
    valid_ratio = np.isfinite(rv5) & np.isfinite(rv20) & (rv5 > 0) & (rv20 > 0)
    ratio[valid_ratio] = np.log(rv5[valid_ratio] / rv20[valid_ratio])
    percentile, history_count = trailing_valid_midrank_percentile(
        rv20,
        lookback_valid_values=percentile_lookback,
        minimum_valid_history=percentile_minimum_history,
    )
    previous_ratio = np.roll(ratio, 1)
    previous_ratio[0] = np.nan
    expansion_cross = (
        np.isfinite(previous_ratio)
        & np.isfinite(ratio)
        & (previous_ratio <= 0.0)
        & (ratio > 0.0)
    )
    result["h00300_log_return"] = log_return
    result["market_rv5"] = rv5
    result["market_rv20"] = rv20
    result["market_rv120"] = rv120
    result["market_vol_percentile_756"] = percentile
    result["market_vol_history_count"] = history_count
    result["market_vol_quintile"] = [classify_quintile(value) for value in percentile]
    result["market_log_rv5_div_rv20"] = ratio
    result["market_vol_falling"] = np.isfinite(ratio) & (ratio <= 0.0)
    result["market_vol_expanding"] = np.isfinite(ratio) & (ratio > 0.0)
    result["market_first_expansion_cross"] = expansion_cross
    result["market_low_and_falling"] = (
        np.isfinite(percentile) & (percentile <= 0.40) & result["market_vol_falling"]
    )
    result["data_available_at"] = result["date"] + pd.Timedelta(hours=15)
    result["data_status"] = np.where(
        np.isfinite(percentile) & np.isfinite(ratio),
        "POINT_IN_TIME_COMPLETE",
        "INSUFFICIENT_HISTORY_OR_MISSING",
    )
    return result


def first_cross_after_origins(
    cross_positions: Iterable[int],
    origin_positions: np.ndarray | pd.Series,
    *,
    maximum_offset: int,
) -> tuple[np.ndarray, np.ndarray]:
    """为每个原点寻找未来固定窗口内第一次已观测穿越。"""

    crosses = np.asarray(sorted(set(int(value) for value in cross_positions)), dtype=np.int64)
    origins = np.asarray(origin_positions, dtype=np.int64)
    first_position = np.full(origins.size, -1, dtype=np.int64)
    offset = np.full(origins.size, -1, dtype=np.int32)
    if crosses.size == 0:
        return first_position, offset
    location = np.searchsorted(crosses, origins + 1, side="left")
    in_range = location < crosses.size
    candidate = np.full(origins.size, -1, dtype=np.int64)
    candidate[in_range] = crosses[location[in_range]]
    valid = in_range & ((candidate - origins) <= maximum_offset)
    first_position[valid] = candidate[valid]
    offset[valid] = (candidate[valid] - origins[valid]).astype(np.int32)
    return first_position, offset


def classify_expansion_price_state(close: float, lower: float, upper: float) -> str:
    """按事件日冻结区间识别个股第一次升波时的价格方向。"""

    if not all(np.isfinite(value) for value in (close, lower, upper)):
        return "EXPANSION_PATH_UNRESOLVED"
    if close > upper:
        return "EXPANSION_UP_BREAK"
    if close < lower:
        return "EXPANSION_DOWN_BREAK"
    return "EXPANSION_WITHIN_RANGE"


def classify_expansion_lead_lag(stock_offset: float, market_offset: float) -> str:
    """分类市场与个股第一次升波的先后顺序。"""

    if not np.isfinite(stock_offset) or not np.isfinite(market_offset):
        return "ONE_SIDE_NOT_OBSERVED"
    if stock_offset < market_offset:
        return "STOCK_EXPANDS_FIRST"
    if stock_offset > market_offset:
        return "MARKET_EXPANDS_FIRST"
    return "SAME_DAY"


def compound_excess(stock_return: pd.Series | np.ndarray, benchmark_return: pd.Series | np.ndarray) -> np.ndarray:
    """计算复合比率超额收益，不使用简单收益差。"""

    stock = np.asarray(stock_return, dtype=float)
    benchmark = np.asarray(benchmark_return, dtype=float)
    result = np.full(np.broadcast(stock, benchmark).shape, np.nan, dtype=float)
    valid = np.isfinite(stock) & np.isfinite(benchmark) & (benchmark > -1.0)
    result[valid] = (1.0 + stock[valid]) / (1.0 + benchmark[valid]) - 1.0
    return result


def holm_adjust(p_values: Iterable[float]) -> np.ndarray:
    """对一个预冻结检验家族执行 Holm step-down 校正。"""

    raw = np.asarray(list(p_values), dtype=float)
    adjusted = np.full(raw.size, np.nan, dtype=float)
    valid_indices = np.flatnonzero(np.isfinite(raw))
    if valid_indices.size == 0:
        return adjusted
    order = valid_indices[np.argsort(raw[valid_indices], kind="mergesort")]
    running = 0.0
    count = order.size
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * raw[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def event_date_equal_mean(frame: pd.DataFrame, value_column: str) -> float:
    """先按唯一事件日等权，再对日期均值等权。"""

    valid = frame.dropna(subset=["event_date", value_column]).copy()
    if valid.empty:
        return float("nan")
    valid["event_date"] = pd.to_datetime(valid["event_date"]).dt.normalize()
    return float(valid.groupby("event_date", sort=True)[value_column].mean().mean())


def cumulative_quintile_membership(quintile: str) -> tuple[str, ...]:
    """返回预冻结累计加入样本的成员箱体。"""

    order = ("Q1", "Q2", "Q3", "Q4", "Q5")
    if quintile not in order:
        raise ValueError(f"未知市场波动五等分：{quintile}")
    return order[: order.index(quintile) + 1]
