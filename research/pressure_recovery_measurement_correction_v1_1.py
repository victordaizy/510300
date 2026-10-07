"""修正成交/申报滚动窗口单位，分解移动价格带的深度变化；不生成策略。"""

from __future__ import annotations

import numpy as np
import pandas as pd


FLOW_COLUMNS = ["trade_volume_5m_source_units", "reported_bs_volume_imbalance_5m"] + [
    f"reported_order_{kind}_{side}_volume_5m" for kind in ("A", "D") for side in ("B", "S")
]


def trailing_values(messages: pd.DataFrame, ends, weights) -> np.ndarray:
    """严格累计(t−5分钟,t]，两侧显式转为纳秒，兼容微秒与纳秒输入。"""
    message_index = pd.DatetimeIndex(messages["_at"])
    endpoint_index = pd.DatetimeIndex(ends)
    if message_index.tz is None or endpoint_index.tz is None:
        raise ValueError("消息和截止时间都必须明确时区。")
    if message_index.hasnans or endpoint_index.hasnans:
        raise ValueError("不接受缺失时间。")
    if not message_index.is_monotonic_increasing:
        raise ValueError("消息必须按时间排序。")
    values = np.asarray(weights, dtype=float)
    if values.ndim != 1 or len(values) != len(message_index) or not np.isfinite(values).all():
        raise ValueError("权重长度或有效性不符合要求。")
    times = message_index.as_unit("ns").asi8
    end = endpoint_index.as_unit("ns").asi8
    begin = end - pd.Timedelta(minutes=5).value
    cumulative = np.r_[0.0, np.cumsum(values)]
    return cumulative[np.searchsorted(times, end, side="right")] - cumulative[np.searchsorted(times, begin, side="right")]


def correct_flow_columns(grid: pd.DataFrame, orders: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    result = grid.copy()
    volume = trades.volume.to_numpy(dtype=float)
    total = trailing_values(trades, result.decision_at, volume)
    signed = volume * np.where(trades.bs_flag.eq("B"), 1, np.where(trades.bs_flag.eq("S"), -1, 0))
    balance = trailing_values(trades, result.decision_at, signed)
    result["trade_volume_5m_source_units"] = total
    result["reported_bs_volume_imbalance_5m"] = np.divide(balance, total, out=np.full(len(total), np.nan), where=total > 0)
    for kind in ("A", "D"):
        for side in ("B", "S"):
            values = orders.volume.where(orders.order_type.eq(kind) & orders.order_code.eq(side), 0)
            result[f"reported_order_{kind}_{side}_volume_5m"] = trailing_values(orders, result.decision_at, values)
    result["reported_bs_direction_independently_validated"] = False
    return result


def bid_book(row: pd.Series) -> dict[int, float]:
    prices = [int(row[f"bid_px{i}"]) for i in range(1, 11)]
    sizes = [float(row[f"bid_vol{i}"]) for i in range(1, 11)]
    if any(p <= 0 for p in prices) or any(prices[i] <= prices[i + 1] for i in range(9)):
        raise ValueError("十档买价非正或不是严格降序。")
    if any(not np.isfinite(q) or q < 0 for q in sizes):
        raise ValueError("买盘份额缺失或为负。")
    return dict(zip(prices, sizes))


def band_value(book: dict[int, float], lower: float, upper: float) -> float:
    return sum(price * size / 10000 for price, size in book.items() if lower <= price <= upper)


def band_visible(book: dict[int, float], lower: float) -> bool:
    """下界被十档覆盖，或整个区间在买一以上时，可确认金额含零。"""
    return min(book) <= lower or max(book) < lower


def depth_decomposition(start: pd.Series, end: pd.Series, band_bps: float = 10) -> dict:
    """共同绝对价位区间净变化与价格带换位精确相加；不推断补单/撤单。"""
    if not 0 < band_bps < 10000:
        raise ValueError("价格带宽度必须在0与10000基点之间。")
    first, last = bid_book(start), bid_book(end)
    midpoint0 = (float(start.bid_px1) + float(start.ask_px1)) / 2
    midpoint1 = (float(end.bid_px1) + float(end.ask_px1)) / 2
    lower0, lower1 = midpoint0 * (1 - band_bps / 10000), midpoint1 * (1 - band_bps / 10000)
    overlap_low, overlap_high = max(lower0, lower1), min(midpoint0, midpoint1)
    first_value = band_value(first, lower0, midpoint0)
    last_value = band_value(last, lower1, midpoint1)
    common0 = band_value(first, overlap_low, overlap_high)
    common1 = band_value(last, overlap_low, overlap_high)
    common_change = common1 - common0
    entered = last_value - common1
    left = first_value - common0
    fixed_end_visible = band_visible(last, lower0)
    fixed_end = band_value(last, lower0, midpoint0) if fixed_end_visible else None
    return {
        "start_band_low_cny": lower0 / 10000, "start_band_high_cny": midpoint0 / 10000,
        "end_band_low_cny": lower1 / 10000, "end_band_high_cny": midpoint1 / 10000,
        "start_moving_depth_cny": first_value, "end_moving_depth_cny": last_value,
        "moving_depth_ratio": last_value / first_value if first_value > 0 else None,
        "moving_depth_change_cny": last_value - first_value,
        "same_absolute_band_net_change_cny": common_change,
        "new_band_part_end_depth_cny": entered, "old_band_part_start_depth_cny": left,
        "band_shift_net_component_cny": entered - left,
        "band_overlap_fraction": max(0.0, overlap_high - overlap_low) / (midpoint0 - lower0),
        "both_moving_bands_fully_visible": band_visible(first, lower0) and band_visible(last, lower1),
        "initial_band_at_endpoint_fully_visible": fixed_end_visible,
        "initial_band_endpoint_depth_cny": fixed_end,
        "initial_band_depth_ratio": fixed_end / first_value if first_value > 0 and fixed_end is not None else None,
        "new_orders_or_cancellations_identified": False,
    }
