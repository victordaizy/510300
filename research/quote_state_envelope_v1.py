"""以全部候选源状态形成逐价位外包络，不估计个人成交概率。"""

from __future__ import annotations

import numpy as np
import pandas as pd


def clock_ms(values) -> np.ndarray:
    numbers = np.asarray(values, dtype=np.int64)
    hours, minutes, seconds, milliseconds = numbers // 10000000, numbers // 100000 % 100, numbers // 1000 % 100, numbers % 1000
    if np.any((hours > 23) | (minutes > 59) | (seconds > 59) | (numbers < 0)):
        raise ValueError("源时间编码无效。")
    return ((hours * 60 + minutes) * 60 + seconds) * 1000 + milliseconds


def clock_windows(quotes: pd.DataFrame, trades: pd.DataFrame, quote_ms: int = 1000, tick_ms: int = 10) -> pd.DataFrame:
    positive = trades.loc[trades.volume.gt(0) & trades.time.lt(150500000)].sort_values("time", kind="stable")
    cumulative = positive.volume.cumsum().to_numpy(dtype=float)
    trade_time = clock_ms(positive.time)
    viewed = quotes.sort_values("time", kind="stable").copy()
    if len(cumulative) == 0 or np.any(np.diff(cumulative) <= 0):
        raise ValueError("无法形成正成交量前缀。")
    nominal = clock_ms(viewed.time)
    positions = np.searchsorted(cumulative, viewed.cum_volume.to_numpy(dtype=float), side="left")
    exact = (positions < len(cumulative)) & (cumulative[np.minimum(positions, len(cumulative) - 1)] == viewed.cum_volume.to_numpy(dtype=float))
    last = trade_time[np.minimum(positions, len(trade_time) - 1)]
    next_indices = positions + 1
    next_ceiling = np.where(next_indices < len(trade_time), trade_time[np.minimum(next_indices, len(trade_time) - 1)] + tick_ms, 24 * 3600 * 1000)
    lower = np.maximum(nominal, last)
    upper = np.minimum(nominal + quote_ms, next_ceiling)
    return pd.DataFrame({"source_index": viewed.index.to_numpy(), "nominal_ms": nominal,
                         "lower_ms": lower, "upper_exclusive_ms": upper, "exact_volume_prefix": exact,
                         "interval_nonempty": exact & (lower < upper)})


def batch_outer_bounds(start: np.ndarray, locations: np.ndarray, delta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """批内次序未知时给出包含所有可行中间数量的外界，不假定独立界可同时实现。"""
    positive, negative = np.zeros(len(start)), np.zeros(len(start))
    np.add.at(positive, locations, np.maximum(delta, 0))
    np.add.at(negative, locations, np.maximum(-delta, 0))
    lower = np.maximum(0, start - negative)
    upper = start + positive
    end = start + positive - negative
    if np.any(end < -1e-6):
        raise ValueError("时间批次结束后存在负库存，不能建立当前模型的外包络。")
    return lower, upper, np.maximum(0, end)


def scalar_features(prices: np.ndarray, bid_low: np.ndarray, bid_high: np.ndarray,
                    ask_low: np.ndarray, ask_high: np.ndarray, band_bps: float = 10) -> dict:
    guaranteed_bids, possible_bids = prices[bid_low > 1e-6], prices[bid_high > 1e-6]
    guaranteed_asks, possible_asks = prices[ask_low > 1e-6], prices[ask_high > 1e-6]
    if any(len(part) == 0 for part in [guaranteed_bids, possible_bids, guaranteed_asks, possible_asks]):
        return {"feature_status": "UNKNOWN_NO_GUARANTEED_TWO_SIDED_BOOK"}
    bid_min, bid_max = guaranteed_bids.max(), possible_bids.max()
    ask_min, ask_max = possible_asks.min(), guaranteed_asks.min()
    mid_min, mid_max = (bid_min + ask_min) / 2, (bid_max + ask_max) / 2
    common_low, common_high = mid_max * (1 - band_bps / 10000), mid_min
    union_low, union_high = mid_min * (1 - band_bps / 10000), mid_max
    common = (prices >= common_low) & (prices <= common_high)
    union = (prices >= union_low) & (prices <= union_high)
    return {"feature_status": "CONDITIONAL_OUTER_ENVELOPE", "mid_lower_cny": mid_min / 10000,
            "mid_upper_cny": mid_max / 10000,
            "spread_lower_ticks": max(1.0, (ask_min - bid_max) / 10),
            "spread_upper_ticks": (ask_max - bid_min) / 10,
            "bid_depth_lower_cny": float(np.dot(prices[common], bid_low[common]) / 10000),
            "bid_depth_upper_cny": float(np.dot(prices[union], bid_high[union]) / 10000)}


def quote_inside_box(row: pd.Series, prices: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> bool:
    width = len(prices)
    for side, offset in [("bid", 0), ("ask", width)]:
        selected_prices = np.array([int(row[f"{side}_px{i}"]) for i in range(1, 11)])
        selected_sizes = np.array([float(row[f"{side}_vol{i}"]) for i in range(1, 11)])
        indices = np.searchsorted(prices, selected_prices)
        if np.any(indices >= width) or not np.array_equal(prices[np.minimum(indices, width - 1)], selected_prices):
            return False
        if np.any(selected_sizes < lower[offset + indices] - 1e-6) or np.any(selected_sizes > upper[offset + indices] + 1e-6):
            return False
        inspected = prices >= selected_prices[-1] if side == "bid" else prices <= selected_prices[-1]
        unknown_visible = inspected & ~np.isin(prices, selected_prices)
        if np.any(lower[offset:offset + width][unknown_visible] > 1e-6):
            return False
        total = float(row["tot_bid_vol" if side == "bid" else "tot_ask_vol"])
        if total < lower[offset:offset + width].sum() - 1e-6 or total > upper[offset:offset + width].sum() + 1e-6:
            return False
    return True


def quote_envelopes(quotes: pd.DataFrame, trades: pd.DataFrame, events: pd.DataFrame, band_bps: float = 10) -> pd.DataFrame:
    selected = quotes.loc[((quotes.time >= 93000000) & (quotes.time < 113000000)) | ((quotes.time >= 130000000) & (quotes.time < 145700000))]
    selected = selected.loc[selected.bid_px1.gt(0) & selected.ask_px1.gt(selected.bid_px1)]
    windows = clock_windows(selected, trades)
    if np.any(events.time.mod(10).ne(0)) or np.any(selected.time.mod(1000).ne(0)):
        raise ValueError("源字段与预先声明的10毫秒/整秒网格不符。")
    order = np.argsort(events.time.to_numpy(), kind="stable")
    events = events.iloc[order]
    times = clock_ms(events.time)
    prices = np.sort(events.price.astype("int64").unique())
    width = len(prices)
    locations = np.searchsorted(prices, events.price.astype("int64").to_numpy()) + np.where(events.side.eq("B"), 0, width)
    delta = events.delta.to_numpy(dtype=float)
    current = np.zeros(width * 2)
    cursor, last_lower = 0, -1
    records = []
    for window in windows.itertuples():
        source = quotes.loc[window.source_index]
        record = {"date": int(source.date), "time": int(source.time), "source_quote_index": int(window.source_index),
                  "nominal_ms": window.nominal_ms, "lower_ms": window.lower_ms, "upper_exclusive_ms": window.upper_exclusive_ms,
                  "conditional_interval_width_ms": window.upper_exclusive_ms - window.lower_ms,
                  "clock_interval_valid": window.interval_nonempty, "source_snapshot_inside_outer_box": False,
                  "availability_at_nominal_time_proven": False, "joint_state_reconstructed": False}
        if not window.interval_nonempty:
            records.append(dict(record, feature_status="INCONSISTENT_CLOCK_INTERVAL"))
            continue
        if window.lower_ms < last_lower:
            raise ValueError("下界时钟不递增，不能沿用顺序重建。")
        last_lower = window.lower_ms
        begin = int(np.searchsorted(times, window.lower_ms, side="left"))
        np.add.at(current, locations[cursor:begin], delta[cursor:begin])
        cursor = begin
        if np.any(current < -1e-6):
            raise ValueError("窗口前库存为负。")
        current = np.maximum(0, current)
        lower, upper, state = current.copy(), current.copy(), current.copy()
        finish = int(np.searchsorted(times, window.upper_exclusive_ms, side="left"))
        subset_times = times[begin:finish]
        _, starts = np.unique(subset_times, return_index=True)
        ends = np.r_[starts[1:], len(subset_times)]
        for start, end in zip(starts, ends):
            span = slice(begin + start, begin + end)
            batch_low, batch_high, state = batch_outer_bounds(state, locations[span], delta[span])
            lower = np.minimum(lower, batch_low)
            upper = np.maximum(upper, batch_high)
        record["source_snapshot_inside_outer_box"] = quote_inside_box(source, prices, lower, upper)
        record.update(scalar_features(prices, lower[:width], upper[:width], lower[width:], upper[width:], band_bps))
        record["source_mid_cny"] = (float(source.bid_px1) + float(source.ask_px1)) / 20000
        records.append(record)
    return pd.DataFrame(records)


def event_comparison(start: pd.Series, end: pd.Series) -> dict:
    if not start.get("source_snapshot_inside_outer_box", True) or not end.get("source_snapshot_inside_outer_box", True):
        return {"comparison_status": "SOURCE_OUTSIDE_CONDITIONAL_BOUND_NO_INFERENCE"}
    if start.feature_status != "CONDITIONAL_OUTER_ENVELOPE" or end.feature_status != "CONDITIONAL_OUTER_ENVELOPE":
        return {"comparison_status": "UNKNOWN_FEATURE_RANGE"}
    price_low = (end.mid_lower_cny / start.mid_upper_cny - 1) * 10000
    price_high = (end.mid_upper_cny / start.mid_lower_cny - 1) * 10000
    depth_low = end.bid_depth_lower_cny / start.bid_depth_upper_cny if start.bid_depth_upper_cny > 0 else np.nan
    depth_high = end.bid_depth_upper_cny / start.bid_depth_lower_cny if start.bid_depth_lower_cny > 0 else np.nan
    spread_low = end.spread_lower_ticks - start.spread_upper_ticks
    spread_high = end.spread_upper_ticks - start.spread_lower_ticks
    return {"comparison_status": "CONDITIONAL_DESCRIPTIVE_BOUNDS_ONLY",
            "observation_price_change_lower_bps": price_low, "observation_price_change_upper_bps": price_high,
            "price_direction": "POSITIVE_THROUGHOUT_BOUND" if price_low > 1e-8 else "NEGATIVE_THROUGHOUT_BOUND" if price_high < -1e-8 else "ZERO_OR_DIRECTION_AMBIGUOUS",
            "moving_bid_depth_ratio_lower": depth_low, "moving_bid_depth_ratio_upper": depth_high,
            "moving_depth_direction": "INCREASE_THROUGHOUT_BOUND" if depth_low > 1 + 1e-8 else "DECREASE_THROUGHOUT_BOUND" if depth_high < 1 - 1e-8 else "UNCHANGED_OR_AMBIGUOUS",
            "spread_tick_change_lower": spread_low, "spread_tick_change_upper": spread_high,
            "spread_direction": "NARROWER_THROUGHOUT_BOUND" if spread_high < -1e-8 else "WIDER_THROUGHOUT_BOUND" if spread_low > 1e-8 else "UNCHANGED_OR_AMBIGUOUS"}
