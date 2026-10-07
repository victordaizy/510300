"""仅使用已存数据的价格压力探索；盘口假设与可验证交易严格分别记录。"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from research.pressure_recovery_v1 import book_features, commission, session


def with_clock(frame):
    result = frame.copy()
    text = result.date.astype(str) + result.time.astype(str).str.zfill(9)
    result["_at"] = pd.to_datetime(text, format="%Y%m%d%H%M%S%f").dt.tz_localize("Asia/Shanghai")
    return result.sort_values("_at", kind="stable").reset_index(drop=True)


def quote_at(quotes, at, after=False):
    """测量只向后匹配；假设入场明确使用观察结束之后的第一条报价。"""
    at = pd.Timestamp(at)
    index = int(quotes._at.searchsorted(at, side="right")) - (0 if after else 1)
    if index < 0 or index >= len(quotes):
        return None
    row = quotes.iloc[index]
    delay = (row._at - at).total_seconds()
    if abs(delay) > 5 or (after and delay <= 0) or (not after and delay > 0):
        return None
    if session(at) == "OTHER" or session(row._at) != session(at):
        return None
    return row


def levels(row, side):
    return [(float(row[f"{side}_px{i}"]) / 10000, float(row[f"{side}_vol{i}"])) for i in range(1, 11)]


def quoted_vwap(side, quantity):
    remaining, value = quantity, 0.0
    for price, available in side:
        used = min(remaining, available)
        value += price * used
        remaining -= used
        if remaining <= 0:
            return value / quantity
    return None


def visible_order_quote(row, budget):
    bids, asks = levels(row, "bid"), levels(row, "ask")
    if not math.isfinite(asks[0][0]) or asks[0][0] <= 0:
        return {"status": "INVALID_QUOTE", "quantity": 0}
    quantity = int(max(0, budget - 5) // (asks[0][0] * 100)) * 100
    while quantity > 0:
        buy = quoted_vwap(asks, quantity)
        if buy is None:
            return {"status": "INSUFFICIENT_VISIBLE_ASK_QUANTITY", "quantity": quantity}
        buy_cost = quantity * buy
        fee = commission(buy_cost)
        if buy_cost + fee <= budget + 1e-8:
            break
        quantity -= 100
    if quantity <= 0:
        return {"status": "BUDGET_BELOW_ONE_LOT_AND_FEE", "quantity": 0}
    sell = quoted_vwap(bids, quantity)
    mid = (asks[0][0] + bids[0][0]) / 2
    constant_fee_price = (buy_cost + fee + 5) / quantity
    proportional_fee_price = (buy_cost + fee) / (quantity * (1 - 0.0002))
    breakeven = proportional_fee_price if proportional_fee_price * quantity * 0.0002 >= 5 else constant_fee_price
    return {
        "status": "VISIBLE_QUOTE_HYPOTHESIS_ONLY", "nominal_budget_cny": budget,
        "quantity": quantity, "buy_vwap_cny": buy, "buy_notional_cny": buy_cost,
        "buy_commission_cny": fee, "cash_required_cny": buy_cost + fee,
        "entry_mid_cny": mid, "buy_vwap_above_mid_bps": (buy / mid - 1) * 10000,
        "same_snapshot_bid_vwap_cny": sell,
        "same_snapshot_roundtrip_cost_bps": (buy_cost + fee - (quantity * sell - commission(quantity * sell))) / buy_cost * 10000 if sell else None,
        "next_day_minute_proxy_breakeven_cny": breakeven / (1 - 5 / 10000),
        "next_day_proxy_required_rise_from_entry_mid_bps": (breakeven / (1 - 5 / 10000) / mid - 1) * 10000,
        "actual_fill_established": False,
    }


def trailing_values(messages, ends, weights):
    times = messages._at.astype("int64").to_numpy()
    cumulative = np.concatenate(([0.0], np.cumsum(np.asarray(weights, dtype=float))))
    end = pd.DatetimeIndex(ends).asi8
    begin = end - pd.Timedelta(minutes=5).value
    return cumulative[np.searchsorted(times, end, side="right")] - cumulative[np.searchsorted(times, begin, side="right")]


def minute_measurements(quotes, orders, trades, day):
    ranges = [pd.date_range(f"{day} 09:30", f"{day} 11:29", freq="min", tz="Asia/Shanghai"),
              pd.date_range(f"{day} 13:00", f"{day} 14:56", freq="min", tz="Asia/Shanghai")]
    rows = []
    for at in ranges[0].append(ranges[1]):
        raw = quote_at(quotes, at)
        result = {"decision_at": at, "trade_date": day, "session": session(at),
                  "formal_m2_status": "NO_VIEW_MISSING_REFERENCE_CLOCK_AND_60_DAY_BASELINE"}
        if raw is None:
            result["measurement_status"] = "NO_FRESH_QUOTE"
        else:
            try:
                feature = book_features(levels(raw, "bid"), levels(raw, "ask"), 100)
                result.update(measurement_status="SOURCE_QUOTE_MEASUREMENT_ONLY", source_quote_at=raw._at,
                              quote_age_seconds=(at - raw._at).total_seconds(),
                              mid=feature["mid"], spread_bps=feature["spread_bps"],
                              spread_ticks=round((raw.ask_px1 - raw.bid_px1) / 10, 6),
                              bid_depth_cny=feature["bid_visible_depth_cny"], ask_depth_cny=feature["ask_visible_depth_cny"],
                              fixed_band_fully_visible=feature["fixed_band_fully_visible"], obi=feature["obi_visible"])
            except ValueError as error:
                result.update(measurement_status="INVALID_BOOK", reason=str(error))
        rows.append(result)
    frame = pd.DataFrame(rows)
    lookup = frame.set_index("decision_at")
    returns = []
    for row in rows:
        lag = row["decision_at"] - pd.Timedelta(minutes=5)
        current_mid = row.get("mid", float("nan"))
        old_mid = lookup.loc[lag, "mid"] if lag in lookup.index else float("nan")
        same_session = session(lag) == row["session"]
        returns.append(math.log(current_mid / old_mid) * 10000 if same_session and current_mid > 0 and old_mid > 0 else np.nan)
    frame["log_mid_return_5m_bps"] = returns
    volumes = trades.volume.to_numpy(dtype=float)
    total = trailing_values(trades, frame.decision_at, volumes)
    signed = volumes * np.where(trades.bs_flag.eq("B"), 1, np.where(trades.bs_flag.eq("S"), -1, 0))
    frame["trade_volume_5m_source_units"] = total
    frame["reported_bs_volume_imbalance_5m"] = np.divide(trailing_values(trades, frame.decision_at, signed), total, out=np.full(len(total), np.nan), where=total > 0)
    frame["reported_bs_direction_independently_validated"] = False
    for kind in ("A", "D"):
        for side in ("B", "S"):
            weights = orders.volume.where(orders.order_type.eq(kind) & orders.order_code.eq(side), 0)
            frame[f"reported_order_{kind}_{side}_volume_5m"] = trailing_values(orders, frame.decision_at, weights)
    return frame


def detect_price_episodes(grid, threshold=-5, quiet_required=10):
    """固定阈值只定义探索样本；保留缺失观察终点，不用未来反弹筛选事件。"""
    indexed = grid.set_index("decision_at")
    events, active, quiet, previous = [], False, 0, None
    for row in grid.sort_values("decision_at").itertuples(index=False):
        at = row.decision_at
        if previous is None or at.date() != previous.date() or session(at) != session(previous):
            active, quiet = False, 0
        elif at - previous != pd.Timedelta(minutes=1):
            quiet = 0
        previous = at
        value = row.log_mid_return_5m_bps
        if not math.isfinite(value):
            quiet = 0
            continue
        if value <= threshold:
            quiet = 0
            if active:
                continue
            active = True
            endpoint = at + pd.Timedelta(minutes=5)
            event = {"event_id": f"P0_{at:%Y%m%d_%H%M}", "trade_date": str(at.date()),
                     "shock_at": at, "observation_end_at": endpoint,
                     "shock_log_mid_return_5m_bps": value, "shock_mid_cny": row.mid,
                     "shock_spread_ticks": row.spread_ticks, "shock_bid_depth_cny": row.bid_depth_cny,
                     "event_kind": "EXPLORATORY_PRICE_PRESSURE_NOT_M2_SIGNAL",
                     "formal_m2_status": "NO_VIEW", "actual_order": False,
                     "observation_status": "NO_FIXED_ENDPOINT"}
            if endpoint in indexed.index and session(endpoint) == session(at):
                end = indexed.loc[endpoint]
                if pd.notna(end.get("mid")) and end.mid > 0:
                    tick_delta = float(end.spread_ticks - row.spread_ticks)
                    event.update(observation_status="OBSERVED_FIXED_5_MINUTES",
                                 observation_price_change_bps=(end.mid / row.mid - 1) * 10000,
                                 endpoint_spread_ticks=float(end.spread_ticks),
                                 spread_tick_change=tick_delta,
                                 spread_label="收窄" if tick_delta < -1e-8 else "扩大" if tick_delta > 1e-8 else "不变",
                                 bid_depth_ratio=float(end.bid_depth_cny / row.bid_depth_cny),
                                 endpoint_mid_cny=float(end.mid))
            events.append(event)
        elif active:
            quiet += 1
            if quiet >= quiet_required:
                active, quiet = False, 0
    return events


def next_day_proxies(day, minute_prices, trading_days):
    later = [value for value in trading_days if value > day]
    if not later:
        return {"status": "NEXT_TRADE_DAY_UNKNOWN", "next_trade_date": None, "prices": {}}
    next_day = later[0]
    target_times = [pd.Timestamp(next_day + " 09:35"), pd.Timestamp(next_day + " 09:36")]
    selected = minute_prices.loc[minute_prices.timestamp.isin(target_times)].sort_values("timestamp")
    prices = {row.timestamp.strftime("%H:%M") + "_close": float(row.close) for row in selected.itertuples()}
    if len(selected) == 2:
        prices.update(two_bar_low=float(selected.low.min()), two_bar_high=float(selected.high.max()))
    return {"status": "MINUTE_PRICE_PROXIES_ONLY" if len(selected) == 2 else "NEXT_DAY_MINUTE_DATA_MISSING_OR_PARTIAL",
            "next_trade_date": next_day, "prices": prices, "price_label_boundaries_verified": False}


def hypothetical_price_cost(quantity, entry_price, proxy_exit_price):
    """算术情景，不能交给实际成交结算或完整账户统计。"""
    buy_notional = quantity * entry_price
    exit_price = proxy_exit_price * (1 - 5 / 10000)
    sell_notional = quantity * exit_price
    fees = commission(buy_notional) + commission(sell_notional)
    return {"gross_price_path_bps": (proxy_exit_price / entry_price - 1) * 10000,
            "assumed_exit_proxy_friction_bps": 5,
            "hypothetical_cash_difference_cny": sell_notional - buy_notional - fees,
            "hypothetical_cash_difference_bps": (sell_notional - buy_notional - fees) / buy_notional * 10000,
            "commission_total_cny": fees, "actual_fill_established": False,
            "verified_strategy_return": False, "dividends_included": False}
