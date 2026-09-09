"""只交易510300的估值仓位、短周期波动与5分钟FVG回测引擎。"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StrategyCosts:
    """510300实际成交成本。"""

    commission_rate: float = 0.0003
    minimum_commission_cny: float = 5.0
    stamp_duty_rate: float = 0.0
    slippage_bps_per_leg: float = 5.0
    price_tick_cny: float = 0.001
    lot_size: int = 100
    cash_annual_rate: float = 0.015


def commission_for_notional(notional_cny: float, costs: StrategyCosts) -> float:
    if notional_cny <= 0:
        return 0.0
    return max(float(costs.minimum_commission_cny), notional_cny * costs.commission_rate)


def unfavorable_price(raw_price: float, side: str, costs: StrategyCosts) -> float:
    """加入滑点，并按最小价位向交易者不利方向取整。"""

    if raw_price <= 0 or costs.price_tick_cny <= 0:
        raise ValueError("价格与最小价位必须为正数")
    adjusted = raw_price * (
        1.0 + costs.slippage_bps_per_leg / 10_000.0
        if side == "BUY"
        else 1.0 - costs.slippage_bps_per_leg / 10_000.0
    )
    if side == "BUY":
        return round(math.ceil((adjusted - 1e-12) / costs.price_tick_cny) * costs.price_tick_cny, 6)
    if side == "SELL":
        return round(math.floor((adjusted + 1e-12) / costs.price_tick_cny) * costs.price_tick_cny, 6)
    raise ValueError(f"不支持的成交方向：{side}")


def roundtrip_cost_fraction(raw_price: float, shares: int, costs: StrategyCosts) -> float:
    if raw_price <= 0 or shares <= 0:
        return np.inf
    buy_price = unfavorable_price(raw_price, "BUY", costs)
    sell_price = unfavorable_price(raw_price, "SELL", costs)
    buy_notional = buy_price * shares
    sell_notional = sell_price * shares
    commissions = commission_for_notional(buy_notional, costs) + commission_for_notional(
        sell_notional, costs
    )
    execution_loss = (buy_price - sell_price) * shares
    return (commissions + execution_loss) / (raw_price * shares)


def build_valuation_signals(valuation: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """用指数隐含盈利/净资产与滚动估值倍数构造12个月估值带。

    每一行只使用该日及此前数据，信号必须到下一交易日开盘才能执行。
    这是一种可回测的指数聚合估值代理，不冒充逐家公司自由现金流DCF。
    """

    required = {"date", "index_close_pe_source", "pe_ttm", "pb"}
    if missing := required - set(valuation.columns):
        raise ValueError(f"估值数据缺少字段：{sorted(missing)}")
    data = valuation[list(required)].copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    data = data.rename(columns={"index_close_pe_source": "index_close"})
    numeric = ["index_close", "pe_ttm", "pb"]
    data[numeric] = data[numeric].apply(pd.to_numeric, errors="coerce")
    if data[numeric].isna().any().any() or (data[numeric] <= 0).any().any():
        raise ValueError("估值价格、PE或PB存在空值、零值或负值")

    trading_days = int(config["trading_days_per_year"])
    growth_window = int(config["growth_median_window_days"])
    multiple_window = int(config["multiple_window_days"])
    minimum_growth = int(config.get("minimum_growth_observations", growth_window // 2))
    minimum_multiple = int(config.get("minimum_multiple_observations", multiple_window // 2))
    earnings_weight = float(config["earnings_anchor_weight"])
    book_weight = float(config["book_anchor_weight"])
    if not np.isclose(earnings_weight + book_weight, 1.0):
        raise ValueError("盈利锚与净资产锚权重之和必须等于1")

    data["implied_eps"] = data["index_close"] / data["pe_ttm"]
    data["implied_bps"] = data["index_close"] / data["pb"]
    eps_clip = tuple(float(value) for value in config["eps_growth_clip"])
    bps_clip = tuple(float(value) for value in config["bps_growth_clip"])
    eps_growth_raw = data["implied_eps"].pct_change(trading_days, fill_method=None).clip(*eps_clip)
    bps_growth_raw = data["implied_bps"].pct_change(trading_days, fill_method=None).clip(*bps_clip)
    data["eps_growth_12m"] = eps_growth_raw.rolling(
        growth_window, min_periods=minimum_growth
    ).median()
    data["bps_growth_12m"] = bps_growth_raw.rolling(
        growth_window, min_periods=minimum_growth
    ).median()
    data["projected_eps_12m"] = data["implied_eps"] * (1.0 + data["eps_growth_12m"])
    data["projected_bps_12m"] = data["implied_bps"] * (1.0 + data["bps_growth_12m"])

    quantiles = {
        "low": float(config["low_quantile"]),
        "mid": float(config["mid_quantile"]),
        "high": float(config["high_quantile"]),
    }
    for label, quantile in quantiles.items():
        data[f"pe_{label}"] = data["pe_ttm"].rolling(
            multiple_window, min_periods=minimum_multiple
        ).quantile(quantile)
        data[f"pb_{label}"] = data["pb"].rolling(
            multiple_window, min_periods=minimum_multiple
        ).quantile(quantile)
        data[f"fair_{label}_12m"] = (
            earnings_weight * data["projected_eps_12m"] * data[f"pe_{label}"]
            + book_weight * data["projected_bps_12m"] * data[f"pb_{label}"]
        )

    ready = data[["fair_low_12m", "fair_mid_12m", "fair_high_12m"]].notna().all(axis=1)
    data["target_position"] = np.nan
    data.loc[ready, "target_position"] = 0.5
    data.loc[ready & data["index_close"].le(data["fair_low_12m"]), "target_position"] = 1.0
    data.loc[ready & data["index_close"].ge(data["fair_high_12m"]), "target_position"] = 0.0
    data["valuation_state"] = data["target_position"].map(
        {0.0: "空仓", 0.5: "半仓", 1.0: "满仓"}
    ).fillna("数据不足")
    return data


def attach_market_cap_context(
    valuation_signals: pd.DataFrame,
    constituent_panel: pd.DataFrame,
) -> pd.DataFrame:
    """加入信号日300只点时成员的总市值、隐含盈利与估值市值带。"""

    required = {"date", "is_index_member", "total_market_cap_cny", "con_code"}
    if missing := required - set(constituent_panel.columns):
        raise ValueError(f"成分股市值数据缺少字段：{sorted(missing)}")
    panel = constituent_panel.loc[constituent_panel["is_index_member"].astype(bool)].copy()
    panel["date"] = pd.to_datetime(panel["date"])
    aggregate = (
        panel.groupby("date", sort=True)
        .agg(
            member_count=("con_code", "nunique"),
            aggregate_market_cap_cny=("total_market_cap_cny", "sum"),
        )
        .reset_index()
    )
    if not aggregate["member_count"].eq(300).all():
        raise ValueError("市值面板并非每个交易日恰好300只点时指数成员")
    result = valuation_signals.merge(aggregate, on="date", how="left", validate="one_to_one")
    valid = result["aggregate_market_cap_cny"].notna()
    result.loc[valid, "aggregate_earnings_proxy_cny"] = (
        result.loc[valid, "aggregate_market_cap_cny"] / result.loc[valid, "pe_ttm"]
    )
    result.loc[valid, "aggregate_book_value_proxy_cny"] = (
        result.loc[valid, "aggregate_market_cap_cny"] / result.loc[valid, "pb"]
    )
    for label in ("low", "mid", "high"):
        result.loc[valid, f"fair_market_cap_{label}_12m_cny"] = (
            result.loc[valid, "aggregate_market_cap_cny"]
            * result.loc[valid, f"fair_{label}_12m"]
            / result.loc[valid, "index_close"]
        )
    cap_ready = valid & result[
        ["fair_market_cap_low_12m_cny", "fair_market_cap_high_12m_cny"]
    ].notna().all(axis=1)
    band_width = (
        result["fair_market_cap_high_12m_cny"]
        - result["fair_market_cap_low_12m_cny"]
    ).replace(0.0, np.nan)
    result["raw_continuous_position"] = (
        (result["fair_market_cap_high_12m_cny"] - result["aggregate_market_cap_cny"])
        / band_width
    ).clip(0.0, 1.0)
    result.loc[cap_ready, "target_position"] = result.loc[
        cap_ready, "raw_continuous_position"
    ]
    result.loc[cap_ready, "valuation_state"] = result.loc[
        cap_ready, "target_position"
    ].map(lambda value: f"连续目标仓位{value:.1%}")
    result.loc[
        cap_ready & result["target_position"].le(0.0), "valuation_state"
    ] = "空仓"
    result.loc[
        cap_ready & result["target_position"].ge(1.0), "valuation_state"
    ] = "满仓"
    return result


def schedule_continuous_positions(
    features: pd.DataFrame,
    rebalance_every_trading_days: int,
) -> pd.DataFrame:
    """慢估值层按固定交易日节奏更新连续仓位，期间保持上次目标。"""

    if rebalance_every_trading_days <= 0:
        raise ValueError("连续仓位再平衡间隔必须为正整数")
    if "raw_continuous_position" not in features.columns:
        raise ValueError("缺少raw_continuous_position，无法生成连续仓位")
    result = features.sort_values("date").reset_index(drop=True).copy()
    scheduled = pd.Series(np.nan, index=result.index, dtype=float)
    ready_indices = result.index[result["raw_continuous_position"].notna()].to_numpy()
    if len(ready_indices):
        rebalance_indices = ready_indices[::rebalance_every_trading_days]
        scheduled.loc[rebalance_indices] = result.loc[
            rebalance_indices, "raw_continuous_position"
        ]
    result["target_position"] = scheduled.ffill()
    result["is_valuation_rebalance_day"] = scheduled.notna()
    result["valuation_state"] = result["target_position"].map(
        lambda value: f"连续目标仓位{value:.1%}" if pd.notna(value) else "数据不足"
    )
    result.loc[result["target_position"].le(0.0), "valuation_state"] = "空仓"
    result.loc[result["target_position"].ge(1.0), "valuation_state"] = "满仓"
    return result


def build_short_wave(daily: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """用短周期涨跌幅相对近期实现波动率的偏离判断做T方向。"""

    required = {"date", "close"}
    if missing := required - set(daily.columns):
        raise ValueError(f"日线数据缺少字段：{sorted(missing)}")
    data = daily[["date", "close"]].copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    data["log_return_1d"] = np.log(data["close"] / data["close"].shift(1))
    wave_days = int(config["wave_days"])
    long_wave_days = int(config.get("long_wave_days", 20))
    volatility_days = int(config["volatility_days"])
    threshold = float(config["z_threshold"])
    data["wave_log_return"] = np.log(data["close"] / data["close"].shift(wave_days))
    data["long_wave_log_return"] = np.log(
        data["close"] / data["close"].shift(long_wave_days)
    )
    data["wave_residual"] = data["wave_log_return"] - (
        wave_days / long_wave_days
    ) * data["long_wave_log_return"]
    data["realized_volatility"] = data["log_return_1d"].rolling(
        volatility_days, min_periods=volatility_days
    ).std(ddof=1)
    denominator = data["realized_volatility"] * np.sqrt(wave_days)
    data["wave_z"] = data["wave_residual"] / denominator.replace(0.0, np.nan)
    data["t_direction"] = "NONE"
    data.loc[data["wave_z"].ge(threshold), "t_direction"] = "SELL_FIRST"
    data.loc[data["wave_z"].le(-threshold), "t_direction"] = "BUY_FIRST"
    return data


def build_tactical_positions(
    signals: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """把短周期波动转为连续战术仓位偏移，不要求当日恢复基础仓位。"""

    required = {"date", "target_position", "wave_z"}
    if missing := required - set(signals.columns):
        raise ValueError(f"战术仓位信号缺少字段：{sorted(missing)}")
    result = signals.copy()
    result["base_position"] = pd.to_numeric(result["target_position"], errors="coerce")
    deadband = float(config["deadband_z"])
    full_strength = float(config["full_strength_z"])
    amplitude = float(config["maximum_tactical_position"])
    if not 0.0 <= amplitude <= 1.0 or full_strength <= deadband:
        raise ValueError("战术仓位幅度或z值区间无效")
    absolute_z = result["wave_z"].abs()
    strength = ((absolute_z - deadband) / (full_strength - deadband)).clip(0.0, 1.0)
    strength = strength.where(result["wave_z"].notna(), 0.0)
    signed_strength = np.sign(result["wave_z"].fillna(0.0)) * strength
    result["tactical_strength"] = strength
    result["tactical_adjustment"] = -amplitude * signed_strength
    result["desired_position"] = (
        result["base_position"] + result["tactical_adjustment"]
    ).clip(0.0, 1.0)
    result["inventory_intent"] = "HOLD"
    result.loc[result["tactical_adjustment"].gt(0.0), "inventory_intent"] = "BUY"
    result.loc[result["tactical_adjustment"].lt(0.0), "inventory_intent"] = "SELL"
    return result


def aggregate_minute_to_5m(minute: pd.DataFrame, records_per_bar: int = 5) -> pd.DataFrame:
    """按上午、下午分别聚合连续5条一分钟记录，绝不跨午休。"""

    required = {"trade_time", "open", "high", "low", "close", "vol", "amount"}
    if missing := required - set(minute.columns):
        raise ValueError(f"分钟数据缺少字段：{sorted(missing)}")
    data = minute[list(required)].copy()
    data["trade_time"] = pd.to_datetime(data["trade_time"])
    data = data.sort_values("trade_time").reset_index(drop=True)
    data["date"] = data["trade_time"].dt.normalize()
    minute_of_day = data["trade_time"].dt.hour * 60 + data["trade_time"].dt.minute
    data["session"] = np.where(minute_of_day <= 11 * 60 + 30, "AM", "PM")
    grouped = data.groupby(["date", "session"], sort=False)
    data["session_record"] = grouped.cumcount()
    data["bar_number"] = data["session_record"] // records_per_bar
    data["records_in_bar"] = grouped["session_record"].transform(
        lambda values: values.groupby(values // records_per_bar).transform("size")
    )
    data = data.loc[data["records_in_bar"].eq(records_per_bar)].copy()
    bars = (
        data.groupby(["date", "session", "bar_number"], sort=False)
        .agg(
            trade_time=("trade_time", "max"),
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            vol=("vol", "sum"),
            amount=("amount", "sum"),
            source_records=("trade_time", "size"),
        )
        .reset_index()
    )
    bars["session_bar"] = bars.groupby(["date", "session"], sort=False).cumcount()
    return bars


def detect_fvg_zones(bars: pd.DataFrame, minimum_gap_bps: float) -> pd.DataFrame:
    """识别标准三根K线FVG，且三根K线必须位于同一交易时段。"""

    required = {"date", "session", "session_bar", "trade_time", "high", "low"}
    if missing := required - set(bars.columns):
        raise ValueError(f"5分钟数据缺少字段：{sorted(missing)}")
    rows: list[dict[str, Any]] = []
    for (date, session), frame in bars.groupby(["date", "session"], sort=False):
        frame = frame.sort_values("session_bar").reset_index(drop=True)
        for index in range(2, len(frame)):
            first = frame.iloc[index - 2]
            third = frame.iloc[index]
            bullish_gap = float(third["low"]) - float(first["high"])
            bearish_gap = float(first["low"]) - float(third["high"])
            reference = float(third["close"])
            if bullish_gap > 0 and bullish_gap / reference * 10_000.0 >= minimum_gap_bps:
                lower, upper, kind = float(first["high"]), float(third["low"]), "BULLISH"
            elif bearish_gap > 0 and bearish_gap / reference * 10_000.0 >= minimum_gap_bps:
                lower, upper, kind = float(third["high"]), float(first["low"]), "BEARISH"
            else:
                continue
            rows.append(
                {
                    "date": pd.Timestamp(date),
                    "session": session,
                    "formation_bar": int(index),
                    "formation_time": pd.Timestamp(third["trade_time"]),
                    "fvg_kind": kind,
                    "zone_lower": lower,
                    "zone_upper": upper,
                    "zone_mid": (lower + upper) / 2.0,
                    "gap_bps": (upper - lower) / reference * 10_000.0,
                }
            )
    return pd.DataFrame(rows)


def _maximum_affordable_buy(
    cash: float, raw_price: float, maximum_shares: int, costs: StrategyCosts
) -> int:
    upper = maximum_shares // costs.lot_size * costs.lot_size
    execution = unfavorable_price(raw_price, "BUY", costs)
    for shares in range(upper, 0, -costs.lot_size):
        notional = shares * execution
        if notional + commission_for_notional(notional, costs) <= cash + 1e-9:
            return shares
    return 0


def _maximum_sell_first(
    cash: float,
    raw_entry: float,
    stop_price: float,
    maximum_shares: int,
    costs: StrategyCosts,
) -> int:
    """按止损价加不利滑点预留回补资金。"""

    upper = maximum_shares // costs.lot_size * costs.lot_size
    sell_execution = unfavorable_price(raw_entry, "SELL", costs)
    reserve_buy_execution = unfavorable_price(stop_price, "BUY", costs)
    for shares in range(upper, 0, -costs.lot_size):
        sell_notional = shares * sell_execution
        available = cash + sell_notional - commission_for_notional(sell_notional, costs)
        buy_notional = shares * reserve_buy_execution
        required = buy_notional + commission_for_notional(buy_notional, costs)
        if required <= available + 1e-9:
            return shares
    return 0


def audit_fvg_funnel(
    ledger: pd.DataFrame,
    bars_5m: pd.DataFrame,
    zones: pd.DataFrame,
    costs: StrategyCosts,
    config: dict[str, Any],
) -> dict[str, Any]:
    """逐层审计FVG候选、方向匹配、回补触发与经济成交门槛。

    此函数只做口径统计，不将“通过全部门槛的往返”冒充为“合格FVG数量”。
    """

    required_ledger = {
        "date",
        "target_position",
        "t_direction",
        "shares_at_start",
        "shares",
        "cash",
    }
    if missing := required_ledger - set(ledger.columns):
        raise ValueError(f"FVG漏斗所需账户字段缺失：{sorted(missing)}")
    account = ledger.copy()
    account["date"] = pd.to_datetime(account["date"])
    bars = bars_5m.copy()
    bars["date"] = pd.to_datetime(bars["date"])
    events = zones.copy()
    if events.empty:
        return {
            "raw_fvg_events": 0,
            "raw_fvg_days": 0,
            "half_position_days": int(account["target_position"].eq(0.5).sum()),
            "half_with_wave_direction_days": 0,
            "t_plus_one_eligible_days": 0,
            "direction_matched_events": 0,
            "direction_matched_days": 0,
            "midpoint_retracement_events": 0,
            "midpoint_retracement_days": 0,
            "next_open_inside_events": 0,
            "next_open_inside_days": 0,
            "economic_gate_events": 0,
            "economic_gate_days": 0,
        }
    events["date"] = pd.to_datetime(events["date"])
    valid_dates = set(account["date"])
    events = events.loc[events["date"].isin(valid_dates)].copy()
    bars_by_date = {date: frame for date, frame in bars.groupby("date", sort=False)}
    zones_by_date = {date: frame for date, frame in events.groupby("date", sort=False)}
    maximum_age = int(config["maximum_zone_age_bars"])
    edge_multiple = float(config["minimum_target_room_cost_multiple"])

    counters = {
        "raw_fvg_events": int(len(events)),
        "raw_fvg_days": int(events["date"].nunique()),
        "half_position_days": 0,
        "half_with_wave_direction_days": 0,
        "t_plus_one_eligible_days": 0,
        "direction_matched_events": 0,
        "midpoint_retracement_events": 0,
        "next_open_inside_events": 0,
        "economic_gate_events": 0,
    }
    day_sets: dict[str, set[pd.Timestamp]] = {
        "direction_matched": set(),
        "midpoint_retracement": set(),
        "next_open_inside": set(),
        "economic_gate": set(),
    }
    for row in account.itertuples(index=False):
        date = pd.Timestamp(row.date)
        if not np.isclose(float(row.target_position), 0.5):
            continue
        counters["half_position_days"] += 1
        direction = str(row.t_direction)
        if direction not in {"BUY_FIRST", "SELL_FIRST"}:
            continue
        counters["half_with_wave_direction_days"] += 1
        core_shares = int(row.shares)
        if core_shares < costs.lot_size or core_shares > int(row.shares_at_start):
            continue
        counters["t_plus_one_eligible_days"] += 1
        day_zones = zones_by_date.get(date)
        day_bars = bars_by_date.get(date)
        if day_zones is None or day_bars is None:
            continue
        wanted_kind = "BULLISH" if direction == "BUY_FIRST" else "BEARISH"
        matched = day_zones.loc[day_zones["fvg_kind"].eq(wanted_kind)]
        counters["direction_matched_events"] += int(len(matched))
        if not matched.empty:
            day_sets["direction_matched"].add(date)
        for zone in matched.itertuples(index=False):
            session = day_bars.loc[day_bars["session"].eq(zone.session)].sort_values(
                "session_bar"
            ).reset_index(drop=True)
            last_touch = min(int(zone.formation_bar) + maximum_age, len(session) - 2)
            for touch_index in range(int(zone.formation_bar) + 1, last_touch + 1):
                touch = session.iloc[touch_index]
                if not (float(touch["low"]) <= zone.zone_mid <= float(touch["high"])):
                    continue
                counters["midpoint_retracement_events"] += 1
                day_sets["midpoint_retracement"].add(date)
                raw_entry = float(session.iloc[touch_index + 1]["open"])
                if direction == "BUY_FIRST":
                    target_price, stop_price = float(zone.zone_upper), float(zone.zone_lower)
                    inside = stop_price < raw_entry < target_price
                    target_room = target_price / raw_entry - 1.0 if inside else 0.0
                else:
                    target_price, stop_price = float(zone.zone_lower), float(zone.zone_upper)
                    inside = target_price < raw_entry < stop_price
                    target_room = 1.0 - target_price / raw_entry if inside else 0.0
                if not inside:
                    break
                counters["next_open_inside_events"] += 1
                day_sets["next_open_inside"].add(date)
                if direction == "BUY_FIRST":
                    trade_shares = _maximum_affordable_buy(
                        float(row.cash), raw_entry, core_shares, costs
                    )
                else:
                    trade_shares = _maximum_sell_first(
                        float(row.cash), raw_entry, stop_price, core_shares, costs
                    )
                if trade_shares < costs.lot_size:
                    break
                estimated_cost = roundtrip_cost_fraction(raw_entry, trade_shares, costs)
                if target_room + 1e-12 < edge_multiple * estimated_cost:
                    break
                counters["economic_gate_events"] += 1
                day_sets["economic_gate"].add(date)
                break

    counters.update(
        {
            "direction_matched_days": len(day_sets["direction_matched"]),
            "midpoint_retracement_days": len(day_sets["midpoint_retracement"]),
            "next_open_inside_days": len(day_sets["next_open_inside"]),
            "economic_gate_days": len(day_sets["economic_gate"]),
        }
    )
    return counters


def find_fvg_trade(
    day_bars: pd.DataFrame,
    day_zones: pd.DataFrame,
    direction: str,
    cash: float,
    old_sellable_shares: int,
    costs: StrategyCosts,
    config: dict[str, Any],
) -> dict[str, Any] | None:
    """返回当日第一笔同时满足方向、T+1和成本门槛的FVG交易。"""

    if direction not in {"BUY_FIRST", "SELL_FIRST"} or old_sellable_shares < costs.lot_size:
        return None
    if day_zones.empty:
        return None
    wanted_kind = "BULLISH" if direction == "BUY_FIRST" else "BEARISH"
    maximum_age = int(config["maximum_zone_age_bars"])
    maximum_hold = int(config["maximum_hold_bars"])
    edge_multiple = float(config["minimum_target_room_cost_multiple"])

    for zone in day_zones.loc[day_zones["fvg_kind"].eq(wanted_kind)].itertuples(index=False):
        session = day_bars.loc[day_bars["session"].eq(zone.session)].sort_values("session_bar")
        session = session.reset_index(drop=True)
        last_touch_index = min(int(zone.formation_bar) + maximum_age, len(session) - 2)
        for touch_index in range(int(zone.formation_bar) + 1, last_touch_index + 1):
            touch = session.iloc[touch_index]
            if not (float(touch["low"]) <= zone.zone_mid <= float(touch["high"])):
                continue
            entry_index = touch_index + 1
            entry_bar = session.iloc[entry_index]
            raw_entry = float(entry_bar["open"])
            if direction == "BUY_FIRST":
                target_price, stop_price = float(zone.zone_upper), float(zone.zone_lower)
                if not stop_price < raw_entry < target_price:
                    continue
                shares = _maximum_affordable_buy(cash, raw_entry, old_sellable_shares, costs)
                target_room = target_price / raw_entry - 1.0
            else:
                target_price, stop_price = float(zone.zone_lower), float(zone.zone_upper)
                if not target_price < raw_entry < stop_price:
                    continue
                shares = _maximum_sell_first(
                    cash, raw_entry, stop_price, old_sellable_shares, costs
                )
                target_room = 1.0 - target_price / raw_entry
            if shares < costs.lot_size:
                continue
            estimated_cost = roundtrip_cost_fraction(raw_entry, shares, costs)
            if target_room + 1e-12 < edge_multiple * estimated_cost:
                continue

            final_index = min(entry_index + maximum_hold - 1, len(session) - 1)
            exit_reason = "TIME_EXIT"
            raw_exit = float(session.iloc[final_index]["close"])
            exit_time = pd.Timestamp(session.iloc[final_index]["trade_time"])
            for exit_index in range(entry_index, final_index + 1):
                bar = session.iloc[exit_index]
                if direction == "BUY_FIRST":
                    stop_hit = float(bar["low"]) <= stop_price
                    target_hit = float(bar["high"]) >= target_price
                else:
                    stop_hit = float(bar["high"]) >= stop_price
                    target_hit = float(bar["low"]) <= target_price
                if stop_hit:
                    raw_exit, exit_reason = stop_price, "STOP"
                    exit_time = pd.Timestamp(bar["trade_time"])
                    break
                if target_hit:
                    raw_exit, exit_reason = target_price, "TARGET"
                    exit_time = pd.Timestamp(bar["trade_time"])
                    break

            entry_side = "BUY" if direction == "BUY_FIRST" else "SELL"
            exit_side = "SELL" if direction == "BUY_FIRST" else "BUY"
            entry_execution = unfavorable_price(raw_entry, entry_side, costs)
            exit_execution = unfavorable_price(raw_exit, exit_side, costs)
            entry_notional = shares * entry_execution
            exit_notional = shares * exit_execution
            entry_commission = commission_for_notional(entry_notional, costs)
            exit_commission = commission_for_notional(exit_notional, costs)
            pnl = (
                exit_notional - entry_notional
                if direction == "BUY_FIRST"
                else entry_notional - exit_notional
            ) - entry_commission - exit_commission
            return {
                "direction": direction,
                "sequence": (
                    "BUY_NEW_THEN_SELL_OLD"
                    if direction == "BUY_FIRST"
                    else "SELL_OLD_THEN_BUY_NEW"
                ),
                "shares": int(shares),
                "formation_time": pd.Timestamp(zone.formation_time),
                "touch_time": pd.Timestamp(touch["trade_time"]),
                "entry_time": pd.Timestamp(entry_bar["trade_time"]),
                "exit_time": exit_time,
                "fvg_kind": zone.fvg_kind,
                "gap_bps": float(zone.gap_bps),
                "zone_lower": float(zone.zone_lower),
                "zone_upper": float(zone.zone_upper),
                "raw_entry_price": raw_entry,
                "raw_exit_price": raw_exit,
                "entry_execution_price": entry_execution,
                "exit_execution_price": exit_execution,
                "target_price": target_price,
                "stop_price": stop_price,
                "target_room_fraction": target_room,
                "estimated_roundtrip_cost_fraction": estimated_cost,
                "entry_commission": entry_commission,
                "exit_commission": exit_commission,
                "net_pnl_cny": pnl,
                "exit_reason": exit_reason,
            }
    return None


def find_inventory_fvg_fill(
    day_bars: pd.DataFrame,
    day_zones: pd.DataFrame,
    side: str,
    costs: StrategyCosts,
    maximum_zone_age_bars: int,
    after_time: pd.Timestamp | None = None,
) -> dict[str, Any] | None:
    """寻找最早的方向性FVG单边成交，不附加强制平仓腿。"""

    if side not in {"BUY", "SELL"} or day_bars.empty or day_zones.empty:
        return None
    wanted_kind = "BULLISH" if side == "BUY" else "BEARISH"
    candidates: list[dict[str, Any]] = []
    for zone in day_zones.loc[day_zones["fvg_kind"].eq(wanted_kind)].itertuples(index=False):
        if after_time is not None and pd.Timestamp(zone.formation_time) <= after_time:
            continue
        session = day_bars.loc[day_bars["session"].eq(zone.session)].sort_values(
            "session_bar"
        ).reset_index(drop=True)
        last_touch = min(int(zone.formation_bar) + maximum_zone_age_bars, len(session) - 1)
        for touch_index in range(int(zone.formation_bar) + 1, last_touch + 1):
            touch = session.iloc[touch_index]
            midpoint = float(zone.zone_mid)
            if not (float(touch["low"]) <= midpoint <= float(touch["high"])):
                continue
            execution = unfavorable_price(midpoint, side, costs)
            if not (float(touch["low"]) <= execution <= float(touch["high"])):
                break
            candidates.append(
                {
                    "side": side,
                    "fvg_kind": wanted_kind,
                    "formation_time": pd.Timestamp(zone.formation_time),
                    "fill_time": pd.Timestamp(touch["trade_time"]),
                    "zone_lower": float(zone.zone_lower),
                    "zone_upper": float(zone.zone_upper),
                    "zone_mid": midpoint,
                    "gap_bps": float(zone.gap_bps),
                    "raw_fill_price": midpoint,
                    "execution_price": execution,
                }
            )
            break
    if not candidates:
        return None
    return min(candidates, key=lambda item: (item["fill_time"], item["formation_time"]))


def run_inventory_backtest(
    daily: pd.DataFrame,
    dividends: pd.DataFrame,
    signals: pd.DataFrame,
    bars_5m: pd.DataFrame,
    zones: pd.DataFrame,
    initial_cash: float,
    costs: StrategyCosts,
    execution_config: dict[str, Any],
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按连续目标库存用FVG完成单边调仓，持仓偏离可以跨日延续。

    每日最多执行一条买入或卖出腿，不为“做完一次T”而强制反向成交。
    新买份额最早下一交易日才能卖出；由于单日只有一条腿，天然满足T+1。
    """

    market, events, targets = _validate_backtest_inputs(daily, dividends, signals)
    if "desired_position" not in targets.columns:
        raise ValueError("库存回测需要desired_position字段")
    start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
    market = market.loc[market["date"].between(start, end)].reset_index(drop=True)
    targets = targets.loc[targets["date"].between(start, end)].copy()
    if len(market) < 2:
        raise ValueError("回测区间至少需要两个交易日")
    desired_map = targets.set_index("date")["desired_position"].to_dict()
    base_map = targets.set_index("date")["base_position"].to_dict()
    wave_map = targets.set_index("date")["wave_z"].to_dict()
    adjustment_map = targets.set_index("date")["tactical_adjustment"].to_dict()
    events_by_ex = {
        date: frame.to_dict("records") for date, frame in events.groupby("ex_date")
    }
    bars_by_date = {date: frame for date, frame in bars_5m.groupby("date", sort=False)}
    zones_by_date = (
        {date: frame for date, frame in zones.groupby("date", sort=False)}
        if not zones.empty
        else {}
    )
    maximum_age = int(execution_config["maximum_zone_age_bars"])
    minimum_notional = float(execution_config["minimum_trade_notional_cny"])
    maximum_commission_fraction = float(
        execution_config["maximum_one_way_commission_fraction"]
    )
    roundtrip_config = execution_config.get("optional_intraday_roundtrip", {})
    allow_optional_roundtrip = bool(roundtrip_config.get("enabled", False))
    minimum_locked_net_profit = float(
        roundtrip_config.get("minimum_locked_net_profit_cny", np.inf)
    )

    cash = float(initial_cash)
    shares = 0
    receivable = 0.0
    scheduled_payments: dict[pd.Timestamp, float] = {}
    previous_date: pd.Timestamp | None = None
    desired_position = 0.0
    base_position = 0.0
    wave_z = np.nan
    tactical_adjustment = 0.0
    ledger_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []

    for row in market.itertuples(index=False):
        date = pd.Timestamp(row.date)
        shares_at_start = shares
        if previous_date is not None and costs.cash_annual_rate:
            cash *= 1.0 + costs.cash_annual_rate / 242.0
        entitlement_today = 0.0
        for event in events_by_ex.get(date, []):
            entitlement = shares_at_start * float(event["cash_dividend_per_share"])
            entitlement_today += entitlement
            payment_date = pd.Timestamp(event["payment_date"])
            scheduled_payments[payment_date] = scheduled_payments.get(payment_date, 0.0) + entitlement
        receivable += entitlement_today

        signal_date = previous_date
        if signal_date in desired_map and pd.notna(desired_map[signal_date]):
            desired_position = float(desired_map[signal_date])
            base_position = float(base_map[signal_date])
            wave_z = float(wave_map.get(signal_date, np.nan))
            tactical_adjustment = float(adjustment_map.get(signal_date, 0.0))

        open_equity = cash + shares * float(row.open) + receivable
        reference_price = float(row.open)
        desired_shares_at_open = int(
            math.floor(desired_position * open_equity / reference_price / costs.lot_size)
        ) * costs.lot_size
        share_gap_at_open = desired_shares_at_open - shares
        requested_side = "HOLD"
        if share_gap_at_open >= costs.lot_size:
            requested_side = "BUY"
        elif share_gap_at_open <= -costs.lot_size:
            requested_side = "SELL"

        skip_reason = "目标份额处于一手以内"
        fill = None
        traded_quantity = 0
        trade_legs_today = 0
        locked_roundtrip_profit = 0.0
        first_trade_record: dict[str, Any] | None = None
        if requested_side in {"BUY", "SELL"}:
            day_bars = bars_by_date.get(date, pd.DataFrame())
            day_zones = zones_by_date.get(date, pd.DataFrame())
            fill = find_inventory_fvg_fill(
                day_bars, day_zones, requested_side, costs, maximum_age
            )
            if fill is None:
                skip_reason = "当日没有匹配并可成交的FVG中点回补"
            else:
                execution = float(fill["execution_price"])
                fill_equity = cash + shares * execution + receivable
                desired_shares = int(
                    math.floor(desired_position * fill_equity / execution / costs.lot_size)
                ) * costs.lot_size
                quantity = desired_shares - shares
                if requested_side == "BUY":
                    quantity = max(quantity, 0) // costs.lot_size * costs.lot_size
                else:
                    quantity = min(max(-quantity, 0), shares_at_start)
                    quantity = quantity // costs.lot_size * costs.lot_size
                notional = quantity * execution
                fee = commission_for_notional(notional, costs)
                if quantity < costs.lot_size:
                    skip_reason = "FVG成交时目标差额已小于一手"
                elif notional < minimum_notional:
                    skip_reason = "目标差额成交额低于最低经济成交金额"
                elif fee / notional > maximum_commission_fraction:
                    skip_reason = "单边佣金占成交额比例过高"
                elif requested_side == "BUY":
                    while quantity >= costs.lot_size:
                        notional = quantity * execution
                        fee = commission_for_notional(notional, costs)
                        if notional + fee <= cash + 1e-9:
                            break
                        quantity -= costs.lot_size
                    if quantity < costs.lot_size or quantity * execution < minimum_notional:
                        skip_reason = "现金不足以完成经济最小买入"
                    else:
                        notional = quantity * execution
                        fee = commission_for_notional(notional, costs)
                        cash -= notional + fee
                        shares += quantity
                        traded_quantity = quantity
                        skip_reason = "已成交"
                else:
                    cash += notional - fee
                    shares -= quantity
                    traded_quantity = quantity
                    skip_reason = "已成交"
                if traded_quantity:
                    first_trade_record = {
                            "date": date,
                            "signal_date": signal_date,
                            "side": requested_side,
                            "quantity": int(traded_quantity),
                            "formation_time": fill["formation_time"],
                            "fill_time": fill["fill_time"],
                            "fvg_kind": fill["fvg_kind"],
                            "zone_lower": fill["zone_lower"],
                            "zone_upper": fill["zone_upper"],
                            "zone_mid": fill["zone_mid"],
                            "gap_bps": fill["gap_bps"],
                            "execution_price": execution,
                            "notional_cny": float(notional),
                            "commission": float(fee),
                            "shares_before": int(shares - traded_quantity if requested_side == "BUY" else shares + traded_quantity),
                            "shares_after": int(shares),
                            "base_position": base_position,
                            "tactical_adjustment": tactical_adjustment,
                            "desired_position": desired_position,
                            "wave_z": wave_z,
                            "held_overnight_allowed": True,
                            "trade_role": "INVENTORY_ADJUSTMENT",
                            "paired_roundtrip_id": None,
                            "locked_pair_net_pnl_cny": np.nan,
                        }
                    trade_rows.append(first_trade_record)
                    trade_legs_today = 1

        if (
            allow_optional_roundtrip
            and first_trade_record is not None
            and fill is not None
            and (requested_side == "SELL" or traded_quantity <= shares_at_start)
        ):
            close_side = "BUY" if requested_side == "SELL" else "SELL"
            close_fill = find_inventory_fvg_fill(
                day_bars,
                day_zones,
                close_side,
                costs,
                maximum_age,
                after_time=pd.Timestamp(fill["fill_time"]),
            )
            if close_fill is not None:
                close_execution = float(close_fill["execution_price"])
                close_notional = traded_quantity * close_execution
                close_fee = commission_for_notional(close_notional, costs)
                if requested_side == "BUY":
                    gross_spread = (
                        close_execution - float(first_trade_record["execution_price"])
                    ) * traded_quantity
                else:
                    gross_spread = (
                        float(first_trade_record["execution_price"]) - close_execution
                    ) * traded_quantity
                locked_net = (
                    gross_spread
                    - float(first_trade_record["commission"])
                    - close_fee
                )
                close_is_economic = (
                    close_notional >= minimum_notional
                    and close_fee / close_notional <= maximum_commission_fraction
                    and locked_net >= minimum_locked_net_profit
                )
                close_is_affordable = (
                    close_side == "SELL"
                    or close_notional + close_fee <= cash + 1e-9
                )
                if close_is_economic and close_is_affordable:
                    shares_before_close = shares
                    if close_side == "SELL":
                        cash += close_notional - close_fee
                        shares -= traded_quantity
                    else:
                        cash -= close_notional + close_fee
                        shares += traded_quantity
                    pair_id = f"{date.date()}-{pd.Timestamp(fill['fill_time']).strftime('%H%M')}"
                    first_trade_record["paired_roundtrip_id"] = pair_id
                    first_trade_record["locked_pair_net_pnl_cny"] = locked_net
                    trade_rows.append(
                        {
                            "date": date,
                            "signal_date": signal_date,
                            "side": close_side,
                            "quantity": int(traded_quantity),
                            "formation_time": close_fill["formation_time"],
                            "fill_time": close_fill["fill_time"],
                            "fvg_kind": close_fill["fvg_kind"],
                            "zone_lower": close_fill["zone_lower"],
                            "zone_upper": close_fill["zone_upper"],
                            "zone_mid": close_fill["zone_mid"],
                            "gap_bps": close_fill["gap_bps"],
                            "execution_price": close_execution,
                            "notional_cny": float(close_notional),
                            "commission": float(close_fee),
                            "shares_before": int(shares_before_close),
                            "shares_after": int(shares),
                            "base_position": base_position,
                            "tactical_adjustment": tactical_adjustment,
                            "desired_position": desired_position,
                            "wave_z": wave_z,
                            "held_overnight_allowed": True,
                            "trade_role": "OPTIONAL_INTRADAY_T_CLOSE",
                            "paired_roundtrip_id": pair_id,
                            "locked_pair_net_pnl_cny": locked_net,
                        }
                    )
                    trade_legs_today = 2
                    locked_roundtrip_profit = float(locked_net)

        payment_today = float(scheduled_payments.pop(date, 0.0))
        if payment_today:
            cash += payment_today
            receivable -= payment_today
            if abs(receivable) < 1e-9:
                receivable = 0.0
        equity = cash + shares * float(row.close) + receivable
        ledger_rows.append(
            {
                "date": date,
                "signal_date_used": signal_date,
                "base_position": base_position,
                "tactical_adjustment": tactical_adjustment,
                "desired_position": desired_position,
                "wave_z": wave_z,
                "requested_side": requested_side,
                "fvg_fill_found": fill is not None,
                "execution_status": skip_reason,
                "traded_quantity": int(traded_quantity),
                "trade_legs_today": int(trade_legs_today),
                "locked_roundtrip_profit_cny": locked_roundtrip_profit,
                "shares_at_start": int(shares_at_start),
                "shares": int(shares),
                "cash": float(cash),
                "dividend_receivable": float(receivable),
                "dividend_entitlement_today": float(entitlement_today),
                "dividend_payment_today": payment_today,
                "close": float(row.close),
                "equity": float(equity),
                "actual_position": float(shares * float(row.close) / equity) if equity else 0.0,
                "position_gap": float(desired_position - shares * float(row.close) / equity) if equity else desired_position,
            }
        )
        previous_date = date

    ledger = pd.DataFrame(ledger_rows)
    ledger["daily_return"] = ledger["equity"].pct_change(fill_method=None).fillna(0.0)
    ledger["equity_peak"] = ledger["equity"].cummax()
    ledger["drawdown"] = ledger["equity"] / ledger["equity_peak"] - 1.0
    return ledger, pd.DataFrame(trade_rows)


def _validate_backtest_inputs(
    daily: pd.DataFrame,
    dividends: pd.DataFrame,
    signals: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    market = daily.copy()
    events = dividends.copy()
    targets = signals.copy()
    for column in ("date",):
        market[column] = pd.to_datetime(market[column])
        targets[column] = pd.to_datetime(targets[column])
    events["ex_date"] = pd.to_datetime(events["ex_date"])
    events["payment_date"] = pd.to_datetime(events["payment_date"])
    return (
        market.sort_values("date").drop_duplicates("date").reset_index(drop=True),
        events.sort_values("ex_date").reset_index(drop=True),
        targets.sort_values("date").drop_duplicates("date").reset_index(drop=True),
    )


def run_integrated_backtest(
    daily: pd.DataFrame,
    dividends: pd.DataFrame,
    signals: pd.DataFrame,
    bars_5m: pd.DataFrame,
    zones: pd.DataFrame,
    initial_cash: float,
    costs: StrategyCosts,
    fvg_config: dict[str, Any],
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    enable_intraday_t: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """执行0/50/100%隔夜仓位，并仅在已有隔夜半仓时完成一次日内T。"""

    market, events, targets = _validate_backtest_inputs(daily, dividends, signals)
    start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
    market = market.loc[market["date"].between(start, end)].reset_index(drop=True)
    targets = targets.loc[targets["date"].between(start, end)].copy()
    if len(market) < 2:
        raise ValueError("回测区间至少需要两个交易日")
    target_map = targets.set_index("date")["target_position"].to_dict()
    direction_map = targets.set_index("date")["t_direction"].to_dict()
    wave_z_map = targets.set_index("date")["wave_z"].to_dict()
    events_by_ex = {date: frame.to_dict("records") for date, frame in events.groupby("ex_date")}
    bars_by_date = {date: frame for date, frame in bars_5m.groupby("date", sort=False)}
    zones_by_date = {date: frame for date, frame in zones.groupby("date", sort=False)} if not zones.empty else {}

    cash = float(initial_cash)
    shares = 0
    receivable = 0.0
    scheduled_payments: dict[pd.Timestamp, float] = {}
    previous_date: pd.Timestamp | None = None
    previous_target = 0.0
    ledger_rows: list[dict[str, Any]] = []
    core_trade_rows: list[dict[str, Any]] = []
    t_trade_rows: list[dict[str, Any]] = []

    for row in market.itertuples(index=False):
        date = pd.Timestamp(row.date)
        shares_at_start = shares
        if previous_date is not None and costs.cash_annual_rate:
            cash *= 1.0 + costs.cash_annual_rate / 242.0

        entitlement_today = 0.0
        for event in events_by_ex.get(date, []):
            entitlement = shares_at_start * float(event["cash_dividend_per_share"])
            entitlement_today += entitlement
            payment_date = pd.Timestamp(event["payment_date"])
            scheduled_payments[payment_date] = scheduled_payments.get(payment_date, 0.0) + entitlement
        receivable += entitlement_today

        signal_date = previous_date
        if signal_date in target_map and pd.notna(target_map[signal_date]):
            previous_target = float(target_map[signal_date])
        target = previous_target
        direction = str(direction_map.get(signal_date, "NONE"))
        wave_z = float(wave_z_map.get(signal_date, np.nan))
        open_equity = cash + shares * float(row.open) + receivable
        target_value = target * open_equity
        current_value = shares * float(row.open)
        reference_side = "BUY" if target_value >= current_value else "SELL"
        reference_price = unfavorable_price(float(row.open), reference_side, costs)
        desired_shares = max(
            int(math.floor(target_value / reference_price / costs.lot_size)) * costs.lot_size,
            0,
        )
        quantity = desired_shares - shares
        old_sellable = shares_at_start
        if quantity > 0:
            while quantity > 0:
                execution = unfavorable_price(float(row.open), "BUY", costs)
                notional = quantity * execution
                fee = commission_for_notional(notional, costs)
                if notional + fee <= cash + 1e-9:
                    break
                quantity -= costs.lot_size
            if quantity > 0:
                execution = unfavorable_price(float(row.open), "BUY", costs)
                notional = quantity * execution
                fee = commission_for_notional(notional, costs)
                cash -= notional + fee
                shares += quantity
                core_trade_rows.append(
                    {"date": date, "signal_date": signal_date, "side": "买入", "quantity": quantity,
                     "execution_price": execution, "commission": fee, "target_position": target}
                )
        elif quantity < 0:
            sell_quantity = min(-quantity, old_sellable) // costs.lot_size * costs.lot_size
            if sell_quantity > 0:
                execution = unfavorable_price(float(row.open), "SELL", costs)
                notional = sell_quantity * execution
                fee = commission_for_notional(notional, costs)
                cash += notional - fee
                shares -= sell_quantity
                old_sellable -= sell_quantity
                core_trade_rows.append(
                    {"date": date, "signal_date": signal_date, "side": "卖出", "quantity": sell_quantity,
                     "execution_price": execution, "commission": fee, "target_position": target}
                )

        core_end_shares = shares
        t_eligible = (
            enable_intraday_t
            and np.isclose(target, 0.5)
            and core_end_shares >= costs.lot_size
            and old_sellable >= core_end_shares
            and direction in {"BUY_FIRST", "SELL_FIRST"}
        )
        t_trade = None
        if t_eligible:
            day_bars = bars_by_date.get(date, pd.DataFrame())
            day_zones = zones_by_date.get(date, pd.DataFrame())
            if not day_bars.empty and not day_zones.empty:
                t_trade = find_fvg_trade(
                    day_bars, day_zones, direction, cash, core_end_shares, costs, fvg_config
                )
        if t_trade is not None:
            quantity_t = int(t_trade["shares"])
            entry_notional = quantity_t * float(t_trade["entry_execution_price"])
            exit_notional = quantity_t * float(t_trade["exit_execution_price"])
            if direction == "BUY_FIRST":
                cash -= entry_notional + float(t_trade["entry_commission"])
                shares += quantity_t
                cash += exit_notional - float(t_trade["exit_commission"])
                shares -= quantity_t
                old_sellable -= quantity_t
            else:
                cash += entry_notional - float(t_trade["entry_commission"])
                shares -= quantity_t
                old_sellable -= quantity_t
                cash -= exit_notional + float(t_trade["exit_commission"])
                shares += quantity_t
            if shares != core_end_shares:
                raise AssertionError("做T后份额没有恢复到日内开始时的核心仓位")
            t_trade_rows.append(
                {"date": date, "signal_date": signal_date, "wave_z": wave_z,
                 "shares_before": core_end_shares, "shares_after": shares, **t_trade}
            )

        payment_today = float(scheduled_payments.pop(date, 0.0))
        if payment_today:
            cash += payment_today
            receivable -= payment_today
            if abs(receivable) < 1e-9:
                receivable = 0.0
        equity = cash + shares * float(row.close) + receivable
        ledger_rows.append(
            {
                "date": date,
                "signal_date_used": signal_date,
                "target_position": target,
                "t_direction": direction,
                "wave_z": wave_z,
                "t_eligible": bool(t_eligible),
                "t_executed": t_trade is not None,
                "shares_at_start": shares_at_start,
                "shares": shares,
                "cash": cash,
                "dividend_receivable": receivable,
                "dividend_entitlement_today": entitlement_today,
                "dividend_payment_today": payment_today,
                "close": float(row.close),
                "equity": equity,
                "actual_position": shares * float(row.close) / equity if equity else 0.0,
            }
        )
        previous_date = date

    ledger = pd.DataFrame(ledger_rows)
    ledger["daily_return"] = ledger["equity"].pct_change(fill_method=None).fillna(0.0)
    ledger["equity_peak"] = ledger["equity"].cummax()
    ledger["drawdown"] = ledger["equity"] / ledger["equity_peak"] - 1.0
    return ledger, pd.DataFrame(core_trade_rows), pd.DataFrame(t_trade_rows)


def summarize_strategy(
    ledger: pd.DataFrame,
    core_trades: pd.DataFrame,
    t_trades: pd.DataFrame,
    initial_cash: float,
) -> dict[str, Any]:
    elapsed_days = max((ledger["date"].iloc[-1] - ledger["date"].iloc[0]).days, 1)
    total_return = float(ledger["equity"].iloc[-1] / initial_cash - 1.0)
    cagr = float((1.0 + total_return) ** (365.25 / elapsed_days) - 1.0)
    volatility = float(ledger["daily_return"].std(ddof=1) * np.sqrt(242))
    mean_return = float(ledger["daily_return"].mean() * 242)
    core_cost = float(core_trades["commission"].sum()) if not core_trades.empty else 0.0
    t_cost = (
        float((t_trades["entry_commission"] + t_trades["exit_commission"]).sum())
        if not t_trades.empty else 0.0
    )
    return {
        "start_date": str(ledger["date"].iloc[0].date()),
        "end_date": str(ledger["date"].iloc[-1].date()),
        "observations": int(len(ledger)),
        "ending_equity_cny": float(ledger["equity"].iloc[-1]),
        "total_return": total_return,
        "cagr": cagr,
        "annualized_volatility": volatility,
        "sharpe_zero_rate": mean_return / volatility if volatility else None,
        "max_drawdown": float(ledger["drawdown"].min()),
        "average_exposure": float(ledger["actual_position"].mean()),
        "core_trade_count": int(len(core_trades)),
        "t_roundtrip_count": int(len(t_trades)),
        "t_net_pnl_cny": float(t_trades["net_pnl_cny"].sum()) if not t_trades.empty else 0.0,
        "explicit_cost_cny": core_cost + t_cost,
    }


def summarize_inventory_backtest(
    ledger: pd.DataFrame,
    trades: pd.DataFrame,
    initial_cash: float,
) -> dict[str, Any]:
    """汇总连续库存策略，不伪造单笔T的配对损益。"""

    if ledger.empty:
        raise ValueError("连续库存净值表为空")
    elapsed_days = max((ledger["date"].iloc[-1] - ledger["date"].iloc[0]).days, 1)
    total_return = float(ledger["equity"].iloc[-1] / initial_cash - 1.0)
    cagr = float((1.0 + total_return) ** (365.25 / elapsed_days) - 1.0)
    volatility = float(ledger["daily_return"].std(ddof=1) * np.sqrt(242))
    annualized_mean = float(ledger["daily_return"].mean() * 242)
    if trades.empty:
        buys = sells = 0
        explicit_cost = turnover = 0.0
        optional_roundtrips = 0
        locked_roundtrip_profit = 0.0
    else:
        buys = int(trades["side"].eq("BUY").sum())
        sells = int(trades["side"].eq("SELL").sum())
        explicit_cost = float(trades["commission"].sum())
        turnover = float(trades["notional_cny"].sum())
        optional_roundtrips = int(
            trades.get("trade_role", pd.Series(index=trades.index, dtype=str))
            .eq("OPTIONAL_INTRADAY_T_CLOSE")
            .sum()
        )
        locked_roundtrip_profit = float(
            trades.loc[
                trades.get("trade_role", pd.Series(index=trades.index, dtype=str)).eq(
                    "OPTIONAL_INTRADAY_T_CLOSE"
                ),
                "locked_pair_net_pnl_cny",
            ].sum()
        ) if "locked_pair_net_pnl_cny" in trades.columns else 0.0
    return {
        "start_date": str(ledger["date"].iloc[0].date()),
        "end_date": str(ledger["date"].iloc[-1].date()),
        "observations": int(len(ledger)),
        "ending_equity_cny": float(ledger["equity"].iloc[-1]),
        "total_return": total_return,
        "cagr": cagr,
        "annualized_volatility": volatility,
        "sharpe_zero_rate": annualized_mean / volatility if volatility else None,
        "max_drawdown": float(ledger["drawdown"].min()),
        "average_exposure": float(ledger["actual_position"].mean()),
        "average_absolute_position_gap": float(ledger["position_gap"].abs().mean()),
        "single_leg_trade_count": int(len(trades)),
        "buy_leg_count": buys,
        "sell_leg_count": sells,
        "optional_intraday_roundtrip_count": optional_roundtrips,
        "locked_intraday_roundtrip_profit_cny": locked_roundtrip_profit,
        "explicit_cost_cny": explicit_cost,
        "turnover_cny": turnover,
    }


def costs_with_slippage(costs: StrategyCosts, slippage_bps: float) -> StrategyCosts:
    return replace(costs, slippage_bps_per_leg=float(slippage_bps))
