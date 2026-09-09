"""510300图形状态马丁-海龟V2：因果特征、纸面账本与固定评估。"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = ROOT / "config" / "graph_regime_martin_turtle_v2.yaml"


VARIANTS: dict[str, dict[str, Any]] = {
    "T0_DONCHIAN_20_10": {
        "module": "NONE",
        "turtle": True,
        "graph_exit": False,
        "registered": False,
    },
    "T1_TURTLE_GRAPH_EXIT": {
        "module": "NONE",
        "turtle": True,
        "graph_exit": True,
        "registered": True,
    },
    "M0_CAPPED_MARTINGALE_ONLY": {
        "module": "MARTINGALE",
        "turtle": False,
        "graph_exit": False,
        "registered": True,
    },
    "P0_BIAS28_SWING_ONLY": {
        "module": "PRING_STYLE",
        "turtle": False,
        "graph_exit": False,
        "registered": True,
    },
    "S1_MARTINGALE_TURTLE_SWITCH": {
        "module": "MARTINGALE",
        "turtle": True,
        "graph_exit": True,
        "registered": True,
    },
    "S2_BIAS28_TURTLE_SWITCH": {
        "module": "PRING_STYLE",
        "turtle": True,
        "graph_exit": True,
        "registered": True,
    },
}


@dataclass(frozen=True)
class CostModel:
    commission_rate: float
    minimum_commission: float
    slippage_rate: float
    multiplier: float = 1.0

    def commission(self, raw_notional: float) -> float:
        return max(
            raw_notional * self.commission_rate * self.multiplier,
            self.minimum_commission * self.multiplier,
        )


def load_config(path: Path = CONFIG_FILE) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("V2配置必须是YAML对象")
    expected = [variant for variant, spec in VARIANTS.items() if spec["registered"]]
    actual = [candidate["id"] for candidate in config["candidates"]]
    if actual != expected:
        raise ValueError("V2候选集合与冻结实现不一致")
    return config


def load_inputs(root: Path, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    contracts = config["data_contracts"]
    market = pd.read_parquet(root / contracts["market"]["file"])
    benchmark = pd.read_parquet(root / contracts["benchmark"]["file"])
    dividends = pd.read_csv(root / contracts["distributions"]["file"])
    market["date"] = pd.to_datetime(market["date"], errors="raise").dt.normalize()
    benchmark["date"] = pd.to_datetime(benchmark["date"], errors="raise").dt.normalize()
    for column in ("record_date", "ex_date", "payment_date"):
        dividends[column] = pd.to_datetime(dividends[column], errors="raise").dt.normalize()
    market = market.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    benchmark = benchmark.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    dividends = dividends.sort_values("ex_date").reset_index(drop=True)
    return market, benchmark, dividends


def add_point_in_time_adjusted_ohlc(
    market: pd.DataFrame, dividends: pd.DataFrame
) -> pd.DataFrame:
    """构造现金分红调整OHLC；未来分红只造成统一比例缩放，不改变当时信号比较。"""

    data = market.copy()
    data = data.sort_values("date").reset_index(drop=True)
    factor = pd.Series(1.0, index=data.index, dtype=float)
    close_by_date = data.set_index("date")["close"].astype(float)
    for event in dividends.itertuples(index=False):
        prior = close_by_date.loc[close_by_date.index < event.ex_date]
        if prior.empty:
            continue
        previous_close = float(prior.iloc[-1])
        event_factor = (previous_close - float(event.cash_dividend_per_share)) / previous_close
        if not 0.0 < event_factor <= 1.0:
            raise ValueError(f"{event.ex_date.date()}分红调整因子非法：{event_factor}")
        factor.loc[data["date"].lt(event.ex_date)] *= event_factor
    data["adjustment_factor"] = factor
    for column in ("open", "high", "low", "close"):
        data[f"adjusted_{column}"] = pd.to_numeric(data[column], errors="raise") * factor
    return data


def _local_pivots(values: np.ndarray, order: int, kind: str) -> list[int]:
    result: list[int] = []
    for index in range(order, len(values) - order):
        center = values[index]
        left = values[index - order : index]
        right = values[index + 1 : index + order + 1]
        if not np.isfinite(center) or not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
            continue
        if kind == "LOW":
            valid = center <= left.min() and center <= right.min() and (
                center < left.min() or center < right.min()
            )
        else:
            valid = center >= left.max() and center >= right.max() and (
                center > left.max() or center > right.max()
            )
        if valid:
            result.append(index)
    return result


def _alternating_significant_price_pivots(
    features: pd.DataFrame, config: dict[str, Any]
) -> list[dict[str, Any]]:
    divergence = config["graph_divergence"]
    order = int(divergence["pivot_left_bars"])
    lows = features["adjusted_low"].to_numpy(float)
    highs = features["adjusted_high"].to_numpy(float)
    atr = features["atr14"].to_numpy(float)
    candidates = [
        {"kind": "LOW", "index": index, "value": lows[index]}
        for index in _local_pivots(lows, order, "LOW")
    ] + [
        {"kind": "HIGH", "index": index, "value": highs[index]}
        for index in _local_pivots(highs, order, "HIGH")
    ]
    candidates.sort(key=lambda item: (item["index"], item["kind"]))
    accepted: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate = dict(candidate)
        candidate["confirmation_index"] = candidate["index"] + order
        if not accepted:
            accepted.append(candidate)
            continue
        last = accepted[-1]
        if candidate["kind"] == last["kind"]:
            more_extreme = (
                candidate["value"] < last["value"]
                if candidate["kind"] == "LOW"
                else candidate["value"] > last["value"]
            )
            if more_extreme:
                accepted[-1] = candidate
            continue
        threshold = float(divergence["price_swing_minimum_atr"]) * atr[candidate["index"]]
        if np.isfinite(threshold) and abs(candidate["value"] - last["value"]) >= threshold:
            accepted.append(candidate)
    for pivot_id, pivot in enumerate(accepted):
        pivot["pivot_id"] = pivot_id
    return accepted


def build_divergence_events(
    features: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    divergence = config["graph_divergence"]
    order = int(divergence["pivot_left_bars"])
    tolerance = int(divergence["indicator_pair_tolerance_bars"])
    dif = features["dif"].to_numpy(float)
    price_pivots = _alternating_significant_price_pivots(features, config)
    dif_candidates = {
        "LOW": _local_pivots(dif, order, "LOW"),
        "HIGH": _local_pivots(dif, order, "HIGH"),
    }
    used_dif: set[int] = set()
    for pivot in price_pivots:
        eligible = [
            index
            for index in dif_candidates[pivot["kind"]]
            if abs(index - pivot["index"]) <= tolerance and index not in used_dif
        ]
        if not eligible:
            pivot["dif_index"] = None
            continue
        chosen = min(eligible, key=lambda index: (abs(index - pivot["index"]), index))
        used_dif.add(chosen)
        pivot["dif_index"] = chosen
        pivot["dif_value"] = float(dif[chosen])
        pivot["dif_confirmation_index"] = chosen + order

    used_price: set[int] = set()
    used_event_dif: set[int] = set()
    events: list[dict[str, Any]] = []
    for kind, event_type in (("LOW", "BOTTOM"), ("HIGH", "TOP")):
        same_side = [pivot for pivot in price_pivots if pivot["kind"] == kind]
        for previous, current in zip(same_side, same_side[1:]):
            gap = current["index"] - previous["index"]
            if not (
                int(divergence["minimum_same_side_pivot_gap"])
                <= gap
                <= int(divergence["maximum_same_side_pivot_gap"])
            ):
                continue
            if previous.get("dif_index") is None or current.get("dif_index") is None:
                continue
            if previous["pivot_id"] in used_price or current["pivot_id"] in used_price:
                continue
            if previous["dif_index"] in used_event_dif or current["dif_index"] in used_event_dif:
                continue
            if kind == "LOW":
                valid = (
                    current["value"] < previous["value"]
                    and current["dif_value"] > previous["dif_value"]
                )
            else:
                valid = (
                    current["value"] > previous["value"]
                    and current["dif_value"] < previous["dif_value"]
                )
            if not valid:
                continue
            confirmation_index = max(
                current["confirmation_index"], current["dif_confirmation_index"]
            )
            if confirmation_index >= len(features):
                continue
            used_price.update({previous["pivot_id"], current["pivot_id"]})
            used_event_dif.update({previous["dif_index"], current["dif_index"]})
            atr_value = float(features.loc[current["index"], "atr14"])
            events.append(
                {
                    "event_id": None,
                    "event_type": event_type,
                    "previous_price_pivot_index": previous["index"],
                    "current_price_pivot_index": current["index"],
                    "previous_price_pivot_date": features.loc[previous["index"], "date"],
                    "current_price_pivot_date": features.loc[current["index"], "date"],
                    "previous_price_value": previous["value"],
                    "current_price_value": current["value"],
                    "previous_dif_pivot_index": previous["dif_index"],
                    "current_dif_pivot_index": current["dif_index"],
                    "previous_dif_pivot_date": features.loc[previous["dif_index"], "date"],
                    "current_dif_pivot_date": features.loc[current["dif_index"], "date"],
                    "previous_dif_value": previous["dif_value"],
                    "current_dif_value": current["dif_value"],
                    "confirmation_index": confirmation_index,
                    "confirmation_date": features.loc[confirmation_index, "date"],
                    "confirmation_lag_from_price_bars": confirmation_index
                    - current["index"],
                    "price_pivot_gap_bars": gap,
                    "atr14_at_second_price_pivot": atr_value,
                    "invalidation_price": current["value"] - atr_value
                    if event_type == "BOTTOM"
                    else current["value"] + atr_value,
                }
            )
    columns = [
        "event_id",
        "event_type",
        "previous_price_pivot_index",
        "current_price_pivot_index",
        "previous_price_pivot_date",
        "current_price_pivot_date",
        "previous_price_value",
        "current_price_value",
        "previous_dif_pivot_index",
        "current_dif_pivot_index",
        "previous_dif_pivot_date",
        "current_dif_pivot_date",
        "previous_dif_value",
        "current_dif_value",
        "confirmation_index",
        "confirmation_date",
        "confirmation_lag_from_price_bars",
        "price_pivot_gap_bars",
        "atr14_at_second_price_pivot",
        "invalidation_price",
    ]
    result = pd.DataFrame(events, columns=columns).sort_values(
        ["confirmation_index", "event_type"]
    ).reset_index(drop=True)
    if not result.empty:
        result["event_id"] = [f"D{number:04d}" for number in range(1, len(result) + 1)]
    return result


def build_features(
    market: pd.DataFrame, dividends: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = add_point_in_time_adjusted_ohlc(market, dividends)
    indicators = config["indicators"]
    close = data["adjusted_close"]
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            data["adjusted_high"] - data["adjusted_low"],
            (data["adjusted_high"] - previous_close).abs(),
            (data["adjusted_low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr_window = int(indicators["atr"]["window"])
    data["atr14"] = true_range.ewm(
        alpha=1.0 / atr_window,
        adjust=False,
        min_periods=atr_window,
    ).mean()
    fast = int(indicators["macd_dif"]["fast_ema"])
    slow = int(indicators["macd_dif"]["slow_ema"])
    signal = int(indicators["macd_dif"]["signal_ema"])
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    data["dif"] = ema_fast - ema_slow
    data["dea"] = data["dif"].ewm(span=signal, adjust=False, min_periods=signal).mean()
    er_window = int(indicators["efficiency_ratio"]["window"])
    data["er20"] = close.diff(er_window).abs() / close.diff().abs().rolling(
        er_window, min_periods=er_window
    ).sum()
    for window in (20, 28, 60):
        data[f"ma{window}"] = close.rolling(window, min_periods=window).mean()
    data["ma60_slope"] = data["ma60"] - data["ma60"].shift(
        int(indicators["moving_averages"]["downtrend_slope_lookback"])
    )
    data["bias28"] = 100.0 * (close / data["ma28"] - 1.0)
    entry_window = int(indicators["donchian"]["entry_window"])
    exit_window = int(indicators["donchian"]["exit_window"])
    data["donchian_entry_high"] = data["adjusted_high"].shift(1).rolling(
        entry_window, min_periods=entry_window
    ).max()
    data["donchian_exit_low"] = data["adjusted_low"].shift(1).rolling(
        exit_window, min_periods=exit_window
    ).min()
    data["donchian_down_low"] = data["adjusted_low"].shift(1).rolling(
        entry_window, min_periods=entry_window
    ).min()
    data["turtle_breakout"] = close.gt(data["donchian_entry_high"])
    data["turtle_regular_exit"] = close.lt(data["donchian_exit_low"])
    data["trend_down"] = (
        close.lt(data["ma60"])
        & data["ma60_slope"].lt(0.0)
        & close.lt(data["donchian_down_low"])
    )
    data["range_eligible"] = (
        data["er20"].le(float(indicators["efficiency_ratio"]["range_maximum"]))
        & ~data["trend_down"]
    )
    data["chart_state"] = np.select(
        [data["trend_down"], data["range_eligible"]],
        ["TREND_DOWN", "RANGE"],
        default="AMBIGUOUS",
    )
    events = build_divergence_events(data, config)
    data["bottom_divergence_event"] = None
    data["top_divergence_event"] = None
    data["bottom_divergence_active"] = False
    data["bottom_divergence_active_id"] = None
    data["bottom_divergence_invalidation"] = np.nan
    active_days = int(config["graph_divergence"]["active_maximum_trading_days"])
    for event in events.itertuples(index=False):
        confirmation = int(event.confirmation_index)
        if event.event_type == "BOTTOM":
            data.at[confirmation, "bottom_divergence_event"] = event.event_id
            for index in range(confirmation, min(len(data), confirmation + active_days)):
                if data.loc[index, "adjusted_close"] < event.invalidation_price:
                    break
                data.at[index, "bottom_divergence_active"] = True
                data.at[index, "bottom_divergence_active_id"] = event.event_id
                data.at[index, "bottom_divergence_invalidation"] = event.invalidation_price
        else:
            data.at[confirmation, "top_divergence_event"] = event.event_id
    return data, events


def _cost_model(config: dict[str, Any], multiplier: float) -> CostModel:
    execution = config["price_and_execution"]
    return CostModel(
        commission_rate=float(execution["commission_rate_per_leg"]),
        minimum_commission=float(execution["minimum_commission_cny_per_leg"]),
        slippage_rate=float(execution["slippage_bps_per_leg"]) / 10_000.0,
        multiplier=multiplier,
    )


def _target_order(
    *,
    target_exposure: float,
    raw_open: float,
    cash: float,
    shares: int,
    receivable: float,
    lot_size: int,
    costs: CostModel,
) -> tuple[float, int, dict[str, Any] | None]:
    wealth_open = cash + shares * raw_open + receivable
    desired = int(np.floor(target_exposure * wealth_open / raw_open / lot_size) * lot_size)
    desired = max(0, desired)
    delta = desired - shares
    if delta > 0:
        execution_price = raw_open * (1.0 + costs.slippage_rate * costs.multiplier)
        while delta > 0:
            raw_notional = delta * raw_open
            commission = costs.commission(raw_notional)
            required = delta * execution_price + commission
            if required <= cash + 1e-9:
                break
            delta -= lot_size
        if delta <= 0:
            return cash, shares, None
        raw_notional = delta * raw_open
        commission = costs.commission(raw_notional)
        cash -= delta * execution_price + commission
        return cash, shares + delta, {
            "side": "BUY",
            "shares": delta,
            "raw_notional": raw_notional,
            "execution_price": execution_price,
            "commission": commission,
            "slippage_cost": delta * (execution_price - raw_open),
        }
    if delta < 0:
        sell_shares = -delta
        execution_price = raw_open * (1.0 - costs.slippage_rate * costs.multiplier)
        raw_notional = sell_shares * raw_open
        commission = costs.commission(raw_notional)
        cash += sell_shares * execution_price - commission
        return cash, shares - sell_shares, {
            "side": "SELL",
            "shares": sell_shares,
            "raw_notional": raw_notional,
            "execution_price": execution_price,
            "commission": commission,
            "slippage_cost": sell_shares * (raw_open - execution_price),
        }
    return cash, shares, None


def _events_by_date(dividends: pd.DataFrame, column: str) -> dict[pd.Timestamp, list[Any]]:
    mapping: dict[pd.Timestamp, list[Any]] = defaultdict(list)
    for event in dividends.itertuples(index=False):
        mapping[getattr(event, column)].append(event)
    return mapping


def simulate_variant(
    features: pd.DataFrame,
    dividends: pd.DataFrame,
    config: dict[str, Any],
    variant_id: str,
    *,
    cost_multiplier: float = 1.0,
) -> dict[str, pd.DataFrame]:
    if variant_id not in VARIANTS:
        raise KeyError(f"未知V2轨道：{variant_id}")
    variant = VARIANTS[variant_id]
    execution = config["price_and_execution"]
    lot_size = int(execution["lot_size_shares"])
    daily_cash_rate = (1.0 + float(execution["cash_annual_rate"])) ** (
        1.0 / int(execution["trading_days_per_year"])
    ) - 1.0
    costs = _cost_model(config, cost_multiplier)
    initial_capital = float(execution["initial_capital_cny"])
    record_events = _events_by_date(dividends, "record_date")
    ex_events = _events_by_date(dividends, "ex_date")
    payment_events = _events_by_date(dividends, "payment_date")
    entitlements: dict[pd.Timestamp, float] = {}
    receivables_by_payment: dict[pd.Timestamp, float] = defaultdict(float)

    cash = initial_capital
    shares = 0
    receivable = 0.0
    pending: dict[str, Any] | None = None
    position_mode = "CASH"
    turtle_rearmed = True
    martin_layer = 0
    last_layer_adjusted_price = np.nan
    module_holding_days = 0
    module_invalidation = np.nan
    used_bottom_events: set[str] = set()
    cycle: dict[str, Any] | None = None
    cycles: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    executions: list[dict[str, Any]] = []

    for index, row in features.iterrows():
        date = row["date"]
        for event in ex_events.get(date, []):
            amount = entitlements.get(event.ex_date, 0.0)
            if amount:
                receivable += amount
                receivables_by_payment[event.payment_date] += amount
        if date in payment_events:
            amount = receivables_by_payment.pop(date, 0.0)
            cash += amount
            receivable -= amount

        shares_before = shares
        mode_before = position_mode
        executed_reason = None
        day_commission = 0.0
        day_slippage = 0.0
        day_notional = 0.0
        if pending is not None:
            cash, shares, trade = _target_order(
                target_exposure=float(pending["target_exposure"]),
                raw_open=float(row["open"]),
                cash=cash,
                shares=shares,
                receivable=receivable,
                lot_size=lot_size,
                costs=costs,
            )
            executed_reason = pending["reason"]
            if trade is not None:
                day_commission = float(trade["commission"])
                day_slippage = float(trade["slippage_cost"])
                day_notional = float(trade["raw_notional"])
                executions.append(
                    {
                        "variant_id": variant_id,
                        "signal_date": pending["signal_date"],
                        "execution_date": date,
                        "reason": pending["reason"],
                        "target_exposure": pending["target_exposure"],
                        "mode_before": mode_before,
                        "side": trade["side"],
                        "trade_shares": trade["shares"],
                        "raw_open": float(row["open"]),
                        "execution_price": trade["execution_price"],
                        "raw_notional": trade["raw_notional"],
                        "commission": trade["commission"],
                        "slippage_cost": trade["slippage_cost"],
                        "shares_after": shares,
                        "bottom_event_id": pending.get("bottom_event_id"),
                        "top_event_id": pending.get("top_event_id"),
                    }
                )
            reason = pending["reason"]
            if reason in {"TURTLE_ENTRY", "MARTIN_TO_TURTLE", "PRING_TO_TURTLE"}:
                position_mode = "TURTLE_LONG"
                martin_layer = 0
                module_holding_days = 0
                module_invalidation = np.nan
            elif reason.startswith("MARTINGALE_LAYER_"):
                position_mode = "MARTINGALE"
                martin_layer = int(reason.rsplit("_", 1)[-1])
                module_holding_days = 0 if martin_layer == 1 else module_holding_days
                module_invalidation = float(pending["invalidation_price"])
                if trade is not None and trade["side"] == "BUY":
                    last_layer_adjusted_price = float(trade["execution_price"]) * float(
                        row["adjustment_factor"]
                    )
            elif reason == "PRING_ENTRY":
                position_mode = "PRING_STYLE"
                module_holding_days = 0
                module_invalidation = float(pending["invalidation_price"])
            elif pending["target_exposure"] == 0.0:
                position_mode = "CASH"
                martin_layer = 0
                module_holding_days = 0
                module_invalidation = np.nan
                last_layer_adjusted_price = np.nan
                if reason == "TURTLE_GRAPH_EXIT":
                    turtle_rearmed = False
            pending = None

        if shares_before == 0 and shares > 0:
            cycle = {
                "variant_id": variant_id,
                "entry_date": date,
                "entry_reason": executed_reason,
                "entry_equity": cash + shares * float(row["open"]) + receivable,
                "switch_date": None,
                "switch_reason": None,
                "switch_equity": None,
            }
        elif (
            shares_before > 0
            and shares > 0
            and cycle is not None
            and executed_reason in {"MARTIN_TO_TURTLE", "PRING_TO_TURTLE"}
        ):
            cycle["switch_date"] = date
            cycle["switch_reason"] = executed_reason
            cycle["switch_equity"] = cash + shares * float(row["open"]) + receivable
        elif shares_before > 0 and shares == 0 and cycle is not None:
            exit_equity = cash + receivable
            switch_equity = cycle.get("switch_equity")
            cycles.append(
                {
                    **cycle,
                    "exit_date": date,
                    "exit_reason": executed_reason,
                    "exit_equity": exit_equity,
                    "net_pnl": exit_equity - float(cycle["entry_equity"]),
                    "pnl_to_switch": (
                        float(switch_equity) - float(cycle["entry_equity"])
                        if switch_equity is not None
                        else None
                    ),
                    "pnl_after_switch": (
                        exit_equity - float(switch_equity)
                        if switch_equity is not None
                        else None
                    ),
                    "holding_calendar_days": (date - cycle["entry_date"]).days,
                }
            )
            cycle = None

        cash *= 1.0 + daily_cash_rate
        equity = cash + shares * float(row["close"]) + receivable
        exposure = shares * float(row["close"]) / equity if equity > 0 else np.nan
        for event in record_events.get(date, []):
            entitlements[event.ex_date] = shares * float(event.cash_dividend_per_share)

        if not turtle_rearmed and pd.notna(row["ma20"]) and row["adjusted_close"] <= row["ma20"]:
            turtle_rearmed = True
        if position_mode in {"MARTINGALE", "PRING_STYLE"}:
            module_holding_days += 1

        next_pending: dict[str, Any] | None = None
        bottom_event_id = row["bottom_divergence_active_id"]
        top_event_id = row["top_divergence_event"]
        has_bottom_event = pd.notna(bottom_event_id) and str(bottom_event_id) != ""
        has_top_event = pd.notna(top_event_id) and str(top_event_id) != ""
        breakout = bool(row["turtle_breakout"]) if pd.notna(row["turtle_breakout"]) else False
        if index + 1 < len(features):
            if position_mode == "TURTLE_LONG":
                if variant["graph_exit"] and has_top_event:
                    next_pending = {
                        "signal_date": date,
                        "target_exposure": 0.0,
                        "reason": "TURTLE_GRAPH_EXIT",
                        "top_event_id": top_event_id,
                    }
                elif bool(row["turtle_regular_exit"]):
                    next_pending = {
                        "signal_date": date,
                        "target_exposure": 0.0,
                        "reason": "TURTLE_REGULAR_EXIT",
                    }
            elif position_mode in {"MARTINGALE", "PRING_STYLE"}:
                if variant["turtle"] and breakout:
                    next_pending = {
                        "signal_date": date,
                        "target_exposure": 1.0,
                        "reason": "MARTIN_TO_TURTLE"
                        if position_mode == "MARTINGALE"
                        else "PRING_TO_TURTLE",
                    }
                else:
                    exit_reason = None
                    if not variant["turtle"] and breakout:
                        exit_reason = "CONTROL_BREAKOUT_EXIT"
                    elif has_top_event:
                        exit_reason = "MODULE_TOP_DIVERGENCE_EXIT"
                    elif bool(row["trend_down"]):
                        exit_reason = "MODULE_TREND_DOWN_EXIT"
                    elif np.isfinite(module_invalidation) and row["adjusted_close"] < module_invalidation:
                        exit_reason = "MODULE_INVALIDATION_EXIT"
                    elif row["adjusted_close"] >= row["ma28"]:
                        exit_reason = "MODULE_MA28_EXIT"
                    elif module_holding_days >= 20:
                        exit_reason = "MODULE_TIME_EXIT"
                    if exit_reason:
                        next_pending = {
                            "signal_date": date,
                            "target_exposure": 0.0,
                            "reason": exit_reason,
                            "top_event_id": top_event_id,
                        }
                    elif (
                        position_mode == "MARTINGALE"
                        and martin_layer < 3
                        and bool(row["range_eligible"])
                        and np.isfinite(last_layer_adjusted_price)
                        and row["adjusted_close"]
                        <= last_layer_adjusted_price - row["atr14"]
                    ):
                        layer = martin_layer + 1
                        target = [0.125, 0.375, 0.875][layer - 1]
                        next_pending = {
                            "signal_date": date,
                            "target_exposure": target,
                            "reason": f"MARTINGALE_LAYER_{layer}",
                            "invalidation_price": module_invalidation,
                        }
            else:
                if variant["turtle"] and turtle_rearmed and breakout:
                    next_pending = {
                        "signal_date": date,
                        "target_exposure": 1.0,
                        "reason": "TURTLE_ENTRY",
                    }
                elif (
                    variant["module"] in {"MARTINGALE", "PRING_STYLE"}
                    and bool(row["range_eligible"])
                    and bool(row["bottom_divergence_active"])
                    and has_bottom_event
                    and bottom_event_id not in used_bottom_events
                ):
                    if variant["module"] == "MARTINGALE":
                        next_pending = {
                            "signal_date": date,
                            "target_exposure": 0.125,
                            "reason": "MARTINGALE_LAYER_1",
                            "bottom_event_id": bottom_event_id,
                            "invalidation_price": float(
                                row["bottom_divergence_invalidation"]
                            ),
                        }
                    elif row["bias28"] < -10.0:
                        next_pending = {
                            "signal_date": date,
                            "target_exposure": 0.50,
                            "reason": "PRING_ENTRY",
                            "bottom_event_id": bottom_event_id,
                            "invalidation_price": float(
                                row["bottom_divergence_invalidation"]
                            ),
                        }
                    if next_pending is not None:
                        used_bottom_events.add(str(bottom_event_id))
        pending = next_pending
        ledger.append(
            {
                "variant_id": variant_id,
                "date": date,
                "raw_open": float(row["open"]),
                "raw_close": float(row["close"]),
                "adjusted_close": float(row["adjusted_close"]),
                "adjustment_factor": float(row["adjustment_factor"]),
                "chart_state": row["chart_state"],
                "position_mode": position_mode,
                "turtle_rearmed": turtle_rearmed,
                "range_eligible": bool(row["range_eligible"]),
                "trend_down": bool(row["trend_down"]),
                "turtle_breakout": bool(row["turtle_breakout"]),
                "turtle_regular_exit": bool(row["turtle_regular_exit"]),
                "bottom_divergence_event": row["bottom_divergence_event"],
                "bottom_divergence_active_id": bottom_event_id,
                "top_divergence_event": top_event_id,
                "bias28": row["bias28"],
                "atr14": row["atr14"],
                "er20": row["er20"],
                "shares": shares,
                "cash": cash,
                "receivable": receivable,
                "equity": equity,
                "exposure": exposure,
                "day_commission": day_commission,
                "day_slippage": day_slippage,
                "day_raw_notional": day_notional,
                "executed_reason": executed_reason,
                "next_signal_reason": next_pending["reason"]
                if next_pending is not None
                else None,
                "next_target_exposure": next_pending["target_exposure"]
                if next_pending is not None
                else np.nan,
                "martin_layer": martin_layer,
                "module_holding_days": module_holding_days,
            }
        )
    return {
        "ledger": pd.DataFrame(ledger),
        "executions": pd.DataFrame(executions),
        "cycles": pd.DataFrame(cycles),
    }


def simulate_static_exposure(
    features: pd.DataFrame,
    dividends: pd.DataFrame,
    config: dict[str, Any],
    exposure_target: float,
    *,
    cost_multiplier: float = 1.0,
) -> pd.DataFrame:
    execution = config["price_and_execution"]
    costs = _cost_model(config, cost_multiplier)
    lot_size = int(execution["lot_size_shares"])
    daily_cash_rate = (1.0 + float(execution["cash_annual_rate"])) ** (
        1.0 / int(execution["trading_days_per_year"])
    ) - 1.0
    cash = float(execution["initial_capital_cny"])
    shares = 0
    receivable = 0.0
    entitlements: dict[pd.Timestamp, float] = {}
    receivables_by_payment: dict[pd.Timestamp, float] = defaultdict(float)
    record_events = _events_by_date(dividends, "record_date")
    ex_events = _events_by_date(dividends, "ex_date")
    payment_events = _events_by_date(dividends, "payment_date")
    rows: list[dict[str, Any]] = []
    for index, row in features.iterrows():
        date = row["date"]
        for event in ex_events.get(date, []):
            amount = entitlements.get(event.ex_date, 0.0)
            receivable += amount
            receivables_by_payment[event.payment_date] += amount
        if date in payment_events:
            amount = receivables_by_payment.pop(date, 0.0)
            cash += amount
            receivable -= amount
        commission = slippage = notional = 0.0
        if index == 0:
            cash, shares, trade = _target_order(
                target_exposure=exposure_target,
                raw_open=float(row["open"]),
                cash=cash,
                shares=shares,
                receivable=receivable,
                lot_size=lot_size,
                costs=costs,
            )
            if trade is not None:
                commission = float(trade["commission"])
                slippage = float(trade["slippage_cost"])
                notional = float(trade["raw_notional"])
        cash *= 1.0 + daily_cash_rate
        equity = cash + shares * float(row["close"]) + receivable
        for event in record_events.get(date, []):
            entitlements[event.ex_date] = shares * float(event.cash_dividend_per_share)
        rows.append(
            {
                "date": date,
                "equity": equity,
                "shares": shares,
                "cash": cash,
                "receivable": receivable,
                "exposure": shares * float(row["close"]) / equity,
                "day_commission": commission,
                "day_slippage": slippage,
                "day_raw_notional": notional,
            }
        )
    return pd.DataFrame(rows)


def performance_metrics(ledger: pd.DataFrame, cash_annual_rate: float = 0.015) -> dict[str, float]:
    equity = pd.to_numeric(ledger["equity"], errors="raise")
    dates = pd.to_datetime(ledger["date"])
    daily = equity.pct_change(fill_method=None).dropna()
    elapsed_years = max((dates.iloc[-1] - dates.iloc[0]).days / 365.2425, 1.0 / 365.2425)
    total_return = equity.iloc[-1] / equity.iloc[0] - 1.0
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1.0 / elapsed_years) - 1.0
    volatility = float(daily.std(ddof=1) * np.sqrt(242)) if len(daily) > 1 else np.nan
    daily_cash = (1.0 + cash_annual_rate) ** (1.0 / 242.0) - 1.0
    sharpe = (
        float((daily.mean() - daily_cash) * 242.0 / volatility)
        if volatility and np.isfinite(volatility) and volatility > 0
        else np.nan
    )
    drawdown = equity / equity.cummax() - 1.0
    return {
        "start_date": str(dates.iloc[0].date()),
        "end_date": str(dates.iloc[-1].date()),
        "ending_equity_cny": float(equity.iloc[-1]),
        "total_return": float(total_return),
        "cagr": float(cagr),
        "annualized_volatility": volatility,
        "sharpe": sharpe,
        "maximum_drawdown": float(drawdown.min()),
        "average_exposure": float(pd.to_numeric(ledger["exposure"]).mean()),
        "turnover_notional_cny": float(pd.to_numeric(ledger["day_raw_notional"]).sum()),
        "commission_cny": float(pd.to_numeric(ledger["day_commission"]).sum()),
        "slippage_cny": float(pd.to_numeric(ledger["day_slippage"]).sum()),
    }


def _period_return(ledger: pd.DataFrame, start: str, end: str) -> float | None:
    part = ledger.loc[ledger["date"].between(pd.Timestamp(start), pd.Timestamp(end))]
    if len(part) < 2:
        return None
    return float(part["equity"].iloc[-1] / part["equity"].iloc[0] - 1.0)


def circular_shift_percentile(
    exposure: pd.Series,
    adjusted_close: pd.Series,
    *,
    repetitions: int,
    minimum_shift: int,
    random_seed: int,
) -> float:
    returns = adjusted_close.pct_change(fill_method=None).fillna(0.0).to_numpy(float)
    position = exposure.fillna(0.0).clip(0.0, 1.0).to_numpy(float)
    cost_rate = 0.0016
    observed = float(np.sum(position[:-1] * returns[1:]) - np.sum(np.abs(np.diff(position))) * cost_rate)
    eligible = np.arange(minimum_shift, len(position) - minimum_shift)
    if len(eligible) == 0:
        return np.nan
    rng = np.random.default_rng(random_seed)
    shifts = rng.choice(eligible, size=repetitions, replace=True)
    placebo = np.empty(repetitions, dtype=float)
    for number, shift in enumerate(shifts):
        shifted = np.roll(position, int(shift))
        placebo[number] = np.sum(shifted[:-1] * returns[1:]) - np.sum(
            np.abs(np.diff(shifted))
        ) * cost_rate
    return float((np.sum(placebo < observed) + 0.5 * np.sum(placebo == observed)) / repetitions)


def block_bootstrap_mean_interval(
    values: pd.Series,
    *,
    repetitions: int,
    block_length: int,
    random_seed: int,
) -> tuple[float, float]:
    clean = values.dropna().to_numpy(float)
    if len(clean) < block_length:
        return np.nan, np.nan
    rng = np.random.default_rng(random_seed)
    block_starts = np.arange(0, len(clean) - block_length + 1)
    blocks_needed = int(np.ceil(len(clean) / block_length))
    means = np.empty(repetitions, dtype=float)
    for repetition in range(repetitions):
        starts = rng.choice(block_starts, size=blocks_needed, replace=True)
        sample = np.concatenate(
            [clean[start : start + block_length] for start in starts]
        )[: len(clean)]
        means[repetition] = sample.mean() * 242.0
    lower, upper = np.quantile(means, [0.025, 0.975])
    return float(lower), float(upper)


def evaluate_results(
    features: pd.DataFrame,
    dividends: pd.DataFrame,
    benchmark: pd.DataFrame,
    config: dict[str, Any],
    base_results: dict[str, dict[str, pd.DataFrame]],
    stress_results: dict[str, dict[str, pd.DataFrame]],
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    cash_rate = float(config["price_and_execution"]["cash_annual_rate"])
    buy_hold = simulate_static_exposure(features, dividends, config, 1.0)
    buy_hold_metrics = performance_metrics(buy_hold, cash_rate)
    h00300 = benchmark.loc[
        benchmark["date"].between(features["date"].min(), features["date"].max())
    ].copy()
    h00300_close = pd.to_numeric(h00300["close"], errors="raise")
    h00300_years = (h00300["date"].iloc[-1] - h00300["date"].iloc[0]).days / 365.2425
    h00300_cagr = float(
        (h00300_close.iloc[-1] / h00300_close.iloc[0]) ** (1.0 / h00300_years) - 1.0
    )
    h00300_daily = h00300_close.pct_change(fill_method=None).dropna()
    h00300_vol = float(h00300_daily.std(ddof=1) * np.sqrt(242))
    daily_cash = (1.0 + cash_rate) ** (1.0 / 242.0) - 1.0
    h00300_sharpe = float((h00300_daily.mean() - daily_cash) * 242.0 / h00300_vol)

    ledgers: dict[str, pd.DataFrame] = {"BUY_HOLD": buy_hold}
    metrics: dict[str, Any] = {}
    static_cache: dict[tuple[float, float], pd.DataFrame] = {}
    for number, variant_id in enumerate(VARIANTS):
        result = base_results[variant_id]
        ledger = result["ledger"]
        stress_ledger = stress_results[variant_id]["ledger"]
        ledgers[variant_id] = ledger
        base = performance_metrics(ledger, cash_rate)
        stress = performance_metrics(stress_ledger, cash_rate)
        cycles = result["cycles"]
        closed_trades = int(len(cycles))
        average_exposure = base["average_exposure"]
        key_mean = (round(average_exposure, 8), 1.0)
        if key_mean not in static_cache:
            static_cache[key_mean] = simulate_static_exposure(
                features, dividends, config, average_exposure
            )
        static_mean = static_cache[key_mean]
        static_mean_metrics = performance_metrics(static_mean, cash_rate)
        vol_ratio = (
            min(1.0, base["annualized_volatility"] / buy_hold_metrics["annualized_volatility"])
            if buy_hold_metrics["annualized_volatility"] > 0
            else 0.0
        )
        key_vol = (round(vol_ratio, 8), 1.0)
        if key_vol not in static_cache:
            static_cache[key_vol] = simulate_static_exposure(features, dividends, config, vol_ratio)
        static_vol = static_cache[key_vol]
        static_vol_metrics = performance_metrics(static_vol, cash_rate)
        stress_static = simulate_static_exposure(
            features,
            dividends,
            config,
            average_exposure,
            cost_multiplier=2.0,
        )
        stress_static_metrics = performance_metrics(stress_static, cash_rate)
        strategy_daily = ledger["equity"].pct_change(fill_method=None)
        static_daily = static_mean["equity"].pct_change(fill_method=None)
        rolling_excess = (
            ledger["equity"].div(ledger["equity"].shift(242))
            / buy_hold["equity"].div(buy_hold["equity"].shift(242))
            - 1.0
        ).dropna()
        periods: list[dict[str, Any]] = []
        for period in config["evaluation"]["predefined_periods"]:
            strategy_return = _period_return(ledger, period["start"], period["end"])
            buy_return = _period_return(buy_hold, period["start"], period["end"])
            periods.append(
                {
                    **period,
                    "strategy_return": strategy_return,
                    "buy_hold_return": buy_return,
                    "excess_return": strategy_return - buy_return
                    if strategy_return is not None and buy_return is not None
                    else None,
                }
            )
        positive_periods = sum(
            item["excess_return"] is not None and item["excess_return"] > 0
            for item in periods
        )
        net_profits = cycles["net_pnl"].clip(lower=0.0) if not cycles.empty else pd.Series(dtype=float)
        profit_sum = float(net_profits.sum())
        maximum_trade_share = (
            float(net_profits.max() / profit_sum) if profit_sum > 0 else np.inf
        )
        shift_percentile = circular_shift_percentile(
            ledger["exposure"],
            features["adjusted_close"],
            repetitions=int(config["evaluation"]["circular_shift_repetitions"]),
            minimum_shift=20,
            random_seed=int(config["evaluation"]["random_seed"]) + number,
        )
        bootstrap_lower, bootstrap_upper = block_bootstrap_mean_interval(
            strategy_daily - static_daily,
            repetitions=int(config["evaluation"]["block_bootstrap_repetitions"]),
            block_length=int(config["evaluation"]["block_length_trading_days"]),
            random_seed=int(config["evaluation"]["random_seed"]) + 100 + number,
        )
        timing_contribution = base["total_return"] - static_mean_metrics["total_return"]
        stress_timing = stress["total_return"] - stress_static_metrics["total_return"]
        gate_config = config["evaluation"]["gates"]
        best_benchmark_sharpe = max(
            buy_hold_metrics["sharpe"],
            static_mean_metrics["sharpe"],
            static_vol_metrics["sharpe"],
        )
        gates = {
            "annualized_excess_vs_510300": base["cagr"] - buy_hold_metrics["cagr"]
            > float(gate_config["annualized_excess_vs_510300_buy_hold_minimum"]),
            "annualized_excess_vs_h00300": base["cagr"] - h00300_cagr
            > float(gate_config["annualized_excess_vs_h00300_minimum"]),
            "timing_contribution_after_cost": timing_contribution
            > float(gate_config["timing_contribution_after_cost_minimum"]),
            "excess_vs_static_mean": base["total_return"] - static_mean_metrics["total_return"]
            > float(gate_config["excess_vs_static_mean_exposure_minimum"]),
            "excess_vs_static_vol": base["total_return"] - static_vol_metrics["total_return"]
            > float(gate_config["excess_vs_static_vol_matched_minimum"]),
            "sharpe_minimum": base["sharpe"] >= float(gate_config["sharpe_minimum"]),
            "sharpe_improvement": base["sharpe"] - best_benchmark_sharpe
            >= float(gate_config["sharpe_improvement_minimum"]),
            "drawdown_ratio": abs(base["maximum_drawdown"])
            <= float(gate_config["drawdown_ratio_to_buy_hold_maximum"])
            * abs(buy_hold_metrics["maximum_drawdown"]),
            "rolling_excess_median": float(rolling_excess.median())
            > float(gate_config["rolling_242d_excess_median_minimum"]),
            "rolling_excess_positive_ratio": float(rolling_excess.gt(0.0).mean())
            >= float(gate_config["rolling_242d_positive_ratio_minimum"]),
            "positive_predefined_periods": positive_periods
            >= int(gate_config["positive_predefined_periods_minimum"]),
            "minimum_closed_trades": closed_trades
            >= int(gate_config["minimum_closed_trades"]),
            "double_cost_timing_positive": stress_timing > 0.0,
            "circular_shift_percentile": shift_percentile
            >= float(gate_config["circular_shift_percentile_minimum"]),
            "single_trade_concentration": maximum_trade_share
            <= float(gate_config["maximum_single_trade_profit_contribution_share"]),
        }
        metrics[variant_id] = {
            "registered_candidate": VARIANTS[variant_id]["registered"],
            "base": base,
            "double_cost": stress,
            "closed_trades": closed_trades,
            "static_mean_exposure": static_mean_metrics,
            "static_vol_matched": static_vol_metrics,
            "vol_matched_exposure": vol_ratio,
            "timing_contribution_after_cost": timing_contribution,
            "double_cost_timing_contribution": stress_timing,
            "annualized_excess_vs_510300": base["cagr"] - buy_hold_metrics["cagr"],
            "annualized_excess_vs_h00300": base["cagr"] - h00300_cagr,
            "rolling_242d_excess_median": float(rolling_excess.median()),
            "rolling_242d_excess_positive_ratio": float(rolling_excess.gt(0.0).mean()),
            "predefined_periods": periods,
            "positive_predefined_periods": positive_periods,
            "maximum_single_trade_profit_contribution_share": maximum_trade_share,
            "circular_shift_percentile": shift_percentile,
            "bootstrap_annualized_timing_mean_95_interval": [
                bootstrap_lower,
                bootstrap_upper,
            ],
            "gates": gates,
        }

    component_pairs = {
        "S1_MARTINGALE_TURTLE_SWITCH": [
            "T1_TURTLE_GRAPH_EXIT",
            "M0_CAPPED_MARTINGALE_ONLY",
        ],
        "S2_BIAS28_TURTLE_SWITCH": [
            "T1_TURTLE_GRAPH_EXIT",
            "P0_BIAS28_SWING_ONLY",
        ],
    }
    for candidate, components in component_pairs.items():
        candidate_sharpe = metrics[candidate]["base"]["sharpe"]
        component_sharpe = max(metrics[item]["base"]["sharpe"] for item in components)
        metrics[candidate]["gates"]["sharpe_exceeds_both_components"] = (
            candidate_sharpe > component_sharpe
        )
        metrics[candidate]["component_sharpe_maximum"] = component_sharpe

    for variant_id, evidence in metrics.items():
        all_pass = all(evidence["gates"].values())
        if evidence["closed_trades"] < int(config["evaluation"]["gates"]["minimum_closed_trades"]):
            evidence["decision"] = "INSUFFICIENT_EVIDENCE"
        else:
            evidence["decision"] = "PASS" if all_pass else "FAIL"

    report = {
        "project_id": config["protocol"]["project_id"],
        "data_cutoff": config["protocol"]["historical_contamination_cutoff"],
        "evidence_label": config["protocol"]["historical_evidence_label"],
        "true_forward_start": None,
        "buy_hold": buy_hold_metrics,
        "h00300": {
            "cagr": h00300_cagr,
            "annualized_volatility": h00300_vol,
            "sharpe": h00300_sharpe,
        },
        "variants": metrics,
        "registered_pass_count": sum(
            evidence["registered_candidate"] and evidence["decision"] == "PASS"
            for evidence in metrics.values()
        ),
        "live_position_mapping_enabled": False,
        "order_generation_enabled": False,
        "broker_connection_enabled": False,
    }
    return report, ledgers
