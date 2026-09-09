"""小账户点时成分股整仓轮换回测引擎。"""

from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SmallAccountCosts:
    """小账户股票交易成本与硬约束。"""

    commission_rate: float = 0.0003
    minimum_commission_cny: float = 5.0
    stamp_duty_sell_rate: float = 0.0005
    stamp_duty_sell_rate_before_reduction: float = 0.001
    stamp_duty_reduction_effective_date: str = "2023-08-28"
    slippage_bps_per_leg: float = 5.0
    cash_annual_rate: float = 0.015
    lot_size: int = 100
    minimum_trade_notional_cny: float = 5000.0
    maximum_positions: int = 3

    def stamp_duty_rate_for_date(self, date: pd.Timestamp) -> float:
        """返回交易日适用的卖出印花税率。"""

        cutoff = pd.Timestamp(self.stamp_duty_reduction_effective_date)
        return (
            self.stamp_duty_sell_rate_before_reduction
            if pd.Timestamp(date) < cutoff
            else self.stamp_duty_sell_rate
        )


def _commission(notional: float, costs: SmallAccountCosts) -> float:
    if notional <= 1e-9:
        return 0.0
    return max(costs.minimum_commission_cny, notional * costs.commission_rate)


def _weights_json(symbols: np.ndarray, values: np.ndarray) -> str:
    return json.dumps(
        {
            str(symbol): round(float(value), 8)
            for symbol, value in zip(symbols, values, strict=True)
            if value > 1e-10
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def load_full_component_history(files: list[str]) -> pd.DataFrame:
    """读取完整证券历史，供成分股调出后继续估值和卖出。"""

    required = [
        "date",
        "con_code",
        "raw_open",
        "total_return_open",
        "total_return_close",
        "volume",
    ]
    frames = [pd.read_parquet(path, columns=required) for path in files]
    if not frames:
        raise ValueError("完整成分股历史文件为空")
    history = pd.concat(frames, ignore_index=True)
    history["date"] = pd.to_datetime(history["date"])
    history["is_suspended"] = history["volume"].fillna(0.0).le(0.0)
    if history[["date", "con_code"]].duplicated().any():
        raise ValueError("完整证券历史存在重复证券日期")
    return history


def run_small_account_open_backtest(
    execution_history: pd.DataFrame,
    targets: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    initial_cash: float,
    costs: SmallAccountCosts,
    maximum_open_gap_for_trade: float = 0.095,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按整仓轮换执行，任何成交腿都不得小于最低金额。"""

    required_market = {
        "date",
        "con_code",
        "raw_open",
        "total_return_open",
        "total_return_close",
        "is_suspended",
    }
    required_targets = {"signal_date", "con_code", "selection_rank", "regime"}
    if missing := required_market.difference(execution_history.columns):
        raise ValueError(f"执行行情缺少字段：{sorted(missing)}")
    if missing := required_targets.difference(targets.columns):
        raise ValueError(f"目标表缺少字段：{sorted(missing)}")
    if initial_cash <= 0:
        raise ValueError("初始资金必须为正")
    if costs.maximum_positions * costs.minimum_trade_notional_cny >= initial_cash:
        raise ValueError("最大持仓数乘最低成交额必须严格小于本金，以预留交易费用")

    market = execution_history.copy()
    market["date"] = pd.to_datetime(market["date"])
    market = market.loc[market["date"].isin(calendar)].copy()
    signals = targets.copy()
    signals["signal_date"] = pd.to_datetime(signals["signal_date"])
    if signals[["signal_date", "con_code"]].duplicated().any():
        raise ValueError("同一信号日出现重复证券")
    if signals.groupby("signal_date").size().max() > costs.maximum_positions:
        raise ValueError("目标证券数超过最大持仓数")

    symbols = np.array(sorted(market["con_code"].astype(str).unique()), dtype=object)
    symbol_to_index = {str(symbol): index for index, symbol in enumerate(symbols)}

    def wide(column: str) -> pd.DataFrame:
        return market.pivot(index="date", columns="con_code", values=column).reindex(
            index=calendar, columns=symbols
        )

    raw_open = wide("raw_open").astype(float)
    observed_total_open = wide("total_return_open").astype(float)
    observed_total_close = wide("total_return_close").astype(float)
    suspended = wide("is_suspended").fillna(True).astype(bool)
    total_close = observed_total_close.ffill()
    previous_close = total_close.shift(1)
    mark_open = observed_total_open.where(observed_total_open.notna(), previous_close)
    open_gap = observed_total_open.divide(previous_close).subtract(1.0)

    signal_targets: dict[pd.Timestamp, list[int]] = {}
    signal_regimes: dict[pd.Timestamp, str] = {}
    for signal_date, frame in signals.groupby("signal_date", sort=True):
        ordered = frame.sort_values(["selection_rank", "con_code"])
        indices = [
            symbol_to_index[str(code)]
            for code in ordered["con_code"]
            if str(code) in symbol_to_index
        ]
        signal_targets[pd.Timestamp(signal_date)] = indices[: costs.maximum_positions]
        signal_regimes[pd.Timestamp(signal_date)] = str(ordered["regime"].iloc[0])

    cash = float(initial_cash)
    units = np.zeros(len(symbols), dtype=float)
    acquired_day = np.full(len(symbols), -1, dtype=int)
    current_regime = "CASH"
    previous_date: pd.Timestamp | None = None
    ledger_rows: list[dict] = []
    trade_rows: list[dict] = []

    for day_number, date in enumerate(calendar):
        if day_number > 0 and costs.cash_annual_rate:
            cash *= 1.0 + costs.cash_annual_rate / 242.0
        open_prices = mark_open.loc[date].to_numpy(dtype=float)
        close_prices = total_close.loc[date].to_numpy(dtype=float)
        raw_prices = raw_open.loc[date].to_numpy(dtype=float)
        gaps = open_gap.loc[date].to_numpy(dtype=float)
        suspended_today = suspended.loc[date].to_numpy(dtype=bool)
        held = units > 1e-12
        if np.any(held & ~np.isfinite(open_prices)):
            missing = symbols[held & ~np.isfinite(open_prices)].tolist()
            raise ValueError(f"持仓证券无法开盘估值：{missing}")
        open_values = np.where(held, units * np.nan_to_num(open_prices), 0.0)

        day_buy = day_sell = day_commission = day_stamp = day_slippage = 0.0
        blocked_small_sell = blocked_buy = blocked_sell = 0
        signal_used: pd.Timestamp | None = None
        desired_indices: list[int] = []

        if previous_date in signal_targets:
            signal_used = previous_date
            desired_indices = signal_targets[previous_date]
            desired_set = set(desired_indices)
            current_regime = signal_regimes.get(previous_date, "UNKNOWN")
            stamp_rate = costs.stamp_duty_rate_for_date(date)

            for index in np.flatnonzero(held):
                if index in desired_set:
                    continue
                notional = float(open_values[index])
                can_sell = (
                    np.isfinite(observed_total_open.loc[date].iloc[index])
                    and np.isfinite(raw_prices[index])
                    and not suspended_today[index]
                    and np.isfinite(gaps[index])
                    and gaps[index] > -maximum_open_gap_for_trade
                    and acquired_day[index] < day_number
                )
                if not can_sell:
                    blocked_sell += 1
                    continue
                if notional + 1e-9 < costs.minimum_trade_notional_cny:
                    blocked_small_sell += 1
                    continue
                commission = _commission(notional, costs)
                stamp = notional * stamp_rate
                slippage = notional * costs.slippage_bps_per_leg / 10000.0
                cash += notional - commission - stamp - slippage
                units[index] = 0.0
                acquired_day[index] = -1
                day_sell += notional
                day_commission += commission
                day_stamp += stamp
                day_slippage += slippage
                trade_rows.append(
                    {
                        "date": date,
                        "signal_date": signal_used,
                        "con_code": str(symbols[index]),
                        "side": "卖出",
                        "raw_open": float(raw_prices[index]),
                        "estimated_quantity": float(notional / raw_prices[index]),
                        "notional": notional,
                        "commission": commission,
                        "stamp_duty": stamp,
                        "slippage": slippage,
                        "regime": current_regime,
                    }
                )

            held = units > 1e-12
            new_indices = [index for index in desired_indices if not held[index]]
            available_slots = max(costs.maximum_positions - int(held.sum()), 0)
            new_indices = new_indices[:available_slots]
            for position, index in enumerate(new_indices):
                remaining_slots = len(new_indices) - position
                observed_open = observed_total_open.loc[date].iloc[index]
                can_buy = (
                    np.isfinite(observed_open)
                    and np.isfinite(raw_prices[index])
                    and raw_prices[index] > 0
                    and not suspended_today[index]
                    and np.isfinite(gaps[index])
                    and gaps[index] < maximum_open_gap_for_trade
                )
                if not can_buy:
                    blocked_buy += 1
                    continue
                per_slot_cash = cash / max(remaining_slots, 1)
                lot_notional = float(raw_prices[index] * costs.lot_size)
                minimum_lots = int(np.ceil(costs.minimum_trade_notional_cny / lot_notional))
                affordable_lots = int(
                    np.floor(
                        max(per_slot_cash - costs.minimum_commission_cny, 0.0)
                        / (lot_notional * (1.0 + costs.slippage_bps_per_leg / 10000.0))
                    )
                )
                lots = max(minimum_lots, affordable_lots)
                notional = lots * lot_notional
                commission = _commission(notional, costs)
                slippage = notional * costs.slippage_bps_per_leg / 10000.0
                outlay = notional + commission + slippage
                if notional + 1e-9 < costs.minimum_trade_notional_cny or outlay > per_slot_cash + 1e-9:
                    blocked_buy += 1
                    continue
                cash -= outlay
                units[index] = notional / float(observed_open)
                acquired_day[index] = day_number
                day_buy += notional
                day_commission += commission
                day_slippage += slippage
                trade_rows.append(
                    {
                        "date": date,
                        "signal_date": signal_used,
                        "con_code": str(symbols[index]),
                        "side": "买入",
                        "raw_open": float(raw_prices[index]),
                        "estimated_quantity": float(lots * costs.lot_size),
                        "notional": notional,
                        "commission": commission,
                        "stamp_duty": 0.0,
                        "slippage": slippage,
                        "regime": current_regime,
                    }
                )

        held = units > 1e-12
        if np.any(held & ~np.isfinite(close_prices)):
            missing = symbols[held & ~np.isfinite(close_prices)].tolist()
            raise ValueError(f"持仓证券无法收盘估值：{missing}")
        close_values = np.where(held, units * np.nan_to_num(close_prices), 0.0)
        equity = float(cash + close_values.sum())
        weights = close_values / equity if equity > 0 else np.zeros(len(symbols))
        target_mask = np.zeros(len(symbols), dtype=float)
        if desired_indices:
            target_mask[desired_indices] = 1.0 / len(desired_indices)
        ledger_rows.append(
            {
                "date": date,
                "signal_date_used": signal_used,
                "regime": current_regime,
                "equity": equity,
                "cash": float(cash),
                "invested_value": float(close_values.sum()),
                "actual_exposure": float(close_values.sum() / equity) if equity > 0 else 0.0,
                "position_count": int(held.sum()),
                "buy_notional": day_buy,
                "sell_notional": day_sell,
                "commission": day_commission,
                "stamp_duty": day_stamp,
                "slippage": day_slippage,
                "blocked_small_sell_count": blocked_small_sell,
                "blocked_buy_count": blocked_buy,
                "blocked_sell_count": blocked_sell,
                "target_weights_json": _weights_json(symbols, target_mask),
                "actual_weights_json": _weights_json(symbols, weights),
            }
        )
        previous_date = date

    ledger = pd.DataFrame(ledger_rows)
    ledger["daily_return"] = ledger["equity"].pct_change().fillna(0.0)
    ledger["equity_peak"] = ledger["equity"].cummax()
    ledger["drawdown"] = ledger["equity"] / ledger["equity_peak"] - 1.0
    trades = pd.DataFrame(trade_rows)
    if not trades.empty and (trades["notional"] + 1e-8 < costs.minimum_trade_notional_cny).any():
        raise AssertionError("回测生成了低于5000元的交易")
    return ledger, trades

