"""点时成分股组合的下一开盘成交回测引擎。"""

from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CrossSectionalCosts:
    """股票组合的显式与隐式单边成本。"""

    commission_rate: float = 0.0003
    minimum_commission_cny: float = 5.0
    stamp_duty_sell_rate: float = 0.0005
    stamp_duty_sell_rate_before_reduction: float = 0.001
    stamp_duty_reduction_effective_date: str = "2023-08-28"
    slippage_bps_per_leg: float = 5.0
    cash_annual_rate: float = 0.015
    lot_size: int = 100

    def stamp_duty_rate_for_date(self, date: str | pd.Timestamp) -> float:
        """按财政部、税务总局2023年减半征收政策返回卖出印花税率。"""

        if pd.Timestamp(date) < pd.Timestamp(self.stamp_duty_reduction_effective_date):
            return self.stamp_duty_sell_rate_before_reduction
        return self.stamp_duty_sell_rate


def _commission(notional: np.ndarray, costs: CrossSectionalCosts) -> np.ndarray:
    result = np.zeros_like(notional, dtype=float)
    traded = notional > 1e-9
    result[traded] = np.maximum(
        costs.minimum_commission_cny,
        notional[traded] * costs.commission_rate,
    )
    return result


def _json_weights(symbols: np.ndarray, values: np.ndarray) -> str:
    payload = {
        str(symbol): round(float(value), 8)
        for symbol, value in zip(symbols, values, strict=True)
        if value > 1e-10
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _validate_inputs(panel: pd.DataFrame, targets: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required_panel = {
        "date",
        "con_code",
        "total_return_open",
        "total_return_close",
        "raw_open",
        "is_suspended",
    }
    required_targets = {"signal_date", "con_code", "target_weight"}
    if missing := required_panel.difference(panel.columns):
        raise ValueError(f"成分股面板缺少字段：{sorted(missing)}")
    if missing := required_targets.difference(targets.columns):
        raise ValueError(f"目标持仓缺少字段：{sorted(missing)}")

    market = panel.copy()
    signals = targets.copy()
    market["date"] = pd.to_datetime(market["date"])
    signals["signal_date"] = pd.to_datetime(signals["signal_date"])
    market = market.sort_values(["date", "con_code"]).reset_index(drop=True)
    signals = signals.sort_values(["signal_date", "con_code"]).reset_index(drop=True)
    if market[["date", "con_code"]].duplicated().any():
        raise ValueError("成分股面板存在重复证券日期")
    if signals[["signal_date", "con_code"]].duplicated().any():
        raise ValueError("目标持仓存在重复证券日期")
    signals["target_weight"] = pd.to_numeric(signals["target_weight"], errors="coerce")
    if signals["target_weight"].isna().any() or (signals["target_weight"] <= 0).any():
        raise ValueError("目标权重必须为正数")
    weight_sums = signals.groupby("signal_date")["target_weight"].sum()
    if (weight_sums > 1.0 + 1e-9).any():
        raise ValueError("同一信号日的目标权重合计不得超过100%")
    return market, signals


def run_weighted_open_backtest(
    panel: pd.DataFrame,
    targets: pd.DataFrame,
    initial_cash: float,
    costs: CrossSectionalCosts,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    maximum_open_gap_for_trade: float = 0.095,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """执行信号日收盘生成、下一交易日开盘成交的多股票回测。

    组合使用总收益价格记账，以纳入分红和送转；成交手数按未复权开盘价向下取整。
    停牌、近似涨停买入和近似跌停卖出均不假设成交，未投出的目标资金保留为现金。
    """

    market, signals = _validate_inputs(panel, targets)
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    market = market.loc[market["date"].between(start, end)].copy()
    if market.empty:
        raise ValueError("回测区间没有成分股行情")
    calendar = pd.DatetimeIndex(sorted(market["date"].unique()))
    if len(calendar) < 2:
        raise ValueError("回测区间至少需要两个交易日")

    symbols = np.array(sorted(market["con_code"].unique()), dtype=object)
    symbol_to_index = {str(symbol): index for index, symbol in enumerate(symbols)}

    def wide(column: str) -> pd.DataFrame:
        return market.pivot(index="date", columns="con_code", values=column).reindex(
            index=calendar, columns=symbols
        )

    total_open = wide("total_return_open").astype(float)
    total_close = wide("total_return_close").astype(float)
    raw_open = wide("raw_open").astype(float)
    suspended = wide("is_suspended").fillna(True).astype(bool)
    previous_total_close = total_close.shift(1)
    open_gap = total_open.divide(previous_total_close).subtract(1.0)

    signal_weights: dict[pd.Timestamp, np.ndarray] = {}
    signal_regimes: dict[pd.Timestamp, str] = {}
    for signal_date, frame in signals.groupby("signal_date", sort=True):
        weights = np.zeros(len(symbols), dtype=float)
        for row in frame.itertuples(index=False):
            index = symbol_to_index.get(str(row.con_code))
            if index is not None:
                weights[index] = float(row.target_weight)
        signal_weights[pd.Timestamp(signal_date)] = weights
        signal_regimes[pd.Timestamp(signal_date)] = (
            str(frame["regime"].iloc[0]) if "regime" in frame.columns else "UNKNOWN"
        )

    cash = float(initial_cash)
    units = np.zeros(len(symbols), dtype=float)
    acquired_ordinal = np.full(len(symbols), -1, dtype=int)
    current_target = np.zeros(len(symbols), dtype=float)
    current_regime = "CASH"
    previous_date: pd.Timestamp | None = None
    ledger_rows: list[dict] = []
    trade_rows: list[dict] = []

    for day_number, date in enumerate(calendar):
        if day_number > 0 and costs.cash_annual_rate:
            cash *= 1.0 + costs.cash_annual_rate / 242.0

        open_prices = total_open.loc[date].to_numpy(dtype=float)
        close_prices = total_close.loc[date].to_numpy(dtype=float)
        raw_open_prices = raw_open.loc[date].to_numpy(dtype=float)
        gaps = open_gap.loc[date].to_numpy(dtype=float)
        suspended_today = suspended.loc[date].to_numpy(dtype=bool)
        held = units > 1e-12
        if np.any(held & ~np.isfinite(open_prices)):
            missing_symbols = symbols[held & ~np.isfinite(open_prices)].tolist()
            raise ValueError(f"持仓证券缺少开盘总收益价格：{missing_symbols}")

        current_values = np.where(held, units * np.nan_to_num(open_prices, nan=0.0), 0.0)
        equity_before_trade = float(cash + current_values.sum())
        buy_notional = 0.0
        sell_notional = 0.0
        commission_total = 0.0
        stamp_total = 0.0
        slippage_total = 0.0
        blocked_buy_count = 0
        blocked_sell_count = 0
        signal_date_used: pd.Timestamp | None = None

        if previous_date in signal_weights:
            signal_date_used = previous_date
            current_target = signal_weights[previous_date].copy()
            current_regime = signal_regimes.get(previous_date, "UNKNOWN")
            stamp_duty_rate = costs.stamp_duty_rate_for_date(date)
            finite_trade_prices = np.isfinite(open_prices) & np.isfinite(raw_open_prices)
            buy_allowed = (
                finite_trade_prices
                & ~suspended_today
                & (np.nan_to_num(gaps, nan=np.inf) < maximum_open_gap_for_trade)
            )
            sell_allowed = (
                finite_trade_prices
                & ~suspended_today
                & (np.nan_to_num(gaps, nan=-np.inf) > -maximum_open_gap_for_trade)
                & (acquired_ordinal < day_number)
            )
            locked = held & ~sell_allowed
            blocked_sell_count = int(np.sum(locked & (current_target <= 1e-12)))
            blocked_buy_count = int(np.sum((current_target > 1e-12) & ~buy_allowed & ~held))
            adjustable_target = (current_target > 1e-12) & (buy_allowed | held) & ~locked
            locked_values = np.where(locked, current_values, 0.0)

            post_cost_equity = equity_before_trade
            desired_values = current_values.copy()
            total_cost = 0.0
            for _ in range(6):
                desired_values = locked_values.copy()
                available = max(post_cost_equity - float(locked_values.sum()), 0.0)
                desired_nominal = current_target * post_cost_equity
                lot_notional = raw_open_prices * costs.lot_size
                rounded = np.zeros(len(symbols), dtype=float)
                valid_lot = adjustable_target & np.isfinite(lot_notional) & (lot_notional > 0)
                rounded[valid_lot] = (
                    np.floor(desired_nominal[valid_lot] / lot_notional[valid_lot])
                    * lot_notional[valid_lot]
                )
                cannot_increase = adjustable_target & held & ~buy_allowed
                rounded[cannot_increase] = np.minimum(
                    rounded[cannot_increase], current_values[cannot_increase]
                )
                rounded_total = float(rounded.sum())
                if rounded_total > available + 1e-9 and rounded_total > 0:
                    scale = available / rounded_total
                    rounded[valid_lot] = (
                        np.floor((rounded[valid_lot] * scale) / lot_notional[valid_lot])
                        * lot_notional[valid_lot]
                    )
                desired_values[adjustable_target] = rounded[adjustable_target]
                tradable_current = held & ~locked
                desired_values[tradable_current & ~adjustable_target] = 0.0
                delta = desired_values - current_values
                buys = np.clip(delta, 0.0, None)
                sells = np.clip(-delta, 0.0, None)
                commissions = _commission(buys, costs) + _commission(sells, costs)
                stamps = sells * stamp_duty_rate
                slippage = (buys + sells) * costs.slippage_bps_per_leg / 10000.0
                total_cost = float(commissions.sum() + stamps.sum() + slippage.sum())
                post_cost_equity = max(equity_before_trade - total_cost, 0.0)

            delta = desired_values - current_values
            buys = np.clip(delta, 0.0, None)
            sells = np.clip(-delta, 0.0, None)
            buy_commissions = _commission(buys, costs)
            sell_commissions = _commission(sells, costs)
            commissions = buy_commissions + sell_commissions
            stamps = sells * stamp_duty_rate
            slippage = (buys + sells) * costs.slippage_bps_per_leg / 10000.0
            buy_notional = float(buys.sum())
            sell_notional = float(sells.sum())
            commission_total = float(commissions.sum())
            stamp_total = float(stamps.sum())
            slippage_total = float(slippage.sum())
            post_cost_equity = max(
                equity_before_trade - commission_total - stamp_total - slippage_total,
                0.0,
            )

            old_units = units.copy()
            units = np.divide(
                desired_values,
                open_prices,
                out=np.zeros_like(desired_values),
                where=np.isfinite(open_prices) & (open_prices > 0),
            )
            increased = units > old_units + 1e-12
            acquired_ordinal[increased] = day_number
            acquired_ordinal[units <= 1e-12] = -1
            cash = float(post_cost_equity - desired_values.sum())
            if cash < -1e-6:
                raise RuntimeError(f"调仓后现金为负：{date.date()}={cash:.2f}")
            cash = max(cash, 0.0)

            traded = (buys + sells) > 1e-9
            for index in np.flatnonzero(traded):
                side = "买入" if buys[index] > 0 else "卖出"
                notional = float(buys[index] if buys[index] > 0 else sells[index])
                trade_rows.append(
                    {
                        "date": date,
                        "signal_date": signal_date_used,
                        "con_code": str(symbols[index]),
                        "side": side,
                        "raw_open": float(raw_open_prices[index]),
                        "estimated_quantity": float(notional / raw_open_prices[index]),
                        "notional": notional,
                        "commission": float(commissions[index]),
                        "stamp_duty": float(stamps[index]),
                        "slippage": float(slippage[index]),
                        "target_weight": float(current_target[index]),
                        "regime": current_regime,
                    }
                )

        if np.any((units > 1e-12) & ~np.isfinite(close_prices)):
            missing_symbols = symbols[(units > 1e-12) & ~np.isfinite(close_prices)].tolist()
            raise ValueError(f"持仓证券缺少收盘总收益价格：{missing_symbols}")
        close_values = np.where(
            units > 1e-12,
            units * np.nan_to_num(close_prices, nan=0.0),
            0.0,
        )
        equity = float(cash + close_values.sum())
        actual_weights = close_values / equity if equity > 0 else np.zeros(len(symbols))
        ledger_rows.append(
            {
                "date": date,
                "signal_date_used": signal_date_used,
                "regime": current_regime,
                "equity": equity,
                "cash": float(cash),
                "invested_value": float(close_values.sum()),
                "actual_exposure": float(close_values.sum() / equity) if equity > 0 else 0.0,
                "position_count": int(np.sum(units > 1e-12)),
                "buy_notional": buy_notional,
                "sell_notional": sell_notional,
                "commission": commission_total,
                "stamp_duty": stamp_total,
                "slippage": slippage_total,
                "blocked_buy_count": blocked_buy_count,
                "blocked_sell_count": blocked_sell_count,
                "target_weights_json": _json_weights(symbols, current_target),
                "actual_weights_json": _json_weights(symbols, actual_weights),
            }
        )
        previous_date = date

    ledger = pd.DataFrame(ledger_rows)
    ledger["daily_return"] = ledger["equity"].pct_change().fillna(0.0)
    ledger["equity_peak"] = ledger["equity"].cummax()
    ledger["drawdown"] = ledger["equity"] / ledger["equity_peak"] - 1.0
    trades = pd.DataFrame(trade_rows)
    return ledger, trades
