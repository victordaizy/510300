"""510300固定连续评分，以及复用既有账户的逐日政策模拟。"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from research.intraday_overnight_increment_v1 import (
    Account, affordable_quantity, execute_order, fill_price, require, return_metrics,
)

FULL_COLUMNS = ["T", "Q", "B", "D", "R"]
SIMPLE_COLUMNS = ["T", "Q", "V_PLUS"]


def features(prices: pd.DataFrame, dividends: pd.DataFrame, floor: float = 0.003) -> pd.DataFrame:
    """每一行仅用当日及过去；保留缺失行，滚动窗口不缩短。"""
    data = prices.sort_values("date").reset_index(drop=True).copy()
    distribution = dividends.set_index("ex_date").cash_dividend_per_share
    data["dividend"] = data.date.map(distribution).fillna(0.0)
    required = ["open", "high", "low", "close", "volume", "amount"]
    values = data[required].to_numpy(float)
    valid = np.isfinite(values).all(axis=1) & (values > 0).all(axis=1)
    valid &= (data.high >= data[["open", "low", "close"]].max(axis=1)).to_numpy()
    valid &= (data.low <= data[["open", "high", "close"]].min(axis=1)).to_numpy()
    data["valid_market"] = valid
    require(not ((data.dividend > 0) & (~np.isfinite(data.close) | (data.close <= 0))).any(),
            "除息日缺少可信收盘价，不能构造信号财富序列")
    adjustment = pd.Series(np.where(data.dividend > 0, 1 + data.dividend / data.close, 1.0)).cumprod()
    data["signal_price"] = data.close * adjustment
    data["r"] = np.log(data.signal_price / data.signal_price.shift(1)).where(valid)
    for h in (5, 60):
        data[f"s{h}"] = np.sqrt(data.r.pow(2).rolling(h, min_periods=h).mean()).clip(lower=floor)
    for h in (5, 20, 60):
        data[f"u{h}"] = np.tanh(data.r.rolling(h, min_periods=h).sum() / (math.sqrt(h) * data.s60))
    spread = data.high - data.low
    data["q"] = ((2 * data.close - data.high - data.low) / spread.where(spread != 0)).where(spread != 0, 0.0).where(valid)
    amount = data.amount.where(valid)
    data["T"] = (data.u20 + data.u60) / 2
    data["Q"] = (amount * data.q).rolling(5, min_periods=5).sum() / amount.rolling(5, min_periods=5).sum()
    data["B"] = (1 + data.u60) / 2 * (-data.u5).clip(lower=0)
    data["v"] = (np.log(data.s5 / data.s60) / math.log(2)).clip(-1, 1)
    data["V_PLUS"] = data.v.clip(lower=0)
    data["D"] = 0.5 * (-data.u20).clip(lower=0) + 0.5 * (1 - data.u5) / 2 * data.V_PLUS
    high = data.signal_price.where(valid).rolling(60, min_periods=60).max()
    data["G"] = (-np.log(data.signal_price / high) / (2 * data.s60 * math.sqrt(20))).clip(0, 1)
    data["R"] = data.G * (data.u5.clip(lower=0) + data.Q.clip(lower=0) + (data.v.shift(5) - data.v).clip(0, 1)) / 3
    data["feature_valid"] = np.isfinite(data[FULL_COLUMNS + SIMPLE_COLUMNS].to_numpy(float)).all(axis=1) & valid
    data["previous_close"] = data.close.shift(1)
    return data


def design(data: pd.DataFrame, model: str) -> np.ndarray:
    if model in ("FULL", "NO_REPAIR"):
        matrix = data[FULL_COLUMNS].to_numpy(float).copy()
        matrix[:, 3] *= -1
    elif model == "SIMPLE":
        matrix = data[SIMPLE_COLUMNS].to_numpy(float).copy()
        matrix[:, 2] *= -1
    else:
        raise ValueError(f"未知模型：{model}")
    return np.column_stack([np.ones(len(data)), matrix])


def score(data: pd.DataFrame, coefficients: np.ndarray, model: str) -> np.ndarray:
    coefficients = np.asarray(coefficients, dtype=float).copy()
    if model == "NO_REPAIR":
        coefficients[5] = 0
    result = 100 / (1 + np.exp(-(design(data, model) @ coefficients.T)))
    result[~data.feature_valid.to_numpy(bool)] = np.nan
    return result


def target_weight(scores: np.ndarray | float) -> np.ndarray:
    return np.clip((np.asarray(scores) - 20.0) / 60.0, 0.0, 1.0)


def choose_request(account: Account, close: float, value: float, config: dict) -> dict:
    equity = account.value(close)
    actual = account.shares * close / equity
    if not np.isfinite(value):
        return {"score": None, "target_weight": None, "adjusted_weight": None,
                "requested_quantity": None, "reason": "NO_VIEW_保持原份额及已冻结余单", "actual_weight": actual}
    target = float(target_weight(value))
    if value <= 20:
        quantity, adjusted, reason = -account.shares, 0.0, "退出_不受不交易带限制"
    elif abs(target - actual) < config["no_trade_band"] - 1e-12:
        quantity, adjusted, reason = 0, actual, "不交易带内"
    else:
        adjusted = min(target, actual + config["maximum_daily_addition"]) if target > actual else target
        target_shares = math.floor(adjusted * equity / close / config["lot"] + 1e-12) * config["lot"]
        quantity = int(target_shares - account.shares)
        reason = "分步加仓" if quantity > 0 else "直接减仓" if quantity < 0 else "不足整手"
    return {"score": float(value), "target_weight": target, "adjusted_weight": adjusted,
            "requested_quantity": quantity, "reason": reason, "actual_weight": actual}


@dataclass
class Period:
    """首行是决策锚，后续每行都是不可删除的账户估值日。"""
    data: pd.DataFrame
    dividends: pd.DataFrame
    record_keys: list[list[int]]
    ex_keys: list[list[int]]
    pay_open_keys: list[list[int]]
    pay_close_keys: list[list[int]]

    @property
    def n(self) -> int:
        return len(self.data) - 1


def period(data: pd.DataFrame, dividends: pd.DataFrame, start: str, end: str) -> Period:
    dates = pd.DatetimeIndex(data.date)
    selected = np.flatnonzero((dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end)))
    require(len(selected) > 1 and selected[0] > 0, "评价期间或前一收盘锚不足")
    selected_data = data.iloc[selected[0] - 1:selected[-1] + 1].reset_index(drop=True)
    require(np.isfinite(selected_data.close).all() and (selected_data.close > 0).all(),
            "缺少可信每日收盘估值，禁止删除日期或填造零收益")
    n = len(selected_data)
    records, exes, pay_open, pay_close = ([[ ] for _ in range(n)] for _ in range(4))
    selected_dates = pd.DatetimeIndex(selected_data.date)
    for k, event in enumerate(dividends.itertuples()):
        for field, schedule in (("record_date", records), ("ex_date", exes)):
            date = getattr(event, field)
            if date in selected_dates:
                schedule[selected_dates.get_loc(date)].append(k)
        position = selected_dates.searchsorted(event.payment_date)
        if 0 < position < n:
            (pay_close if selected_dates[position] == event.payment_date else pay_open)[position].append(k)
    return Period(selected_data, dividends, records, exes, pay_open, pay_close)


def simulate_reference(p: Period, scores: np.ndarray, cost: dict, config: dict,
                       model: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """交易直接调用既有账户引擎；新增的只有评分决策与逐日调度。"""
    account = Account(config["initial_capital"])
    prices = p.data
    amounts = p.dividends.cash_dividend_per_share.to_numpy(float)
    rows, trades, decisions = [], [], []
    anchor = prices.iloc[0]
    if model == "BUY_HOLD":
        px = fill_price(float(anchor.close), 1, cost, config["tick"])
        request = affordable_quantity(account.cash, px, cost, config["lot"])
        decision = {"score": None, "target_weight": 1.0, "adjusted_weight": 1.0,
                    "requested_quantity": request, "reason": "买入持有初始请求", "actual_weight": 0.0}
    else:
        decision = choose_request(account, float(anchor.close), float(scores[0]), config)
        request = int(decision["requested_quantity"] or 0)
    pending_origin = anchor.date
    if model != "BUY_HOLD":
        decision.update({name: float(anchor[name]) for name in FULL_COLUMNS + ["v", "G", "u5", "u20", "u60"]})
    decisions.append({"origin": anchor.date, "execution_date": prices.date.iloc[1],
                      "simulation_only": True, **decision})
    previous_nav, previous_mark = config["initial_capital"], float(anchor.close)
    for day in range(1, len(prices)):
        row = prices.iloc[day]
        old_shares = account.shares
        recognized, paid = 0.0, 0.0
        for key in p.ex_keys[day]:
            value = account.entitlements.get(key, 0) * amounts[key]
            account.receivables[key] = value
            recognized += value
        for key in p.pay_open_keys[day]:
            value = account.receivables.pop(key, 0.0)
            paid += value
            account.cash += value
        open_valid = bool(np.isfinite(row.open) and row.open > 0 and np.isfinite(row.volume) and row.volume > 0)
        pretrade_nav = account.value(float(row.open)) if open_valid else account.value(float(row.close))
        request_origin = pending_origin
        if open_valid:
            execution = execute_order(account, request, float(row.open), float(row.previous_close),
                                      float(row.dividend), day, cost, config)
        else:
            execution = {"requested_quantity": request, "filled_quantity": 0, "fill_price": None,
                         "commission": 0.0, "slippage_cost": 0.0, "notional": 0.0,
                         "status": "UNFILLED_开盘输入不可信", "open_price": None}
        filled = int(execution["filled_quantity"])
        remainder = request - filled
        if model == "BUY_HOLD" and filled and filled != request:
            remainder = 0
        if request:
            trades.append({"date": row.date, "origin": request_origin, "pre_trade_nav": pretrade_nav, **execution})
        cash_after_execution = account.cash
        for key in p.pay_close_keys[day]:
            value = account.receivables.pop(key, 0.0)
            account.cash += value
            paid += value
        for key in p.record_keys[day]:
            account.entitlements[key] = account.shares
        nav = account.value(float(row.close))
        if open_valid:
            price_pnl = old_shares * (float(row.open) - previous_mark) + account.shares * (float(row.close) - float(row.open))
        else:
            price_pnl = old_shares * (float(row.close) - previous_mark)
        error = nav - previous_nav - price_pnl - recognized + execution["commission"] + execution["slippage_cost"]
        require(abs(error) < 1e-6, f"账户财富不守恒：{row.date}，差额{error}")
        account.assert_valid()
        require(account.shares % config["lot"] == 0, "期末份额不是整手")
        exposure = account.shares * float(row.close) / nav
        rows.append({"date": row.date, "open": row.open, "close": row.close,
                     "equity": nav, "cash": account.cash, "cash_after_execution": cash_after_execution, "shares": account.shares,
                     "dividend_receivable": account.receivable(), "dividend_recognized": recognized,
                     "dividend_paid": paid, "net_return": nav / previous_nav - 1,
                     "pnl": nav - previous_nav, "price_pnl": price_pnl, "exposure": exposure,
                     "commission": execution["commission"], "slippage_cost": execution["slippage_cost"],
                     "traded_notional": execution["notional"], "turnover": execution["notional"] / pretrade_nav,
                     "requested_quantity": request, "filled_quantity": filled, "request_origin": request_origin,
                     "execution_status": execution["status"], "accounting_error": error,
                     "feature_valid": row.feature_valid, "mark_clock": "CLOSE"})
        previous_nav, previous_mark = nav, float(row.close)
        if model == "BUY_HOLD":
            request = remainder
        else:
            decision = choose_request(account, float(row.close), float(scores[day]), config)
            if decision["requested_quantity"] is None:
                request = remainder
            else:
                request = int(decision["requested_quantity"])
                pending_origin = row.date
            values = {name: float(row[name]) for name in FULL_COLUMNS + ["v", "G", "u5", "u20", "u60"]}
            decisions.append({"origin": row.date,
                              "execution_date": prices.date.iloc[day + 1] if day + 1 < len(prices) else pd.NaT,
                              "simulation_only": True, "pending_quantity_after_close": request,
                              "pending_original_origin": pending_origin,
                              "next_execution_within_window": day + 1 < len(prices), **values, **decision})
    return pd.DataFrame(rows), pd.DataFrame(trades), pd.DataFrame(decisions)


def batch_simulate(p: Period, scores: np.ndarray, config: dict, *, details: bool = False) -> dict:
    """批量训练加速器；两种成本逐日独立现金/股数，原始评分完全共享。"""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim == 1:
        scores = scores[:, None]
    members = scores.shape[1]
    size = members * 2
    price = p.data
    opens, closes = price.open.to_numpy(float), price.close.to_numpy(float)
    volumes, distributions = price.volume.to_numpy(float), price.dividend.to_numpy(float)
    previous_closes = price.previous_close.to_numpy(float)
    cash = np.full(size, config["initial_capital"], dtype=float)
    shares = np.zeros(size, dtype=np.int64)
    receivables = np.zeros((size, len(p.dividends)), dtype=float)
    entitlements = np.zeros((size, len(p.dividends)), dtype=np.int64)
    amounts = p.dividends.cash_dividend_per_share.to_numpy(float)
    rates = np.repeat([config["costs"][name]["commission"] for name in ("BASE", "STRESS")], members)
    slips = np.repeat([config["costs"][name]["slippage"] for name in ("BASE", "STRESS")], members)
    minimums = np.repeat([config["costs"][name]["minimum"] for name in ("BASE", "STRESS")], members)
    lot, tick = config["lot"], config["tick"]
    previous_nav = cash.copy()
    sum_r, sum_r2, turnover, sum_exposure, exposed_days = (np.zeros(size) for _ in range(5))
    total_fee, total_slip = np.zeros(size), np.zeros(size)
    path = np.empty((p.n, size, 5)) if details else None
    pending = np.zeros(size, dtype=np.int64)
    for day in range(len(price)):
        if day:
            for key in p.ex_keys[day]:
                receivables[:, key] = entitlements[:, key] * amounts[key]
            for key in p.pay_open_keys[day]:
                cash += receivables[:, key]
                receivables[:, key] = 0
            open_valid = np.isfinite(opens[day]) and opens[day] > 0 and np.isfinite(volumes[day]) and volumes[day] > 0
            fee, slip_cost, notional = np.zeros(size), np.zeros(size), np.zeros(size)
            filled = np.zeros(size, dtype=np.int64)
            if open_valid:
                nav_open = cash + shares * opens[day] + receivables.sum(axis=1)
                buy_price = np.ceil(opens[day] * (1 + slips) / tick - 1e-10) * tick
                sell_price = np.floor(opens[day] * (1 - slips) / tick + 1e-10) * tick
                buy_price = np.where(slips == 0, opens[day], buy_price)
                sell_price = np.where(slips == 0, opens[day], sell_price)
                basis = previous_closes[day] - distributions[day]
                upper = math.floor(basis * (1 + config["limit_fraction"]) / tick + 0.5 + 1e-9) * tick
                lower = math.floor(basis * (1 - config["limit_fraction"]) / tick + 0.5 + 1e-9) * tick
                # 与既有affordable_quantity相同，先估整手再扣佣金缩量。
                affordable = np.floor(np.maximum(cash + 1e-9, 0) / (buy_price * lot)).astype(np.int64) * lot
                too_much = affordable * buy_price + np.where(affordable > 0, np.maximum(affordable * buy_price * rates, minimums), 0) > cash + 1e-8
                while np.any(too_much & (affordable > 0)):
                    affordable = np.where(too_much, np.maximum(affordable - lot, 0), affordable)
                    too_much = affordable * buy_price + np.where(affordable > 0, np.maximum(affordable * buy_price * rates, minimums), 0) > cash + 1e-8
                buy_ok = (pending > 0) & (opens[day] < upper - 1e-9) & (buy_price <= upper + 1e-9)
                sell_ok = (pending < 0) & (opens[day] > lower + 1e-9) & (sell_price >= lower - 1e-9)
                # 每天仅在开盘执行一次，全部旧份额已度过T+1；当天新买不再卖。
                filled = np.where(buy_ok, np.minimum(pending, affordable), np.where(sell_ok, -np.minimum(-pending, shares), 0))
                execution_price = np.where(filled > 0, buy_price, sell_price)
                notional = np.abs(filled) * execution_price
                fee = np.where(filled != 0, np.maximum(notional * rates, minimums), 0)
                slip_cost = np.abs(filled) * np.abs(execution_price - opens[day])
                cash -= filled * execution_price + fee
                shares += filled
                turnover += notional / nav_open
            pending -= filled
            for key in p.pay_close_keys[day]:
                cash += receivables[:, key]
                receivables[:, key] = 0
            for key in p.record_keys[day]:
                entitlements[:, key] = shares
            nav = cash + shares * closes[day] + receivables.sum(axis=1)
            returns = nav / previous_nav - 1
            exposure = shares * closes[day] / nav
            sum_r += returns
            sum_r2 += returns * returns
            sum_exposure += exposure
            exposed_days += shares > 0
            total_fee += fee
            total_slip += slip_cost
            if details:
                path[day - 1, :, :] = np.column_stack([nav, cash, shares, filled, returns])
            previous_nav = nav
        nav_close = cash + shares * closes[day] + receivables.sum(axis=1)
        actual = shares * closes[day] / nav_close
        value = np.tile(scores[day], 2)
        valid = np.isfinite(value)
        target = target_weight(np.where(valid, value, 50.0))
        no_trade = np.abs(target - actual) < config["no_trade_band"] - 1e-12
        adjusted = np.where(target > actual, np.minimum(target, actual + config["maximum_daily_addition"]), target)
        desired = np.floor(adjusted * nav_close / closes[day] / lot + 1e-12).astype(np.int64) * lot
        new_request = np.where(value <= 20, -shares, np.where(no_trade, 0, desired - shares))
        pending = np.where(valid, new_request, pending)
    n, annual = p.n, config["annual_days"]
    variance = np.maximum((sum_r2 - sum_r * sum_r / n) / (n - 1), 0) * annual
    result = {"mu": sum_r / n * annual, "variance": variance,
              "turnover": turnover / n * annual, "mean_exposure": sum_exposure / n,
              "exposed_days": exposed_days, "equity": previous_nav,
              "commission": total_fee, "slippage_cost": total_slip}
    for key, values in result.items():
        result[key] = values.reshape(2, members)
    if details:
        result["path"] = path
    require((cash >= -1e-7).all() and (shares >= 0).all(), "批量账户出现透支或卖空")
    return result


def metrics(ledger: pd.DataFrame, config: dict) -> dict[str, Any]:
    result = return_metrics(ledger.net_return.to_numpy(float), config["annual_days"])
    result.update({"days": len(ledger), "ending_equity": float(ledger.equity.iloc[-1]),
                   "ending_cash": float(ledger.cash.iloc[-1]), "ending_shares": int(ledger.shares.iloc[-1]),
                   "mean_exposure": float(ledger.exposure.mean()),
                   "annualized_two_way_turnover": float(ledger.turnover.mean() * config["annual_days"]),
                   "commission": float(ledger.commission.sum()), "slippage_cost": float(ledger.slippage_cost.sum()),
                   "total_friction": float(ledger.commission.sum() + ledger.slippage_cost.sum()),
                   "trade_days": int((ledger.filled_quantity != 0).sum()),
                   "exposed_days": int((ledger.shares > 0).sum()),
                   "dividend_recognized": float(ledger.dividend_recognized.sum()),
                   "maximum_accounting_error": float(ledger.accounting_error.abs().max()),
                   "no_view_days": int((~ledger.feature_valid).sum())})
    result["sharpe_eligible"] = bool(result["mean_exposure"] >= config["minimum_mean_exposure_for_sharpe"] and
                                     result["exposed_days"] >= config["minimum_exposed_days_for_sharpe"] and
                                     result["annualized_volatility"] > 1e-8)
    if not result["sharpe_eligible"]:
        result["net_sharpe"] = None
    result["utility_gamma5"] = result["annualized_arithmetic_mean"] - 2.5 * result["annualized_volatility"] ** 2
    return result


def assert_batch_parity(p: Period, scores: np.ndarray, config: dict, model: str) -> dict:
    batch = batch_simulate(p, scores, config, details=True)
    differences = {}
    for cost_index, name in enumerate(("BASE", "STRESS")):
        ledger, _, _ = simulate_reference(p, scores, config["costs"][name], config, model)
        expected = ledger[["equity", "cash", "shares", "filled_quantity", "net_return"]].to_numpy(float)
        actual = batch["path"][:, cost_index, :]
        error = np.max(np.abs(expected - actual), axis=0)
        require(error[2] == 0 and error[3] == 0 and error[0] < 1e-6 and error[1] < 1e-6 and error[4] < 1e-12,
                f"训练/既有账户路径不一致：{model}/{name} {error}")
        differences[name] = dict(zip(["equity", "cash", "shares", "filled_quantity", "net_return"], error.tolist()))
    return differences
