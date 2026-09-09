"""固定截距归因：四条新增历史账户与四条事后理想统计路径，不含训练。"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from research.conditional_score_policy_v1 import Period, choose_request, target_weight
from research.intraday_overnight_increment_v1 import (
    Account, commission, execute_order, fill_price, require, return_metrics,
)


def analytic_bounds(coefficients: list[float]) -> dict:
    b0, bt, bq, bb, bd, br = map(float, coefficients)
    require(min(bt, bq, bb, bd, br) >= 0, "原系数不满足非负斜率约束")
    lower, upper = b0 - bt - bq - bd, b0 + bt + bq + bb + br
    scores = 100 / (1 + np.exp(-np.array([lower, upper])))
    weights = target_weight(scores)
    s0 = float(100 / (1 + math.exp(-b0)))
    return {"intercept": b0, "constant_score": s0, "constant_target": float(target_weight(s0)),
            "z_bounds": [lower, upper], "score_bounds": scores.tolist(),
            "target_bounds": weights.tolist(), "exit_z_threshold": math.log(.2 / .8),
            "score20_exit_reachable": bool(lower <= math.log(.2 / .8)),
            "addition25_cap_can_bind": bool(weights[1] > .25),
            "universal_no_trade_exposure_open_interval": [float(weights[1] - .1), float(weights[0] + .1)],
            "interpretation": "合法特征矩形包络；不宣称端点能由真实市场同时达到；目标界不约束价格漂移后的实际暴露"}


def opening_inputs_valid(row: pd.Series) -> bool:
    """只接触开盘时可知字段；交易日资格由上游日历合同保证。"""
    return bool(np.isfinite([row.open, row.previous_close, row.dividend]).all()
                and row.open > 0 and row.previous_close - row.dividend > 0)


def simulate_control(p: Period, constant_score: float, model: str,
                     cost: dict, config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    require(model in {"C0", "C1"}, "新增账户仅允许C0/C1")
    require(np.isfinite(constant_score), "常数评分非法")
    account = Account(config["initial_capital"])
    amounts = p.dividends.cash_dividend_per_share.to_numpy(float)
    anchor = p.data.iloc[0]
    w0 = float(target_weight(constant_score))
    if model == "C0":
        quantity = math.floor(w0 * account.cash / anchor.close / config["lot"] + 1e-12) * config["lot"]
        decision = {"score": constant_score, "target_weight": w0, "adjusted_weight": w0,
                    "requested_quantity": int(quantity), "actual_weight": 0.0, "reason": "静态底仓首次建仓"}
    else:
        decision = choose_request(account, float(anchor.close), constant_score if anchor.feature_valid else np.nan, config)
    request = int(decision["requested_quantity"] or 0)
    origin = anchor.date
    decisions = [{"origin": anchor.date, "execution_date": p.data.date.iloc[1],
                  "simulation_only": True, **decision}]
    rows, trades = [], []
    previous_nav, previous_mark = config["initial_capital"], float(anchor.close)
    for day in range(1, len(p.data)):
        row = p.data.iloc[day]
        old_shares = account.shares
        recognized = paid = 0.0
        for key in p.ex_keys[day]:
            value = account.entitlements.get(key, 0) * amounts[key]
            account.receivables[key] = value
            recognized += value
        for key in p.pay_open_keys[day]:
            value = account.receivables.pop(key, 0.0)
            account.cash += value
            paid += value
        valid_open = opening_inputs_valid(row)
        pretrade_nav = account.value(float(row.open)) if valid_open else account.value(float(row.close))
        if valid_open:
            execution = execute_order(account, request, float(row.open), float(row.previous_close),
                                      float(row.dividend), day, cost, config)
        else:
            execution = {"requested_quantity": request, "filled_quantity": 0, "fill_price": None,
                         "commission": 0.0, "slippage_cost": 0.0, "notional": 0.0,
                         "status": "UNFILLED_开盘输入不可信", "open_price": None}
        filled = int(execution["filled_quantity"])
        remainder = 0 if model == "C0" and filled > 0 else request - filled
        if request:
            trades.append({"date": row.date, "origin": origin, "pre_trade_nav": pretrade_nav, **execution})
        cash_after_execution = account.cash
        for key in p.pay_close_keys[day]:
            value = account.receivables.pop(key, 0.0)
            account.cash += value
            paid += value
        for key in p.record_keys[day]:
            account.entitlements[key] = account.shares
        nav = account.value(float(row.close))
        if valid_open:
            price_pnl = old_shares * (float(row.open) - previous_mark) + account.shares * (float(row.close) - float(row.open))
        else:
            price_pnl = old_shares * (float(row.close) - previous_mark)
        error = nav - previous_nav - price_pnl - recognized + execution["commission"] + execution["slippage_cost"]
        require(abs(error) < 1e-6, f"新增账户财富不守恒：{row.date}，差额{error}")
        account.assert_valid()
        require(account.shares % config["lot"] == 0, "期末份额不符合整手")
        rows.append({"date": row.date, "open": row.open, "close": row.close,
                     "equity": nav, "cash": account.cash, "cash_after_execution": cash_after_execution,
                     "shares": account.shares, "dividend_receivable": account.receivable(),
                     "dividend_recognized": recognized, "dividend_paid": paid,
                     "net_return": nav / previous_nav - 1, "pnl": nav - previous_nav,
                     "price_pnl": price_pnl, "exposure": account.shares * float(row.close) / nav,
                     "commission": execution["commission"], "slippage_cost": execution["slippage_cost"],
                     "traded_notional": execution["notional"], "turnover": execution["notional"] / pretrade_nav,
                     "requested_quantity": request, "filled_quantity": filled, "request_origin": origin,
                     "execution_status": execution["status"], "accounting_error": error,
                     "feature_valid": bool(row.feature_valid), "opening_inputs_valid": valid_open,
                     "mark_clock": "CLOSE", "model": model, "simulation_only": True})
        previous_nav, previous_mark = nav, float(row.close)
        if model == "C0":
            request = remainder
        else:
            decision = choose_request(account, float(row.close), constant_score if row.feature_valid else np.nan, config)
            if decision["requested_quantity"] is None:
                request = remainder
            else:
                request, origin = int(decision["requested_quantity"]), row.date
            decisions.append({"origin": row.date,
                              "execution_date": p.data.date.iloc[day + 1] if day + 1 < len(p.data) else pd.NaT,
                              "next_execution_within_window": day + 1 < len(p.data),
                              "pending_quantity_after_close": request, "pending_original_origin": origin,
                              "simulation_only": True, **decision})
    return pd.DataFrame(rows), pd.DataFrame(trades), pd.DataFrame(decisions)


def audit_ledger(p: Period, ledger: pd.DataFrame, trades: pd.DataFrame, cost: dict, config: dict) -> dict:
    """用保存成交独立核对会计；不运行评分、订单选择或账户模拟器。"""
    require(pd.DatetimeIndex(pd.to_datetime(ledger.date)).equals(pd.DatetimeIndex(p.data.date.iloc[1:])), "账本与完整日历不符")
    require(not trades.date.duplicated().any() if len(trades) else True, "每日多于一条成交请求记录")
    trade_map = {pd.Timestamp(row.date): row for row in trades.itertuples()}
    cash, shares, nav = float(config["initial_capital"]), 0, float(config["initial_capital"])
    receivables, entitlements = {}, {}
    maximum_errors = {}
    mark = float(p.data.close.iloc[0])
    amounts = p.dividends.cash_dividend_per_share.to_numpy(float)

    def check(name: str, observed: float, expected: float, tolerance: float = 1e-7) -> None:
        error = abs(float(observed) - float(expected))
        require(np.isfinite(error) and error <= tolerance, f"会计核对失败：{name}，差额{error}")
        maximum_errors[name] = max(maximum_errors.get(name, 0.0), error)

    for day, saved in enumerate(ledger.itertuples(), 1):
        market = p.data.iloc[day]
        check("open", saved.open, market.open)
        check("close", saved.close, market.close)
        old_shares, recognized, paid = shares, 0.0, 0.0
        for key in p.ex_keys[day]:
            amount = entitlements.get(key, 0) * amounts[key]
            receivables[key] = amount
            recognized += amount
        for key in p.pay_open_keys[day]:
            amount = receivables.pop(key, 0.0)
            paid += amount
            cash += amount
        pretrade = cash + shares * float(market.open) + sum(receivables.values())
        check("filled_integral", saved.filled_quantity, int(saved.filled_quantity), 0)
        quantity = int(saved.filled_quantity)
        require(quantity % config["lot"] == 0 and shares + quantity >= 0, "成交违反整手/无卖空")
        require(quantity >= 0 or -quantity <= old_shares, "卖出超过前日可用份额，违反T+1")
        require(pd.Timestamp(saved.request_origin) < market.date, "请求来源不是此前收盘")
        fee = slip = notional = 0.0
        if saved.requested_quantity:
            require(market.date in trade_map, "请求缺少成交日志")
            trade = trade_map[market.date]
            check("trade_request", trade.requested_quantity, saved.requested_quantity, 0)
            check("trade_fill", trade.filled_quantity, quantity, 0)
            require(pd.Timestamp(trade.origin) == pd.Timestamp(saved.request_origin), "成交来源日期不符")
            check("pre_trade_nav", trade.pre_trade_nav, pretrade)
        else:
            require(market.date not in trade_map and quantity == 0, "无请求却存在成交")
        if quantity:
            require(opening_inputs_valid(market), "非法开盘输入发生交易")
            side = 1 if quantity > 0 else -1
            px = fill_price(float(market.open), side, cost, config["tick"])
            fee, slip = commission(quantity, px, cost), abs(quantity) * abs(px - market.open)
            notional = abs(quantity) * px
            check("fill_price", trade.fill_price, px, 1e-12)
            check("trade_commission", trade.commission, fee)
            check("trade_slippage", trade.slippage_cost, slip)
            basis, tick = float(market.previous_close - market.dividend), config["tick"]
            lower = math.floor(basis * (1 - config["limit_fraction"]) / tick + .5 + 1e-9) * tick
            upper = math.floor(basis * (1 + config["limit_fraction"]) / tick + .5 + 1e-9) * tick
            require((side > 0 and market.open < upper - 1e-9 and px <= upper + 1e-9)
                    or (side < 0 and market.open > lower + 1e-9 and px >= lower - 1e-9), "成交违反冻结方向涨跌停规则")
            cash -= quantity * px + fee
            shares += quantity
        check("cash_after_execution", saved.cash_after_execution, cash)
        for key in p.pay_close_keys[day]:
            amount = receivables.pop(key, 0.0)
            paid += amount
            cash += amount
        for key in p.record_keys[day]:
            entitlements[key] = shares
        wealth = cash + shares * market.close + sum(receivables.values())
        price_pnl = old_shares * (market.open - mark) + shares * (market.close - market.open)
        require(cash >= -1e-7 and wealth > 0, "现金透支或权益非正")
        checks = {"cash": cash, "shares": shares, "equity": wealth, "dividend_receivable": sum(receivables.values()),
                  "dividend_recognized": recognized, "dividend_paid": paid, "commission": fee,
                  "slippage_cost": slip, "traded_notional": notional, "pnl": wealth - nav, "price_pnl": price_pnl}
        for name, expected in checks.items():
            check(name, getattr(saved, name), expected)
        check("wealth_identity", wealth - nav, price_pnl + recognized - fee - slip)
        check("net_return", saved.net_return, wealth / nav - 1, 1e-12)
        check("exposure", saved.exposure, shares * market.close / wealth, 1e-12)
        check("turnover", saved.turnover, notional / pretrade, 1e-12)
        nav, mark = wealth, float(market.close)
    return {"status": "PASS_ACCOUNT_IDENTITIES", "days": len(ledger), "maximum_errors": maximum_errors,
            "ending_equity_reconstructed": nav, "entitlement_shares_by_dividend_row": entitlements,
            "account_simulator_calls": 0, "scope": "保存成交会计复算；不证明历史每笔开盘真实可成交"}


def ideal_paths(ledgers: dict[str, pd.DataFrame], config: dict) -> tuple[dict, dict]:
    paths, matching = {}, {}
    for cost in ("BASE", "STRESS"):
        full, bh = ledgers[f"FULL_{cost}"], ledgers[f"BUY_HOLD_{cost}"]
        ratios = {"EXPOSURE": full.exposure.mean() / bh.exposure.mean(),
                  "VOLATILITY": full.net_return.std(ddof=1) / bh.net_return.std(ddof=1)}
        for method, ratio in ratios.items():
            key = f"IDEAL_{method}_{cost}"
            require(np.isfinite(ratio) and ratio > 0, "理想匹配比例不可用")
            returns = ratio * bh.net_return.to_numpy(float)
            require((returns > -1).all(), "理想路径无法形成正财富")
            frame = pd.DataFrame({"date": full.date, "net_return": returns,
                                  "ideal_equity": config["initial_capital"] * np.cumprod(1 + returns),
                                  "reference_exposure": ratio * bh.exposure.to_numpy(float),
                                  "matching_ratio": float(ratio), "tradable": False,
                                  "hindsight_matching": True})
            paths[key] = frame
            info = {"ratio": float(ratio), "reference": f"BUY_HOLD_{cost}", "method": method,
                    "target_full_mean_exposure": float(full.exposure.mean()),
                    "ideal_mean_reference_exposure": float(frame.reference_exposure.mean()),
                    "target_full_daily_volatility": float(full.net_return.std(ddof=1)),
                    "ideal_daily_volatility": float(frame.net_return.std(ddof=1)),
                    "net_return_linear_scaling_max_error": float(np.abs(frame.net_return - ratio * bh.net_return).max()),
                    "tradable": False, "hindsight_matching": True, "bootstrap_ratio_refitted": False}
            error = abs(info["target_full_mean_exposure"] - info["ideal_mean_reference_exposure"]) if method == "EXPOSURE" else abs(info["target_full_daily_volatility"] - info["ideal_daily_volatility"])
            require(error < 1e-12, "理想标准化未匹配指定统计量")
            info["matching_error"] = error
            matching[key] = info
    return paths, matching


STAT_NAMES = ["net_sharpe", "annualized_return", "utility_gamma5", "annualized_arithmetic_mean", "ending_equity"]


def statistic_matrix(returns: np.ndarray, annual: int, capital: float) -> np.ndarray:
    n = len(returns)
    mu, sigma = returns.mean(axis=0) * annual, returns.std(axis=0, ddof=1) * math.sqrt(annual)
    logs = np.log1p(returns).sum(axis=0)
    return np.column_stack([np.divide(mu, sigma, out=np.full_like(mu, np.nan), where=sigma > 1e-8),
                            np.expm1(logs * annual / n), mu - 2.5 * sigma ** 2, mu, capital * np.exp(logs)])


def joint_bootstrap(paths: dict[str, pd.DataFrame], pairs: list[tuple[str, str]],
                    config: dict, diagnostic: dict) -> tuple[dict, np.ndarray, np.ndarray]:
    keys = list(paths)
    dates = pd.DatetimeIndex(pd.to_datetime(paths[keys[0]].date))
    for key in keys:
        require(pd.DatetimeIndex(pd.to_datetime(paths[key].date)).equals(dates), "联合区块路径日期不一致")
    returns = np.column_stack([paths[key].net_return.to_numpy(float) for key in keys])
    require(np.isfinite(returns).all() and (returns > -1).all(), "统计路径收益非法")
    n = len(returns)
    repetitions, block = diagnostic["bootstrap_repetitions"], diagnostic["bootstrap_block_days"]
    rng = np.random.default_rng(diagnostic["bootstrap_seed"])
    samples = np.empty((repetitions, len(keys), len(STAT_NAMES)))
    indices = np.empty((repetitions, n), dtype=np.int32)
    for rep in range(repetitions):
        starts = rng.integers(0, n, size=math.ceil(n / block))
        ix = ((starts[:, None] + np.arange(block)) % n).reshape(-1)[:n]
        indices[rep] = ix
        samples[rep] = statistic_matrix(returns[ix], config["annual_days"], config["initial_capital"])
    point = statistic_matrix(returns, config["annual_days"], config["initial_capital"])
    result = {"path_order": keys, "statistic_order": STAT_NAMES, "block_days": block,
              "repetitions": repetitions, "seed": diagnostic["bootstrap_seed"],
              "method": "共同索引的循环移动区块；95%百分位区间；匹配比例固定；仅条件于已保存路径",
              "selection_adjusted": False, "joint_calls": 1, "models": {}, "paired_increments": {}}
    for j, key in enumerate(keys):
        result["models"][key] = {name: {"point": float(point[j, k]),
                                               "interval95": np.nanquantile(samples[:, j, k], [.025, .975]).tolist()}
                                  for k, name in enumerate(STAT_NAMES)}
    for left, right in pairs:
        i, j = keys.index(left), keys.index(right)
        result["paired_increments"][f"{left}_MINUS_{right}"] = {
            name: {"point": float(point[i, k] - point[j, k]),
                   "interval95": np.nanquantile(samples[:, i, k] - samples[:, j, k], [.025, .975]).tolist()}
            for k, name in enumerate(STAT_NAMES)}
    return result, indices, samples
