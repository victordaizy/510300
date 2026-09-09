"""510300 日内—隔夜条件增量的冻结历史研究与现金账户模拟。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/510300_intraday_overnight_increment_v1.json"
FEATURES = {"M0": ["M20", "LOG_RV20"], "M1": ["M20", "LOG_RV20", "D20"]}


def now() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def safe_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [safe_json(v) for v in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def write_json(path: Path, value: dict, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(safe_json(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if exclusive:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(text)
    else:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def normalize_prices(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["date", "open", "high", "low", "close", "volume"]
    require(set(required).issubset(frame), "行情缺少必需字段")
    data = frame.copy()
    data["date"] = pd.to_datetime(data.date, errors="raise").dt.normalize()
    require(not data.date.duplicated().any(), "行情日期重复，禁止自动去重")
    data = data.sort_values("date").reset_index(drop=True)
    values = data[["open", "high", "low", "close", "volume"]].to_numpy(float)
    require(np.isfinite(values).all() and (values > 0).all(), "行情缺失、非正价或无成交；不能填补")
    require((data.high >= data[["open", "close", "low"]].max(axis=1)).all(), "最高价关系错误")
    require((data.low <= data[["open", "close", "high"]].min(axis=1)).all(), "最低价关系错误")
    if "symbol" in data:
        require(set(data.symbol) == {"510300.SH"}, "行情标的超出510300范围")
    return data


def normalize_dividends(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    required = {"record_date", "ex_date", "payment_date", "cash_dividend_per_share"}
    require(required.issubset(data), "分红日期或金额不完整")
    for column in ("record_date", "ex_date", "payment_date"):
        data[column] = pd.to_datetime(data[column], errors="raise").dt.normalize()
    require(not data.ex_date.duplicated().any(), "同一除息日重复，禁止自动合并来源")
    require((data.record_date < data.ex_date).all(), "登记日必须早于除息日")
    require((data.ex_date <= data.payment_date).all(), "付款日不能早于除息日")
    cash = data.cash_dividend_per_share.to_numpy(float)
    require(np.isfinite(cash).all() and (cash > 0).all(), "分红金额非法")
    if "symbol" in data:
        require(set(data.symbol) == {"510300.SH"}, "分红标的不符")
    return data.sort_values("ex_date").reset_index(drop=True)


def build_features(prices: pd.DataFrame, dividends: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """除息现金加入日内外财富分解；滚动计算严格只访问当前及过去。"""
    data = prices.copy()
    distribution = dividends.set_index("ex_date").cash_dividend_per_share
    data["dividend"] = data.date.map(distribution).fillna(0.0)
    prev = data.close.shift(1)
    data["previous_close"] = prev
    data["overnight_log"] = np.log((data.open + data.dividend) / prev)
    data["intraday_log"] = np.log((data.close + data.dividend) / (data.open + data.dividend))
    data["total_simple"] = (data.close + data.dividend) / prev - 1.0
    data["total_log"] = np.log1p(data.total_simple)
    data["identity_error"] = data.overnight_log + data.intraday_log - data.total_log
    data["M20"] = data.total_log.rolling(lookback, min_periods=lookback).sum()
    data["D20"] = (data.intraday_log - data.overnight_log).rolling(lookback, min_periods=lookback).sum()
    data["RV20"] = data.total_simple.rolling(lookback, min_periods=lookback).var(ddof=1)
    data["LOG_RV20"] = np.log(data.RV20.where(data.RV20 > 0))
    return data


def load_inputs(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    inputs = {key: ROOT / path for key, path in config["inputs"].items()}
    price_receipt = json.loads(inputs["price_receipt"].read_text(encoding="utf-8"))
    coverage = json.loads(inputs["dividend_coverage"].read_text(encoding="utf-8"))
    require(price_receipt["status"] == "PASS", "既有修正价格未获准入")
    require(digest(inputs["prices"]) == price_receipt["canonical_price"]["output_sha256"], "价格与既有修正回执不一致")
    require(coverage["complete_history_confirmed"], "缺少完整分红覆盖证明")
    require(digest(inputs["dividends"]) == coverage["distribution_file_sha256"], "分红表与覆盖证明不一致")
    require(pd.Timestamp(coverage["coverage_end"]) >= pd.Timestamp(config["data_cutoff"]), "分红覆盖早于本轮截止")
    prices = normalize_prices(pd.read_parquet(inputs["prices"]))
    dividends = normalize_dividends(pd.read_csv(inputs["dividends"]))
    require(len(dividends) == int(coverage["event_count"]), "分红事件总数不符")
    calendar = pd.read_parquet(inputs["calendar"])
    dates = pd.DatetimeIndex(pd.to_datetime(calendar.loc[calendar.is_open, "trade_date"]))
    require(not dates.duplicated().any(), "交易日历重复")
    first = pd.Timestamp(config["train_origin_start"])
    cutoff = pd.Timestamp(config["data_cutoff"])
    before = prices.loc[prices.date < first].tail(config["lookback"])
    selected = prices.loc[prices.date.between(first, cutoff)]
    require(len(before) == config["lookback"], "训练开始前不足20日暖启动")
    prices = pd.concat([before, selected], ignore_index=True)
    expected = dates[(dates >= prices.date.iloc[0]) & (dates <= cutoff)].sort_values()
    require(pd.DatetimeIndex(prices.date).equals(expected), "行情与独立日期日历不完全相同")
    require(prices.date.iloc[-1] == cutoff, "行情未到冻结终点")
    for event in dividends.itertuples():
        if first <= event.ex_date <= cutoff:
            require(event.record_date in expected and event.ex_date in expected, "分红登记或除息日不在交易日历")
            ex_pos = expected.get_loc(event.ex_date)
            require(expected[ex_pos - 1] == event.record_date, "登记到除息间存在未定义的权益日")
    features = build_features(prices, dividends, config["lookback"])
    relevant = features.loc[features.date >= first]
    require(np.isfinite(relevant[FEATURES["M1"] + ["RV20"]].to_numpy()).all(), "所需原点存在不合格特征")
    identity_max = float(features.identity_error.abs().max())
    require(identity_max < 1e-12, "分红财富分解不守恒")
    original = normalize_prices(pd.read_parquet(inputs["original_prices"]))
    price_fields = ["date", "open", "high", "low", "close"]
    overlap = prices[price_fields].merge(original[price_fields], on="date", suffixes=("_fixed", "_original"), validate="one_to_one")
    differences = []
    for column in ["open", "high", "low", "close"]:
        mask = (overlap[column + "_fixed"] - overlap[column + "_original"]).abs() > 1e-12
        for row in overlap.loc[mask].to_dict("records"):
            differences.append({"date": row["date"], "field": column, "original": row[column + "_original"], "fixed": row[column + "_fixed"]})
    report = {
        "study_id": config["study_id"], "created_at": now(), "status": "PASS_FIXED_DATA_CONTRACT",
        "start": first, "cutoff": cutoff, "research_days": len(selected), "warmup_days": len(before),
        "missing_calendar_days": 0, "nonpositive_or_missing_prices": 0,
        "dividend_events_in_research_window": int(dividends.ex_date.between(first, cutoff).sum()),
        "identity_max_absolute_error": identity_max, "existing_price_corrections": differences,
        "price_source": "既有新浪未复权价格，三处收盘价沿用已核验的Tushare/腾讯一致裁决",
        "source_limit": "历史下载及更正数据，不声称拥有每个历史交易日原始发布时刻的快照",
        "cash_yield": "明确零收益假设", "new_future_labels_computed": False,
        "new_models_fitted": False, "new_portfolios_computed": False,
        "input_identities": {key: {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "bytes": path.stat().st_size, "sha256": digest(path)} for key, path in inputs.items()},
    }
    return features, dividends, report


def holding_total_return(data: pd.DataFrame, dividends: pd.DataFrame, entry: int, end: int) -> tuple[float, float]:
    """买入后的五日财富；除息日买入没有此前登记的权益。"""
    entry_date, exit_date = data.date.iloc[entry], data.date.iloc[end]
    eligible = dividends.loc[(dividends.record_date >= entry_date) & (dividends.record_date < exit_date) & (dividends.ex_date <= exit_date)]
    distribution = float(eligible.cash_dividend_per_share.sum())
    total = (float(data.open.iloc[end]) + distribution) / float(data.open.iloc[entry]) - 1.0
    return total, distribution


def sample_schedule(data: pd.DataFrame, config: dict, *, with_labels: bool, dividends: pd.DataFrame) -> pd.DataFrame:
    dates = pd.DatetimeIndex(data.date)
    first_eval = int(np.flatnonzero(data.date.dt.year.to_numpy() >= config["evaluation_start_year"])[0])
    anchor = first_eval - 1
    first_allowed = pd.Timestamp(config["train_origin_start"])
    train_cutoff = pd.Timestamp(config["train_cutoff"])
    rows = []
    for t in range(len(data)):
        if data.date.iloc[t] < first_allowed or (t - anchor) % config["horizon"]:
            continue
        entry = t + 1
        end = t + 1 + config["horizon"]
        mature = end < len(data)
        row = {"origin_index": t, "origin": dates[t], "entry_index": entry, "exit_index": end,
               "entry_date": dates[entry] if entry < len(data) else pd.NaT,
               "exit_date": dates[end] if mature else pd.NaT,
               "label_mature": mature, "train_eligible": bool(mature and dates[end] <= train_cutoff),
               "evaluation_eligible": bool(entry >= first_eval and mature),
               "quality": "PASS", "label_status": "MATURE" if mature else "CENSORED_AFTER_CUTOFF"}
        for column in FEATURES["M1"] + ["RV20"]:
            row[column] = float(data[column].iloc[t])
        if with_labels:
            row["Y5"] = np.nan
            row["label_dividend_per_share"] = np.nan
            if mature:
                total, distribution = holding_total_return(data, dividends, entry, end)
                row["label_dividend_per_share"] = distribution
                row["Y5"] = total
        rows.append(row)
    return pd.DataFrame(rows)


def fit_ridge(sample: pd.DataFrame, model_id: str, penalty: float) -> dict:
    columns = FEATURES[model_id]
    x, y = sample[columns].to_numpy(float), sample.Y5.to_numpy(float)
    require(len(x) > len(columns) + 1 and np.isfinite(x).all() and np.isfinite(y).all(), "训练数据不足或非法")
    mean, scale = x.mean(axis=0), x.std(axis=0, ddof=0)
    require((scale > 0).all(), "训练特征无方差")
    z = (x - mean) / scale
    intercept = float(y.mean())
    slopes = np.linalg.solve(z.T @ z / len(z) + penalty * np.eye(len(columns)), z.T @ (y - intercept) / len(z))
    return {"model_id": model_id, "features": columns, "mean": mean.tolist(), "scale": scale.tolist(),
            "intercept": intercept, "slopes": slopes.tolist(), "ridge_lambda": penalty,
            "training_observations": len(sample), "training_origin_start": sample.origin.min(),
            "training_origin_end": sample.origin.max(), "last_training_label_exit": sample.exit_date.max(),
            "standardization_ddof": 0, "label_retraining_allowed": False}


def predict(model: dict, sample: pd.DataFrame) -> np.ndarray:
    x = sample[model["features"]].to_numpy(float)
    return float(model["intercept"]) + ((x - np.array(model["mean"])) / np.array(model["scale"])) @ np.array(model["slopes"])


def commission(quantity: int, price: float, cost: dict) -> float:
    return max(abs(quantity) * price * cost["commission"], cost["minimum"]) if quantity else 0.0


def fill_price(price: float, side: int, cost: dict, tick: float) -> float:
    if cost["slippage"] == 0:
        return float(price)
    value = price * (1.0 + side * cost["slippage"]) / tick
    return (math.ceil(value - 1e-10) if side > 0 else math.floor(value + 1e-10)) * tick


def affordable_quantity(cash: float, price: float, cost: dict, lot: int) -> int:
    if cash <= 0:
        return 0
    lots = int(max(0, math.floor((cash + 1e-9) / (price * lot))))
    while lots > 0 and lots * lot * price + commission(lots * lot, price, cost) > cash + 1e-8:
        lots -= 1
    return lots * lot


@dataclass
class Account:
    cash: float
    shares: int = 0
    receivables: dict[int, float] = field(default_factory=dict)
    entitlements: dict[int, int] = field(default_factory=dict)
    purchase_lots: list[tuple[int, int]] = field(default_factory=list)

    def receivable(self) -> float:
        return float(sum(self.receivables.values()))

    def value(self, price: float) -> float:
        return self.cash + self.shares * price + self.receivable()

    def sellable(self, day_index: int) -> int:
        return sum(quantity for day, quantity in self.purchase_lots if day < day_index)

    def assert_valid(self) -> None:
        require(self.cash >= -1e-7 and self.shares >= 0, "账户发生透支或卖空")
        require(sum(quantity for _, quantity in self.purchase_lots) == self.shares, "T+1份额批次与总份额不符")
        require(all(quantity >= 0 for _, quantity in self.purchase_lots), "持仓批次出现负值")


def choose_order(account: Account, close: float, mu: float, variance5: float, cost: dict, config: dict) -> dict:
    """只使用收盘时的账户和特征；结果是次日固定份额请求。"""
    nav = account.value(close)
    require(nav > 0 and np.isfinite([close, mu, variance5]).all(), "决策输入非法")
    candidates = [("HOLD", account.shares)]
    buy_price = fill_price(close, 1, cost, config["tick"])
    max_buy = affordable_quantity(account.cash, buy_price, cost, config["lot"])
    for weight in config["weights"]:
        target = int(math.floor(weight * nav / close / config["lot"])) * config["lot"]
        target = min(target, account.shares + max_buy)
        candidates.append((f"TARGET_{int(weight * 100)}", target))
    best = None
    seen = set()
    for action, target in candidates:
        if target in seen:
            continue
        seen.add(target)
        quantity = target - account.shares
        side = 1 if quantity > 0 else -1
        px = fill_price(close, side, cost, config["tick"])
        friction = abs(quantity) * abs(px - close) + commission(quantity, px, cost)
        weight = target * close / nav
        score = weight * mu - config["gamma"] / 2 * weight * weight * variance5 - friction / nav
        item = {"action": action, "requested_quantity": quantity, "reference_weight": weight,
                "score": score, "estimated_cost": friction, "reference_close": close,
                "mu5": mu, "variance5": variance5, "nav_at_decision": nav}
        if best is None or score > best["score"] + 1e-12:
            best = item
    require(best is not None, "缺少有效候选动作")
    return best


def execute_order(account: Account, quantity: int, open_price: float, previous_close: float,
                  dividend: float, day_index: int, cost: dict, config: dict) -> dict:
    requested = int(quantity)
    row = {"requested_quantity": requested, "filled_quantity": 0, "open_price": open_price,
           "fill_price": None, "commission": 0.0, "slippage_cost": 0.0, "notional": 0.0,
           "status": "NO_TRADE", "cash_before": account.cash, "shares_before": account.shares}
    if not requested:
        return row
    require(requested % config["lot"] == 0, "普通买卖不符合100份整手")
    side = 1 if requested > 0 else -1
    px = fill_price(open_price, side, cost, config["tick"])
    basis = previous_close - dividend
    tick = config["tick"]
    lower = math.floor(basis * (1 - config["limit_fraction"]) / tick + 0.5 + 1e-9) * tick
    upper = math.floor(basis * (1 + config["limit_fraction"]) / tick + 0.5 + 1e-9) * tick
    if (side > 0 and (open_price >= upper - 1e-9 or px > upper + 1e-9)) or (side < 0 and (open_price <= lower + 1e-9 or px < lower - 1e-9)):
        row["status"] = "UNFILLED_DIRECTIONAL_LIMIT"
        return row
    if requested > 0:
        quantity = min(requested, affordable_quantity(account.cash, px, cost, config["lot"]))
    else:
        quantity = -min(abs(requested), account.sellable(day_index))
    if not quantity:
        row["status"] = "UNFILLED_CASH_OR_T_PLUS_ONE"
        return row
    fee = commission(quantity, px, cost)
    account.cash -= quantity * px + fee
    account.shares += quantity
    if quantity > 0:
        account.purchase_lots.append((day_index, quantity))
    else:
        left = -quantity
        remaining = []
        for bought, count in account.purchase_lots:
            remove = min(left, count) if bought < day_index else 0
            left -= remove
            if count > remove:
                remaining.append((bought, count - remove))
        require(left == 0, "卖出超出可用旧库存")
        account.purchase_lots = remaining
    account.assert_valid()
    row.update({"filled_quantity": quantity, "fill_price": px, "commission": fee,
                "slippage_cost": abs(quantity) * abs(px - open_price), "notional": abs(quantity) * px,
                "status": "FILLED" if quantity == requested else "PARTIALLY_FILLED_CASH_OR_T_PLUS_ONE",
                "cash_after": account.cash, "shares_after": account.shares})
    return row


def simulate(data: pd.DataFrame, dividends: pd.DataFrame, predictions: dict[int, float],
             cost: dict, config: dict, model_id: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    first_eval = int(np.flatnonzero(data.date.dt.year.to_numpy() >= config["evaluation_start_year"])[0])
    last = len(data) - 1
    account = Account(float(config["initial_capital"]))
    events = dividends.to_dict("records")
    anchor = first_eval - 1
    if model_id == "BUY_HOLD":
        price = fill_price(float(data.close.iloc[anchor]), 1, cost, config["tick"])
        pending = {"requested_quantity": affordable_quantity(account.cash, price, cost, config["lot"]), "action": "INITIAL_BUY_HOLD"}
    else:
        pending = choose_order(account, float(data.close.iloc[anchor]), predictions[anchor], config["horizon"] * float(data.RV20.iloc[anchor]), cost, config)
    decisions = [{"origin": data.date.iloc[anchor], "execution_date": data.date.iloc[first_eval], **pending}]
    ledgers, trades = [], []
    previous_nav = float(config["initial_capital"])
    previous_mark = float(data.open.iloc[first_eval])
    for day in range(first_eval, last + 1):
        market = data.iloc[day]
        date = pd.Timestamp(market.date)
        old_shares = account.shares
        recognized, paid = 0.0, 0.0
        for key, event in enumerate(events):
            if event["ex_date"] == date:
                amount = account.entitlements.get(key, 0) * event["cash_dividend_per_share"]
                if amount:
                    account.receivables[key] = amount
                    recognized += amount
            if event["payment_date"] < date and key in account.receivables:
                amount = account.receivables.pop(key)
                account.cash += amount
                paid += amount
        pre_trade_nav = account.value(float(market.open))
        if day == last:
            request = -account.shares
            action = "TERMINAL_LIQUIDATION"
        elif pending is not None:
            request, action = int(pending["requested_quantity"]), pending["action"]
        else:
            request, action = 0, "NO_SCHEDULED_ORDER"
        execution = execute_order(account, request, float(market.open), float(market.previous_close), float(market.dividend), day, cost, config)
        if request:
            trades.append({"date": date, "origin": data.date.iloc[day - 1], "action": action,
                           "pre_trade_nav": pre_trade_nav, **execution})
        pending = None
        terminal = day == last
        mark = float(market.open if terminal else market.close)
        if not terminal:
            for key, event in enumerate(events):
                if event["payment_date"] == date and key in account.receivables:
                    amount = account.receivables.pop(key)
                    account.cash += amount
                    paid += amount
                if event["record_date"] == date:
                    account.entitlements[key] = account.shares
        nav = account.value(mark)
        price_pnl = old_shares * (float(market.open) - previous_mark) + account.shares * (mark - float(market.open))
        expected_change = price_pnl + recognized - execution["commission"] - execution["slippage_cost"]
        accounting_error = (nav - previous_nav) - expected_change
        require(abs(accounting_error) < 1e-6, f"账户财富变化不守恒：{date.date()} {accounting_error}")
        account.assert_valid()
        ledgers.append({"date": date, "mark_clock": "OPEN_TERMINAL" if terminal else "CLOSE",
                        "open": float(market.open), "mark": mark, "shares": account.shares,
                        "cash": account.cash, "dividend_receivable": account.receivable(), "equity": nav,
                        "net_return": nav / previous_nav - 1.0, "pnl": nav - previous_nav,
                        "price_pnl": price_pnl, "dividend_recognized": recognized, "dividend_paid": paid,
                        "commission": execution["commission"], "slippage_cost": execution["slippage_cost"],
                        "traded_notional": execution["notional"], "turnover": execution["notional"] / pre_trade_nav,
                        "exposure": account.shares * mark / nav, "accounting_error": accounting_error,
                        "execution_status": execution["status"], "terminal_unliquidated": bool(terminal and account.shares)})
        previous_nav, previous_mark = nav, mark
        if not terminal and (day - anchor) % config["horizon"] == 0 and day + 1 < last:
            if model_id != "BUY_HOLD":
                pending = choose_order(account, float(market.close), predictions[day], config["horizon"] * float(market.RV20), cost, config)
                decisions.append({"origin": date, "execution_date": data.date.iloc[day + 1], **pending})
    trade_columns = ["date", "origin", "action", "pre_trade_nav", "requested_quantity", "filled_quantity", "open_price", "fill_price", "commission", "slippage_cost", "notional", "status"]
    return pd.DataFrame(ledgers), pd.DataFrame(trades).reindex(columns=trade_columns), pd.DataFrame(decisions)


def return_metrics(returns: np.ndarray, annual_days: int) -> dict:
    values = np.asarray(returns, dtype=float)
    require(len(values) > 1 and np.isfinite(values).all() and (values > -1).all(), "绩效收益序列非法")
    log_equity = np.r_[0.0, np.cumsum(np.log1p(values))]
    volatility = float(values.std(ddof=1) * np.sqrt(annual_days))
    mean = float(values.mean() * annual_days)
    return {"cumulative_return": float(np.expm1(log_equity[-1])),
            "annualized_return": float(np.expm1(log_equity[-1] * annual_days / len(values))),
            "annualized_arithmetic_mean": mean, "annualized_volatility": volatility,
            "net_sharpe": mean / volatility if volatility > 1e-15 else None,
            "max_drawdown": float(np.expm1(log_equity - np.maximum.accumulate(log_equity)).min())}


def block_indices(rng: np.random.Generator, size: int, block: int) -> np.ndarray:
    starts = rng.integers(0, size, size=math.ceil(size / block))
    return ((starts[:, None] + np.arange(block)) % size).reshape(-1)[:size]


def interval(values: list[float]) -> list[float | None]:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    return np.quantile(clean, [0.025, 0.975]).tolist() if len(clean) else [None, None]


def prediction_statistics(evaluation: pd.DataFrame, models: dict, config: dict) -> dict:
    y = evaluation.Y5.to_numpy(float)
    losses, report = {}, {}
    for key in FEATURES:
        prediction = evaluation[f"prediction_{key}"].to_numpy(float)
        losses[key] = (y - prediction) ** 2
        correlation = float(np.corrcoef(y, prediction)[0, 1]) if prediction.std() > 1e-15 else None
        report[key] = {"MSE": float(losses[key].mean()), "MAE": float(np.abs(y - prediction).mean()), "pearson": correlation}
    improvement = 1 - float(losses["M1"].mean() / losses["M0"].mean())
    x = evaluation[FEATURES["M1"]].to_numpy(float)
    z = (x - np.asarray(models["M1"]["mean"])) / np.asarray(models["M1"]["scale"])
    diagnostic = np.linalg.lstsq(np.column_stack([np.ones(len(y)), z]), y, rcond=None)[0]
    train_direction = float(models["M1"]["slopes"][-1])
    eval_direction = float(diagnostic[-1])
    rng = np.random.default_rng(config["random_seed"])
    samples = []
    for _ in range(config["bootstrap_repetitions"]):
        index = block_indices(rng, len(y), config["bootstrap_origin_block"])
        samples.append(1 - float(losses["M1"][index].mean() / losses["M0"][index].mean()))
    report.update({"observations": len(y), "M1_relative_MSE_improvement": improvement,
                   "MSE_improvement_95pct_block_interval": interval(samples),
                   "training_standardized_D20_coefficient": train_direction,
                   "evaluation_D20_conditional_OLS_diagnostic": eval_direction,
                   "direction_reversed": train_direction * eval_direction < 0,
                   "evaluation_coefficient_used_for_prediction": False})
    return report


def account_uncertainty(ledgers: dict[str, pd.DataFrame], config: dict) -> dict:
    arrays = {key: ledger.net_return.to_numpy(float) for key, ledger in ledgers.items()}
    rng = np.random.default_rng(config["random_seed"])
    results = {"M1_sharpe": [], "M1_minus_M0_sharpe": [], "M1_minus_M0_annualized_return": [], "M1_minus_BUY_HOLD_annualized_return": []}
    for _ in range(config["bootstrap_repetitions"]):
        index = block_indices(rng, len(arrays["M0"]), config["bootstrap_day_block"])
        metrics = {key: return_metrics(value[index], config["annual_days"]) for key, value in arrays.items()}
        s0, s1 = metrics["M0"]["net_sharpe"], metrics["M1"]["net_sharpe"]
        results["M1_sharpe"].append(float(s1) if s1 is not None else np.nan)
        results["M1_minus_M0_sharpe"].append(s1 - s0 if s1 is not None and s0 is not None else np.nan)
        results["M1_minus_M0_annualized_return"].append(metrics["M1"]["annualized_return"] - metrics["M0"]["annualized_return"])
        results["M1_minus_BUY_HOLD_annualized_return"].append(metrics["M1"]["annualized_return"] - metrics["BUY_HOLD"]["annualized_return"])
    return {key: interval(values) for key, values in results.items()}


def evaluate(data: pd.DataFrame, dividends: pd.DataFrame, config: dict) -> dict:
    sample = sample_schedule(data, config, with_labels=True, dividends=dividends)
    training = sample.loc[sample.train_eligible].copy()
    models = {key: fit_ridge(training, key, config["ridge_lambda"]) for key in FEATURES}
    require(training.exit_date.max() <= pd.Timestamp(config["train_cutoff"]), "训练使用了跨截止标签")
    for key in FEATURES:
        sample[f"prediction_{key}"] = predict(models[key], sample)
    evaluation = sample.loc[sample.evaluation_eligible].copy()
    predictions = {key: dict(zip(sample.origin_index.astype(int), sample[f"prediction_{key}"].astype(float))) for key in FEATURES}
    metrics, annual, all_ledgers, all_trades, all_decisions, uncertainties = [], [], {}, {}, {}, {}
    for scenario, cost in config["costs"].items():
        scenario_ledgers = {}
        for key in ["BUY_HOLD", "M0", "M1"]:
            ledger, trades, decisions = simulate(data, dividends, predictions.get(key, {}), cost, config, key)
            identity = f"{scenario}_{key}"
            all_ledgers[identity], all_trades[identity], all_decisions[identity] = ledger, trades, decisions
            scenario_ledgers[key] = ledger
            values = return_metrics(ledger.net_return.to_numpy(float), config["annual_days"])
            values.update({"scenario": scenario, "model": key, "start_date": ledger.date.iloc[0], "end_date": ledger.date.iloc[-1],
                           "days": len(ledger), "ending_equity": float(ledger.equity.iloc[-1]),
                           "commission_cny": float(ledger.commission.sum()), "slippage_cny": float(ledger.slippage_cost.sum()),
                           "total_cost_cny": float(ledger.commission.sum() + ledger.slippage_cost.sum()),
                           "annualized_turnover": float(ledger.turnover.sum() * config["annual_days"] / len(ledger)),
                           "mean_exposure": float(ledger.exposure.mean()),
                           "filled_orders": int((trades.filled_quantity != 0).sum()),
                           "unfilled_orders": int((trades.filled_quantity == 0).sum()),
                           "partial_orders": int(trades.status.eq("PARTIALLY_FILLED_CASH_OR_T_PLUS_ONE").sum()),
                           "terminal_liquidated": not bool(ledger.terminal_unliquidated.iloc[-1]),
                           "maximum_accounting_error": float(ledger.accounting_error.abs().max())})
            metrics.append(values)
            for year, part in ledger.groupby(ledger.date.dt.year, sort=True):
                annual.append({"scenario": scenario, "model": key, "year": int(year), "days": len(part),
                               "net_return": float(np.expm1(np.log1p(part.net_return).sum())),
                               "pnl_cny": float(part.pnl.sum()), "contribution_to_initial_capital": float(part.pnl.sum() / config["initial_capital"]),
                               "cost_cny": float(part.commission.sum() + part.slippage_cost.sum())})
        if scenario != "ZERO_COST_DIAGNOSTIC":
            uncertainties[scenario] = account_uncertainty(scenario_ledgers, config)
    table = pd.DataFrame(metrics)
    for scenario in config["costs"]:
        mask = table.scenario.eq(scenario)
        subset = table.loc[mask].set_index("model")
        for metric in ["cumulative_return", "annualized_return", "net_sharpe"]:
            table.loc[mask, f"{metric}_minus_BUY_HOLD"] = table.loc[mask, metric] - subset.loc["BUY_HOLD", metric]
            table.loc[mask, f"{metric}_minus_M0"] = table.loc[mask, metric] - subset.loc["M0", metric]
    annual_table = pd.DataFrame(annual)
    for scenario in config["costs"]:
        for year in annual_table.year.unique():
            mask = annual_table.scenario.eq(scenario) & annual_table.year.eq(year)
            subset = annual_table.loc[mask].set_index("model")
            annual_table.loc[mask, "return_minus_M0"] = annual_table.loc[mask, "net_return"] - subset.loc["M0", "net_return"]
            annual_table.loc[mask, "return_minus_BUY_HOLD"] = annual_table.loc[mask, "net_return"] - subset.loc["BUY_HOLD", "net_return"]
    prediction_report = prediction_statistics(evaluation, models, config)
    m1 = table.loc[table.model.eq("M1") & table.scenario.isin(["BASE", "STRESS"])]
    positive_increment = bool((m1.cumulative_return_minus_M0 > 0).all())
    positive_return = bool((m1.cumulative_return > 0).all())
    liquidated = bool(table.loc[table.scenario.isin(["BASE", "STRESS"]), "terminal_liquidated"].all())
    retain = positive_increment and positive_return and not prediction_report["direction_reversed"] and liquidated
    high_sharpe = bool((m1.net_sharpe.notna() & (m1.net_sharpe >= config["high_sharpe_target"])).all())
    final_models = {key: fit_ridge(sample.loc[sample.label_mature], key, config["ridge_lambda"]) for key in FEATURES} if retain else None
    result = {"study_id": config["study_id"], "completed_at": now(),
              "status": "RETAIN_RESEARCH_REPRESENTATION_FINAL_PARAMETERS_FROZEN" if retain else "STOP_REPRESENTATION_NO_PARAMETER_RESCUE",
              "evidence_class": config["evidence_class"], "historical_runs_consumed": 1,
              "training_observations": len(training), "evaluation_observations": len(evaluation),
              "censored_origins": int((~sample.label_mature).sum()),
              "training_last_label_exit": training.exit_date.max(),
              "evaluation_first_origin": evaluation.origin.min(), "evaluation_last_mature_origin": evaluation.origin.max(),
              "prediction": prediction_report, "account_uncertainty_95pct": uncertainties,
              "retention_conditions": {"positive_M1_net_increment_in_both_costs": positive_increment,
                                       "positive_M1_net_return_in_both_costs": positive_return,
                                       "D20_direction_not_reversed": not prediction_report["direction_reversed"],
                                       "terminal_liquidation_complete": liquidated},
              "historical_point_estimate_sharpe_1_2_in_both_costs": high_sharpe,
              "final_full_history_models_fitted": final_models is not None,
              "future_label_training_allowed": False, "live_trading_authorized": False,
              "current_model_action": "ABSTAIN", "current_model_position_target": "UNSET", "position_impact": 0,
              "comparisons": table.to_dict("records"),
              "limitations": ["历史已被项目反复观察，不能称为真正未见样本外", "日线开盘加摩擦是假想成交，缺少开盘逐笔成交深度", "现金收益采用零假设", "买入持有分红现金留存", "年度和时间区块不是独立市场历史", "来源更正已固定，不按结果选择数据版本"]}
    return {"result": result, "samples": sample, "models": models, "final_models": final_models,
            "comparison": table, "annual": annual_table, "ledgers": all_ledgers, "trades": all_trades, "decisions": all_decisions}


def percent(value: Any, digits: int = 2) -> str:
    return "不可定义" if value is None or pd.isna(value) else f"{value * 100:.{digits}f}%"


def number(value: Any, digits: int = 3) -> str:
    return "不可定义" if value is None or pd.isna(value) else f"{value:.{digits}f}"


def render_report(bundle: dict, config: dict) -> str:
    result, table = bundle["result"], bundle["comparison"]
    p = result["prediction"]
    lines = ["# 510300 日内—隔夜条件增量 V1 结果", "", f"状态：`{result['status']}`。研究观察历史中的冻结回放；当前仓位影响为0。", "",
             f"训练原点从2015年开始，训练截止2019-12-31；训练{result['training_observations']}个五日原点，最后训练标签退出开盘为{str(result['training_last_label_exit'])[:10]}。评价{result['evaluation_observations']}个成熟原点。", "",
             "账户连续覆盖2020年第一个交易日开盘至2026-08-14开盘；末尾未满五日部分仍计入账户。模型仅在训练段拟合一次，评价期间不重训。", "",
             "| 费用情景 | 模型 | 净累计收益 | 年化收益 | 净夏普 | 最大回撤 | 年化收益减买持 | 年化收益减M0 | 总费用(元) | 平均暴露 | 年化单边换手 |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in table.itertuples():
        lines.append(f"| {row.scenario} | {row.model} | {percent(row.cumulative_return)} | {percent(row.annualized_return)} | {number(row.net_sharpe)} | {percent(row.max_drawdown)} | {percent(row.annualized_return_minus_BUY_HOLD)} | {percent(row.annualized_return_minus_M0)} | {row.total_cost_cny:,.2f} | {percent(row.mean_exposure)} | {row.annualized_turnover:.2f} |")
    lines.extend(["", "基础：佣金万分之二、最低5元、单边5bp；压力：万分之四、最低5元、单边10bp。滑点成交价按0.001元向不利方向舍入。初始20万元，100份整手，T+1，零现金收益假设。买持初次买入后持有固定份额，股息现金留存。零费用情景预先登记，仅供摩擦归因；因费用进入决策，其动作可能与含费用情景不同。", "",
                  f"M0 MSE={p['M0']['MSE']:.10f}；M1 MSE={p['M1']['MSE']:.10f}；M1相对误差改善={percent(p['M1_relative_MSE_improvement'], 4)}。其95%时间区块区间为[{percent(p['MSE_improvement_95pct_block_interval'][0])}, {percent(p['MSE_improvement_95pct_block_interval'][1])}]。这是误差变化，不是收益率。", "",
                  f"训练D20标准化系数={p['training_standardized_D20_coefficient']:.8f}；评价条件OLS诊断系数={p['evaluation_D20_conditional_OLS_diagnostic']:.8f}；方向反转={p['direction_reversed']}。后者仅作诊断，未改变预测或交易。", "",
                  "| 情景 | M1夏普95%区间 | M1减M0年化收益95%区间 | M1减买持年化收益95%区间 |", "|---|---|---|---|"])
    for scenario, intervals in result["account_uncertainty_95pct"].items():
        sharpe = intervals["M1_sharpe"]
        delta = intervals["M1_minus_M0_annualized_return"]
        bh = intervals["M1_minus_BUY_HOLD_annualized_return"]
        lines.append(f"| {scenario} | [{number(sharpe[0])}, {number(sharpe[1])}] | [{percent(delta[0])}, {percent(delta[1])}] | [{percent(bh[0])}, {percent(bh[1])}] |")
    lines.extend(["", "区间使用20交易日配对移动区块的循环bootstrap，2000次；不要求每年显著，也不把区块当成新增独立历史。", "", "## 年度结果与贡献", "",
                  "| 年份 | 基础买持 | 基础M0 | 基础M1 | M1减M0 | M1减买持 | M1净损益(元) | 压力M1 |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
    annual = bundle["annual"]
    for year in sorted(annual.year.unique()):
        rows = annual.loc[annual.year.eq(year)].set_index(["scenario", "model"])
        m1 = rows.loc[("BASE", "M1")]
        lines.append(f"| {year} | {percent(rows.loc[('BASE','BUY_HOLD'),'net_return'])} | {percent(rows.loc[('BASE','M0'),'net_return'])} | {percent(m1.net_return)} | {percent(m1.return_minus_M0)} | {percent(m1.return_minus_BUY_HOLD)} | {m1.pnl_cny:,.2f} | {percent(rows.loc[('STRESS','M1'),'net_return'])} |")
    lines.extend(["", "2026年为截至8月14日开盘的部分年度。年度净损益可按初始本金相加核对全期财富变化，年度收益率不可直接相加。", "", "## 冻结裁决", ""])
    for key, value in result["retention_conditions"].items():
        lines.append(f"- `{key}`：{value}。")
    lines.extend([f"- 基础及压力两种情景的历史净夏普点估计均达到1.2：{result['historical_point_estimate_sharpe_1_2_in_both_costs']}。",
                  f"- 是否拟合最终全历史参数：{result['final_full_history_models_fitted']}。", "",
                  "未满足保留条件时，本表征停止，不修改20日窗口、五日目标、方向、岭惩罚、风险偏好或交易规则。任何正向信息都按真实证据等级保留，不能由失败推出所有日内隔夜研究均无效。", "",
                  "## 可核对文件", "",
                  "`comparison.csv`为同一账户口径比较表，`annual_contributions.csv`为年度全表，`conditional_samples.csv`为样本与训练资格，`models.json`为训练时冻结参数，`ledgers/`、`trades/`和`decisions/`分别保存日账、假想成交和收盘时确定的份额请求。`freeze_manifest.json`、`run_claim.json`与`receipt.json`记录一次性冻结运行。", "",
                  "实际持仓未知，当前模型ABSTAIN、目标UNSET，未连接券商、未产生真实订单或Paper/Shadow信号。未来标签仅用于冻结模型评分。", "",
                  "[上交所ETF常见问题](https://www.sse.com.cn/assortment/fund/etf/question/)支持本轮T+1、100份和0.001元价位约束。真实开盘成交深度没有日线证据，不能把本回放当成已经成交的业绩。", ""])
    return "\n".join(lines)


def freeze_files(config: dict) -> list[Path]:
    documents = [
        "docs/510300_INTRADAY_OVERNIGHT_INCREMENT_V1_AUTHORIZATION_20260905.md",
        "docs/510300_INTRADAY_OVERNIGHT_INCREMENT_V1_PROTOCOL_20260905.md",
        "docs/510300_INTRADAY_OVERNIGHT_INCREMENT_V1_USER_TEXT_20260905.txt",
        "tests/test_intraday_overnight_increment_v1.py",
    ]
    return [CONFIG, Path(__file__).resolve(), ROOT / config["authorization"]] + [ROOT / p for p in documents] + [ROOT / p for p in config["inputs"].values()]


def verify_freeze(output: Path) -> dict:
    manifest = json.loads((output / "freeze_manifest.json").read_text(encoding="utf-8"))
    for relative, expected in manifest["files"].items():
        require(digest(ROOT / relative) == expected["sha256"], f"冻结文件漂移：{relative}")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="510300日内—隔夜条件增量的一次性历史研究")
    parser.add_argument("command", choices=["preflight", "freeze", "run"])
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    output = ROOT / config["output"]
    output.mkdir(parents=True, exist_ok=True)
    authority = json.loads((ROOT / config["authorization"]).read_text(encoding="utf-8"))
    require(config["study_id"] in authority["authorized_historical_studies"], "当前权限不包含本轮研究")
    require(not authority["live_trading_authorized"], "研究不能升级实盘权限")
    if args.command in {"preflight", "freeze"}:
        require(not (output / "run_claim.json").exists(), "历史运行已领用，不能重新预检或冻结")
        require(not (output / "freeze_manifest.json").exists(), "已经冻结，禁止覆盖清单")
        data, dividends, report = load_inputs(config)
        write_json(output / "data_preflight.json", report)
        schedule = sample_schedule(data, config, with_labels=False, dividends=dividends)
        schedule.to_csv(output / "schedule_before_outcomes.csv", index=False, encoding="utf-8-sig")
        if args.command == "freeze":
            test_evidence = output / "synthetic_tests.txt"
            require(test_evidence.exists() and "passed" in test_evidence.read_text(encoding="utf-8"), "缺少冻结前合成测试结果")
            files = freeze_files(config) + [output / "data_preflight.json", test_evidence]
            manifest = {"study_id": config["study_id"], "frozen_at": now(), "new_labels_read_before_freeze": False,
                        "new_models_fit_before_freeze": False, "new_portfolios_before_freeze": False,
                        "history_previously_observed_by_project": True, "run_limit": 1,
                        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                        "files": {p.relative_to(ROOT).as_posix(): {"bytes": p.stat().st_size, "sha256": digest(p)} for p in files}}
            write_json(output / "freeze_manifest.json", manifest, exclusive=True)
        print(json.dumps({"步骤": args.command, "状态": report["status"], "样本日": report["research_days"], "计划原点": len(schedule), "新结果读取": False}, ensure_ascii=False))
        return 0
    manifest = verify_freeze(output)
    write_json(output / "run_claim.json", {"claimed_at": now(), "study_id": config["study_id"], "freeze_sha256": digest(output / "freeze_manifest.json"), "run_limit_consumed": 1}, exclusive=True)
    try:
        data, dividends, _ = load_inputs(config)
        bundle = evaluate(data, dividends, config)
        write_json(output / "models.json", bundle["models"])
        if bundle["final_models"] is not None:
            write_json(output / "final_full_history_models.json", {"fit_cutoff": config["data_cutoff"], "future_labels_only_for_scoring": True, "models": bundle["final_models"]})
        bundle["samples"].to_csv(output / "conditional_samples.csv", index=False, encoding="utf-8-sig")
        bundle["comparison"].to_csv(output / "comparison.csv", index=False, encoding="utf-8-sig")
        bundle["annual"].to_csv(output / "annual_contributions.csv", index=False, encoding="utf-8-sig")
        for category in ["ledgers", "trades", "decisions"]:
            (output / category).mkdir(exist_ok=True)
            for key, frame in bundle[category].items():
                frame.to_csv(output / category / f"{key}.csv", index=False, encoding="utf-8-sig")
        features = data.loc[data.date >= pd.Timestamp(config["train_origin_start"]), ["date", "dividend", "overnight_log", "intraday_log", "total_log", "identity_error", "M20", "D20", "RV20"]]
        features.to_csv(output / "feature_accounting.csv", index=False, encoding="utf-8-sig")
        write_json(output / "result.json", bundle["result"])
        (output / "REPORT.md").write_text(render_report(bundle, config), encoding="utf-8")
        verify_freeze(output)
        files = sorted(p for p in output.rglob("*") if p.is_file() and p.name not in {"receipt.json", "failure.json"})
        receipt = {"study_id": config["study_id"], "completed_at": now(), "status": bundle["result"]["status"],
                   "freeze_at": manifest["frozen_at"], "freeze_sha256": digest(output / "freeze_manifest.json"),
                   "historical_runs_consumed": 1, "evaluation_complete": True, "position_impact": 0,
                   "output_files": {p.relative_to(output).as_posix(): {"bytes": p.stat().st_size, "sha256": digest(p)} for p in files}}
        write_json(output / "receipt.json", receipt, exclusive=True)
        print(json.dumps({"状态": bundle["result"]["status"], "训练样本": bundle["result"]["training_observations"], "评价样本": bundle["result"]["evaluation_observations"], "报告": str(output / "REPORT.md")}, ensure_ascii=False))
        return 0
    except Exception as exc:
        write_json(output / "failure.json", {"failed_at": now(), "status": "PROGRAM_FAILED_RUN_CONSUMED", "error": str(exc), "repeat_without_versioned_correction_allowed": False})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
