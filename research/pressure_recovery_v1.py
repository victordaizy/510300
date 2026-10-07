"""510300 价格让步与压力恢复的纯测量函数；不连接交易或自动采集。"""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from statistics import mean

import numpy as np

CN = timezone(timedelta(hours=8))


def timestamp(value: str) -> datetime:
    result = datetime.fromisoformat(str(value))
    if result.tzinfo is None:
        raise ValueError("时间必须包含时区，禁止默认补成已校准经济时间")
    return result.astimezone(CN)


def finite(value: object, positive: bool = False) -> float:
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        raise ValueError("数值缺失、非有限数或不满足正值要求")
    return result


def session(value: datetime) -> str:
    clock = value.strftime("%H:%M:%S")
    if "09:30:00" <= clock < "11:30:00":
        return "AM"
    if "13:00:00" <= clock < "14:57:00":
        return "PM"
    return "OTHER"


def book_features(bids: list, asks: list, quantity: int, band_bps: float = 10) -> dict:
    """数量单位为份；只返回可见盘口范围内的价格与深度。"""
    if not isinstance(quantity, int) or quantity <= 0 or quantity % 100:
        raise ValueError("研究买入数量必须是正的100份整数倍")
    if not bids or not asks:
        raise ValueError("缺少买卖盘口")
    for side in (bids, asks):
        for price, size in side:
            finite(price, True)
            if finite(size) < 0:
                raise ValueError("申报数量不能为负")
    if any(bids[i][0] <= bids[i + 1][0] for i in range(len(bids) - 1)):
        raise ValueError("买盘价格必须逐档递减")
    if any(asks[i][0] >= asks[i + 1][0] for i in range(len(asks) - 1)):
        raise ValueError("卖盘价格必须逐档递增")
    bid, ask = float(bids[0][0]), float(asks[0][0])
    if bid >= ask:
        raise ValueError("连续竞价盘口锁定或交叉，不能解释为常规价差")
    mid = (bid + ask) / 2
    lower, upper = mid * (1 - band_bps / 1e4), mid * (1 + band_bps / 1e4)
    bid_depth = sum(p * q for p, q in bids if p >= lower)
    ask_depth = sum(p * q for p, q in asks if p <= upper)
    bid_qty, ask_qty = sum(q for _, q in bids), sum(q for _, q in asks)
    remaining, cost = quantity, 0.0
    for price, available in asks:
        take = min(remaining, available)
        cost += take * price
        remaining -= take
    acquired = quantity - remaining
    return {
        "mid": mid, "spread_bps": 1e4 * (ask - bid) / mid,
        "bid_visible_depth_cny": bid_depth, "ask_visible_depth_cny": ask_depth,
        "fixed_band_fully_visible": bids[-1][0] <= lower and asks[-1][0] >= upper,
        "obi_visible": (bid_qty - ask_qty) / (bid_qty + ask_qty) if bid_qty + ask_qty else None,
        "quoted_available_quantity": acquired,
        "buy_vwap": cost / quantity if remaining == 0 else None,
        "full_quantity_visible": remaining == 0,
        "actual_fill_established": False,
    }


def concession_interval(price: float, reference_low: float, reference_high: float) -> dict:
    price, low, high = [finite(v, True) for v in (price, reference_low, reference_high)]
    if low > high:
        raise ValueError("参考估值区间颠倒")
    return {"concession_low_bps": 1e4 * (1 - price / low),
            "concession_high_bps": 1e4 * (1 - price / high)}


def point_in_time_errors(row: dict, deadline: str, close: bool = False) -> list[str]:
    """保守使用接收时间加时钟误差；网页外层时间不替代经济时间。"""
    errors = []
    try:
        cutoff = timestamp(deadline)
        price_time = timestamp(row["price_economic_at"])
        ref_time = timestamp(row["reference_economic_at"])
        received = timestamp(row["received_at"])
        reference_received = timestamp(row["reference_received_at"])
        clock = finite(row["clock_error_bound_seconds"])
        if not 0 <= clock <= 1:
            errors.append("CLOCK_ERROR_UNQUALIFIED")
        if max(received, reference_received) + timedelta(seconds=max(0, clock)) > cutoff:
            errors.append("LATE_INPUT")
        if price_time > cutoff or ref_time > cutoff:
            errors.append("FUTURE_ECONOMIC_TIME")
        if price_time > received + timedelta(seconds=max(0, clock)) or ref_time > reference_received + timedelta(seconds=max(0, clock)):
            errors.append("ECONOMIC_TIME_AFTER_RECEIPT")
        if abs((price_time - ref_time).total_seconds()) > 1:
            errors.append("ASYNCHRONOUS_REFERENCE")
        if price_time.date() != cutoff.date() or ref_time.date() != cutoff.date():
            errors.append("WRONG_ECONOMIC_DAY")
        if close:
            if price_time.strftime("%H:%M:%S") != "15:00:00" or ref_time.strftime("%H:%M:%S") != "15:00:00":
                errors.append("NOT_SYNCHRONIZED_CLOSE")
            if not row.get("close_components_confirmed"):
                errors.append("CLOSE_COMPONENTS_NOT_CONFIRMED")
        elif session(price_time) == "OTHER" or (cutoff - min(price_time, ref_time)).total_seconds() > 5:
            errors.append("STALE_OR_NONCONTINUOUS_QUOTE")
        if not row.get("reference_error_bound_evidence"):
            errors.append("VALUATION_ERROR_UNKNOWN")
        for field in ("raw_sha256", "reference_raw_sha256"):
            value = row.get(field, "")
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                errors.append("MISSING_RAW_IDENTITY")
        concession_interval(row["price"], row["reference_low"], row["reference_high"])
    except (KeyError, TypeError, ValueError, OverflowError):
        errors.append("MISSING_OR_INVALID_POINT_IN_TIME_FIELDS")
    return sorted(set(errors))


def m1_measure(row: dict) -> dict:
    day = str(row["trade_date"])
    cutoff = f"{day}T15:06:00+08:00"
    errors = point_in_time_errors(row, cutoff, close=True)
    if day < "2026-07-06":
        errors.append("BEFORE_AFTER_HOURS_ETF_REGIME")
    if errors:
        return {"status": "NO_VIEW", "reasons": errors, "concession_low_bps": None}
    result = concession_interval(row["price"], row["reference_low"], row["reference_high"])
    d = result["concession_low_bps"]
    bins = [-math.inf, 0, 5, 10, 20, 40, math.inf]
    result.update(status="MEASUREMENT_ONLY", bin_index=next(i for i in range(6) if bins[i] <= d < bins[i + 1]),
                  fill_status="UNKNOWN_NO_QUEUE_OR_ORDER_RECEIPTS", actual_order=False)
    return result


def detect_m2_events(rows: list[dict], baseline_days: int = 60) -> list[dict]:
    """输入为通过时点检查且整分钟对齐的可见盘口；过去同分钟日样本形成阈值。"""
    ordered = sorted(rows, key=lambda r: timestamp(r["decision_at"]))
    by_time = {}
    for row in ordered:
        t = timestamp(row["decision_at"])
        if t.second or t.microsecond:
            raise ValueError("M2规范输入必须为整分钟决策网格")
        if t in by_time:
            raise ValueError("同一决策时刻重复，需先根据原始回执去重")
        by_time[t] = row
    qualified = {}
    history = defaultdict(list)
    for t, row in by_time.items():
        if point_in_time_errors(row, row["decision_at"]) or not row.get("fixed_band_fully_visible"):
            continue
        lag = by_time.get(t - timedelta(minutes=5))
        if not lag or point_in_time_errors(lag, lag["decision_at"]):
            continue
        if session(t) != session(t - timedelta(minutes=5)) or session(t) == "OTHER":
            continue
        try:
            etf = math.log(finite(row["mid"], True) / finite(lag["mid"], True))
            basket = math.log(finite(row["reference_mid"], True) / finite(lag["reference_mid"], True))
            spread = finite(row["spread_bps"], True)
            depth = finite(row["bid_visible_depth_cny"], True)
        except (KeyError, TypeError, ValueError):
            continue
        regime = "POST_20260706" if t.date().isoformat() >= "2026-07-06" else "PRE_20260706"
        key = (regime, t.strftime("%H:%M"))
        past = [x for x in history[key] if x[0] < t.date()][-baseline_days:]
        observation = {"time": t, "etf_return": etf, "basket_return": basket,
                       "relative_return": etf - basket, "spread": spread, "depth": depth}
        if len(past) == baseline_days:
            reference = np.array([x[1:] for x in past], dtype=float)
            observation.update(return_q05=float(np.quantile(reference[:, 0], .05)),
                               spread_q90=float(np.quantile(reference[:, 1], .90)),
                               spread_median=float(np.median(reference[:, 1])),
                               depth_median=float(np.median(reference[:, 2])))
            observation["shock"] = etf < observation["return_q05"] and basket < 0 and spread > observation["spread_q90"]
            qualified[t] = observation
        history[key].append((t.date(), etf, spread, depth))
    events, active, quiet, previous = [], None, 0, None
    for t, row in qualified.items():
        if previous is None or t.date() != previous.date() or session(t) != session(previous):
            active, quiet = None, 0
        elif t - previous != timedelta(minutes=1):
            quiet = 0
        previous = t
        if row["shock"]:
            quiet = 0
            if active is None:
                endpoint = t + timedelta(minutes=5)
                after = qualified.get(endpoint)
                state = "NO_VIEW_NO_FIXED_ENDPOINT"
                if after is not None and session(endpoint) == session(t):
                    recovered = (after["spread"] <= row["spread_median"] and after["spread"] < row["spread"]
                                 and after["depth"] >= row["depth_median"] and after["depth"] > row["depth"]
                                 and after["relative_return"] > row["relative_return"])
                    state = "RECOVERED_MEASUREMENT_ONLY" if recovered else "NOT_RECOVERED"
                active = {"event_id": f"M2_{t:%Y%m%d_%H%M}", "shock_at": t.isoformat(),
                          "observation_end_at": endpoint.isoformat(), "state": state,
                          "shock_return": row["etf_return"], "shock_relative_return": row["relative_return"],
                          "shock_spread_bps": row["spread"], "shock_depth_cny": row["depth"],
                          "included_in_observation_rebound": True, "actual_order": False,
                          "entry_status": "NOT_MAPPED_NO_REMAINING_NET_SPACE_EVIDENCE"}
                events.append(active)
        elif active is not None:
            quiet += 1
            if quiet >= 10:
                active = None
    return events


def commission(notional: float, rate: float = .0002, minimum: float = 5) -> float:
    n = finite(notional)
    if n < 0 or rate < 0 or minimum < 0:
        raise ValueError("成交额与费用参数不能为负")
    return max(n * rate, minimum) if n > 0 else 0.0


def next_exit_request(day: str, trading_days: list[str]) -> str:
    days = sorted(set(trading_days))
    if day not in days:
        raise ValueError("入场日期不在官方交易日历")
    later = [d for d in days if d > day]
    if not later:
        raise ValueError("缺少下一交易日，禁止用自然日替代")
    return later[0] + "T09:35:00+08:00"


def settle_completed_trade(trade: dict, trading_days: list[str]) -> dict:
    """只对已经完整退出的证据记录结算；拒绝未知、未成交及T+1前退出。"""
    if trade.get("evidence_type") not in {"ACTUAL_BROKER_RECEIPTS", "INDEPENDENTLY_VALIDATED_REPLAY"}:
        raise ValueError("没有合格成交证据，不能生成净收益")
    if not trade.get("evidence_path"):
        raise ValueError("成交证据路径缺失")
    buy, sell = timestamp(trade["buy_at"]), timestamp(trade["sell_at"])
    request = timestamp(next_exit_request(buy.date().isoformat(), trading_days))
    if sell < request or trade.get("exit_request_at") != request.isoformat():
        raise ValueError("退出不符合下一交易日9:35请求合同")
    quantity = finite(trade["buy_quantity"], True)
    if quantity != finite(trade["sell_quantity"], True):
        raise ValueError("仍有未退出份额，应保留未结算状态")
    buy_cash = finite(trade["buy_vwap"], True) * quantity
    sell_cash = finite(trade["sell_vwap"], True) * quantity
    fees = finite(trade["buy_commission"]) + finite(trade["sell_commission"]) + finite(trade["other_fees"])
    if any(finite(trade[k]) < 0 for k in ("buy_commission", "sell_commission", "other_fees")):
        raise ValueError("费用不能为负")
    dividend = finite(trade["dividend_cash"])
    pnl = sell_cash + dividend - buy_cash - fees
    return {"trade_date": buy.date().isoformat(), "net_pnl_cny": pnl,
            "net_return": pnl / buy_cash, "buy_notional_cny": buy_cash,
            "gross_price_return": sell_cash / buy_cash - 1, "total_fees_cny": fees,
            "extra_price_slippage_deducted": False}


def expectancy(rows: list[dict]) -> dict:
    if not rows:
        return {"status": "NOT_COMPUTED_NO_COMPLETED_TRADES", "trades": 0,
                "win_rate": None, "payoff_ratio": None, "net_expectancy": None}
    returns = [finite(r["net_return"]) for r in rows]
    wins, losses = [r for r in returns if r > 0], [-r for r in returns if r < 0]
    p, p_loss = len(wins) / len(rows), len(losses) / len(rows)
    gain, loss = mean(wins) if wins else None, mean(losses) if losses else None
    decomposition = p * (gain or 0) - p_loss * (loss or 0)
    if not math.isclose(decomposition, mean(returns), abs_tol=1e-12):
        raise ValueError("净期望分解不相等")
    return {"status": "DESCRIPTIVE_COMPLETED_TRADES", "trades": len(rows),
            "win_rate": p, "loss_rate": p_loss, "flat_rate": returns.count(0) / len(rows),
            "mean_win": gain, "mean_loss_abs": loss,
            "payoff_ratio": gain / loss if gain is not None and loss is not None else None,
            "net_expectancy": decomposition, "worst_trade": min(returns)}


def clustered_expectancy_interval(rows: list[dict], replications: int = 5000) -> dict:
    clusters = defaultdict(list)
    for row in rows:
        clusters[row["trade_date"]].append(finite(row["net_return"]))
    if len(clusters) < 30:
        return {"status": "NOT_COMPUTED_INSUFFICIENT_INDEPENDENT_DAYS", "independent_days": len(clusters), "lower": None, "upper": None}
    sums = np.array([sum(v) for v in clusters.values()])
    counts = np.array([len(v) for v in clusters.values()])
    rng = np.random.default_rng(510300)
    estimates = []
    for _ in range(replications):
        selected = rng.integers(0, len(clusters), len(clusters))
        estimates.append(float(sums[selected].sum() / counts[selected].sum()))
    # 三个首要比较的保守区间；仍不替代独立样本或序列相关诊断。
    tail = .05 / (2 * 3)
    return {"status": "DAY_CLUSTER_BONFERRONI_3", "independent_days": len(clusters),
            "lower": float(np.quantile(estimates, tail)), "upper": float(np.quantile(estimates, 1-tail))}
