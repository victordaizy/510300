"""整窗已知现金单独处理，缺失证据和其余退化仍保持无观点。"""
import numpy as np
import pandas as pd
from research.intraday_overnight_increment_v1 import require
from research.tangency_reference_budget_inputs_v1 import tangency_budget


def observed_cash_days(ledger, dates, first):
    dates = pd.DatetimeIndex(dates)
    require(pd.DatetimeIndex(ledger.date).equals(dates[first:]), "现金证据没有覆盖完整参考日历")
    fields = ledger[["shares", "filled_quantity", "net_return", "cash", "dividend_receivable", "equity"]].to_numpy(float)
    require(np.isfinite(fields).all(), "参考现金证据有缺失")
    require((ledger.equity-ledger.cash-ledger.shares*ledger.mark-ledger.dividend_receivable).abs().max() < 1e-6, "现金证据与原账户权益不符")
    result = np.full(len(dates), np.nan)
    result[first:] = (ledger.shares.eq(0) & ledger.filled_quantity.eq(0) & ledger.net_return.eq(0)).to_numpy(float)
    return result


def risk_support_budget(mean, covariance, previous, confirmed_cash):
    mean, cov = np.asarray(mean, float), np.asarray(covariance, float)
    flags = np.asarray(confirmed_cash)
    require(flags.shape == (2,), "现金支路证据形状错误")
    proven = flags == 1
    for k in np.flatnonzero(proven):
        require(mean[k] == 0 and (cov[k] == 0).all() and (cov[:, k] == 0).all(), "整窗现金证据与收益矩矛盾")
    weights, status, score = tangency_budget(mean, cov, previous)
    if status != "NO_VIEW_NONPOSITIVE_DEFINITE_COVARIANCE_KEEP_BUDGET" or proven.sum() != 1:
        return weights, status, score
    active = int(np.flatnonzero(~proven)[0])
    if not np.isfinite(mean[active]) or mean[active] <= 0 or not np.isfinite(cov[active, active]) or cov[active, active] <= 0:
        return weights, status, score
    # 与零收益现金混合时，任何正比例的历史夏普相同；并列时最大化历史均值，选择全部策略预算。
    selected = np.zeros(2)
    selected[active] = 1.
    return selected, "KNOWN_CASH_SINGLE_RISK_BUDGET_AVAILABLE", float(mean[active]/np.sqrt(cov[active, active]))


def budget_frame(dates, reference_returns, expert_states, confirmed_cash_days, first, window=242):
    dates = pd.DatetimeIndex(dates)
    returns, states = np.asarray(reference_returns, float), np.asarray(expert_states, float)
    require(dates.is_monotonic_increasing and not dates.has_duplicates and returns.shape == states.shape == (len(dates), 2), "切点参考日历或形状不符")
    require(1 <= first < len(dates)-1 and isinstance(window, int) and window >= 2 and (np.isnan(states) | np.isin(states, [0., 1.])).all(), "切点起点、窗口或状态不符")
    cash_days = np.asarray(confirmed_cash_days, float)
    require(cash_days.shape == returns.shape and (np.isnan(cash_days) | np.isin(cash_days, [0., 1.])).all(), "现金证据日历或取值错误")
    cash_support = np.array([False, False])
    weights = np.array([.5, .5])
    mean, cov, score = np.full(2, np.nan), np.full((2, 2), np.nan), float("nan")
    status, attempted, successful = "INITIAL_EQUAL_BUDGET_NO_MOMENT_ESTIMATE", pd.NaT, pd.NaT
    rows, count, left = [], 0, pd.NaT
    for t in range(len(dates)):
        outside = t < first-1 or t == len(dates)-1
        scheduled = not outside and t >= first and dates[t].to_period("M") != dates[t-1].to_period("M")
        if scheduled:
            attempted, count = dates[t], min(window, t-first+1)
            left = dates[t-count+1]
            values = returns[t-count+1:t+1]
            mean, cov, score = np.full(2, np.nan), np.full((2, 2), np.nan), float("nan")
            cash_support = np.all(cash_days[t-count+1:t+1] == 1., axis=0)
            if count < window:
                status = "NO_VIEW_WARMUP_KEEP_BUDGET"
            elif not np.isfinite(values).all() or (values <= -1).any():
                status = "NO_VIEW_INCOMPLETE_WINDOW_KEEP_BUDGET"
            else:
                mean, cov = values.mean(axis=0), np.cov(values, rowvar=False, ddof=1)
                weights, status, score = risk_support_budget(mean, cov, weights, cash_support)
                if status in ["TANGENCY_BUDGET_AVAILABLE", "KNOWN_CASH_SINGLE_RISK_BUDGET_AVAILABLE", "EXPLICIT_CASH_NO_POSITIVE_PAST_MEAN"]:
                    successful = dates[t]
        require(np.isfinite(weights).all() and (weights >= 0).all() and weights.sum() <= 1+1e-12, "切点预算超出自身资金")
        target = float("nan")
        if not outside:
            if weights.sum() == 0:
                target = 0.
            elif np.isfinite(states[t]).all():
                target = float(states[t]@weights)
        rows.append({"date": dates[t], "panic_budget": float("nan") if outside else weights[0], "learned_budget": float("nan") if outside else weights[1],
            "panic_confirmed_cash_window": bool(cash_support[0]), "learned_confirmed_cash_window": bool(cash_support[1]),
            "budget_status": "NO_VIEW_OUTSIDE_DECISION_PERIOD" if outside else status, "budget_update_scheduled": scheduled, "moment_attempt_origin": attempted,
            "last_successful_budget_origin": successful, "window_start": left, "window_observations": count, "panic_mean": mean[0], "learned_mean": mean[1],
            "panic_variance": cov[0, 0], "learned_variance": cov[1, 1], "reference_covariance": cov[0, 1], "estimated_daily_ratio": score,
            "panic_state": states[t, 0], "learned_state": states[t, 1], "target": target, "panic_reference_return": returns[t, 0], "learned_reference_return": returns[t, 1]})
    return pd.DataFrame(rows)
