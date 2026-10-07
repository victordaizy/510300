"""T11日期代理账户：固定信号、逐日风险、完整现金与分红账本。"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


POLICIES = ("FULL", "FINANCIAL_ALL", "FINANCIAL_COMMON", "ABOVE_MEDIAN")
BENCHMARKS = ("BUY_HOLD_50", "CASH")
COSTS = {"BASE": {"commission": .0002, "minimum": 5., "slippage": .0005},
         "STRESS": {"commission": .0004, "minimum": 5., "slippage": .001}}


def signal_table(o02, companies, market, features):
    """保留四个预定方向；未知反应或财务值不能成为FULL信号。"""
    index = {pd.Timestamp(d): i for i, d in enumerate(market.date)}
    assert market.date.equals(features.date)
    records = []
    for row in o02.itertuples(index=False):
        if row.date not in index:
            continue
        i = index[row.date]
        positive = bool(row.cohort_known and pd.notna(row.positive_financial_measurement)
                        and row.positive_financial_measurement)
        common = bool(positive and row.condition_known and row.reference_count >= 2
                      and np.isfinite([row.first_response, row.reference_median]).all())
        full = bool(common and row.first_response <= row.reference_median)
        group = companies.loc[companies.date.eq(row.date)].sort_values("ts_code")
        if positive:
            assert len(group) and group.known.eq(True).all(), "正向信号的完整公司篮子缺失"
            assert group.ts_code.is_unique
        anchors = tuple((str(r.ts_code), pd.Timestamp(r.report_period)) for r in group.itertuples())
        records.append({"date": row.date, "observation_idx": i, "signal_id": row.date.strftime("%Y%m%d"),
                        "FULL": full, "FINANCIAL_ALL": positive, "FINANCIAL_COMMON": common,
                        "ABOVE_MEDIAN": bool(common and not full), "frozen_stop": float(features.low_w.iloc[i]),
                        "first_response": row.first_response, "reference_count": row.reference_count,
                        "reference_median": row.reference_median, "measurement_status": row.status,
                        "company_count": len(anchors), "anchors": anchors})
    return records


def fact_events(frontier, companies, market):
    """使用完整报告前沿；后续报告缺字段或离开成分范围仍可触发未知退出。"""
    dates = {pd.Timestamp(d): i for i, d in enumerate(market.date)}
    score = {str(r.announcement_id): r for r in companies.itertuples(index=False)}
    events, audit = {}, []
    for row in frontier.itertuples(index=False):
        if row.available_date not in dates:
            continue
        value = score.get(str(row.announcement_id))
        known = bool(value is not None and value.known
                     and np.isfinite([value.L02, value.L04_change, value.L04_industry_z]).all())
        adverse = bool(not known or value.L02 <= 0 or value.L04_change < 0 or value.L04_industry_z < 0)
        event = {"observation_idx": dates[row.available_date], "date": row.available_date,
                 "ts_code": str(row.ts_code), "report_period": pd.Timestamp(row.report_period),
                 "announcement_id": str(row.announcement_id), "measurement_known": known, "adverse": adverse,
                 "L02": float(value.L02) if value is not None else np.nan,
                 "L04_change": float(value.L04_change) if value is not None else np.nan,
                 "L04_industry_z": float(value.L04_industry_z) if value is not None else np.nan}
        events.setdefault(event["observation_idx"], []).append(event)
        audit.append(event)
    return events, pd.DataFrame(audit)


def simulate(market, features, dividends, signals, facts, policy, lag, capital, cost_name, start, end,
             engine, risk_target):
    """开盘只成交已知条件，收盘更新风险；完整保留现金日及未成交请求。"""
    assert policy in POLICIES + BENCHMARKS
    assert lag in (1, 2)
    assert market.date.equals(features.date)
    indices = np.flatnonzero(market.date.between(start, end))
    assert len(indices) > 1 and indices[0] >= 2
    first, last = int(indices[0]), int(indices[-1])
    account = engine.Account(float(capital))
    cost, config = COSTS[cost_name], {"lot": 100, "tick": .001, "limit_fraction": .1}
    scheduled = {int(s["observation_idx"]) + lag: s for s in signals if s.get(policy, False)}
    events = dividends.to_dict("records")
    previous_equity, previous_mark, previous_reserve, peak = float(capital), float(market.close.iloc[first-1]), 0., float(capital)
    entry_idx, entry_signal_idx, stop_level, active_id = None, None, np.nan, None
    anchors, seen_periods = {}, {}
    stopped, benchmark_entered, pending_reason = False, False, None
    ledgers, decisions, orders, fact_checks = [], [], [], []
    for i in indices:
        row, f = market.iloc[i], features.iloc[i-1]
        day, op, close, old = row.date, float(row.open), float(row.close), account.shares
        recognized, paid = 0., 0.
        for k, event in enumerate(events):
            if event["ex_date"] == day:
                amount = account.entitlements.get(k, 0) * event["cash_dividend_per_share"]
                if amount:
                    account.receivables[k] = amount
                    recognized += amount
            if event["payment_date"] < day and k in account.receivables:
                amount = account.receivables.pop(k)
                account.cash += amount
                paid += amount
        reference = float(market.close.iloc[i-1] - row.dividend)
        signal = scheduled.get(int(i))
        quantity, reason, trigger = 0, "无预定买入信号", None
        price_invalid, time_due, fact_invalid = False, False, False
        if policy == "BUY_HOLD_50":
            if not benchmark_entered:
                quantity = math.floor(.5 * account.value(reference) / reference / 100) * 100
                reason = "期初半仓买入持有"
        elif policy != "CASH":
            if old:
                for event in facts.get(int(i)-lag, []):
                    issuer = event["ts_code"]
                    if issuer not in anchors or event["report_period"] <= seen_periods[issuer]:
                        continue
                    seen_periods[issuer] = event["report_period"]
                    fact_checks.append({"execution_date": day, "execution_idx": int(i),
                                        "entry_signal_id": active_id, "lag": lag, **event})
                    if event["adverse"]:
                        fact_invalid = True
                        trigger = event["announcement_id"]
                price_invalid = bool(float(f.wealth) < stop_level)
                time_due = bool(i >= entry_idx + 20)
                if pending_reason is None:
                    if stopped:
                        pending_reason = "账户回撤停机"
                    elif fact_invalid:
                        pending_reason = "后续财报反证或未知"
                    elif price_invalid:
                        pending_reason = "跌破观察日财富低点"
                    elif time_due:
                        pending_reason = "二十个开盘间隔到期"
                if pending_reason is not None:
                    quantity, reason = -old, pending_reason
                else:
                    target = min(old, risk_target(engine, account, reference, peak, float(f.es95)))
                    quantity, reason = target-old, "风险预算只减仓"
                    if target == 0:
                        pending_reason = reason = "风险预算归零持续退出"
            elif stopped:
                reason = "账户已停机不再入场"
            elif signal is not None:
                price_invalid = bool(float(f.wealth) < signal["frozen_stop"])
                if price_invalid:
                    reason = "延迟入场前价格条件已失效"
                else:
                    quantity = risk_target(engine, account, reference, peak, float(f.es95))
                    reason = "预定信号开盘请求" if quantity else "风险预算不足未入场"
        before_gap = quantity
        if quantity > 0 and policy != "BUY_HOLD_50":
            quantity = min(quantity, risk_target(engine, account, op, peak, float(f.es95)))
            if quantity == 0:
                reason = "开盘风险约束取消买入"
        sellable = account.sellable(int(i))
        execution = engine.execute_order(account, quantity, op, float(row.previous_close), float(row.dividend),
                                         int(i), cost, config)
        if execution["filled_quantity"] > 0 and old == 0:
            entry_idx, benchmark_entered = int(i), True
            if policy not in BENCHMARKS:
                entry_signal_idx = int(signal["observation_idx"])
                stop_level, active_id = float(signal["frozen_stop"]), signal["signal_id"]
                anchors = dict(signal["anchors"])
                seen_periods = anchors.copy()
                pending_reason = None
        if quantity:
            orders.append({"date": day, "idx": int(i), "origin": market.date.iloc[i-1],
                           "signal_id": active_id if old else signal["signal_id"] if signal else None,
                           "reason": reason, "sellable_before": sellable, **execution})
        for k, event in enumerate(events):
            if event["payment_date"] == day and k in account.receivables:
                amount = account.receivables.pop(k)
                account.cash += amount
                paid += amount
            if event["record_date"] == day:
                account.entitlements[k] = account.shares
        reserve = 0.
        if i == last and account.shares:
            exit_price = engine.fill_price(close, -1, COSTS["STRESS"], .001)
            reserve = account.shares * (close-exit_price) + engine.commission(account.shares, exit_price, COSTS["STRESS"])
        equity = account.value(close)-reserve
        price_pnl = old*(op-previous_mark) + account.shares*(close-op)
        error = equity-previous_equity-price_pnl-recognized + execution["commission"] + execution["slippage_cost"] + reserve-previous_reserve
        assert abs(error) < 1e-6, f"账户恒等式异常：{day} {error}"
        account.assert_valid()
        peak = max(peak, equity)
        drawdown = 1-equity/peak
        if policy not in BENCHMARKS:
            stopped |= drawdown >= .1
        ledgers.append({"date": day, "idx": int(i), "policy": policy, "open": op, "mark": close,
                        "cash": account.cash, "shares": account.shares, "dividend_receivable": account.receivable(),
                        "terminal_exit_reserve": reserve, "equity": equity, "net_return": equity/previous_equity-1,
                        "price_pnl": price_pnl, "dividend_recognized": recognized, "dividend_paid": paid,
                        "exposure": account.shares*close/equity, "accounting_error": error, "drawdown": drawdown,
                        "risk_stopped": stopped, "sellable_before": sellable, "entry_signal_id": active_id,
                        "terminal_unliquidated": bool(i == last and account.shares), **execution})
        decisions.append({"date": day, "idx": int(i), "origin": market.date.iloc[i-1], "risk_origin_idx": int(i)-1,
                          "signal_scheduled": signal is not None, "scheduled_signal_id": signal["signal_id"] if signal else None,
                          "scheduled_observation_idx": signal["observation_idx"] if signal else None,
                          "scheduled_signal_ignored_while_held": bool(signal is not None and old),
                          "es95_5d": float(f.es95), "reason": reason, "pre_open_request": before_gap,
                          "requested_quantity": quantity, "filled_quantity": execution["filled_quantity"],
                          "entry_idx": entry_idx, "entry_signal_idx": entry_signal_idx, "entry_signal_id": active_id,
                          "frozen_stop": stop_level, "price_invalid": price_invalid, "time_due": time_due,
                          "new_fact_invalid": fact_invalid, "trigger_announcement_id": trigger,
                          "exit_pending": pending_reason, "risk_stopped": stopped})
        if account.shares == 0:
            entry_idx, entry_signal_idx, active_id, pending_reason = None, None, None, None
            anchors, seen_periods, stop_level = {}, {}, np.nan
        previous_equity, previous_mark, previous_reserve = equity, close, reserve
    order_columns = ["date", "idx", "origin", "signal_id", "reason", "sellable_before", "requested_quantity",
                     "filled_quantity", "open_price", "fill_price", "commission", "slippage_cost", "notional",
                     "status", "cash_before", "shares_before", "cash_after", "shares_after"]
    fact_columns = ["execution_date", "execution_idx", "entry_signal_id", "lag", "observation_idx", "date", "ts_code",
                    "report_period", "announcement_id", "measurement_known", "adverse", "L02", "L04_change", "L04_industry_z"]
    return (pd.DataFrame(ledgers), pd.DataFrame(decisions), pd.DataFrame(orders).reindex(columns=order_columns),
            pd.DataFrame(fact_checks).reindex(columns=fact_columns))


def paired_uncertainty(ledgers, repetitions=4000, seed=20260927, block=20):
    """预定配对区块，现金日包含在样本中；不据区间选择方向。"""
    arrays = {k: v.net_return.to_numpy(float) for k, v in ledgers.items()}
    assert set(arrays) == {"FULL", "FINANCIAL_COMMON", "ABOVE_MEDIAN"}
    for ledger in ledgers.values():
        assert ledger.date.equals(ledgers["FULL"].date)
    rng, n = np.random.default_rng(seed), len(arrays["FULL"])
    values = {"FULL_sharpe": [], "FULL_cagr": []}
    for comparator in ("FINANCIAL_COMMON", "ABOVE_MEDIAN"):
        values[f"FULL_minus_{comparator}_sharpe"] = []
        values[f"FULL_minus_{comparator}_cagr"] = []
    for _ in range(repetitions):
        starts = rng.integers(0, n, size=math.ceil(n/block))
        indices = ((starts[:, None] + np.arange(block)) % n).reshape(-1)[:n]
        metrics = {}
        for key, r in arrays.items():
            sample = r[indices]
            sd = sample.std(ddof=1)
            metrics[key] = (np.sqrt(242)*sample.mean()/sd if sd > 1e-15 else np.nan,
                            np.expm1(np.log1p(sample).sum()*242/n))
        values["FULL_sharpe"].append(metrics["FULL"][0])
        values["FULL_cagr"].append(metrics["FULL"][1])
        for comparator in ("FINANCIAL_COMMON", "ABOVE_MEDIAN"):
            values[f"FULL_minus_{comparator}_sharpe"].append(metrics["FULL"][0]-metrics[comparator][0])
            values[f"FULL_minus_{comparator}_cagr"].append(metrics["FULL"][1]-metrics[comparator][1])
    result = {"repetitions": repetitions, "seed": seed, "block_days": block,
              "interpretation": "历史配对日收益重采样；没有校正全部家族搜索，也不是独立前向验证。"}
    for key, raw in values.items():
        a = np.asarray(raw)
        finite = a[np.isfinite(a)]
        result[key] = {"interval_95": np.quantile(finite, [.025, .975]).tolist() if len(finite) else [None, None],
                       "finite_samples": len(finite)}
    return result
