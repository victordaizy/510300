"""分红日历毛收益初筛通过后的单账户兑现检验，固定份额、期限和风控。"""
from __future__ import annotations

import argparse
from copy import deepcopy
import math
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from research import mechanism_odds_open_contract_v1 as shared
from research import constituent_dividend_calendar_v1 as signal

ROOT = shared.ROOT
OUT = signal.OUT / "account_stage"
PRIMARY = "DIVIDEND_CALENDAR"
POLICIES = [PRIMARY, "WEEKLY_ALL", "BUY_HOLD_50", "CASH"]
START, END, HOLD = "2017-01-03", "2025-12-31", 20
save, read, now, digest = shared.save, shared.read, shared.now, shared.digest


def risk_series(market, labels, neighbors):
    """只从已保存、当时已成熟的近邻读取五日尾部，不使用其方向预测。"""
    assert len(market) == len(labels) == len(neighbors)
    values = np.full(len(market), np.nan)
    rows = []
    dates = pd.DatetimeIndex(market.date)
    for i, ids in enumerate(neighbors):
        if ids[0] < 0:
            continue
        assert len(ids) == 126 and len(np.unique(ids)) == 126
        assert (ids + 6 <= i).all(), "风险场景尚未成熟。"
        assert (dates[ids] >= dates[i] - pd.DateOffset(years=2)).all(), "风险场景超过两年窗口。"
        sample = labels[ids, 4]
        assert np.isfinite(sample).all()
        values[i] = max(0., -float(np.sort(sample)[:math.ceil(.05 * len(ids))].mean()))
        rows.append({"idx": i, "date": dates[i], "es95": values[i], "neighbors": len(ids),
                     "latest_maturity_idx": int(ids.max() + 6), "earliest_origin_date": dates[ids.min()]})
    return values, pd.DataFrame(rows)


def simulate(market, dividends, risk, origins, policy, capital, cost_name, e):
    account = e.Account(float(capital))
    cost = shared.COSTS[cost_name]
    ids = np.flatnonzero(market.date.between(START, END))
    first, last = int(ids[0]), int(ids[-1])
    rows = list(market.itertuples(index=False))
    events = dividends.to_dict("records")
    peak, previous_equity = float(capital), float(capital)
    previous_mark, previous_reserve = float(rows[first - 1].close), 0.
    expiry, stopped, exit_pending, entered = None, False, False, False
    plan = {"quantity": 0, "limit_price": None, "reason": "首日尚无前收盘计划", "decision_idx": first - 1}
    ledger, orders, decisions = [], [], []
    for i in ids:
        i = int(i)
        row, day = rows[i], rows[i].date
        old, sellable_before = account.shares, account.sellable(i)
        recognized, paid = 0., 0.
        for k, event in enumerate(events):
            if event["ex_date"] == day:
                amount = account.entitlements.get(k, 0) * event["cash_dividend_per_share"]
                if amount:
                    account.receivables[k] = amount
                    recognized += amount
            if event["payment_date"] < day and k in account.receivables:
                value = account.receivables.pop(k)
                account.cash += value
                paid += value
        assert plan["decision_idx"] < i
        fill = shared.execute(e, account, plan, row, i, cost)
        assert fill["filled_quantity"] >= -sellable_before
        if fill["filled_quantity"] > 0 and old == 0:
            expiry, entered = i + HOLD, True
        if plan["quantity"]:
            orders.append({"date": day, "idx": i, **plan, **fill})
        if account.shares == 0:
            expiry, exit_pending = None, False
        for k, event in enumerate(events):
            if event["payment_date"] == day and k in account.receivables:
                value = account.receivables.pop(k)
                account.cash += value
                paid += value
            if event["record_date"] == day:
                account.entitlements[k] = account.shares
        reserve = 0.
        if i == last and account.shares:
            px = e.fill_price(row.close, -1, cost, shared.TICK)
            reserve = account.shares * (row.close - px) + e.commission(account.shares, px, cost)
        equity = account.value(row.close) - reserve
        price_pnl = old * (row.open - previous_mark) + account.shares * (row.close - row.open)
        error = equity - previous_equity - price_pnl - recognized + fill["commission"] + fill["slippage_cost"] + reserve - previous_reserve
        assert abs(error) < 1e-6, (day, error)
        account.assert_valid()
        peak = max(peak, equity)
        dd = 1 - equity / peak
        if policy in [PRIMARY, "WEEKLY_ALL"]:
            stopped |= dd >= .1
        ledger.append({"date": day, "idx": i, "equity": equity, "net_return": equity / previous_equity - 1,
                       "shares": account.shares, "cash": account.cash, "dividend_receivable": account.receivable(),
                       "dividend_recognized": recognized, "dividend_paid": paid, "mark": row.close,
                       "exposure": account.shares * row.close / equity, "drawdown": dd, "risk_stopped": stopped,
                       "terminal_exit_reserve": reserve, "accounting_error": error, "decision_idx": plan["decision_idx"],
                       "sellable_before": sellable_before, "locked_after": account.shares - account.sellable(i),
                       "holding_expiry_idx": expiry, **fill})
        previous_equity, previous_mark, previous_reserve = equity, row.close, reserve
        if i == last:
            continue
        reference = float(row.close - rows[i + 1].dividend)
        es = float(risk[i])
        origin = origins.get(i)
        source_ready = origin is not None and origin["information_status"] == "READY"
        trigger = source_ready and (policy == "WEEKLY_ALL" or bool(origin["selected"]))
        plan = {"quantity": 0, "limit_price": None, "reason": "继续持有或等待周原点", "decision_idx": i,
                "decision_date": day, "reference_price": reference, "equity_at_decision": equity,
                "peak_at_decision": peak, "es95": es, "expiry_idx": expiry,
                "source_ready_week": source_ready, "entry_trigger": trigger}
        if policy == "BUY_HOLD_50":
            if not entered:
                plan.update(quantity=math.floor(.5 * equity / reference / shared.LOT) * shared.LOT, reason="半仓持有背景")
        elif policy in [PRIMARY, "WEEKLY_ALL"]:
            if account.shares:
                exit_pending |= stopped or i + 1 >= expiry
                if exit_pending:
                    plan.update(quantity=-account.shares, reason="回撤永久停止" if stopped else "原20日期限到达")
                elif np.isfinite(es):
                    target = min(account.shares, shared.risk_quantity(e, equity, account.cash, peak, reference, es, account.shares))
                    plan.update(quantity=target - account.shares, reason="按事前风险预算减仓" if target < account.shares else "持有至原期限，忽略新信号")
            elif not stopped and trigger and np.isfinite(es):
                quantity = shared.risk_quantity(e, equity, account.cash, peak, reference, es)
                plan.update(quantity=quantity, reason="固定日历信号买入" if quantity else "风险预算不足一手")
            elif stopped:
                plan["reason"] = "回撤后永久停止"
            elif trigger:
                plan["reason"] = "NO_VIEW：条件风险样本不足"
        decisions.append(deepcopy(plan))
    return pd.DataFrame(ledger), pd.DataFrame(orders), pd.DataFrame(decisions)


def freeze():
    assert not (OUT / "freeze.json").exists(), "账户版本已冻结。"
    assert read(signal.OUT / "result.json")["status"] == "PASS_GROSS_SCREEN_ACCOUNT_REQUIRED"
    assert read(OUT / "prefreeze_tests.json")["exit_code"] == 0
    assert digest(signal.OUT / "inputs/market.parquet") == digest(shared.OUT / "inputs/market.parquet")
    for rel in ["inputs", "code"]:
        (OUT / rel).mkdir(exist_ok=True)
    for name in ["market.parquet", "dividends.csv", "dividend_coverage.json"]:
        shutil.copy2(shared.OUT / "inputs" / name, OUT / "inputs" / name)
    shutil.copy2(shared.OUT / "forecasts.npz", OUT / "inputs/risk_cache.npz")
    shutil.copy2(shared.OUT / "dividend_source_clock.json", OUT / "inputs/dividend_source_clock.json")
    shutil.copy2(shared.OUT / "code/account_engine.py", OUT / "code/account_engine.py")
    source = pd.read_parquet(signal.OUT / "events.parquet")
    source[["date", "idx", "information_status", "selected"]].to_parquet(OUT / "inputs/signal_schedule.parquet", index=False)
    protocol = {**read(OUT / "design_registration.json"), "frozen_at": now(),
                "signal_source": str((signal.OUT / "result.json").relative_to(ROOT)),
                "risk_cache_source": str((shared.OUT / "forecasts.npz").relative_to(ROOT)),
                "signal_labels_not_used_in_orders": True}
    save(OUT / "protocol.json", protocol)
    paths = [Path(__file__), ROOT / "tests/test_constituent_dividend_calendar_account_v1.py",
             Path(shared.__file__), OUT / "design_registration.json", OUT / "protocol.json", OUT / "prefreeze_tests.json",
             signal.OUT / "result.json", signal.OUT / "freeze.json", *list((OUT / "inputs").glob("*")), *list((OUT / "code").glob("*"))]
    save(OUT / "freeze.json", {"at": now(), "files": [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in paths]})
    print("完整账户实现已冻结，原日历信号与已存条件风险序列分别固定。", flush=True)


def run():
    for row in read(OUT / "freeze.json")["files"]:
        assert digest(ROOT / row["path"]) == row["sha256"], row["path"]
    save(OUT / "run_started.json", {"at": now(), "expected_accounts": 16})
    market = pd.read_parquet(OUT / "inputs/market.parquet")
    market.date = pd.to_datetime(market.date)
    market = market.loc[market.date.le(END)].reset_index(drop=True)
    e = shared.engine(OUT / "code/account_engine.py")
    dividends = e.normalize_dividends(pd.read_csv(OUT / "inputs/dividends.csv"))
    np.testing.assert_allclose(market.dividend, market.date.map(dividends.set_index("ex_date").cash_dividend_per_share).fillna(0.), rtol=0, atol=1e-12)
    cache = np.load(OUT / "inputs/risk_cache.npz")
    es, risk_receipts = risk_series(market, cache["labels"], cache["neighbors"])
    risk_receipts.to_parquet(OUT / "risk_input_receipts.parquet", index=False)
    origins = pd.read_parquet(OUT / "inputs/signal_schedule.parquet").set_index("idx").to_dict("index")
    records, yearly, periods, ledgers = [], [], [], {}
    checked_rows, maximum_identity_error = 0, 0.
    for capital in [200000, 20000]:
        for cost in shared.COSTS:
            for policy in POLICIES:
                ledger, orders, decisions = simulate(market, dividends, es, origins, policy, capital, cost, e)
                folder = OUT / "accounts" / f"{capital}_{cost}_{policy}"
                folder.mkdir(parents=True)
                ledger.to_parquet(folder / "ledger.parquet", index=False)
                orders.to_parquet(folder / "orders.parquet", index=False)
                decisions.to_parquet(folder / "decisions.parquet", index=False)
                trades = shared.cycles(ledger)
                trades.to_csv(folder / "cycles.csv", index=False, encoding="utf-8-sig")
                metric = shared.metrics(ledger, capital)
                np.testing.assert_allclose(trades.pnl.sum(), metric["net_profit"], rtol=0, atol=1e-6)
                metric.update(capital=capital, cost=cost, policy=policy, closed_cycles=int(trades.closed.sum()),
                              largest_winner=trades.pnl.max(), worst_cycle=trades.pnl.min(),
                              largest_cycle_share_of_total_profit=trades.pnl.max() / metric["net_profit"] if len(trades) and metric["net_profit"] > 0 else None)
                records.append(metric)
                for name, start, end in [(str(y), f"{y}-01-01", f"{y}-12-31") for y in range(2017, 2026)] + [("2017-2020", START, "2020-12-31"), ("2021-2025", "2021-01-01", END)]:
                    part = ledger.loc[ledger.date.between(start, end)]
                    initial = float(part.equity.iloc[0] / (1 + part.net_return.iloc[0]))
                    rec = {"capital": capital, "cost": cost, "policy": policy, "period": name, **shared.metrics(part, initial)}
                    (periods if "-" in name else yearly).append(rec)
                # 读回保存账本核对金额、全部日收益、T+1和事前股数，不重新跑策略。
                saved = pd.read_parquet(folder / "ledger.parquet")
                np.testing.assert_allclose(saved.equity, saved.cash + saved.shares * saved.mark + saved.dividend_receivable - saved.terminal_exit_reserve, rtol=0, atol=1e-7)
                assert (saved.filled_quantity >= -saved.sellable_before).all() and (saved.cash >= -1e-7).all()
                assert (saved.decision_idx < saved.idx).all()
                plans = decisions.set_index("decision_idx")
                assert all(rec.requested_quantity == plans.loc[rec.idx - 1, "quantity"] for rec in saved.iloc[1:].itertuples())
                recomputed = shared.metrics(saved, capital)
                for key, value in recomputed.items():
                    if value is not None:
                        np.testing.assert_allclose(metric[key], value, rtol=1e-10, atol=1e-9)
                checked_rows += len(saved)
                maximum_identity_error = max(maximum_identity_error, metric["maximum_identity_error"])
                if capital == 200000 and cost == "STRESS":
                    ledgers[policy] = ledger
            print(f"已完成{capital}元{cost}的四个完整账户。", flush=True)
    table = pd.DataFrame(records)
    table["joint_historical_target"] = table.net_sharpe.ge(1.2) & table.calendar_cagr.ge(.1) & table.max_drawdown.le(.1)
    table.to_csv(OUT / "metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(yearly).to_csv(OUT / "annual_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(periods).to_csv(OUT / "period_metrics.csv", index=False, encoding="utf-8-sig")
    primary = table.loc[table.capital.eq(200000) & table.cost.eq("STRESS") & table.policy.eq(PRIMARY)].iloc[0]
    small = table.loc[table.capital.eq(20000) & table.cost.eq("STRESS") & table.policy.eq(PRIMARY)].iloc[0]
    # 保存给定历史路径的区块区间，同时披露相同持有风控的周原点对照差。
    values = ledgers[PRIMARY].net_return.to_numpy(float)
    n = len(values)
    rng = np.random.default_rng(20260928)
    index = ((rng.integers(0, n, size=(2000, math.ceil(n / 20)))[:, :, None] + np.arange(20)) % n).reshape(2000, -1)[:, :n]
    draws = values[index]
    sds = draws.std(axis=1, ddof=1)
    sharpes = np.divide(np.sqrt(242) * draws.mean(axis=1), sds, out=np.full(2000, np.nan), where=sds > 1e-15)
    excess = values - ledgers["WEEKLY_ALL"].net_return.to_numpy(float)
    save(OUT / "uncertainty.json", {"primary_sharpe_95_interval": np.nanquantile(sharpes, [.025, .975]).tolist(),
         "annualized_arithmetic_excess_vs_weekly_all": float(excess.mean() * 242),
         "paired_excess_95_interval": np.quantile(excess[index].mean(axis=1) * 242, [.025, .975]).tolist(),
         "method": "20交易日循环区块2000次，未校正历史研究选择偏差"})
    passed = bool(primary.joint_historical_target)
    result = {"at": now(), "study_id": "510300_CONSTITUENT_DIVIDEND_CALENDAR_ACCOUNT_V1",
              "status": "HISTORICAL_POINT_TARGET_PASSED_INDEPENDENT_VALIDATION_REQUIRED" if passed else "REJECTED_ACCOUNT_TARGET_FAILED_NO_PARAMETER_RESCUE",
              "gross_screen_status": read(signal.OUT / "result.json")["status"], "primary": primary.to_dict(), "small_account": small.to_dict(),
              "new_accounts": len(table), "qualified_candidates": 0, "historical_point_passing_scenarios": int(table.joint_historical_target.sum()),
              "new_model_fits": 0, "parameter_searches": 0, "independent_forward_observations": 0,
              "historical_first_vintage_verified": False, "strict_M05_validated": False, "goal_achieved": False, "orders_authorized": False}
    save(OUT / "result.json", result)
    save(OUT / "verification_receipt.json", {"at": now(), "status": "PASS_SAVED_ACCOUNT_CLOCK_CASH_AND_METRICS",
         "accounts": len(table), "saved_ledger_rows_checked": checked_rows, "maximum_account_identity_error": maximum_identity_error,
         "risk_origins_rechecked": len(risk_receipts), "new_accounts_in_verification": 0,
         "boundary": "账本和时序核对不是独立前向验证或真实成交证明。"})
    print(f"分红日历完整账户完成：20万元压力夏普{primary.net_sharpe:.6f}，日历年化{primary.calendar_cagr:.4%}，回撤{primary.max_drawdown:.4%}。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="分红日历固定完整账户检验")
    parser.add_argument("action", choices=["freeze", "run"])
    {"freeze": freeze, "run": run}[parser.parse_args().action]()
