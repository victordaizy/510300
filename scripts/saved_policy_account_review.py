"""从已保存账户核对经济结果和请求，不重新模拟、优化或扩大研究范围。"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from research.adaptive_allocation_v1 import normalize_dividends, summarize
from research.intraday_overnight_increment_v1 import now, require
from scripts.finalize_round66_20260907 import saved_cycles

ROOT = Path(__file__).resolve().parents[1]


def review_accounts(cfg, result, out, primary, expected_targets):
    data = pd.read_parquet(ROOT / cfg["features"])
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    metrics, differences, cycles_all, groups, no_views = [], [], [], [], []
    origin_count = 0
    for period, key in [("evaluation", "all_metrics"), ("earlier_diagnostic", "earlier_diagnostics")]:
        for cost in cfg["costs"]:
            folder = out / period / cost
            accounts = {}
            for record in result[key]:
                if record["cost"] != cost:
                    continue
                ledger = pd.read_parquet(folder / f"{record['model']}_ledger.parquet")
                accounts[record["model"]] = ledger
                actual = summarize(ledger, cfg)
                for field in ["net_sharpe", "annualized_return", "max_drawdown", "cumulative_return", "mean_exposure", "commission", "slippage_cost"]:
                    require((actual[field] is None and record[field] is None) or
                        (actual[field] is not None and record[field] is not None and abs(actual[field] - record[field]) < 1e-10), "保存账户指标不能复算")
                metrics.append({"period": period, "cost": cost, "model": record["model"], "net_sharpe": actual["net_sharpe"]})
            current = accounts[primary]
            require(np.allclose(current.cash + current.shares * current.mark + current.dividend_receivable, current.equity, atol=1e-6, rtol=0), "新账户资产合计与净值不符")
            previous = np.r_[cfg["initial_capital"], current.equity.to_numpy()[:-1]]
            require(np.allclose(current.equity / previous - 1, current.net_return, atol=1e-13, rtol=0), "每日收益与净值不符")
            for control in ["REARM_RIDGE", "PANIC_LEARNED_HALF", "BUY_HOLD"]:
                old = accounts[control]
                require(pd.DatetimeIndex(old.date).equals(pd.DatetimeIndex(current.date)), "新账户与对照日历不同")
                delta = {"period": period, "cost": cost, "control": control,
                    "terminal_nav_difference": float(current.equity.iloc[-1] - old.equity.iloc[-1]),
                    "price_difference": float(current.price_pnl.sum() - old.price_pnl.sum()),
                    "dividend_difference": float(current.dividend_recognized.sum() - old.dividend_recognized.sum()),
                    "commission_difference": float(current.commission.sum() - old.commission.sum()),
                    "slippage_difference": float(current.slippage_cost.sum() - old.slippage_cost.sum())}
                delta["identity_error"] = delta["terminal_nav_difference"] - delta["price_difference"] - delta["dividend_difference"] + delta["commission_difference"] + delta["slippage_difference"]
                require(abs(delta["identity_error"]) < 1e-6, "账户经济差额没有核对完全")
                differences.append(delta)
            decisions = pd.read_parquet(folder / f"{primary}_decisions.parquet")
            by_date = current.set_index("date")
            for row in decisions.itertuples():
                t = int(row.origin_index)
                target = expected_targets[t]
                require(row.origin == data.date.iloc[t] and row.execution_date == data.date.iloc[t + 1], "原点和下一开盘执行错位")
                require((np.isnan(target) and np.isnan(row.reference_weight)) or target == row.reference_weight, "请求目标不同于独立计算目标")
                account = by_date.loc[row.origin] if row.origin in by_date.index else None
                shares = int(account.shares) if account is not None else 0
                nav = float(account.equity) if account is not None else cfg["initial_capital"]
                close, request = float(data.close.iloc[t]), 0
                if np.isfinite(target):
                    desired = math.floor(target * nav / close / cfg["lot"]) * cfg["lot"]
                    if target != 0 and shares > 0 and abs(target - shares * close / nav) < cfg["weight_band"]:
                        desired = shares
                    request = desired - shares
                require(request == row.requested_quantity, "请求份额与账户预算或无观点规则不一致")
                if np.isnan(target):
                    execution = by_date.loc[row.execution_date]
                    no_views.append({"period": period, "cost": cost, "origin": row.origin, "execution_date": row.execution_date,
                        "shares_before": int(execution.shares_before), "filled_quantity": int(execution.filled_quantity),
                        "shares_after": int(execution.shares), "terminal": execution.mark_clock == "OPEN_TERMINAL"})
                origin_count += 1
            cycles = saved_cycles(current, dividends, cfg)
            cycles_all.extend({"period": period, "cost": cost, **cycle} for cycle in cycles)
            groups.append({"period": period, "cost": cost, "cycles": len(cycles), "positive_cycles": sum(x["net_profit"] > 0 for x in cycles),
                "negative_cycles": sum(x["net_profit"] < 0 for x in cycles), "net_profit": sum(x["net_profit"] for x in cycles),
                "gross_price_profit": sum(x["gross_price_profit"] for x in cycles), "dividend_recognized": sum(x["dividend_recognized"] for x in cycles),
                "commission": sum(x["commission"] for x in cycles), "slippage": sum(x["slippage"] for x in cycles),
                "held_closes": sum(x["held_closes"] for x in cycles), "mean_held_closes": float(np.mean([x["held_closes"] for x in cycles])) if cycles else None,
                "additional_buys": int((current.shares_before.gt(0) & current.filled_quantity.gt(0)).sum()),
                "initial_target": float(decisions.reference_weight.iloc[0]), "last_decision_target": float(decisions.reference_weight.iloc[-1])})
    for name, rows in [("saved_metrics_recomputation.csv", metrics), ("saved_account_differences.csv", differences),
        ("saved_actual_cycles.csv", cycles_all), ("saved_cycle_profit_groups.csv", groups)]:
        pd.DataFrame(rows).to_csv(out / name, index=False, encoding="utf-8-sig")
    pd.DataFrame(no_views, columns=["period", "cost", "origin", "execution_date", "shares_before", "filled_quantity", "shares_after", "terminal"]).to_csv(out / "saved_no_view_requests.csv", index=False, encoding="utf-8-sig")
    return {"verified_at": now(), "status": "SAVED_ACTUAL_ACCOUNT_ECONOMICS_AND_REQUESTS_RECONCILED", "recomputed_metrics": len(metrics),
        "account_differences": len(differences), "replayed_decision_origins": origin_count, "actual_cycles": len(cycles_all), "no_view_records": len(no_views),
        "new_account_simulations": 0, "new_model_fits": 0, "security_audit_performed": False}, groups
