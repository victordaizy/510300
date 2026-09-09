"""复用连续目标账户的必要保存核对，支持部分买卖和再次进入。"""
import numpy as np
import pandas as pd
from research.adaptive_allocation_v1 import summarize
from research.intraday_overnight_increment_v1 import require
from scripts.finalize_round60_20260907 import metric
from scripts.finalize_round96_20260908 import value_equal
from scripts.review_round74_saved import saved_cycles


def verify_saved_target_accounts(out, cfg, result, data, dividends, expected_targets, comparison_models=()):
    """策略只提供独立目标；共用核对不重新运行账户或模型。"""
    primary = cfg["primary"]
    accounts, all_cycles, differences, decision_count = [], [], [], 0
    for period in ["evaluation", "earlier_diagnostic"]:
        start, end = (cfg["evaluation_start"], cfg["data_cutoff"]) if period == "evaluation" else (cfg["earlier_start"], cfg["earlier_terminal"])
        frame = data[data.date.le(end)]
        first = int(np.flatnonzero(frame.date.ge(start))[0])
        indices = np.arange(first-1, len(frame)-1)
        for cost_id in cfg["costs"]:
            folder = out / period / cost_id
            ledger = pd.read_parquet(folder / f"{primary}_ledger.parquet")
            decisions = pd.read_parquet(folder / f"{primary}_decisions.parquet")
            full_targets = np.asarray(expected_targets(period, cost_id, frame, start), dtype=float)
            require(full_targets.shape == (len(frame),), "独立目标没有保留完整准备日历")
            targets = full_targets[indices]
            require(pd.DatetimeIndex(ledger.date).equals(pd.DatetimeIndex(frame.date.iloc[first:])), "目标账户交易日历不同")
            require(np.array_equal(decisions.origin_index, indices), "目标账户判断收盘日历不同")
            require(pd.DatetimeIndex(decisions.origin).equals(pd.DatetimeIndex(frame.date.iloc[indices])), "目标账户判断时点不同")
            require(pd.DatetimeIndex(decisions.execution_date).equals(pd.DatetimeIndex(ledger.date)), "目标账户未在下一开盘执行")
            np.testing.assert_allclose(decisions.reference_weight, targets, atol=1e-12, rtol=0, equal_nan=True)
            prior = np.r_[cfg["initial_capital"], ledger.equity.iloc[:-1]]
            old_shares = np.r_[0, ledger.shares.iloc[:-1]].astype(int)
            prices = frame.close.iloc[indices].to_numpy(float)
            known = np.isfinite(targets)
            desired = old_shares.copy()
            desired[known] = (np.floor(targets[known]*prior[known]/prices[known]/cfg["lot"])*cfg["lot"]).astype(int)
            within_band = known & (targets > 0) & (old_shares > 0) & (np.abs(targets-old_shares*prices/prior) < cfg["weight_band"])
            desired[within_band] = old_shares[within_band]
            requests = desired-old_shares
            require(np.array_equal(decisions.requested_quantity, requests), "目标账户请求未使用自身净值和既有带宽")
            execution_requests = requests.copy(); execution_requests[-1] = -old_shares[-1]
            require(np.array_equal(ledger.requested_quantity, execution_requests), "目标账户执行请求或终点全退不同")
            require(np.array_equal(ledger.shares, old_shares+ledger.filled_quantity), "目标账户份额未按实际成交衔接")
            require(ledger.shares.iloc[-1] == 0 and ledger.mark.iloc[-1] == ledger.open.iloc[-1], "目标账户终点未按开盘清仓")
            np.testing.assert_allclose(ledger.cash+ledger.shares*ledger.mark+ledger.dividend_receivable, ledger.equity, atol=1e-6, rtol=0)
            np.testing.assert_allclose(ledger.equity/prior-1, ledger.net_return, atol=1e-12, rtol=0)
            np.testing.assert_allclose(ledger.equity-prior, ledger.price_pnl+ledger.dividend_recognized-ledger.commission-ledger.slippage_cost, atol=1e-6, rtol=0)
            measured, stored = summarize(ledger, cfg), metric(result, primary, period, cost_id)
            for key in ["net_sharpe", "annualized_return", "annualized_arithmetic_mean", "annualized_volatility", "max_drawdown", "cumulative_return", "trade_count", "commission", "slippage_cost", "mean_exposure"]:
                require(value_equal(measured[key], stored[key]), "目标账户保存绩效不能复算")
            cycles = saved_cycles(ledger, dividends, cfg)
            net = float(ledger.equity.iloc[-1]-cfg["initial_capital"])
            require(abs(sum(c["net_profit"] for c in cycles)-net) < 1e-5, "实际完整周期利润不能还原总账户")
            all_cycles.extend({"period": period, "cost": cost_id, **c} for c in cycles)
            accounts.append({"period": period, "cost": cost_id, "net_sharpe": measured["net_sharpe"], "cycles": len(cycles),
                "holding_closes": int(ledger.shares.gt(0).sum()), "gross_price_dividend_profit": float(ledger.price_pnl.sum()+ledger.dividend_recognized.sum()),
                "commission_and_slippage": float(ledger.commission.sum()+ledger.slippage_cost.sum()), "net_profit": net})
            for model in comparison_models:
                previous = pd.read_parquet(folder / f"{model}_ledger.parquet")
                old_metric = summarize(previous, cfg)
                differences.append({"period": period, "cost": cost_id, "comparison": model,
                    "terminal_equity_difference": float(ledger.equity.iloc[-1]-previous.equity.iloc[-1]),
                    **{f"{k}_difference": float(ledger[k].sum()-previous[k].sum()) for k in ["price_pnl", "dividend_recognized", "commission", "slippage_cost"]},
                    "sharpe_difference": measured["net_sharpe"]-old_metric["net_sharpe"]})
            decision_count += len(decisions)
    for name, records in [("saved_account_checks.csv", accounts), ("saved_actual_cycles.csv", all_cycles), ("saved_comparison_differences.csv", differences)]:
        pd.DataFrame(records).to_csv(out / name, index=False, encoding="utf-8-sig")
    return accounts, all_cycles, differences, decision_count
