"""只读取第75轮保存内容，复算因果因子、时点请求、净值和含分红周期。"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

from research.afternoon_entry_v1_output_fix import ROOT, OUT, CONFIG, PRIMARY, PARENT
from research.adaptive_allocation_v1 import normalize_dividends, summarize
from research.intraday_overnight_increment_v1 import digest, now, require, write_json
from scripts.review_round74_saved import saved_cycles


def known_quantity(cash, price, cost, cfg):
    execution_price = np.ceil(price * (1 + cost["slippage"]) / cfg["tick"] - 1e-10) * cfg["tick"]
    quantity = int(np.floor(cash / execution_price / cfg["lot"]) * cfg["lot"])
    while quantity > 0 and quantity * execution_price + max(cost["minimum"], quantity * execution_price * cost["commission"]) > cash + 1e-10:
        quantity -= cfg["lot"]
    return quantity


def main():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    result = json.loads((OUT / "result.json").read_text(encoding="utf-8"))
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "下午进入固定来源改变")
    for item in cfg["preserved_files"]:
        require(digest(OUT / item["path"]) == item["sha256"], "前三账户或已有因子未按原字节保留")
    data = pd.read_parquet(ROOT / cfg["features"])
    dates = pd.DatetimeIndex(data.date)
    div = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    intraday = np.log((data.close + data.dividend) / (data.open + data.dividend))
    overnight = np.log((data.open + data.dividend) / data.previous_close)
    difference = intraday - overnight
    factor = difference.rolling(60).sum() / (difference.rolling(60).std(ddof=1) * np.sqrt(60))
    entry = (factor > 1) & (factor.shift() > 1) & data.feature_valid.fillna(False)
    snapshot = pd.read_parquet(OUT / "minute_snapshots.parquet")
    partial = pd.read_csv(OUT / "all_partial_entry_factors.csv", parse_dates=["date"])
    factor_checks = []
    for row in partial.itertuples():
        t = dates.get_loc(row.date)
        price, opening, dividend, previous = snapshot.loc[row.date, "signal_price"], data.open.iloc[t], data.dividend.iloc[t], data.close.iloc[t-1]
        night = np.log((opening + dividend) / previous)
        current = np.log((price + dividend) / (opening + dividend)) - night
        values = np.r_[difference.iloc[t-59:t].to_numpy(), current]
        expected = values.sum() / (values.std(ddof=1) * np.sqrt(60))
        error = max(abs(expected-row.partial_d60), abs(factor.iloc[t-1]-row.previous_d60))
        require(error < 1e-11, "下午因子与原始已知价格计算不符")
        require(row.entry_signal == bool(expected > 1 and factor.iloc[t-1] > 1), "下午进入阈值不符")
        factor_checks.append({"date": row.date, "error": error, "partial_entry_signal": row.entry_signal,
            "full_close_entry": bool(entry.iloc[t]), "partial_signal_disappeared_at_close": bool(row.entry_signal and not entry.iloc[t])})
    partial_by_date = partial.set_index("date")
    metrics, differences, cycles_all, groups, requests, account_checks = [], [], [], [], [], []
    predictions = 0
    models = json.loads((ROOT / cfg["saved_models"]).read_text(encoding="utf-8"))["models"]["D60_INTRA__RIDGE"]
    model_by_date = {pd.Timestamp(data.date.iloc[m["fit_index"]]): m for m in models}
    for period, key in [("evaluation", "all_metrics"), ("earlier_diagnostic", "earlier_diagnostics")]:
        for cost_id, cost in cfg["costs"].items():
            folder = OUT / period / cost_id
            candidate = pd.read_parquet(folder / f"{PRIMARY}_ledger.parquet")
            decisions = pd.read_parquet(folder / f"{PRIMARY}_decisions.parquet")
            afternoon = pd.read_parquet(folder / "afternoon_decisions.parquet")
            for row in afternoon.itertuples():
                requested = int(row.requested_quantity)
                if requested:
                    t = dates.get_loc(row.date)
                    require(row.entry_rearmed and row.days_since_exit >= cfg["specification"]["cooldown"], "下午请求未满足原资格和冷却")
                    require(bool(partial_by_date.loc[row.date, "entry_signal"]), "下午请求没有当时的进入信号")
                    px = snapshot.loc[row.date, "signal_price"]
                    require(requested == known_quantity(row.cash_at_signal, px, cost, cfg), "下午请求使用了未来执行价或错误费用")
                    require(row.observation_label + pd.Timedelta(minutes=1) < row.signal_time < row.execution_label - pd.Timedelta(minutes=1), "下午观察决定执行时钟不符")
                    actual = candidate[candidate.date.eq(row.date)].iloc[0]
                    if row.filled_quantity:
                        require(row.filled_quantity == actual.filled_quantity and actual.shares_before == 0 and actual.opening_filled_quantity == 0, "下午成交没有映射到真实空仓账户")
                        require(requested <= snapshot.loc[row.date, "execution_volume"] * cfg["minute_participation_cap"], "超出分钟容量仍成交")
                    else:
                        require(row.execution_status == "UNFILLED_MINUTE_CAPACITY", "本轮未成交原因出现未解释变化")
                    requests.append({"period": period, "cost": cost_id, "date": row.date, "requested_quantity": requested,
                        "filled_quantity": row.filled_quantity, "status": row.execution_status, "original_entry_at_close": bool(entry.iloc[t]),
                        "partial_signal_disappeared_at_close": not bool(entry.iloc[t])})
            held = decisions[decisions.learning_status.eq("PREDICTION_AVAILABLE")]
            for row in held.itertuples():
                stored = model_by_date[pd.Timestamp(row.learning_fit_origin)]
                require(stored["fit_index"] <= row.origin_index and stored["latest_exit_index"] <= stored["fit_index"], "保存预测使用未来模型或标签")
                columns = ["log_holding_days", "cycle_return", "cycle_drawdown", "entry_mode", "mom5", "mom20", "sma120", "vol20"]
                values = np.array([getattr(row, x) for x in columns])
                model = stored["model"]
                z = np.clip((values-np.array(model["mean"]))/np.array(model["scale"]), -5, 5)
                estimate = float(model["intercept"] + np.sum(z * np.array(model["coefficients"])))
                require(abs(estimate-row.continuation_prediction) < 1e-12, "保存日终学习预测不符")
                predictions += 1
            cs = saved_cycles(candidate, div, cfg)
            for c in cs:
                c.update(period=period, cost=cost_id)
            cycles_all += cs
            groups.append({"period": period, "cost": cost_id, "cycles": len(cs), "positive_cycles": sum(c["net_profit"] > 0 for c in cs),
                "negative_cycles": sum(c["net_profit"] < 0 for c in cs), "net_profit": sum(c["net_profit"] for c in cs),
                "price_pnl": float(candidate.price_pnl.sum()), "dividend": float(candidate.dividend_recognized.sum()),
                "commission": float(candidate.commission.sum()), "slippage": float(candidate.slippage_cost.sum()),
                "held_closes": int(candidate.shares.gt(0).sum())})
            for m in [x for x in result[key] if x["cost"] == cost_id]:
                ledger = pd.read_parquet(folder / f"{m['model']}_ledger.parquet")
                nav = ledger.cash + ledger.shares * ledger.mark + ledger.dividend_receivable
                prior = np.r_[cfg["initial_capital"], ledger.equity.iloc[:-1].to_numpy()]
                np.testing.assert_allclose(nav, ledger.equity, atol=1e-6, rtol=0)
                np.testing.assert_allclose(nav/prior-1, ledger.net_return, atol=1e-12, rtol=0)
                np.testing.assert_allclose(nav-prior, ledger.price_pnl+ledger.dividend_recognized-ledger.commission-ledger.slippage_cost, atol=1e-6, rtol=0)
                recomputed = summarize(ledger, cfg)
                for k in ["net_sharpe", "cumulative_return", "annualized_return", "max_drawdown", "mean_exposure", "trade_count", "commission", "slippage_cost"]:
                    require(abs(recomputed[k]-m[k]) < 1e-9, "完整账户汇总与保存结果不符")
                metrics.append({"period": period, "cost": cost_id, "model": m["model"], **recomputed})
                if m["model"] != PRIMARY:
                    d = {"period": period, "cost": cost_id, "control": m["model"], "terminal_equity_difference": float(candidate.equity.iloc[-1]-ledger.equity.iloc[-1])}
                    d.update({f"{c}_difference": float(candidate[c].sum()-ledger[c].sum()) for c in ["price_pnl", "dividend_recognized", "commission", "slippage_cost"]})
                    require(abs(d["terminal_equity_difference"]-(d["price_pnl_difference"]+d["dividend_recognized_difference"]-d["commission_difference"]-d["slippage_cost_difference"])) < 1e-6, "新旧经济差额未对平")
                    differences.append(d)
            first = int(np.flatnonzero(data.date >= pd.Timestamp(cfg["evaluation_start"] if period=="evaluation" else cfg["earlier_start"]))[0])
            previous_mark = data.close.iloc[first-1]
            for row in candidate.itertuples():
                # 只有下午真正成交时分割日内价格区间；无成交请求不制造敞口。
                af = row.execution_clock == "AFTERNOON_LATER_BAR_OPEN_PROXY" and row.filled_quantity != 0
                if af:
                    expected_pnl = row.shares_before * (row.open-previous_mark) + row.shares * (row.mark-row.open_price)
                else:
                    expected_pnl = row.shares_before * (row.open-previous_mark) + row.shares * (row.mark-row.open)
                require(abs(expected_pnl-row.price_pnl) < 1e-6, "下午真实持仓分段价格收益不符")
                previous_mark = row.mark
            account_checks.append({"period": period, "cost": cost_id, "account_days": len(candidate), "decision_origins": len(decisions), "afternoon_checks": len(afternoon),
                "original_fallback_exact": period=="earlier_diagnostic", "terminal_liquidated": not bool(candidate.terminal_unliquidated.iloc[-1])})
    for name, rows in [("saved_factor_replay.csv", factor_checks), ("saved_metrics_recomputation.csv", metrics), ("saved_account_differences.csv", differences),
        ("saved_actual_cycles.csv", cycles_all), ("saved_cycle_profit_groups.csv", groups), ("saved_afternoon_requests.csv", requests), ("saved_account_checks.csv", account_checks)]:
        pd.DataFrame(rows).to_csv(OUT / name, index=False, encoding="utf-8-sig")
    receipt = {"verified_at": now(), "status": "SAVED_AFTERNOON_FACTORS_EXECUTION_AND_FULL_ACCOUNT_RECONCILED", "recomputed_account_records": len(metrics),
        "account_differences": len(differences), "partial_factor_days": len(factor_checks), "maximum_factor_error": max(x["error"] for x in factor_checks),
        "afternoon_requests": len(requests), "daily_saved_predictions": predictions, "complete_cycles": len(cycles_all), "account_days": sum(x["account_days"] for x in account_checks),
        "new_diagnostic_accounts": 0, "new_models": 0, "security_audit_performed": False, "reviewer_sha256": digest(Path(__file__)),
        "cycle_helper_sha256": digest(ROOT / "scripts/review_round74_saved.py")}
    write_json(OUT / "saved_verification_receipt.json", receipt, exclusive=True)
    print(json.dumps({"核对": receipt, "周期与经济": groups, "原账户差额": [d for d in differences if d['control']=='REARM_RIDGE']}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
