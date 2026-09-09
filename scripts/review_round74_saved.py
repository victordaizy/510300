"""复算波幅价格转向、完整成交周期和阶段差异，交付中文规则。"""
from __future__ import annotations

import json
from pathlib import Path
import math
import shutil

import numpy as np
import pandas as pd

from research.adaptive_allocation_v1 import normalize_dividends, summarize
from research.parabolic_reversal_v1 import ROOT, OUT as RESEARCH, CONFIG, PRIMARY
from research.intraday_overnight_increment_v1 import digest, now, require, write_json
from scripts.finalize_round60_20260907 import metric, table

INDEX = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
OUT = ROOT / "deliverables/510300真实波幅趋势价格转向_第66轮_20260907"
DOCUMENT = OUT / "真实波幅趋势价格转向_全部因子规则和历史表现.md"
NEXT_NOTE = ROOT / "docs/510300_AFTER_ATR_CONTINUOUS_TREND_20260907.md"


def saved_cycles(ledger, dividends, cfg):
    cycles, owners = [], {}
    active = None
    for row in ledger.itertuples():
        if row.filled_quantity > 0 and row.shares_before == 0:
            require(active is None, "保存账户周期重叠")
            active = {"cycle": len(cycles) + 1, "entry_date": row.date, "entry_origin": row.origin,
                "buy_debit": 0., "gross_price_profit": 0., "commission": 0., "slippage": 0.,
                "dividend_recognized": 0., "held_closes": 0, "buy_trades": 0, "sell_trades": 0}
        if row.filled_quantity != 0:
            require(active is not None, "实际成交没有归属周期")
            active["gross_price_profit"] -= row.filled_quantity * row.open_price
            active["commission"] += row.commission
            active["slippage"] += row.slippage_cost
            if row.filled_quantity > 0:
                active["buy_debit"] += row.filled_quantity * row.fill_price + row.commission
                active["buy_trades"] += 1
            else:
                active["sell_trades"] += 1
        if row.shares > 0:
            require(active is not None and row.mark_clock != "OPEN_TERMINAL", "未平仓终点不能算完成周期")
            active["held_closes"] += 1
            owners[row.date] = (active["cycle"], int(row.shares))
        if active is not None and row.filled_quantity < 0 and row.shares == 0:
            active.update(exit_date=row.date, exit_origin=row.origin, terminal_exit=row.mark_clock == "OPEN_TERMINAL")
            cycles.append(active)
            active = None
    require(active is None, "仍有未完成周期")
    dates = set(ledger.date)
    for event in dividends.itertuples():
        owner = owners.get(event.record_date)
        if owner is not None and event.ex_date in dates:
            number, shares = owner
            cycles[number - 1]["dividend_recognized"] += shares * event.cash_dividend_per_share
    for row in cycles:
        row["net_profit"] = row["gross_price_profit"] + row["dividend_recognized"] - row["commission"] - row["slippage"]
        row["cycle_net_return"] = row["net_profit"] / row["buy_debit"]
    require(abs(sum(x["net_profit"] for x in cycles) - (ledger.equity.iloc[-1] - cfg["initial_capital"])) < 1e-6, "周期利润含分红权利后仍不等于终值")
    require(abs(sum(x["dividend_recognized"] for x in cycles) - ledger.dividend_recognized.sum()) < 1e-6, "周期分红权利与实际应收不符")
    return cycles


def verify(cfg, result):
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "价格转向冻结来源改变")
    data = pd.read_parquet(ROOT / cfg["features"])
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    f = pd.read_parquet(RESEARCH / "factors.parquet")
    require(pd.DatetimeIndex(f.date).equals(pd.DatetimeIndex(data.date)), "价格转向因子日历改变")
    raw_return = (data.close + data.dividend) / data.previous_close - 1
    wealth = (1 + raw_return.fillna(0)).cumprod()
    for column, raw in [("wealth_high", data.high), ("wealth_low", data.low)]:
        expected = wealth.shift() * (raw + data.dividend) / data.previous_close
        expected.iloc[0] = raw.iloc[0] / data.close.iloc[0]
        np.testing.assert_allclose(expected, f[column], atol=2e-12, rtol=0)
    np.testing.assert_allclose(wealth, f.wealth_close, atol=2e-12, rtol=0)
    require(f.input_valid.all(), "当前保存原数据应完整，不能忽略缺失后做状态复算")
    require(pd.isna(f.target.iloc[0]) and f.target.iloc[1:].notna().all(), "首根准备及之后成熟状态不符")
    checks, sar_error_max = [], 0.
    for t in range(1, len(f)):
        high, low = float(f.wealth_high.iloc[t]), float(f.wealth_low.iloc[t])
        previous_high, previous_low = float(f.wealth_high.iloc[t-1]), float(f.wealth_low.iloc[t-1])
        if t == 1:
            rise, fall = high-previous_high, previous_low-low
            direction = int(not (fall > 0 and fall > rise))
            extreme, speed = high if direction else low, cfg["acceleration"]
            current = previous_low if direction else previous_high
            previous_high, previous_low = high, low
        else:
            direction = int(f.trend_state.iloc[t-1])
            extreme, speed, current = float(f.extreme_price.iloc[t-1]), float(f.acceleration.iloc[t-1]), float(f.next_sar.iloc[t-1])
        crossed = low <= current if direction else high >= current
        next_direction = 1-direction if crossed else direction
        if crossed:
            displayed = min(extreme, previous_low, low) if next_direction else max(extreme, previous_high, high)
            updated_extreme, updated_speed = high if next_direction else low, cfg["acceleration"]
        else:
            displayed = current
            updated_extreme = max(extreme, high) if direction else min(extreme, low)
            updated_speed = min(cfg["maximum_acceleration"], speed + cfg["acceleration"]) if updated_extreme != extreme else speed
        advanced = math.fma(updated_speed, updated_extreme-displayed, displayed)
        projected = min(advanced, previous_low, low) if next_direction else max(advanced, previous_high, high)
        values = {"sar_input": current, "sar_level": displayed, "next_sar": projected, "extreme_price": updated_extreme,
            "acceleration": updated_speed, "trend_state": next_direction, "target": next_direction, "reversed": int(crossed), "initialized": int(t==1)}
        error = max(abs(float(f[name].iloc[t])-value) for name, value in values.items())
        require(error < 1e-12, "逐日初值、触线、极值或加速递推不符")
        require(f.source_state.iloc[t] == ("VIEW_UP_STATE" if next_direction else "VIEW_DOWN_STATE"), "方向中文状态对应不符")
        sar_error_max = max(sar_error_max, error)
        checks.append({"date": f.date.iloc[t], "direction": next_direction, "reversed": crossed, "maximum_state_error": error})
    pd.DataFrame(checks).to_csv(RESEARCH / "saved_parabolic_state_replay.csv", index=False, encoding="utf-8-sig")
    metrics, differences, replay, cycle_rows, groups = [], [], [], [], []
    for period, key in [("evaluation", "all_metrics"), ("earlier_diagnostic", "earlier_diagnostics")]:
        for cost in cfg["costs"]:
            folder = RESEARCH / period / cost
            accounts = {}
            for m in result[key]:
                if m["cost"] != cost:
                    continue
                ledger = pd.read_parquet(folder / f"{m['model']}_ledger.parquet")
                accounts[m["model"]] = ledger
                require(np.allclose(ledger.cash + ledger.shares * ledger.mark + ledger.dividend_receivable, ledger.equity, atol=1e-6, rtol=0), "现金份额和分红应收不等于净值")
                previous = np.r_[cfg["initial_capital"], ledger.equity.to_numpy()[:-1]]
                require(np.allclose(ledger.equity / previous - 1, ledger.net_return, atol=1e-13, rtol=0), "净收益不能从完整净值复算")
                actual = summarize(ledger, cfg)
                for field in ["net_sharpe", "annualized_return", "max_drawdown", "cumulative_return", "commission", "slippage_cost", "mean_exposure"]:
                    require(abs(actual[field] - m[field]) < 1e-10, "保存指标不能复算")
                metrics.append({"period": period, "cost": cost, "model": m["model"], "days": len(ledger), "net_sharpe": actual["net_sharpe"]})
            current = accounts[PRIMARY]
            for control in ["REARM_RIDGE", "PANIC_LEARNED_HALF", "BUY_HOLD"]:
                old = accounts[control]
                require(pd.DatetimeIndex(current.date).equals(pd.DatetimeIndex(old.date)), "对照日历不同")
                d = {"period": period, "cost": cost, "control": control,
                    "terminal_nav_difference": float(current.equity.iloc[-1] - old.equity.iloc[-1]),
                    "price_difference": float(current.price_pnl.sum() - old.price_pnl.sum()),
                    "dividend_difference": float(current.dividend_recognized.sum() - old.dividend_recognized.sum()),
                    "commission_difference": float(current.commission.sum() - old.commission.sum()),
                    "slippage_difference": float(current.slippage_cost.sum() - old.slippage_cost.sum())}
                d["reconciliation_error"] = d["terminal_nav_difference"] - d["price_difference"] - d["dividend_difference"] + d["commission_difference"] + d["slippage_difference"]
                require(abs(d["reconciliation_error"]) < 1e-6, "新旧账户差额没有解释完全")
                differences.append(d)
            decisions = pd.read_parquet(folder / f"{PRIMARY}_decisions.parquet")
            by_date = current.set_index("date")
            for row in decisions.itertuples():
                t = int(row.origin_index)
                require(row.origin == f.date.iloc[t] and row.execution_date == data.date.iloc[t + 1], "收盘判断与下一开盘时钟不符")
                require(row.reference_weight == f.target.iloc[t], "账户目标与当时价格转向不符")
                actual = by_date.loc[row.origin] if row.origin in by_date.index else None
                shares = int(actual.shares) if actual is not None else 0
                nav = float(actual.equity) if actual is not None else cfg["initial_capital"]
                close, target = float(data.close.iloc[t]), float(row.reference_weight)
                target_shares = math.floor(target * nav / close / cfg["lot"]) * cfg["lot"]
                if target == 1 and shares > 0 and abs(target - shares * close / nav) < cfg["weight_band"]:
                    target_shares = shares
                require(row.requested_quantity == target_shares - shares, "请求没有使用实际净值和整手")
            cycles = saved_cycles(current, dividends, cfg)
            cycle_rows.extend({"period": period, "cost": cost, **x} for x in cycles)
            positive = sorted((x["net_profit"] for x in cycles if x["net_profit"] > 0), reverse=True)
            groups.append({"period": period, "cost": cost, "cycles": len(cycles), "positive_cycles": len(positive),
                "negative_cycles": sum(x["net_profit"] < 0 for x in cycles), "net_profit": sum(x["net_profit"] for x in cycles),
                "largest_positive_cycle_profit": positive[0] if positive else 0., "top_two_positive_profit": sum(positive[:2]),
                "initial_target": float(decisions.reference_weight.iloc[0]), "additional_buys": int((current.shares_before.gt(0) & current.filled_quantity.gt(0)).sum())})
            replay.append({"period": period, "cost": cost, "replayed_decision_origins": len(decisions), "held_closes": int(current.shares.gt(0).sum())})
    for name, rows in [("saved_metrics_recomputation.csv", metrics), ("saved_account_differences.csv", differences),
        ("saved_state_and_request_replay.csv", replay), ("saved_actual_cycles.csv", cycle_rows), ("saved_cycle_profit_groups.csv", groups)]:
        pd.DataFrame(rows).to_csv(RESEARCH / name, index=False, encoding="utf-8-sig")
    return {"verified_at": now(), "status": "SAVED_PARABOLIC_STATES_ACCOUNTS_CYCLES_AND_REQUESTS_RECONCILED",
        "recomputed_account_records": len(metrics), "account_differences": len(differences), "factor_rows_recomputed": len(f),
        "maximum_state_recomputation_error": sar_error_max,
        "replayed_decision_origins": sum(x["replayed_decision_origins"] for x in replay), "recomputed_closed_cycles": len(cycle_rows),
        "new_diagnostic_accounts": 0, "new_models": 0, "security_audit_performed": False}, differences, groups


def main():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    result = json.loads((RESEARCH / "result.json").read_text(encoding="utf-8"))
    receipt, differences, groups = verify(cfg, result)
    receipt["reviewer_source_sha256"] = digest(Path(__file__))
    write_json(RESEARCH / "saved_verification_receipt.json", receipt, exclusive=True)
    print(json.dumps({"核对": receipt, "周期": groups}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
