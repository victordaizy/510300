"""比较50、60、70日窗口，固定已选组合的其他规则及已存模型。"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import time

import numpy as np
import pandas as pd

from research.strategy_review_diagnostics_v1 import ROOT, MAIN, INPUTS, MODEL, read, write, digest, inputs, metrics
from research.strategy_review_full_graph_v1 import DiagnosticPipeline
from research.simple_session_divergence_v1 import make_rule

OUT = ROOT / "reports/research/510300_session_window_neighborhood_v1"
PERIODS = {"main": ("2020-01-02", "2026-09-11", "2026-09-14"), "early": ("2015-01-05", "2019-12-31", "2020-01-02")}
WINDOWS = [50, 60, 70]


def factor_rule(data, window, dispersion=None):
    difference = data.intraday_log - data.overnight_log
    std = difference.rolling(dispersion or window).std(ddof=1)
    factor = difference.rolling(window).sum() / (std.replace(0, np.nan) * np.sqrt(window))
    entry = ((factor > 1) & (factor.shift(1) > 1)).fillna(False) & data.feature_valid.fillna(False)
    exit_flag = ((factor < 0) & (factor.shift(1) < 0)).fillna(False)
    return factor, {"entry": entry.to_numpy(int), "exit": {1: exit_flag.to_numpy(bool)}}


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = OUT / "protocol.json"
    if protocol.exists():
        return
    graph = read(INPUTS / "dependency_graph.json")
    paths = {Path(__file__), ROOT / "research/strategy_review_full_graph_v1.py",
             ROOT / "research/post_selection_continuous_replay_v1.py", ROOT / "research/post_selection_continuous_accounts_v1.py",
             INPUTS / "candidate_features.parquet", INPUTS / "dependency_graph.json",
             MAIN / "within_models.json", MAIN / "ridge_models.json", ROOT / "data/reference/510300_dividends.csv"}
    paths.update(ROOT / node["configuration"] for node in graph["nodes"])
    for cost in ["BASE", "STRESS"]:
        paths.update(MAIN / "accounts" / cost / MODEL / name for name in ["ledger.parquet", "decisions.parquet"])
    spec = {"study_id": "510300_SESSION_WINDOW_NEIGHBORHOOD_V1", "registered_at": datetime.now().astimezone().isoformat(),
        "user_request": "我的意思是不去六十日 如果改成五十日，七十日，结果有很大影响吗？",
        "windows": WINDOWS, "primary_variants": "SUM_N(d)/(STDEV.S_N(d)*SQRT(N))", "entry_threshold": 1,
        "exit_threshold": 0, "two_day_confirmation": True,
        "secondary_signal_only": "累计改N，标准差窗口仍为60，检验源码只改window参数的另一口径",
        "periods": PERIODS, "all_other_rules_fixed": True, "max_holding_days_stays_60": True,
        "risk_budget_windows_30_60_unchanged": True, "costs_and_initial_capital_unchanged": True,
        "saved_monthly_model_coefficients_fixed": True, "monthly_fit_origin_selection_retained": True,
        "new_model_fits": 0, "automatic_strategy_promotion": False,
        "main_60_uses_current_saved_baseline": True,
        "planned_new_full_graph_runs": ["main_50", "main_70", "early_50", "early_60", "early_70"],
        "planned_new_accounts": 110, "position_impact": 0,
        "evidence_class": "PREVIOUSLY_OBSERVED_HISTORY_FIXED_MODEL_LOCAL_PARAMETER_DIAGNOSTIC",
        "frozen_files": [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in sorted(paths)]}
    write(protocol, spec)
    print("已登记50／60／70日局部对照，尚未读取本次变体收益。", flush=True)


def check_frozen():
    spec = read(OUT / "protocol.json")
    for item in spec["frozen_files"]:
        if digest(ROOT / item["path"]) != item["sha256"]:
            raise RuntimeError("对照登记后的输入发生变化：" + item["path"])
    return spec


def signal_comparison():
    cfg, data, div = inputs()
    baseline, base_rule = factor_rule(data, 60)
    original, original_factor = make_rule(data, {"kind": "DIFFERENCE", "window": 60, "direction": 1})
    np.testing.assert_allclose(baseline, original_factor, rtol=0, atol=1e-12, equal_nan=True)
    np.testing.assert_array_equal(base_rule["entry"], original["entry"])
    np.testing.assert_array_equal(base_rule["exit"][1], original["exit"][1])
    factors, rows = {"date": data.date}, []
    for variant in ["MATCHED_N", "STD_FIXED_60"]:
        for window in WINDOWS:
            factor, rule = factor_rule(data, window, window if variant == "MATCHED_N" else 60)
            factors[f"{variant}_{window}"] = factor
            for period, (start, end, _) in PERIODS.items():
                mask = (data.date >= start) & (data.date <= end) & data.feature_valid.fillna(False)
                mask &= factor.notna() & factor.shift(1).notna() & baseline.notna() & baseline.shift(1).notna()
                indices = mask.to_numpy(bool)
                entry = rule["entry"].astype(bool)
                exits = rule["exit"][1]
                entry_changed = (entry != base_rule["entry"].astype(bool)) & indices
                exit_changed = (exits != base_rule["exit"][1]) & indices
                rows.append({"period": period, "variant": variant, "window": window, "days": int(mask.sum()),
                    "score_correlation_vs_60": float(factor[mask].corr(baseline[mask])),
                    "entry_condition_days": int((entry & indices).sum()), "entry_changed_days": int(entry_changed.sum()),
                    "entry_changed_fraction": float(entry_changed.sum() / mask.sum()),
                    "exit_condition_days": int((exits & indices).sum()), "exit_changed_days": int(exit_changed.sum()),
                    "either_changed_days": int((entry_changed | exit_changed).sum()),
                    "last_factor": float(factor[mask].iloc[-1]), "last_entry": bool(entry[indices][-1]),
                    "last_exit": bool(exits[indices][-1])})
    pd.DataFrame(rows).to_csv(OUT / "signal_comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(factors).to_csv(OUT / "all_factor_values.csv", index=False, encoding="utf-8-sig")
    data[["date", "previous_close", "open", "close", "dividend", "overnight_log", "intraday_log", "feature_valid"]].to_csv(
        OUT / "signal_inputs.csv", index=False, encoding="utf-8-sig")
    print(pd.DataFrame(rows).query("period == 'main' and variant == 'MATCHED_N'").to_string(index=False), flush=True)


class WindowPipeline(DiagnosticPipeline):
    def __init__(self, period, window):
        self.variant = f"{period}_N{window}"
        self.destination = OUT / "full_graph" / self.variant
        self.destination.mkdir(parents=True, exist_ok=True)
        self.cfg, data, self.div = inputs()
        self.start, cutoff, self.next_date = PERIODS[period]
        self.data = data[data.date <= cutoff].reset_index(drop=True)
        self.capital = 200000.0
        self.cfg["initial_capital"] = self.capital
        self.cfg["data_cutoff"] = cutoff
        self.first = int(np.flatnonzero(self.data.date.ge(self.start))[0])
        self.graph = {n["node"]: deepcopy(n) for n in read(INPUTS / "dependency_graph.json")["nodes"]}
        if period == "early":
            for node in self.graph.values():
                if node["replay_start"] == "2020-01-02":
                    node["replay_start"] = self.start
        self.accounts, self.targets, self.settings, self.calls = {}, {}, {}, {}
        self.checks, self.target_checks = [], []
        self.within = read(MAIN / "within_models.json")["models"]
        self.ridge = read(MAIN / "ridge_models.json")["models"]
        self.factor, self.session = factor_rule(self.data, window)


def account_run(period, window):
    folder = OUT / "full_graph" / f"{period}_N{window}"
    if (folder / "result.json").exists():
        print("该窗口完整组合已完成，复用保存结果。", flush=True)
        return
    if period == "main" and window == 60:
        raise ValueError("主历史60日使用已保存正式账户，不重复生成")
    start = time.perf_counter()
    pipeline = WindowPipeline(period, window).run()
    rows = []
    for cost in ["BASE", "STRESS"]:
        ledger, decisions, state = pipeline.accounts[MODEL, cost]
        rows.append({"period": period, "window": window, "cost": cost, **metrics(ledger)})
    write(folder / "result.json", {"status": "COMPLETED_FIXED_MODEL_NEIGHBOR_WINDOW_FULL_GRAPH", "period": period,
        "window": window, "accounts": len(pipeline.accounts), "nodes": len(pipeline.graph), "metrics": rows,
        "new_model_fits": 0, "monthly_coefficients_retrained_for_window": False,
        "seconds": time.perf_counter() - start})
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


def summarize_saved():
    rows, comparison, yearly, cycle_rows, identities = [], [], [], [], []
    for period in PERIODS:
        baseline = None
        for window in [60, 50, 70]:
            for cost in ["BASE", "STRESS"]:
                folder = MAIN / "accounts" / cost / MODEL if period == "main" and window == 60 else OUT / "full_graph" / f"{period}_N{window}" / "accounts" / cost / MODEL
                ledger = pd.read_parquet(folder / "ledger.parquet")
                decisions = pd.read_parquet(folder / "decisions.parquet")
                rows.append({"period": period, "window": window, "cost": cost, **metrics(ledger)})
                export = OUT / "master_accounts" / f"{period}_N{window}_{cost}"
                export.mkdir(parents=True, exist_ok=True)
                ledger.to_csv(export / "ledger.csv", index=False, encoding="utf-8-sig")
                decisions.to_csv(export / "decisions.csv", index=False, encoding="utf-8-sig")
                base_folder = MAIN / "accounts" / cost / MODEL if period == "main" else OUT / "full_graph" / "early_N60" / "accounts" / cost / MODEL
                base = pd.read_parquet(base_folder / "decisions.parquet")
                matched = decisions.merge(base, on="origin", suffixes=("_n", "_60"))
                p, q = matched.reference_weight_n.to_numpy(float), matched.reference_weight_60.to_numpy(float)
                valid = np.isfinite(p) & np.isfinite(q)
                comparison.append({"period": period, "window": window, "cost": cost, "decision_days": int(valid.sum()),
                    "target_changed_days": int((np.abs(p[valid]-q[valid]) > 1e-10).sum()),
                    "zero_positive_changed_days": int(((p[valid] == 0) != (q[valid] == 0)).sum()),
                    "average_absolute_target_change": float(np.mean(np.abs(p[valid]-q[valid])))})
                previous = 200000.0
                for year, group in ledger.groupby(ledger.date.dt.year):
                    yearly.append({"period": period, "window": window, "cost": cost, "year": int(year),
                                   "return": float(group.equity.iloc[-1]/previous-1), "fills": int(group.filled_quantity.ne(0).sum())})
                    previous = float(group.equity.iloc[-1])
    for folder in sorted((OUT / "full_graph").glob("*/accounts/*/*")):
        ledger = pd.read_parquet(folder / "ledger.parquet")
        np.testing.assert_allclose(ledger.equity, ledger.cash + ledger.shares*ledger.mark + ledger.dividend_receivable, atol=1e-7, rtol=0)
        previous = np.r_[200000., ledger.equity.to_numpy(float)[:-1]]
        np.testing.assert_allclose(ledger.equity-previous, ledger.price_pnl+ledger.dividend_recognized-ledger.commission-ledger.slippage_cost, atol=1e-7, rtol=0)
        np.testing.assert_allclose(ledger.net_return, ledger.equity/previous-1, atol=1e-12, rtol=0)
        identities.append({"folder": str(folder.relative_to(OUT)), "rows": len(ledger)})
    pd.DataFrame(rows).to_csv(OUT / "portfolio_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(comparison).to_csv(OUT / "portfolio_decision_changes.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(yearly).to_csv(OUT / "yearly_returns.csv", index=False, encoding="utf-8-sig")
    write(OUT / "result.json", {"status": "COMPLETED_FIXED_MODEL_50_60_70_SENSITIVITY", "metrics": rows,
        "new_full_graphs": 5, "new_accounts": len(identities), "verified_ledger_rows": sum(x["rows"] for x in identities),
        "ledger_identities": identities, "new_model_fits": 0, "changed_current_strategy": False,
        "strict_forward_days": 0, "full_retraining_for_50_70": "NOT_RUN_FIXED_MODEL_SENSITIVITY_ONLY"})
    print(pd.DataFrame(rows).query("period == 'main'").to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="固定其他规则，比较50／60／70交易日窗口")
    parser.add_argument("action", choices=["freeze", "signals", "run", "summarize"])
    parser.add_argument("--period", choices=list(PERIODS))
    parser.add_argument("--window", type=int, choices=WINDOWS)
    args = parser.parse_args()
    if args.action == "freeze":
        freeze()
    else:
        check_frozen()
        if args.action == "signals":
            signal_comparison()
        elif args.action == "run":
            if args.period is None or args.window is None:
                parser.error("账户对照需要明确period及window")
            account_run(args.period, args.window)
        else:
            summarize_saved()
