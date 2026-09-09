"""登记买入与等待策略后，共用月度模型直接计算八个完整账户。"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from research.entry_wait_policy_inputs_v1 import FEATURES, CN, select_training_rows, action_arrays, fit_policy, observed_signal_age, action_views
from research.entry_wait_policy_account_v1 import simulate_entry_wait_policy
from research.entry_payoff_gate_v1 import schedule
from research.adaptive_allocation_v1 import normalize_dividends, save_account, summarize
from research.intraday_overnight_increment_v1 import digest, now, require, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_entry_wait_policy_v1"
CONFIG = ROOT / "config/510300_entry_wait_policy_v1.json"
LINKS = ROOT / "reports/research/510300_entry_wait_transition_20260908"
P59 = ROOT / "reports/research/510300_entry_path_coverage_v1"
P95 = ROOT / "reports/research/510300_entry_payoff_gate_v1"
P91 = ROOT / "reports/research/510300_continuous_reference_min_variance_v1"
P32 = ROOT / "reports/research/510300_rearmed_session_exit_v1"
PRIMARY = "ENTRY_WAIT_POLICY"
IMMEDIATE = "IMMEDIATE_VALUE_CONTROL"
SETTINGS = {PRIMARY: ("waiting_policy_margin", "买入与等待价值比较"), IMMEDIATE: ("immediate_value_margin", "同模型只看买入价值为正")}
CONTROLS = {"ENTRY_PAYOFF_RIDGE_NATURAL_EXIT": (P95, "原第95轮完整收益筛选"), "CONTINUOUS_REFERENCE_MIN_VARIANCE": (P91, "原第91轮局部候选"), "BUY_HOLD": (P32, "买入持有")}


def freeze():
    require(not CONFIG.exists(), "第96轮已经登记，不能重新冻结")
    old = json.loads((ROOT / "config/510300_entry_payoff_gate_v1.json").read_text(encoding="utf-8"))
    fields = ["evaluation_start", "data_cutoff", "initial_capital", "lot", "tick", "limit_fraction", "annual_days", "cash_annual_rate_assumption", "high_sharpe_target", "costs", "features", "dividends", "earlier_start", "earlier_terminal", "minimum_rows", "feature_clip", "ridge_alpha", "recent_episodes", "minimum_episodes", "specification"]
    cfg = {k: old[k] for k in fields}
    cfg.update(study_id="510300_ENTRY_WAIT_POLICY_V1", round=96, registered_at=now(), primary=PRIMARY, candidate_configurations=2, settings=SETTINGS,
        feature_columns=FEATURES, feature_names=CN, maximum_policy_iterations=50, training_method="EVALUATE_FIXED_POLICY_REALIZED_PAYOFF_THEN_IMPROVE_PREDICTED_ACTIONS",
        training_weights="ONE_PER_COMPLETE_GROUP_EQUAL_STATES", cash_step_reward=0., discount_factor=1., objective="LOCAL_OPPORTUNITY_TERMINAL_NET_RETURN_ON_ORIGINAL_200000",
        no_model_rule="BOTH_SETTINGS_NO_VIEW_NO_NEW_BUY_EXISTING_NATURAL_EXIT", reference_cost="BASE", rules="docs/510300_ENTRY_WAIT_POLICY_V1.md",
        new_reference_accounts=0, independent_validation="NOT_ESTABLISHED", goal_achieved=False, position_impact=0)
    tests = json.loads((OUT / "tests_receipt.json").read_text(encoding="utf-8"))
    require(tests["exit_code"] == 0 and tests["passed"] == 8, "动作与账户必要测试未通过")
    links = pd.read_parquet(LINKS / "entry_action_links.parquet")
    factors = pd.read_parquet(P59 / "entry_factors.parquet")
    np.testing.assert_allclose(links.signal_age_observed_days, observed_signal_age(factors)[links.path_id.to_numpy(int)], atol=0, rtol=0)
    rows, groups = select_training_rows(links, cfg["data_cutoff"], 100000)
    action_arrays(rows)
    require(len(rows) == 1180 and len(groups) == 46, "已完成动作组范围改变")
    receipt = json.loads((LINKS / "result.json").read_text(encoding="utf-8"))
    paths = [Path(__file__), ROOT / "research/entry_wait_policy_inputs_v1.py", ROOT / "research/entry_wait_policy_account_v1.py", ROOT / "research/entry_payoff_gate_inputs_v1.py", ROOT / "research/entry_payoff_gate_v1.py",
        ROOT / "research/adaptive_allocation_v1.py", ROOT / "research/intraday_overnight_increment_v1.py", ROOT / cfg["features"], ROOT / cfg["dividends"], ROOT / cfg["rules"], ROOT / "config/510300_entry_payoff_gate_v1.json",
        ROOT / "tests/test_entry_wait_policy_v1.py", OUT / "tests_receipt.json", LINKS / "entry_action_links.parquet", LINKS / "result.json", P59 / "entry_factors.parquet"]
    for item in receipt["source_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "准备好的动作来源发生变化")
        paths.append(ROOT / item["path"])
    for period in ["evaluation", "earlier_diagnostic"]:
        for cost in cfg["costs"]:
            paths.extend(parent / period / cost / f"{model}_ledger.parquet" for model, (parent, _) in CONTROLS.items())
    cfg["frozen_files"] = [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in sorted(set(paths))]
    write_json(CONFIG, cfg, exclusive=True)
    print("第96轮两个设置已冻结；完整动作连接1180个状态、46组，尚未训练或计算新账户。", flush=True)


def train(data, links, cfg):
    models, memberships, receipts, iterations, coefficients = [], [], [], [], []
    for t in schedule(data, cfg["earlier_start"]):
        rows, groups = select_training_rows(links, data.date.iloc[t], cfg["recent_episodes"])
        eligible = len(groups) >= cfg["minimum_episodes"] and len(rows) >= cfg["minimum_rows"]
        fitted = fit_policy(rows, cfg) if eligible else {"status": "NO_VIEW_MINIMUM_MATURE_GROUPS_OR_ROWS", "model": None, "iterations": [], "scalar_action_regressions": 0, "matrix_factorizations": 0}
        stored = {"fit_index": t, "fit_origin": str(data.date.iloc[t].date()), "fit_time": data.date.iloc[t]+pd.Timedelta(hours=15, minutes=5),
            "status": fitted["status"], "training_groups": groups, "training_group_count": len(groups), "training_rows": len(rows), "eligible": eligible,
            "latest_mature_group_date": str(rows.original_group_mature_date.max().date()) if len(rows) else None,
            "iterations": len(fitted["iterations"]), "scalar_action_regressions": fitted["scalar_action_regressions"], "matrix_factorizations": fitted["matrix_factorizations"], "model": fitted["model"]}
        models.append(stored)
        receipts.append({k: v for k, v in stored.items() if k not in ["training_groups", "model"]})
        if eligible:
            memberships.extend({"fit_index": t, "path_id": row.path_id, "episode_id": row.episode_id, "mature_date": row.original_group_mature_date, "sample_weight": row.sample_weight} for row in rows.itertuples())
            iterations.extend({"fit_index": t, "fit_origin": stored["fit_origin"], **iteration} for iteration in fitted["iterations"])
        if fitted["model"] is not None:
            m = fitted["model"]
            coefficients.append({"模型日期": stored["fit_origin"], "因子": "截距", "均值": None, "标准差": None, "买入价值系数": m["buy_coefficients_with_intercept"][0], "等待价值系数": m["wait_coefficients_with_intercept"][0]})
            coefficients.extend({"模型日期": stored["fit_origin"], "因子": name, "均值": mean, "标准差": scale, "买入价值系数": b, "等待价值系数": w} for name, mean, scale, b, w in zip(CN, m["mean"], m["scale"], m["buy_coefficients_with_intercept"][1:], m["wait_coefficients_with_intercept"][1:]))
    write_json(OUT / "saved_models.json", {"models": models, "features": dict(zip(FEATURES, CN))}, exclusive=True)
    write_json(OUT / "saved_policy_iterations.json", {"iterations": iterations}, exclusive=True)
    pd.DataFrame(memberships).to_parquet(OUT / "training_memberships.parquet", index=False)
    pd.DataFrame(receipts).to_csv(OUT / "training_receipts.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(coefficients).to_csv(OUT / "每月十项因子及两个动作的实际系数.csv", index=False, encoding="utf-8-sig")
    print("月度动作模型完成："+json.dumps(pd.Series([m["status"] for m in models]).value_counts().to_dict(), ensure_ascii=False), flush=True)
    return models


def run():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "第96轮冻结输入改变")
    write_json(OUT / "RUN_STARTED.json", {"started_at": now(), "config_sha256": digest(CONFIG)}, exclusive=True)
    data = pd.read_parquet(ROOT / cfg["features"])
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    factors = pd.read_parquet(P59 / "entry_factors.parquet")
    models = train(data, pd.read_parquet(LINKS / "entry_action_links.parquet"), cfg)
    views = action_views(data, factors, models)
    views.to_parquet(OUT / "action_views.parquet", index=False)
    main, earlier, yearly, eras, coverage = [], [], [], [], []
    for period, frame, start, destination in [("evaluation", data, cfg["evaluation_start"], main), ("earlier_diagnostic", data[data.date.le(cfg["earlier_terminal"])], cfg["earlier_start"], earlier)]:
        local_views, local_factors = views.iloc[:len(frame)], factors.iloc[:len(frame)]
        rule = {"entry": local_factors.raw_entry.to_numpy(int), "exit": {1: local_factors.raw_exit.to_numpy(bool)}}
        first = int(np.flatnonzero(frame.date.ge(start))[0])
        for cost_id, cost in cfg["costs"].items():
            folder = OUT / period / cost_id
            accounts, names = {}, {}
            for model, (margin, name) in SETTINGS.items():
                local = local_views.assign(entry_value_margin=local_views[margin], entry_decision_rule=model)
                ledger, decisions, cycles = simulate_entry_wait_policy(frame, dividends, cfg, cost, start, rule, cfg["specification"], local)
                decisions = decisions.merge(local[["date", *FEATURES]].rename(columns={"date": "origin"}), on="origin", how="left", validate="one_to_one")
                save_account(folder, model, ledger, decisions)
                cycles.to_csv(folder / f"{model}_cycles.csv", index=False, encoding="utf-8-sig")
                require(ledger.accounting_error.abs().max() < 1e-6 and not ledger.terminal_unliquidated.iloc[-1], "动作策略账户未完整结算")
                raw = rule["entry"][first-1:-1].astype(bool)
                known = decisions.entry_value_margin.notna().to_numpy()
                buys = ledger.loc[ledger.filled_quantity.gt(0), "date"]
                coverage.append({"period": period, "cost": cost_id, "model": model, "holding_closes": int(ledger.shares.gt(0).sum()), "buy_trades": len(buys), "sell_trades": int(ledger.filled_quantity.lt(0).sum()),
                    "unfilled_requests": int((ledger.requested_quantity.ne(0) & ledger.filled_quantity.eq(0)).sum()), "mean_exposure": float(ledger.exposure.mean()), "raw_entry_origins": int(raw.sum()),
                    "raw_opportunities_without_model": int((raw & ~known).sum()), "raw_opportunities_accepted_by_value": int((raw & decisions.entry_value_margin.gt(0).to_numpy()).sum()),
                    "origins_without_prediction": int((~known).sum()), "first_actual_entry": str(buys.min().date()) if len(buys) else None, "completed_cycles": len(cycles)})
                accounts[model], names[model] = ledger, name
            for model, (parent, name) in CONTROLS.items():
                saved = pd.read_parquet(parent / period / cost_id / f"{model}_ledger.parquet")
                saved.to_parquet(folder / f"{model}_ledger.parquet", index=False)
                accounts[model], names[model] = saved, name
            bh = summarize(accounts["BUY_HOLD"], cfg)
            for model, saved in accounts.items():
                require(pd.DatetimeIndex(saved.date).equals(pd.DatetimeIndex(ledger.date)), "新旧账户日历不同")
                m = {"cost": cost_id, "model": model, "name": names[model], **summarize(saved, cfg)}
                m["annualized_return_excess_vs_buy_hold"] = m["annualized_return"]-bh["annualized_return"]
                m["meets_point_target"] = m["net_sharpe"] is not None and m["net_sharpe"] >= cfg["high_sharpe_target"]
                destination.append(m)
                for year, group in saved.groupby(saved.date.dt.year):
                    yearly.append({"period": period, "cost": cost_id, "model": model, "year": int(year), **summarize(group, cfg)})
                if period == "evaluation":
                    for label, left, right in [("2020—2021", "2020-01-01", "2021-12-31"), ("2022—2023", "2022-01-01", "2023-12-31"), ("2024—终点", "2024-01-01", cfg["data_cutoff"])]:
                        eras.append({"cost": cost_id, "model": model, "era": label, **summarize(saved[saved.date.between(left, right)], cfg)})
            print(f"{period}／{cost_id}：两个新完整账户与三个保存对照完成。", flush=True)
    for name, rows in [("metrics.csv", main), ("earlier_diagnostics.csv", earlier), ("yearly_metrics.csv", yearly), ("era_metrics.csv", eras), ("account_coverage.csv", coverage)]:
        pd.DataFrame(rows).to_csv(OUT / name, index=False, encoding="utf-8-sig")
    new = [m for m in main if m["model"] in SETTINGS]
    summary = {"study_id": cfg["study_id"], "completed_at": now(), "status": "ENTRY_WAIT_POLICY_ACCOUNTS_COMPLETE", "candidate_configurations": 2,
        "evaluation_accounts": 10, "new_accounts_generated": 4, "reused_control_accounts": 6, "earlier_diagnostic_accounts": 10, "new_earlier_diagnostic_accounts": 4, "reused_earlier_accounts": 6,
        "new_model_fits": sum(m["eligible"] for m in models), "stable_monthly_models": sum(m["status"] == "FIT_COMPLETE" for m in models),
        "scalar_action_regressions": sum(m["scalar_action_regressions"] for m in models), "matrix_factorizations": sum(m["matrix_factorizations"] for m in models),
        "model_status_counts": pd.Series([m["status"] for m in models]).value_counts().to_dict(), "new_reference_accounts": 0,
        "all_metrics": main, "earlier_diagnostics": earlier, "primary": [m for m in new if m["model"] == PRIMARY], "account_coverage": coverage,
        "post_selected_best_base": max((m for m in new if m["cost"] == "BASE"), key=lambda m: m["net_sharpe"] if m["net_sharpe"] is not None else float("-inf")),
        "historical_point_target_met": any(m["meets_point_target"] for m in new), "goal_achieved": False, "independent_validation": "NOT_ESTABLISHED", "position_impact": 0}
    write_json(OUT / "result.json", summary, exclusive=True)
    print(json.dumps({"主新账户": new, "较早新账户": [m for m in earlier if m["model"] in SETTINGS], "模型状态": summary["model_status_counts"], "实际标量回归次数": summary["scalar_action_regressions"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    import sys
    {"freeze": freeze, "run": run}[sys.argv[1]]()
