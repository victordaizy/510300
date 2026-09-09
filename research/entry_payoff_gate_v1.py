"""先用成熟的完整交易学习进入，再直接检验四个真实账户。"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from research.entry_payoff_gate_inputs_v1 import FEATURES, CN, full_entry_label, choose_rows, fit_model, entry_views
from research.entry_payoff_gate_account_v1 import simulate_entry_payoff_gate
from research.adaptive_allocation_v1 import normalize_dividends, save_account, summarize
from research.intraday_overnight_increment_v1 import digest, now, require, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_entry_payoff_gate_v1"
CONFIG = ROOT / "config/510300_entry_payoff_gate_v1.json"
P59 = ROOT / "reports/research/510300_entry_path_coverage_v1"
CONTEXT = ROOT / "reports/research/510300_entry_context_coverage_20260908"
P32 = ROOT / "reports/research/510300_rearmed_session_exit_v1"
P91 = ROOT / "reports/research/510300_continuous_reference_min_variance_v1"
PRIMARY = "ENTRY_PAYOFF_RIDGE_NATURAL_EXIT"
NAME = "九因子完整交易收益筛选进入"
CONTROLS = {"REARM_NONE": (P32, "原条件进入及自然退出"), "REARM_RIDGE": (P32, "原条件进入及线性提前退出"), "CONTINUOUS_REFERENCE_MIN_VARIANCE": (P91, "原第91轮局部候选"), "BUY_HOLD": (P32, "买入持有")}


def old_config():
    return json.loads((ROOT / "config/510300_entry_path_coverage_v1.json").read_text(encoding="utf-8"))


def schedule(data, earlier_start):
    anchor = int(np.flatnonzero(data.date.ge(earlier_start))[0])-1
    first_month = data.date.dt.to_period("M").ne(data.date.shift(1).dt.to_period("M"))
    return sorted(set([anchor] + [int(t) for t in np.flatnonzero(first_month) if anchor <= t < len(data)-1]))


def prepare():
    require(not (OUT / "input_receipt.json").exists(), "完整交易标签已保存，不重复生成")
    cfg = old_config()
    contexts = pd.read_parquet(CONTEXT / "saved_entry_context_rows.parquet")
    paths = pd.read_csv(P59 / "reference_paths.csv", parse_dates=["entry_date", "exit_date"])
    groups = pd.read_csv(P59 / "reference_episodes.csv", parse_dates=["group_mature_date"])
    ledgers = pd.read_parquet(P59 / "reference_path_ledgers.parquet")
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    by_path = {int(key): frame for key, frame in ledgers.groupby("path_id", sort=False)}
    labels, states = [], []
    for row in paths.itertuples():
        if not row.natural_exit:
            states.append({"path_id": row.path_id, "status": "NO_VIEW_NO_NATURAL_COMPLETED_ENTRY_LABEL", "original_status": row.status})
            continue
        path = by_path[int(row.path_id)]
        label = full_entry_label(path, dividends, cfg["initial_capital"])
        require(path.date.iloc[-1] == row.exit_date and abs(path.equity.iloc[-1]-row.final_observed_nav) < 1e-6, "原自然退出元数据与保存账本不符")
        labels.append({"path_id": row.path_id, "episode_id": row.episode_id, **label})
        states.append({"path_id": row.path_id, "status": "COMPLETE_ENTRY_PAYOFF_LABEL", "original_status": row.status})
    labels = pd.DataFrame(labels)
    latest = labels.groupby("episode_id").economic_maturity_date.max()
    group_dates = {}
    for row in groups.itertuples():
        group_dates[row.episode_id] = max(pd.Timestamp(row.group_mature_date), latest.get(row.episode_id, pd.Timestamp(row.group_mature_date))) if pd.notna(row.group_mature_date) else pd.NaT
    samples = contexts.merge(labels.drop(columns="episode_id"), on="path_id", how="inner", validate="one_to_one")
    samples["training_maturity_date"] = samples.episode_id.map(group_dates)
    require(np.isfinite(samples[FEATURES]).all().all() and samples.path_id.is_unique, "完整交易样本原点因子缺失或编号重复")
    require((samples.entry_origin < samples.entry_date).all() and (samples.entry_date < samples.natural_exit_date).all(), "完整交易产生顺序不符")
    OUT.mkdir(parents=True, exist_ok=True)
    samples.to_parquet(OUT / "entry_payoff_samples.parquet", index=False)
    samples.to_csv(OUT / "完整交易训练标签及进入因子.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(states).to_csv(OUT / "全部原路径的标签状态.csv", index=False, encoding="utf-8-sig")
    data = pd.read_parquet(ROOT / cfg["features"])
    coverage = []
    for t in schedule(data, cfg["earlier_start"]):
        rows, ids = choose_rows(samples, data.date.iloc[t], cfg["recent_episodes"])
        coverage.append({"fit_index": t, "fit_origin": data.date.iloc[t], "groups": len(ids), "rows": len(rows), "eligible": len(ids) >= cfg["minimum_episodes"] and len(rows) >= cfg["minimum_rows"]})
    pd.DataFrame(coverage).to_csv(OUT / "月度训练覆盖.csv", index=False, encoding="utf-8-sig")
    paths_for_freeze = [Path(__file__), ROOT / "research/entry_payoff_gate_inputs_v1.py", ROOT / "research/entry_payoff_gate_account_v1.py", ROOT / "docs/510300_ENTRY_PAYOFF_GATE_V1.md", OUT / "entry_payoff_samples.parquet"]
    receipt = {"prepared_at": now(), "status": "COMPLETE_ENTRY_PAYOFF_LABELS_READY_NO_NEW_MODEL_OR_ACCOUNT", "saved_paths": len(paths), "natural_completed_label_rows": len(samples), "groups_with_label_rows": int(samples.episode_id.nunique()),
        "rows_in_mature_groups_at_cutoff": int(samples.training_maturity_date.notna().sum()), "unlabeled_non_natural_paths": len(paths)-len(samples),
        "labels_with_extra_rights_after_exit": int(samples.extra_owned_dividend_after_exit.gt(0).sum()), "first_eligible_month": next((row for row in coverage if row["eligible"]), None),
        "eligible_months": sum(row["eligible"] for row in coverage), "monthly_origins": len(coverage), "new_reference_accounts": 0, "new_models": 0,
        "input_files": [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in paths_for_freeze]}
    write_json(OUT / "input_receipt.json", receipt, exclusive=True)
    print(json.dumps({k: v for k, v in receipt.items() if k != "input_files"}, ensure_ascii=False, default=str), flush=True)


def freeze():
    require(not CONFIG.exists(), "完整交易进入筛选已登记，不能重复冻结")
    old = old_config()
    cfg = {k: old[k] for k in ["evaluation_start", "data_cutoff", "initial_capital", "lot", "tick", "limit_fraction", "annual_days", "cash_annual_rate_assumption", "high_sharpe_target", "costs", "features", "dividends", "earlier_start", "earlier_terminal", "minimum_rows", "feature_clip", "ridge_alpha", "recent_episodes", "minimum_episodes", "specification"]}
    cfg.update(study_id="510300_ENTRY_PAYOFF_GATE_V1", round=95, registered_at=now(), primary=PRIMARY, candidate_configurations=1, feature_columns=FEATURES, feature_names=CN,
        model_refit="ORIGINAL_ANCHOR_AND_MONTH_FIRST_COMPLETE_CLOSE", training_target="COMPLETE_NATURAL_ENTRY_ACCOUNT_NET_RETURN_WITH_OWNED_DIVIDEND", training_group_weights="EACH_MATURE_SIGNAL_GROUP_ONE_EQUAL_NATURAL_PATHS",
        no_model_rule="NO_VIEW_NO_NEW_BUY_EXISTING_NATURAL_EXIT_UNCHANGED", entry_threshold=0., exit_method="ORIGINAL_NATURAL_PRICE_AND_TIME_ONLY", reference_cost="BASE",
        rules="docs/510300_ENTRY_PAYOFF_GATE_V1.md", new_reference_accounts=0, independent_validation="NOT_ESTABLISHED", goal_achieved=False, position_impact=0,
        previous_goal_turn_classification="PROGRESS_ROUNDS93_94_COMPLETE_AND_ENTRY_CONTEXT_PREPARED")
    tests = json.loads((OUT / "tests_receipt.json").read_text(encoding="utf-8"))
    require(tests["exit_code"] == 0, "完整交易进入筛选必要测试未通过")
    receipt = json.loads((OUT / "input_receipt.json").read_text(encoding="utf-8"))
    paths = [ROOT / "config/510300_entry_path_coverage_v1.json", ROOT / cfg["features"], ROOT / cfg["dividends"], ROOT / "research/adaptive_allocation_v1.py", ROOT / "research/intraday_overnight_increment_v1.py", ROOT / "research/rearmed_cycle_exit_account_v1.py", ROOT / "tests/test_entry_payoff_gate_v1.py", OUT / "input_receipt.json", OUT / "tests_receipt.json", CONTEXT / "saved_entry_context_rows.parquet", CONTEXT / "result.json"]
    paths.extend(P59 / name for name in ["reference_paths.csv", "reference_episodes.csv", "reference_path_ledgers.parquet", "entry_factors.parquet"])
    for item in receipt["input_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "准备阶段定义或标签改变")
        paths.append(ROOT / item["path"])
    for period in ["evaluation", "earlier_diagnostic"]:
        for cost in cfg["costs"]:
            paths.extend(folder / period / cost / f"{model}_ledger.parquet" for model, (folder, _) in CONTROLS.items())
    cfg["frozen_files"] = [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in sorted(set(paths))]
    write_json(CONFIG, cfg, exclusive=True)
    print("第95轮完整交易收益进入模型已冻结，尚未训练或计算新策略账户收益。", flush=True)


def train(data, samples, cfg):
    models, receipts, memberships = [], [], []
    chinese = ["# 第95轮每月九项进入因子模型", "", "预测对象是整次自然退出交易的完整净收益，各原信号组必须先成熟。", ""]
    for t in schedule(data, cfg["earlier_start"]):
        rows, groups = choose_rows(samples, data.date.iloc[t], cfg["recent_episodes"])
        usable = len(groups) >= cfg["minimum_episodes"] and len(rows) >= cfg["minimum_rows"]
        stored = {"fit_index": t, "fit_origin": str(data.date.iloc[t].date()), "fit_time": data.date.iloc[t]+pd.Timedelta(hours=15, minutes=5), "status": "FIT_COMPLETE" if usable else "NO_VIEW_MINIMUM_MATURE_GROUPS_OR_ROWS",
            "training_groups": groups, "training_group_count": len(groups), "training_rows": len(rows), "latest_training_maturity": str(rows.training_maturity_date.max().date()) if len(rows) else None,
            "model": fit_model(rows, cfg) if usable else None}
        models.append(stored)
        receipts.append({k: v for k, v in stored.items() if k not in ["model", "training_groups"]})
        chinese += [f"## {stored['fit_origin']}", ""]
        if usable:
            memberships.extend({"fit_index": t, "episode_id": row.episode_id, "path_id": row.path_id, "entry_origin": row.entry_origin, "training_maturity_date": row.training_maturity_date, "sample_weight": row.sample_weight} for row in rows.itertuples())
            model = stored["model"]
            chinese += [f"使用{len(groups)}个成熟信号组、{len(rows)}条完整交易标签，最后成熟日期{stored['latest_training_maturity']}。", "", f"截距为{model['intercept']:.10f}。每项因子减下列均值、除标准差，再限制为负五至五后，乘对应系数；九项与截距相加即预测完整交易收益。", ""]
            chinese += [f"- {name}：均值{mean:.10f}，标准差{scale:.10f}，系数{coefficient:.10f}。" for name, mean, scale, coefficient in zip(CN, model["mean"], model["scale"], model["coefficients"])]
        else:
            chinese += [f"只有{len(groups)}个成熟组、{len(rows)}条标签，未达到原十组一百条门槛。没有模型判断，不发起新的买入；原可知退出照常。"]
        chinese += [""]
    write_json(OUT / "saved_models.json", {"models": models, "features": dict(zip(FEATURES, CN))}, exclusive=True)
    pd.DataFrame(receipts).to_csv(OUT / "training_receipts.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(memberships).to_parquet(OUT / "training_memberships.parquet", index=False)
    (OUT / "每月九项进入模型中文规则.md").write_text("\n".join(chinese), encoding="utf-8")
    print(f"本轮{sum(row['status']=='FIT_COMPLETE' for row in models)}次月度训练完成，直接运行四个完整账户。", flush=True)
    return models


def run():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "完整交易进入筛选冻结来源改变")
    write_json(OUT / "RUN_STARTED.json", {"started_at": now(), "config_sha256": digest(CONFIG)}, exclusive=True)
    data = pd.read_parquet(ROOT / cfg["features"])
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    factors = pd.read_parquet(P59 / "entry_factors.parquet")
    require(pd.DatetimeIndex(factors.date).equals(pd.DatetimeIndex(data.date)), "原进入因子与账户日历不同")
    samples = pd.read_parquet(OUT / "entry_payoff_samples.parquet")
    models = train(data, samples, cfg)
    views = entry_views(data, factors.d60_factor, models)
    views.to_parquet(OUT / "entry_views.parquet", index=False)
    main, earlier, yearly, eras, coverage = [], [], [], [], []
    for period, frame, start, destination in [("evaluation", data, cfg["evaluation_start"], main), ("earlier_diagnostic", data[data.date.le(cfg["earlier_terminal"])], cfg["earlier_start"], earlier)]:
        local_views = views.iloc[:len(frame)]
        local_factors = factors.iloc[:len(frame)]
        rule = {"entry": local_factors.raw_entry.to_numpy(int), "exit": {1: local_factors.raw_exit.to_numpy(bool)}}
        first = int(np.flatnonzero(frame.date.ge(start))[0])
        for cost_id, cost in cfg["costs"].items():
            folder = OUT / period / cost_id
            ledger, decisions, cycles = simulate_entry_payoff_gate(frame, dividends, cfg, cost, start, rule, cfg["specification"], local_views)
            decisions = decisions.merge(local_views[["date", *FEATURES, "entry_accepted_by_model"]].rename(columns={"date": "origin"}), on="origin", how="left", validate="one_to_one")
            save_account(folder, PRIMARY, ledger, decisions)
            cycles.to_csv(folder / f"{PRIMARY}_cycles.csv", index=False, encoding="utf-8-sig")
            require(ledger.accounting_error.abs().max() < 1e-6 and not ledger.terminal_unliquidated.iloc[-1], "完整交易进入账户未结算")
            raw = rule["entry"][first-1:-1].astype(bool)
            known = decisions.predicted_entry_return.notna().to_numpy()
            coverage.append({"period": period, "cost": cost_id, "holding_closes": int(ledger.shares.gt(0).sum()), "buy_trades": int(ledger.filled_quantity.gt(0).sum()), "sell_trades": int(ledger.filled_quantity.lt(0).sum()),
                "unfilled_requests": int((ledger.requested_quantity.ne(0) & ledger.filled_quantity.eq(0)).sum()), "mean_exposure": float(ledger.exposure.mean()),
                "raw_entry_origins": int(raw.sum()), "raw_opportunities_without_model": int((raw & ~known).sum()), "raw_opportunities_with_positive_prediction": int((raw & decisions.predicted_entry_return.gt(0).to_numpy()).sum()),
                "origins_without_prediction": int((~known).sum()), "first_actual_entry": str(cycles.entry_date.min().date()) if len(cycles) else None,
                "completed_cycles": len(cycles), "no_model_period_is_not_forecast_success": True})
            accounts, names = {PRIMARY: ledger}, {PRIMARY: NAME}
            for model, (parent, name) in CONTROLS.items():
                saved = pd.read_parquet(parent / period / cost_id / f"{model}_ledger.parquet")
                saved.to_parquet(folder / f"{model}_ledger.parquet", index=False)
                accounts[model], names[model] = saved, name
            bh = summarize(accounts["BUY_HOLD"], cfg)
            for model, saved in accounts.items():
                require(pd.DatetimeIndex(saved.date).equals(pd.DatetimeIndex(ledger.date)), "新旧完整账户日期不同")
                m = {"cost": cost_id, "model": model, "name": names[model], **summarize(saved, cfg)}
                m["annualized_return_excess_vs_buy_hold"] = m["annualized_return"]-bh["annualized_return"]
                m["meets_point_target"] = m["net_sharpe"] is not None and m["net_sharpe"] >= cfg["high_sharpe_target"]
                destination.append(m)
                for year, group in saved.groupby(saved.date.dt.year):
                    yearly.append({"period": period, "cost": cost_id, "model": model, "year": int(year), **summarize(group, cfg)})
                if period == "evaluation":
                    for label, left, right in [("2020—2021", "2020-01-01", "2021-12-31"), ("2022—2023", "2022-01-01", "2023-12-31"), ("2024—终点", "2024-01-01", cfg["data_cutoff"])]:
                        eras.append({"cost": cost_id, "model": model, "era": label, **summarize(saved[saved.date.between(left, right)], cfg)})
            print(f"{period}／{cost_id}：进入筛选完整账户和四个保存对照完成。", flush=True)
    for name, rows in [("metrics.csv", main), ("earlier_diagnostics.csv", earlier), ("yearly_metrics.csv", yearly), ("era_metrics.csv", eras), ("account_coverage.csv", coverage)]:
        pd.DataFrame(rows).to_csv(OUT / name, index=False, encoding="utf-8-sig")
    primary = [m for m in main if m["model"] == PRIMARY]
    write_json(OUT / "result.json", {"study_id": cfg["study_id"], "completed_at": now(), "status": "ENTRY_PAYOFF_GATE_ACCOUNTS_COMPLETE", "candidate_configurations": 1,
        "evaluation_accounts": 10, "new_accounts_generated": 2, "reused_control_accounts": 8, "earlier_diagnostic_accounts": 10, "new_earlier_diagnostic_accounts": 2, "reused_earlier_accounts": 8,
        "new_model_fits": sum(m["status"] == "FIT_COMPLETE" for m in models), "new_reference_accounts": 0,
        "all_metrics": main, "earlier_diagnostics": earlier, "primary": primary, "post_selected_best_base": next(m for m in primary if m["cost"] == "BASE"), "account_coverage": coverage,
        "historical_point_target_met": any(m["meets_point_target"] for m in primary), "goal_achieved": False, "independent_validation": "NOT_ESTABLISHED", "position_impact": 0}, exclusive=True)
    print(json.dumps({"主结果": primary, "较早结果": [m for m in earlier if m["model"] == PRIMARY], "账户覆盖": coverage}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    import sys
    {"prepare": prepare, "freeze": freeze, "run": run}[sys.argv[1]]()
