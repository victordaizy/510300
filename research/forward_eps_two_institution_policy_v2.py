"""两机构前瞻盈利与来源质量的共同月份模型和完整账户研究。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.financial_annual_components_v1 import read, save, now
from research.forward_eps_guosen_history_v1 import identity
from research.forward_eps_monthly_policy_v1 import PRICE, mature_training
from research.event_clock_account_v1 import simulate_event_account
from research.adaptive_allocation_v1 import save_account, summarize
from research.intraday_overnight_increment_v1 import normalize_dividends, holding_total_return, block_indices, return_metrics, interval

CONFIG = ROOT / "config/510300_forward_eps_two_institution_policy_v2.json"
MANIFEST = ROOT / "config/510300_forward_eps_two_institution_policy_v2_manifest.json"
OUT = ROOT / "reports/research/510300_forward_eps_two_institution_policy_v2"
SOURCE = ROOT / "reports/research/510300_forward_eps_monthly_policy_v2_csi"
FEATURE_SOURCE = ROOT / "reports/research/510300_forward_eps_two_institution_features_v2"
PARENT = ROOT / "reports/research/510300_adaptive_allocation_v1"
POOLED = ["pooled_eps_growth_median", "pooled_profit_revision_median", "pooled_reported_earnings_yield_median"]
SOOCHOW = ["soochow_eps_growth_median", "soochow_profit_revision_median", "soochow_reported_earnings_yield_median"]
QUALITY = ["pooled_profit_revision_breadth", "dual_institution_growth_share", "pooled_report_age_mean_days"]
MODELS = {"S1_POOLED_EPS": POOLED, "S2_MATCHED_SOOCHOW_EPS": SOOCHOW, "S3_POOLED_EPS_INFORMATION_QUALITY": POOLED + QUALITY}
CONTROLS = {"C0_ORIGINAL_GUOSEN_EPS": "E3_FORWARD_EPS", "BUY_HOLD": "BUY_HOLD"}
PAIRS = {"S1_POOLED_EPS": "S2_MATCHED_SOOCHOW_EPS", "S2_MATCHED_SOOCHOW_EPS": "C0_ORIGINAL_GUOSEN_EPS", "S3_POOLED_EPS_INFORMATION_QUALITY": "S1_POOLED_EPS"}


def freeze() -> None:
    config = read(ROOT / "config/510300_forward_eps_monthly_policy_v2.json")
    for key in ["rule", "source_layout_correction_only", "source_fact_version", "primary_scope", "diagnostic_scope"]:
        config.pop(key, None)
    config.update({"study_id": "510300_FORWARD_EPS_TWO_INSTITUTION_POLICY_V2", "primary": "S1_POOLED_EPS",
                   "models": MODELS, "bootstrap_day_blocks": [20, 60], "candidate_configurations": 3,
                   "planned_evaluation_accounts": 10, "planned_new_accounts": 6, "reused_control_accounts": 4,
                   "prepare_gpt_numerical_review_package": False, "common_months_for_all_models": True,
                   "new_source_originals_in_progress_at_registration": True,
                   "same_institution_profit_revision": True, "source_definition_unknown_preserved": True,
                   "source_layout_correction_only": True, "soochow_source_fact_version": 3,
                   "replaces_unrun_policy_binding": "510300_FORWARD_EPS_TWO_INSTITUTION_POLICY_V1"})
    paths = [Path(__file__), ROOT / "docs/510300_FORWARD_EPS_TWO_INSTITUTION_POLICY_V2.md",
             ROOT / "research/forward_eps_two_institution_features_v2.py",
             ROOT / "tests/test_forward_eps_two_institution_features_v2.py", FEATURE_SOURCE / "manifest.json",
             ROOT / "research/forward_eps_soochow_facts_v3.py", ROOT / "tests/test_forward_eps_soochow_layout_v3.py",
             ROOT / "research/event_clock_account_v1.py", ROOT / "research/adaptive_allocation_v1.py",
             ROOT / "research/intraday_overnight_increment_v1.py", ROOT / "research/forward_eps_monthly_policy_v1.py",
             ROOT / "research/financial_annual_components_v1.py", ROOT / "research/forward_eps_guosen_history_v1.py",
             ROOT / "config/510300_research_authority_v6.json", PARENT / "features.parquet",
             ROOT / config["inputs"]["dividends"], SOURCE / "signals.parquet", SOURCE / "result.json"]
    for cost in config["costs"]:
        for original in CONTROLS.values():
            paths.extend([SOURCE / "evaluation" / cost / (original + "_ledger.parquet"),
                          SOURCE / "evaluation" / cost / (original + "_decisions.parquet")])
    prior = read(ROOT / "config/510300_forward_eps_two_institution_policy_v1.json")
    changed_keys = {"study_id", "source_layout_correction_only", "soochow_source_fact_version", "replaces_unrun_policy_binding"}
    assert {k: v for k, v in prior.items() if k not in changed_keys} == {k: v for k, v in config.items() if k not in changed_keys}, "来源更正不允许改变模型和账户规则"
    save(CONFIG, config, exclusive=True)
    save(MANIFEST, {"registered_at": now(), "new_candidate_returns_read_before_freeze": False,
         "old_results_already_observed": True, "source_collection_in_progress": True,
         "candidate_configurations": 3, "planned_new_accounts": 6, "planned_evaluation_accounts": 10,
         "files": [identity(p) for p in [CONFIG, *paths]]}, exclusive=True)
    print("第十九轮三候选来源更正绑定已登记；方法与进出场未变，尚未运行账户。", flush=True)


def run() -> None:
    config = read(CONFIG)
    for item in read(MANIFEST)["files"]:
        if identity(ROOT / item["path"])["sha256"] != item["sha256"]:
            raise ValueError("第十九轮冻结规则或输入变化")
    if not (FEATURE_SOURCE / "result.json").exists():
        raise RuntimeError("两机构完整来源和月度因子尚未完成")
    for item in read(FEATURE_SOURCE / "completed_source_receipt.json")["files"]:
        if identity(ROOT / item["path"])["sha256"] != item["sha256"]:
            raise ValueError("已完成来源发生变化")
    OUT.mkdir(parents=True, exist_ok=False)
    feature_paths = [FEATURE_SOURCE / name for name in ["result.json", "monthly_two_institution_features.parquet", "pooled_company_month_features.parquet", "institution_company_month_evidence.parquet", "completed_source_receipt.json"]]
    save(OUT / "RUN_STARTED.json", {"started_at": now(), "manifest": identity(MANIFEST),
         "completed_feature_inputs_bound_before_new_labels": [identity(p) for p in feature_paths]}, exclusive=True)
    data = pd.read_parquet(PARENT / "features.parquet")
    dividends = normalize_dividends(pd.read_csv(ROOT / config["inputs"]["dividends"]))
    monthly = pd.read_parquet(FEATURE_SOURCE / "monthly_two_institution_features.parquet")
    monthly = monthly.merge(data[["date", *PRICE]], left_on="origin", right_on="date", validate="one_to_one").drop(columns="date")
    monthly["all_features_valid"] = monthly.common_sources_valid & np.isfinite(monthly[PRICE + POOLED + SOOCHOW + QUALITY]).all(axis=1)
    monthly["origin_index"] = pd.DatetimeIndex(data.date).get_indexer(monthly.origin)
    assert monthly.origin_index.ge(0).all()
    monthly["Y60"] = np.nan
    monthly["label_exit_date"] = pd.NaT
    for i, row in monthly.iterrows():
        t = int(row.origin_index)
        if row.all_features_valid and t + 61 < len(data):
            monthly.loc[i, "Y60"] = holding_total_return(data, dividends, t + 1, t + 61)[0]
            monthly.loc[i, "label_exit_date"] = data.date.iloc[t + 61]
    monthly.to_parquet(OUT / "monthly_features_and_mature_labels.parquet", index=False)
    originals = pd.read_parquet(SOURCE / "signals.parquet")
    assert data.date.tolist() == originals.date.tolist()
    mask = originals.event_mask.to_numpy(bool)
    events = monthly.loc[monthly.origin_index.isin(np.flatnonzero(mask))]
    assert len(events) == int(mask.sum())
    predictions = {model: np.full(len(data), np.nan) for model in MODELS}
    receipts = []
    for row in events.itertuples():
        t = int(row.origin_index)
        train = mature_training(monthly, row.origin)
        for key, columns in MODELS.items():
            valid = bool(row.all_features_valid) and len(train) >= config["minimum_train_samples"]
            receipt = {"model": key, "origin": str(row.origin.date()), "training_months": train.origin.dt.strftime("%Y-%m-%d").tolist(),
                       "training_samples": len(train), "latest_label_exit_date": str(train.label_exit_date.max()) if len(train) else None,
                       "columns": columns, "status": "TRAINED_MONTHLY_MODEL" if valid else "NO_VIEW_FEATURES_OR_MATURE_MONTHS_INSUFFICIENT", "prediction": None}
            if valid:
                assert train.origin.lt(row.origin).all() and train.label_exit_date.le(row.origin).all()
                model = make_pipeline(StandardScaler(), Ridge(alpha=config["ridge_alpha"]))
                model.fit(train[columns], train.Y60)
                sample = pd.DataFrame([{name: getattr(row, name) for name in columns}], columns=columns)
                value = float(model.predict(sample)[0])
                predictions[key][t] = value
                path = OUT / "fitted_models" / f"{key}_{row.origin:%Y%m%d}.joblib"
                path.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(model, path)
                receipt.update({"prediction": value, "model_file": identity(path)})
            receipts.append(receipt)
    save(OUT / "training_receipts.json", {"rows": receipts}, exclusive=True)
    pd.DataFrame({"date": data.date, "event_mask": mask, **predictions}).to_parquet(OUT / "signals.parquet", index=False)
    print("双机构三模型训练完成，保存拟合：", sum(r["status"] == "TRAINED_MONTHLY_MODEL" for r in receipts), flush=True)
    metrics, yearly, eras, checks, uncertainty = [], [], [], [], {}
    for cost_name, cost in config["costs"].items():
        accounts = {}
        for model in MODELS:
            ledger, decisions = simulate_event_account(data, dividends, config, cost, config["evaluation_start"], model,
                                                       prediction=predictions[model], horizon=60, event_mask=mask)
            outside = decisions.loc[~decisions.origin_index.isin(np.flatnonzero(mask))]
            assert outside.requested_quantity.eq(0).all()
            assert not ledger.terminal_unliquidated.iloc[-1]
            ledger.loc[ledger.filled_quantity.ne(0)].to_csv(OUT / f"{cost_name}_{model}_全部进出场成交.csv", index=False, encoding="utf-8-sig")
            save_account(OUT / "evaluation" / cost_name, model, ledger, decisions)
            accounts[model] = ledger
            checks.append({"cost": cost_name, "model": model, "new_account": True, "outside_month_end_orders": 0,
                           "max_accounting_error": float(ledger.accounting_error.abs().max())})
        for name, original in CONTROLS.items():
            ledger = pd.read_parquet(SOURCE / "evaluation" / cost_name / (original + "_ledger.parquet"))
            decisions = pd.read_parquet(SOURCE / "evaluation" / cost_name / (original + "_decisions.parquet"))
            save_account(OUT / "evaluation" / cost_name, name, ledger, decisions)
            accounts[name] = ledger
            checks.append({"cost": cost_name, "model": name, "new_account": False, "reused_saved_model": original})
        benchmark = summarize(accounts["BUY_HOLD"], config)
        for model, ledger in accounts.items():
            metric = {"cost": cost_name, "model": model, **summarize(ledger, config)}
            metric["annualized_return_excess_vs_buy_hold"] = metric["annualized_return"] - benchmark["annualized_return"]
            metric["meets_point_target"] = metric["net_sharpe"] is not None and metric["net_sharpe"] >= 1.2
            metrics.append(metric)
            for year, group in ledger.groupby(ledger.date.dt.year):
                yearly.append({"cost": cost_name, "model": model, "year": int(year), **summarize(group, config)})
            for label, start, end in [("2020—2021", "2020-01-01", "2021-12-31"), ("2022—2023", "2022-01-01", "2023-12-31"), ("2024—终点", "2024-01-01", config["data_cutoff"])]:
                eras.append({"cost": cost_name, "model": model, "era": label, **summarize(ledger.loc[ledger.date.between(start, end)], config)})
        returns = pd.DataFrame({"date": accounts["BUY_HOLD"].date, **{model: ledger.net_return.to_numpy() for model, ledger in accounts.items()}})
        returns.to_parquet(OUT / f"{cost_name}_all_evaluation_returns.parquet", index=False)
        uncertainty[cost_name] = {}
        for block in config["bootstrap_day_blocks"]:
            rng = np.random.default_rng(config["random_seed"])
            draws = []
            for _ in range(config["bootstrap_repetitions"]):
                ix = block_indices(rng, len(returns), block)
                draw = {}
                for model, control in PAIRS.items():
                    values = returns[model].to_numpy()[ix]
                    draw[model + "_sharpe"] = return_metrics(values, config["annual_days"])["net_sharpe"]
                    draw[model + "_minus_paired_control"] = float((values - returns[control].to_numpy()[ix]).mean() * config["annual_days"])
                    draw[model + "_minus_buy_hold"] = float((values - returns.BUY_HOLD.to_numpy()[ix]).mean() * config["annual_days"])
                draws.append(draw)
            saved = pd.DataFrame(draws)
            saved.to_parquet(OUT / f"{cost_name}_block{block}_saved_bootstrap_statistics.parquet", index=False)
            uncertainty[cost_name][str(block)] = {column + "_95_interval": interval(saved[column].dropna().tolist()) for column in saved}
        print("双机构前瞻盈利账户完成", cost_name, "三条新账户和两条复用对照", flush=True)
    frame = pd.DataFrame(metrics)
    frame.to_csv(OUT / "metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(yearly).to_csv(OUT / "yearly_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(eras).to_csv(OUT / "era_metrics.csv", index=False, encoding="utf-8-sig")
    save(OUT / "execution_checks.json", {"rows": checks}, exclusive=True)
    save(OUT / "uncertainty.json", uncertainty, exclusive=True)
    result = {"study_id": config["study_id"], "completed_at": now(), "status": "TWO_INSTITUTION_COMMON_CLOCK_COMPLETE_ACCOUNTS_FINISHED",
              "primary": frame.loc[frame.model.eq(config["primary"])].to_dict("records"), "all_metrics": metrics, "uncertainty": uncertainty,
              "candidate_configurations": 3, "evaluation_accounts": 10, "new_accounts_generated": 6, "reused_control_accounts": 4,
              "trained_models": sum(r["status"] == "TRAINED_MONTHLY_MODEL" for r in receipts),
              "valid_monthly_feature_origins": int(monthly.all_features_valid.sum()),
              "first_valid_feature_origin": str(monthly.loc[monthly.all_features_valid, "origin"].min()),
              "evaluation_month_end_events": len(events),
              "historical_point_target_met": bool(frame.loc[frame.model.eq(config["primary"]), "meets_point_target"].any()),
              "goal_achieved": False, "independent_validation": "NOT_ESTABLISHED_ALREADY_OBSERVED_HISTORY", "position_impact": 0}
    save(OUT / "result.json", result, exclusive=True)
    print(json.dumps({"status": result["status"], "primary": result["primary"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--freeze", action="store_true")
    actions.add_argument("--run", action="store_true")
    args = parser.parse_args()
    freeze() if args.freeze else run()
