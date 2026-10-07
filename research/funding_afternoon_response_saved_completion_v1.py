"""恢复八个已保存账户的剩余核验，修正索引取值，不重跑策略账户。"""
from __future__ import annotations

from pathlib import Path
import sys
from types import FunctionType

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import research.funding_afternoon_response_daily_v1 as experiment
from research.selected_mix_reappraisal_v1 import read, save, now, digest
from research.selected_mix_daily_two_year_v1 import paired_interval
from research.strategy_review_diagnostics_v1 import metrics

DATA = experiment.OUT
OUT = ROOT / "reports/research/510300_funding_afternoon_response_saved_completion_v1"
COMPLETION_STUDY = "510300_FUNDING_AFTERNOON_RESPONSE_SAVED_COMPLETION_V1"


def run():
    (OUT / "results").mkdir(parents=True, exist_ok=True)
    failure = read(DATA / "RUN_FAILURE.json")
    assert failure["error"] == "AttributeError: 'DataFrame' object has no attribute 'idx'"
    assert not (DATA / "result.json").exists()
    save(OUT / "completion_plan.json", {"at": now(), "study_id": COMPLETION_STUDY,
         "original_failure": failure,
         "repair": "labels.set_index('idx')后idx是行索引；核验采用pool.index.to_numpy(int)，不改训练、预测、账户或统计定义。",
         "new_accounts": 0, "new_funding_fits": 0, "original_files_mutated": False,
         "scope": "复算已保存8账户和分布，完成原定两个未来标签隔离检查及同一种子区间。"}, True)
    frozen = read(DATA / "freeze.json")
    assert digest(DATA / "protocol.json") == frozen["protocol_sha256"]
    for path, sha in frozen["sources"].items():
        assert digest(ROOT / path) == sha, path
    artifacts = {p.relative_to(ROOT).as_posix(): digest(p) for p in DATA.rglob("*") if p.is_file()}
    save(OUT / "original_artifacts.json", artifacts, True)
    pred = pd.read_parquet(DATA / "results/predictions.parquet")
    models = read(DATA / "results/saved_distributions.json")
    x = pd.read_parquet(DATA / "inputs/decision_information.parquet")
    response = pd.read_parquet(DATA / "inputs/afternoon_response.parquet")
    labels = pd.read_parquet(experiment.parent.OUT / "inputs/mature_labels.parquet")
    nested = pd.read_parquet(experiment.prior.OUT / "results/nested_funding_predictions.parquet")
    source = pd.read_parquet(experiment.prior.OUT / "inputs/funding_features.parquet")
    schedule = pd.read_parquet(DATA / "results/update_receipts.parquet")
    sample, lookup = labels.set_index("idx"), pred.set_index(["idx", "model"])
    for record in models:
        i = record["idx"]
        pool = sample.loc[record["training_indices"]]
        assert pool.date.ge(x.date.iloc[i]-pd.DateOffset(years=2)).all() and pool.exit_idx.lt(i).all()
        assert x.common_known.iloc[pool.index.to_numpy(int)].all()
        values = experiment.distribution.empirical_statistics(sample.loc[record["selected_indices"], "gross_return5"])
        for key in ["mu5", "variance5", "q05", "es95"]:
            np.testing.assert_allclose(values[key], lookup.loc[(i, record["model"]), key], atol=1e-14, rtol=0)
    verified_dates = []
    for day in ["2023-06-30", "2025-09-24"]:
        i = int(x.date.searchsorted(day, side="right"))-1
        restricted = labels[labels.date.le(x.date.iloc[i])].copy()
        restricted.loc[restricted.exit_idx.ge(i), "gross_return5"] = -9999.
        p, _, _ = experiment.fit_day(i, x.iloc[:i+1], nested[nested.decision_idx.eq(i)], len(source), restricted)
        assert len(p) == len(experiment.MODELS)
        for row in p:
            for key in ["mu5", "variance5", "q05", "es95"]:
                assert row[key] == lookup.loc[(i, row["model"]), key]
        verified_dates.append(day)
    original = experiment.prior.forecast_evaluation
    evaluate = FunctionType(original.__code__, {**original.__globals__, "OUT": OUT, "PRIMARY": experiment.PRIMARY, "RAW": experiment.RESPONSE},
                            original.__name__, original.__defaults__, original.__closure__)
    forecast, scored, monitor = evaluate(pred, labels, x)
    for record in forecast:
        record["primary_mse_improvement_vs_afternoon_only"] = record.pop("primary_mse_improvement_vs_raw")
    measures = read(DATA / "results/account_metrics.json")
    rolling = pd.read_parquet(DATA / "results/rolling_two_years.parquet")
    accounts, account_checks, cycle_rows = {}, [], []
    for name in ["PRICE", experiment.FUNDING, experiment.RESPONSE, experiment.PRIMARY]:
        for cost in experiment.distribution.COSTS:
            folder = DATA / "accounts" / cost / name
            ledger = pd.read_parquet(folder / "ledger.parquet")
            decisions = pd.read_parquet(folder / "decisions.parquet")
            accounts[name, cost] = ledger
            stored = next(row for row in measures if row["policy"] == name and row["cost"] == cost)
            for key, value in metrics(ledger).items():
                if value is None:
                    assert stored[key] is None
                else:
                    np.testing.assert_allclose(value, stored[key], rtol=1e-12, atol=1e-12)
            account_checks.append({"model": name, "cost": cost, **experiment.parent.account_verification(ledger, decisions)})
            closed = pd.read_parquet(folder / "cycles.parquet")
            pending = read(folder / "pending_cycle.json")
            cycle_rows.append({"model": name, "cost": cost, "closed": len(closed), "open": pending is not None})
    rng = np.random.default_rng(2026092601)
    comparisons, factorial = [], []
    for cost in experiment.distribution.COSTS:
        left = accounts[experiment.PRIMARY, cost]
        for control in [experiment.RESPONSE, experiment.FUNDING, "PRICE"]:
            right = accounts[control, cost]
            periods = []
            for lo, hi in [(experiment.START, "2023-12-31"), ("2024-01-01", experiment.END)]:
                mask = left.date.between(lo, hi)
                periods.append({"start": lo, "end": hi,
                                "annual_arithmetic_difference": float((left.loc[mask, "net_return"]-right.loc[mask, "net_return"]).mean()*242)})
            comparisons.append({"cost": cost, "control": control, **paired_interval(left, right, rng), "fixed_periods": periods})
        daily = left.net_return-accounts[experiment.RESPONSE, cost].net_return-accounts[experiment.FUNDING, cost].net_return+accounts["PRICE", cost].net_return
        factorial.append({"cost": cost, **paired_interval(pd.DataFrame({"net_return": daily}), pd.DataFrame({"net_return": np.zeros(len(daily))}), rng), "causal_identification": False})
    for path, sha in artifacts.items():
        assert digest(ROOT / path) == sha, path
    primary = next(row for row in measures if row["policy"] == experiment.PRIMARY and row["cost"] == "STRESS")
    increment = all(row["lower_95"] > 0 and all(era["annual_arithmetic_difference"] > 0 for era in row["fixed_periods"]) for row in comparisons if row["cost"] == "STRESS")
    updated = schedule[schedule.status.eq("UPDATED")]
    ready = pred[["idx", "date"]].drop_duplicates()
    result = {"at": now(), "study_id": experiment.STUDY, "completion_study_id": COMPLETION_STUDY,
              "status": "DEVELOPMENT_CANDIDATE_REQUIRES_INDEPENDENT_VALIDATION" if primary["historical_point_targets_met"] and increment else "FROZEN_NO_QUALIFIED_FUNDING_RESPONSE_STRATEGY",
              "primary": primary, "all_accounts": measures, "comparisons": comparisons, "factorial_diagnostic": factorial,
              "forecast_evaluation": forecast, "new_full_accounts": 8, "new_accounts_in_completion": 0,
              "new_funding_regression_fits": 0, "reused_nested_funding_predictions_each_model": len(nested),
              "daily_distribution_updates": len(models), "unique_prediction_days": len(ready),
              "first_prediction_date": ready.date.min(), "last_prediction_date": ready.date.max(),
              "mature_prediction_days": int(scored[scored.gross_return5.notna()].idx.nunique()),
              "source_response_pairs": len(response), "source_first_date": response.source_date.min(), "source_last_date": response.source_date.max(),
              "schedule_statuses": schedule.status.value_counts().to_dict(),
              "training_rows_min": int(updated.n_train.min()), "training_rows_max": int(updated.n_train.max()),
              "account_checks": account_checks, "completed_and_pending_cycles": cycle_rows,
              "saved_distributions_recomputed": len(models), "future_label_exclusion_dates": verified_dates,
              "primary_rolling_two_year_joint_passes": int(rolling.loc[rolling.policy.eq(experiment.PRIMARY)&rolling.cost.eq("STRESS"), "joint_point_pass"].sum()),
              "monitor_alerts": monitor.groupby("model").alert.sum().to_dict(), "increment_gate_met": increment,
              "original_failure_preserved": True, "original_files_unchanged": len(artifacts),
              "repair_changed_strategy_or_account_values": False,
              "new_parameter_searches": 0, "new_market_downloads": 0, "historical_first_vintage_authenticated": False,
              "new_independent_forward_observations": 0, "goal_achieved": False, "orders_authorized": False}
    save(OUT / "result.json", result, True)
    print(f"剩余核验已完成：{len(models)}份分布复算一致、8账户保持原值，索引错误未改变策略结果。", flush=True)


if __name__ == "__main__":
    run()
