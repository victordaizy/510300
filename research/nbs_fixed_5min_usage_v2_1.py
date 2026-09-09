"""执行明确授权的 NBS 五分钟用途准入，复用冻结 G2 并保持逐级停止线。"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from research.nbs_v2_common import (
    ContractError, committed, csv_bytes, git, identity, load_json, now, verify, write_once,
)
from research.nbs_v2_g2_engine import StatisticalNoView, construct_labels, g2_statistics, prequential
from research.stk_mins_source_admission_v2 import normalize

CONFIG = "config/510300_nbs_fixed_5min_usage_v2_1.json"
AUTH = "docs/510300_NBS_FIXED_5MIN_USAGE_AUTHORIZATION_20260905.md"
MODEL = "config/510300_nbs_1000_negative_information_drift_v2.yaml"
BASE_SOURCE = "config/510300_stk_mins_source_admission_v2.yaml"
BASE_REPORT = "reports/data_quality/510300_nbs_v2_source_refetch_v1_2_20260905/source_adjudication.json"
SOURCE_OUT = "reports/data_quality/510300_nbs_fixed_5min_usage_v2_1"
OUT = "reports/research/510300_nbs_fixed_5min_usage_v2_1"
MANIFEST = f"{SOURCE_OUT}/protocol_implementation_manifest.json"
SOURCE_RESULT = f"{SOURCE_OUT}/source_adjudication.json"
PARENT_PLAN = "reports/research/510300_nbs_v2_g2_execution_v1/event_baseline_placebo_plan_pre_return.csv"
PLAN = f"{OUT}/event_baseline_placebo_plan_pre_return.csv"
PASS = "PASS_NBS_FIXED_5MIN_USAGE_ADMISSION_V2_1"
SCOPE = [CONFIG, AUTH, "research/nbs_fixed_5min_usage_v2_1.py",
         "scripts/run_510300_nbs_fixed_5min_usage_v2_1.py",
         "tests/test_510300_nbs_fixed_5min_usage_v2_1.py"]


def fixed_windows(source: dict, model: dict) -> dict:
    result = copy.deepcopy(source["windows"])
    result.update({f"shift_{n}": model["placebos"]["B"][f"{n}_labels"] for n in ["pre", "reaction", "entry", "exit"]})
    if len(result) != 8 or any(len(x) != 5 or len(set(x)) != 5 for x in result.values()):
        raise ContractError("准入只接受原主实验和已授权安慰剂 B 的八个固定五分钟窗口")
    return result


def window_measurements(minute: pd.DataFrame, dates: pd.DatetimeIndex, windows: dict, tolerance: float) -> pd.DataFrame:
    """各窗口独立测量，禁止在数据审核阶段构造跨窗口收益。"""
    frame = normalize(minute)
    frame["valid_measurement"] = (np.isfinite(frame[["vol", "amount", "low", "high"]]).all(axis=1)
        & frame.vol.ge(0) & frame.amount.ge(0) & frame.vol.eq(0).eq(frame.amount.eq(0))
        & frame.low.gt(0) & frame.high.ge(frame.low))
    rows = []
    for name, labels in windows.items():
        part = frame.loc[frame.clock.isin(labels)]
        values = part.groupby("date").agg(
            rows=("clock", "size"), unique_labels=("clock", "nunique"), valid=("valid_measurement", "all"),
            volume_shares=("vol", "sum"), amount_cny=("amount", "sum"),
            minimum_low=("low", "min"), maximum_high=("high", "max"))
        values = values.reindex(dates)
        values["complete"] = values.rows.eq(5) & values.unique_labels.eq(5) & values.valid.eq(True)
        values["positive_totals"] = values.volume_shares.gt(0) & values.amount_cny.gt(0)
        values["vwap"] = values.amount_cny / values.volume_shares.where(values.positive_totals)
        values["inside_range"] = (values.vwap.ge(values.minimum_low - tolerance)
                                  & values.vwap.le(values.maximum_high + tolerance))
        values["pass"] = values.complete & values.positive_totals & values.inside_range
        values["window"] = name
        values.index.name = "date"
        rows.append(values.reset_index())
    return pd.concat(rows, ignore_index=True)


def inherited_checks(prior: dict) -> tuple[dict, dict]:
    """只移除用户明确授权调整的两项单分钟检查，其余原门全部保留。"""
    removed = {k: prior["checks"][k] for k in ["all_bar_vwap_defined", "all_bar_vwap_inside_low_high"]}
    kept = {k: v for k, v in prior["checks"].items() if k not in removed}
    return kept, removed


def admit(report: dict, loader):
    if (report.get("state") != PASS or report.get("source_pass") is not True
            or report.get("g1_pass") is not True or report.get("scope") != "EXACT_EIGHT_FIXED_5MIN_WINDOWS_ONLY"
            or report.get("explicit_user_authorization") is not True
            or not report.get("checks") or not all(report["checks"].values())):
        raise ContractError("五分钟用途准入或 G1 未通过，禁止读取事件收益")
    return loader()


def freeze(root: Path) -> dict:
    cfg = load_json(root / CONFIG)
    model = yaml.safe_load((root / MODEL).read_text("utf-8"))
    source = yaml.safe_load((root / BASE_SOURCE).read_text("utf-8"))
    verify(root, load_json(root / "config/510300_stk_mins_source_admission_v2_manifest.json")["identities"])
    prior_manifest = load_json(root / "reports/research/510300_nbs_v2_g2_execution_v1/implementation_manifest.json")
    verify(root, prior_manifest["identities"])
    parent_stop = load_json(root / "reports/research/510300_nbs_v2_g2_execution_v1/gate_check_receipt.json")
    if parent_stop["event_return_values_read"] or parent_stop["event_labels_created"] != 0:
        raise ContractError("父执行版本已读取真实标签，不得登记为首次结果读取")
    if (root / "reports/research/510300_nbs_v2_g2_execution_v1/return_read_claim.json").exists():
        raise ContractError("父版本已有真实收益读取 claim，停止")
    prior = load_json(root / BASE_REPORT)
    verify(root, [prior["candidate"]])
    if cfg["fixed_windows"] != fixed_windows(source, model):
        raise ContractError("用途合同窗口与原策略不一致")
    if cfg["authorization_quote"] != "明确批准此前的五分钟用途合同 ，然后请继续推进":
        raise ContractError("缺少本轮明确来源合同授权")
    write_once(root / PLAN, (root / PARENT_PLAN).read_bytes())
    paths = SCOPE + [MODEL, BASE_SOURCE, BASE_REPORT, PARENT_PLAN, PLAN,
        "docs/510300_NBS_V2_DATA_REMEDIATION_REVIEW_20260905.md",
        "research/nbs_v2_common.py", "research/stk_mins_source_admission_v2.py", "research/nbs_v2_g2_engine.py",
        "tests/test_510300_nbs_v2_g2.py",
        "config/510300_stk_mins_source_admission_v2_manifest.json",
        "reports/research/510300_nbs_v2_g2_execution_v1/implementation_manifest.json",
        "reports/research/510300_nbs_v2_g2_execution_v1/gate_check_receipt.json",
        "reports/data_quality/510300_nbs_v2_source_refetch_v1_2_20260905/manifest.json",
        "reports/data_quality/510300_nbs_v2_source_refetch_v1_2_20260905/acquisition_receipt.json",
        prior["candidate"]["path"]]
    result = {"state": "FROZEN_AUTHORIZED_5MIN_USAGE_BEFORE_FIRST_EVENT_RETURN_READ", "frozen_at": now(),
              "authorization": identity(root, AUTH), "identities": [identity(root, p) for p in sorted(set(paths))],
              "event_return_values_read": False, "strategy_numerical_core_changed": False,
              "prior_bar_contract_stays_failed": True, "position_impact": 0}
    write_once(root / MANIFEST, result)
    return {k: v for k, v in result.items() if k != "identities"}


def validate(root: Path) -> tuple[dict, dict, dict]:
    verify(root, load_json(root / MANIFEST)["identities"])
    verify(root, load_json(root / "config/510300_stk_mins_source_admission_v2_manifest.json")["identities"])
    verify(root, load_json(root / "reports/data_quality/510300_nbs_v2_source_refetch_v1_2_20260905/manifest.json")["identities"])
    committed(root, SCOPE + [MANIFEST, PLAN])
    return (load_json(root / CONFIG), yaml.safe_load((root / BASE_SOURCE).read_text("utf-8")),
            yaml.safe_load((root / MODEL).read_text("utf-8")))


def source_admission(root: Path) -> dict:
    cfg, source, model = validate(root)
    if (root / SOURCE_RESULT).exists():
        raise ContractError("用途准入已有不可变回执，请读取既有结果")
    prior = load_json(root / BASE_REPORT)
    verify(root, [prior["candidate"]])
    acq = load_json(root / source["inputs"]["acquisition"])
    cal_rec = load_json(root / acq["records"][0]["receipt_relative_path"])
    cal = pd.read_parquet(root / cal_rec["normalized_relative_path"])
    dates = pd.DatetimeIndex(pd.to_datetime(cal.loc[cal.is_open.astype(int).eq(1), "cal_date"].astype(str))).sort_values()
    minute = pd.read_parquet(root / prior["candidate"]["path"])
    q = source["quality_gates"]
    tolerance = q["bar_vwap_price_tolerance_cny"] + q["binary_float_tolerance_cny"]
    windows = fixed_windows(source, model)
    measured = window_measurements(minute, dates, windows, tolerance)
    overlap = dates[(dates >= q["independent_overlap_start"]) & (dates <= q["independent_overlap_end"])]
    independent = window_measurements(pd.read_parquet(root / source["inputs"]["independent_minute"]), overlap, windows, tolerance)
    pairs = measured.merge(independent, on=["date", "window"], suffixes=("_new", "_old"), validate="one_to_one")
    pairs["vwap_match"] = (pairs.complete_new & pairs.complete_old & pairs.positive_totals_new & pairs.positive_totals_old
        & np.isclose(pairs.vwap_new, pairs.vwap_old, atol=q["independent_vwap_absolute_tolerance_cny"],
                     rtol=q["independent_vwap_relative_tolerance"]))
    checks, bar_diagnostics = inherited_checks(prior)
    checks.update({"all_minute_volume_amount_finite_nonnegative": bool(np.isfinite(minute[["vol", "amount"]]).all().all()
                          and minute.vol.ge(0).all() and minute.amount.ge(0).all()),
                   "all_minute_zero_volume_amount_paired": bool(minute.vol.eq(0).eq(minute.amount.eq(0)).all()),
                   "all_eight_windows_on_all_open_days": len(measured) == len(dates) * 8 and measured["pass"].all(),
                   "independent_eight_window_vwap": len(pairs) == len(overlap) * 8
                          and pairs.vwap_match.mean() >= q["independent_four_window_vwap_match_minimum"]})
    ledger = pd.read_csv(root / model["sample"]["ledger"])
    eligible = pd.to_datetime(ledger.loc[ledger.final_event_eligibility.eq(True), "scheduled_date"])
    on_event = measured.loc[measured.date.isin(eligible)]
    checks["all_event_eight_windows"] = len(on_event) == 88 * 8 and on_event["pass"].all()
    plan = pd.read_csv(root / PLAN)
    counts = plan.era.value_counts().to_dict()
    g1 = (len(plan) == 86 and len(eligible) == 88 and counts == {"TRAINING_ORIGIN": 36, "ERA_1": 17, "ERA_2": 17, "ERA_3": 16})
    source_pass = bool(all(checks.values()))
    result = {"state": PASS if source_pass else "BLOCKED_NBS_FIXED_5MIN_USAGE_V2_1", "generated_at": now(),
              "scope": "EXACT_EIGHT_FIXED_5MIN_WINDOWS_ONLY", "source_pass": source_pass, "g1_pass": g1,
              "explicit_user_authorization": True, "authorization": identity(root, AUTH), "contract": identity(root, CONFIG),
              "checks": checks, "original_bar_checks_diagnostic_only": bar_diagnostics,
              "original_bar_contract_pass": False, "bar_vwap_out_of_range_rows": prior["bar_vwap_out_of_range_rows"],
              "zero_volume_zero_amount_rows": int((minute.vol.eq(0) & minute.amount.eq(0)).sum()),
              "original_source_checks_reused_from_verified_identical_inputs": identity(root, BASE_REPORT),
              "minute_rows": len(minute), "open_days": len(dates), "fixed_window_count": len(measured),
              "passing_fixed_windows": int(measured["pass"].sum()), "eligible_events": len(eligible),
              "complete_event_fixed_windows": int(on_event["pass"].sum()), "model_events": len(plan), "era_counts": counts,
              "independent_capture_window_count": len(pairs), "independent_matching_windows": int(pairs.vwap_match.sum()),
              "independent_capture_same_provider": True, "independent_provider_claimed": False,
              "maximum_volume_relative_error": prior["maximum_volume_relative_error"],
              "maximum_amount_relative_error": prior["maximum_amount_relative_error"],
              "candidate": prior["candidate"], "freeze_commit": git(root, "rev-parse", "HEAD"),
              "event_return_values_read": False, "event_labels_created": 0, "position_impact": 0}
    write_once(root / SOURCE_OUT / "fixed_window_measurements.csv", csv_bytes(measured))
    write_once(root / SOURCE_OUT / "independent_capture_window_comparison.csv", csv_bytes(pairs))
    result["outputs"] = [identity(root, f"{SOURCE_OUT}/{p}") for p in ["fixed_window_measurements.csv", "independent_capture_window_comparison.csv"]]
    write_once(root / SOURCE_RESULT, result)
    return result


def run_g2(root: Path) -> dict:
    cfg, source, model = validate(root)
    report = load_json(root / SOURCE_RESULT)
    admit(report, lambda: None)
    verify(root, [report["candidate"]] + report["outputs"])
    committed(root, [SOURCE_RESULT])
    claim_path = root / OUT / "return_read_claim.json"
    if claim_path.exists():
        raise ContractError("已领取本次唯一真实收益读取 claim，禁止重复运行")
    claim = {"state": "CLAIMED_ONCE_AFTER_AUTHORIZED_G0_G1_PASS", "claimed_at": now(),
             "authorization": identity(root, AUTH), "source": identity(root, SOURCE_RESULT),
             "candidate": report["candidate"], "implementation": identity(root, MANIFEST),
             "freeze_commit": git(root, "rev-parse", "HEAD")}
    write_once(claim_path, claim)
    completed_labels = 0
    try:
        arms = admit(report, lambda: construct_labels(pd.read_parquet(root / report["candidate"]["path"]),
                        pd.read_csv(root / PLAN), source, model))
        for arm, frame in arms.items():
            write_once(root / OUT / f"labels_{arm}.csv", csv_bytes(frame))
            completed_labels += len(frame)
        predictions = {arm: prequential(frame, model["models"]["initial_training"]) for arm, frame in arms.items()}
        stats, draws = g2_statistics(predictions, model["statistics"]["bootstrap_repetitions"], model["statistics"]["seed"])
        outputs = []
        for arm, frame in predictions.items():
            path = f"{OUT}/predictions_{arm}.csv"
            write_once(root / path, csv_bytes(frame))
            outputs.append(identity(root, path))
        write_once(root / OUT / "paired_bootstrap_slopes.csv", csv_bytes(draws))
        stats.update({"state": "PASS_G2_READY_FOR_FROZEN_G3" if stats["g2_pass"] else "REJECTED_FROZEN_NBS_FAMILY_NO_RESCUE",
                      "generated_at": now(), "event_return_values_read": True, "labels_created": completed_labels,
                      "numerical_core_changed": False, "model_contract": identity(root, MODEL),
                      "G0": "PASS_FIXED_5MIN_USAGE", "G1": "PASS_36_17_17_16", "G3": "NOT_RUN", "G4": "NOT_RUN",
                      "signal_density_read": False, "position_impact": 0, "claim": identity(root, claim_path),
                      "outputs": outputs + [identity(root, f"{OUT}/paired_bootstrap_slopes.csv")],
                      "next_mode": "FROZEN_G3" if stats["g2_pass"] else "STRICT_FORWARD_ONLY"})
    except StatisticalNoView as exc:
        stats = {"state": "NO_VIEW_FROZEN_STATISTICAL_DEGENERACY", "g2_pass": False, "generated_at": now(),
                 "reason": str(exc), "return_access_consumed": True, "label_construction_started": True,
                 "completed_label_rows_saved": completed_labels, "G3": "NOT_RUN", "G4": "NOT_RUN",
                 "signal_density_read": False, "position_impact": 0, "next_mode": "STRICT_FORWARD_ONLY",
                 "claim": identity(root, claim_path)}
    write_once(root / OUT / "G2_adjudication.json", stats)
    return stats


def g3_statistics(prediction: pd.DataFrame, model: dict) -> tuple[dict, pd.DataFrame]:
    gates = model["gates"]["G3"]
    selected = prediction.loc[prediction.X.gt(0) & prediction.beta.gt(0) & prediction.lcb.ge(.0042)].copy()
    counts = selected.era.value_counts().to_dict()
    density = len(selected) >= gates["minimum_independent_signals"] and sum(n >= 3 for n in counts.values()) >= gates["eras_with_3_signals_minimum"]
    result = {"signal_count": len(selected), "signal_counts_by_era": counts, "density_pass": density,
              "frozen_lcb_threshold": .0042, "signal_density_read": True}
    if not density:
        result.update({"state": "NO_VIEW_INSUFFICIENT_HIGH_MARGIN_SIGNAL_DENSITY", "g3_pass": False,
                       "net_margin_statistics": "NOT_RUN_AFTER_DENSITY_FAILURE"})
        return result, selected
    selected["base_net_avoided_loss"] = selected.Y - gates["base_round_trip_reference_cost"]
    selected["stress_net_avoided_loss"] = selected.Y - gates["stress_round_trip_reference_cost"]
    values = selected.stress_net_avoided_loss.to_numpy()
    indices = np.random.default_rng(model["statistics"]["seed"]).integers(0, len(values),
                    size=(model["statistics"]["bootstrap_repetitions"], len(values)))
    lower = float(np.quantile(values[indices].mean(axis=1), .10))
    total = float(values.sum())
    share = float(max(values.max(), 0) / total) if total > 0 else None
    checks = {"signal_density": density, "base_net_mean_positive": selected.base_net_avoided_loss.mean() > 0,
              "stress_net_mean_positive": values.mean() > 0, "stress_bootstrap_lower_positive": lower > 0,
              "stress_net_hit_rate_at_least_60pct": (values > 0).mean() >= gates["stress_net_hit_rate_minimum"],
              "single_event_share_at_most_30pct": share is not None and share <= gates["maximum_single_event_share_of_total_net"]}
    result.update({"g3_pass": all(checks.values()), "checks": checks,
                   "state": "PASS_G3_READY_FOR_FROZEN_G4" if all(checks.values()) else "REJECTED_G3_NET_MARGIN_OR_CONCENTRATION",
                   "base_mean_net_avoided_loss": selected.base_net_avoided_loss.mean(), "stress_mean_net_avoided_loss": values.mean(),
                   "stress_mean_bootstrap_lower_10pct": lower, "stress_net_hit_rate": (values > 0).mean(),
                   "maximum_single_event_share": share})
    return result, selected


def run_g3(root: Path) -> dict:
    _, _, model = validate(root)
    source_result = load_json(root / SOURCE_RESULT)
    admit(source_result, lambda: None)
    g2 = load_json(root / OUT / "G2_adjudication.json")
    if g2.get("g2_pass") is not True or g2.get("state") != "PASS_G2_READY_FOR_FROZEN_G3":
        raise ContractError("G2 未通过，禁止读取 42bp 信号密度或账户收益")
    if (root / OUT / "G3_adjudication.json").exists():
        raise ContractError("G3 已执行，禁止重复运行")
    verify(root, g2["outputs"])
    result, signals = g3_statistics(pd.read_csv(root / OUT / "predictions_MAIN.csv"), model)
    write_once(root / OUT / "G3_signals.csv", csv_bytes(signals))
    result.update({"generated_at": now(), "G4": "NOT_RUN", "position_impact": 0,
                   "next_mode": "FROZEN_G4" if result["g3_pass"] else "STRICT_FORWARD_ONLY",
                   "g2": identity(root, f"{OUT}/G2_adjudication.json"),
                   "signals": identity(root, f"{OUT}/G3_signals.csv")})
    write_once(root / OUT / "G3_adjudication.json", result)
    return result
