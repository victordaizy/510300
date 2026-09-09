"""在新模型结果生成前冻结510300 UP20预测V1.1。"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "research") not in sys.path:
    sys.path.insert(0, str(ROOT / "research"))

from up20_rare_event_forecast_v1_1_full_breadth import (  # noqa: E402
    CONFIG_PATH,
    MANIFEST_PATH,
    PROJECT_ID,
    load_config,
    sha256_file,
    validate_all_inputs,
)


def main() -> int:
    if MANIFEST_PATH.exists():
        print(f"V1.1冻结清单已存在，拒绝覆盖：{MANIFEST_PATH}", file=sys.stderr)
        return 2
    try:
        config = load_config(CONFIG_PATH)
        validate_all_inputs(config)
    except Exception as exc:
        print(f"V1.1协议或输入合同失败：{exc}", file=sys.stderr)
        return 2
    protocol = config["protocol"]
    if protocol["state"] != "READY_TO_FREEZE_BEFORE_MODEL_OUTCOME_CALCULATION":
        print("V1.1协议尚未处于可冻结状态", file=sys.stderr)
        return 2
    if protocol["parent_v1_result_known"] is not True:
        print("必须声明父版本负结果已经可见", file=sys.stderr)
        return 2
    if protocol["new_model_outcomes_read_before_freeze"] is not False:
        print("V1.1新模型结果在冻结前已被读取", file=sys.stderr)
        return 2

    tracked_relatives = [
        "config/510300_up20_rare_event_forecast_v1_1_full_breadth.yaml",
        "docs/510300_UP20_RARE_EVENT_FORECAST_V1_1_FULL_BREADTH_SPEC.md",
        "research/up20_rare_event_forecast_v1_1_full_breadth.py",
        "scripts/build_510300_up20_rare_event_forecast_v1_1_full_breadth_features.py",
        "scripts/freeze_510300_up20_rare_event_forecast_v1_1_full_breadth.py",
        "scripts/run_510300_up20_rare_event_forecast_v1_1_full_breadth.py",
        "tests/test_510300_up20_rare_event_forecast_v1_1_full_breadth.py",
    ]
    input_relatives: list[str] = []
    for group_name in ("inputs", "parent_protocol", "additional_inputs"):
        for receipt in config[group_name].values():
            if isinstance(receipt, dict) and "path" in receipt and "sha256" in receipt:
                input_relatives.append(receipt["path"])
    for receipt in config["source_results"].values():
        if isinstance(receipt, dict) and "path" in receipt and "sha256" in receipt:
            input_relatives.append(receipt["path"])
    input_relatives.extend(
        [config["paths"]["features_prefreeze"], config["paths"]["feature_audit"]]
    )
    input_relatives = list(dict.fromkeys(input_relatives))
    missing = [
        relative
        for relative in tracked_relatives + input_relatives
        if not (ROOT / relative).exists()
    ]
    if missing:
        print(f"V1.1冻结文件缺失：{missing}", file=sys.stderr)
        return 2

    audit_path = ROOT / config["paths"]["feature_audit"]
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != (
        "PASS_OUTCOME_FREE_FULL_PERIOD_OFFICIAL_BREADTH_READY_TO_FREEZE"
    ):
        print("V1.1冻结前特征审计未通过", file=sys.stderr)
        return 2
    if (
        audit.get("new_model_outcomes_read") is not False
        or audit.get("future_return_created") is not False
        or audit.get("feature_contract", {}).get("future_columns_present") is not False
        or audit.get("feature_contract", {}).get("labels_present") is not False
        or audit.get("feature_contract", {}).get("external_membership_used") is not False
        or audit.get("feature_contract", {}).get("historical_weights_used") is not False
        or audit.get("feature_contract", {}).get("model_or_threshold_changed_from_parent") is not False
    ):
        print("V1.1冻结前特征越过结果、成员、权重或模型边界", file=sys.stderr)
        return 2

    formal_keys = (
        "daily_predictions",
        "primary_blocks",
        "phase_results",
        "primary_base_ledger",
        "primary_base_trades",
        "primary_stress_ledger",
        "primary_stress_trades",
        "result_json",
        "result_markdown",
    )
    existing = [
        config["paths"][key]
        for key in formal_keys
        if (ROOT / config["paths"][key]).exists()
    ]
    if existing:
        print(f"V1.1冻结前已存在新模型或账户结果：{existing}", file=sys.stderr)
        return 2

    prior_manifests = sorted((ROOT / "config").glob("*manifest*.json"))
    expected_prior = int(config["selection_bias"]["prior_manifest_count"])
    if len(prior_manifests) != expected_prior:
        print(
            f"V1.1冻结前清单计数异常：期望{expected_prior}，实际{len(prior_manifests)}",
            file=sys.stderr,
        )
        return 2
    total_trials = len(prior_manifests) + 1
    if total_trials != int(
        config["selection_bias"]["expected_total_trial_count_including_current"]
    ):
        print("V1.1累计试验计数异常", file=sys.stderr)
        return 2

    manifest = {
        "schema_version": "1.0.0",
        "project_id": PROJECT_ID,
        "state": "FROZEN_BEFORE_NEW_MODEL_OUTCOME_CALCULATION",
        "frozen_at_asia_shanghai": datetime.now(
            ZoneInfo(config["protocol"]["timezone"])
        ).isoformat(),
        "config_path": CONFIG_PATH.relative_to(ROOT).as_posix(),
        "config_sha256": sha256_file(CONFIG_PATH),
        "parent_v1_result_known": True,
        "parent_v1_rejection_preserved": True,
        "new_model_outcomes_computed_before_freeze": False,
        "only_permitted_change": config["protocol"]["only_permitted_change"],
        "tracked_files": {
            relative: sha256_file(ROOT / relative) for relative in tracked_relatives
        },
        "input_files": {
            relative: sha256_file(ROOT / relative) for relative in input_relatives
        },
        "fixed_parent_protocol": {
            "features": config["model"]["features"],
            "model": config["model"],
            "decision_rule": config["decision_rule"],
            "evaluation": config["evaluation"],
            "historical_acceptance_gates": config["historical_acceptance_gates"],
            "feature_modules": config["feature_modules"],
        },
        "selection_bias_control": {
            "prior_manifest_count": len(prior_manifests),
            "total_trial_count_including_current": total_trials,
            "parent_v1_result_known_and_declared": True,
            "no_post_result_parameter_selection": True,
            "data_coverage_change_only": True,
        },
        "no_rescue": {
            "model": True,
            "regularization": True,
            "probability_thresholds": True,
            "stress_veto": True,
            "holding_period": True,
            "phase": True,
            "costs": True,
            "controls_cannot_replace_primary": True,
        },
        "governance": {
            "historical_research_only": True,
            "paper_signal_allowed": False,
            "shadow_signal_allowed": False,
            "position_mapping_enabled": False,
            "order_generation": False,
            "broker_connection": False,
            "position_change": False,
            "live_trading_authorized": False,
        },
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": manifest["state"],
                "manifest": MANIFEST_PATH.relative_to(ROOT).as_posix(),
                "manifest_sha256": sha256_file(MANIFEST_PATH),
                "prior_manifest_count": len(prior_manifests),
                "total_trial_count_including_current": total_trials,
                "parent_v1_result_known": True,
                "new_model_outcomes_computed_before_freeze": False,
                "live_trading_authorized": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

