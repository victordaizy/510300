"""冻结 Oracle 信息预算 V1.0.1 保守验收修正。"""

from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import sys
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "research") not in sys.path:
    sys.path.insert(0, str(ROOT / "research"))

from oracle_information_budget_v1_0_1_acceptance import (  # noqa: E402
    CONFIG_PATH,
    MANIFEST_PATH,
    PROJECT_ID,
    load_config,
    project_path,
    sha256_file,
)


def main() -> int:
    if MANIFEST_PATH.exists():
        print(f"V1.0.1保守验收冻结清单已存在，拒绝覆盖：{MANIFEST_PATH}", file=sys.stderr)
        return 2
    try:
        config = load_config(CONFIG_PATH)
    except Exception as exc:
        print(f"V1.0.1保守验收配置合同失败：{exc}", file=sys.stderr)
        return 2
    protocol = config["protocol"]
    if protocol["state"] != "READY_TO_FREEZE_BEFORE_CONSERVATIVE_SCENARIO_METRICS":
        print("V1.0.1协议尚未处于可冻结状态", file=sys.stderr)
        return 2
    if protocol["source_v1_outputs_known"] is not True:
        print("必须声明原V1结果已经可见", file=sys.stderr)
        return 2
    if protocol["source_robustness_audit_known"] is not True:
        print("必须声明Oracle稳健性审计已经可见", file=sys.stderr)
        return 2
    if protocol["conservative_scenario_metrics_computed_before_freeze"] is not False:
        print("冻结前已经计算V1.0.1保守场景指标", file=sys.stderr)
        return 2

    tracked_relatives = [
        "config/510300_oracle_information_budget_v1_0_1_acceptance.yaml",
        "docs/510300_ORACLE_INFORMATION_BUDGET_V1_0_1_ACCEPTANCE_SPEC.md",
        "research/oracle_information_budget_v1_0_1_acceptance.py",
        "scripts/freeze_510300_oracle_information_budget_v1_0_1_acceptance.py",
        "scripts/run_510300_oracle_information_budget_v1_0_1_acceptance.py",
        "tests/test_510300_oracle_information_budget_v1_0_1_acceptance.py",
    ]
    input_relatives = [
        record["path"]
        for section in ("source_v1", "source_atlas", "preexisting_up20_protocol")
        for record in config[section].values()
        if isinstance(record, dict) and "path" in record
    ]
    missing = [
        relative
        for relative in tracked_relatives + input_relatives
        if not project_path(relative).exists()
    ]
    if missing:
        print(f"V1.0.1冻结文件缺失：{missing}", file=sys.stderr)
        return 2
    source_hash_failures = {
        record["path"]: {
            "expected": record["sha256"],
            "actual": sha256_file(project_path(record["path"])),
        }
        for section in ("source_v1", "source_atlas", "preexisting_up20_protocol")
        for record in config[section].values()
        if isinstance(record, dict) and "path" in record
        if sha256_file(project_path(record["path"])) != record["sha256"]
    }
    if source_hash_failures:
        print(
            "V1.0.1冻结源哈希失败："
            + json.dumps(source_hash_failures, ensure_ascii=False),
            file=sys.stderr,
        )
        return 2
    existing_outputs = [
        value
        for value in config["paths"].values()
        if project_path(value).exists()
    ]
    if existing_outputs:
        print(f"V1.0.1冻结前已存在候选输出：{existing_outputs}", file=sys.stderr)
        return 2
    prior_manifests = sorted((ROOT / "config").glob("*manifest*.json"))
    minimum_prior = int(config["selection_bias"]["minimum_prior_manifest_count"])
    if len(prior_manifests) < minimum_prior:
        print(
            f"V1.0.1冻结前清单计数异常：至少{minimum_prior}，实际{len(prior_manifests)}",
            file=sys.stderr,
        )
        return 2

    manifest = {
        "schema_version": "1.0.1",
        "project_id": PROJECT_ID,
        "state": "FROZEN_BEFORE_CONSERVATIVE_SCENARIO_METRICS",
        "frozen_at_asia_shanghai": datetime.now(
            ZoneInfo(protocol["timezone"])
        ).isoformat(),
        "config_path": CONFIG_PATH.relative_to(ROOT).as_posix(),
        "config_sha256": sha256_file(CONFIG_PATH),
        "source_v1_outputs_known": True,
        "source_robustness_audit_known": True,
        "conservative_scenario_metrics_computed_before_freeze": False,
        "tracked_files": {
            relative: sha256_file(project_path(relative))
            for relative in tracked_relatives
        },
        "input_files": {
            relative: sha256_file(project_path(relative))
            for relative in input_relatives
        },
        "fixed_correction": {
            "target_net_sharpe": 1.2,
            "conservative_scenario_sharpe_definition": config[
                "frozen_measurement"
            ]["conservative_scenario_sharpe_definition"],
            "lo_bartlett_max_lag": 20,
            "random_seed": 51030020260831,
            "repetitions_per_random_cell": 500,
            "joint_grid": config["frozen_measurement"]["joint_grid"],
            "timing_error_days": list(range(11)),
            "reported_budget_fields": config["reported_budget_fields"],
            "nondegenerate_advancement_gate": config[
                "nondegenerate_advancement_gate"
            ],
        },
        "selection_bias_control": {
            "prior_manifest_count": len(prior_manifests),
            "total_manifest_count_including_current": len(prior_manifests) + 1,
            "source_outputs_known_and_declared": True,
            "correction_can_only_tighten_v1": True,
        },
        "no_rescue": {
            "ordinary_result": True,
            "error_grid": True,
            "random_seed": True,
            "adversarial_rule": True,
            "timing_grid": True,
            "serial_metric": True,
            "advancement_gate": True,
        },
        "governance": config["governance"],
    }
    temporary = MANIFEST_PATH.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, MANIFEST_PATH)
    print(
        json.dumps(
            {
                "status": manifest["state"],
                "project_id": PROJECT_ID,
                "manifest_path": MANIFEST_PATH.relative_to(ROOT).as_posix(),
                "manifest_sha256": sha256_file(MANIFEST_PATH),
                "source_v1_outputs_known": True,
                "source_robustness_audit_known": True,
                "conservative_scenario_metrics_computed_before_freeze": False,
                "prior_manifest_count": len(prior_manifests),
                "total_manifest_count_including_current": len(prior_manifests) + 1,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
