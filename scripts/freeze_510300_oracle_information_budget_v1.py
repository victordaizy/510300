"""冻结510300未来20日上涨机会信息质量预算V1。"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "research") not in sys.path:
    sys.path.insert(0, str(ROOT / "research"))

from oracle_information_budget_v1 import (  # noqa: E402
    CONFIG_PATH,
    MANIFEST_PATH,
    PROJECT_ID,
    load_config,
    sha256_file,
)


def main() -> int:
    if MANIFEST_PATH.exists():
        print(f"信息预算冻结清单已存在，拒绝覆盖：{MANIFEST_PATH}", file=sys.stderr)
        return 2
    try:
        config = load_config(CONFIG_PATH)
    except Exception as exc:
        print(f"信息预算配置合同失败：{exc}", file=sys.stderr)
        return 2
    if (
        config["protocol"]["state"]
        != "READY_TO_FREEZE_BEFORE_INFORMATION_BUDGET_CALCULATION"
    ):
        print("信息预算协议尚未处于可冻结状态", file=sys.stderr)
        return 2
    if config["protocol"]["source_oracle_outcomes_already_known"] is not True:
        print("必须如实声明源Oracle结果已经可见", file=sys.stderr)
        return 2
    if (
        config["protocol"]["information_budget_outputs_computed_before_freeze"]
        is not False
    ):
        print("信息预算新输出在冻结前已经被计算", file=sys.stderr)
        return 2
    tracked_relatives = [
        "config/510300_oracle_information_budget_v1.yaml",
        "docs/510300_ORACLE_INFORMATION_BUDGET_V1_SPEC.md",
        "research/oracle_information_budget_v1.py",
        "research/known_state_policy_oracle_v1.py",
        "research/three_state_trend_router_v1_0_1.py",
        "scripts/freeze_510300_oracle_information_budget_v1.py",
        "scripts/run_510300_oracle_information_budget_v1.py",
        "tests/test_510300_oracle_information_budget_v1.py",
    ]
    input_relatives = [
        config["source_oracle"][key]["path"]
        for key in ("manifest", "result", "blocks", "stress_ledger", "stress_trades")
    ] + [
        config["inputs"][key]["path"]
        for key in (
            "etf_daily",
            "index_daily",
            "benchmark_total_return",
            "dividends",
            "backtest_engine",
        )
    ]
    missing = [
        relative
        for relative in tracked_relatives + input_relatives
        if not (ROOT / relative).exists()
    ]
    if missing:
        print(f"信息预算冻结文件缺失：{missing}", file=sys.stderr)
        return 2
    output_paths = [ROOT / value for key, value in config["paths"].items() if key != "spec"]
    existing_outputs = [path.relative_to(ROOT).as_posix() for path in output_paths if path.exists()]
    if existing_outputs:
        print(f"信息预算冻结前已存在候选输出：{existing_outputs}", file=sys.stderr)
        return 2
    prior_manifest_paths = sorted((ROOT / "config").glob("*manifest*.json"))
    expected_prior = int(config["selection_bias"]["prior_manifest_count"])
    if len(prior_manifest_paths) != expected_prior:
        print(
            f"信息预算冻结前清单计数异常：期望{expected_prior}，实际{len(prior_manifest_paths)}",
            file=sys.stderr,
        )
        return 2
    total_trials = len(prior_manifest_paths) + 1
    if total_trials != int(
        config["selection_bias"]["expected_total_trial_count_including_current"]
    ):
        print("信息预算累计试验计数异常", file=sys.stderr)
        return 2
    manifest = {
        "schema_version": "1.0.0",
        "project_id": PROJECT_ID,
        "state": "FROZEN_BEFORE_INFORMATION_BUDGET_CALCULATION",
        "frozen_at_asia_shanghai": datetime.now(
            ZoneInfo(config["protocol"]["timezone"])
        ).isoformat(),
        "config_path": CONFIG_PATH.relative_to(ROOT).as_posix(),
        "config_sha256": sha256_file(CONFIG_PATH),
        "source_oracle_outcomes_already_known": True,
        "information_budget_outputs_computed_before_freeze": False,
        "tracked_files": {
            relative: sha256_file(ROOT / relative) for relative in tracked_relatives
        },
        "input_files": {
            relative: sha256_file(ROOT / relative) for relative in input_relatives
        },
        "fixed_information_budget": {
            "target_net_sharpe": 1.2,
            "source_threshold": 0.05,
            "source_horizon_days": 20,
            "random_seed": config["random_error_budget"]["seed"],
            "random_repetitions_per_cell": config["random_error_budget"][
                "repetitions_per_cell"
            ],
            "joint_grid": config["random_error_budget"]["joint_grid"],
            "entry_delay_days": config["timing_and_persistence_budget"][
                "entry_delay_days"
            ],
            "phase_offsets": config["robustness_annex"]["block_phase_offsets"],
            "event_bootstrap_repetitions": config["robustness_annex"][
                "concentration"
            ]["event_bootstrap_repetitions"],
        },
        "selection_bias_control": {
            "prior_manifest_count": len(prior_manifest_paths),
            "total_trial_count_including_current": total_trials,
            "source_oracle_known_and_declared": True,
            "diagnostic_not_promoted_to_strategy": True,
        },
        "no_rescue": {
            "error_grid": True,
            "random_seed": True,
            "adversarial_rule": True,
            "timing_grid": True,
            "robustness_rule": True,
        },
        "governance": config["governance"],
    }
    temporary = MANIFEST_PATH.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(MANIFEST_PATH)
    print(
        json.dumps(
            {
                "status": manifest["state"],
                "project_id": PROJECT_ID,
                "manifest_path": MANIFEST_PATH.relative_to(ROOT).as_posix(),
                "manifest_sha256": sha256_file(MANIFEST_PATH),
                "source_oracle_outcomes_already_known": True,
                "information_budget_outputs_computed_before_freeze": False,
                "prior_manifest_count": len(prior_manifest_paths),
                "total_trial_count_including_current": total_trials,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
