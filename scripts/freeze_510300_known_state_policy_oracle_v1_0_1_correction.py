"""冻结510300状态已知Oracle V1.0.1开封前字段兼容修正。"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "research") not in sys.path:
    sys.path.insert(0, str(ROOT / "research"))

from known_state_policy_oracle_v1 import (  # noqa: E402
    CONFIG_PATH,
    MANIFEST_PATH,
    load_config,
    sha256_file,
)


PREDECESSOR_PATH = ROOT / "config" / "510300_known_state_policy_oracle_v1_manifest.json"
AUDIT_PATH = (
    ROOT
    / "reports"
    / "audit"
    / "510300_known_state_policy_oracle_v1_prefirstoutcome_contract_key_failure.json"
)


def main() -> int:
    if MANIFEST_PATH.exists():
        print(f"Oracle修正清单已存在，拒绝覆盖：{MANIFEST_PATH}", file=sys.stderr)
        return 2
    try:
        config = load_config(CONFIG_PATH)
    except Exception as exc:
        print(f"Oracle修正配置合同失败：{exc}", file=sys.stderr)
        return 2
    if (
        config["protocol"]["state"]
        != "READY_TO_FREEZE_PRE_OUTCOME_CONTRACT_COMPATIBILITY_CORRECTION"
    ):
        print("Oracle修正协议状态异常", file=sys.stderr)
        return 2
    if not PREDECESSOR_PATH.exists() or not AUDIT_PATH.exists():
        print("Oracle修正缺少前序清单或失败审计", file=sys.stderr)
        return 2
    predecessor_hash = sha256_file(PREDECESSOR_PATH)
    if predecessor_hash != config["protocol"]["predecessor_manifest_sha256"]:
        print("Oracle前序清单哈希不匹配", file=sys.stderr)
        return 2
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    visibility_fields = (
        "candidate_oracle_blocks_computed_before_failure",
        "candidate_oracle_states_read_before_failure",
        "candidate_portfolio_returns_computed_before_failure",
    )
    if any(audit.get(field) is not False for field in visibility_fields):
        print("Oracle修正前候选结果可见性证明失败", file=sys.stderr)
        return 2
    if config["data_contract"]["expected_evaluation_rows"] != config[
        "data_contract"
    ]["expected_evaluation_trading_days"]:
        print("Oracle兼容字段值不相等", file=sys.stderr)
        return 2
    tracked_relatives = [
        "config/510300_known_state_policy_oracle_v1.yaml",
        "docs/510300_KNOWN_STATE_POLICY_ORACLE_V1_SPEC.md",
        "research/known_state_policy_oracle_v1.py",
        "research/three_state_trend_router_v1_0_1.py",
        "scripts/freeze_510300_known_state_policy_oracle_v1_0_1_correction.py",
        "scripts/run_510300_known_state_policy_oracle_v1.py",
        "tests/test_510300_known_state_policy_oracle_v1.py",
        "reports/audit/510300_known_state_policy_oracle_v1_prefirstoutcome_contract_key_failure.json",
        "reports/data_quality/510300_causal_gaussian_hmm_regime_router_v1_inputs.json",
        "backtest/engine.py",
    ]
    input_relatives = [
        "data/raw/r6/510300_daily.parquet",
        "data/raw/r6/000300_daily.parquet",
        "data/raw/r6/H00300_total_return_daily.parquet",
        "data/reference/510300_dividends.csv",
        "config/510300_known_state_policy_oracle_v1_manifest.json",
    ]
    missing = [
        relative
        for relative in tracked_relatives + input_relatives
        if not (ROOT / relative).exists()
    ]
    if missing:
        print(f"Oracle修正冻结文件缺失：{missing}", file=sys.stderr)
        return 2
    prior_manifest_paths = sorted((ROOT / "config").glob("*manifest*.json"))
    expected_prior_count = int(config["selection_bias"]["prior_manifest_count"])
    if len(prior_manifest_paths) != expected_prior_count:
        print(
            f"Oracle修正冻结前清单计数异常：期望{expected_prior_count}，得到{len(prior_manifest_paths)}",
            file=sys.stderr,
        )
        return 2
    non_manifest_count = int(
        config["selection_bias"]["non_manifest_prefreeze_failed_attempt_count"]
    )
    total_trials = len(prior_manifest_paths) + non_manifest_count + 1
    if total_trials != int(
        config["selection_bias"]["expected_total_trial_count_including_current"]
    ):
        print("Oracle修正累计试验计数异常", file=sys.stderr)
        return 2
    manifest = {
        "schema_version": "1.0.1",
        "project_id": config["protocol"]["project_id"],
        "state": "FROZEN_BEFORE_ORACLE_OUTCOME_CALCULATION",
        "frozen_at_asia_shanghai": datetime.now(
            ZoneInfo(config["protocol"]["timezone"])
        ).isoformat(),
        "config_path": CONFIG_PATH.relative_to(ROOT).as_posix(),
        "config_sha256": sha256_file(CONFIG_PATH),
        "oracle_outcomes_read_before_freeze": False,
        "state_predictability_evaluated": False,
        "oracle_future_information_declared": True,
        "tracked_files": {
            relative: sha256_file(ROOT / relative) for relative in tracked_relatives
        },
        "input_files": {
            relative: sha256_file(ROOT / relative) for relative in input_relatives
        },
        "selection_bias_control": {
            "prior_manifest_count": len(prior_manifest_paths),
            "non_manifest_prefreeze_failed_attempt_count": non_manifest_count,
            "total_trial_count_including_current": total_trials,
            "diagnostic_oracle_not_promoted_to_strategy": True,
        },
        "fixed_oracle": {
            "horizon_trading_days": 20,
            "primary_threshold": 0.05,
            "robustness_thresholds": [0.03, 0.07],
            "primary_policy": {
                "BULL_TREND": 1.0,
                "BEAR_TREND": 0.0,
                "RANGE": 0.0,
            },
            "account_cny": 20000.0,
            "base_slippage_bps": 5.0,
            "stress_slippage_bps": 10.0,
        },
        "correction": {
            "class": "PRE_OUTCOME_REQUIRED_FIELD_ALIAS_ONLY",
            "predecessor_manifest": PREDECESSOR_PATH.relative_to(ROOT).as_posix(),
            "predecessor_manifest_sha256": predecessor_hash,
            "failure_audit": AUDIT_PATH.relative_to(ROOT).as_posix(),
            "oracle_or_strategy_parameters_changed": False,
            "outcomes_read_before_correction": False,
        },
        "no_rescue": {
            "threshold": True,
            "horizon": True,
            "policy_mapping": True,
            "range_strategy": True,
        },
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
                "correction": manifest["correction"]["class"],
                "manifest_path": MANIFEST_PATH.relative_to(ROOT).as_posix(),
                "manifest_sha256": sha256_file(MANIFEST_PATH),
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
