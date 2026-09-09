"""冻结510300因果高斯HMM三阶段路由V1。"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "research") not in sys.path:
    sys.path.insert(0, str(ROOT / "research"))

from causal_gaussian_hmm_regime_router_v1 import (  # noqa: E402
    CONFIG_PATH,
    MANIFEST_PATH,
    MODEL_PATH,
    load_config,
    sha256_file,
)


def main() -> int:
    if MANIFEST_PATH.exists():
        print(f"冻结清单已存在，拒绝覆盖：{MANIFEST_PATH}", file=sys.stderr)
        return 2
    try:
        config = load_config(CONFIG_PATH)
    except Exception as exc:
        print(f"配置合同失败：{exc}", file=sys.stderr)
        return 2
    if config["protocol"]["state"] != "READY_TO_PREFIT_ON_PRE_2015_DATA_THEN_FREEZE":
        print("协议尚未处于可冻结状态", file=sys.stderr)
        return 2
    visibility_fields = (
        "candidate_2015_plus_states_read_before_freeze",
        "candidate_2015_plus_outcomes_read_before_freeze",
        "candidate_portfolio_returns_read_before_freeze",
    )
    if any(config["protocol"][field] is not False for field in visibility_fields):
        print("冻结前候选可见性字段异常", file=sys.stderr)
        return 2
    tracked_relatives = [
        "config/510300_causal_gaussian_hmm_regime_router_v1.yaml",
        "config/510300_causal_gaussian_hmm_regime_router_v1_model.json",
        "docs/510300_CAUSAL_GAUSSIAN_HMM_REGIME_ROUTER_V1_SPEC.md",
        "research/causal_gaussian_hmm_regime_router_v1.py",
        "research/three_state_trend_router_v1_0_1.py",
        "scripts/prefit_510300_causal_gaussian_hmm_regime_router_v1.py",
        "scripts/freeze_510300_causal_gaussian_hmm_regime_router_v1.py",
        "scripts/run_510300_causal_gaussian_hmm_regime_router_v1.py",
        "tests/test_510300_causal_gaussian_hmm_regime_router_v1.py",
        "reports/data_quality/510300_causal_gaussian_hmm_regime_router_v1_inputs.json",
        "backtest/engine.py",
    ]
    input_relatives = [
        "data/raw/r6/510300_daily.parquet",
        "data/raw/r6/000300_daily.parquet",
        "data/raw/r6/H00300_total_return_daily.parquet",
        "data/reference/510300_dividends.csv",
        "data/raw/market/510300_15m_from_1m_raw.parquet",
        "reports/data_quality/510300_15m_from_1m_quality.json",
        "reports/research/510300_intraday_binary_livermore_screen_v1.json",
    ]
    missing = [
        relative
        for relative in tracked_relatives + input_relatives
        if not (ROOT / relative).exists()
    ]
    if missing:
        print(f"冻结文件缺失：{missing}", file=sys.stderr)
        return 2
    model = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    if model.get("status") != "PASS_PREFIT_TRAINING_ONLY_MODEL_IDENTIFIABILITY":
        print("训练期模型门未通过", file=sys.stderr)
        return 2
    if model.get("config_sha256_at_prefit") != sha256_file(CONFIG_PATH):
        print("模型拟合后配置发生漂移", file=sys.stderr)
        return 2
    if model.get("training", {}).get("end") != "2014-12-31":
        print("模型训练截止日越过2014年", file=sys.stderr)
        return 2
    model_visibility = (
        "candidate_2015_plus_states_computed",
        "candidate_2015_plus_outcomes_read_or_computed",
        "candidate_portfolio_returns_read_or_computed",
    )
    if any(model.get("training", {}).get(field) is not False for field in model_visibility):
        print("训练模型使用了2015年后的候选信息", file=sys.stderr)
        return 2
    audit_path = ROOT / config["paths"]["input_audit"]
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    required_status = config["inputs"]["input_audit"]["required_status"]
    if audit.get("status") != required_status:
        print("预冻结输入审计未通过", file=sys.stderr)
        return 2
    audit_visibility = (
        "candidate_2015_plus_states_read_or_computed",
        "candidate_2015_plus_outcomes_read_or_computed",
        "candidate_portfolio_returns_read_or_computed",
    )
    if any(audit.get(field) is not False for field in audit_visibility):
        print("输入审计显示2015年后的候选信息已读取", file=sys.stderr)
        return 2
    if audit.get("range_t_data_gate", {}).get("range_t_return_evaluation") != "NOT_ALLOWED":
        print("震荡做T数据门没有保持NOT_ALLOWED", file=sys.stderr)
        return 2
    prior_manifest_paths = sorted((ROOT / "config").glob("*manifest*.json"))
    expected_prior_count = int(config["selection_bias"]["prior_manifest_count"])
    if len(prior_manifest_paths) != expected_prior_count:
        print(
            f"冻结前清单计数异常：期望{expected_prior_count}，得到{len(prior_manifest_paths)}",
            file=sys.stderr,
        )
        return 2
    prior_manifest_hashes = {
        path.relative_to(ROOT).as_posix(): sha256_file(path)
        for path in prior_manifest_paths
    }
    non_manifest_count = int(
        config["selection_bias"]["non_manifest_prefreeze_failed_attempt_count"]
    )
    total_trials = len(prior_manifest_paths) + non_manifest_count + 1
    expected_total = int(
        config["selection_bias"]["expected_total_trial_count_including_current"]
    )
    if total_trials != expected_total:
        print(
            f"累计试验计数异常：期望{expected_total}，得到{total_trials}",
            file=sys.stderr,
        )
        return 2
    manifest = {
        "schema_version": "1.0.0",
        "project_id": config["protocol"]["project_id"],
        "state": "FROZEN_BEFORE_FIRST_2015_PLUS_CANDIDATE_EVALUATION",
        "frozen_at_asia_shanghai": datetime.now(
            ZoneInfo(config["protocol"]["timezone"])
        ).isoformat(),
        "config_path": CONFIG_PATH.relative_to(ROOT).as_posix(),
        "config_sha256": sha256_file(CONFIG_PATH),
        "candidate_2015_plus_states_read_before_freeze": False,
        "candidate_2015_plus_outcomes_read_before_freeze": False,
        "candidate_portfolio_returns_read_before_freeze": False,
        "range_t_return_evaluation_before_freeze": "NOT_ALLOWED",
        "tracked_files": {
            relative: sha256_file(ROOT / relative) for relative in tracked_relatives
        },
        "input_files": {
            relative: sha256_file(ROOT / relative) for relative in input_relatives
        },
        "selection_bias_control": {
            "rule": "COUNT_ALL_EXISTING_MANIFEST_FILES_PLUS_NAMED_NON_MANIFEST_PREFREEZE_ATTEMPTS",
            "prior_manifest_count": len(prior_manifest_paths),
            "non_manifest_prefreeze_failed_attempt_count": non_manifest_count,
            "named_non_manifest_attempts": config["selection_bias"][
                "named_non_manifest_attempts"
            ],
            "total_trial_count_including_current": total_trials,
            "prior_manifest_hashes": prior_manifest_hashes,
            "count_is_conservative_upper_bound_not_independence_claim": True,
        },
        "fixed_candidate": {
            "model_family": "DIAGONAL_GAUSSIAN_HMM",
            "training_window": ["2012-06-26", "2014-12-31"],
            "feature_names": model["feature_names"],
            "model_sha256": sha256_file(MODEL_PATH),
            "state_mapping": model["state_mapping"],
            "decision_probability": "ONE_STEP_AHEAD_NEXT_STATE",
            "majority_threshold": 0.5,
            "position_mapping": {
                "BULL_TREND": 1.0,
                "BEAR_TREND": 0.0,
                "RANGE_OR_UNCONFIRMED": 0.0,
            },
            "range_t_policy": "NO_TRADE_DATA_GATE_FAILED",
        },
        "no_rescue": {
            "parameter": True,
            "threshold": True,
            "window": True,
            "feature": True,
            "state_label": True,
            "range_policy": True,
            "direction": True,
            "combination": True,
        },
    }
    temporary = MANIFEST_PATH.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(MANIFEST_PATH)
    print(
        json.dumps(
            {
                "status": manifest["state"],
                "project_id": manifest["project_id"],
                "manifest_path": MANIFEST_PATH.relative_to(ROOT).as_posix(),
                "manifest_sha256": sha256_file(MANIFEST_PATH),
                "model_sha256": sha256_file(MODEL_PATH),
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
