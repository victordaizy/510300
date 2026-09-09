"""在首次读取2015年后候选结果前冻结510300多尺度非线性趋势V1。"""

from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path, PurePosixPath
import sys
from typing import Any
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "research") not in sys.path:
    sys.path.insert(0, str(ROOT / "research"))

from multi_scale_nonlinear_trend_reversion_v1 import (  # noqa: E402
    CONFIG_PATH,
    MANIFEST_PATH,
    load_config,
    sha256_file,
    trend_weights,
)


TRACKED_FILES = [
    "config/510300_multi_scale_nonlinear_trend_reversion_v1.yaml",
    "research/multi_scale_nonlinear_trend_reversion_v1.py",
    "scripts/freeze_510300_multi_scale_nonlinear_trend_reversion_v1.py",
    "scripts/run_510300_multi_scale_nonlinear_trend_reversion_v1.py",
    "tests/test_510300_multi_scale_nonlinear_trend_reversion_v1.py",
]


def _project_path(value: str) -> Path:
    return ROOT / PurePosixPath(value)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def freeze() -> dict[str, Any]:
    config = load_config()
    if MANIFEST_PATH.exists():
        raise FileExistsError(f"冻结清单已经存在，禁止覆盖：{MANIFEST_PATH}")
    result_paths = [
        config["paths"][key]
        for key in (
            "result_json",
            "result_markdown",
            "mechanism_table",
            "target_table",
            "base_ledger",
            "base_trades",
            "stress_ledger",
            "stress_trades",
            "buy_hold_ledger",
            "buy_hold_trades",
        )
    ]
    preexisting = [relative for relative in result_paths if _project_path(relative).exists()]
    if preexisting:
        raise RuntimeError(f"冻结前已经存在候选结果：{preexisting}")

    tracked: dict[str, str] = {}
    for relative in TRACKED_FILES:
        path = _project_path(relative)
        if not path.exists():
            raise FileNotFoundError(f"缺少待冻结实现：{relative}")
        tracked[relative] = sha256_file(path)

    inputs: dict[str, str] = {}
    for specification in config["inputs"].values():
        relative = specification["path"]
        path = _project_path(relative)
        if not path.exists():
            raise FileNotFoundError(f"缺少待绑定输入：{relative}")
        inputs[relative] = sha256_file(path)

    candidate_audit_path = _project_path(
        config["inputs"]["candidate_input_audit"]["path"]
    )
    candidate_audit = json.loads(candidate_audit_path.read_text(encoding="utf-8"))
    if candidate_audit.get("status") != config["inputs"]["candidate_input_audit"][
        "required_status"
    ]:
        raise RuntimeError("候选预冻结输入审计未通过")
    if candidate_audit.get("candidate_2015_plus_outcomes_read_or_computed") is not False:
        raise RuntimeError("冻结前已经读取2015年后候选结果")
    if candidate_audit.get("candidate_fixed_score_computed") is not False:
        raise RuntimeError("冻结前已经计算候选固定分数")
    if candidate_audit.get("candidate_portfolio_returns_read_or_computed") is not False:
        raise RuntimeError("冻结前已经计算候选组合收益")
    if candidate_audit.get("candidate_sharpe_computed") is not False:
        raise RuntimeError("冻结前已经计算候选夏普率")

    captures: dict[str, dict[str, float | int]] = {}
    for horizon in config["trend"]["horizons_trading_days"]:
        weights, capture = trend_weights(
            int(horizon), float(config["trend"]["lag_multiplier"])
        )
        if capture < float(config["trend"]["minimum_squared_weight_energy_capture"]):
            raise RuntimeError(f"尺度{horizon}的权重能量覆盖不足")
        captures[str(int(horizon))] = {
            "lag_count": int(len(weights)),
            "squared_weight_energy_capture": float(capture),
        }

    prior_manifests = sorted(
        path
        for path in (ROOT / "config").glob("*manifest*.json")
        if path.resolve() != MANIFEST_PATH.resolve()
    )
    expected_prior = int(
        config["selection_bias"]["expected_prior_manifest_count_at_freeze"]
    )
    if len(prior_manifests) != expected_prior:
        raise RuntimeError(
            f"冻结前清单数量漂移：期望{expected_prior}，实际{len(prior_manifests)}；"
            "必须先显式更新试验总数，不能静默少计"
        )
    failed_attempts = int(
        config["selection_bias"]["non_manifest_prefreeze_failed_attempt_count"]
    )
    total_trials = len(prior_manifests) + failed_attempts + 1
    if total_trials != int(
        config["selection_bias"]["expected_total_trial_count_including_current"]
    ):
        raise RuntimeError("冻结试验总数与协议不一致")
    prior_hashes = {
        path.relative_to(ROOT).as_posix(): sha256_file(path)
        for path in prior_manifests
    }
    frozen_at = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    manifest = {
        "project_id": config["protocol"]["project_id"],
        "candidate_model_id": config["protocol"]["candidate_model_id"],
        "state": "FROZEN_BEFORE_FIRST_2015_PLUS_CANDIDATE_OUTCOME",
        "implementation_frozen": True,
        "frozen_at_asia_shanghai": frozen_at,
        "result_preexisted_at_freeze": False,
        "pre_2015_training_normalization_read_before_freeze": True,
        "candidate_2015_plus_outcomes_read_before_freeze": False,
        "candidate_fixed_score_computed_before_freeze": False,
        "portfolio_returns_read_before_freeze": False,
        "sharpe_computed_before_freeze": False,
        "config_sha256": sha256_file(CONFIG_PATH),
        "tracked_files": tracked,
        "input_files": inputs,
        "weight_contract": captures,
        "selection_bias_control": {
            "rule": config["selection_bias"]["trial_count_rule"],
            "prior_manifest_count": len(prior_manifests),
            "non_manifest_prefreeze_failed_attempt_count": failed_attempts,
            "named_non_manifest_attempts": config["selection_bias"][
                "named_non_manifest_attempts"
            ],
            "total_trial_count_including_current": total_trials,
            "prior_manifest_hashes": prior_hashes,
            "count_is_conservative_upper_bound_not_independence_claim": True,
        },
        "execution_boundaries": {
            "execution_asset": "510300.SH",
            "allowed_holdings": ["510300.SH", "CASH_CNY"],
            "paper_signal_allowed": False,
            "shadow_signal_allowed": False,
            "order_generation": False,
            "broker_connection": False,
            "position_change": False,
            "live_trading_authorized": False,
        },
    }
    _atomic_json(MANIFEST_PATH, manifest)
    receipt_path = _project_path(config["paths"]["freeze_receipt"])
    receipt = {
        "project_id": manifest["project_id"],
        "status": manifest["state"],
        "frozen_at_asia_shanghai": frozen_at,
        "manifest_path": MANIFEST_PATH.relative_to(ROOT).as_posix(),
        "manifest_sha256": sha256_file(MANIFEST_PATH),
        "config_sha256": manifest["config_sha256"],
        "tracked_file_count": len(tracked),
        "input_file_count": len(inputs),
        "prior_manifest_count": len(prior_manifests),
        "non_manifest_prefreeze_failed_attempt_count": failed_attempts,
        "total_trial_count_including_current": total_trials,
        "candidate_2015_plus_outcomes_read_before_freeze": False,
        "candidate_fixed_score_computed_before_freeze": False,
        "portfolio_returns_read_before_freeze": False,
        "sharpe_computed_before_freeze": False,
        "live_trading_authorized": False,
    }
    _atomic_json(receipt_path, receipt)
    return receipt


def main() -> int:
    receipt = freeze()
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
