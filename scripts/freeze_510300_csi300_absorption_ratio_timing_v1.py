"""在首次读取市场结果前冻结510300沪深300PCA吸收率V1。"""

from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import sys
from typing import Any
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "research") not in sys.path:
    sys.path.insert(0, str(ROOT / "research"))

from csi300_absorption_ratio_timing_v1 import (  # noqa: E402
    CONFIG_PATH,
    MANIFEST_PATH,
    load_config,
    project_path,
    sha256_file,
)


TRACKED_FILES = [
    "config/510300_csi300_absorption_ratio_timing_v1.yaml",
    "backtest/engine.py",
    "research/csi300_absorption_ratio_timing_v1.py",
    "scripts/build_510300_csi300_absorption_ratio_inputs_v1.py",
    "scripts/download_000300_constituent_daily_sina.py",
    "scripts/freeze_510300_csi300_absorption_ratio_timing_v1.py",
    "scripts/run_510300_csi300_absorption_ratio_timing_v1.py",
    "tests/test_510300_csi300_absorption_ratio_timing_v1.py",
    "reports/frozen/510300_csi300_absorption_ratio_timing_v1_pre_factor_membership_failure.json",
]


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
    result_keys = [
        "result_json",
        "result_markdown",
        "mechanism_table",
        "portfolio_targets",
        "base_ledger",
        "base_trades",
        "stress_ledger",
        "stress_trades",
        "buy_hold_ledger",
        "buy_hold_trades",
    ]
    preexisting = [
        config["paths"][key]
        for key in result_keys
        if project_path(config["paths"][key]).exists()
    ]
    if preexisting:
        raise RuntimeError(f"冻结前已经存在候选结果：{preexisting}")

    tracked: dict[str, str] = {}
    for relative in TRACKED_FILES:
        path = project_path(relative)
        if not path.exists():
            raise FileNotFoundError(f"缺少待冻结文件：{relative}")
        tracked[relative] = sha256_file(path)

    inputs: dict[str, str] = {}
    for specification in config["inputs"].values():
        relative = specification["path"]
        path = project_path(relative)
        if not path.exists():
            raise FileNotFoundError(f"缺少待绑定输入：{relative}")
        inputs[relative] = sha256_file(path)

    input_audit_path = project_path(config["inputs"]["input_audit"]["path"])
    input_audit = json.loads(input_audit_path.read_text(encoding="utf-8"))
    if input_audit.get("status") != "PASS":
        raise RuntimeError("吸收率输入审计不是PASS")
    if input_audit.get("candidate_factor_values_read_before_freeze") is not True:
        raise RuntimeError("输入审计没有如实声明已读取因子值")
    if input_audit.get("candidate_market_return_outcomes_read_or_computed_before_freeze") is not False:
        raise RuntimeError("冻结前已经读取或计算候选市场结果")
    if input_audit.get("candidate_portfolio_returns_read_or_computed_before_freeze") is not False:
        raise RuntimeError("冻结前已经读取或计算组合收益")
    if input_audit.get("calendar_access", {}).get("columns_read") != ["date"]:
        raise RuntimeError("冻结前交易日历不只读取了日期列")

    source_inventory_path = project_path(config["inputs"]["source_inventory"]["path"])
    source_inventory = json.loads(source_inventory_path.read_text(encoding="utf-8"))
    if source_inventory.get("status") != "PASS":
        raise RuntimeError("源文件清单不是PASS")
    if source_inventory.get("tls_certificate_verification") is not True:
        raise RuntimeError("源文件清单未确认TLS证书校验")
    for archive_name in ("reused_archive", "incremental_archive"):
        if source_inventory.get(archive_name, {}).get("crc_status") != "PASS":
            raise RuntimeError(f"{archive_name}没有通过ZIP CRC")

    prior_manifests = sorted(
        path
        for path in (ROOT / "config").glob("*manifest.json")
        if path.resolve() != MANIFEST_PATH.resolve()
    )
    expected_prior = int(config["selection_bias"]["expected_prior_manifest_count"])
    if len(prior_manifests) != expected_prior:
        raise RuntimeError(
            f"冻结前manifest计数从{expected_prior}漂移为{len(prior_manifests)}；必须先重新裁决试验次数，禁止静默继续"
        )
    prior_manifest_hashes = {
        path.relative_to(ROOT).as_posix(): sha256_file(path)
        for path in prior_manifests
    }
    frozen_at = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    total_trials = len(prior_manifests) + 1
    if total_trials != int(
        config["selection_bias"]["expected_total_trial_count_including_current"]
    ):
        raise RuntimeError("含当前候选的总试验次数漂移")
    manifest = {
        "project_id": config["protocol"]["project_id"],
        "candidate_model_id": config["protocol"]["candidate_model_id"],
        "state": "FROZEN_BEFORE_FIRST_MARKET_RESULT",
        "implementation_frozen": True,
        "frozen_at_asia_shanghai": frozen_at,
        "result_preexisted_at_freeze": False,
        "candidate_factor_values_read_before_freeze": True,
        "candidate_market_return_outcomes_read_before_freeze": False,
        "candidate_portfolio_returns_read_before_freeze": False,
        "historical_510300_prices_previously_observed_in_other_research": True,
        "pristine_blind_holdout_claimed": False,
        "evidence_class": config["protocol"]["evidence_class"],
        "config_sha256": sha256_file(CONFIG_PATH),
        "tracked_files": tracked,
        "input_files": inputs,
        "factor_input_summary": input_audit["factor"],
        "calendar_access_before_freeze": input_audit["calendar_access"],
        "source_archive_summary": input_audit["source_archives"],
        "pre_factor_membership_correction": {
            "failure_receipt_path": config["protocol"]["pre_factor_failure_receipt"],
            "failure_receipt_sha256": sha256_file(
                project_path(config["protocol"]["pre_factor_failure_receipt"])
            ),
            "corrections": input_audit["membership"][
                "pre_factor_interval_corrections"
            ],
            "factor_values_computed_before_correction": False,
            "market_return_outcomes_read_before_correction": False,
            "portfolio_returns_read_before_correction": False,
        },
        "selection_bias_control": {
            "rule": config["selection_bias"]["trial_count_rule"],
            "glob_pattern": "config/*manifest.json",
            "prior_manifest_count": len(prior_manifests),
            "total_trial_count_including_current": total_trials,
            "prior_manifest_hashes": prior_manifest_hashes,
            "count_includes_non_510300_and_non_candidate_manifests": True,
            "count_is_conservative_upper_bound_not_independence_claim": True,
        },
        "no_rescue_boundaries": {
            "parameter_rescue_after_result": "FORBIDDEN",
            "alternate_window_after_result": "FORBIDDEN",
            "threshold_change_after_result": "FORBIDDEN",
            "direction_reversal_after_result": "FORBIDDEN",
            "continuous_mapping_after_result": "FORBIDDEN",
            "combination_with_rejected_candidates_after_result": "FORBIDDEN",
        },
        "execution_boundaries": {
            "execution_asset": "510300.SH",
            "allowed_holdings": ["510300.SH", "CASH_CNY"],
            "paper_or_shadow_position_mapping": "DISABLED",
            "order_generation": "DISABLED",
            "broker_connection": "DISABLED",
            "position_change": "DISABLED",
            "live_trading_authorized": False,
        },
    }
    _atomic_json(MANIFEST_PATH, manifest)
    receipt_path = project_path(config["paths"]["freeze_receipt"])
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
        "total_trial_count_including_current": total_trials,
        "result_preexisted_at_freeze": False,
        "candidate_factor_values_read_before_freeze": True,
        "candidate_market_return_outcomes_read_before_freeze": False,
        "candidate_portfolio_returns_read_before_freeze": False,
        "historical_510300_prices_previously_observed_in_other_research": True,
        "pristine_blind_holdout_claimed": False,
        "return_evaluation": "NOT_ALLOWED_BEFORE_FROZEN_RUN",
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
