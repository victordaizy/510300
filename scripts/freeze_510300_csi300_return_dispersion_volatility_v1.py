"""在首次读取未来波动与组合结果前冻结510300横截面收益离散度V1。"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
import sys
from typing import Any
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "research") not in sys.path:
    sys.path.insert(0, str(ROOT / "research"))

from csi300_return_dispersion_volatility_v1 import (  # noqa: E402
    CONFIG_PATH,
    MANIFEST_PATH,
    load_config,
    project_path,
    sha256_file,
)


TRACKED_FILES = [
    "config/510300_csi300_return_dispersion_volatility_v1.yaml",
    "backtest/engine.py",
    "research/csi300_return_dispersion_volatility_v1.py",
    "scripts/build_510300_csi300_return_dispersion_inputs_v1.py",
    "scripts/download_000300_constituent_daily_sina.py",
    "scripts/freeze_510300_csi300_return_dispersion_volatility_v1.py",
    "scripts/run_510300_csi300_return_dispersion_volatility_v1.py",
    "tests/test_510300_csi300_return_dispersion_volatility_v1.py",
    "reports/frozen/510300_csi300_return_dispersion_volatility_v1_pre_result_contract_failure.json",
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
        "forecast_table",
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

    protocol = config["protocol"]
    superseded_manifest = project_path(protocol["supersedes_manifest"])
    failure_receipt_path = project_path(protocol["superseded_failure_receipt"])
    if not superseded_manifest.exists() or not failure_receipt_path.exists():
        raise FileNotFoundError("预结果修正版缺少原冻结清单或失败收据")
    failure_receipt = json.loads(failure_receipt_path.read_text(encoding="utf-8"))
    if failure_receipt.get("status") != "NO_VIEW_DATA_OR_PROTOCOL_CONTRACT_FAILED":
        raise RuntimeError("被替代尝试的失败状态不正确")
    if failure_receipt.get("candidate_future_volatility_outcomes_computed") is not False:
        raise RuntimeError("被替代尝试已经计算未来波动结果，禁止按预结果修正处理")
    if failure_receipt.get("candidate_portfolio_returns_computed") is not False:
        raise RuntimeError("被替代尝试已经计算组合收益，禁止按预结果修正处理")

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
        raise RuntimeError("离散度输入审计不是PASS")
    if input_audit.get("candidate_future_volatility_outcomes_computed") is not False:
        raise RuntimeError("冻结前不得计算候选未来波动结果")
    if input_audit.get("candidate_portfolio_returns_computed") is not False:
        raise RuntimeError("冻结前不得计算候选组合收益")

    prior_manifests = sorted(
        path
        for path in (ROOT / "config").glob("510300_*_manifest.json")
        if path.resolve() != MANIFEST_PATH.resolve()
    )
    prior_manifest_hashes = {
        path.relative_to(ROOT).as_posix(): sha256_file(path)
        for path in prior_manifests
    }
    frozen_at = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    manifest = {
        "project_id": config["protocol"]["project_id"],
        "candidate_model_id": config["protocol"]["candidate_model_id"],
        "state": "FROZEN_BEFORE_FIRST_RESULT",
        "implementation_frozen": True,
        "frozen_at_asia_shanghai": frozen_at,
        "result_preexisted_at_freeze": False,
        "candidate_factor_values_read_before_freeze": protocol[
            "candidate_factor_values_read_before_freeze"
        ],
        "candidate_future_volatility_outcomes_read_before_freeze": protocol[
            "candidate_future_volatility_outcomes_read_before_freeze"
        ],
        "portfolio_returns_read_before_freeze": protocol[
            "candidate_portfolio_returns_read_before_freeze"
        ],
        "pre_result_contract_correction": {
            "supersedes_project_id": protocol["supersedes_project_id"],
            "superseded_manifest_path": superseded_manifest.relative_to(ROOT).as_posix(),
            "superseded_manifest_sha256": sha256_file(superseded_manifest),
            "failure_receipt_path": failure_receipt_path.relative_to(ROOT).as_posix(),
            "failure_receipt_sha256": sha256_file(failure_receipt_path),
            "future_outcomes_computed_in_superseded_attempt": False,
            "portfolio_returns_computed_in_superseded_attempt": False,
            "correction_scope": failure_receipt["allowed_correction_scope"],
        },
        "historical_510300_prices_previously_observed_in_other_research": True,
        "pristine_blind_holdout_claimed": False,
        "evidence_class": config["protocol"]["evidence_class"],
        "config_sha256": sha256_file(CONFIG_PATH),
        "tracked_files": tracked,
        "input_files": inputs,
        "pre_freeze_data_contract_corrections": input_audit.get(
            "pre_freeze_contract_corrections", []
        ),
        "selection_bias_control": {
            "rule": config["selection_bias"]["trial_count_rule"],
            "prior_510300_manifest_count": len(prior_manifests),
            "total_trial_count_including_current": len(prior_manifests) + 1,
            "prior_manifest_hashes": prior_manifest_hashes,
            "count_is_conservative_upper_bound_not_independence_claim": True,
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
        "status": "FROZEN_BEFORE_FIRST_RESULT",
        "frozen_at_asia_shanghai": frozen_at,
        "manifest_path": MANIFEST_PATH.relative_to(ROOT).as_posix(),
        "manifest_sha256": sha256_file(MANIFEST_PATH),
        "config_sha256": manifest["config_sha256"],
        "tracked_file_count": len(tracked),
        "input_file_count": len(inputs),
        "prior_510300_manifest_count": len(prior_manifests),
        "total_trial_count_including_current": len(prior_manifests) + 1,
        "result_preexisted_at_freeze": False,
        "candidate_factor_values_read_before_freeze": manifest[
            "candidate_factor_values_read_before_freeze"
        ],
        "candidate_future_volatility_outcomes_read_before_freeze": manifest[
            "candidate_future_volatility_outcomes_read_before_freeze"
        ],
        "portfolio_returns_read_before_freeze": manifest[
            "portfolio_returns_read_before_freeze"
        ],
        "pre_result_contract_correction": manifest["pre_result_contract_correction"],
        "historical_510300_prices_previously_observed_in_other_research": True,
        "pristine_blind_holdout_claimed": False,
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
