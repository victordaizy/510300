"""在首次读取未来风险结果前冻结510300下行风险预算V1。"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path, PurePosixPath
import sys
from typing import Any
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "research") not in sys.path:
    sys.path.insert(0, str(ROOT / "research"))

from downside_risk_budget_v1 import (  # noqa: E402
    CONFIG_PATH,
    MANIFEST_PATH,
    load_config,
    sha256_file,
)


TRACKED_FILES = [
    "config/510300_downside_risk_budget_v1.yaml",
    "data/reference/510300_downside_risk_price_corrections_v1.csv",
    "research/downside_risk_budget_v1.py",
    "scripts/build_510300_downside_risk_inputs_v1.py",
    "scripts/freeze_510300_downside_risk_budget_v1.py",
    "scripts/run_510300_downside_risk_budget_v1.py",
    "tests/test_510300_downside_risk_budget_v1.py",
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
    result_keys = [
        "result_json",
        "result_markdown",
        "feature_table",
        "portfolio_targets",
        "base_ledger",
        "base_trades",
        "double_cost_ledger",
        "double_cost_trades",
        "buy_hold_ledger",
        "buy_hold_trades",
    ]
    preexisting = [
        config["paths"][key]
        for key in result_keys
        if _project_path(config["paths"][key]).exists()
    ]
    if preexisting:
        raise RuntimeError(f"冻结前已经存在候选结果：{preexisting}")

    tracked: dict[str, str] = {}
    for relative in TRACKED_FILES:
        path = _project_path(relative)
        if not path.exists():
            raise FileNotFoundError(f"缺少待冻结文件：{relative}")
        tracked[relative] = sha256_file(path)

    inputs: dict[str, str] = {}
    for specification in config["inputs"].values():
        relative = specification["path"]
        path = _project_path(relative)
        if not path.exists():
            raise FileNotFoundError(f"缺少待绑定输入：{relative}")
        inputs[relative] = sha256_file(path)

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
        "candidate_future_risk_outcomes_read_before_freeze": False,
        "portfolio_returns_read_before_freeze": False,
        "evidence_class": config["protocol"]["evidence_class"],
        "config_sha256": sha256_file(CONFIG_PATH),
        "tracked_files": tracked,
        "input_files": inputs,
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
    receipt_path = _project_path(config["paths"]["freeze_receipt"])
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
        "candidate_future_risk_outcomes_read_before_freeze": False,
        "portfolio_returns_read_before_freeze": False,
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
