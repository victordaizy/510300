"""冻结510300期权-期货-全A四域二元候选的协议与开发证据。"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "510300_option_futures_breadth_domain_union_v1.yaml"
TIMEZONE = ZoneInfo("Asia/Shanghai")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    if not CONFIG.is_file():
        raise FileNotFoundError(CONFIG)
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    evidence = config["development_evidence"]
    evidence_paths = {
        "discovery_script": ROOT / evidence["discovery_script"],
        "discovery_report": ROOT / evidence["discovery_report"],
        "discovery_features": ROOT / evidence["discovery_features"],
        "discovery_metrics": ROOT / evidence["discovery_metrics"],
    }
    missing = [str(path) for path in evidence_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"冻结证据缺失：{missing}")

    report = json.loads(evidence_paths["discovery_report"].read_text(encoding="utf-8"))
    if report.get("status") != "DEVELOPMENT_CANDIDATE_FOUND_FREEZE_REQUIRED_NO_VALIDATION_READ":
        raise RuntimeError("开发报告状态不允许冻结")
    best = report.get("best_candidate", {})
    expected = {
        "policy": config["protocol"]["candidate_name"],
        "stress_annualized_excess_minimum": float(evidence["stress_annualized_excess_minimum"]),
        "stress_annualized_excess_median": float(evidence["stress_annualized_excess_median"]),
        "stress_rolling_242d_excess_minimum": float(evidence["stress_rolling_242d_excess_minimum"]),
        "stress_perturbation_pass_ratio": float(evidence["stress_start_pass_ratio"]),
        "development_gate": True,
    }
    for key, value in expected.items():
        actual = best.get(key)
        if isinstance(value, float):
            if actual is None or abs(float(actual) - value) > 1e-12:
                raise RuntimeError(f"开发证据字段漂移：{key}，实际={actual}，期望={value}")
        elif actual != value:
            raise RuntimeError(f"开发证据字段漂移：{key}，实际={actual}，期望={value}")

    manifest_path = ROOT / config["paths"]["manifest"]
    receipt_path = ROOT / config["paths"]["freeze_receipt"]
    frozen_at = datetime.now(TIMEZONE).isoformat()
    manifest = {
        "schema_version": "1.0.0",
        "project_id": config["protocol"]["project_id"],
        "state": "FROZEN_BEFORE_FIXED_RULE_VALIDATION",
        "frozen_at": frozen_at,
        "config_path": str(CONFIG.relative_to(ROOT)).replace("\\", "/"),
        "config_sha256": sha256(CONFIG),
        "candidate_name": config["protocol"]["candidate_name"],
        "development_evidence_sha256": {
            name: sha256(path) for name, path in evidence_paths.items()
        },
        "fixed_contract": {
            "execution_asset": config["scope"]["execution_asset"],
            "states": config["scope"]["allowed_states"],
            "benchmark": config["objective"]["benchmark"],
            "stress_slippage_bps_per_leg": config["costs"]["stress_slippage_bps_per_leg"],
            "minimum_annualized_net_excess": config["objective"][
                "minimum_annualized_net_excess"
            ],
            "minimum_rolling_242d_excess_median": config["objective"][
                "minimum_rolling_242d_excess_median"
            ],
            "all_five_starts_required": config["objective"][
                "every_start_must_pass_both_gates"
            ],
        },
        "boundaries": {
            "validation_read_before_manifest": False,
            "formula_change_after_freeze": "FORBIDDEN",
            "threshold_change_after_freeze": "FORBIDDEN",
            "orders": "DISABLED",
            "broker": "DISABLED",
            "live_trading": "NOT_AUTHORIZED",
        },
    }
    atomic_json(manifest, manifest_path)
    receipt = {
        "status": "FROZEN_SUCCESS",
        "project_id": config["protocol"]["project_id"],
        "frozen_at": frozen_at,
        "manifest_path": str(manifest_path.relative_to(ROOT)).replace("\\", "/"),
        "manifest_sha256": sha256(manifest_path),
        "config_path": str(CONFIG.relative_to(ROOT)).replace("\\", "/"),
        "config_sha256": sha256(CONFIG),
        "candidate": best,
        "development_evidence_sha256": manifest["development_evidence_sha256"],
        "next_allowed_action": "运行不改变公式、阈值、成本的2024-2025固定规则验证",
        "live_trading_authorized": False,
    }
    atomic_json(receipt, receipt_path)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
