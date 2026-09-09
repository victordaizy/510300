"""按 V1.0.1 修正重做同规模四态来源探测。"""

from __future__ import annotations

import argparse
import copy
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo

import tushare as ts

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.project_evidence_contract_v1 import (
    EvidenceContractError,
    atomic_write_json_new,
    file_evidence,
    normalize_project_relative_path,
)
from scripts.download_csi300_all_etf_momentum_v1 import credentials
from scripts.freeze_510300_stress_transmission_hazard_v2_four_state_sources_v1 import (
    DEFAULT_CONFIG as V1_CONFIG,
    load_config as load_v1_config,
)
from scripts.freeze_510300_stress_transmission_hazard_v2_four_state_sources_v1_0_1 import (
    DEFAULT_CONFIG,
    load_config,
    verify_frozen_manifest,
)
from scripts.probe_510300_stress_transmission_hazard_v2_four_state_sources_v1 import (
    _endpoint_candidates,
    _query_frames,
    _write_parquet_new,
    validate_endpoint_frames,
)


def effective_config(correction: Mapping[str, Any]) -> dict[str, Any]:
    base = copy.deepcopy(load_v1_config(V1_CONFIG))
    fixed_fields = [
        str(value)
        for value in correction["schema_correction"]["required_fields_after_correction"]
    ]
    base["provider_contract"]["dividend"]["required_fields"] = fixed_fields
    base["provider_contract"]["dividend"]["optional_fields"] = [
        str(value) for value in correction["schema_correction"]["optional_fields"]
    ]
    corrected_probe = correction["corrected_probe_contract"]
    base["probe_contract"]["snapshot_directory"] = corrected_probe["snapshot_directory"]
    base["probe_contract"]["receipt_path"] = corrected_probe["receipt_path"]
    base["probe_contract"]["status_path"] = corrected_probe["status_path"]
    base["probe_contract"]["full_history_or_bulk_member_download_allowed"] = False
    return base


def _output_paths(config: Mapping[str, Any]) -> tuple[Path, Path, Path]:
    probe = config["probe_contract"]
    return (
        ROOT / Path(normalize_project_relative_path(str(probe["snapshot_directory"]))),
        ROOT / Path(normalize_project_relative_path(str(probe["receipt_path"]))),
        ROOT / Path(normalize_project_relative_path(str(probe["status_path"]))),
    )


def _write_failed_probe(
    config: Mapping[str, Any],
    receipt_path: Path,
    status_path: Path,
    failures: list[dict[str, str]],
) -> None:
    now = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    receipt = {
        "receipt_id": "510300_STRESS_TRANSMISSION_HAZARD_V2_FOUR_STATE_SOURCE_PROBE_V1_0_1_FAILURE",
        "created_at": now,
        "status": "BLOCKED_ALL_ALLOWED_ENDPOINT_PROBES_FAILED",
        "failures": failures,
        "snapshots_written": False,
        "bulk_collection_authorized": False,
        "performance_values_read": False,
        "credential_persisted": False,
        "security_audit_performed": False,
        "position_impact": 0,
    }
    atomic_write_json_new(receipt_path, receipt)
    status = {
        "program_id": config["program"]["program_id"],
        "updated_at": now,
        "source_probe": "BLOCKED",
        "next_allowed_step": "SOURCE_REMEDIATION_ONLY",
        "model_state": "NO_VIEW_SOURCE_PROBE_BLOCKED",
        "g0_status": "NOT_PASSED",
        "return_evaluation": "NOT_ALLOWED",
        "position_impact": 0,
        "receipt": file_evidence(receipt_path, project_root=ROOT),
    }
    atomic_write_json_new(status_path, status)


def run_probe(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    verify_frozen_manifest(config_path)
    correction = load_config(config_path)
    config = effective_config(correction)
    snapshot_dir, receipt_path, status_path = _output_paths(config)
    if receipt_path.exists() or status_path.exists() or snapshot_dir.exists():
        raise EvidenceContractError("V1.0.1 探测输出已存在，禁止重复或覆盖")

    secret, preferred = credentials()
    failures: list[dict[str, str]] = []
    selected_endpoint: str | None = None
    normalized = None
    for endpoint in _endpoint_candidates(preferred):
        api = ts.pro_api(secret)
        api._DataApi__http_url = endpoint
        try:
            normalized = validate_endpoint_frames(config, _query_frames(api, config))
            selected_endpoint = endpoint
            break
        except Exception as exc:
            message = str(exc).replace(secret, "<EPHEMERAL_CREDENTIAL_REMOVED>")
            failures.append(
                {
                    "endpoint": endpoint,
                    "error_type": type(exc).__name__,
                    "message": message[:500],
                }
            )
    if selected_endpoint is None or normalized is None:
        _write_failed_probe(config, receipt_path, status_path, failures)
        raise EvidenceContractError("V1.0.1 所有允许节点的同规模探测均失败")

    snapshots: dict[str, dict[str, Any]] = {}
    for source_name, frame in normalized.items():
        target = snapshot_dir / f"{source_name}_probe.parquet"
        _write_parquet_new(frame, target)
        snapshots[source_name] = file_evidence(target, project_root=ROOT)
        snapshots[source_name]["row_count"] = int(len(frame))
        snapshots[source_name]["columns"] = frame.columns.astype(str).tolist()

    now = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    receipt: dict[str, Any] = {
        "receipt_id": "510300_STRESS_TRANSMISSION_HAZARD_V2_FOUR_STATE_SOURCE_PROBE_V1_0_1",
        "created_at": now,
        "status": "PASS_BOUNDED_ENDPOINT_SCHEMA_AND_IDENTITY_PROBE_AFTER_OPTIONAL_FIELD_CORRECTION",
        "selected_endpoint": selected_endpoint,
        "snapshots": snapshots,
        "failed_endpoint_attempts": failures,
        "scope_expanded_from_v1_probe": False,
        "semantic_boundaries": {
            "suspend_d_official_exchange_evidence_eligible": False,
            "suspend_d_zero_return_authorization": False,
            "adj_factor_direct_return_construction_allowed": False,
            "historical_four_state_coverage_claimed": False,
            "bulk_collection_authorized_by_probe": False,
            "next_execution_contract_freeze_allowed": True,
        },
        "credential_persisted": False,
        "performance_values_read": False,
        "security_audit_performed": False,
        "position_impact": 0,
    }
    atomic_write_json_new(receipt_path, receipt)
    status = {
        "program_id": config["program"]["program_id"],
        "correction_id": correction["program"]["correction_id"],
        "updated_at": now,
        "source_probe": "PASS",
        "next_allowed_step": correction["corrected_probe_contract"]["pass_unlocks_only"],
        "bulk_collection_authorized_now": False,
        "official_exchange_suspension_source": "NOT_ADMITTED",
        "historical_four_state_coverage": "NOT_RUN",
        "model_state": "NO_VIEW_PENDING_FOUR_STATE_LEDGER",
        "g0_status": "NOT_PASSED",
        "g1_through_g7_status": "NOT_RUN",
        "return_evaluation": "NOT_ALLOWED",
        "portfolio_results_read": False,
        "position_impact": 0,
        "receipt": file_evidence(receipt_path, project_root=ROOT),
    }
    atomic_write_json_new(status_path, status)
    print("V1.0.1 同规模来源探测通过；下一步仅允许冻结历史采集与四态账本执行。")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    try:
        run_probe(args.config)
        return 0
    except (EvidenceContractError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        print(f"V1.0.1 有界探测失败：{exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
