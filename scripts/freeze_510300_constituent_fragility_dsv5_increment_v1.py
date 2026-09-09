"""在读取任何新 DSV5 标签前冻结协议、实现、谱系和输入身份。"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import pyarrow.parquet as pq
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.constituent_fragility_dsv5_increment_v1 import (  # noqa: E402
    DSV5ProtocolError,
    MODEL_ID,
    assert_file_identity,
    canonical_json_bytes,
    git_branch,
    git_head,
    load_protocol,
    payload_sha256,
    resolve_path,
    sha256_file,
    write_json_once,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="冻结DSV5预测增量V1，不读取任何DSV5数值。")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("config/510300_constituent_fragility_dsv5_increment_v1.yaml"),
    )
    return parser.parse_args()


def _load_structured(path: Path) -> Mapping[str, Any]:
    if path.suffix.lower() in {".yaml", ".yml"}:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise DSV5ProtocolError(f"谱系文件顶层不是映射：{path}")
    return payload


def _lineage_fact(model_id: str, config: Mapping[str, Any], result: Mapping[str, Any]) -> dict[str, Any]:
    if model_id == "510300_DOWNSIDE_RISK_BUDGET_V1":
        return {
            "primary_horizon_trading_days": config["targets"]["primary_horizon_trading_days"],
            "primary_target": config["targets"]["primary_target"],
            "secondary_target": config["targets"]["secondary_target"],
            "result_status": result["status"],
        }
    if model_id == "HAR_VOLATILITY_CHALLENGER":
        return {
            "target": config["model"]["target"],
            "minimum_training_samples": config["model"]["minimum_training_samples"],
            "result_status": result["status"],
        }
    if model_id == "510300_OPTION_IMPLIED_DOWNSIDE_RISK_BUDGET_V1":
        return {
            "label_name": config["labels"]["primary"]["name"],
            "label_definition": config["labels"]["primary"]["definition"],
            "horizon_trading_days": config["labels"]["primary"]["horizon_trading_days"],
            "incremental_features": config["features"]["b2_option_incremental_features"],
            "result_status": result.get("status", result.get("G0_G1", "UNKNOWN")),
        }
    if model_id == "510300_STRESS_TRANSMISSION_HAZARD_V2":
        return {
            "target": config["model_contract"]["target"],
            "parent_B2": config["model_contract"]["fixed_models"]["B2"],
            "terminal_G1B": result["G1B_PREQUENTIAL_ERA_IDENTIFIABILITY"],
            "terminal_G2": result["G2_STRUCTURAL_INCREMENT"],
            "model_trained": result["MODEL_TRAINED"],
        }
    raise DSV5ProtocolError(f"未登记的谱系模型：{model_id}")


def build_lineage_audit(project_root: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    comparisons: list[dict[str, Any]] = []
    for specification in protocol["lineage"]["comparisons"]:
        config_path = resolve_path(project_root, str(specification["config_path"]))
        result_path = resolve_path(project_root, str(specification["result_path"]))
        if not config_path.is_file() or not result_path.is_file():
            raise DSV5ProtocolError(f"谱系证据缺失：{config_path} 或 {result_path}")
        config = _load_structured(config_path)
        result = _load_structured(result_path)
        model_id = str(specification["id"])
        comparisons.append(
            {
                "id": model_id,
                "declared_target": str(specification["target"]),
                "decision": str(specification["decision"]),
                "config": {
                    "path": str(specification["config_path"]),
                    "bytes": config_path.stat().st_size,
                    "sha256": sha256_file(config_path),
                },
                "result": {
                    "path": str(specification["result_path"]),
                    "bytes": result_path.stat().st_size,
                    "sha256": sha256_file(result_path),
                },
                "observed_contract_facts": _lineage_fact(model_id, config, result),
            }
        )
    if any(item["decision"] == "EXACT_MATCH" for item in comparisons):
        raise DSV5ProtocolError("发现完全相同旧模型，协议要求停止并复用旧B1账本")
    payload: dict[str, Any] = {
        "model_id": MODEL_ID,
        "status": "PASS_LINEAGE_NO_EXACT_DSV5_B1_TO_B2_INCREMENT_TEST_FOUND",
        "similar_components_already_implemented": True,
        "exact_end_to_end_strategy_implemented": False,
        "new_orthogonal_alpha_family": False,
        "new_estimand": True,
        "comparisons": comparisons,
        "next_action": "RUN_ONE_FROZEN_PREDICTION_INCREMENT_TEST",
        "dsv5_values_read": False,
        "portfolio_return_or_sharpe_read": False,
        "position_impact": 0,
    }
    payload["audit_payload_sha256"] = payload_sha256(payload, "audit_payload_sha256")
    return payload


def _parquet_metadata(project_root: Path, specification: Mapping[str, Any]) -> dict[str, Any]:
    identity = assert_file_identity(project_root, specification)
    path = resolve_path(project_root, str(specification["path"]))
    parquet = pq.ParquetFile(path)
    columns = list(parquet.schema_arrow.names)
    required = [str(value) for value in specification.get("required_columns", [])]
    missing = [column for column in required if column not in columns]
    if missing:
        raise DSV5ProtocolError(f"冻结Parquet元数据缺列：{missing}")
    return {
        **identity,
        "format": "PARQUET",
        "row_count": parquet.metadata.num_rows,
        "column_count": parquet.metadata.num_columns,
        "schema_columns": columns,
        "row_values_read": False,
    }


def _csv_metadata(project_root: Path, specification: Mapping[str, Any]) -> dict[str, Any]:
    identity = assert_file_identity(project_root, specification)
    path = resolve_path(project_root, str(specification["path"]))
    header = path.read_text(encoding="utf-8-sig").splitlines()[0].split(",")
    required = [str(value) for value in specification.get("required_columns", [])]
    missing = [column for column in required if column not in header]
    if missing:
        raise DSV5ProtocolError(f"冻结CSV元数据缺列：{missing}")
    return {
        **identity,
        "format": "CSV",
        "schema_columns": header,
        "data_row_values_read": False,
    }


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    protocol_path = args.protocol
    if not protocol_path.is_absolute():
        protocol_path = project_root / protocol_path
    protocol = load_protocol(protocol_path)
    required_branch = str(protocol["freeze"]["required_branch"])
    branch = git_branch(project_root)
    if branch != required_branch:
        raise DSV5ProtocolError(f"冻结分支错误：预期{required_branch}，实际{branch}")

    outputs = protocol["outputs"]
    forbidden_existing = [
        "manifest",
        "freeze_receipt",
        "g0_g1_audit",
        "origin_schedule",
        "one_shot_claim",
        "label_ledger",
        "predictions",
        "coefficients",
        "result_json",
        "result_markdown",
        "execution_receipt",
        "final_status",
    ]
    existing = [
        str(outputs[key])
        for key in forbidden_existing
        if resolve_path(project_root, str(outputs[key])).exists()
    ]
    if existing:
        raise DSV5ProtocolError(f"冻结前发现本V1输出，拒绝覆盖：{existing}")

    lineage = build_lineage_audit(project_root, protocol)
    lineage_path = resolve_path(project_root, str(outputs["lineage_audit"]))
    write_json_once(lineage_path, lineage)

    implementation: list[dict[str, Any]] = []
    for logical_path in protocol["freeze"]["implementation_files"]:
        path = resolve_path(project_root, str(logical_path))
        if not path.is_file():
            raise DSV5ProtocolError(f"缺少冻结实现文件：{logical_path}")
        implementation.append(
            {
                "path": str(logical_path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    implementation.append(
        {
            "path": str(outputs["lineage_audit"]),
            "bytes": lineage_path.stat().st_size,
            "sha256": sha256_file(lineage_path),
        }
    )

    source_identities = [
        assert_file_identity(project_root, protocol["source_adjudication"]["initial_proposal"]),
        assert_file_identity(project_root, protocol["source_adjudication"]["final_review"]),
    ]
    parent_identities = [
        assert_file_identity(project_root, protocol["immutable_parent"]["g0_1_manifest"]),
        assert_file_identity(project_root, protocol["immutable_parent"]["g0_1_clean_replay"]),
        assert_file_identity(project_root, protocol["immutable_parent"]["g1b_terminal_status"]),
    ]
    input_metadata = [
        _parquet_metadata(project_root, protocol["inputs"]["parent_internal_features"]),
        _parquet_metadata(project_root, protocol["inputs"]["etf_unadjusted_daily"]),
        _csv_metadata(project_root, protocol["inputs"]["etf_cash_dividends"]),
    ]

    manifest: dict[str, Any] = {
        "model_id": MODEL_ID,
        "version": str(protocol["protocol"]["version"]),
        "status": "FROZEN_BEFORE_DSV5_LABEL_VALUE_READ",
        "frozen_at": datetime.now().astimezone().isoformat(),
        "pre_freeze_branch": branch,
        "pre_freeze_head": git_head(project_root),
        "protocol_file": str(protocol_path.relative_to(project_root)).replace("\\", "/"),
        "protocol_sha256": sha256_file(protocol_path),
        "source_adjudication": source_identities,
        "implementation_files": implementation,
        "immutable_parent_identities": parent_identities,
        "input_metadata": input_metadata,
        "lineage_status": lineage["status"],
        "primary_offset": int(protocol["origin_schedule"]["primary_offset"]),
        "registered_offsets": protocol["origin_schedule"]["registered_offsets"],
        "target": protocol["label"]["name"],
        "horizon_trading_days": int(protocol["label"]["horizon_trading_days"]),
        "parent_feature_values_reused_without_recomputation": True,
        "new_252_day_parent_feature_warmup_overlay": False,
        "dsv5_values_read": False,
        "future_total_variance_values_read": False,
        "model_trained": False,
        "portfolio_return_or_sharpe_read": False,
        "position_impact": 0,
    }
    manifest["manifest_payload_sha256"] = payload_sha256(
        manifest, "manifest_payload_sha256"
    )
    manifest_path = resolve_path(project_root, str(outputs["manifest"]))
    write_json_once(manifest_path, manifest)
    receipt: dict[str, Any] = {
        "model_id": MODEL_ID,
        "status": "PASS_PROTOCOL_IMPLEMENTATION_LINEAGE_AND_INPUT_IDENTITY_FREEZE",
        "created_at": datetime.now().astimezone().isoformat(),
        "branch": branch,
        "pre_freeze_head": manifest["pre_freeze_head"],
        "manifest": {
            "path": str(outputs["manifest"]),
            "bytes": manifest_path.stat().st_size,
            "sha256": sha256_file(manifest_path),
        },
        "dsv5_values_read": False,
        "model_trained": False,
        "portfolio_return_or_sharpe_read": False,
        "next_step": "COMMIT_FREEZE_THEN_RUN_PRELABEL_G0_G1_ONLY",
        "position_impact": 0,
    }
    receipt["receipt_payload_sha256"] = payload_sha256(receipt, "receipt_payload_sha256")
    receipt_path = resolve_path(project_root, str(outputs["freeze_receipt"]))
    write_json_once(receipt_path, receipt)
    print(
        json.dumps(
            {
                "状态": receipt["status"],
                "manifest": str(outputs["manifest"]),
                "manifest_sha256": receipt["manifest"]["sha256"],
                "DSV5数值已读取": False,
                "模型已训练": False,
                "下一步": receipt["next_step"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
