from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
import sys
from typing import Any
from zoneinfo import ZoneInfo

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from research.csi300_pit_fundamental_underreaction_official_facts_v1_5 import (  # noqa: E402
    PARSER_VERSION,
    REQUIRED_METRICS,
)
from scripts import (  # noqa: E402
    freeze_csi300_pit_fundamental_underreaction_official_facts_v1_1 as _base,
)


CONFIG_PATH = ROOT / "config/csi300_pit_fundamental_underreaction_official_facts_v1_5.yaml"
TIMEZONE = ZoneInfo("Asia/Shanghai")
FROZEN_STATUS = (
    "FROZEN_DEFERRED_EXPENSIVE_FALLBACK_AND_QUARTERLY_TABLE_CONTAMINATION_"
    "CORRECTION_AFTER_V1_4_PARTIAL_BEFORE_V1_5_BULK_AND_ANY_FUTURE_RETURN_READ"
)


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _checked_json(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"{label}不存在：{path}")
    actual = _base.sha256_file(path)
    if actual != expected_sha256:
        raise RuntimeError(f"{label}哈希不一致：{actual}|{expected_sha256}")
    return json.loads(path.read_text(encoding="utf-8"))


def _unique_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    result: list[Path] = []
    for path in paths:
        key = str(path.resolve()).casefold()
        if key not in seen:
            result.append(path)
            seen.add(key)
    return result


def frozen_paths(config: dict[str, Any]) -> list[Path]:
    superseded_manifest_path = _base.project_path(
        config["supersedes"]["protocol_manifest"]["path"]
    )
    superseded_manifest = json.loads(
        superseded_manifest_path.read_text(encoding="utf-8")
    )
    inherited = [
        superseded_manifest_path,
        *[
            _base.project_path(entry["path"])
            for entry in superseded_manifest.get("files") or []
        ],
    ]
    gate = config["representative_revision_gate"]
    correction = config["extraction"]["quarterly_contamination_corrections"]
    current = [
        CONFIG_PATH,
        _base.project_path(config["artifacts"]["protocol_document"]),
        ROOT / "research/csi300_pit_fundamental_underreaction_official_facts_v1_5.py",
        ROOT / "scripts/collect_csi300_pit_fundamental_underreaction_official_facts_v1_5.py",
        ROOT / "scripts/freeze_csi300_pit_fundamental_underreaction_official_facts_v1_5.py",
        ROOT / "scripts/replay_csi300_pit_official_facts_v1_5.py",
        ROOT / "tests/test_csi300_pit_fundamental_underreaction_official_facts_v1_5.py",
        _base.project_path(correction["path"]),
        _base.project_path(gate["v1_5_replay_receipt"]["path"]),
    ]
    return _unique_paths([*inherited, *current])


def validate_existing(path: Path) -> None:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    _base.verify_manifest(
        path,
        expected_status=FROZEN_STATUS,
        expected_content_sha256=str(manifest.get("content_sha256") or ""),
    )
    print(f"V1.5官方财务事实冻结清单已存在且匹配：{_base.relative_path(path)}")


def validate_superseded_v1_4(config: dict[str, Any]) -> dict[str, Any]:
    supersedes = config["supersedes"]
    protocol = supersedes["protocol_manifest"]
    manifest_path = _base.project_path(protocol["path"])
    manifest = _base.verify_manifest(
        manifest_path,
        expected_status=protocol["expected_status"],
        expected_content_sha256=protocol["expected_content_sha256"],
        expected_file_sha256=protocol["expected_file_sha256"],
    )

    receipt_rule = supersedes["pilot_receipt"]
    receipt_path = _base.project_path(receipt_rule["path"])
    receipt = _checked_json(
        receipt_path,
        str(receipt_rule["expected_sha256"]),
        "V1.4部分采集收据",
    )
    counts = receipt.get("counts") or {}
    checks = {
        "status": receipt.get("status") == receipt_rule["expected_status"],
        "parser_version": receipt.get("parser_version")
        == receipt_rule["expected_parser_version"],
        "terminal_document_count": int(counts.get("terminal_document_count", -1))
        == int(receipt_rule["expected_terminal_document_count"]),
        "complete_document_count": int(counts.get("complete_document_count", -1))
        == int(receipt_rule["expected_complete_document_count"]),
        "incomplete_terminal_document_count": int(
            counts.get("incomplete_terminal_document_count", -1)
        )
        == int(receipt_rule["expected_incomplete_terminal_document_count"]),
        "market_price_read": receipt.get("market_price_read") is False,
        "future_return_read": receipt.get("future_return_read") is False,
        "return_evaluation": receipt.get("return_evaluation")
        == receipt_rule["required_return_evaluation"],
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"V1.4部分采集收据不满足继承门：{failed}")

    checkpoint_root = _base.project_path(supersedes["legacy_checkpoint_root"])
    checkpoint_count = sum(1 for _ in checkpoint_root.rglob("*.json.gz"))
    if checkpoint_count != int(supersedes["expected_legacy_checkpoint_count"]):
        raise RuntimeError(
            f"V1.4检查点数量不一致：{checkpoint_count}|"
            f"{supersedes['expected_legacy_checkpoint_count']}"
        )
    return {
        "manifest_path": _base.relative_path(manifest_path),
        "manifest_file_sha256": _base.sha256_file(manifest_path),
        "manifest_content_sha256": manifest["content_sha256"],
        "manifest_status": manifest["status"],
        "partial_receipt_path": _base.relative_path(receipt_path),
        "partial_receipt_sha256": _base.sha256_file(receipt_path),
        "partial_receipt_status": receipt["status"],
        "partial_parser_version": receipt["parser_version"],
        "legacy_checkpoint_root": _base.relative_path(checkpoint_root),
        "legacy_checkpoint_count": checkpoint_count,
        "legacy_checkpoint_mutation_forbidden": True,
        "market_price_read": False,
        "future_return_read": False,
        "return_evaluation": "NOT_ALLOWED",
    }


def validate_representative_revision_gate(config: dict[str, Any]) -> dict[str, Any]:
    gate = config["representative_revision_gate"]
    expected_ids = [str(value) for value in gate["target_announcement_ids"]]
    if len(expected_ids) != len(set(expected_ids)):
        raise RuntimeError("V1.5固定回放公告ID存在重复")

    source_rule = gate["source_manifest"]
    source_path = _base.project_path(source_rule["path"])
    source = _checked_json(
        source_path,
        str(source_rule["expected_sha256"]),
        "V1.5固定回放源清单",
    )
    source_results = {
        str(row["announcement_id"]): row for row in source.get("results") or []
    }
    if set(source_results) != set(expected_ids):
        raise RuntimeError("V1.5固定回放源清单公告集合不一致")

    reference_rule = gate["v1_4_reference_receipt"]
    reference_path = _base.project_path(reference_rule["path"])
    reference = _checked_json(
        reference_path,
        str(reference_rule["expected_sha256"]),
        "V1.4固定回放参考收据",
    )
    reference_results = {
        str(row["announcement_id"]): row for row in reference.get("results") or []
    }
    if set(reference_results) != set(expected_ids):
        raise RuntimeError("V1.4固定回放参考收据公告集合不一致")

    correction_rule = config["extraction"]["quarterly_contamination_corrections"]
    correction_path = _base.project_path(correction_rule["path"])
    correction = _checked_json(
        correction_path,
        str(correction_rule["expected_sha256"]),
        "V1.5季度表污染纠错契约",
    )
    correction_documents = correction.get("documents") or {}
    correction_metric_count = sum(
        len(document.get("corrected_metrics") or {})
        for document in correction_documents.values()
    )
    if len(correction_documents) != int(correction_rule["expected_document_count"]):
        raise RuntimeError("V1.5季度表污染纠错契约文档数不一致")
    if correction_metric_count != int(correction_rule["expected_metric_count"]):
        raise RuntimeError("V1.5季度表污染纠错契约指标数不一致")

    receipt_rule = gate["v1_5_replay_receipt"]
    receipt_path = _base.project_path(receipt_rule["path"])
    receipt = _checked_json(
        receipt_path,
        str(receipt_rule["expected_sha256"]),
        "V1.5固定回放收据",
    )
    receipt_checks = {
        "parser_version": receipt.get("parser_version")
        == receipt_rule["expected_parser_version"]
        == PARSER_VERSION,
        "target_count": int(receipt.get("target_count", -1))
        == int(receipt_rule["expected_target_count"]),
        "complete_count": int(receipt.get("complete_count", -1))
        == int(receipt_rule["expected_complete_count"]),
        "authoritative_match_count": int(
            receipt.get("authoritative_metric_value_match_count", -1)
        )
        == int(receipt_rule["expected_authoritative_match_count"]),
        "exact_v1_4_match_count": int(
            receipt.get("exact_v1_4_metric_value_match_count", -1)
        )
        == int(receipt_rule["expected_exact_v1_4_match_count"]),
        "correction_document_count": int(
            receipt.get("quarterly_contamination_corrected_document_count", -1)
        )
        == int(receipt_rule["expected_correction_document_count"]),
        "correction_metric_count": int(
            receipt.get("quarterly_contamination_corrected_metric_count", -1)
        )
        == int(receipt_rule["expected_correction_metric_count"]),
        "full_v1_4_path_count": int(
            receipt.get("full_v1_4_path_executed_count", -1)
        )
        == int(receipt_rule["expected_full_v1_4_path_count"]),
        "gate": receipt.get(
            "all_nine_metric_replay_and_authoritative_value_match_passed"
        )
        is True,
        "market_price_read": receipt.get("market_price_read") is False,
        "future_return_read": receipt.get("future_return_read") is False,
        "return_evaluation": receipt.get("return_evaluation")
        == receipt_rule["required_return_evaluation"],
    }
    failed = [name for name, passed in receipt_checks.items() if not passed]
    if failed:
        raise RuntimeError(f"V1.5固定回放收据不满足冻结门：{failed}")

    replay_results = {
        str(row["announcement_id"]): row for row in receipt.get("results") or []
    }
    if set(replay_results) != set(expected_ids):
        raise RuntimeError("V1.5固定回放收据公告集合不一致")

    correction_ids = {
        str(document["announcement_id"])
        for document in correction_documents.values()
    }
    if correction_ids != {str(value) for value in gate["correction_announcement_ids"]}:
        raise RuntimeError("V1.5季度表污染纠错公告集合不一致")

    pdf_entries: list[dict[str, Any]] = []
    for announcement_id in expected_ids:
        source_row = source_results[announcement_id]
        replay_row = replay_results[announcement_id]
        pdf_path = _base.project_path(source_row["path"])
        if not pdf_path.exists():
            raise FileNotFoundError(f"V1.5固定回放PDF不存在：{pdf_path}")
        content = pdf_path.read_bytes()
        observed_sha256 = _sha256_bytes(content)
        differences = replay_row.get("metric_differences_from_v1_4") or {}
        checks = {
            "source_status": source_row.get("status") == "DOWNLOADED_SHA256_VERIFIED",
            "same_path": str(replay_row.get("path")) == str(source_row.get("path")),
            "sha256": observed_sha256 == str(replay_row["expected_sha256"]),
            "size": len(content) == int(source_row["size_bytes"]),
            "document_complete": replay_row.get("document_complete") is True,
            "authoritative_match": replay_row.get("authoritative_metric_values_match")
            is True,
            "metric_count": int(replay_row.get("metric_count", -1))
            == int(gate["every_document_metric_count"]),
            "missing_metrics": not (replay_row.get("missing_metrics") or []),
            "only_registered_differences": all(
                value.get("registered_quarterly_contamination_correction") is True
                for value in differences.values()
            ),
            "difference_membership": bool(differences)
            == (announcement_id in correction_ids),
        }
        failed_document = [name for name, passed in checks.items() if not passed]
        if failed_document:
            raise RuntimeError(
                f"V1.5固定回放PDF校验失败：{announcement_id}|{failed_document}"
            )
        if announcement_id in correction_ids:
            guard = replay_row.get("quarterly_contamination_guard_receipt") or {}
            if guard.get("status") != (
                "PASS_HASH_BOUND_V1_4_QUARTERLY_TABLE_CONTAMINATION_CORRECTED"
            ):
                raise RuntimeError(f"V1.5纠错守卫未通过：{announcement_id}")
        pdf_entries.append(
            {
                "announcement_id": announcement_id,
                "path": _base.relative_path(pdf_path),
                "bytes": len(content),
                "sha256": observed_sha256,
                "metric_count": int(replay_row["metric_count"]),
                "difference_metric_count": len(differences),
            }
        )

    return {
        "source_manifest_path": _base.relative_path(source_path),
        "source_manifest_sha256": _base.sha256_file(source_path),
        "v1_4_reference_receipt_path": _base.relative_path(reference_path),
        "v1_4_reference_receipt_sha256": _base.sha256_file(reference_path),
        "v1_5_replay_receipt_path": _base.relative_path(receipt_path),
        "v1_5_replay_receipt_sha256": _base.sha256_file(receipt_path),
        "correction_contract_path": _base.relative_path(correction_path),
        "correction_contract_sha256": _base.sha256_file(correction_path),
        "target_document_count": len(expected_ids),
        "complete_target_count": int(receipt["complete_count"]),
        "authoritative_match_target_count": int(
            receipt["authoritative_metric_value_match_count"]
        ),
        "exact_v1_4_match_target_count": int(
            receipt["exact_v1_4_metric_value_match_count"]
        ),
        "correction_document_count": len(correction_documents),
        "correction_metric_count": correction_metric_count,
        "full_v1_4_path_executed_count": int(
            receipt["full_v1_4_path_executed_count"]
        ),
        "pdf_entries": pdf_entries,
        "market_price_read": False,
        "future_return_read": False,
        "return_evaluation": "NOT_ALLOWED",
    }


def main() -> int:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if config["extraction"]["parser_version"] != PARSER_VERSION:
        raise RuntimeError("V1.5配置解析器版本与代码不一致")
    if tuple(config["extraction"]["required_metrics"]) != tuple(REQUIRED_METRICS):
        raise RuntimeError("V1.5配置九项指标顺序与代码不一致")

    manifest_path = _base.project_path(config["artifacts"]["protocol_manifest"])
    if manifest_path.exists():
        validate_existing(manifest_path)
        return 0

    checkpoint_root = _base.project_path(config["artifacts"]["checkpoint_root"])
    if checkpoint_root.exists() and any(checkpoint_root.rglob("*.json.gz")):
        raise RuntimeError("已发现V1.5官方PDF采集检查点，拒绝事后冻结")

    parent = _base.validate_parent_manifest(config)
    superseded = validate_superseded_v1_4(config)
    inputs = _base.validate_inputs(config)
    replay_gate = validate_representative_revision_gate(config)
    paths = frozen_paths(config)
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"缺少待冻结文件：{missing}")
    files = [_base.file_entry(path) for path in paths]
    parent_bundle = {
        **parent,
        "superseded_official_fact_v1_4": superseded,
        "representative_revision_gate": replay_gate,
    }
    manifest: dict[str, Any] = {
        "manifest_id": "CSI300_PIT_FUNDAMENTAL_UNDERREACTION_OFFICIAL_FACTS_V1_5_MANIFEST",
        "status": FROZEN_STATUS,
        "created_at": datetime.now(TIMEZONE).isoformat(),
        "protocol_id": config["protocol"]["protocol_id"],
        "version": config["protocol"]["version"],
        "parent_freeze": parent_bundle,
        "inputs": inputs,
        "files": files,
        "new_checkpoint_root": _base.relative_path(checkpoint_root),
        "bulk_network_run_started": False,
        "market_price_read": False,
        "future_return_read": False,
        "future_label_created": False,
        "portfolio_return_calculated": False,
        "trading_authorization": False,
    }
    manifest["content_sha256"] = _base.canonical_sha256(
        {
            "status": manifest["status"],
            "protocol_id": manifest["protocol_id"],
            "version": manifest["version"],
            "parent_freeze": manifest["parent_freeze"],
            "inputs": manifest["inputs"],
            "files": manifest["files"],
        }
    )
    _base.atomic_write_json(manifest_path, manifest)
    for path in [*paths, manifest_path]:
        _base.read_only(path)
    print(f"V1.5官方财务事实协议已冻结：{_base.relative_path(manifest_path)}")
    print(f"V1.5官方财务事实内容SHA-256：{manifest['content_sha256']}")
    print("V1.4检查点未修改；冻结过程未读取市场价格或未来收益。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
