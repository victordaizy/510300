from __future__ import annotations

from datetime import datetime
import gzip
import hashlib
import json
from pathlib import Path
import sys
from typing import Any
from zoneinfo import ZoneInfo

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from research.csi300_pit_fundamental_underreaction_official_facts_v1_7 import (  # noqa: E402
    PARSER_VERSION,
    REQUIRED_METRICS,
)
from scripts import (  # noqa: E402
    freeze_csi300_pit_fundamental_underreaction_official_facts_v1_1 as _base,
)


CONFIG_PATH = ROOT / "config/csi300_pit_fundamental_underreaction_official_facts_v1_7.yaml"
TIMEZONE = ZoneInfo("Asia/Shanghai")
FROZEN_STATUS = (
    "FROZEN_ADJACENT_NOTE_EXTENDED_PROFIT_QUARTERLY_IMAGE_IDENTITY_"
    "AFTER_V1_6_FULL_REPARSE_BEFORE_V1_7_GAP_REPARSE_AND_ANY_FUTURE_RETURN_READ"
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
    gate = config["representative_revision_gate"]
    source_manifest_path = _base.project_path(
        gate["v1_7_gap_source_manifest"]["path"]
    )
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    pdf_paths = [
        _base.project_path(row["path"])
        for row in source_manifest.get("results") or []
    ]
    return _unique_paths(
        [
            CONFIG_PATH,
            _base.project_path(config["artifacts"]["protocol_document"]),
            ROOT / "research/csi300_pit_fundamental_underreaction_official_facts_v1_7.py",
            ROOT / "scripts/collect_csi300_pit_fundamental_underreaction_official_facts_v1_7.py",
            ROOT / "scripts/freeze_csi300_pit_fundamental_underreaction_official_facts_v1_7.py",
            ROOT / "scripts/migrate_csi300_pit_official_facts_v1_6_complete_checkpoints_to_v1_7.py",
            ROOT / "scripts/prepare_csi300_pit_official_facts_v1_7_replay_sources.py",
            ROOT / "scripts/replay_csi300_pit_official_facts_v1_7.py",
            ROOT / "tests/test_csi300_pit_fundamental_underreaction_official_facts_v1_7.py",
            source_manifest_path,
            _base.project_path(gate["v1_6_reference_replay_receipt"]["path"]),
            _base.project_path(gate["v1_7_replay_receipt"]["path"]),
            *pdf_paths,
        ]
    )


def validate_existing(path: Path) -> None:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    _base.verify_manifest(
        path,
        expected_status=FROZEN_STATUS,
        expected_content_sha256=str(manifest.get("content_sha256") or ""),
    )
    print(f"V1.7官方财务事实冻结清单已存在且匹配：{_base.relative_path(path)}")


def validate_superseded_v1_6(config: dict[str, Any]) -> dict[str, Any]:
    supersedes = config["supersedes"]
    manifest_rule = supersedes["protocol_manifest"]
    manifest_path = _base.project_path(manifest_rule["path"])
    manifest = _base.verify_manifest(
        manifest_path,
        expected_status=manifest_rule["expected_status"],
        expected_content_sha256=manifest_rule["expected_content_sha256"],
        expected_file_sha256=manifest_rule["expected_file_sha256"],
    )
    receipt_rule = supersedes["full_collection_receipt"]
    receipt_path = _base.project_path(receipt_rule["path"])
    receipt = _checked_json(
        receipt_path,
        str(receipt_rule["expected_sha256"]),
        "V1.6全量复解析收据",
    )
    counts = receipt.get("counts") or {}
    checks = {
        "status": receipt.get("status") == receipt_rule["expected_status"],
        "parser_version": receipt.get("parser_version")
        == receipt_rule["expected_parser_version"],
        "queued_document_count": int(counts.get("queued_document_count", -1))
        == int(receipt_rule["expected_queued_document_count"]),
        "terminal_document_count": int(counts.get("terminal_document_count", -1))
        == int(receipt_rule["expected_terminal_document_count"]),
        "complete_document_count": int(counts.get("complete_document_count", -1))
        == int(receipt_rule["expected_complete_document_count"]),
        "incomplete_terminal_document_count": int(
            counts.get("incomplete_terminal_document_count", -1)
        )
        == int(receipt_rule["expected_incomplete_terminal_document_count"]),
        "download_failed_document_count": int(
            counts.get("download_failed_document_count", -1)
        )
        == int(receipt_rule["expected_download_failed_document_count"]),
        "target_event_count": int(counts.get("target_event_count", -1))
        == int(receipt_rule["expected_target_event_count"]),
        "ready_target_event_count": int(counts.get("ready_target_event_count", -1))
        == int(receipt_rule["expected_ready_target_event_count"]),
        "market_price_read": receipt.get("market_price_read") is False,
        "future_return_read": receipt.get("future_return_read") is False,
        "return_evaluation": receipt.get("return_evaluation")
        == receipt_rule["required_return_evaluation"],
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"V1.6全量收据不满足继承门：{failed}")

    checkpoint_root = _base.project_path(supersedes["legacy_checkpoint_root"])
    checkpoint_paths = sorted(checkpoint_root.rglob("*.json.gz"))
    complete_count = 0
    incomplete_count = 0
    for path in checkpoint_paths:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        status = str(payload.get("checkpoint_status") or "")
        if status == "PARSED_COMPLETE":
            metric_ids = [
                str(metric.get("metric_id")) for metric in payload.get("metrics") or []
            ]
            valid = (
                payload.get("document_complete") is True
                and not (payload.get("missing_metrics") or [])
                and len(metric_ids) == len(REQUIRED_METRICS)
                and set(metric_ids) == set(REQUIRED_METRICS)
                and len(metric_ids) == len(set(metric_ids))
            )
            if not valid:
                raise RuntimeError(f"V1.6完整检查点不是九项唯一：{path}")
            complete_count += 1
        elif status == "PARSED_INCOMPLETE":
            if payload.get("document_complete") is True:
                raise RuntimeError(f"V1.6不完整检查点状态矛盾：{path}")
            incomplete_count += 1
        else:
            raise RuntimeError(f"V1.6检查点存在非终态：{path}|{status}")
    expected_total = int(supersedes["expected_legacy_checkpoint_count"])
    expected_complete = int(supersedes["expected_complete_checkpoint_count"])
    expected_incomplete = int(supersedes["expected_incomplete_checkpoint_count"])
    if len(checkpoint_paths) != expected_total:
        raise RuntimeError("V1.6检查点总数不一致")
    if complete_count != expected_complete or incomplete_count != expected_incomplete:
        raise RuntimeError("V1.6完整或不完整检查点计数不一致")
    return {
        "manifest_path": _base.relative_path(manifest_path),
        "manifest_file_sha256": _base.sha256_file(manifest_path),
        "manifest_content_sha256": manifest["content_sha256"],
        "manifest_status": manifest["status"],
        "full_collection_receipt_path": _base.relative_path(receipt_path),
        "full_collection_receipt_sha256": _base.sha256_file(receipt_path),
        "full_collection_status": receipt["status"],
        "legacy_checkpoint_root": _base.relative_path(checkpoint_root),
        "legacy_checkpoint_count": len(checkpoint_paths),
        "legacy_complete_checkpoint_count": complete_count,
        "legacy_incomplete_checkpoint_count": incomplete_count,
        "legacy_checkpoint_mutation_forbidden": True,
        "market_price_read": False,
        "future_return_read": False,
        "return_evaluation": "NOT_ALLOWED",
    }


def validate_representative_revision_gate(config: dict[str, Any]) -> dict[str, Any]:
    gate = config["representative_revision_gate"]
    reference_rule = gate["v1_6_reference_replay_receipt"]
    reference_path = _base.project_path(reference_rule["path"])
    reference = _checked_json(
        reference_path,
        str(reference_rule["expected_sha256"]),
        "V1.6固定回放参考收据",
    )
    if int(reference.get("target_count", -1)) != int(reference_rule["expected_target_count"]):
        raise RuntimeError("V1.6固定回放参考目标数不一致")
    if reference.get("all_nine_metric_replay_and_authoritative_value_match_passed") is not True:
        raise RuntimeError("V1.6固定回放参考门未通过")

    source_rule = gate["v1_7_gap_source_manifest"]
    source_path = _base.project_path(source_rule["path"])
    source = _checked_json(
        source_path,
        str(source_rule["expected_sha256"]),
        "V1.7缺口回放源清单",
    )
    source_rows = source.get("results") or []
    if len(source_rows) != int(source_rule["expected_target_count"]):
        raise RuntimeError("V1.7缺口回放源数量不一致")
    expected_ids = {str(value) for value in gate["gap_target_announcement_ids"]}
    if {str(row["announcement_id"]) for row in source_rows} != expected_ids:
        raise RuntimeError("V1.7缺口回放源公告集合不一致")
    pdf_entries = []
    for row in source_rows:
        pdf_path = _base.project_path(row["path"])
        content = pdf_path.read_bytes()
        checks = {
            "pdf_header": content.startswith(b"%PDF-"),
            "size": len(content) == int(row["size_bytes"]),
            "sha256": _sha256_bytes(content) == str(row["sha256"]),
            "expected_size": len(content) == int(row["expected_size_bytes"]),
            "expected_sha256": _sha256_bytes(content) == str(row["expected_sha256"]),
            "source_host": str(row["official_pdf_url"]).startswith(
                "https://static.cninfo.com.cn/"
            ),
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise RuntimeError(
                f"V1.7缺口固定PDF校验失败：{row['announcement_id']}|{failed}"
            )
        pdf_entries.append(
            {
                "announcement_id": str(row["announcement_id"]),
                "path": _base.relative_path(pdf_path),
                "bytes": len(content),
                "sha256": _sha256_bytes(content),
                "fix_family": str(row["fix_family"]),
            }
        )

    replay_rule = gate["v1_7_replay_receipt"]
    replay_path = _base.project_path(replay_rule["path"])
    replay = _checked_json(
        replay_path,
        str(replay_rule["expected_sha256"]),
        "V1.7固定回放收据",
    )
    replay_checks = {
        "parser_version": replay.get("parser_version")
        == replay_rule["expected_parser_version"]
        == PARSER_VERSION,
        "target_count": int(replay.get("target_count", -1))
        == int(replay_rule["expected_target_count"]),
        "v1_6_nonregression": int(replay.get("v1_6_nonregression_target_count", -1))
        == int(replay_rule["expected_v1_6_nonregression_target_count"]),
        "v1_7_positive": int(replay.get("v1_7_positive_gap_target_count", -1))
        == int(replay_rule["expected_v1_7_positive_gap_target_count"]),
        "v1_7_negative": int(replay.get("v1_7_negative_guard_target_count", -1))
        == int(replay_rule["expected_v1_7_negative_guard_target_count"]),
        "complete_positive": int(
            replay.get("complete_positive_and_nonregression_count", -1)
        )
        == int(replay_rule["expected_complete_positive_and_nonregression_count"]),
        "value_match": int(replay.get("expected_metric_values_match_count", -1))
        == int(replay_rule["expected_metric_values_match_count"]),
        "baseline_unchanged": int(
            replay.get("baseline_metric_values_unchanged_count", -1)
        )
        == int(replay_rule["expected_baseline_metric_values_unchanged_count"]),
        "targeted_receipts": int(replay.get("targeted_receipt_pass_count", -1))
        == int(replay_rule["expected_targeted_receipt_pass_count"]),
        "negative_guard": replay.get("negative_guard_passed") is True,
        "gate": replay.get("frozen_replay_contract_passed") is True,
        "market_price_read": replay.get("market_price_read") is False,
        "future_return_read": replay.get("future_return_read") is False,
        "return_evaluation": replay.get("return_evaluation")
        == replay_rule["required_return_evaluation"],
    }
    failed_replay = [name for name, passed in replay_checks.items() if not passed]
    if failed_replay:
        raise RuntimeError(f"V1.7固定回放收据不满足冻结门：{failed_replay}")
    replay_rows = replay.get("results") or []
    negative_id = str(gate["negative_guard_announcement_id"])
    if len(replay_rows) != int(replay_rule["expected_target_count"]):
        raise RuntimeError("V1.7固定回放结果行数不一致")
    for row in replay_rows:
        is_negative = str(row["announcement_id"]) == negative_id
        checks = {
            "parser_version": row.get("parser_version") == PARSER_VERSION,
            "complete_state": row.get("document_complete") is (not is_negative),
            "metric_count": int(row.get("metric_count", -1))
            == (len(REQUIRED_METRICS) - 1 if is_negative else len(REQUIRED_METRICS)),
            "missing_metrics": (
                row.get("missing_metrics") == ["ACCOUNTS_RECEIVABLE_END"]
                if is_negative
                else not (row.get("missing_metrics") or [])
            ),
            "value_match": row.get("expected_metric_values_match") is True,
            "baseline_unchanged": row.get("baseline_metric_values_unchanged") is True,
            "targeted_receipt": row.get("targeted_receipt_passed") is True,
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise RuntimeError(f"V1.7固定回放文档门失败：{row['announcement_id']}|{failed}")
    return {
        "v1_6_reference_replay_receipt_path": _base.relative_path(reference_path),
        "v1_6_reference_replay_receipt_sha256": _base.sha256_file(reference_path),
        "v1_7_gap_source_manifest_path": _base.relative_path(source_path),
        "v1_7_gap_source_manifest_sha256": _base.sha256_file(source_path),
        "v1_7_replay_receipt_path": _base.relative_path(replay_path),
        "v1_7_replay_receipt_sha256": _base.sha256_file(replay_path),
        "target_document_count": len(replay_rows),
        "v1_6_nonregression_target_count": replay["v1_6_nonregression_target_count"],
        "v1_7_positive_gap_target_count": replay["v1_7_positive_gap_target_count"],
        "v1_7_negative_guard_target_count": replay["v1_7_negative_guard_target_count"],
        "pdf_entries": pdf_entries,
        "market_price_read": False,
        "future_return_read": False,
        "return_evaluation": "NOT_ALLOWED",
    }


def main() -> int:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if config["extraction"]["parser_version"] != PARSER_VERSION:
        raise RuntimeError("V1.7配置解析器版本与代码不一致")
    if tuple(config["extraction"]["required_metrics"]) != tuple(REQUIRED_METRICS):
        raise RuntimeError("V1.7配置九项指标顺序与代码不一致")
    manifest_path = _base.project_path(config["artifacts"]["protocol_manifest"])
    if manifest_path.exists():
        validate_existing(manifest_path)
        return 0
    checkpoint_root = _base.project_path(config["artifacts"]["checkpoint_root"])
    if checkpoint_root.exists() and any(checkpoint_root.rglob("*.json.gz")):
        raise RuntimeError("已发现V1.7官方PDF采集检查点，拒绝事后冻结")

    parent = _base.validate_parent_manifest(config)
    superseded = validate_superseded_v1_6(config)
    inputs = _base.validate_inputs(config)
    replay_gate = validate_representative_revision_gate(config)
    paths = frozen_paths(config)
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"缺少待冻结文件：{missing}")
    files = [_base.file_entry(path) for path in paths]
    parent_bundle = {
        **parent,
        "superseded_official_fact_v1_6": superseded,
        "representative_revision_gate": replay_gate,
    }
    manifest: dict[str, Any] = {
        "manifest_id": "CSI300_PIT_FUNDAMENTAL_UNDERREACTION_OFFICIAL_FACTS_V1_7_MANIFEST",
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
    print(f"V1.7官方财务事实协议已冻结：{_base.relative_path(manifest_path)}")
    print(f"V1.7官方财务事实内容SHA-256：{manifest['content_sha256']}")
    print("V1.6产物未修改；冻结过程未读取市场价格或未来收益。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
