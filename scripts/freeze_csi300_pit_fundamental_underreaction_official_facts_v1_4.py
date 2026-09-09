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


from research.csi300_pit_fundamental_underreaction_official_facts_v1_4 import (  # noqa: E402
    PARSER_VERSION,
    REQUIRED_METRICS,
)
from scripts import (  # noqa: E402
    freeze_csi300_pit_fundamental_underreaction_official_facts_v1_1 as _base,
)


CONFIG_PATH = ROOT / "config/csi300_pit_fundamental_underreaction_official_facts_v1_4.yaml"
TIMEZONE = ZoneInfo("Asia/Shanghai")
FROZEN_STATUS = (
    "FROZEN_SEMANTIC_TABLE_AND_HASH_BOUND_IMAGE_APPENDIX_REVISION_AFTER_V1_3_"
    "PARTIAL_BEFORE_V1_4_BULK_AND_ANY_FUTURE_RETURN_READ"
)


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def frozen_paths(config: dict[str, Any]) -> list[Path]:
    gate = config["representative_revision_gate"]
    manual = config["extraction"]["manual_appendix"]
    return [
        CONFIG_PATH,
        _base.project_path(config["artifacts"]["protocol_document"]),
        ROOT / "research/csi300_pit_fundamental_underreaction_official_facts_v1.py",
        ROOT / "research/csi300_pit_fundamental_underreaction_official_facts_v1_1.py",
        ROOT / "research/csi300_pit_fundamental_underreaction_official_facts_v1_2.py",
        ROOT / "research/csi300_pit_fundamental_underreaction_official_facts_v1_3.py",
        ROOT / "research/csi300_pit_fundamental_underreaction_official_facts_v1_4.py",
        ROOT / "scripts/windows_ocr_financial_statement_v1_3.ps1",
        ROOT / "scripts/collect_csi300_pit_fundamental_underreaction_official_facts_v1.py",
        ROOT / "scripts/collect_csi300_pit_fundamental_underreaction_official_facts_v1_4.py",
        ROOT / "scripts/freeze_csi300_pit_fundamental_underreaction_official_facts_v1_4.py",
        ROOT / "scripts/prepare_csi300_pit_official_facts_v1_4_replay_sources.py",
        ROOT / "scripts/replay_csi300_pit_official_facts_v1_4.py",
        ROOT / "tests/test_csi300_pit_fundamental_underreaction_official_facts_v1_4.py",
        _base.project_path(manual["path"]),
        _base.project_path(gate["source_manifest"]["path"]),
        _base.project_path(gate["replay_receipt"]["path"]),
    ]


def validate_existing(path: Path) -> None:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    _base.verify_manifest(
        path,
        expected_status=FROZEN_STATUS,
        expected_content_sha256=str(manifest.get("content_sha256") or ""),
    )
    print(f"V1.4官方财务事实冻结清单已存在且匹配：{_base.relative_path(path)}")


def _checked_json(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"{label}不存在：{path}")
    actual = _base.sha256_file(path)
    if actual != expected_sha256:
        raise RuntimeError(f"{label}哈希不一致：{actual}|{expected_sha256}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_representative_revision_gate(config: dict[str, Any]) -> dict[str, Any]:
    gate = config["representative_revision_gate"]
    expected_ids = [str(value) for value in gate["target_announcement_ids"]]
    if len(expected_ids) != len(set(expected_ids)):
        raise RuntimeError("V1.4固定回放公告ID存在重复")

    source_rule = gate["source_manifest"]
    source_path = _base.project_path(source_rule["path"])
    source = _checked_json(
        source_path,
        str(source_rule["expected_sha256"]),
        "V1.4固定回放源清单",
    )
    source_results = {
        str(row["announcement_id"]): row for row in source.get("results") or []
    }
    if set(source_results) != set(expected_ids):
        raise RuntimeError("V1.4固定回放源清单公告集合不一致")
    if int(source.get("target_count") or -1) != len(expected_ids):
        raise RuntimeError("V1.4固定回放源清单数量不一致")
    if source.get("market_price_read") is not False or source.get("future_return_read") is not False:
        raise RuntimeError("V1.4固定回放源清单违反收益禁读规则")
    if source.get("return_evaluation") != "NOT_ALLOWED":
        raise RuntimeError("V1.4固定回放源清单收益评价状态不一致")

    receipt_rule = gate["replay_receipt"]
    receipt_path = _base.project_path(receipt_rule["path"])
    receipt = _checked_json(
        receipt_path,
        str(receipt_rule["expected_sha256"]),
        "V1.4固定回放收据",
    )
    receipt_checks = {
        "parser_version": receipt.get("parser_version")
        == receipt_rule["expected_parser_version"]
        == PARSER_VERSION,
        "target_count": int(receipt.get("target_count", -1))
        == int(receipt_rule["expected_target_count"]),
        "complete_count": int(receipt.get("complete_count", -1))
        == int(receipt_rule["expected_complete_count"]),
        "incomplete_count": int(receipt.get("incomplete_count", -1))
        == int(receipt_rule["expected_incomplete_count"]),
        "all_nine_metric_replay_passed": receipt.get("all_nine_metric_replay_passed")
        is True,
        "market_price_read": receipt.get("market_price_read") is False,
        "future_return_read": receipt.get("future_return_read") is False,
        "return_evaluation": receipt.get("return_evaluation")
        == receipt_rule["required_return_evaluation"],
    }
    failed = [name for name, passed in receipt_checks.items() if not passed]
    if failed:
        raise RuntimeError(f"V1.4固定回放收据不满足冻结门：{failed}")

    replay_results = {
        str(row["announcement_id"]): row for row in receipt.get("results") or []
    }
    if set(replay_results) != set(expected_ids):
        raise RuntimeError("V1.4固定回放收据公告集合不一致")
    pdf_entries: list[dict[str, Any]] = []
    for announcement_id in expected_ids:
        source_row = source_results[announcement_id]
        replay_row = replay_results[announcement_id]
        pdf_path = _base.project_path(source_row["path"])
        if not pdf_path.exists():
            raise FileNotFoundError(f"V1.4固定回放PDF不存在：{pdf_path}")
        content = pdf_path.read_bytes()
        observed_sha256 = _sha256_bytes(content)
        expected_sha256 = str(replay_row["expected_sha256"])
        checks = {
            "source_status": source_row.get("status") == "DOWNLOADED_SHA256_VERIFIED",
            "same_path": str(replay_row.get("path")) == str(source_row.get("path")),
            "sha256": observed_sha256 == expected_sha256,
            "size": len(content) == int(source_row["size_bytes"]),
            "document_complete": replay_row.get("document_complete") is True,
            "metric_count": int(replay_row.get("metric_count", -1))
            == int(gate["every_document_metric_count"]),
            "missing_metrics": not (replay_row.get("missing_metrics") or []),
        }
        failed_document = [name for name, passed in checks.items() if not passed]
        if failed_document:
            raise RuntimeError(
                f"V1.4固定回放PDF校验失败：{announcement_id}|{failed_document}"
            )
        pdf_entries.append(
            {
                "announcement_id": announcement_id,
                "path": _base.relative_path(pdf_path),
                "bytes": len(content),
                "sha256": observed_sha256,
                "metric_count": int(replay_row["metric_count"]),
            }
        )
        content = b""

    manual_rule = config["extraction"]["manual_appendix"]
    manual_path = _base.project_path(manual_rule["path"])
    manual = _checked_json(
        manual_path,
        str(manual_rule["expected_sha256"]),
        "V1.4哈希绑定核验附录",
    )
    documents = manual.get("documents") or {}
    if len(documents) != int(manual_rule["expected_document_count"]):
        raise RuntimeError("V1.4哈希绑定核验附录文档数不一致")
    required = set(REQUIRED_METRICS)
    for pdf_sha256, document in documents.items():
        metric_ids = [str(fact.get("metric_id") or "") for fact in document.get("facts") or []]
        if str(document.get("official_pdf_sha256")) != str(pdf_sha256):
            raise RuntimeError("V1.4哈希绑定核验附录键与文档哈希不一致")
        if set(metric_ids) != required or len(metric_ids) != len(required):
            raise RuntimeError("V1.4哈希绑定核验附录未恰好覆盖九项事实")

    manual_ids = set(str(value) for value in gate["hash_bound_appendix_announcement_ids"])
    observed_manual_ids = {
        str(document["document_key"]).rsplit("_", 1)[-1]
        for document in documents.values()
    }
    if manual_ids != observed_manual_ids:
        raise RuntimeError("V1.4哈希绑定核验附录公告集合不一致")

    return {
        "source_manifest_path": _base.relative_path(source_path),
        "source_manifest_sha256": _base.sha256_file(source_path),
        "replay_receipt_path": _base.relative_path(receipt_path),
        "replay_receipt_sha256": _base.sha256_file(receipt_path),
        "target_document_count": len(expected_ids),
        "complete_target_count": int(receipt["complete_count"]),
        "incomplete_target_count": int(receipt["incomplete_count"]),
        "hash_bound_appendix_path": _base.relative_path(manual_path),
        "hash_bound_appendix_sha256": _base.sha256_file(manual_path),
        "hash_bound_appendix_document_count": len(documents),
        "pdf_entries": pdf_entries,
        "market_price_read": False,
        "future_return_read": False,
        "return_evaluation": "NOT_ALLOWED",
    }


def main() -> int:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if config["extraction"]["parser_version"] != PARSER_VERSION:
        raise RuntimeError("V1.4配置解析器版本与代码不一致")
    if tuple(config["extraction"]["required_metrics"]) != tuple(REQUIRED_METRICS):
        raise RuntimeError("V1.4配置九项指标顺序与代码不一致")

    manifest_path = _base.project_path(config["artifacts"]["protocol_manifest"])
    if manifest_path.exists():
        validate_existing(manifest_path)
        return 0

    checkpoint_root = _base.project_path(config["artifacts"]["checkpoint_root"])
    if checkpoint_root.exists() and any(checkpoint_root.rglob("*.json.gz")):
        raise RuntimeError("已发现V1.4官方PDF采集检查点，拒绝事后冻结")

    parent = _base.validate_parent_manifest(config)
    superseded = _base.validate_superseded_v1(config)
    inputs = _base.validate_inputs(config)
    replay_gate = validate_representative_revision_gate(config)
    paths = frozen_paths(config)
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"缺少待冻结文件：{missing}")
    files = [_base.file_entry(path) for path in paths]
    parent_bundle = {
        **parent,
        "superseded_official_fact_v1_3": superseded,
        "representative_revision_gate": replay_gate,
    }
    manifest: dict[str, Any] = {
        "manifest_id": "CSI300_PIT_FUNDAMENTAL_UNDERREACTION_OFFICIAL_FACTS_V1_4_MANIFEST",
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
    print(f"V1.4官方财务事实协议已冻结：{_base.relative_path(manifest_path)}")
    print(f"V1.4官方财务事实内容SHA-256：{manifest['content_sha256']}")
    print("V1.3检查点未修改；冻结过程未读取市场价格或未来收益。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
