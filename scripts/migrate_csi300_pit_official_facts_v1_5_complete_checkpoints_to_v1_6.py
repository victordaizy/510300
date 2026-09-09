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


from research.csi300_pit_fundamental_underreaction_official_facts_v1_6 import (  # noqa: E402
    PARSER_VERSION,
    REQUIRED_METRICS,
)
from scripts import (  # noqa: E402
    collect_csi300_pit_fundamental_underreaction_official_facts_v1 as _collector_base,
)


CONFIG_PATH = ROOT / "config/csi300_pit_fundamental_underreaction_official_facts_v1_6.yaml"
TIMEZONE = ZoneInfo("Asia/Shanghai")
FROZEN_STATUS = (
    "FROZEN_ANCHORED_IMAGE_OCR_INVERSE_CORE_LABEL_AND_NET_RECEIVABLE_"
    "RECONCILIATION_AFTER_V1_5_FULL_COLLECTION_BEFORE_V1_6_GAP_REPARSE_"
    "AND_ANY_FUTURE_RETURN_READ"
)


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_gzip_json(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _atomic_write_gzip_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as handle:
            handle.write(encoded)
    temporary.replace(path)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _restamp_complete_checkpoint(
    source: dict[str, Any],
    *,
    source_path: Path,
    source_sha256: str,
) -> dict[str, Any]:
    if source.get("checkpoint_status") != "PARSED_COMPLETE":
        raise RuntimeError(f"拒绝迁移非完整检查点：{source_path}")
    if source.get("document_complete") is not True:
        raise RuntimeError(f"完整检查点的document_complete不为true：{source_path}")
    if source.get("missing_metrics"):
        raise RuntimeError(f"完整检查点仍有缺失指标：{source_path}")
    metrics = source.get("metrics") or []
    observed_ids = {str(metric.get("metric_id")) for metric in metrics}
    if observed_ids != set(REQUIRED_METRICS) or len(metrics) != len(REQUIRED_METRICS):
        raise RuntimeError(f"完整检查点不是九项唯一指标：{source_path}")
    source_parser_version = str(source.get("parser_version") or "")
    if not source_parser_version:
        raise RuntimeError(f"完整检查点缺少源解析器版本：{source_path}")

    migrated = dict(source)
    migrated_metrics: list[dict[str, Any]] = []
    for source_metric in metrics:
        metric = dict(source_metric)
        try:
            locator = json.loads(str(metric.get("source_locator") or "{}"))
        except json.JSONDecodeError:
            locator = {"legacy_source_locator": str(metric.get("source_locator") or "")}
        previous = str(locator.get("parser_version") or source_parser_version)
        if previous and previous != PARSER_VERSION:
            locator.setdefault("base_parser_version", previous)
        locator["parser_version"] = PARSER_VERSION
        metric["source_locator"] = json.dumps(
            locator,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        migrated_metrics.append(metric)
    migrated["metrics"] = migrated_metrics
    migrated["parser_version"] = PARSER_VERSION
    migrated["v1_6_checkpoint_migration_receipt"] = {
        "status": "PASS_REUSED_V1_5_PARSED_COMPLETE_WITH_NO_MISSING_METRICS",
        "source_checkpoint_path": source_path.relative_to(ROOT).as_posix(),
        "source_checkpoint_sha256": source_sha256,
        "source_parser_version": source_parser_version,
        "target_parser_version": PARSER_VERSION,
        "metric_count": len(migrated_metrics),
        "source_document_complete": True,
        "source_missing_metric_count": 0,
        "market_price_read": False,
        "future_return_read": False,
        "return_evaluation": "NOT_ALLOWED",
    }
    return migrated


def _validate_existing_destination(
    destination: Path,
    *,
    source_sha256: str,
) -> None:
    payload = _read_gzip_json(destination)
    migration = payload.get("v1_6_checkpoint_migration_receipt") or {}
    checks = {
        "parser_version": payload.get("parser_version") == PARSER_VERSION,
        "checkpoint_status": payload.get("checkpoint_status") == "PARSED_COMPLETE",
        "document_complete": payload.get("document_complete") is True,
        "missing_metrics": not (payload.get("missing_metrics") or []),
        "metric_count": len(payload.get("metrics") or []) == len(REQUIRED_METRICS),
        "migration_status": migration.get("status")
        == "PASS_REUSED_V1_5_PARSED_COMPLETE_WITH_NO_MISSING_METRICS",
        "source_sha256": migration.get("source_checkpoint_sha256") == source_sha256,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"已有V1.6迁移检查点校验失败：{destination}|{failed}")


def _verify_frozen_inputs(config: dict[str, Any]) -> None:
    manifest_rule = config["supersedes"]["protocol_manifest"]
    _collector_base.verify_manifest(
        _collector_base.project_path(manifest_rule["path"]),
        expected_status=manifest_rule["expected_status"],
        expected_content_sha256=manifest_rule["expected_content_sha256"],
        expected_file_sha256=manifest_rule["expected_file_sha256"],
    )
    self_manifest_path = _collector_base.project_path(
        config["artifacts"]["protocol_manifest"]
    )
    _collector_base.verify_manifest(
        self_manifest_path,
        expected_status=FROZEN_STATUS,
        expected_content_sha256=None,
        verify_own_content_payload=True,
    )
    receipt_rule = config["supersedes"]["full_collection_receipt"]
    receipt_path = _collector_base.project_path(receipt_rule["path"])
    if _sha256_file(receipt_path) != receipt_rule["expected_sha256"]:
        raise RuntimeError("V1.5全量收据哈希不一致")


def main() -> int:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if config["extraction"]["parser_version"] != PARSER_VERSION:
        raise RuntimeError("V1.6配置解析器版本与迁移脚本不一致")
    _verify_frozen_inputs(config)

    source_root = _collector_base.project_path(
        config["supersedes"]["legacy_checkpoint_root"]
    )
    target_root = _collector_base.project_path(config["artifacts"]["checkpoint_root"])
    receipt_path = _collector_base.project_path(
        config["artifacts"]["checkpoint_migration_receipt"]
    )
    source_paths = sorted(source_root.rglob("*.json.gz"))
    if len(source_paths) != int(
        config["supersedes"]["expected_legacy_checkpoint_count"]
    ):
        raise RuntimeError(
            f"V1.5源检查点数量不一致：{len(source_paths)}|"
            f"{config['supersedes']['expected_legacy_checkpoint_count']}"
        )

    complete_count = 0
    incomplete_count = 0
    created_count = 0
    reused_count = 0
    digest = hashlib.sha256()
    for completed, source_path in enumerate(source_paths, start=1):
        source_bytes = source_path.read_bytes()
        source_sha256 = _sha256_bytes(source_bytes)
        source = _read_gzip_json(source_path)
        status = str(source.get("checkpoint_status") or "")
        if status == "PARSED_INCOMPLETE":
            incomplete_count += 1
            continue
        if status != "PARSED_COMPLETE":
            raise RuntimeError(f"V1.5源检查点存在非终态状态：{source_path}|{status}")
        complete_count += 1
        relative = source_path.relative_to(source_root)
        destination = target_root / relative
        if destination.exists():
            _validate_existing_destination(
                destination,
                source_sha256=source_sha256,
            )
            reused_count += 1
        else:
            migrated = _restamp_complete_checkpoint(
                source,
                source_path=source_path,
                source_sha256=source_sha256,
            )
            _atomic_write_gzip_json(destination, migrated)
            created_count += 1
        target_sha256 = _sha256_file(destination)
        digest.update(
            (
                f"{relative.as_posix()}|{source_sha256}|{target_sha256}\n"
            ).encode("utf-8")
        )
        if completed % 500 == 0 or completed == len(source_paths):
            print(
                f"V1.5→V1.6检查点 {completed:,}/{len(source_paths):,}｜"
                f"完整={complete_count:,}｜缺口跳过={incomplete_count:,}｜"
                f"新建={created_count:,}｜复用={reused_count:,}",
                flush=True,
            )

    if complete_count != int(config["supersedes"]["expected_complete_checkpoint_count"]):
        raise RuntimeError("V1.5完整检查点计数不一致")
    if incomplete_count != int(
        config["supersedes"]["expected_incomplete_checkpoint_count"]
    ):
        raise RuntimeError("V1.5不完整检查点计数不一致")
    target_paths = sorted(target_root.rglob("*.json.gz"))
    if len(target_paths) != complete_count:
        raise RuntimeError(
            f"V1.6迁移目标检查点计数不一致：{len(target_paths)}|{complete_count}"
        )
    receipt = {
        "protocol_id": "CSI300_PIT_OFFICIAL_FACTS_V1_5_COMPLETE_TO_V1_6_MIGRATION",
        "status": "PASS_V1_5_COMPLETE_CHECKPOINTS_MIGRATED_TO_V1_6",
        "created_at": datetime.now(TIMEZONE).isoformat(),
        "source_checkpoint_root": source_root.relative_to(ROOT).as_posix(),
        "target_checkpoint_root": target_root.relative_to(ROOT).as_posix(),
        "source_parser_version": config["supersedes"]["full_collection_receipt"][
            "expected_parser_version"
        ],
        "target_parser_version": PARSER_VERSION,
        "source_checkpoint_count": len(source_paths),
        "source_complete_count": complete_count,
        "source_incomplete_count": incomplete_count,
        "migrated_complete_count": len(target_paths),
        "migrated_incomplete_count": 0,
        "created_in_this_run_count": created_count,
        "reused_in_this_run_count": reused_count,
        "source_to_target_hash_ledger_sha256": digest.hexdigest(),
        "legacy_checkpoint_mutation_forbidden": True,
        "market_price_read": False,
        "future_return_read": False,
        "return_evaluation": "NOT_ALLOWED",
    }
    _atomic_write_json(receipt_path, receipt)
    print(
        f"V1.5完整检查点迁移完成：{complete_count:,}份；"
        f"V1.5缺口未迁移：{incomplete_count:,}份。",
        flush=True,
    )
    print("迁移过程未读取市场价格或未来收益。", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
