from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


CONFIG_PATH = ROOT / "config/csi300_pit_fundamental_underreaction_official_facts_v1_1.yaml"
TIMEZONE = ZoneInfo("Asia/Shanghai")
FROZEN_STATUS = (
    "FROZEN_PARSER_REVISION_AFTER_REPRESENTATIVE_PILOT_"
    "BEFORE_V1_1_BULK_AND_ANY_FUTURE_RETURN_READ"
)


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def relative_path(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_only(path: Path) -> None:
    mode = path.stat().st_mode
    path.chmod(mode & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)


def file_entry(path: Path) -> dict[str, Any]:
    return {
        "path": relative_path(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def frozen_paths(config: dict[str, Any]) -> list[Path]:
    return [
        CONFIG_PATH,
        project_path(config["artifacts"]["protocol_document"]),
        ROOT / "research/csi300_pit_fundamental_underreaction_official_facts_v1_1.py",
        ROOT / "scripts/collect_csi300_pit_fundamental_underreaction_official_facts_v1_1.py",
        ROOT / "scripts/freeze_csi300_pit_fundamental_underreaction_official_facts_v1_1.py",
        ROOT / "tests/test_csi300_pit_fundamental_underreaction_official_facts_v1_1.py",
    ]


def verify_manifest(
    path: Path,
    *,
    expected_status: str,
    expected_content_sha256: str,
    expected_file_sha256: str | None = None,
    verify_own_content_payload: bool = True,
) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"冻结清单不存在：{path}")
    if expected_file_sha256 and sha256_file(path) != expected_file_sha256:
        raise RuntimeError(f"冻结清单文件哈希不一致：{relative_path(path)}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != expected_status:
        raise RuntimeError(f"冻结状态不一致：{relative_path(path)}")
    if manifest.get("content_sha256") != expected_content_sha256:
        raise RuntimeError(f"冻结内容哈希不一致：{relative_path(path)}")
    mismatches = [
        item["path"]
        for item in manifest.get("files", [])
        if not project_path(item["path"]).exists()
        or sha256_file(project_path(item["path"])) != item["sha256"]
    ]
    if mismatches:
        raise RuntimeError(f"冻结文件漂移：{mismatches}")
    if verify_own_content_payload:
        payload = {
            "status": manifest["status"],
            "protocol_id": manifest["protocol_id"],
            "version": manifest["version"],
            "parent_freeze": manifest["parent_freeze"],
            "inputs": manifest["inputs"],
            "files": manifest["files"],
        }
        if canonical_sha256(payload) != manifest.get("content_sha256"):
            raise RuntimeError(f"冻结清单自身内容校验失败：{relative_path(path)}")
    return manifest


def validate_parent_manifest(config: dict[str, Any]) -> dict[str, Any]:
    parent = config["parent_freeze"]
    path = project_path(parent["path"])
    manifest = verify_manifest(
        path,
        expected_status=parent["expected_status"],
        expected_content_sha256=parent["expected_content_sha256"],
        verify_own_content_payload=False,
    )
    return {
        "path": relative_path(path),
        "file_sha256": sha256_file(path),
        "content_sha256": manifest["content_sha256"],
        "status": manifest["status"],
    }


def validate_superseded_v1(config: dict[str, Any]) -> dict[str, Any]:
    supersedes = config["supersedes"]
    protocol = supersedes["protocol_manifest"]
    manifest_path = project_path(protocol["path"])
    manifest = verify_manifest(
        manifest_path,
        expected_status=protocol["expected_status"],
        expected_content_sha256=protocol["expected_content_sha256"],
        expected_file_sha256=protocol["expected_file_sha256"],
    )

    receipt_rule = supersedes["pilot_receipt"]
    receipt_path = project_path(receipt_rule["path"])
    if not receipt_path.exists():
        raise FileNotFoundError(f"V1试运行收据不存在：{receipt_path}")
    receipt_sha256 = sha256_file(receipt_path)
    if receipt_sha256 != receipt_rule["expected_sha256"]:
        raise RuntimeError("V1试运行收据哈希不一致")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    checks = {
        "status": receipt.get("status") == receipt_rule["expected_status"],
        "parser_version": (
            receipt.get("parser_version") == receipt_rule["expected_parser_version"]
        ),
        "terminal_document_count": (
            receipt.get("counts", {}).get("terminal_document_count")
            == receipt_rule["expected_terminal_document_count"]
        ),
        "complete_document_count": (
            receipt.get("counts", {}).get("complete_document_count")
            == receipt_rule["expected_complete_document_count"]
        ),
        "incomplete_terminal_document_count": (
            receipt.get("counts", {}).get("incomplete_terminal_document_count")
            == receipt_rule["expected_incomplete_terminal_document_count"]
        ),
        "return_evaluation": (
            receipt.get("return_evaluation")
            == receipt_rule["required_return_evaluation"]
        ),
        "market_price_read": receipt.get("market_price_read") is False,
        "future_return_read": receipt.get("future_return_read") is False,
        "future_label_created": receipt.get("future_label_created") is False,
        "portfolio_return_calculated": (
            receipt.get("portfolio_return_calculated") is False
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"V1试运行收据不满足版本化修订前提：{failed}")

    legacy_root = project_path(supersedes["legacy_checkpoint_root"])
    legacy_checkpoint_count = (
        sum(1 for _ in legacy_root.rglob("*.json.gz")) if legacy_root.exists() else 0
    )
    if legacy_checkpoint_count != receipt_rule["expected_terminal_document_count"]:
        raise RuntimeError(
            "V1检查点数量与冻结试运行收据不一致："
            f"{legacy_checkpoint_count}|{receipt_rule['expected_terminal_document_count']}"
        )
    return {
        "manifest_path": relative_path(manifest_path),
        "manifest_file_sha256": sha256_file(manifest_path),
        "manifest_content_sha256": manifest["content_sha256"],
        "manifest_status": manifest["status"],
        "pilot_receipt_path": relative_path(receipt_path),
        "pilot_receipt_sha256": receipt_sha256,
        "pilot_receipt_status": receipt["status"],
        "pilot_parser_version": receipt["parser_version"],
        "legacy_checkpoint_root": relative_path(legacy_root),
        "legacy_checkpoint_count": legacy_checkpoint_count,
        "legacy_checkpoint_mutation_forbidden": True,
        "market_price_read": False,
        "future_return_read": False,
    }


def validate_inputs(config: dict[str, Any]) -> list[dict[str, Any]]:
    inputs: list[dict[str, Any]] = []
    for label, item in (
        ("target_inventory", config["inputs"]["target_inventory"]),
        ("official_events", config["inputs"]["official_events"]),
    ):
        path = project_path(item["path"])
        if not path.exists():
            raise FileNotFoundError(f"冻结输入不存在：{path}")
        actual = sha256_file(path)
        if actual != item["expected_sha256"]:
            raise RuntimeError(f"冻结输入哈希不一致：{label}")
        entry: dict[str, Any] = {
            "label": label,
            "path": relative_path(path),
            "bytes": path.stat().st_size,
            "sha256": actual,
        }
        if label == "target_inventory":
            inventory = pd.read_parquet(path, columns=[item["target_flag"]])
            target_count = int(
                inventory[item["target_flag"]].fillna(False).astype(bool).sum()
            )
            if target_count != int(item["expected_target_event_count"]):
                raise RuntimeError(
                    f"目标事件数不一致：{target_count}|{item['expected_target_event_count']}"
                )
            entry["target_event_count"] = target_count
        inputs.append(entry)
    return inputs


def validate_existing(path: Path) -> None:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    verify_manifest(
        path,
        expected_status=FROZEN_STATUS,
        expected_content_sha256=str(manifest.get("content_sha256") or ""),
    )
    print(f"V1.1官方财务事实冻结清单已存在且匹配：{relative_path(path)}")


def main() -> int:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    manifest_path = project_path(config["artifacts"]["protocol_manifest"])
    if manifest_path.exists():
        validate_existing(manifest_path)
        return 0

    checkpoint_root = project_path(config["artifacts"]["checkpoint_root"])
    if checkpoint_root.exists() and any(checkpoint_root.rglob("*.json.gz")):
        raise RuntimeError("已发现V1.1官方PDF采集检查点，拒绝事后冻结")

    parent = validate_parent_manifest(config)
    superseded = validate_superseded_v1(config)
    inputs = validate_inputs(config)
    paths = frozen_paths(config)
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"缺少待冻结文件：{missing}")
    files = [file_entry(path) for path in paths]
    parent_bundle = {
        **parent,
        "superseded_official_fact_v1": superseded,
        "representative_revision_gate": config["representative_revision_gate"],
    }
    manifest: dict[str, Any] = {
        "manifest_id": "CSI300_PIT_FUNDAMENTAL_UNDERREACTION_OFFICIAL_FACTS_V1_1_MANIFEST",
        "status": FROZEN_STATUS,
        "created_at": datetime.now(TIMEZONE).isoformat(),
        "protocol_id": config["protocol"]["protocol_id"],
        "version": config["protocol"]["version"],
        "parent_freeze": parent_bundle,
        "inputs": inputs,
        "files": files,
        "new_checkpoint_root": relative_path(checkpoint_root),
        "bulk_network_run_started": False,
        "market_price_read": False,
        "future_return_read": False,
        "future_label_created": False,
        "portfolio_return_calculated": False,
        "trading_authorization": False,
    }
    manifest["content_sha256"] = canonical_sha256(
        {
            "status": manifest["status"],
            "protocol_id": manifest["protocol_id"],
            "version": manifest["version"],
            "parent_freeze": manifest["parent_freeze"],
            "inputs": manifest["inputs"],
            "files": manifest["files"],
        }
    )
    atomic_write_json(manifest_path, manifest)
    for path in [*paths, manifest_path]:
        read_only(path)
    print(f"V1.1官方财务事实协议已冻结：{relative_path(manifest_path)}")
    print(f"V1.1官方财务事实内容SHA-256：{manifest['content_sha256']}")
    print("旧V1检查点未修改；冻结过程未读取市场价格或未来收益。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
