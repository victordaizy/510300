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


CONFIG_PATH = ROOT / "config/csi300_pit_fundamental_underreaction_official_facts_v1.yaml"
TIMEZONE = ZoneInfo("Asia/Shanghai")
FROZEN_STATUS = (
    "FROZEN_BEFORE_BULK_OFFICIAL_PDF_ACQUISITION_AND_ANY_FUTURE_RETURN_READ"
)


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


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
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def frozen_paths(config: dict[str, Any]) -> list[Path]:
    return [
        CONFIG_PATH,
        project_path(config["artifacts"]["protocol_document"]),
        ROOT / "research/csi300_pit_fundamental_underreaction_official_facts_v1.py",
        ROOT / "scripts/collect_csi300_pit_fundamental_underreaction_official_facts_v1.py",
        ROOT / "scripts/freeze_csi300_pit_fundamental_underreaction_official_facts_v1.py",
        ROOT / "tests/test_csi300_pit_fundamental_underreaction_official_facts_v1.py",
        ROOT
        / "tests/test_csi300_pit_fundamental_underreaction_official_facts_collection_v1.py",
    ]


def validate_parent_manifest(config: dict[str, Any]) -> dict[str, Any]:
    parent = config["parent_freeze"]
    path = project_path(parent["path"])
    if not path.exists():
        raise FileNotFoundError(f"主研究冻结清单不存在：{path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != parent["expected_status"]:
        raise RuntimeError("主研究冻结状态不一致")
    if manifest.get("content_sha256") != parent["expected_content_sha256"]:
        raise RuntimeError("主研究冻结内容哈希不一致")
    mismatches = [
        item["path"]
        for item in manifest.get("files", [])
        if not project_path(item["path"]).exists()
        or sha256_file(project_path(item["path"])) != item["sha256"]
    ]
    if mismatches:
        raise RuntimeError(f"主研究冻结文件漂移：{mismatches}")
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "file_sha256": sha256_file(path),
        "content_sha256": manifest["content_sha256"],
        "status": manifest["status"],
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
        entry = {
            "label": label,
            "path": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": actual,
        }
        if label == "target_inventory":
            inventory = pd.read_parquet(path, columns=[item["target_flag"]])
            target_count = int(inventory[item["target_flag"]].fillna(False).astype(bool).sum())
            if target_count != int(item["expected_target_event_count"]):
                raise RuntimeError(
                    f"目标事件数不一致：{target_count}|{item['expected_target_event_count']}"
                )
            entry["target_event_count"] = target_count
        inputs.append(entry)
    return inputs


def validate_existing(path: Path) -> None:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != FROZEN_STATUS:
        raise RuntimeError("已有官方财务事实冻结清单状态错误")
    mismatches = [
        item["path"]
        for item in manifest.get("files", [])
        if not project_path(item["path"]).exists()
        or sha256_file(project_path(item["path"])) != item["sha256"]
    ]
    if mismatches:
        raise RuntimeError(f"已有官方财务事实冻结文件漂移：{mismatches}")
    payload = {
        "status": manifest["status"],
        "protocol_id": manifest["protocol_id"],
        "version": manifest["version"],
        "parent_freeze": manifest["parent_freeze"],
        "inputs": manifest["inputs"],
        "files": manifest["files"],
    }
    if canonical_sha256(payload) != manifest.get("content_sha256"):
        raise RuntimeError("已有官方财务事实冻结清单自身哈希不一致")
    print(f"官方财务事实冻结清单已存在且匹配：{path.relative_to(ROOT).as_posix()}")


def main() -> int:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    manifest_path = project_path(config["artifacts"]["protocol_manifest"])
    if manifest_path.exists():
        validate_existing(manifest_path)
        return 0

    checkpoint_root = project_path(config["artifacts"]["checkpoint_root"])
    if checkpoint_root.exists() and any(checkpoint_root.rglob("*.json.gz")):
        raise RuntimeError("已发现官方PDF采集检查点，拒绝事后冻结")
    forbidden_pre_freeze_outputs = [
        project_path(config["artifacts"]["facts"]),
        project_path(config["artifacts"]["dependency_ledger"]),
        project_path(config["artifacts"]["receipt"]),
    ]
    seen = [path.relative_to(ROOT).as_posix() for path in forbidden_pre_freeze_outputs if path.exists()]
    if seen:
        raise RuntimeError(f"已发现正式采集输出，拒绝事后冻结：{seen}")

    parent = validate_parent_manifest(config)
    inputs = validate_inputs(config)
    paths = frozen_paths(config)
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"缺少待冻结文件：{missing}")
    files = [file_entry(path) for path in paths]
    manifest: dict[str, Any] = {
        "manifest_id": "CSI300_PIT_FUNDAMENTAL_UNDERREACTION_OFFICIAL_FACTS_V1_MANIFEST",
        "status": FROZEN_STATUS,
        "created_at": datetime.now(TIMEZONE).isoformat(),
        "protocol_id": config["protocol"]["protocol_id"],
        "version": config["protocol"]["version"],
        "parent_freeze": parent,
        "inputs": inputs,
        "files": files,
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
    print(f"官方财务事实协议已冻结：{manifest_path.relative_to(ROOT).as_posix()}")
    print(f"官方财务事实内容SHA-256：{manifest['content_sha256']}")
    print("冻结过程未读取市场价格或未来收益。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
