"""为行业预期差的可变结果输入建立内容寻址快照。"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


class IndustrySnapshotError(RuntimeError):
    """输入快照不完整或发生漂移。"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _exclusive_bytes(path: Path, content: bytes) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        return False
    except FileExistsError:
        if path.read_bytes() != content:
            raise IndustrySnapshotError(f"不可变文件已存在但内容不同：{path}")
        return True


def write_content_addressed_copy(
    source: Path,
    snapshot_root: Path,
    role: str,
    *,
    workspace_root: Path = ROOT,
) -> dict[str, Any]:
    if not source.is_file():
        raise IndustrySnapshotError(f"共享输入不存在：{source}")
    digest = _sha256(source)
    suffix = "".join(source.suffixes) or ".bin"
    target = snapshot_root / role / f"{digest}{suffix}"
    target.parent.mkdir(parents=True, exist_ok=True)
    existing_identical = False
    if target.is_file():
        if _sha256(target) != digest or target.stat().st_size != source.stat().st_size:
            raise IndustrySnapshotError(f"内容寻址目标与文件名哈希不一致：{target}")
        existing_identical = True
    else:
        created = False
        try:
            with target.open("xb") as destination, source.open("rb") as origin:
                created = True
                for block in iter(lambda: origin.read(1024 * 1024), b""):
                    destination.write(block)
                destination.flush()
                os.fsync(destination.fileno())
            if _sha256(target) != digest:
                raise IndustrySnapshotError(f"快照复制后哈希不一致：{target}")
        except FileExistsError:
            if _sha256(target) != digest:
                raise IndustrySnapshotError(f"并发快照内容不一致：{target}")
            existing_identical = True
        except Exception:
            if created and target.exists():
                target.unlink()
            raise
    return {
        "role": role,
        "source_path": _relative(source, workspace_root),
        "snapshot_path": _relative(target, workspace_root),
        "sha256": digest,
        "bytes": source.stat().st_size,
        "content_addressed": target.stem == digest,
        "existing_identical": existing_identical,
    }


def _latest_date(path: Path, column: str | None) -> str | None:
    if not column:
        return None
    frame = pd.read_parquet(path, columns=[column])
    values = pd.to_datetime(frame[column], errors="coerce").dropna()
    if values.empty:
        raise IndustrySnapshotError(f"输入日期列为空：{path}:{column}")
    return values.max().date().isoformat()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f"{path.name}.{os.getpid()}.{os.urandom(6).hex()}.tmp"
    )
    content = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode(
        "utf-8"
    )
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def freeze_inputs(
    config: dict[str, Any],
    generated_at: datetime,
    *,
    root: Path = ROOT,
) -> dict[str, Any]:
    contract = config["input_snapshot"]
    snapshot_root = root / contract["snapshot_directory"]
    records: list[dict[str, Any]] = []
    for role, specification in contract["roles"].items():
        source = root / specification["source"]
        record = write_content_addressed_copy(
            source,
            snapshot_root,
            role,
            workspace_root=root,
        )
        record["latest_date"] = _latest_date(
            source, specification.get("date_column")
        )
        records.append(record)
    records.sort(key=lambda item: item["role"])
    core = {
        "schema_version": "1.2.0",
        "snapshot_status": "PASS_CONTENT_ADDRESSED_INPUT_SNAPSHOT",
        "generated_at": generated_at.isoformat(),
        "shared_latest_consumed_by_evaluation": False,
        "records": records,
    }
    snapshot_id = hashlib.sha256(
        json.dumps(core, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    immutable_directory = root / contract["immutable_manifest_directory"]
    immutable_path = immutable_directory / f"{snapshot_id}.json"
    payload = {
        **core,
        "snapshot_id": snapshot_id,
        "immutable_manifest_path": _relative(immutable_path, root),
        "immutable_receipt": True,
    }
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode(
        "utf-8"
    )
    _exclusive_bytes(immutable_path, encoded)
    _atomic_json(root / contract["current_manifest"], payload)
    return payload


def load_verified_snapshot_manifest(
    config: dict[str, Any], *, root: Path = ROOT
) -> tuple[dict[str, Any], dict[str, Path]]:
    contract = config["input_snapshot"]
    current = root / contract["current_manifest"]
    if not current.is_file():
        raise IndustrySnapshotError(f"当前输入快照清单缺失：{current}")
    payload = json.loads(current.read_text(encoding="utf-8"))
    if payload.get("snapshot_status") != "PASS_CONTENT_ADDRESSED_INPUT_SNAPSHOT":
        raise IndustrySnapshotError("当前输入快照状态未通过")
    immutable = root / str(payload.get("immutable_manifest_path", ""))
    if not immutable.is_file() or immutable.read_bytes() != current.read_bytes():
        raise IndustrySnapshotError("当前指针与不可变快照清单不一致")
    records = payload.get("records")
    if not isinstance(records, list):
        raise IndustrySnapshotError("输入快照 records 格式错误")
    required_roles = set(contract["roles"])
    actual_roles = {str(record.get("role")) for record in records}
    if actual_roles != required_roles:
        raise IndustrySnapshotError(
            f"输入快照角色不一致：required={sorted(required_roles)}, actual={sorted(actual_roles)}"
        )
    mapping: dict[str, Path] = {}
    for record in records:
        snapshot = root / str(record["snapshot_path"])
        expected = str(record["sha256"])
        if (
            not snapshot.is_file()
            or snapshot.stem != expected
            or _sha256(snapshot) != expected
            or snapshot.stat().st_size != int(record["bytes"])
        ):
            raise IndustrySnapshotError(
                f"内容寻址输入验证失败：{record.get('role')}"
            )
        mapping[str(record["role"])] = snapshot
    return payload, mapping
