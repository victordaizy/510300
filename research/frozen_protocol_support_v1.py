from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


class ProtocolError(RuntimeError):
    """冻结协议、输入证据或研究边界不成立。"""


def resolve_path(root: Path, configured_path: str | Path) -> Path:
    path = Path(configured_path)
    return path if path.is_absolute() else root / path


def relative_path(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, set):
        return [jsonable(item) for item in sorted(value, key=str)]
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.datetime64):
        return None if np.isnat(value) else pd.Timestamp(value).isoformat()
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def strict_json_dumps(value: Any, *, indent: int | None = 2) -> str:
    return json.dumps(
        jsonable(value),
        ensure_ascii=False,
        indent=indent,
        sort_keys=False,
        allow_nan=False,
    )


def canonical_json_sha256(
    value: Mapping[str, Any],
    *,
    excluded_keys: Iterable[str] = (),
) -> str:
    excluded = set(excluded_keys)
    material = {key: item for key, item in value.items() if key not in excluded}
    payload = json.dumps(
        jsonable(material),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(strict_json_dumps(value) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def atomic_write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def frozen_file_inventory(
    root: Path,
    entries: Sequence[tuple[str | Path, str]],
) -> list[dict[str, Any]]:
    inventory: dict[str, dict[str, Any]] = {}
    for configured_path, role in entries:
        path = resolve_path(root, configured_path)
        if not path.is_file():
            raise ProtocolError(f"冻结文件不存在：{configured_path}")
        relative = relative_path(root, path)
        inventory[relative] = {
            "path": relative,
            "role": role,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return [inventory[key] for key in sorted(inventory)]


def validate_frozen_manifest(
    root: Path,
    manifest_path: Path,
    expected_manifest_sha256: str,
    *,
    project_id: str,
    bundle_status: str,
) -> dict[str, Any]:
    if not manifest_path.is_file():
        raise ProtocolError("冻结清单不存在")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("project_id") != project_id:
        raise ProtocolError("冻结清单项目标识不匹配")
    if manifest.get("bundle_status") != bundle_status:
        raise ProtocolError("冻结清单状态不匹配")
    recorded = str(manifest.get("manifest_sha256", ""))
    actual = canonical_json_sha256(manifest, excluded_keys=("manifest_sha256",))
    if not recorded or recorded != actual:
        raise ProtocolError("冻结清单自身内容摘要不匹配")
    if expected_manifest_sha256.lower() != recorded.lower():
        raise ProtocolError("命令提供的冻结清单摘要不匹配")

    drift: list[dict[str, Any]] = []
    for entry in manifest.get("frozen_files", []):
        path = resolve_path(root, entry["path"])
        if not path.is_file():
            drift.append({"path": entry["path"], "reason": "MISSING"})
            continue
        actual_size = path.stat().st_size
        actual_sha256 = sha256_file(path)
        if actual_size != int(entry["size_bytes"]) or actual_sha256 != entry["sha256"]:
            drift.append(
                {
                    "path": entry["path"],
                    "reason": "HASH_OR_SIZE_DRIFT",
                    "expected_size": entry["size_bytes"],
                    "actual_size": actual_size,
                    "expected_sha256": entry["sha256"],
                    "actual_sha256": actual_sha256,
                }
            )
    if drift:
        raise ProtocolError(f"冻结文件发生漂移：{strict_json_dumps(drift)}")
    return manifest


def acquire_immutable_claim(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    descriptor = os.open(path, flags)
    try:
        content = (strict_json_dumps(payload) + "\n").encode("utf-8")
        os.write(descriptor, content)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
