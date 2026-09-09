"""零付费来源 V1.2 内容寻址原始层。"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from market_data.etf_primary_market import ProviderSnapshot


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def write_content_addressed_raw(
    snapshot: ProviderSnapshot,
    raw_root: Path,
    kind: str,
    observed_at: datetime,
    *,
    root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """按响应内容 SHA-256 落盘；同内容复用，绝不覆盖既有文件。"""

    if not kind or any(character in kind for character in "\\/:\0"):
        raise ValueError(f"原始响应类别无效：{kind}")
    envelope = {
        "provider": snapshot.record.get("source"),
        "observed_at": observed_at.isoformat(),
        "normalized_record": snapshot.record,
        "raw_payload": snapshot.raw_payload,
    }
    encoded = (
        json.dumps(envelope, ensure_ascii=False, indent=2, default=str) + "\n"
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    target_directory = raw_root / kind / observed_at.strftime("%Y%m%d")
    target_directory.mkdir(parents=True, exist_ok=True)
    target = target_directory / f"{digest}.json"
    existing_identical = False
    try:
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError:
        if target.read_bytes() != encoded:
            raise RuntimeError(f"内容寻址路径发生哈希碰撞或文件损坏：{target}")
        existing_identical = True
    else:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    return {
        "path": _relative(target, root),
        "sha256": digest,
        "bytes": len(encoded),
        "content_addressed": True,
        "existing_identical": existing_identical,
    }


def persist_record_atomic(
    record: dict[str, Any],
    output_file: Path,
    key: str,
) -> dict[str, Any]:
    """以同文件系统临时文件原子替换解析层，避免半写入 Parquet。"""

    if key not in record:
        raise ValueError(f"解析记录缺少去重键：{key}")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fetched = pd.DataFrame([record])
    existing = pd.read_parquet(output_file) if output_file.exists() else pd.DataFrame()
    combined = (
        pd.concat([existing, fetched], ignore_index=True)
        if not existing.empty
        else fetched
    )
    combined = combined.drop_duplicates(key, keep="last").sort_values(key)
    temporary = output_file.with_name(
        f"{output_file.name}.{os.getpid()}.{os.urandom(6).hex()}.tmp"
    )
    try:
        combined.to_parquet(temporary, index=False)
        with temporary.open("r+b") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, output_file)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "path": _relative(output_file, PROJECT_ROOT),
        "sha256": hashlib.sha256(output_file.read_bytes()).hexdigest(),
        "rows": int(len(combined)),
        "atomic_replace": True,
    }
