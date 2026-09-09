"""NBS V2 与只读观察器使用的不可变文件和版本工具。"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

TZ = ZoneInfo("Asia/Shanghai")


class ContractError(RuntimeError):
    """冻结合同不能满足时停止。"""


def now() -> str:
    return datetime.now(TZ).isoformat()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def encoded(value) -> bytes:
    return (json.dumps(clean(value), ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


def write_once(path: Path, value) -> None:
    payload = value if isinstance(value, bytes) else encoded(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ContractError(f"禁止覆盖不可变文件：{path}")
        return
    with path.open("xb") as stream:
        stream.write(payload)


def identity(root: Path, path: str | Path) -> dict:
    full = root / path
    return {"path": full.relative_to(root).as_posix(), "bytes": full.stat().st_size,
            "sha256": sha(full)}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def verify(root: Path, identities: list[dict]) -> None:
    for spec in identities:
        full = root / spec["path"]
        if not full.is_file() or sha(full) != spec["sha256"]:
            raise ContractError(f"冻结文件哈希不匹配：{spec['path']}")


def git(root: Path, *args: str) -> str:
    p = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", check=False)
    if p.returncode:
        raise ContractError(p.stderr.strip())
    return p.stdout.strip()


def committed(root: Path, paths: list[str]) -> None:
    for path in paths:
        git(root, "ls-files", "--error-unmatch", "--", path)
        if git(root, "status", "--porcelain", "--untracked-files=no", "--", path):
            raise ContractError(f"冻结范围存在未提交修改：{path}")


def csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8-sig")
