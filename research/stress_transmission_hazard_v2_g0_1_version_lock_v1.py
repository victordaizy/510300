"""510300 压力传导危险率 V2 修复后版本锁定的通用能力。

本模块只负责文件身份、Parquet 持久化语义、Git 范围解析和只增不改的
数据物化。它不读取收益，不拟合模型，也不产生任何 G2 指标。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd


PROGRAM_ID = "510300_STRESS_TRANSMISSION_HAZARD_V2"
EXECUTION_ID = (
    "510300_STRESS_TRANSMISSION_HAZARD_V2_"
    "G0_1_POST_REMEDIATION_VERSION_LOCK_V1"
)


class VersionLockError(RuntimeError):
    """版本锁定输入、文件身份或 Git 状态违反冻结契约。"""


def _lexical_absolute(path: Path) -> Path:
    """生成不解引用 Windows Junction 的绝对路径。"""

    return Path(os.path.abspath(os.fspath(path)))


def canonical_sha256(value: Any) -> str:
    """计算稳定 JSON 载荷摘要。"""

    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    """流式计算文件 SHA-256。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_relative_path(value: str) -> str:
    """把项目内路径规范为无上跳的 POSIX 相对路径。"""

    text = str(value).strip().replace("\\", "/")
    path = Path(text)
    if not text or path.is_absolute() or re.match(r"^[A-Za-z]:", text):
        raise VersionLockError(f"路径必须是项目相对路径：{value!r}")
    normalized = Path(os.path.normpath(text)).as_posix()
    if normalized in {".", ""} or normalized == ".." or normalized.startswith("../"):
        raise VersionLockError(f"路径越出项目根目录：{value!r}")
    return normalized


def project_path(root: Path, relative: str) -> Path:
    """按词法边界解析项目路径，允许项目内显式配置的 Junction。"""

    root = _lexical_absolute(root)
    normalized = normalize_relative_path(relative)
    candidate = _lexical_absolute(root / Path(normalized))
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise VersionLockError(f"路径越出项目根目录：{relative}") from exc
    return candidate


def file_evidence(path: Path, *, root: Path) -> dict[str, Any]:
    """生成普通文件的项目相对身份。"""

    path = _lexical_absolute(path)
    root = _lexical_absolute(root)
    if not path.is_file():
        raise VersionLockError(f"文件不存在：{path}")
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError as exc:
        raise VersionLockError(f"文件不在项目根目录内：{path}") from exc
    return {
        "path": relative,
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def frame_semantic_sha256(frame: pd.DataFrame) -> str:
    """计算含列顺序、dtype、行顺序和数值的 Parquet 持久化语义摘要。"""

    digest = hashlib.sha256()
    digest.update("\x1f".join(str(column) for column in frame.columns).encode("utf-8"))
    digest.update("\x1f".join(str(dtype) for dtype in frame.dtypes).encode("utf-8"))
    row_hashes = pd.util.hash_pandas_object(frame, index=False, categorize=True)
    digest.update(row_hashes.to_numpy(dtype="uint64").tobytes())
    digest.update(str(len(frame)).encode("ascii"))
    return digest.hexdigest()


def parquet_evidence(path: Path, *, root: Path) -> dict[str, Any]:
    """生成满足 G0.1 的 Parquet 文件身份和持久化语义身份。"""

    evidence = file_evidence(path, root=root)
    frame = pd.read_parquet(path)
    evidence.update(
        {
            "format": "PARQUET",
            "row_count": int(len(frame)),
            "column_count": int(len(frame.columns)),
            "columns": [str(column) for column in frame.columns],
            "persisted_semantic_sha256": frame_semantic_sha256(frame),
        }
    )
    return evidence


def verify_file_identity(
    root: Path,
    contract: Mapping[str, Any],
    *,
    label: str,
) -> Path:
    """核对路径、字节数和 SHA-256。"""

    path = project_path(root, str(contract["path"]))
    actual = file_evidence(path, root=root)
    expected = {
        "path": normalize_relative_path(str(contract["path"])),
        "bytes": int(contract["bytes"]),
        "sha256": str(contract["sha256"]),
    }
    if actual != expected:
        raise VersionLockError(
            f"{label}文件身份漂移：actual={actual}, expected={expected}"
        )
    return path


def verify_payload_sha256(
    payload: Mapping[str, Any],
    *,
    field: str,
    label: str,
) -> None:
    """核对 JSON 自包含载荷摘要。"""

    expected = str(payload.get(field) or "")
    if not expected:
        raise VersionLockError(f"{label}缺少 {field}")
    body = {key: value for key, value in payload.items() if key != field}
    actual = canonical_sha256(body)
    if actual != expected:
        raise VersionLockError(
            f"{label}的 {field} 不一致：actual={actual}, expected={expected}"
        )


def merge_expected_contract(
    contracts: dict[str, dict[str, Any]],
    contract: Mapping[str, Any],
    *,
    source: str,
) -> None:
    """去重合并一个带路径、字节数和 SHA-256 的数据契约。"""

    required = {"path", "bytes", "sha256"}
    if not required.issubset(contract):
        raise VersionLockError(f"{source} 缺少文件身份字段")
    relative = normalize_relative_path(str(contract["path"]))
    normalized = {
        "path": relative,
        "bytes": int(contract["bytes"]),
        "sha256": str(contract["sha256"]),
        "contract_sources": [source],
    }
    prior = contracts.get(relative)
    if prior is None:
        contracts[relative] = normalized
        return
    if any(prior[key] != normalized[key] for key in ("bytes", "sha256")):
        raise VersionLockError(
            f"同一数据路径存在冲突身份：{relative}，来源={source}"
        )
    if source not in prior["contract_sources"]:
        prior["contract_sources"].append(source)
        prior["contract_sources"].sort()


def resolve_git_scope(
    root: Path,
    *,
    roots: Sequence[str],
    filename_regex: str,
    allowed_suffixes: Sequence[str],
    explicit_paths: Sequence[str],
    excluded_paths: Sequence[str] = (),
) -> list[Path]:
    """按冻结规则解析本地提交范围，不读取 Git 之外的无关文件。"""

    pattern = re.compile(filename_regex)
    suffixes = {str(value).casefold() for value in allowed_suffixes}
    excluded = {normalize_relative_path(value) for value in excluded_paths}
    root = _lexical_absolute(root)
    resolved: dict[str, Path] = {}
    for relative_root in roots:
        directory = project_path(root, relative_root)
        if not directory.is_dir():
            raise VersionLockError(f"Git 范围根目录不存在：{directory}")
        for path in directory.rglob("*"):
            if not path.is_file() or path.suffix.casefold() not in suffixes:
                continue
            if not pattern.search(path.name):
                continue
            path = _lexical_absolute(path)
            relative = path.relative_to(root).as_posix()
            if relative not in excluded:
                resolved[relative] = path
    for value in explicit_paths:
        path = project_path(root, value)
        if not path.is_file():
            raise VersionLockError(f"显式 Git 文件不存在：{path}")
        relative = path.relative_to(root).as_posix()
        if relative not in excluded:
            resolved[relative] = path
    return [resolved[key] for key in sorted(resolved)]


def run_git(
    root: Path,
    arguments: Sequence[str],
    *,
    check: bool = True,
    text: bool = True,
) -> subprocess.CompletedProcess[Any]:
    """在指定工作树运行非交互 Git 命令。"""

    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=False,
        capture_output=True,
        text=text,
        encoding="utf-8" if text else None,
        errors="replace" if text else None,
    )
    if check and completed.returncode != 0:
        stderr = completed.stderr.strip() if text else repr(completed.stderr)
        raise VersionLockError(
            f"Git 命令失败：git {' '.join(arguments)}；{stderr}"
        )
    return completed


def git_head(root: Path) -> str:
    """返回当前工作树 HEAD 完整 SHA。"""

    return run_git(root, ["rev-parse", "HEAD"]).stdout.strip()


def git_branch(root: Path) -> str:
    """返回当前分支名；detached HEAD 返回空字符串。"""

    result = run_git(root, ["branch", "--show-current"])
    return result.stdout.strip()


def verify_commit_contains_files(
    root: Path,
    *,
    commit_sha: str,
    contracts: Iterable[Mapping[str, Any]],
) -> None:
    """确认指定提交包含冻结文件的精确字节内容。"""

    if git_head(root) != commit_sha:
        raise VersionLockError(
            f"工作树 HEAD 不是指定锁定提交：actual={git_head(root)}, expected={commit_sha}"
        )
    for contract in contracts:
        relative = normalize_relative_path(str(contract["path"]))
        result = run_git(
            root,
            ["show", f"{commit_sha}:{relative}"],
            check=False,
            text=False,
        )
        if result.returncode != 0:
            raise VersionLockError(f"锁定提交缺少必需文件：{relative}")
        content = bytes(result.stdout)
        actual = {
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
        expected = {
            "bytes": int(contract["bytes"]),
            "sha256": str(contract["sha256"]),
        }
        if actual != expected:
            raise VersionLockError(
                f"提交中的文件字节与冻结身份不一致：{relative}，"
                f"actual={actual}, expected={expected}"
            )


def verify_tracked_worktree_clean(root: Path) -> None:
    """只检查受 Git 跟踪文件；允许按 manifest 物化的未跟踪数据。"""

    result = run_git(root, ["status", "--porcelain=v1", "--untracked-files=no"])
    if result.stdout.strip():
        raise VersionLockError(
            "独立 worktree 的受跟踪文件不干净：\n" + result.stdout.strip()
        )


def verify_data_manifest(root: Path, manifest: Mapping[str, Any]) -> None:
    """逐文件核对数据清单，包括所有 Parquet 的持久化语义。"""

    verify_payload_sha256(
        manifest,
        field="manifest_payload_sha256",
        label="G0.1 数据清单",
    )
    for contract in manifest["data_files"]:
        path = verify_file_identity(root, contract, label=str(contract["path"]))
        if str(contract.get("format")) == "PARQUET":
            actual = parquet_evidence(path, root=root)
            expected_keys = (
                "path",
                "bytes",
                "sha256",
                "format",
                "row_count",
                "column_count",
                "columns",
                "persisted_semantic_sha256",
            )
            expected = {key: contract[key] for key in expected_keys}
            actual_subset = {key: actual[key] for key in expected_keys}
            if actual_subset != expected:
                raise VersionLockError(
                    f"Parquet 持久化语义漂移：{contract['path']}"
                )


def materialize_data_files(
    *,
    source_root: Path,
    target_root: Path,
    manifest: Mapping[str, Any],
) -> dict[str, int]:
    """把 manifest 文件只增不改地物化到独立 worktree。"""

    source_root = source_root.resolve()
    target_root = target_root.resolve()
    if source_root == target_root:
        raise VersionLockError("数据源根目录和目标 worktree 不能相同")
    copied_files = 0
    reused_files = 0
    copied_bytes = 0
    for contract in manifest["data_files"]:
        relative = str(contract["path"])
        source = verify_file_identity(source_root, contract, label=f"源数据 {relative}")
        target = project_path(target_root, relative)
        if target.exists():
            if not target.is_file():
                raise VersionLockError(f"目标路径已存在但不是文件：{target}")
            actual = file_evidence(target, root=target_root)
            expected = {
                "path": normalize_relative_path(relative),
                "bytes": int(contract["bytes"]),
                "sha256": str(contract["sha256"]),
            }
            if actual != expected:
                raise VersionLockError(f"目标文件已存在且身份不同，禁止覆盖：{target}")
            reused_files += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        verify_file_identity(target_root, contract, label=f"目标数据 {relative}")
        copied_files += 1
        copied_bytes += int(contract["bytes"])
    return {
        "copied_file_count": copied_files,
        "reused_file_count": reused_files,
        "copied_bytes": copied_bytes,
    }


def validate_expected_replay_metrics(
    metrics: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    """核对 G0.1 任务卡要求的 31、1,102、1,279 与时代分布。"""

    g1 = metrics["g1"]
    mft = metrics["mft"]
    actual_scalars = {
        "b2_identifiable_event_count": int(g1["b2_identifiable_event_count"]),
        "b2_eligible_non_event_risk_day_count": int(
            g1["b2_eligible_non_event_risk_day_count"]
        ),
        "b2_common_sample_day_count": int(g1["b2_common_sample_day_count"]),
        "b2_view_day_count": int(mft["b2_view_day_count"]),
    }
    expected_scalars = {
        key: int(expected[key])
        for key in (
            "b2_identifiable_event_count",
            "b2_eligible_non_event_risk_day_count",
            "b2_common_sample_day_count",
            "b2_view_day_count",
        )
    }
    if actual_scalars != expected_scalars:
        raise VersionLockError(
            f"G0.1 关键指标漂移：actual={actual_scalars}, expected={expected_scalars}"
        )
    actual_eras = metrics["event_era_distribution"]
    if canonical_sha256(actual_eras) != canonical_sha256(expected["event_era_distribution"]):
        raise VersionLockError("G0.1 年代事件分布漂移")


def atomic_write_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    """以独占创建方式写入 JSON，禁止覆盖既有证据。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0)
    try:
        descriptor = os.open(path, flags)
    except FileExistsError as exc:
        raise VersionLockError(f"输出已存在，禁止覆盖：{path}") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
