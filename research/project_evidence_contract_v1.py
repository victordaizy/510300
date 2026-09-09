"""项目级证据、权威状态、外部数据根与 Git 提交边界。"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import tempfile
from dataclasses import asdict, dataclass, is_dataclass
from datetime import date, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence


class EvidenceContractError(RuntimeError):
    """证据契约不满足时的封闭失败。"""


REQUIRED_AUTHORITY_FIELDS = {
    "research_state": "DISCOVERY_ONLY",
    "model_position_target": "UNSET",
    "order_authorization": "NOT_AUTHORIZED",
    "actual_holdings_state": "UNKNOWN_OUT_OF_SCOPE",
    "position_impact": 0,
}

PROHIBITED_AUTHORITY_KEYS = {
    "current_holding_route",
}


@dataclass(frozen=True)
class RegisteredFileEvidence:
    """已经通过物理根、身份、契约版本和内容哈希校验的文件。"""

    input_id: str
    logical_path: str
    physical_root_id: str
    resolved_path: str
    content_sha256: str
    bytes: int
    data_contract_version: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _json_ready(value: Any) -> Any:
    if is_dataclass(value):
        return _json_ready(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise EvidenceContractError("严格 JSON 禁止 NaN 或 Infinity")
        return value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if hasattr(value, "item"):
        return _json_ready(value.item())
    raise EvidenceContractError(f"严格 JSON 不支持类型：{type(value).__name__}")


def strict_json_text(payload: Any, *, pretty: bool = True) -> str:
    ready = _json_ready(payload)
    return json.dumps(
        ready,
        ensure_ascii=False,
        sort_keys=True,
        indent=2 if pretty else None,
        separators=None if pretty else (",", ":"),
        allow_nan=False,
    )


def canonical_sha256(payload: Any) -> str:
    return sha256_bytes(strict_json_text(payload, pretty=False).encode("utf-8"))


def read_json_strict(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise EvidenceContractError(f"严格 JSON 出现非法常量：{value}")

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle, parse_constant=reject_constant)


def _iter_mapping_keys(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key)
            yield from _iter_mapping_keys(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_mapping_keys(item)


def validate_authority_state(authority: Mapping[str, Any]) -> None:
    """验证研究、模型、订单、实持仓四层语义互不替代。"""

    missing = sorted(set(REQUIRED_AUTHORITY_FIELDS).difference(authority))
    if missing:
        raise EvidenceContractError(f"权威状态缺少字段：{missing}")
    mismatches = {
        key: {"expected": expected, "actual": authority.get(key)}
        for key, expected in REQUIRED_AUTHORITY_FIELDS.items()
        if authority.get(key) != expected
    }
    if mismatches:
        raise EvidenceContractError(f"权威状态越过研究边界：{mismatches}")
    present_keys = {key.casefold() for key in _iter_mapping_keys(authority)}
    prohibited = sorted(PROHIBITED_AUTHORITY_KEYS.intersection(present_keys))
    if prohibited:
        raise EvidenceContractError(f"权威状态仍使用被禁止字段：{prohibited}")
    for key in (
        "paper_signal_authorized",
        "shadow_signal_authorized",
        "broker_connection_authorized",
        "live_trading_authorized",
    ):
        if authority.get(key) is not False:
            raise EvidenceContractError(f"{key} 必须显式为 false")
    if authority.get("abstain_implies_cash_target") is not False:
        raise EvidenceContractError("ABSTAIN 不得被翻译为现金仓位")


def normalize_project_relative_path(value: str) -> str:
    normalized = value.replace("\\", "/")
    candidate = PurePosixPath(normalized)
    if candidate.is_absolute() or not candidate.parts:
        raise EvidenceContractError(f"逻辑路径必须是项目相对路径：{value}")
    if any(part in {"", ".", ".."} for part in candidate.parts):
        raise EvidenceContractError(f"逻辑路径含有不允许的路径段：{value}")
    if ":" in candidate.parts[0]:
        raise EvidenceContractError(f"逻辑路径不得包含盘符：{value}")
    return candidate.as_posix()


def _same_path(left: Path, right: Path) -> bool:
    return os.path.normcase(os.path.normpath(str(left))) == os.path.normcase(
        os.path.normpath(str(right))
    )


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def resolve_registered_input(
    *,
    project_root: Path,
    input_id: str,
    root_contract: Mapping[str, Any],
    input_contract: Mapping[str, Any],
) -> RegisteredFileEvidence:
    """按注册物理根校验文件，不再要求 Junction 目标位于项目物理目录内。"""

    project_root = project_root.resolve(strict=True)
    logical_root = normalize_project_relative_path(str(root_contract["logical_path"]))
    logical_path = normalize_project_relative_path(str(input_contract["logical_path"]))
    logical_root_parts = PurePosixPath(logical_root).parts
    logical_parts = PurePosixPath(logical_path).parts
    if logical_parts[: len(logical_root_parts)] != logical_root_parts:
        raise EvidenceContractError(
            f"{input_id} 的逻辑路径不在注册逻辑根 {logical_root} 下"
        )

    root_id = str(root_contract["physical_root_id"])
    if str(input_contract.get("physical_root_id")) != root_id:
        raise EvidenceContractError(f"{input_id} 的 physical_root_id 不一致")
    contract_version = str(input_contract.get("data_contract_version", "")).strip()
    if not contract_version:
        raise EvidenceContractError(f"{input_id} 缺少 data_contract_version")

    expected_root = Path(str(root_contract["expected_resolved_root"])).resolve(
        strict=True
    )
    actual_root = (project_root / Path(logical_root)).resolve(strict=True)
    if not _same_path(actual_root, expected_root):
        raise EvidenceContractError(
            f"逻辑根 {logical_root} 解析到 {actual_root}，不等于注册物理根 {expected_root}"
        )

    logical_file = project_root / Path(logical_path)
    resolved_file = logical_file.resolve(strict=True)
    if not resolved_file.is_file():
        raise EvidenceContractError(f"{input_id} 不是常规文件：{resolved_file}")
    if not _is_within(resolved_file, expected_root):
        raise EvidenceContractError(
            f"{input_id} 解析到注册物理根之外：{resolved_file}"
        )

    actual_bytes = resolved_file.stat().st_size
    expected_bytes = int(input_contract["bytes"])
    if actual_bytes != expected_bytes:
        raise EvidenceContractError(
            f"{input_id} 字节数不一致：expected={expected_bytes}, actual={actual_bytes}"
        )
    actual_sha256 = sha256_file(resolved_file)
    expected_sha256 = str(input_contract["content_sha256"]).lower()
    if actual_sha256 != expected_sha256:
        raise EvidenceContractError(
            f"{input_id} 内容哈希不一致：expected={expected_sha256}, actual={actual_sha256}"
        )

    return RegisteredFileEvidence(
        input_id=input_id,
        logical_path=logical_path,
        physical_root_id=root_id,
        resolved_path=str(resolved_file),
        content_sha256=actual_sha256,
        bytes=actual_bytes,
        data_contract_version=contract_version,
    )


def validate_registered_inputs(
    config: Mapping[str, Any], *, project_root: Path
) -> dict[str, RegisteredFileEvidence]:
    roots = config.get("data_roots")
    sources = config.get("source_contract")
    if not isinstance(roots, Mapping) or not isinstance(sources, Mapping):
        raise EvidenceContractError("配置缺少 data_roots 或 source_contract")
    evidence: dict[str, RegisteredFileEvidence] = {}
    for input_id, input_contract in sources.items():
        if not isinstance(input_contract, Mapping):
            raise EvidenceContractError(f"{input_id} 输入契约不是对象")
        root_name = str(input_contract.get("root_contract", ""))
        if root_name not in roots:
            raise EvidenceContractError(f"{input_id} 引用了未知数据根：{root_name}")
        root_contract = roots[root_name]
        if not isinstance(root_contract, Mapping):
            raise EvidenceContractError(f"数据根 {root_name} 不是对象")
        evidence[str(input_id)] = resolve_registered_input(
            project_root=project_root,
            input_id=str(input_id),
            root_contract=root_contract,
            input_contract=input_contract,
        )
    return evidence


def verify_manifest_payload(manifest: Mapping[str, Any]) -> None:
    expected = str(manifest.get("manifest_payload_sha256", ""))
    payload = dict(manifest)
    payload.pop("manifest_payload_sha256", None)
    actual = canonical_sha256(payload)
    if expected != actual:
        raise EvidenceContractError(
            f"manifest 自身哈希失败：expected={expected}, actual={actual}"
        )


def verify_manifest_files(
    manifest: Mapping[str, Any], *, project_root: Path
) -> None:
    verify_manifest_payload(manifest)
    file_groups = (
        "implementation_files",
        "governance_files",
    )
    for group_name in file_groups:
        group = manifest.get(group_name)
        if not isinstance(group, Mapping):
            raise EvidenceContractError(f"manifest 缺少 {group_name}")
        for relative, expected_sha256 in group.items():
            normalized = normalize_project_relative_path(str(relative))
            path = project_root / Path(normalized)
            if not path.is_file():
                raise EvidenceContractError(f"manifest 文件缺失：{normalized}")
            actual_sha256 = sha256_file(path)
            if actual_sha256 != str(expected_sha256):
                raise EvidenceContractError(
                    f"manifest 文件漂移：{normalized}, expected={expected_sha256}, actual={actual_sha256}"
                )


def _run_git(
    project_root: Path,
    arguments: Sequence[str],
    *,
    binary: bool = False,
) -> bytes | str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=not binary,
        encoding=None if binary else "utf-8",
        errors=None if binary else "replace",
    )
    if completed.returncode != 0:
        stderr = (
            completed.stderr.decode("utf-8", errors="replace")
            if binary
            else completed.stderr
        )
        raise EvidenceContractError(
            f"Git 命令失败：git {' '.join(arguments)}；{stderr.strip()}"
        )
    return completed.stdout


def git_head(project_root: Path) -> str:
    return str(_run_git(project_root, ["rev-parse", "HEAD"])).strip()


def ensure_paths_committed_at_head(
    project_root: Path, relative_paths: Iterable[str]
) -> dict[str, dict[str, Any]]:
    """确保每个冻结文件已进入 HEAD，且 Git 认为工作区没有漂移。"""

    evidence: dict[str, dict[str, Any]] = {}
    for value in sorted(set(relative_paths)):
        relative = normalize_project_relative_path(value)
        _run_git(project_root, ["cat-file", "-e", f"HEAD:{relative}"])
        head_bytes = _run_git(
            project_root, ["show", f"HEAD:{relative}"], binary=True
        )
        if not isinstance(head_bytes, bytes):
            raise EvidenceContractError(f"读取 HEAD 文件失败：{relative}")
        diff = subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--", relative],
            cwd=project_root,
            check=False,
            capture_output=True,
        )
        if diff.returncode == 1:
            raise EvidenceContractError(
                f"工作区文件未提交或提交后漂移：{relative}"
            )
        if diff.returncode != 0:
            raise EvidenceContractError(
                f"Git 无法校验工作区文件：{relative}；"
                f"{diff.stderr.decode('utf-8', errors='replace').strip()}"
            )
        working_path = project_root / Path(relative)
        if not working_path.is_file():
            raise EvidenceContractError(f"工作区文件缺失：{relative}")
        working_sha256 = sha256_file(working_path)
        head_sha256 = sha256_bytes(head_bytes)
        evidence[relative] = {
            "head_blob_sha256": head_sha256,
            "working_sha256": working_sha256,
            "bytes": working_path.stat().st_size,
        }
    return evidence


def atomic_write_bytes_new(path: Path, payload: bytes) -> None:
    """只允许创建新文件；临时文件与最终文件位于同一目录。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise EvidenceContractError(f"输出已存在，禁止覆盖：{path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise EvidenceContractError(f"并发创建冲突，禁止覆盖：{path}")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_text_new(path: Path, text: str) -> None:
    atomic_write_bytes_new(path, text.encode("utf-8"))


def atomic_write_json_new(path: Path, payload: Any) -> None:
    atomic_write_text_new(path, strict_json_text(payload, pretty=True) + "\n")


def create_exclusive_claim(path: Path, payload: Mapping[str, Any]) -> None:
    """使用 O_EXCL 创建一次性 claim；已存在时绝不重写。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (strict_json_text(payload, pretty=True) + "\n").encode("utf-8")
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0)
    try:
        descriptor = os.open(path, flags)
    except FileExistsError as exc:
        raise EvidenceContractError(f"一次性 claim 已存在：{path}") from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        # claim 一旦创建即代表尝试已发生；异常时也不得删除或重用。
        raise


def file_evidence(path: Path, *, project_root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(project_root).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }
