"""校验510300方向切换V1的一次性研究终态证据链。"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_RELATIVE_PATH = "config/510300_direction_switch_v1_manifest.json"
RESULT_RELATIVE_PATH = "reports/research/510300_direction_switch_v1_result.json"
TERMINAL_STATUS = "REJECTED_FROZEN_IF_FORCED_FLOW_MECHANISM_GATE_FAILED_NO_RESCUE"


def sha256_file(path: Path) -> str:
    """以二进制口径计算文件SHA-256。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON顶层必须为对象：{path}")
    return payload


def _canonical_manifest_bytes(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _require_hash_registry(
    project_root: Path, registry: dict[str, Any], registry_name: str
) -> int:
    if not isinstance(registry, dict) or not registry:
        raise ValueError(f"{registry_name}为空或类型无效")
    for relative_path, expected_hash in registry.items():
        if not isinstance(relative_path, str) or not isinstance(expected_hash, str):
            raise ValueError(f"{registry_name}包含无效条目")
        path = project_root / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"{registry_name}文件不存在：{relative_path}")
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"{registry_name}哈希漂移：{relative_path}；"
                f"expected={expected_hash}；actual={actual_hash}"
            )
    return len(registry)


def verify_consumed_direction_switch_v1(
    project_root: Path | None = None,
) -> dict[str, Any]:
    """验证旧研究已消费、不可重跑且全部冻结证据仍闭环。"""

    root = (project_root or ROOT).resolve()
    manifest_path = root / MANIFEST_RELATIVE_PATH
    result_path = root / RESULT_RELATIVE_PATH
    manifest = _load_json(manifest_path)
    result = _load_json(result_path)

    if manifest.get("study_id") != "510300_DIRECTION_SWITCH_V1":
        raise ValueError("方向切换终态清单研究编号无效")
    if manifest.get("freeze_state") != "PREREGISTERED_BEFORE_NEW_OUTCOMES":
        raise ValueError("方向切换终态清单冻结状态无效")
    if manifest.get("new_outcomes_read_before_freeze") is not False:
        raise ValueError("方向切换终态清单未证明先冻结后读结果")
    if int(manifest.get("historical_run_limit", -1)) != 1:
        raise ValueError("方向切换历史运行上限不是1")
    if int(manifest.get("historical_runs_consumed", -1)) != 1:
        raise ValueError("方向切换唯一历史运行尚未完整消费")
    if not isinstance(manifest.get("first_completed_run_at"), str):
        raise ValueError("方向切换终态清单缺少首次完成时间")

    protocol_relative_path = manifest.get("protocol_file")
    if not isinstance(protocol_relative_path, str):
        raise ValueError("方向切换终态清单缺少协议路径")
    protocol_path = root / protocol_relative_path
    if sha256_file(protocol_path) != manifest.get("protocol_sha256"):
        raise ValueError("方向切换冻结协议哈希漂移")

    implementation_count = _require_hash_registry(
        root,
        manifest.get("implementation_sha256"),
        "方向切换冻结实现",
    )
    result_count = _require_hash_registry(
        root,
        manifest.get("result_sha256"),
        "方向切换终态结果",
    )

    if result.get("status") != TERMINAL_STATUS:
        raise ValueError("方向切换终态不是冻结拒绝状态")
    if int(result.get("historical_run_count", -1)) != 1:
        raise ValueError("方向切换总报告未记录唯一历史运行")

    pre_run_manifest = copy.deepcopy(manifest)
    pre_run_manifest["historical_runs_consumed"] = 0
    pre_run_manifest.pop("first_completed_run_at", None)
    pre_run_manifest.pop("result_sha256", None)
    reconstructed_pre_run_hash = hashlib.sha256(
        _canonical_manifest_bytes(pre_run_manifest)
    ).hexdigest()
    recorded_pre_run_hash = result.get("pre_run_freeze_manifest_sha256")
    if reconstructed_pre_run_hash != recorded_pre_run_hash:
        raise ValueError(
            "方向切换预运行清单哈希无法从终态无损重建；"
            f"recorded={recorded_pre_run_hash}；"
            f"reconstructed={reconstructed_pre_run_hash}"
        )

    return {
        "status": "VERIFIED_TERMINAL_ONE_SHOT_CONSUMED",
        "study_id": manifest["study_id"],
        "terminal_result_status": result["status"],
        "historical_run_limit": 1,
        "historical_runs_consumed": 1,
        "implementation_hash_count": implementation_count,
        "result_hash_count": result_count,
        "pre_run_manifest_sha256": reconstructed_pre_run_hash,
        "rerun_authorized": False,
    }
