"""行业预期差冻结输入的可审计恢复与替代验证。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from research.industry_expectation_gap_forward_evaluation import ForwardEvaluationError


class FrozenInputRecoveryError(ForwardEvaluationError):
    """冻结输入恢复证据不完整或不一致。"""


def validate_recovery_config(config: Mapping[str, Any]) -> None:
    """确认恢复附录只开放精确身份恢复，不改变规则或交易授权。"""

    if config.get("status") != "RECOVERY_ADDENDUM_AFTER_FIRST_FORWARD_OBSERVATION":
        raise FrozenInputRecoveryError("恢复附录状态非法")
    if config.get("purpose") != "EXACT_FROZEN_INPUT_RECOVERY_ONLY":
        raise FrozenInputRecoveryError("恢复附录用途必须仅为精确冻结输入恢复")
    constraints = config.get("constraints")
    if not isinstance(constraints, Mapping):
        raise FrozenInputRecoveryError("恢复附录缺少 constraints")
    required_true = {
        "parent_rules_must_remain_exact",
        "original_prediction_must_remain_exact",
        "recovered_file_may_only_satisfy_original_manifest_verification",
        "recovered_file_may_not_replace_current_operational_readiness",
    }
    required_false = {
        "partial_forward_return_read",
        "prediction_change",
        "rule_change",
        "threshold_change",
    }
    invalid_true = [name for name in required_true if constraints.get(name) is not True]
    invalid_false = [name for name in required_false if constraints.get(name) is not False]
    if invalid_true or invalid_false:
        raise FrozenInputRecoveryError(
            f"恢复约束非法：true={sorted(invalid_true)}, false={sorted(invalid_false)}"
        )
    safety = config.get("safety")
    if not isinstance(safety, Mapping) or safety.get("research_only") is not True:
        raise FrozenInputRecoveryError("恢复附录必须保持 research_only")
    forbidden = {
        "may_upgrade_original_no_view",
        "position_mapping_enabled",
        "order_generation_enabled",
        "broker_connection_enabled",
        "live_trading_enabled",
    }
    unsafe = [name for name in forbidden if safety.get(name) is not False]
    if unsafe:
        raise FrozenInputRecoveryError(f"恢复附录安全边界非法：{sorted(unsafe)}")
    recoveries = config.get("recoveries")
    if not isinstance(recoveries, list) or len(recoveries) != 1:
        raise FrozenInputRecoveryError("本附录只允许一个已知恢复目标")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def workspace_path(workspace_root: Path, relative_path: str) -> Path:
    root = workspace_root.resolve()
    path = (root / relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise FrozenInputRecoveryError(f"路径越出工作区：{relative_path}") from exc
    return path


def load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FrozenInputRecoveryError(f"无法读取 JSON：{path}") from exc
    if not isinstance(value, dict):
        raise FrozenInputRecoveryError(f"JSON 顶层必须是对象：{path}")
    return value


def reconstruct_json_bytes_from_log(
    log_path: Path,
    *,
    marker: str,
    newline: str,
) -> bytes:
    """从采集日志最后一次匹配位置恢复完整 JSON，并按原换行编码。"""

    if newline not in {"LF", "CRLF"}:
        raise FrozenInputRecoveryError(f"不支持的换行格式：{newline}")
    try:
        text = log_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise FrozenInputRecoveryError(f"无法读取恢复来源日志：{log_path}") from exc
    marker_position = text.rfind(marker)
    if marker_position < 0:
        raise FrozenInputRecoveryError("恢复来源日志中不存在唯一指定标记")
    object_start = text.rfind("{", 0, marker_position)
    if object_start < 0:
        raise FrozenInputRecoveryError("恢复来源日志中找不到 JSON 起点")
    try:
        payload, _ = json.JSONDecoder().raw_decode(text[object_start:])
    except json.JSONDecodeError as exc:
        raise FrozenInputRecoveryError("恢复来源日志中的 JSON 无法解析") from exc
    if not isinstance(payload, dict):
        raise FrozenInputRecoveryError("恢复来源日志中的 JSON 顶层不是对象")
    serialized = json.dumps(payload, ensure_ascii=False, indent=2)
    if newline == "CRLF":
        serialized = serialized.replace("\n", "\r\n")
    return serialized.encode("utf-8")


def validate_recovered_readiness_semantics(
    recovered: Mapping[str, Any],
    frozen_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    """核对恢复报告与冻结 ETF 快照内嵌的就绪语义。"""

    snapshot = frozen_snapshot.get("forward_readiness")
    if not isinstance(snapshot, Mapping):
        raise FrozenInputRecoveryError("冻结 ETF 快照缺少 forward_readiness")
    pairs = {
        "status": (recovered.get("status"), snapshot.get("status")),
        "observed_trading_days": (
            recovered.get("observed_trading_days"),
            snapshot.get("observed_trading_days"),
        ),
        "full_coverage_days": (
            recovered.get("full_coverage_days"),
            snapshot.get("full_coverage_days"),
        ),
        "minimum_full_coverage_days": (
            recovered.get("minimum_full_coverage_days"),
            snapshot.get("minimum_required_days"),
        ),
        "recommended_full_coverage_days": (
            recovered.get("recommended_full_coverage_days"),
            snapshot.get("recommended_days"),
        ),
        "eligible_for_research_evaluation": (
            recovered.get("eligible_for_research_evaluation"),
            snapshot.get("eligible_for_research_evaluation"),
        ),
        "eligible_for_position_mapping": (
            recovered.get("eligible_for_position_mapping"),
            snapshot.get("eligible_for_position_mapping"),
        ),
    }
    mismatches = {
        name: {"recovered": values[0], "snapshot": values[1]}
        for name, values in pairs.items()
        if values[0] != values[1]
    }
    if mismatches:
        raise FrozenInputRecoveryError(f"恢复语义与冻结快照不一致：{mismatches}")
    if recovered.get("eligible_for_research_evaluation") is not False:
        raise FrozenInputRecoveryError("恢复报告不得授权研究评价")
    if recovered.get("eligible_for_position_mapping") is not False:
        raise FrozenInputRecoveryError("恢复报告不得授权仓位映射")
    return {name: values[0] for name, values in pairs.items()}


def verify_manifest_with_declared_recoveries(
    workspace_root: Path,
    *,
    manifest_relative_path: str,
    expected_manifest_sha256: str,
    recoveries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """验证原清单；只允许由精确恢复副本覆盖已声明的单点漂移。"""

    manifest_path = workspace_path(workspace_root, manifest_relative_path)
    if not manifest_path.is_file():
        raise FrozenInputRecoveryError(f"原冻结清单不存在：{manifest_relative_path}")
    manifest_hash = file_sha256(manifest_path)
    if manifest_hash.lower() != expected_manifest_sha256.lower():
        raise FrozenInputRecoveryError("原预测冻结清单哈希发生变化")
    manifest = load_json_object(manifest_path)
    recovery_by_target = {str(item["target_path"]): item for item in recoveries}
    if len(recovery_by_target) != len(recoveries):
        raise FrozenInputRecoveryError("恢复声明包含重复 target_path")

    records: list[dict[str, Any]] = []
    used_targets: set[str] = set()
    for frozen_item in manifest.get("frozen_files", []):
        relative = str(frozen_item["path"])
        expected_hash = str(frozen_item["sha256"]).lower()
        expected_bytes = int(frozen_item["bytes"])
        current_path = workspace_path(workspace_root, relative)
        current_hash = file_sha256(current_path) if current_path.is_file() else None
        if current_hash == expected_hash:
            records.append(
                {
                    "path": relative,
                    "verification": "CURRENT_FILE_EXACT",
                    "sha256": current_hash,
                    "bytes": current_path.stat().st_size,
                }
            )
            continue

        recovery = recovery_by_target.get(relative)
        if recovery is None:
            reason = "缺失" if current_hash is None else f"当前哈希 {current_hash}"
            raise FrozenInputRecoveryError(f"存在未声明的冻结文件漂移：{relative}（{reason}）")
        if str(recovery.get("expected_sha256", "")).lower() != expected_hash:
            raise FrozenInputRecoveryError(f"恢复声明哈希不等于原清单：{relative}")
        if int(recovery.get("expected_bytes", -1)) != expected_bytes:
            raise FrozenInputRecoveryError(f"恢复声明字节数不等于原清单：{relative}")

        recovered_path = workspace_path(workspace_root, str(recovery["recovered_path"]))
        if not recovered_path.is_file():
            raise FrozenInputRecoveryError(f"恢复副本不存在：{recovery['recovered_path']}")
        recovered_bytes = recovered_path.read_bytes()
        recovered_hash = hashlib.sha256(recovered_bytes).hexdigest()
        if recovered_hash != expected_hash or len(recovered_bytes) != expected_bytes:
            raise FrozenInputRecoveryError(f"恢复副本不能逐字节匹配原清单：{relative}")

        source_log_path = workspace_path(workspace_root, str(recovery["source_log_path"]))
        source_log_hash = file_sha256(source_log_path)
        if source_log_hash != str(recovery["source_log_sha256"]).lower():
            raise FrozenInputRecoveryError(f"恢复来源日志哈希不一致：{relative}")
        reconstructed = reconstruct_json_bytes_from_log(
            source_log_path,
            marker=str(recovery["source_marker"]),
            newline=str(recovery["newline"]),
        )
        if reconstructed != recovered_bytes:
            raise FrozenInputRecoveryError(f"恢复副本无法由声明日志确定性重建：{relative}")

        snapshot_path = workspace_path(
            workspace_root, str(recovery["frozen_semantic_snapshot_path"])
        )
        snapshot_hash = file_sha256(snapshot_path)
        if snapshot_hash != str(recovery["frozen_semantic_snapshot_sha256"]).lower():
            raise FrozenInputRecoveryError(f"冻结语义快照哈希不一致：{relative}")
        semantic_projection = validate_recovered_readiness_semantics(
            json.loads(recovered_bytes.decode("utf-8")),
            load_json_object(snapshot_path),
        )
        used_targets.add(relative)
        records.append(
            {
                "path": relative,
                "verification": "RECOVERED_FILE_EXACT",
                "current_sha256": current_hash,
                "recovered_path": str(recovery["recovered_path"]),
                "sha256": recovered_hash,
                "bytes": len(recovered_bytes),
                "source_log_sha256": source_log_hash,
                "semantic_projection": semantic_projection,
            }
        )

    undeclared_in_manifest = set(recovery_by_target) - {
        str(item["path"]) for item in manifest.get("frozen_files", [])
    }
    if undeclared_in_manifest:
        raise FrozenInputRecoveryError(
            f"恢复目标不在原清单中：{sorted(undeclared_in_manifest)}"
        )
    return {
        "manifest_path": manifest_relative_path,
        "manifest_sha256": manifest_hash,
        "frozen_file_count": len(records),
        "recovered_target_count": len(used_targets),
        "records": records,
    }
