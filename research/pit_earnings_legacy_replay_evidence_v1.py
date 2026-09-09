"""校验PIT盈利扩散V1.0.2旧重放失败及V1.0.3替代契约。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LEGACY_MANIFEST_RELATIVE_PATH = (
    "config/510300_pit_earnings_information_diffusion_10d_risk_v1_manifest.json"
)
LEGACY_REPLAY_STATUS_RELATIVE_PATH = (
    "reports/audit/"
    "510300_pit_earnings_information_diffusion_v1_0_2_legacy_replay_status.json"
)
EXPECTED_MUTABLE_PATH = (
    "data/raw/cninfo/"
    "a_share_hs_first_preliminary_core_earnings_pdf_inventory_v1.parquet"
)
EXPECTED_LEGACY_HASH = (
    "49465b1190f813d7e179ee102d9f566f67a13c25a53f242ef194ed729f3e3e95"
)
EXPECTED_CURRENT_HASH = (
    "632b7fbca399945681b831435b759409de38f55d53eaeb3541aad144294f6fbc"
)


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


def _validate_entry(root: Path, item: dict[str, Any], label: str) -> None:
    relative_path = item.get("path")
    expected_hash = item.get("sha256")
    expected_bytes = item.get("bytes")
    if not isinstance(relative_path, str) or not isinstance(expected_hash, str):
        raise ValueError(f"{label}条目无效")
    path = root / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"{label}文件不存在：{relative_path}")
    actual_bytes = path.stat().st_size
    if expected_bytes is not None and actual_bytes != int(expected_bytes):
        raise ValueError(
            f"{label}字节数漂移：{relative_path}；"
            f"expected={expected_bytes}；actual={actual_bytes}"
        )
    actual_hash = sha256_file(path)
    if actual_hash != expected_hash:
        raise ValueError(
            f"{label}哈希漂移：{relative_path}；"
            f"expected={expected_hash}；actual={actual_hash}"
        )


def verify_pit_earnings_legacy_replay_supersession(
    project_root: Path | None = None,
) -> dict[str, Any]:
    """验证旧路径漂移被如实保留，且替代契约完整、结果读取仍被禁止。"""

    root = (project_root or ROOT).resolve()
    legacy_manifest = _load_json(root / LEGACY_MANIFEST_RELATIVE_PATH)
    replay_status = _load_json(root / LEGACY_REPLAY_STATUS_RELATIVE_PATH)

    expected_legacy_status = (
        "FROZEN_OUTCOME_BLIND_PIT_EARNINGS_DATA_FEASIBILITY_"
        "BEFORE_ANY_POST_EVENT_OUTCOME"
    )
    if legacy_manifest.get("status") != expected_legacy_status:
        raise ValueError("PIT盈利扩散旧清单状态无效")

    legacy_file_entries = legacy_manifest.get("files")
    legacy_input_entries = legacy_manifest.get("fixed_input_evidence")
    if not isinstance(legacy_file_entries, list) or not legacy_file_entries:
        raise ValueError("PIT盈利扩散旧清单缺少冻结文件")
    if not isinstance(legacy_input_entries, list) or not legacy_input_entries:
        raise ValueError("PIT盈利扩散旧清单缺少冻结输入")
    for item in legacy_file_entries:
        _validate_entry(root, item, "PIT盈利扩散旧冻结文件")

    observed_mismatches: list[dict[str, Any]] = []
    matched_legacy_input_count = 0
    for item in legacy_input_entries:
        relative_path = item.get("path")
        path = root / str(relative_path)
        if not path.is_file():
            observed_mismatches.append(
                {"path": relative_path, "state": "MISSING"}
            )
            continue
        actual_hash = sha256_file(path)
        actual_bytes = path.stat().st_size
        if actual_hash != item.get("sha256") or actual_bytes != int(item.get("bytes")):
            observed_mismatches.append(
                {
                    "path": relative_path,
                    "state": "DRIFT",
                    "expected_sha256": item.get("sha256"),
                    "actual_sha256": actual_hash,
                    "expected_bytes": int(item.get("bytes")),
                    "actual_bytes": actual_bytes,
                }
            )
        else:
            matched_legacy_input_count += 1

    expected_mismatch = {
        "path": EXPECTED_MUTABLE_PATH,
        "state": "DRIFT",
        "expected_sha256": EXPECTED_LEGACY_HASH,
        "actual_sha256": EXPECTED_CURRENT_HASH,
        "expected_bytes": 7807930,
        "actual_bytes": 7807560,
    }
    if observed_mismatches != [expected_mismatch]:
        raise ValueError(
            "PIT盈利扩散旧清单漂移集合与正式裁定不一致："
            f"{observed_mismatches!r}"
        )

    if replay_status.get("status") != (
        "EXPECTED_LEGACY_REPLAY_FAILED_UPSTREAM_MUTABLE_PATH_OVERWRITTEN"
    ):
        raise ValueError("PIT盈利扩散旧重放状态未正式关闭")
    legacy_record = replay_status.get("legacy_manifest", {})
    current_record = replay_status.get("current_same_path", {})
    if legacy_record.get("frozen_pdf_inventory_path") != EXPECTED_MUTABLE_PATH:
        raise ValueError("旧重放裁定中的冻结路径不一致")
    if legacy_record.get("expected_sha256") != EXPECTED_LEGACY_HASH:
        raise ValueError("旧重放裁定中的预期哈希不一致")
    if int(legacy_record.get("expected_bytes", -1)) != 7807930:
        raise ValueError("旧重放裁定中的预期字节数不一致")
    if current_record.get("sha256") != EXPECTED_CURRENT_HASH:
        raise ValueError("旧重放裁定中的当前哈希不一致")
    if int(current_record.get("bytes", -1)) != 7807560:
        raise ValueError("旧重放裁定中的当前字节数不一致")

    interpretation = replay_status.get("interpretation", {})
    required_false_flags = (
        "old_hash_silently_updated",
        "old_bytes_reconstructed_or_fabricated",
        "formal_candidate_status_changed",
        "portfolio_evaluation_authorized",
        "trading_authorized",
    )
    if any(interpretation.get(name) is not False for name in required_false_flags):
        raise ValueError("旧重放裁定越过了禁止救援或交易边界")

    superseding_record = replay_status.get("superseding_current_data_contract", {})
    superseding_relative_path = superseding_record.get("path")
    if not isinstance(superseding_relative_path, str):
        raise ValueError("旧重放裁定缺少替代契约路径")
    superseding_path = root / superseding_relative_path
    if sha256_file(superseding_path) != superseding_record.get("sha256"):
        raise ValueError("PIT盈利扩散替代契约哈希漂移")
    superseding_manifest = _load_json(superseding_path)
    if superseding_manifest.get("status") != (
        "FROZEN_OUTCOME_BLIND_SOURCE_CORRECTION_BEFORE_OUTPUT"
    ):
        raise ValueError("PIT盈利扩散替代契约状态无效")
    if superseding_manifest.get("market_price_read") is not False:
        raise ValueError("PIT盈利扩散替代契约读取了市场价格")
    if superseding_manifest.get("future_return_read") is not False:
        raise ValueError("PIT盈利扩散替代契约读取了未来收益")
    if superseding_manifest.get("return_evaluation") != "NOT_ALLOWED":
        raise ValueError("PIT盈利扩散替代契约错误授权收益评估")

    superseding_file_entries = superseding_manifest.get("frozen_files")
    superseding_input_entries = superseding_manifest.get("frozen_inputs")
    if not isinstance(superseding_file_entries, list):
        raise ValueError("PIT盈利扩散替代契约缺少冻结文件")
    if not isinstance(superseding_input_entries, list):
        raise ValueError("PIT盈利扩散替代契约缺少冻结输入")
    for item in superseding_file_entries:
        _validate_entry(root, item, "PIT盈利扩散替代冻结文件")
    for item in superseding_input_entries:
        _validate_entry(root, item, "PIT盈利扩散替代冻结输入")

    correction = superseding_manifest.get("post_output_serialization_correction")
    if not isinstance(correction, dict):
        raise ValueError("PIT盈利扩散替代契约缺少严格JSON修正记录")
    _validate_entry(root, correction, "PIT盈利扩散严格JSON修正")
    if correction.get("data_or_gate_logic_changed") is not False:
        raise ValueError("严格JSON修正不得改变数据或门槛逻辑")

    expected_outputs = superseding_manifest.get("expected_outputs")
    if not isinstance(expected_outputs, list) or not expected_outputs:
        raise ValueError("PIT盈利扩散替代契约缺少预期输出")
    for relative_path in expected_outputs:
        if not isinstance(relative_path, str) or not (root / relative_path).is_file():
            raise FileNotFoundError(f"PIT盈利扩散替代输出不存在：{relative_path}")

    return {
        "status": "VERIFIED_LEGACY_REPLAY_FAILURE_SUPERSEDED",
        "legacy_replay_status": replay_status["status"],
        "legacy_frozen_file_count": len(legacy_file_entries),
        "legacy_matched_input_count": matched_legacy_input_count,
        "legacy_expected_drift_count": len(observed_mismatches),
        "superseding_frozen_file_count": len(superseding_file_entries),
        "superseding_frozen_input_count": len(superseding_input_entries),
        "superseding_output_count": len(expected_outputs),
        "market_price_read": False,
        "future_return_read": False,
        "return_evaluation": "NOT_ALLOWED",
        "portfolio_evaluation_authorized": False,
        "trading_authorized": False,
    }
