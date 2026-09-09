"""在首次读取BAD20历史结果前冻结X1实现与全部输入。"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_FILE = ROOT / "config" / "x1_common_tail_prediction_v1.yaml"
MANIFEST_FILE = ROOT / "config" / "x1_common_tail_prediction_v1_manifest.json"
PARENT_MANIFEST_FILE = ROOT / "config" / "return_tail_gating_manifest.json"
FROZEN_FILES = (
    "docs/X1_COMMON_TAIL_PREDICTION_V1_SPEC.md",
    "config/x1_common_tail_prediction_v1.yaml",
    "research/x1_common_tail_prediction.py",
    "scripts/audit_x1_common_tail_signal_v1.py",
    "scripts/freeze_x1_common_tail_prediction_v1.py",
    "scripts/run_x1_common_tail_prediction_v1.py",
    "tests/test_x1_common_tail_prediction.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(directory: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(path for path in directory.glob("*.parquet") if path.is_file())
    for path in files:
        digest.update(path.name.encode("utf-8"))
        digest.update(sha256(path).encode("ascii"))
    return digest.hexdigest()


def _git_value(*args: str) -> str | None:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
    )
    value = result.stdout.strip()
    return value if result.returncode == 0 and value else None


def _validate_contract(contract: dict) -> None:
    protocol = contract["protocol"]
    probability = contract["probability"]
    governance = contract["governance"]
    if protocol["state"] != "DISCOVERY_ONLY":
        raise ValueError("X1只能处于DISCOVERY_ONLY")
    if protocol["candidate_id"] != "X1" or protocol["candidate_budget"] != 1:
        raise ValueError("X1实现候选必须严格等于1")
    if protocol["true_forward_start"] is not None:
        raise ValueError("历史评价前true_forward_start必须为空")
    if not protocol["candidate_addition_after_freeze_forbidden"]:
        raise ValueError("冻结后必须禁止追加候选")
    if not protocol["threshold_change_after_freeze_forbidden"]:
        raise ValueError("冻结后必须禁止修改阈值")
    if probability["hyperparameter_search_allowed"]:
        raise ValueError("X1不得搜索超参数")
    forbidden_true = {
        "historical_run_may_emit_forecast_eligible",
        "combination_model_build_allowed",
        "position_policy_build_allowed",
        "r6_candidate_reselection_allowed",
        "options_branch_recovery_allowed",
        "earnings_consensus_branch_recovery_allowed",
        "position_mapping_enabled",
        "order_generation_enabled",
        "broker_connection_enabled",
        "live_trading_authorized",
    }
    enabled = sorted(key for key in forbidden_true if governance.get(key) is True)
    if enabled:
        raise ValueError(f"X1安全开关不得启用：{enabled}")


def _parent_integrity(contract: dict) -> dict:
    parent = json.loads(PARENT_MANIFEST_FILE.read_text(encoding="utf-8"))
    registry_relative = contract["inputs"]["parent_registry"]
    spec_relative = contract["inputs"]["parent_spec"]
    for relative in (registry_relative, spec_relative):
        if sha256(ROOT / relative) != parent["frozen_files"][relative]:
            raise RuntimeError(f"X1核心父协议哈希变化：{relative}")
    drift = []
    for relative, expected in parent["frozen_files"].items():
        path = ROOT / relative
        if not path.exists() or sha256(path) != expected:
            drift.append(relative)
    return {
        "parent_manifest_sha256": sha256(PARENT_MANIFEST_FILE),
        "core_registry_and_spec_match": True,
        "full_parent_manifest_drift": drift,
    }


def main() -> int:
    contract = yaml.safe_load(CONTRACT_FILE.read_text(encoding="utf-8"))
    _validate_contract(contract)
    parent_integrity = _parent_integrity(contract)
    missing = [relative for relative in FROZEN_FILES if not (ROOT / relative).exists()]
    if missing:
        raise FileNotFoundError(f"X1冻结文件缺失：{missing}")

    inputs = contract["inputs"]
    directory_relative = inputs["component_history_cache"]
    file_inputs = {
        key: value for key, value in inputs.items() if key != "component_history_cache"
    }
    missing_inputs = [
        value for value in file_inputs.values() if not (ROOT / value).exists()
    ]
    if missing_inputs:
        raise FileNotFoundError(f"X1冻结输入缺失：{missing_inputs}")
    history_directory = ROOT / directory_relative
    history_files = sorted(history_directory.glob("*.parquet"))
    if len(history_files) != 493:
        raise RuntimeError("X1完整个股历史缓存必须恰有493个文件")

    feasibility = json.loads(
        (ROOT / inputs["current_feasibility_status"]).read_text(encoding="utf-8")
    )
    if feasibility.get("branch_status", {}).get("CONSTITUENTS") != "PASS":
        raise RuntimeError("当前成分股分支不是PASS")
    if feasibility.get("hypotheses_allowed_for_return_test") != ["X1"]:
        raise RuntimeError("收益检验许可必须只有X1")
    signal_status = json.loads(
        (ROOT / inputs["x1_signal_status"]).read_text(encoding="utf-8")
    )
    if signal_status.get("status") != "READY_FOR_OUTCOME_FREEZE":
        raise RuntimeError("X1信号数据门尚未通过")
    if signal_status.get("future_return_read") is not False:
        raise RuntimeError("X1信号数据门已经读取未来收益")
    if signal_status.get("predictive_model_fitted") is not False:
        raise RuntimeError("X1信号数据门已经拟合预测模型")
    signal_relative = inputs["x1_signal_feature"]
    if (
        signal_status.get("output_hashes", {}).get(signal_relative)
        != sha256(ROOT / signal_relative)
    ):
        raise RuntimeError("X1信号文件与数据门哈希不一致")
    if (
        signal_status.get("input_hashes", {}).get(directory_relative)
        != tree_sha256(history_directory)
    ):
        raise RuntimeError("X1完整个股历史缓存与信号数据门哈希不一致")

    test = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_x1_common_tail_prediction.py", "-q"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if test.returncode != 0:
        raise RuntimeError("X1冻结测试失败：\n" + test.stdout + "\n" + test.stderr)
    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    input_hashes = {value: sha256(ROOT / value) for value in file_inputs.values()}
    input_hashes[directory_relative] = tree_sha256(history_directory)
    manifest = {
        "project_id": contract["protocol"]["project_id"],
        "version": contract["protocol"]["version"],
        "state": "FROZEN_BEFORE_HISTORICAL_OUTCOME_READ",
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "historical_contamination_cutoff": contract["protocol"][
            "historical_contamination_cutoff"
        ],
        "source_commit": _git_value("rev-parse", "HEAD"),
        "source_worktree_dirty": bool(status.stdout.strip()),
        "test_command": f"{sys.executable} -m pytest tests/test_x1_common_tail_prediction.py -q",
        "test_result": test.stdout.strip(),
        "frozen_files": {relative: sha256(ROOT / relative) for relative in FROZEN_FILES},
        "input_files": input_hashes,
        "component_history_file_count": len(history_files),
        "parent_integrity": parent_integrity,
        "signal_data_gate": {
            "status": signal_status["status"],
            "ready_signal_day_count": signal_status["ready_signal_day_count"],
            "first_ready_signal_date": signal_status["first_ready_signal_date"],
            "last_ready_signal_date": signal_status["last_ready_signal_date"],
            "future_return_read": signal_status["future_return_read"],
        },
        "governance": contract["governance"],
    }
    MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary = MANIFEST_FILE.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(MANIFEST_FILE)
    print(
        json.dumps(
            {
                "状态": manifest["state"],
                "候选": "X1",
                "有效信号日": signal_status["ready_signal_day_count"],
                "冻结实现文件": len(manifest["frozen_files"]),
                "冻结输入": len(manifest["input_files"]),
                "父协议核心哈希一致": True,
                "父清单已知漂移": parent_integrity["full_parent_manifest_drift"],
                "测试": manifest["test_result"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
