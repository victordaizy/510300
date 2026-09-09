"""冻结或验证市场状态感知策略配置器 V1。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.regime_aware_strategy_allocator_v1 import CONFIG, load_contract, sha256_file


MANIFEST = ROOT / "config" / "regime_aware_strategy_allocator_v1_manifest.json"
TRACKED_FILES = (
    "docs/REGIME_AWARE_STRATEGY_ALLOCATOR_V1_SPEC.md",
    "config/regime_aware_strategy_allocator_v1.yaml",
    "data/research/regime_aware_strategy_allocator_v1/README.md",
    "research/regime_aware_strategy_allocator_v1.py",
    "research/regime_aware_strategy_forward_evidence_v1.py",
    "scripts/freeze_regime_aware_strategy_allocator_v1.py",
    "scripts/build_regime_aware_strategy_forward_evidence_v1.py",
    "scripts/run_regime_aware_strategy_allocator_v1.py",
    "tests/test_regime_aware_strategy_allocator_v1.py",
    "tests/test_regime_aware_strategy_forward_evidence_v1.py",
)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """同目录临时文件写入后原子替换。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _git_value(*args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _file_record(relative: str) -> dict[str, Any]:
    path = ROOT / PurePosixPath(relative)
    if not path.is_file():
        raise FileNotFoundError(f"V1 冻结文件缺失：{relative}")
    return {
        "path": relative,
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def current_tracked_hashes() -> dict[str, str]:
    return {relative: _file_record(relative)["sha256"] for relative in TRACKED_FILES}


def tracked_content_sha256(files: dict[str, str]) -> str:
    canonical = "\n".join(f"{path}:{files[path]}" for path in sorted(files))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def dependency_paths(contract: dict[str, Any]) -> tuple[str, ...]:
    paths: list[str] = []
    for strategy in contract["strategy_pool"]:
        paths.extend(
            [
                strategy["component_manifest"],
                strategy["component_status_source"],
            ]
        )
    if len(paths) != len(set(paths)):
        raise ValueError("组件依赖路径重复")
    return tuple(paths)


def current_dependency_snapshot(contract: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {relative: _file_record(relative) for relative in dependency_paths(contract)}


def current_component_status_snapshot(contract: dict[str, Any]) -> list[dict[str, Any]]:
    """冻结组件身份、终态和授权；不读取收益用于排序。"""

    result: list[dict[str, Any]] = []
    for strategy in contract["strategy_pool"]:
        path = ROOT / PurePosixPath(strategy["component_status_source"])
        payload = json.loads(path.read_text(encoding="utf-8"))
        status = payload.get("status")
        decision = payload.get("decision") if isinstance(payload.get("decision"), dict) else {}
        authorized = decision.get(
            "paper_or_live_authorized",
            payload.get("paper_or_live_authorized", False),
        )
        if not isinstance(status, str) or not status:
            raise ValueError(f"组件状态文件缺少 status：{strategy['strategy_id']}")
        if authorized is not False:
            raise ValueError(f"组件不得具有 Paper 或实盘授权：{strategy['strategy_id']}")
        result.append(
            {
                "strategy_id": strategy["strategy_id"],
                "status": status,
                "paper_or_live_authorized": False,
                "status_relabelled": False,
                "performance_used_for_ranking": False,
            }
        )
    return result


def _assert_prefreeze_outputs_and_evidence_absent(contract: dict[str, Any]) -> dict[str, Any]:
    outputs = contract["outputs"]
    forbidden_outputs = [
        outputs["latest_status_json"],
        outputs["latest_status_markdown"],
        contract["forward_evaluation"]["latest_build_status_json"],
    ]
    existing_outputs = [relative for relative in forbidden_outputs if (ROOT / PurePosixPath(relative)).exists()]
    receipt_dir = ROOT / PurePosixPath(outputs["immutable_receipt_directory"])
    if receipt_dir.exists() and any(receipt_dir.iterdir()):
        existing_outputs.append(outputs["immutable_receipt_directory"])
    evidence_build_receipt_dir = ROOT / PurePosixPath(
        contract["forward_evaluation"]["immutable_build_receipt_directory"]
    )
    if evidence_build_receipt_dir.exists() and any(evidence_build_receipt_dir.iterdir()):
        existing_outputs.append(
            contract["forward_evaluation"]["immutable_build_receipt_directory"]
        )
    evidence_report_dir = ROOT / PurePosixPath(
        contract["forward_evaluation"]["immutable_report_directory"]
    )
    if evidence_report_dir.exists() and any(evidence_report_dir.iterdir()):
        existing_outputs.append(
            contract["forward_evaluation"]["immutable_report_directory"]
        )
    if existing_outputs:
        raise RuntimeError(f"V1 冻结前已经存在运行输出：{existing_outputs}")

    evidence_paths = [
        contract["inputs"]["environment_evidence_jsonl"],
        contract["inputs"]["strategy_evidence_jsonl"],
        contract["inputs"]["forward_observation_ledger_jsonl"],
    ]
    nonempty_evidence: list[str] = []
    for relative in evidence_paths:
        path = ROOT / PurePosixPath(relative)
        if path.is_file() and path.stat().st_size > 0:
            nonempty_evidence.append(relative)
    if nonempty_evidence:
        raise RuntimeError(f"V1 冻结前已经存在资格证据，禁止事后冻结：{nonempty_evidence}")
    return {
        "environment_evidence_record_count": 0,
        "strategy_evidence_record_count": 0,
        "forward_observation_record_count": 0,
        "future_return_or_conditional_metric_read": False,
        "historical_winner_ranking_read": False,
        "all_checks_pass": True,
    }


def freeze() -> dict[str, Any]:
    """只在首个资格证据和运行输出之前创建不可变清单。"""

    contract = load_contract(CONFIG)
    if MANIFEST.exists():
        verification = verify()
        if verification["failure_count"]:
            raise RuntimeError(f"现有 V1 清单验证失败：{verification['failures']}")
        return json.loads(MANIFEST.read_text(encoding="utf-8"))

    prefreeze = _assert_prefreeze_outputs_and_evidence_absent(contract)
    tracked_files = current_tracked_hashes()
    dependencies = current_dependency_snapshot(contract)
    component_statuses = current_component_status_snapshot(contract)
    frozen_at = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    frozen_scope_git_status = _git_value(
        "status",
        "--porcelain=v1",
        "--",
        *TRACKED_FILES,
    )
    git_tracking_state = {
        relative: _git_value("ls-files", "--error-unmatch", "--", relative) is not None
        for relative in TRACKED_FILES
    }
    manifest = {
        "schema_version": "1.0.0",
        "manifest_id": "REGIME_AWARE_STRATEGY_ALLOCATOR_V1_MANIFEST",
        "candidate_id": contract["protocol"]["candidate_id"],
        "version": contract["protocol"]["version"],
        "status": "FROZEN_BEFORE_FIRST_FORWARD_EVIDENCE",
        "evidence_label": contract["protocol"]["evidence_label"],
        "frozen_at": frozen_at,
        "true_forward_start_at": frozen_at,
        "true_forward_start_policy": contract["protocol"]["true_forward_start_policy"],
        "historical_component_result_cutoff": contract["protocol"][
            "historical_component_result_cutoff"
        ],
        "source_commit": _git_value("rev-parse", "HEAD"),
        "frozen_scope_git_status_at_freeze": frozen_scope_git_status,
        "frozen_file_git_tracking_state": git_tracking_state,
        "all_frozen_files_git_tracked_at_freeze": all(git_tracking_state.values()),
        "tracked_files": tracked_files,
        "tracked_content_sha256": tracked_content_sha256(tracked_files),
        "component_dependencies": dependencies,
        "component_status_snapshot": component_statuses,
        "prefreeze_evidence_audit": prefreeze,
        "performance_or_future_return_computed_during_freeze": False,
        "historical_strategy_return_ranking_used_during_freeze": False,
        "cash_is_valid_state": True,
        "governance": contract["governance"],
        "safety": contract["safety"],
    }
    atomic_json(MANIFEST, manifest)
    return manifest


def verify() -> dict[str, Any]:
    """验证受控实现和组件依赖是否仍与冻结清单逐字节一致。"""

    if not MANIFEST.is_file():
        return {
            "status": "FAIL_MISSING_REGIME_AWARE_ALLOCATOR_V1_MANIFEST",
            "failure_count": 1,
            "failures": ["missing_manifest"],
        }
    contract = load_contract(CONFIG)
    existing = json.loads(MANIFEST.read_text(encoding="utf-8"))
    live_files = current_tracked_hashes()
    live_dependencies = current_dependency_snapshot(contract)
    live_statuses = current_component_status_snapshot(contract)
    failures: list[str] = []
    if existing.get("candidate_id") != contract["protocol"]["candidate_id"]:
        failures.append("candidate_id")
    if existing.get("version") != contract["protocol"]["version"]:
        failures.append("version")
    if existing.get("tracked_files") != live_files:
        failures.append("tracked_files")
    if existing.get("tracked_content_sha256") != tracked_content_sha256(live_files):
        failures.append("tracked_content_sha256")
    if existing.get("component_dependencies") != live_dependencies:
        failures.append("component_dependencies")
    if existing.get("component_status_snapshot") != live_statuses:
        failures.append("component_status_snapshot")
    if existing.get("true_forward_start_at") != existing.get("frozen_at"):
        failures.append("true_forward_start_at")
    if existing.get("true_forward_start_policy") != "STRICTLY_AFTER_MANIFEST_FROZEN_AT":
        failures.append("true_forward_start_policy")
    if existing.get("performance_or_future_return_computed_during_freeze") is not False:
        failures.append("freeze_read_future_return")
    if existing.get("historical_strategy_return_ranking_used_during_freeze") is not False:
        failures.append("freeze_used_winner_ranking")
    if existing.get("cash_is_valid_state") is not True:
        failures.append("cash_is_valid_state")
    if existing.get("governance") != contract["governance"]:
        failures.append("governance")
    if existing.get("safety") != contract["safety"]:
        failures.append("safety")
    return {
        "status": (
            "PASS_REGIME_AWARE_ALLOCATOR_V1_MANIFEST_VERIFIED"
            if not failures
            else "FAIL_REGIME_AWARE_ALLOCATOR_V1_MANIFEST"
        ),
        "failure_count": int(len(failures)),
        "failures": failures,
        "manifest_path": MANIFEST.relative_to(ROOT).as_posix(),
        "manifest_sha256": sha256_file(MANIFEST),
        "tracked_content_sha256": tracked_content_sha256(live_files),
        "tracked_file_count": int(len(live_files)),
        "component_dependency_count": int(len(live_dependencies)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="冻结或验证市场状态感知策略配置器 V1")
    parser.add_argument("--mode", choices=("freeze", "verify"), default="verify")
    args = parser.parse_args()
    payload = freeze() if args.mode == "freeze" else verify()
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if payload.get("failure_count", 0) == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
