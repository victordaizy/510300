"""冻结沪深300官方调入候选的预收益协议、候选和输入哈希。"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.a_share_hs_csi300_official_addition_forced_demand_v1 import (  # noqa: E402
    STRATEGY_ID,
    sha256_file,
    validate_event_ledger,
    validate_no_result_columns,
)


CONFIG_FILE = ROOT / "config" / "a_share_hs_csi300_official_addition_forced_demand_v1.yaml"
FROZEN_CODE_AND_PROTOCOL_FILES = (
    "config/a_share_hs_csi300_official_addition_forced_demand_v1.yaml",
    "docs/A_SHARE_HS_CSI300_OFFICIAL_ADDITION_FORCED_DEMAND_V1_PROTOCOL.md",
    "research/a_share_hs_csi300_official_addition_data_admission_v1.py",
    "research/a_share_hs_csi300_official_addition_forced_demand_v1.py",
    "scripts/acquire_a_share_hs_csi300_official_addition_events_v1.py",
    "scripts/admit_a_share_hs_csi300_official_addition_data_v1.py",
    "scripts/freeze_a_share_hs_csi300_official_addition_forced_demand_v1.py",
    "tests/test_a_share_hs_csi300_official_addition_data_admission_v1.py",
    "tests/test_a_share_hs_csi300_official_addition_forced_demand_v1.py",
)
TARGETED_TESTS = (
    "tests/test_a_share_hs_csi300_official_addition_forced_demand_v1.py",
    "tests/test_a_share_hs_csi300_official_addition_data_admission_v1.py",
)


def now_shanghai() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_bytes(
        path,
        (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 顶层不是对象：{path}")
    return payload


def assert_hash(path: Path, expected: str, label: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 不一致：expected={expected}, actual={actual}")


def run_targeted_tests() -> str:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", *TARGETED_TESTS, "-q"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    output = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())
    if result.returncode != 0:
        raise RuntimeError(f"D6 核心机制测试失败：\n{output}")
    return output


def flatten_reported_inputs(report: dict[str, Any]) -> dict[str, str]:
    flattened: dict[str, str] = {}
    for value in report.get("inputs", {}).values():
        entries = value if isinstance(value, list) else [value]
        for entry in entries:
            if not isinstance(entry, dict) or not entry.get("path") or not entry.get("sha256"):
                raise ValueError("D4/D5 报告中的输入证据格式无效")
            path = str(entry["path"])
            digest = str(entry["sha256"])
            if path in flattened and flattened[path] != digest:
                raise ValueError(f"D4/D5 报告对同一路径给出冲突哈希：{path}")
            flattened[path] = digest
    return flattened


def expected_admission_input_hashes(contract: dict[str, Any]) -> dict[str, str]:
    admission = contract["data_admission_contract"]
    market = admission["market_panel"]
    expected = {
        str(market["path"]): str(market["sha256"]),
        str(market["input_manifest_path"]): str(market["input_manifest_sha256"]),
        str(market["receipt_path"]): str(market["receipt_sha256"]),
        str(admission["security_master"]["path"]): str(admission["security_master"]["sha256"]),
        str(admission["security_status_intervals"]["path"]): str(
            admission["security_status_intervals"]["sha256"]
        ),
        str(contract["calendar_contract"]["path"]): str(contract["calendar_contract"]["sha256"]),
    }
    expected.update(
        {
            str(item["path"]): str(item["sha256"])
            for item in admission["corporate_action_success_receipts"]["paths"]
        }
    )
    return expected


def archive_existing(path: Path, history_root: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    digest = sha256_file(path)
    archive_path = history_root / f"{path.stem}_{digest[:16]}{path.suffix}"
    if archive_path.is_file():
        if sha256_file(archive_path) != digest:
            raise RuntimeError(f"冻结历史文件名冲突：{archive_path}")
    else:
        atomic_write_bytes(archive_path, path.read_bytes())
    return {
        "source_path": path.relative_to(ROOT).as_posix(),
        "archive_path": archive_path.relative_to(ROOT).as_posix(),
        "sha256": digest,
    }


def main() -> int:
    contract = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    if contract.get("strategy_id") != STRATEGY_ID:
        raise ValueError("配置策略标识与代码不一致")
    if contract["data_gates"]["D8"]["return_output_allowed"]:
        raise ValueError("D8 配置未禁止收益输出")

    missing_frozen = [name for name in FROZEN_CODE_AND_PROTOCOL_FILES if not (ROOT / name).is_file()]
    if missing_frozen:
        raise FileNotFoundError(f"冻结文件缺失：{missing_frozen}")

    artifacts = contract["artifacts"]
    acquisition_report_path = resolve_path(artifacts["acquisition_report_json"])
    official_manifest_path = resolve_path(artifacts["official_announcement_manifest"])
    ledger_path = resolve_path(artifacts["event_ledger_without_returns"])
    candidate_path = resolve_path(artifacts["candidate_list_without_returns"])
    admission_event_status_path = resolve_path(artifacts["data_admission_event_status_without_prices"])
    admission_report_path = resolve_path(artifacts["data_admission_report_json"])
    admission_report_md_path = resolve_path(artifacts["data_admission_report_md"])
    calendar_path = resolve_path(contract["calendar_contract"]["path"])
    required_inputs = [
        acquisition_report_path,
        official_manifest_path,
        ledger_path,
        candidate_path,
        admission_event_status_path,
        admission_report_path,
        admission_report_md_path,
        calendar_path,
    ]
    missing_inputs = [str(path) for path in required_inputs if not path.is_file()]
    if missing_inputs:
        raise FileNotFoundError(f"冻结输入缺失：{missing_inputs}")

    acquisition_report = load_json(acquisition_report_path)
    official_manifest = load_json(official_manifest_path)
    admission_report = load_json(admission_report_path)
    if acquisition_report.get("strategy_id") != STRATEGY_ID or official_manifest.get("strategy_id") != STRATEGY_ID:
        raise ValueError("采集报告或官方清单策略标识错配")
    if admission_report.get("strategy_id") != STRATEGY_ID:
        raise ValueError("D4/D5 准入报告策略标识错配")
    for gate in ("D1", "D2", "D3", "D8"):
        if acquisition_report["gates"][gate]["status"] != "PASS":
            raise RuntimeError(f"{gate} 尚未通过，禁止冻结：{acquisition_report['gates'][gate]}")
    if acquisition_report.get("return_evaluation") != "NOT_ALLOWED":
        raise RuntimeError("采集阶段不得允许收益评估")

    assert_hash(official_manifest_path, acquisition_report["official_manifest_sha256"], "官方公告清单")
    assert_hash(ledger_path, acquisition_report["event_ledger_sha256"], "无收益事件账本")
    assert_hash(candidate_path, acquisition_report["candidate_list_sha256"], "无收益候选清单")
    assert_hash(calendar_path, contract["calendar_contract"]["sha256"], "事件时钟交易日历")

    for gate in ("D4", "D5"):
        status = admission_report.get("gates", {}).get(gate, {}).get("status")
        if status not in {"PASS", "FAIL"}:
            raise RuntimeError(f"{gate} 未形成终局 PASS/FAIL：{status}")
    if admission_report.get("gates", {}).get("D8", {}).get("status") != "PASS":
        raise RuntimeError("D4/D5 准入的无收益边界未通过")
    admission_pass = all(admission_report["gates"][gate]["status"] == "PASS" for gate in ("D4", "D5"))
    expected_admission_status = (
        "DATA_CONTRACT_PASS_READY_FOR_SINGLE_HISTORICAL_EVALUATION"
        if admission_pass
        else "NO_VIEW_DATA_CONTRACT_FAILED"
    )
    expected_return_evaluation = "ALLOWED_ONCE_AFTER_FINAL_FREEZE" if admission_pass else "NOT_ALLOWED"
    expected_decision = "READY_FOR_FINAL_PRE_RETURN_FREEZE" if admission_pass else "STOP_NO_HISTORICAL_RETURN_EVALUATION"
    if admission_report.get("status") != expected_admission_status:
        raise ValueError("D4/D5 准入报告总状态与门状态不一致")
    if admission_report.get("return_evaluation") != expected_return_evaluation:
        raise ValueError("D4/D5 准入报告收益权限与门状态不一致")
    if admission_report.get("decision") != expected_decision:
        raise ValueError("D4/D5 准入报告决策与门状态不一致")
    if int(admission_report.get("historical_return_attempts_used", -1)) != 0:
        raise RuntimeError("D4/D5 前不得消耗历史收益检验次数")
    if admission_report.get("source_rescue_allowed") is not False:
        raise RuntimeError("D4/D5 失败后不得开放来源救援")

    reported_input_hashes = flatten_reported_inputs(admission_report)
    expected_input_hashes = expected_admission_input_hashes(contract)
    if reported_input_hashes != expected_input_hashes:
        raise ValueError("D4/D5 准入报告中的冻结输入路径或哈希与协议不一致")
    event_artifact = admission_report.get("event_status_artifact", {})
    if event_artifact.get("path") != admission_event_status_path.relative_to(ROOT).as_posix():
        raise ValueError("D4/D5 事件状态产物路径与协议不一致")
    assert_hash(admission_event_status_path, str(event_artifact.get("sha256")), "无价格事件准入状态")

    ledger = pd.read_parquet(ledger_path)
    candidates = pd.read_csv(candidate_path, dtype=str, encoding="utf-8-sig")
    admission_event_status = pd.read_parquet(admission_event_status_path)
    validate_event_ledger(ledger)
    validate_no_result_columns(candidates.columns)
    validate_no_result_columns(admission_event_status.columns)
    if len(ledger) != len(candidates):
        raise ValueError(f"事件账本与候选清单行数不一致：{len(ledger)} != {len(candidates)}")
    if set(ledger["event_key"].astype(str)) != set(candidates["event_key"].astype(str)):
        raise ValueError("事件账本与候选清单事件键不一致")
    if len(admission_event_status) != len(ledger):
        raise ValueError("D4/D5 事件状态行数与事件账本不一致")
    if set(admission_event_status["event_key"].astype(str)) != set(ledger["event_key"].astype(str)):
        raise ValueError("D4/D5 事件状态键与事件账本不一致")
    if set(admission_event_status["data_admission_status"].astype(str)) != {expected_admission_status}:
        raise ValueError("事件级数据准入状态与汇总报告不一致")
    if set(admission_event_status["return_evaluation"].astype(str)) != {"NOT_ALLOWED"}:
        raise RuntimeError("事件级产物在最终冻结前必须保持 NOT_ALLOWED")
    if int(event_artifact.get("rows", -1)) != len(admission_event_status):
        raise ValueError("D4/D5 报告记录的事件状态行数不一致")
    if list(event_artifact.get("columns", [])) != list(admission_event_status.columns):
        raise ValueError("D4/D5 报告记录的事件状态字段不一致")

    curated_root = resolve_path(artifacts["curated_root"])
    allowed_curated_names = {
        ledger_path.name,
        candidate_path.name,
        official_manifest_path.name,
        admission_event_status_path.name,
    }
    unexpected_outputs = [
        path.relative_to(ROOT).as_posix()
        for path in curated_root.iterdir()
        if path.is_file() and path.name not in allowed_curated_names
    ]
    if unexpected_outputs:
        raise RuntimeError(f"D8 发现未授权的同策略产物：{unexpected_outputs}")

    test_output = run_targeted_tests()
    seed = int(contract["bootstrap"]["seed"])
    if seed != 20260829 or int(contract["bootstrap"]["replications"]) != 10000:
        raise ValueError("bootstrap 随机种子或次数偏离冻结协议")

    code_hashes = {name: sha256_file(ROOT / name) for name in FROZEN_CODE_AND_PROTOCOL_FILES}
    input_hashes = {
        official_manifest_path.relative_to(ROOT).as_posix(): sha256_file(official_manifest_path),
        acquisition_report_path.relative_to(ROOT).as_posix(): sha256_file(acquisition_report_path),
        ledger_path.relative_to(ROOT).as_posix(): sha256_file(ledger_path),
        candidate_path.relative_to(ROOT).as_posix(): sha256_file(candidate_path),
        admission_event_status_path.relative_to(ROOT).as_posix(): sha256_file(admission_event_status_path),
        admission_report_path.relative_to(ROOT).as_posix(): sha256_file(admission_report_path),
        admission_report_md_path.relative_to(ROOT).as_posix(): sha256_file(admission_report_md_path),
        calendar_path.relative_to(ROOT).as_posix(): sha256_file(calendar_path),
    }
    for path, digest in reported_input_hashes.items():
        if path in input_hashes and input_hashes[path] != digest:
            raise ValueError(f"冻结输入哈希冲突：{path}")
        input_hashes[path] = digest

    test_command = f"{sys.executable} -m pytest {' '.join(TARGETED_TESTS)} -q"
    data_gates = {
        "D1": acquisition_report["gates"]["D1"],
        "D2": acquisition_report["gates"]["D2"],
        "D3": acquisition_report["gates"]["D3"],
        "D4": admission_report["gates"]["D4"],
        "D5": admission_report["gates"]["D5"],
        "D6": {"status": "PASS", "test_command": test_command, "result": test_output},
        "D7": {
            "status": "PASS",
            "code_file_count": len(code_hashes),
            "input_count": len(input_hashes),
            "seed_frozen": True,
            "large_admission_inputs_reused_from_verified_admission_report": sorted(reported_input_hashes),
        },
        "D8": {
            "status": "PASS",
            "source_admission_status": acquisition_report["gates"]["D8"]["status"],
            "data_admission_status": admission_report["gates"]["D8"]["status"],
            "unexpected_outputs": [],
            "return_columns_read": admission_report["gates"]["D8"]["return_columns_read"],
            "price_values_output": admission_report["gates"]["D8"]["price_values_output"],
        },
    }
    failed_data_gates = [gate for gate, result in data_gates.items() if result["status"] != "PASS"]
    all_data_gates_pass = not failed_data_gates
    final_status = (
        "DATA_CONTRACT_PASS_READY_FOR_SINGLE_HISTORICAL_EVALUATION"
        if all_data_gates_pass
        else "NO_VIEW_DATA_CONTRACT_FAILED"
    )
    final_return_evaluation = "ALLOWED_ONCE_AFTER_FINAL_FREEZE" if all_data_gates_pass else "NOT_ALLOWED"
    next_allowed_step = (
        "仅允许按本清单执行一次冻结历史评价；不得改参数、改窗口或补来源"
        if all_data_gates_pass
        else "停止；不得计算或查看历史收益，不得补来源救援本候选"
    )

    protocol_manifest_path = resolve_path(artifacts["protocol_manifest"])
    receipt_path = resolve_path(artifacts["freeze_receipt"])
    history_root = resolve_path(artifacts["freeze_history_root"])
    archived_prior_artifacts = [
        archived
        for archived in (
            archive_existing(protocol_manifest_path, history_root),
            archive_existing(receipt_path, history_root),
        )
        if archived is not None
    ]
    protocol_manifest = {
        "schema_version": "1.1.0",
        "strategy_id": STRATEGY_ID,
        "state": "PRE_RETURN_PROTOCOL_CANDIDATES_AND_DATA_ADMISSION_FROZEN",
        "frozen_at": now_shanghai(),
        "evidence_cutoff": contract["evidence_cutoff"],
        "code_and_protocol_files": code_hashes,
        "frozen_inputs": input_hashes,
        "input_hash_verification": {
            "directly_rehashed_small_or_primary_artifacts": sorted(
                set(input_hashes) - set(reported_input_hashes)
            ),
            "reused_from_just_completed_verified_data_admission": sorted(reported_input_hashes),
        },
        "superseded_artifacts_archived": archived_prior_artifacts,
        "candidate_count": int(len(candidates)),
        "cycle_count": int(ledger["cycle_id"].nunique()),
        "bootstrap": {
            "cluster": contract["bootstrap"]["cluster"],
            "replications": int(contract["bootstrap"]["replications"]),
            "seed": seed,
            "confidence_level": float(contract["bootstrap"]["confidence_level"]),
        },
        "historical_return_attempts_allowed": 1 if all_data_gates_pass else 0,
        "historical_return_attempts_used": 0,
        "data_gates": data_gates,
        "failed_data_gates": failed_data_gates,
        "status": final_status,
        "return_evaluation": final_return_evaluation,
        "next_allowed_step": next_allowed_step,
        "execution_permissions": {
            "research_only": True,
            "shadow": False,
            "position_mapping": False,
            "orders": False,
            "broker": False,
            "live": False,
        },
    }
    atomic_write_json(protocol_manifest_path, protocol_manifest)

    receipt = {
        "schema_version": "1.1.0",
        "strategy_id": STRATEGY_ID,
        "generated_at": now_shanghai(),
        "state": "PRE_RETURN_FREEZE_COMPLETE",
        "protocol_manifest_path": protocol_manifest_path.relative_to(ROOT).as_posix(),
        "protocol_manifest_sha256": sha256_file(protocol_manifest_path),
        "data_gate_summary": {
            gate: protocol_manifest["data_gates"][gate]["status"]
            for gate in ("D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8")
        },
        "failed_data_gates": failed_data_gates,
        "status": final_status,
        "return_evaluation": final_return_evaluation,
        "historical_return_attempts_allowed": 1 if all_data_gates_pass else 0,
        "historical_return_attempts_used": 0,
        "next_allowed_step": protocol_manifest["next_allowed_step"],
        "superseded_artifacts_archived": archived_prior_artifacts,
    }
    atomic_write_json(receipt_path, receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
