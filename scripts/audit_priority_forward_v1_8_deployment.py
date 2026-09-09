"""核验并登记 Priority Forward V1.8 的实际部署状态。

默认只读核验；仅在显式传入 ``--write``、当日 V1.7 收盘验收已通过、
Windows 任务与两个 Codex 自动化均已切换且没有同日 PCF 重试时，才写入
V1.8 调度面审计和部署凭据。部署凭据最后写入，因此中途失败不会激活 V1.8。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from scripts import audit_zero_paid_research_plan_completion_v1 as completion


ROOT = Path(__file__).resolve().parents[1]
TIMEZONE = completion.TIMEZONE
MANIFEST = ROOT / "config" / "priority_forward_research_operations_v1_8_manifest.json"
SUPERVISOR = ROOT / "config" / "priority_forward_supervisor_v1_4.yaml"
INSTALLER = ROOT / "scripts" / "install_priority_forward_morning_task_v1_8.ps1"
CENTRAL_AUDIT = ROOT / "reports" / "audit" / "ZERO_PAID_RESEARCH_PLAN_COMPLETION_V1.json"
SCHEDULE_AUDIT = (
    ROOT / "reports" / "audit" / "PRIORITY_FORWARD_ACTIVE_SCHEDULE_SURFACE_V1_8.json"
)
DEPLOYMENT_AUDIT = (
    ROOT / "reports" / "audit" / "PRIORITY_FORWARD_V1_8_DEPLOYMENT_VALIDATION.json"
)
MANIFEST_SHA256 = completion.V1_8_MANIFEST_SHA256
RUNNER = "run_priority_forward_codex_automation_v1_8.py"
SUPERVISOR_NAME = "priority_forward_supervisor_v1_4.yaml"
MANIFEST_NAME = "priority_forward_research_operations_v1_8_manifest.json"
EFFECTIVE_FROM = "2026-08-27T09:20:00+08:00"
EARLIEST_WRITE_AT = datetime(2026, 8, 26, 19, 0, tzinfo=TIMEZONE)

MORNING_AUTOMATION_PROMPT = (
    "每个工作日北京时间09:20继续当前任务，固定工作目录为 "
    r"C:\Users\戴周阳\Documents\New project 8。"
    r"先读取 config\priority_forward_research_operations_v1_8_manifest.json 和 "
    r"config\priority_forward_supervisor_v1_4.yaml，再前台运行 "
    r".\.venv\Scripts\python.exe scripts\run_priority_forward_codex_automation_v1_8.py "
    r"--phase morning --config config\priority_forward_supervisor_v1_4.yaml "
    r"--manifest config\priority_forward_research_operations_v1_8_manifest.json "
    f"--expected-manifest-sha256 {MANIFEST_SHA256}。"
    "入口必须先通过冻结清单校验，只以当天不可变任务回执和原子 claim 去重，"
    "只在09:25至09:35启动带 Python 模块入口修复的 PCF/IOPV V1.2.1 运行；"
    "不得盘后补跑或回填。命令结束后读取 "
    r"reports\audit\priority_forward_codex_run_status_v1_8.json、对应 V1.8 编排收据、"
    r"当天 reports\data_quality\primary_market_task_runs_v1_2_1 收据、免费来源回执、"
    r"官方日端点复核及 reports\data_quality\510300_primary_market_readiness_v1_2.json，"
    "发送中文日结：明确完成、休市、错过窗口、免费外部来源失败（退出码3）或程序失败"
    "（退出码1），并报告完整质量日及20/40/80/120门槛。没有当天内容寻址原始响应、"
    "第二官方端点复核与不可变收据时不得计为完整质量日；不运行其他策略、仓位、订单或券商流程。"
)

CLOSE_AUTOMATION_PROMPT = (
    "每个工作日北京时间19:00继续优先前瞻研究收盘后验收，固定工作目录为 "
    r"C:\Users\戴周阳\Documents\New project 8。"
    r"先读取 config\priority_forward_research_operations_v1_8_manifest.json 和 "
    r"config\priority_forward_supervisor_v1_4.yaml，再前台运行 "
    r".\.venv\Scripts\python.exe scripts\run_priority_forward_codex_automation_v1_8.py "
    r"--phase close --config config\priority_forward_supervisor_v1_4.yaml "
    r"--manifest config\priority_forward_research_operations_v1_8_manifest.json "
    f"--expected-manifest-sha256 {MANIFEST_SHA256}。"
    "入口必须先通过冻结清单校验，只以当天不可变任务回执和原子 claim 去重，"
    "只运行尚未尝试的行业预期差刷新与 V1.6 权威状态任务，不在收盘后补跑 PCF。"
    r"命令结束后读取 reports\audit\priority_forward_codex_run_status_v1_8.json、"
    r"对应 V1.8 编排收据和 reports\audit\priority_forward_authoritative_status_v1_6.json，"
    "发送中文日结：报告 PCF 当天真实状态、行业独立原点与成熟块、20/40/80/120以及"
    "20原点加4块和40原点加8块门槛，并保留未成熟、失败或错过状态。没有当天不可变证据时"
    "不得计为完成；不运行其他策略、仓位、订单或券商流程。"
)

AUTOMATION_SPECS = (
    {
        "id": "510300-pcf-iopv",
        "path": Path.home()
        / ".codex"
        / "automations"
        / "510300-pcf-iopv"
        / "automation.toml",
        "name": "510300 PCF IOPV严格前瞻采集",
        "phase": "morning",
        "rrule": "FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR;BYHOUR=9;BYMINUTE=20",
        "target_thread_id": "01a0166e-14e0-77e0-ba63-d118a1a80b0d",
        "prompt": MORNING_AUTOMATION_PROMPT,
    },
    {
        "id": "510300",
        "path": Path.home()
        / ".codex"
        / "automations"
        / "510300"
        / "automation.toml",
        "name": "优先前瞻研究收盘后验收",
        "phase": "close",
        "rrule": "FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR;BYHOUR=19;BYMINUTE=0",
        "target_thread_id": "019ffba4-af4f-7053-97b1-25a4f51957cf",
        "prompt": CLOSE_AUTOMATION_PROMPT,
    },
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 顶层不是对象：{path}")
    return payload


def relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def validate_automation(specification: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(specification["path"])
    failures: list[str] = []
    if not path.is_file():
        return {
            "id": specification["id"],
            "status": "FAILED",
            "path": str(path),
            "failures": ["AUTOMATION_TOML_MISSING"],
        }
    try:
        with path.open("rb") as handle:
            config = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        return {
            "id": specification["id"],
            "status": "FAILED",
            "path": str(path),
            "failures": [f"AUTOMATION_TOML_UNREADABLE:{exc}"],
        }

    prompt = str(config.get("prompt") or "")
    rrule = str(config.get("rrule") or "")
    required_tokens = (
        RUNNER,
        SUPERVISOR_NAME,
        MANIFEST_NAME,
        MANIFEST_SHA256,
        f"--phase {specification['phase']}",
    )
    if config.get("id") != specification["id"]:
        failures.append("AUTOMATION_ID_MISMATCH")
    if config.get("kind") != "heartbeat":
        failures.append("AUTOMATION_KIND_NOT_HEARTBEAT")
    if config.get("name") != specification["name"]:
        failures.append("AUTOMATION_NAME_MISMATCH")
    if config.get("status") != "ACTIVE":
        failures.append("AUTOMATION_NOT_ACTIVE")
    if rrule != specification["rrule"]:
        failures.append("AUTOMATION_SCHEDULE_MISMATCH")
    if config.get("target_thread_id") != specification["target_thread_id"]:
        failures.append("AUTOMATION_TARGET_THREAD_MISMATCH")
    if prompt != specification["prompt"]:
        failures.append("AUTOMATION_PROMPT_NOT_EXACT_V1_8")
    missing_tokens = [token for token in required_tokens if token not in prompt]
    if missing_tokens:
        failures.append("AUTOMATION_PROMPT_NOT_PINNED:" + ",".join(missing_tokens))
    if "run_priority_forward_codex_automation_v1_7.py" in prompt:
        failures.append("AUTOMATION_PROMPT_STILL_REFERENCES_V1_7_RUNNER")
    if completion.MAIN_MANIFEST_SHA256 in prompt:
        failures.append("AUTOMATION_PROMPT_STILL_REFERENCES_V1_7_MANIFEST")

    return {
        "id": specification["id"],
        "status": "PASS" if not failures else "FAILED",
        "path": str(path),
        "sha256": sha256_file(path),
        "target_thread_id": config.get("target_thread_id"),
        "rrule": rrule,
        "failures": failures,
    }


def windows_task_installer_audit() -> dict[str, Any]:
    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(INSTALLER),
            "-AuditOnly",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=60,
    )
    stdout = completed.stdout.lstrip("\ufeff").strip()
    try:
        payload = json.loads(stdout) if stdout else {}
    except json.JSONDecodeError:
        payload = {}
    return {
        "status": (
            "PASS"
            if completed.returncode == 0
            and isinstance(payload, dict)
            and payload.get("exact_match") is True
            and payload.get("runtime_patch_id")
            == completion.PCF_V1_8_RUNTIME_PATCH_ID
            else "FAILED"
        ),
        "exit_code": completed.returncode,
        "audit": payload,
        "stderr": completed.stderr.strip(),
    }


def close_acceptance_snapshot() -> dict[str, Any]:
    if not CENTRAL_AUDIT.is_file():
        return {
            "status": "FAILED",
            "failures": ["CENTRAL_AUDIT_MISSING"],
        }
    payload = read_json(CENTRAL_AUDIT)
    requirement = next(
        (
            item
            for item in payload.get("requirements", [])
            if isinstance(item, Mapping) and item.get("id") == "R22"
        ),
        None,
    )
    failures: list[str] = []
    requirement_state = (
        requirement.get("current_state")
        if isinstance(requirement, Mapping)
        and isinstance(requirement.get("current_state"), Mapping)
        else {}
    )
    close_attempt_accepted = bool(
        isinstance(requirement, Mapping)
        and (
            requirement.get("status") == "PASS"
            or (
                requirement.get("status") == "PENDING_NEXT_WINDOW"
                and requirement_state.get("acceptance_status") == "FAILED_RUN"
                and requirement_state.get("contract_valid") is True
            )
        )
    )
    if requirement is None:
        failures.append("R22_MISSING")
    elif not close_attempt_accepted:
        failures.append(f"R22_CLOSE_ATTEMPT_NOT_ACCEPTED:{requirement.get('status')}")
    if payload.get("contradiction_requirement_ids"):
        failures.append("CENTRAL_AUDIT_HAS_CONTRADICTIONS")
    generated_at = completion._parse_local_timestamp(payload.get("generated_at"))
    if generated_at is None or generated_at < EARLIEST_WRITE_AT:
        failures.append("CENTRAL_AUDIT_NOT_POST_CLOSE")
    return {
        "status": "PASS" if not failures else "FAILED",
        "path": relative_path(CENTRAL_AUDIT),
        "sha256": sha256_file(CENTRAL_AUDIT),
        "generated_at": payload.get("generated_at"),
        "r22_status": requirement.get("status") if requirement else None,
        "close_attempt_accepted": close_attempt_accepted,
        "r22_state": requirement_state,
        "contradiction_requirement_ids": payload.get(
            "contradiction_requirement_ids", []
        ),
        "failures": failures,
    }


def same_day_v1_8_activity(target_date: str) -> dict[str, Any]:
    task_receipts = completion.receipt_paths_for_local_date(
        completion.PCF_TASK_RECEIPT_DIRECTORY_V1_2_1, target_date
    )
    codex_receipts = completion.receipt_paths_for_local_date(
        completion.PCF_CODEX_RECEIPT_DIRECTORY_V1_8, target_date
    )
    claims = completion.receipt_paths_for_local_date(
        completion.PCF_CLAIM_DIRECTORY_V1_8,
        target_date,
        timestamp_fields=("claimed_at",),
    )
    paths = [*task_receipts, *codex_receipts, *claims]
    return {
        "status": "PASS_NO_SAME_DAY_V1_8_ACTIVITY" if not paths else "FAILED",
        "target_date": target_date,
        "task_receipt_count": len(task_receipts),
        "codex_receipt_count": len(codex_receipts),
        "claim_count": len(claims),
        "paths": [relative_path(path) for path in paths],
    }


def build_validation(now: datetime | None = None) -> dict[str, Any]:
    current = (now or datetime.now(TIMEZONE)).astimezone(TIMEZONE)
    manifest_check = completion.verify_manifest(MANIFEST)
    test_boundary = completion.read_json(completion.V1_8_TEST_BOUNDARY_AUDIT)
    runtime_boundary = completion.verify_runtime_test_boundary(MANIFEST, SUPERVISOR)
    windows_task = windows_task_installer_audit()
    automations = [validate_automation(specification) for specification in AUTOMATION_SPECS]
    pcf_task = completion.windows_task_snapshot(completion.PCF_TASK_NAME)
    t_only_task = completion.windows_task_snapshot(completion.T_ONLY_TASK_NAME)
    schedule_surface = completion.verify_active_schedule_surface(
        pcf_task,
        t_only_task,
        pcf_runner=RUNNER,
        pcf_manifest_sha256=MANIFEST_SHA256,
    )
    close_acceptance = close_acceptance_snapshot()
    same_day_activity = same_day_v1_8_activity("2026-08-26")

    checks = {
        "write_window_open": current >= EARLIEST_WRITE_AT,
        "manifest_exact": bool(
            manifest_check.get("status") == "PASS"
            and manifest_check.get("manifest_sha256") == MANIFEST_SHA256
        ),
        "test_boundary_exact": bool(
            test_boundary.get("status")
            == "PASS_TESTS_EXCLUDED_FROM_RUNTIME_MODEL_BOUNDARY"
            and test_boundary.get("frozen_manifest", {}).get("manifest_sha256")
            == MANIFEST_SHA256
            and runtime_boundary.get("status") == "PASS"
        ),
        "close_attempt_accepted": close_acceptance.get("status") == "PASS",
        "windows_task_exact": windows_task.get("status") == "PASS",
        "automations_exact": all(item.get("status") == "PASS" for item in automations),
        "schedule_surface_exact": schedule_surface.get("status") == "PASS",
        "no_same_day_v1_8_activity": same_day_activity.get("status")
        == "PASS_NO_SAME_DAY_V1_8_ACTIVITY",
        "deployment_audit_not_already_present": not DEPLOYMENT_AUDIT.exists(),
    }
    return {
        "schema_version": "1.0.0",
        "audit_id": "PRIORITY_FORWARD_V1_8_DEPLOYMENT_PREWRITE_VALIDATION",
        "generated_at": current.isoformat(),
        "status": "READY_TO_WRITE" if all(checks.values()) else "FAILED_VALIDATION",
        "checks": checks,
        "manifest": manifest_check,
        "test_boundary": {
            "frozen": test_boundary,
            "live": runtime_boundary,
        },
        "close_acceptance": close_acceptance,
        "windows_task": windows_task,
        "automations": automations,
        "schedule_surface": schedule_surface,
        "same_day_v1_8_activity": same_day_activity,
    }


def build_schedule_audit(validation: Mapping[str, Any]) -> dict[str, Any]:
    schedule = validation["schedule_surface"]
    windows_inventory = schedule["windows_inventory"]
    automation_inventory = schedule["automation_inventory"]
    return {
        "schema_version": "1.0.0",
        "audit_id": "PRIORITY_FORWARD_ACTIVE_SCHEDULE_SURFACE_V1_8",
        "generated_at": validation["generated_at"],
        "status": "PASS_UNAUTHORIZED_SCHEDULES_PAUSED_OR_DISABLED",
        "frozen_manifest": {
            "path": relative_path(MANIFEST),
            "sha256": MANIFEST_SHA256,
        },
        "runtime_patch_id": completion.PCF_V1_8_RUNTIME_PATCH_ID,
        "effective_from": EFFECTIVE_FROM,
        "checks": schedule["checks"],
        "codex_automations": {
            "authorized_active": validation["automations"],
            "inventory": automation_inventory.get("automations", []),
        },
        "windows_tasks": {
            "authorized_enabled": schedule["enabled_windows_task_names"],
            "disabled_legacy": schedule["disabled_legacy_task_names"],
            "inventory": windows_inventory.get("tasks", []),
        },
        "actions": {
            "research_run_triggered_by_deployment": False,
            "same_day_pcf_retry_performed": False,
            "historical_evidence_modified": False,
            "old_schedule_audit_overwritten": False,
        },
    }


def write_deployment_evidence(validation: Mapping[str, Any]) -> dict[str, Any]:
    if validation.get("status") != "READY_TO_WRITE":
        raise RuntimeError("部署前置核验未通过，拒绝写入 V1.8 部署凭据")

    schedule_audit = build_schedule_audit(validation)
    atomic_write_json(SCHEDULE_AUDIT, schedule_audit)
    schedule_sha256 = sha256_file(SCHEDULE_AUDIT)
    deployment = {
        "schema_version": "1.0.0",
        "audit_id": "PRIORITY_FORWARD_V1_8_DEPLOYMENT_VALIDATION",
        "generated_at": validation["generated_at"],
        "status": "PASS_V1_8_DEPLOYED_FOR_NEXT_ELIGIBLE_WINDOW",
        "effective_from": EFFECTIVE_FROM,
        "frozen_manifest": {
            "path": relative_path(MANIFEST),
            "sha256": MANIFEST_SHA256,
            "content_sha256": validation["manifest"].get("content_sha256"),
            "tracked_file_count": validation["manifest"].get(
                "tracked_file_count"
            ),
            "failure_count": validation["manifest"].get("failure_count"),
        },
        "runtime": {
            "runner": f"scripts/{RUNNER}",
            "supervisor": relative_path(SUPERVISOR),
            "runtime_patch_id": completion.PCF_V1_8_RUNTIME_PATCH_ID,
            "python_module_entrypoint": True,
        },
        "close_acceptance": validation["close_acceptance"],
        "windows_task": validation["windows_task"],
        "automations": validation["automations"],
        "schedule_surface_audit": {
            "path": relative_path(SCHEDULE_AUDIT),
            "sha256": schedule_sha256,
        },
        "same_day_pcf_retry_performed": False,
        "same_day_backfill_performed": False,
        "research_run_triggered_by_deployment": False,
        "historical_evidence_modified": False,
        "model_or_candidate_changed": False,
        "source_contract_changed": False,
        "maturity_thresholds_changed": False,
        "execution_authorization_changed": False,
    }
    atomic_write_json(DEPLOYMENT_AUDIT, deployment)
    activated = completion.active_runtime_contract()
    if not (
        activated.get("runtime_version") == "V1_8"
        and activated.get("deployment_audit_valid") is True
    ):
        raise RuntimeError("部署凭据写入后，中央运行契约未识别 V1.8")
    return {
        "status": deployment["status"],
        "deployment_audit": {
            "path": relative_path(DEPLOYMENT_AUDIT),
            "sha256": sha256_file(DEPLOYMENT_AUDIT),
        },
        "schedule_surface_audit": deployment["schedule_surface_audit"],
        "active_runtime": activated["runtime_version"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help="全部前置核验通过后写入 V1.8 调度面审计和部署凭据",
    )
    parser.add_argument(
        "--print-automation-update-specs",
        action="store_true",
        help="只输出两个现有 heartbeat 自动化的精确 V1.8 更新字段",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.print_automation_update_specs:
        payload = [
            {
                "id": specification["id"],
                "mode": "update",
                "kind": "heartbeat",
                "name": specification["name"],
                "prompt": specification["prompt"],
                "status": "ACTIVE",
                "rrule": specification["rrule"],
                "targetThreadId": specification["target_thread_id"],
            }
            for specification in AUTOMATION_SPECS
        ]
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    validation = build_validation()
    if args.write:
        if validation["status"] != "READY_TO_WRITE":
            print(json.dumps(validation, ensure_ascii=False, indent=2))
            return 2
        result = write_deployment_evidence(validation)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    return 0 if validation["status"] == "READY_TO_WRITE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
