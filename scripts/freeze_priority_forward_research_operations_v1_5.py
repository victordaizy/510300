"""冻结零付费数据基础设施与 V1.5 原子编排入口。"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml


ROOT = Path(__file__).resolve().parents[1]
PREVIOUS_MANIFEST = (
    ROOT / "config" / "priority_forward_research_operations_v1_4_manifest.json"
)
MANIFEST = ROOT / "config" / "priority_forward_research_operations_v1_5_manifest.json"
REGISTRY = ROOT / "data" / "governance" / "FREE_SOURCE_REGISTRY.csv"
MATRIX = (
    ROOT
    / "data"
    / "curated"
    / "a_share_hs_official_cash_option_floor_alpha_v1_0_2_free_data_admission"
    / "free_data_admission_matrix_v1_0_2.csv"
)
AUDIT = ROOT / "reports" / "audit" / "ZERO_PAID_DATA_AND_CASH_OPTION_ADMISSION_V1.json"
SUPERVISOR = ROOT / "config" / "priority_forward_supervisor_v1_1.yaml"
AUTOMATION_ROOT = Path(
    os.environ.get(
        "CODEX_HOME",
        str(Path(os.environ["USERPROFILE"]) / ".codex"),
    )
) / "automations"
AUTOMATIONS = {
    "510300-pcf-iopv": AUTOMATION_ROOT / "510300-pcf-iopv" / "automation.toml",
    "510300": AUTOMATION_ROOT / "510300" / "automation.toml",
}
FROZEN_FILES = sorted(
    [
        "config/primary_market_forward.yaml",
        "config/priority_forward_supervisor_v1_1.yaml",
        "data/curated/a_share_hs_official_cash_option_floor_alpha_v1_0_2_free_data_admission/free_data_admission_matrix_v1_0_2.csv",
        "data/governance/FREE_SOURCE_REGISTRY.csv",
        "docs/A_SHARE_HS_OFFICIAL_CASH_OPTION_FLOOR_ALPHA_V1_0_2_FREE_DATA_ADMISSION_PROTOCOL.md",
        "docs/PRIORITY_FORWARD_RESEARCH_OPERATIONS_V1_5_ZERO_PAID_ATOMIC_CLAIM_ADDENDUM.md",
        "docs/ZERO_PAID_DATA_POLICY.md",
        "reports/audit/ZERO_PAID_DATA_AND_CASH_OPTION_ADMISSION_V1.json",
        "scripts/collect_510300_primary_market.py",
        "scripts/freeze_priority_forward_research_operations_v1_5.py",
        "scripts/run_510300_primary_market_collection_task_v1_1.ps1",
        "scripts/run_510300_primary_market_forward_v1_1.ps1",
        "scripts/run_priority_forward_codex_automation_v1_5.py",
        "tests/test_primary_market_free_source_contract_v1.py",
        "tests/test_priority_forward_codex_automation_v1_5.py",
        "tests/test_priority_forward_research_operations_v1_5_manifest.py",
    ]
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(path: Path, *, relative_to_root: bool = True) -> dict[str, Any]:
    resolved = path.resolve()
    display = (
        resolved.relative_to(ROOT.resolve()).as_posix()
        if relative_to_root
        else resolved.as_posix()
    )
    return {
        "path": display,
        "sha256": _sha256(resolved),
        "bytes": resolved.stat().st_size,
    }


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON 顶层必须是对象：{path}")
    return value


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _validate_policy_and_tables() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    policy = (ROOT / "docs" / "ZERO_PAID_DATA_POLICY.md").read_text(
        encoding="utf-8"
    )
    for required in (
        "DATA_PURCHASE_BUDGET = 0 CNY",
        "最多 2 条",
        "A_SHARE_HS_OFFICIAL_CASH_OPTION_FLOOR_ALPHA_V1",
        "CONDITIONAL_FORWARD_ONLY",
    ):
        if required not in policy:
            raise ValueError(f"零付费合同缺少冻结条款：{required}")

    registry = _csv_rows(REGISTRY)
    source_ids = [row["source_id"] for row in registry]
    if len(registry) != 16 or len(set(source_ids)) != 16:
        raise ValueError("免费来源登记册必须包含 16 个唯一 source_id")
    if any(float(row["access_cost_cny"]) != 0 for row in registry):
        raise ValueError("免费来源登记册出现非零采购成本")
    required_sources = {
        "SSE_PCF_COMMON_QUERY",
        "SSE_YUNHQ_IOPV_SNAPSHOT",
        "OFFICIAL_CASH_OPTION_TERMS_ARCHIVE",
        "SINA_HISTDATA_KLC2_RAW_UNADJUSTED",
        "EASTMONEY_PUSH2HIS_RAW_UNADJUSTED",
        "BAOSTOCK_QUERY_TRADE_DATES",
    }
    if not required_sources <= set(source_ids):
        raise ValueError("免费来源登记册缺少核心来源")

    matrix = _csv_rows(MATRIX)
    event_ids = [row["event_id"] for row in matrix]
    if len(matrix) != 21 or len(set(event_ids)) != 21:
        raise ValueError("现金选择权准入矩阵必须包含 21 个唯一事件")
    if any(row["price_values_read_by_matrix"].lower() != "false" for row in matrix):
        raise ValueError("现金选择权准入矩阵意外读取价格")
    if any(row["return_values_read_by_matrix"].lower() != "false" for row in matrix):
        raise ValueError("现金选择权准入矩阵意外读取收益")
    if any(
        row["data_gate_status"]
        != "BLOCKED_FREE_DATA_INCOMPLETE_BEFORE_FINAL_REPAIR"
        for row in matrix
    ):
        raise ValueError("现金选择权准入矩阵状态发生漂移")
    return registry, matrix


def _validate_audit(registry: list[dict[str, str]], matrix: list[dict[str, str]]) -> dict[str, Any]:
    audit = _json(AUDIT)
    if audit.get("data_purchase_budget_cny") != 0:
        raise ValueError("审计文件采购预算不是 0")
    if audit.get("audit_status") != "PASS_TABLE_STRUCTURE_BLOCKED_DATA_ADMISSION":
        raise ValueError("审计文件状态不一致")
    registry_audit = audit["free_source_registry"]
    matrix_audit = audit["cash_option_admission"]
    if registry_audit["sha256"] != _sha256(REGISTRY):
        raise ValueError("免费来源登记册哈希与审计文件不一致")
    if matrix_audit["sha256"] != _sha256(MATRIX):
        raise ValueError("现金选择权准入矩阵哈希与审计文件不一致")
    if registry_audit["row_count"] != len(registry):
        raise ValueError("免费来源登记册行数与审计文件不一致")
    if matrix_audit["event_count"] != len(matrix):
        raise ValueError("现金选择权事件数与审计文件不一致")
    if matrix_audit["data_gate_pass_count"] != 0:
        raise ValueError("现金选择权数据门不应已有通过事件")
    if matrix_audit["price_values_read_by_matrix"] is not False:
        raise ValueError("审计文件意外声明已读取价格")
    if matrix_audit["return_values_read_by_matrix"] is not False:
        raise ValueError("审计文件意外声明已读取收益")
    return audit


def _validate_supervisor() -> dict[str, Any]:
    config = yaml.safe_load(SUPERVISOR.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("V1.5 监督器配置顶层必须是对象")
    if config.get("version") != "PRIORITY_FORWARD_SUPERVISOR_V1_1":
        raise ValueError("V1.5 监督器配置版本不一致")
    by_id = {task["id"]: task for task in config["tasks"]}
    if by_id["PRIMARY_MARKET_PCF_IOPV"]["launcher"] != (
        "scripts/run_510300_primary_market_collection_task_v1_1.ps1"
    ):
        raise ValueError("PCF/IOPV 任务未绑定 V1.1 入口")
    if config["outputs"].get("codex_task_claim_directory") != (
        "reports/audit/priority_forward_task_claims_v1_5"
    ):
        raise ValueError("V1.5 监督器未配置原子占位目录")
    return config


def _validate_powershell_entrypoints() -> None:
    for relative in (
        "scripts/run_510300_primary_market_forward_v1_1.ps1",
        "scripts/run_510300_primary_market_collection_task_v1_1.ps1",
    ):
        if not (ROOT / relative).read_bytes().startswith(b"\xef\xbb\xbf"):
            raise ValueError(f"Windows PowerShell 入口缺少 UTF-8 BOM：{relative}")


def _validate_previous_repository_freeze() -> None:
    previous = _json(PREVIOUS_MANIFEST)
    for item in previous["frozen_files"]:
        path = ROOT / item["path"]
        if path.stat().st_size != item["bytes"] or _sha256(path) != item["sha256"]:
            raise ValueError(f"V1.4 仓库冻结文件漂移：{item['path']}")


def _validate_automations() -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    expected = {
        "510300-pcf-iopv": "morning",
        "510300": "close",
    }
    for automation_id, phase in expected.items():
        path = AUTOMATIONS[automation_id]
        value = tomllib.loads(path.read_text(encoding="utf-8"))
        if value.get("id") != automation_id or value.get("status") != "ACTIVE":
            raise ValueError(f"Codex 自动任务未保持 ACTIVE：{automation_id}")
        prompt = str(value.get("prompt", ""))
        required_fragments = (
            "config\\priority_forward_supervisor_v1_1.yaml",
            "docs\\PRIORITY_FORWARD_RESEARCH_OPERATIONS_V1_5_ZERO_PAID_ATOMIC_CLAIM_ADDENDUM.md",
            f"run_priority_forward_codex_automation_v1_5.py --phase {phase}",
        )
        for fragment in required_fragments:
            if fragment not in prompt:
                raise ValueError(f"Codex 自动任务未切换到 V1.5：{fragment}")
        evidence.append(
            {
                "id": automation_id,
                **_record(path, relative_to_root=False),
                "name": value["name"],
                "status": value["status"],
                "rrule": value["rrule"],
                "target_thread_id": value["target_thread_id"],
                "entrypoint_phase": phase,
            }
        )
    return evidence


def _git_text(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    ).stdout.strip()


def _validate_existing_manifest() -> None:
    manifest = _json(MANIFEST)
    for item in manifest["frozen_files"]:
        path = ROOT / item["path"]
        if path.stat().st_size != item["bytes"] or _sha256(path) != item["sha256"]:
            raise ValueError(f"V1.5 冻结文件漂移：{item['path']}")
    for item in manifest["external_automation_evidence"]:
        path = Path(item["path"])
        if path.stat().st_size != item["bytes"] or _sha256(path) != item["sha256"]:
            raise ValueError(f"V1.5 外部自动任务漂移：{item['id']}")


def main() -> int:
    if MANIFEST.exists():
        _validate_existing_manifest()
        print(f"V1.5 冻结清单：EXISTING_VERIFIED\n输出：{MANIFEST}")
        return 0

    registry, matrix = _validate_policy_and_tables()
    audit = _validate_audit(registry, matrix)
    _validate_supervisor()
    _validate_powershell_entrypoints()
    _validate_previous_repository_freeze()
    automations = _validate_automations()

    frozen_records = []
    for relative in FROZEN_FILES:
        path = ROOT / relative
        if not path.is_file():
            raise ValueError(f"待冻结文件不存在：{relative}")
        frozen_records.append(_record(path))

    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    matrix_audit = audit["cash_option_admission"]
    manifest = {
        "manifest_version": "PRIORITY_FORWARD_RESEARCH_OPERATIONS_V1_5_MANIFEST",
        "status": "FROZEN_ZERO_PAID_FOUNDATION_ACTIVE_AWAITING_NEXT_WINDOW",
        "frozen_at": now.isoformat(timespec="seconds"),
        "supersedes": {
            **_record(PREVIOUS_MANIFEST),
            "previous_manifest_preserved": True,
            "previous_repository_frozen_files_verified": True,
            "previous_external_automation_evidence_is_historical_snapshot": True,
            "reason": "ZERO_PAID_SOURCE_RECEIPTS_AND_ATOMIC_SAME_DAY_CLAIM_REPLACE_ACTIVE_DEPLOYMENT",
        },
        "frozen_files": frozen_records,
        "external_automation_evidence": automations,
        "contract": {
            "data_purchase_budget_cny": 0,
            "maximum_active_research_streams": 2,
            "only_new_frozen_test": "A_SHARE_HS_OFFICIAL_CASH_OPTION_FLOOR_ALPHA_V1",
            "option_orderbook_status": "CONDITIONAL_FORWARD_ONLY",
        },
        "current_state": {
            "overall_research_status": "NO_VERIFIED_TRADABLE_ALPHA_OR_STRONG_BETA_YET",
            "decision": "PAUSE_AND_FIX_FOUNDATION",
            "pcf_iopv": "V1_5_ACTIVE_AWAITING_NEXT_REAL_COLLECTION_WINDOW",
            "cash_option_admission": matrix_audit["current_status"],
            "cash_option_event_count": matrix_audit["event_count"],
            "cash_option_data_gate_pass_count": matrix_audit["data_gate_pass_count"],
            "cash_option_price_values_read": False,
            "cash_option_return_values_read": False,
        },
        "deployment": {
            "codex_morning_heartbeat_active": True,
            "codex_close_heartbeat_active": True,
            "single_entrypoint_v1_5_active": True,
            "atomic_same_day_claim_enabled": True,
            "source_receipt_contract_enabled": True,
            "next_real_collection_window_observed": False,
            "windows_task_scheduler_verified": False,
        },
        "authorization": {
            "research_only": True,
            "shadow_enabled": False,
            "broker_connection_enabled": False,
            "position_mapping_enabled": False,
            "order_generation_enabled": False,
            "live_trading_enabled": False,
        },
        "git": {
            "commit": _git_text("rev-parse", "HEAD"),
            "branch": _git_text("branch", "--show-current"),
            "dirty_before_freeze": bool(_git_text("status", "--porcelain")),
        },
    }
    encoded = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode(
        "utf-8"
    )
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(MANIFEST, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    print(f"V1.5 冻结清单：CREATED\n输出：{MANIFEST}\n冻结文件：{len(frozen_records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
