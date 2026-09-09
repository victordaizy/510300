"""免费 PCF/IOPV 目录兼容修复入口；复用原有采集窗口、回执与原子占位。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.priority_forward_data_paths_v1 import PATHS

BASE_MANIFEST = ROOT / "config/priority_forward_research_operations_v1_10_manifest.json"
BASE_HASH = "fe1192ffd8f76aeb2ff1ba486c6f1a299e941519ef9a4d5ba098c85733820b94"
MANIFEST = ROOT / "config/priority_forward_research_operations_v1_11_manifest.json"
CONFIG = ROOT / "config/priority_forward_supervisor_v1_7.yaml"
REQUIRED_ADDITIONS = {
    "scripts/priority_forward_data_paths_v1.py",
    "scripts/free_source_storage_v1_3.py",
    "scripts/collect_510300_primary_market.py",
    "scripts/collect_510300_primary_market_v1_4.py",
    "scripts/run_priority_forward_codex_automation_v1_11.py",
    "scripts/run_510300_primary_market_forward_v1_4.ps1",
    "scripts/run_510300_primary_market_collection_task_v1_4.ps1",
    "scripts/freeze_priority_forward_research_operations_v1_11.py",
    "scripts/install_priority_forward_morning_task_v1_11.ps1",
    "config/priority_forward_supervisor_v1_7.yaml",
    "data/reference/sse_trade_calendar_2026.csv",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def records(paths: list[str]) -> list[dict[str, Any]]:
    result = []
    for name in sorted(paths):
        path = PATHS.checked(name)
        result.append({"path": name, "sha256": digest(path), "bytes": path.stat().st_size})
    return result


def content_digest(rows: list[dict[str, Any]]) -> str:
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def verified_predecessor() -> dict[str, Any]:
    PATHS.validate_data_junction()
    if digest(PATHS.checked(BASE_MANIFEST)) != BASE_HASH:
        raise ValueError("原 V1.10 冻结清单字节发生变化，停止兼容修复")
    base = json.loads(BASE_MANIFEST.read_text(encoding="utf-8"))
    expected = base["files"]
    current = records([row["path"] for row in expected])
    if current != expected or content_digest(current) != base["content_sha256"]:
        raise ValueError("原 V1.10 受控文件发生变化，不能通过重冻结接纳漂移")
    return base


def verify_manifest(path: Path, expected_hash: str) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
        raise ValueError("必须提供固定的 64 位清单 SHA-256")
    if PATHS.checked(path) != MANIFEST or digest(MANIFEST) != expected_hash:
        raise ValueError("V1.11 清单身份或字节哈希不一致")
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if payload.get("manifest_id") != "PRIORITY_FORWARD_RESEARCH_OPERATIONS_V1_11":
        raise ValueError("V1.11 清单标识不一致")
    if payload.get("status") != "FROZEN_OPERATIONAL_REPAIR_READY":
        raise ValueError("V1.11 清单尚未冻结")
    base = verified_predecessor()
    if payload.get("predecessor_sha256") != BASE_HASH:
        raise ValueError("新清单未绑定原始 V1.10 清单")
    names = [row["path"] for row in payload["files"]]
    required = {row["path"] for row in base["files"]} | REQUIRED_ADDITIONS
    if len(names) != len(set(names)) or set(names) != required:
        raise ValueError("新清单的受控文件集合不一致")
    current = records(names)
    if current != payload["files"] or content_digest(current) != payload.get("content_sha256"):
        raise ValueError("V1.11 受控文件字节与冻结记录不一致")
    if payload.get("pcf_runtime") != base["pcf_runtime"]:
        raise ValueError("原 PCF 来源及质量合同发生变化")
    expected_fields = {
        "approved_resolved_data_root": str(PATHS.approved_data_root),
        "data_purchase_budget_cny": 0,
        "research_protocol_changed": False,
        "maturity_thresholds_changed": False,
        "position_impact": 0,
        "active_execution_phase": "morning_pcf_only",
    }
    if any(payload.get(key) != value for key, value in expected_fields.items()):
        raise ValueError("修复范围与已冻结的免费采集边界不一致")
    return {
        "status": "PASS_V1_11_FROZEN_ENTRYPOINT_VERIFIED",
        "manifest_sha256": expected_hash,
        "predecessor_sha256": BASE_HASH,
        "predecessor_files_unchanged": len(base["files"]),
        "tracked_file_count": len(current),
        "storage": PATHS.validate_data_junction(),
        "data_purchase_budget_cny": 0,
        "position_impact": 0,
        "collection_triggered": False,
    }


def collection_status(now: datetime) -> dict[str, Any]:
    """仅汇总直接回执与原成熟度记录，不触发其他研究或读取策略收益。"""
    today = now.date().isoformat()
    receipt_dir = PATHS.checked("reports/data_quality/primary_market_task_runs_v1_3")
    receipts = []
    for path in sorted(receipt_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        if payload.get("immutable_receipt") is not True:
            continue
        if payload.get("task_name") != "Codex-510300-Primary-Market-Collector":
            continue
        if PATHS.checked(str(payload.get("receipt_file") or "")).resolve() != path.resolve():
            raise ValueError("直接采集回执声明路径不一致")
        started = str(payload.get("started_at") or payload.get("scheduled_task_started_at") or "")
        if started[:10] == today:
            receipts.append((path, payload))
    readiness_path = PATHS.checked("reports/data_quality/510300_primary_market_readiness_v1_2.json")
    readiness = json.loads(readiness_path.read_text(encoding="utf-8")) if readiness_path.exists() else {}
    strict_days = sorted({
        str(row["trade_date"]) for row in readiness.get("daily_quality", [])
        if str(row.get("trade_date", "")) >= "2026-08-31"
        and row.get("complete_quality_day") is True
        and row.get("crosscheck", {}).get("required") is True
    })
    latest = max(receipts, key=lambda item: str(item[1].get("ended_at") or item[1].get("started_at") or "")) if receipts else None
    return {
        "generated_at": now.isoformat(), "target_date": today,
        "status": latest[1].get("collection_status") if latest else "NO_SAME_DAY_TASK_RECEIPT",
        "direct_task_exit_code": latest[1].get("task_exit_code") if latest else None,
        "same_day_receipts": [PATHS.relative(item[0]) for item in receipts],
        "readiness_generated_at": readiness.get("generated_at"),
        "strict_route_quality_dates_in_readiness": strict_days,
        "strict_route_quality_days_in_readiness": len(strict_days),
        "legacy_complete_quality_days_separate": readiness.get("legacy_complete_quality_day_count"),
        "thresholds": [20, 40, 80, 120],
        "data_purchase_budget_cny": 0, "position_impact": 0,
        "research_only": True, "position_mapping_enabled": False,
        "order_generation_enabled": False, "broker_connection_enabled": False,
        "live_trading_enabled": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="运行免费 PCF/IOPV V1.11 目录兼容版本")
    parser.add_argument("--phase", choices=("morning",), default="morning")
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--status-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--now")
    args = parser.parse_args()
    if args.now and not args.dry_run:
        raise ValueError("指定时间只允许用于不采集、不写回执的模拟检查")
    if PATHS.checked(args.config) != CONFIG:
        raise ValueError("仅接受 V1.11 固定编排配置")
    verified = verify_manifest(args.manifest, args.expected_manifest_sha256)
    if args.verify_only:
        print(json.dumps(verified, ensure_ascii=False, indent=2))
        return 0
    now = datetime.fromisoformat(args.now) if args.now else datetime.now(ZoneInfo("Asia/Shanghai"))
    if args.status_only:
        print(json.dumps(collection_status(now), ensure_ascii=False, indent=2))
        return 0
    from scripts import run_priority_forward_codex_automation_v1_5 as delegate

    config = delegate.load_config(CONFIG)
    original_renderer = delegate._run_renderer

    def render_pcf_status(root: Path, log_handle: Any) -> int:
        status = collection_status(datetime.now(ZoneInfo("Asia/Shanghai")))
        delegate._atomic_json(PATHS.checked("reports/audit/priority_forward_collection_status_v1_11.json"), status)
        log_handle.write((json.dumps(status, ensure_ascii=False) + "\n").encode("utf-8"))
        return 0

    delegate._run_renderer = render_pcf_status
    try:
        payload, exit_code = delegate.run_phase(config, "morning", now, dry_run=args.dry_run, wait_for_window=not args.dry_run)
    finally:
        delegate._run_renderer = original_renderer
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
