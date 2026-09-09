"""冻结优先前瞻研究的收据、门槛事件和统一状态实现。"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "priority_forward_research_operations_v1.yaml"
MANIFEST = ROOT / "config" / "priority_forward_research_operations_v1_manifest.json"
FROZEN_FILES = [
    "config/priority_forward_research_operations_v1.yaml",
    "docs/PRIORITY_FORWARD_RESEARCH_OPERATIONS_V1.md",
    "research/priority_forward_threshold_events.py",
    "scripts/freeze_priority_forward_research_operations_v1.py",
    "scripts/install_510300_daily_collection_task.ps1",
    "scripts/install_industry_expectation_gap_forward_operations_task.ps1",
    "scripts/install_priority_forward_research_status_task.ps1",
    "scripts/render_priority_forward_research_status.py",
    "scripts/run_510300_primary_market_collection_task.ps1",
    "scripts/run_industry_expectation_gap_forward_operations_task.ps1",
    "scripts/run_priority_forward_research_status_task.ps1",
    "tests/test_industry_expectation_gap_forward_operations.py",
    "tests/test_primary_market_forward.py",
    "tests/test_priority_forward_research_status.py",
    "tests/test_priority_forward_threshold_events.py",
]
SAFETY_FIELDS = (
    "position_mapping_enabled",
    "order_generation_enabled",
    "broker_connection_enabled",
    "live_trading_enabled",
)


def _path(relative: str) -> Path:
    root = ROOT.resolve()
    path = (root / relative).resolve()
    path.relative_to(root)
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON顶层必须是对象：{path}")
    return value


def _assert_safety(payload: dict[str, Any], label: str) -> None:
    safety = payload.get("safety", payload)
    for field in SAFETY_FIELDS:
        if safety.get(field) is not False:
            raise ValueError(f"{label}未明确关闭安全字段：{field}")


def _latest_receipt(directory: Path) -> tuple[Path, dict[str, Any]]:
    candidates = sorted(directory.glob("*.json"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise ValueError(f"没有运行收据：{directory}")
    path = candidates[-1]
    payload = _json(path)
    if payload.get("immutable_receipt") is not True:
        raise ValueError(f"运行收据未声明不可变：{path}")
    _assert_safety(payload, str(path))
    return path, payload


def _record(path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _git_text(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _write_once(path: Path, content: str) -> str:
    encoded = content.encode("utf-8")
    if path.exists():
        if path.read_bytes() != encoded:
            raise ValueError(f"冻结清单已存在且内容不同：{path}")
        return "EXISTING_IDENTICAL"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return "CREATED"


def main() -> int:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("优先前瞻配置顶层必须是对象")
    _assert_safety(config, "优先前瞻配置")
    primary_gates = config["gates"]["primary_market"]
    if primary_gates != {
        "minimum_quality_days": 20,
        "feature_freeze_days": 40,
        "first_unseen_evaluation_days": 80,
        "replication_days": 120,
    }:
        raise ValueError("PCF/IOPV阶段门槛发生变化")
    industry_gates = config["gates"]["industry_expectation_gap"]
    if industry_gates != {
        "calibration_origin_clusters": 20,
        "calibration_non_overlapping_60d_blocks": 4,
        "model_comparison_origin_clusters": 40,
        "model_comparison_non_overlapping_60d_blocks": 8,
    }:
        raise ValueError("行业预期差阶段门槛发生变化")

    industry_receipt_path, industry_receipt = _latest_receipt(
        _path("reports/forward/industry_expectation_gap_v1_evaluation/task_runs")
    )
    priority_receipt_path, priority_receipt = _latest_receipt(
        _path("reports/audit/priority_forward_status_task_runs")
    )
    report_snapshot = _path(str(priority_receipt["report_snapshot_file"]))
    markdown_snapshot = _path(str(priority_receipt["report_markdown_snapshot_file"]))
    report = _json(report_snapshot)
    _assert_safety(report, "优先前瞻报告快照")
    if report.get("governance_status") == "PASS":
        raise ValueError("当前治理证据意外为PASS，必须独立审计后再冻结")
    low_vol = report["directions"]["orthogonal_low_vol_replication"]
    if low_vol.get("eligible_to_start") is not False:
        raise ValueError("正交低波复制未被治理闸门阻断")
    threshold = report.get("threshold_events", {})
    if threshold.get("total_event_count") != 0 or threshold.get("new_event_count") != 0:
        raise ValueError("冻结时不应已有门槛跨越事件")

    frozen_records = []
    for relative in sorted(FROZEN_FILES):
        path = _path(relative)
        if not path.is_file():
            raise ValueError(f"待冻结文件不存在：{relative}")
        frozen_records.append(_record(path))
    primary_status = _path("reports/data_quality/510300_primary_market_task_status_20260819.json")
    runtime_evidence = {
        "primary_market_last_failure": _record(primary_status),
        "industry_task_receipt": _record(industry_receipt_path),
        "priority_status_task_receipt": _record(priority_receipt_path),
        "priority_status_json_snapshot": _record(report_snapshot),
        "priority_status_markdown_snapshot": _record(markdown_snapshot),
    }
    now = datetime.now(ZoneInfo(str(config["timezone"])))
    manifest = {
        "manifest_version": "PRIORITY_FORWARD_RESEARCH_OPERATIONS_V1_MANIFEST",
        "status": "FROZEN_MONITORING_IMPLEMENTATION",
        "frozen_at": now.isoformat(timespec="seconds"),
        "frozen_files": frozen_records,
        "runtime_evidence": runtime_evidence,
        "current_state": {
            "primary_market_full_coverage_days": int(
                report["directions"]["primary_market_pcf_iopv"]["full_coverage_days"]
            ),
            "industry_origin_cluster_count": int(
                report["directions"]["industry_expectation_gap"]["origin_cluster_count"]
            ),
            "industry_mature_origin_cluster_count": int(
                report["directions"]["industry_expectation_gap"][
                    "mature_origin_cluster_count"
                ]
            ),
            "threshold_event_count": 0,
            "governance_status": report["governance_status"],
            "orthogonal_low_vol_started": False,
        },
        "authorization": {
            "research_only": True,
            **{field: False for field in SAFETY_FIELDS},
            "automatic_trading_authorized": False,
        },
        "git": {
            "commit": _git_text("rev-parse", "HEAD"),
            "branch": _git_text("branch", "--show-current"),
            "dirty_before_freeze": bool(_git_text("status", "--porcelain")),
            "all_monitoring_files_tracked": False,
        },
    }
    state = _write_once(
        MANIFEST,
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    )
    print(f"优先前瞻监控清单：{state}")
    print(f"输出：{MANIFEST}")
    print(f"冻结文件：{len(frozen_records)}")
    print("门槛事件：0；交易授权：false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
