"""汇总两个优先前瞻方向和治理停止条件。"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.priority_forward_threshold_events import (
    append_new_threshold_events,
    derive_threshold_candidates,
    next_thresholds,
)


CONFIG_PATH = ROOT / "config" / "priority_forward_research_operations_v1.yaml"


def _path(relative: str) -> Path:
    root = ROOT.resolve()
    path = (root / relative).resolve()
    path.relative_to(root)
    return path


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON顶层必须是对象：{path}")
    return value


def _atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _status_timestamp(payload: dict[str, Any], path: Path) -> float:
    for field in ("ended_at", "failure_log_ended_at", "generated_at", "audit_date"):
        value = payload.get(field)
        if value:
            try:
                return datetime.fromisoformat(str(value)).timestamp()
            except ValueError:
                continue
    return path.stat().st_mtime


def _task_status_sort_key(payload: dict[str, Any], path: Path) -> tuple[float, int, str]:
    """同一运行的当前指针和不可变收据并存时，优先选择收据。"""

    return (
        _status_timestamp(payload, path),
        1 if payload.get("immutable_receipt") is True else 0,
        path.as_posix(),
    )


def _latest_task_status(
    current_relative: str,
    historical_globs: list[str],
    label: str,
) -> tuple[Path, dict[str, Any]]:
    candidates: list[tuple[Path, dict[str, Any]]] = []
    current = _path(current_relative)
    if current.exists():
        candidates.append((current, _json(current)))
    for pattern in historical_globs:
        for path in ROOT.glob(pattern):
            candidates.append((path, _json(path)))
    if not candidates:
        raise ValueError(f"没有{label}任务状态证据")
    return max(candidates, key=lambda item: _task_status_sort_key(item[1], item[0]))


def _assert_safety_disabled(payload: dict[str, Any], label: str) -> None:
    safety = payload.get("safety", payload)
    for field in (
        "position_mapping_enabled",
        "order_generation_enabled",
        "broker_connection_enabled",
        "live_trading_enabled",
    ):
        if safety.get(field) is not False:
            raise ValueError(f"{label}安全开关未明确关闭：{field}")


def build_report(config: dict[str, Any], generated_at: datetime) -> dict[str, Any]:
    inputs = config["inputs"]
    readiness = _json(_path(inputs["primary_market_readiness"]))
    primary_task_path, primary_task = _latest_task_status(
        inputs["primary_market_current_task_status"],
        [
            inputs["primary_market_historical_task_status_glob"],
            inputs["primary_market_task_receipt_glob"],
        ],
        "PCF/IOPV",
    )
    industry = _json(_path(inputs["industry_operations_status"]))
    industry_task_path, industry_task = _latest_task_status(
        inputs["industry_task_status"],
        [inputs["industry_task_receipt_glob"]],
        "行业预期差",
    )
    governance = _json(_path(inputs["governance_status"]))
    scheduler = _json(_path(inputs["scheduler_installation_status"]))
    supervisor = _json(_path(inputs["supervisor_status"]))
    codex_automation = _json(_path(inputs["codex_automation_status"]))

    _assert_safety_disabled(primary_task, "PCF/IOPV任务")
    _assert_safety_disabled(industry, "行业预期差")
    _assert_safety_disabled(industry_task, "行业预期差任务")
    _assert_safety_disabled(scheduler, "计划任务安装状态")
    _assert_safety_disabled(supervisor, "登录期守护进程状态")
    _assert_safety_disabled(codex_automation, "Codex心跳自动化状态")
    _assert_safety_disabled(config, "汇总配置")

    heartbeat = datetime.fromisoformat(
        str(supervisor["heartbeat_at"]).replace("Z", "+00:00")
    )
    if heartbeat.tzinfo is None:
        heartbeat = heartbeat.replace(tzinfo=generated_at.tzinfo)
    heartbeat_age_seconds = (generated_at - heartbeat).total_seconds()
    supervisor_heartbeat_fresh = 0 <= heartbeat_age_seconds <= 120
    morning_automation = codex_automation["automations"]["primary_market_morning"]
    close_automation = codex_automation["automations"]["close_validation"]
    codex_heartbeats_active = (
        codex_automation.get("status") == "ACTIVE_CODEX_HEARTBEAT_AUTOMATION"
        and morning_automation.get("status") == "ACTIVE"
        and close_automation.get("status") == "ACTIVE"
        and morning_automation.get("late_backfill_authorized") is False
        and close_automation.get("paper_signal_execution_authorized") is False
        and close_automation.get("orthogonal_low_vol_start_authorized") is False
    )

    primary_gates = config["gates"]["primary_market"]
    full_days = int(readiness["full_coverage_days"])
    primary = {
        "priority": 1,
        "status": readiness["status"],
        "stage": readiness["stage"],
        "latest_observed_trade_date": readiness["last_observed_trade_date"],
        "observed_trading_days": readiness["observed_trading_days"],
        "full_coverage_days": full_days,
        "task_evidence": {
            "path": primary_task_path.relative_to(ROOT).as_posix(),
            "run_status": primary_task.get("run_status"),
            "collection_status": primary_task.get("collection_status"),
            "error": primary_task.get("error"),
        },
        "gates": {
            "quality_audit": {
                "required": primary_gates["minimum_quality_days"],
                "eligible": full_days >= primary_gates["minimum_quality_days"],
            },
            "feature_freeze": {
                "required": primary_gates["feature_freeze_days"],
                "eligible": full_days >= primary_gates["feature_freeze_days"],
            },
            "first_unseen_evaluation": {
                "required": primary_gates["first_unseen_evaluation_days"],
                "eligible": full_days >= primary_gates["first_unseen_evaluation_days"],
            },
            "replication": {
                "required": primary_gates["replication_days"],
                "eligible": full_days >= primary_gates["replication_days"],
            },
        },
        "next_action": "CONTINUE_STRICT_FORWARD_COLLECTION",
    }

    maturity = industry["maturity"]
    industry_summary = {
        "priority": 2,
        "status": industry["status"],
        "origin_collection_status": industry["origin_collection"]["status"],
        "outcome_collection_status": industry["outcome_collection"]["status"],
        "origin_cluster_count": maturity["origin_cluster_count"],
        "mature_origin_cluster_count": maturity["mature_origin_cluster_count"],
        "non_overlapping_60d_block_count": maturity["non_overlapping_60d_block_count"],
        "origin_blockers": industry["origin_collection"]["blockers"],
        "outcome_blockers": industry["outcome_collection"]["blockers"],
        "task_run_status": industry_task["run_status"],
        "task_evidence_path": industry_task_path.relative_to(ROOT).as_posix(),
        "calibration": maturity["calibration"],
        "model_comparison": maturity["model_comparison"],
        "next_action": "WAIT_FOR_NEW_OFFICIAL_PACKAGE_AND_CURRENT_OUTCOME_INPUTS",
    }

    governance_status = str(governance.get("overall_status"))
    orthogonal = {
        "priority": 3,
        "status": "NOT_STARTED_GOVERNANCE_GATE_BLOCKED",
        "required_governance_status": config["gates"]["orthogonal_low_vol"][
            "required_governance_status"
        ],
        "actual_governance_status": governance_status,
        "eligible_to_start": governance_status
        == config["gates"]["orthogonal_low_vol"]["required_governance_status"],
        "next_action": "WAIT_FOR_GOVERNANCE_PASS",
    }
    if orthogonal["eligible_to_start"]:
        orthogonal["status"] = "GOVERNANCE_GATE_PASSED_REQUIRES_SEPARATE_PROTOCOL"

    alerts: list[dict[str, str]] = []
    if primary_task.get("run_status") != "SUCCESS":
        alerts.append(
            {
                "severity": "WARNING",
                "direction": "PRIMARY_MARKET_PCF_IOPV",
                "code": str(primary_task.get("collection_status", "TASK_FAILED")),
                "message": str(primary_task.get("error") or "采集任务未成功"),
            }
        )
    if not industry["outcome_collection"]["ready"]:
        alerts.append(
            {
                "severity": "WARNING",
                "direction": "INDUSTRY_EXPECTATION_GAP",
                "code": industry["outcome_collection"]["status"],
                "message": "结果输入尚未达到严格前瞻评价日期闸门。",
            }
        )
    if not orthogonal["eligible_to_start"]:
        alerts.append(
            {
                "severity": "BLOCK",
                "direction": "ORTHOGONAL_LOW_VOL_REPLICATION",
                "code": governance_status,
                "message": "治理未通过，禁止启动正交低波复制。",
            }
        )
    if scheduler.get("registered_tasks_verified") is not True:
        if codex_heartbeats_active:
            scheduler_code = "CODEX_HEARTBEATS_ACTIVE_WINDOWS_FALLBACK_BLOCKED"
            scheduler_message = (
                "早间严格采集与收盘后验收的Codex心跳均已启用；Windows任务调度器和"
                "登录启动项仍无权限，只缺本机持久备用通道。"
            )
        elif supervisor_heartbeat_fresh:
            scheduler_code = "BLOCKED_PERSISTENT_AUTOMATION_PERMISSION"
            scheduler_message = (
                "当前登录会话的隐藏守护进程正在运行，但任务调度器和登录自启动均未获权限；"
                "重启或退出登录后不能自动恢复。"
            )
        else:
            scheduler_code = str(
                scheduler.get("status", "SCHEDULER_AND_SUPERVISOR_NOT_VERIFIED")
            )
            scheduler_message = "计划任务未安装，且登录期守护进程心跳未通过新鲜度核验。"
        alerts.append(
            {
                "severity": "WARNING" if codex_heartbeats_active else "BLOCK",
                "direction": "WINDOWS_TASK_SCHEDULER",
                "code": scheduler_code,
                "message": scheduler_message,
            }
        )

    return {
        "schema_version": "1.0.0",
        "report_version": config["version"],
        "generated_at": generated_at.isoformat(),
        "overall_status": "COLLECTING_FORWARD_WITH_BLOCKERS",
        "directions": {
            "primary_market_pcf_iopv": primary,
            "industry_expectation_gap": industry_summary,
            "orthogonal_low_vol_replication": orthogonal,
        },
        "governance_status": governance_status,
        "scheduler": {
            "status": (
                "CODEX_HEARTBEATS_ACTIVE_WINDOWS_FALLBACK_BLOCKED"
                if codex_heartbeats_active
                else (
                    "RUNNING_SESSION_SCOPED_DEGRADED_AUTOMATION"
                    if supervisor_heartbeat_fresh
                    else scheduler["status"]
                )
            ),
            "registered_tasks_verified": scheduler["registered_tasks_verified"],
            "manual_runners_verified": scheduler["manual_runners_verified"],
            "windows_powershell_5_1_parser_verified": scheduler[
                "windows_powershell_5_1_parser_verified"
            ],
            "evidence_path": _path(inputs["scheduler_installation_status"])
            .relative_to(ROOT)
            .as_posix(),
            "session_supervisor_status": supervisor["status"],
            "session_supervisor_heartbeat_at": supervisor["heartbeat_at"],
            "session_supervisor_heartbeat_age_seconds": heartbeat_age_seconds,
            "session_supervisor_heartbeat_fresh": supervisor_heartbeat_fresh,
            "login_autostart_verified": supervisor["login_autostart_verified"],
            "supervisor_evidence_path": _path(inputs["supervisor_status"])
            .relative_to(ROOT)
            .as_posix(),
            "codex_heartbeats_active": codex_heartbeats_active,
            "primary_market_heartbeat_id": morning_automation["id"],
            "close_validation_heartbeat_id": close_automation["id"],
            "codex_automation_evidence_path": _path(
                inputs["codex_automation_status"]
            )
            .relative_to(ROOT)
            .as_posix(),
        },
        "alerts": alerts,
        "safety": dict(config["safety"]),
    }


def _markdown(report: dict[str, Any]) -> str:
    primary = report["directions"]["primary_market_pcf_iopv"]
    industry = report["directions"]["industry_expectation_gap"]
    orthogonal = report["directions"]["orthogonal_low_vol_replication"]
    scheduler = report["scheduler"]
    threshold_events = report.get("threshold_events", {})
    lines = [
        "# 优先前瞻研究状态",
        "",
        f"- 总状态：`{report['overall_status']}`",
        f"- 生成时间：`{report['generated_at']}`",
        f"- 治理状态：`{report['governance_status']}`",
        "",
        "## 1. PCF/IOPV执行信息",
        "",
        f"- 成熟度：{primary['full_coverage_days']}/20 个最低质量日；首次未见评价要求80日，复制要求120日。",
        f"- 最新观测交易日：`{primary['latest_observed_trade_date']}`",
        f"- 最近任务：`{primary['task_evidence']['run_status']}` / `{primary['task_evidence']['collection_status']}`",
        "- 动作：只继续严格前瞻采集，不生成仓位或订单。",
        "",
        "## 2. 行业预期差",
        "",
        f"- 独立原点：{industry['origin_cluster_count']}；成熟原点：{industry['mature_origin_cluster_count']}；非重叠60日块：{industry['non_overlapping_60d_block_count']}。",
        f"- 新原点闸门：`{industry['origin_collection_status']}`",
        f"- 结果输入闸门：`{industry['outcome_collection_status']}`",
        f"- 最近任务收据：`{industry['task_evidence_path']}`",
        "- 动作：等待新的官方证据包和更新后的月度点时面板；不复制旧预测。",
        "",
        "## 3. 正交低波复制",
        "",
        f"- 状态：`{orthogonal['status']}`",
        "- 动作：治理通过前不启动。",
        "",
        "## 4. Windows计划任务",
        "",
        f"- 状态：`{scheduler['status']}`",
        f"- 已核验安装：`{scheduler['registered_tasks_verified']}`",
        f"- 手动运行器已验证：`{scheduler['manual_runners_verified']}`",
        f"- Windows PowerShell 5.1解析：`{scheduler['windows_powershell_5_1_parser_verified']}`",
        f"- 当前会话守护进程：`{scheduler['session_supervisor_status']}`",
        f"- 心跳新鲜：`{scheduler['session_supervisor_heartbeat_fresh']}`",
        f"- 登录自启动已验证：`{scheduler['login_autostart_verified']}`",
        f"- Codex心跳自动化：`{scheduler['codex_heartbeats_active']}`",
        f"- 证据：`{scheduler['evidence_path']}`",
        f"- 守护进程证据：`{scheduler['supervisor_evidence_path']}`",
        f"- Codex自动化证据：`{scheduler['codex_automation_evidence_path']}`",
        "",
        "## 告警",
        "",
    ]
    if report["alerts"]:
        lines.extend(
            f"- `{item['severity']}` `{item['direction']}` `{item['code']}`：{item['message']}"
            for item in report["alerts"]
        )
    else:
        lines.append("- 无。")
    if threshold_events:
        lines.extend(
            [
                "",
                "## 门槛事件",
                "",
                f"- 首次跨越事件总数：{threshold_events['total_event_count']}",
                f"- 本次新增事件：{threshold_events['new_event_count']}",
                f"- 追加式账本：`{threshold_events['ledger_path']}`",
                "- 门槛事件只开放对应研究步骤，不授权仓位、订单或实盘。",
            ]
        )
    lines.extend(
        [
            "",
            "仓位映射、订单生成、券商连接与实盘均为关闭状态。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("汇总配置顶层必须是对象")
    now = datetime.now(ZoneInfo(config["timezone"]))
    report = build_report(config, now)
    threshold_path = _path(config["outputs"]["threshold_event_ledger"])
    threshold_summary = append_new_threshold_events(
        threshold_path,
        derive_threshold_candidates(report),
    )
    threshold_summary["ledger_path"] = threshold_path.relative_to(ROOT).as_posix()
    threshold_summary["next_thresholds"] = next_thresholds(report)
    report["threshold_events"] = threshold_summary
    for event_id in threshold_summary["new_event_ids"]:
        report["alerts"].append(
            {
                "severity": "MILESTONE",
                "direction": "THRESHOLD_MONITOR",
                "code": event_id,
                "message": "研究门槛首次达到；仅开放对应研究步骤，不构成仓位或交易授权。",
            }
        )
    _atomic(
        _path(config["outputs"]["json"]),
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    _atomic(_path(config["outputs"]["markdown"]), _markdown(report))
    print(f"优先前瞻研究状态：{report['overall_status']}")
    print(f"告警数量：{len(report['alerts'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
