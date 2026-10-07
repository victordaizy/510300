"""核对实际接续证据并记录无进展；不拟合、不采集、不改变研究目标。"""
from __future__ import annotations

import copy
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).absolute().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research import all_factor_joint_path_attribution_v1 as io

CONTEXT = Path(__file__).absolute().parent
STATE = CONTEXT / "state.json"
BLOCKER = "NO_ADMITTED_COMPLETE_INFORMATION_USE_OR_ACTUAL_NEW_INDEPENDENT_EVIDENCE_AFTER_R260"
FINANCE = ["latest_actual_financial_decision", "latest_actual_financial_result", "latest_actual_financial_status",
    "current_new_strategy_return_sharpe", "latest_actual_financial_primary_four_scene_metrics"]
FORWARD = ["forward_protocol", "forward_registry", "new_prospective_observations", "earliest_future_exchange_session",
    "registered_candidate_intents", "new_prospective_completed_points", "next_new_close_eligible_at", "current_validated_candidates",
    "forward_account_comparison_protocol", "latest_forward_account_check", "new_prospective_sessions_this_continuation",
    "new_prospective_cycles_this_continuation", "next_experiment"]


def main():
    original = io.read(STATE)
    service = io.read(CONTEXT / "goal_snapshot_before_post_R260_check.json")
    io.require(service["goal"]["status"] == "active", "当前目标不是active，不能自动核对接续。")
    before_git = subprocess.check_output(["git", "status", "--short", "--untracked-files=no"], cwd=ROOT)
    observer = ROOT / "reports/research/510300_point_forward_observer_v1"
    clock = io.read(observer / "latest_check.json")
    version = io.read(observer / "state.json")
    prices = pd.read_parquet(ROOT / version["version"] / "inputs/candidate_prices.parquet", columns=["date"])
    points = pd.read_parquet(ROOT / version["version"] / "results/全部自然点位.parquet", columns=["entry_date", "exit_date", "status"])
    new_points = points[points.entry_date.ge(pd.Timestamp(version["first_execution"])) & points.status.eq("COMPLETE")]
    now = pd.Timestamp(io.now())
    io.require(0 <= (now - pd.Timestamp(clock["at"])).total_seconds() < 600, "接续实际时钟检查过期，应先执行原check。")
    io.require(clock["latest_completed_session"] >= str(pd.Timestamp(prices.date.max()).date()), "时钟截止早于当前已接纳行情。")
    snapshots = ROOT / "reports/research/510300_current_official_share_source_v1/daily_observations"
    actual_share_versions = list(snapshots.glob("**/normalized.json")) if snapshots.exists() else []
    financial_path = ROOT / original["latest_actual_financial_result"]
    financial = io.read(financial_path)
    recent = [row for row in financial["primary_four_scene_metrics"] if row["period"] == "2020_2026" and row["cost"] == "STRESS"][0]
    terminal_paths = [io.FINANCE / "run_completed.json", io.OUT / "run_completed.json",
        ROOT / "reports/research/510300_all_factor_catalyst_source_alignment_v1/run_completed.json"]
    terminal = [{"path": str(path.absolute().relative_to(ROOT)), "actual": io.read(path)} for path in terminal_paths]
    known_terminal = all(row["actual"].get("terminal") is True for row in terminal)
    no_action = bool(clock["new_completed_sessions"] == 0 and not len(new_points) and not actual_share_versions
        and original["current_admitted_unrun_complete_uses"] == 0 and original["current_admitted_unrun_numeric_candidates"] == 0
        and known_terminal and original["independent_official_share_observer"].get("live_process_handle") is None
        and not original.get("background_monitor_running", False))
    io.require(no_action, "出现新收盘/来源/完成点位/准入用途或运行状态不确定，应该执行具体工作，不能计无进展。")
    prior_same = original.get("latest_goal_turn_classification") == "NO_PROGRESS" and original.get("blocked_audit_key") == BLOCKER
    count = int(original.get("blocked_audit_count", 0)) + 1 if prior_same else 1
    folder = CONTEXT / "continuation_audits" / ("post_R260_" + str(count) + "_" + now.strftime("%Y%m%d_%H%M%S_%f"))
    folder.mkdir(parents=True, exist_ok=False)
    record_path = folder / "record.json"
    protected = {key: copy.deepcopy(original.get(key)) for key in FINANCE + FORWARD + ["independent_official_share_observer"]}
    record = {
        "at": now.isoformat(), "objective": service["goal"]["objective"],
        "previous_goal_turn_classification": original.get("latest_goal_turn_classification"),
        "classification": "NO_PROGRESS", "research_progress": False, "verified_live_wait": False,
        "blocker_id": BLOCKER, "consecutive_no_progress_count": count, "blocked_threshold": 3,
        "blocked_status_eligible_now": count >= 3, "goal_service_status_before_check": "active", "goal_achieved": False,
        "actual_current_clock_check": clock, "actual_current_price_rows": len(prices),
        "actual_current_price_last_date": str(pd.Timestamp(prices.date.max()).date()),
        "actual_new_completed_point_rows": len(new_points), "actual_new_share_normalized_versions": len(actual_share_versions),
        "admitted_unrun_complete_uses": original["current_admitted_unrun_complete_uses"],
        "admitted_unrun_numeric_candidates": original["current_admitted_unrun_numeric_candidates"],
        "known_recent_terminal_runs": terminal, "known_live_collection_handle": None,
        "all_four_latest_financial_gates_passed": financial["all_four_economic_gates_passed"],
        "actual_latest_financial_decision": original["latest_actual_financial_decision"],
        "actual_latest_financial_status": financial["status"], "latest_recent_stress_financial_metrics": recent,
        "financial_summary_sha256": io.sha(financial_path),
        "new_fits": 0, "new_accounts": 0, "new_labels": 0, "new_source_requests": 0,
        "safe_action_disposition": "原声明有限用途已完成；无新增合格收盘/来源/独立点位或已准入不同完整用途。重扫旧65节点和调原失败参数不提供合格证据。",
        "remaining_requirement": "同资金/费用/风险提高净收益与完整日历Sharpe，同时实际净pB>1、净期望>0和独立验证；目标不缩为文档、框架或空仓。",
        "reentry": "真实合格新输入/独立样本到达或不同完整用途获得来源依据；原日线最早10月8日15:05、份额首采窗10月8日23:05。",
        "state_updates": 1, "orders_authorized": False,
    }
    io.write(folder / "state_before.json", original, exclusive=True)
    io.write(folder / "goal_status_before.json", service, exclusive=True)
    io.write(record_path, record, exclusive=True)
    report = f"""# R260之后接续核对：第{count}次无进展

实际核对时间：{now.isoformat()}。上一轮完成全量来源对应，属于研究进展；本轮仅核对新证据，分类NO_PROGRESS，不是运行中任务的等待。

原观察器check确认新完整收盘0、最新可完成交易日仍2026-09-30；原日历覆盖检查确认应有新观察0、独立验证未建立。实际价格末日2026-09-30，新自然完成点位0、实际新官方份额归一版本0、已准入未运行完整用途/数值候选均0。R256/R258/R260三次已知最近运行均terminal，没有可继续轮询的采集句柄。

最新真实金融仍TECH.R256固定拒绝，四场景经济门未通过。近期STRESS净年化{recent['net_cagr']:.4%}、净Sharpe {recent['net_sharpe']:.6f}、实际净pB {recent['p_times_b']:.6f}；该失败不因状态核对改变。

剩余目标仍为同口径提高收益、净Sharpe、实际净pB/净期望和独立验证，不能重命名旧失败或把归档称进展。相同阻碍本轮计{count}/3，目标保持active；首次核对不调用blocked。原10月8日15:05新收盘和23:05份额采集窗仅是协议时点，须实际输入到达才接续。

本核对不修改模型或长期研究结论；记录路径：{record_path.absolute().relative_to(ROOT).as_posix()}。
"""
    (folder / "接续核对.md").write_text(report, encoding="utf-8")
    updated = copy.deepcopy(original)
    updated.update({"updated_at": now.isoformat(), "previous_goal_turn_classification": original.get("latest_goal_turn_classification"),
        "latest_goal_turn_classification": "NO_PROGRESS", "blocked_audit_count": count, "goal_blocked_audit_count": count,
        "blocked_audit_key": BLOCKER, "current_goal_turn_blocker_id": BLOCKER,
        "blocked_reason": "当前无已准入不同完整用途、真实新收盘/官方份额版本或独立完成点位；原有限来源用途已终止。",
        "blocking_decision": "FIRST_NO_PROGRESS_CHECK_GOAL_ACTIVE" if count == 1 else "REPEATED_NO_PROGRESS_CHECK_PENDING_THRESHOLD",
        "latest_no_progress_check": record_path.absolute().relative_to(ROOT).as_posix(),
        "latest_continuation_outcome": f"R260后第{count}次NO_PROGRESS：实际check新收盘0、新独立点位0、新官方份额版本0、待运行合格用途0；目标未达。",
        "status": "research_active", "goal_achieved": False})
    io.write(STATE, updated)
    after = io.read(STATE)
    io.require(all(after.get(k) == v for k, v in protected.items()), "金融/前瞻/份额合同在无进展核对中改变。")
    io.require(subprocess.check_output(["git", "status", "--short", "--untracked-files=no"], cwd=ROOT) == before_git, "已有Git跟踪状态改变。")
    print({"分类": "NO_PROGRESS", "连续次数": count, "可标记blocked": count >= 3, "记录": str(record_path)})


if __name__ == "__main__":
    main()
