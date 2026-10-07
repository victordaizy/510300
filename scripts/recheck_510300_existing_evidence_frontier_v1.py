"""每个新目标续轮最多调用一次，发现输入变化就停止累计无进展。"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from record_510300_existing_evidence_frontier_v1 import MAIN, ROOT, save, source_extent


def main(previous_number, current_number):
    if current_number != previous_number + 1:
        raise ValueError("只能记录紧接上一目标轮次的一次检查。")
    target = MAIN / f"continuation_checks/check_{current_number:02d}.json"
    if target.exists():
        raise RuntimeError("本目标轮次已记录，不能重复增加次数。")
    previous_path = MAIN / f"continuation_checks/check_{previous_number:02d}.json"
    previous = json.loads(previous_path.read_text(encoding="utf-8"))
    status_path = MAIN / "current_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["same_condition_consecutive_no_progress_goal_turns"] == previous_number
    assert status["active_blocker_id"] == previous["blocker_id"]
    mandate = json.loads((ROOT / "config/510300_existing_data_training_mandate_v1.json").read_text(encoding="utf-8-sig"))
    changes, sources = [], []
    for item in previous["candidate_route_evidence"]:
        path = ROOT / item["path"]
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            changes.append({"kind": "研究结果变化", "path": item["path"]})
    for prior in previous["latest_local_source_extents"]:
        current = source_extent(prior["path"], [prior["field"]])
        sources.append(current)
        if current != prior:
            changes.append({"kind": "已用资料变化", "previous": prior, "current": current})
    observed = datetime.fromisoformat(previous["checked_at"]).timestamp()
    # 仅补查一级新增入口，不对大目录或目录联接做全盘递归扫描。
    for folder in [ROOT / "reports/research", ROOT / "data/raw", ROOT / "data/curated"]:
        for path in folder.iterdir():
            if path == MAIN or not path.is_dir():
                continue
            if path.stat().st_mtime > observed:
                changes.append({"kind": "一级目录更新待核对", "path": str(path.relative_to(ROOT))})
    command = (
        "$task = Get-ScheduledTask -TaskName 'Codex-510300-Primary-Market-Collector' -ErrorAction SilentlyContinue; "
        "$jobs = @(Get-CimInstance Win32_Process | Where-Object { $_.Name -Match '^python(w)?\\.exe$' -and "
        "$_.CommandLine -Match 'rr3|state_mechanism|daily_state_v3|macro_micro_prediction|remaining_holding_value|equity_support_policy|priority_forward' } "
        "| Select-Object ProcessId,Name); "
        "[pscustomobject]@{task_found=($null -ne $task); task_state=[string]$task.State; task_enabled=$task.Settings.Enabled; matching_processes=$jobs} "
        "| ConvertTo-Json -Depth 4 -Compress"
    )
    probe = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
                           capture_output=True, text=True, encoding="utf-8", check=True, timeout=45,
                           creationflags=subprocess.CREATE_NO_WINDOW)
    runtime = json.loads(probe.stdout)
    if mandate["new_market_data_collection_enabled"] or not runtime["task_found"] or runtime["task_enabled"] is not False or runtime["matching_processes"]:
        changes.append({"kind": "授权或运行状态变化", "runtime": runtime})
    timestamp = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    if changes:
        observation = {"checked_at": timestamp, "status": "CHANGE_REQUIRES_INSPECTION_COUNT_NOT_INCREMENTED", "changes": changes}
        save(MAIN / f"continuation_checks/change_observation_{current_number:02d}.json", observation)
        print(json.dumps(observation, ensure_ascii=False, indent=2))
        return
    record = {
        **previous,
        "checked_at": timestamp,
        "previous_goal_turn": previous["current_goal_turn"],
        "current_goal_turn": "NO_PROGRESS_SAME_EVIDENCE_BLOCKER_RECONFIRMED",
        "same_condition_consecutive_no_progress_goal_turns": current_number,
        "latest_local_source_extents": sources,
        "candidate_route_evidence": previous["candidate_route_evidence"],
        "scheduler_observed_this_goal_turn": runtime,
        "matching_research_processes_observed_this_goal_turn": runtime["matching_processes"],
        "source_or_result_changes": [],
        "new_fits": 0, "new_accounts": 0, "new_event_paths": 0,
        "goal_achieved": False,
        "goal_remains_active": True,
        "blocked_threshold_reached": current_number >= 3,
    }
    save(target, record)
    status.update({"updated_at": timestamp, "latest_goal_turn_classification": record["current_goal_turn"],
                   "same_condition_consecutive_no_progress_goal_turns": current_number,
                   "latest_continuation_check": str(target.relative_to(ROOT)), "goal_achieved": False})
    save(status_path, status)
    gap_path = MAIN / "后续证据缺口.md"
    gap = gap_path.read_text(encoding="utf-8")
    previous_text = f"当前记录为本次新的连续无进展检查第{previous_number}轮"
    current_text = f"当前记录为本次新的连续无进展检查第{current_number}轮"
    if previous_text in gap:
        gap_path.write_text(gap.replace(previous_text, current_text, 1), encoding="utf-8")
    print(json.dumps({"本轮分类": record["current_goal_turn"], "连续轮次": current_number,
                      "研究结果和数据变化": [], "当前运行状态": runtime, "目标完成": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="仅在新目标续轮核对既有证据是否变化")
    parser.add_argument("--previous", type=int, required=True)
    parser.add_argument("--current", type=int, required=True)
    args = parser.parse_args()
    main(args.previous, args.current)
