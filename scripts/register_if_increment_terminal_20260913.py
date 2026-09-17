"""把已完成IF信息增量实验登记到总索引，不改旧方案及其观察流程。"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
STUDY = ROOT / "reports/research/510300_if_open_interest_increment_v1"
OUT = ROOT / "reports/research/510300_goal_followup_20260913"
FIELD = "supplemental_if_open_interest_increment_v1"


def sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    before_bytes = INDEX.read_bytes()
    before = json.loads(before_bytes)
    if FIELD in before:
        raise FileExistsError("本轮IF结论已经登记，无须再次更新索引")
    result = json.loads((STUDY / "result.json").read_text(encoding="utf-8"))
    verified = json.loads((STUDY / "saved_verification_receipt.json").read_text(encoding="utf-8"))
    if result["status"] != "REJECTED_FROZEN_NO_INCREMENT_UNDER_REGISTERED_TEST" or result["accounts"]:
        raise ValueError("IF终态或未运行账户状态与登记预期不同")
    if verified["new_fits"] or verified["new_accounts"]:
        raise ValueError("保存核对不应产生新研究活动")
    moment = datetime.now(ZoneInfo("Asia/Shanghai"))
    runtime = json.loads((ROOT / "config/510300_new_daily_input_adapter_runtime_v1.json").read_text(encoding="utf-8"))
    calendar = pd.read_csv(ROOT / runtime["calendar"])
    sessions = pd.DatetimeIndex(pd.to_datetime(calendar.trade_date)).sort_values()
    local_day = pd.Timestamp(moment.date())
    complete_days = sessions[(sessions < local_day) | ((sessions == local_day) & ((moment.hour, moment.minute) >= (15, 5)))]
    latest_complete = str(complete_days[-1].date())
    next_day = str(sessions[sessions > complete_days[-1]][0].date())
    state_dir = ROOT / runtime["output_root"]
    latest = json.loads((state_dir / "latest_check.json").read_text(encoding="utf-8"))
    independent_latest_exists = (state_dir / "latest_completed.json").exists()
    observed_input_dates = pd.read_parquet(ROOT / runtime["initial_source"]["prices"], columns=["date"])
    source_cutoff = str(pd.to_datetime(observed_input_dates.date).max().date())

    entry = {
        "registered_at": moment.isoformat(), "study_id": result["study_id"], "status": result["status"],
        "result": (STUDY / "result.json").relative_to(ROOT).as_posix(),
        "result_sha256": sha((STUDY / "result.json").read_bytes()),
        "report": (STUDY / "研究结论与下一步.md").relative_to(ROOT).as_posix(),
        "review_zip": "deliverables/510300_IF持仓信息增量_V1_GPT审阅_20260913.zip",
        "mature_paired_origins": result["evaluation"]["paired_origins"],
        "positive_eras": result["evaluation"]["positive_eras"],
        "relative_mse_improvement": result["evaluation"]["overall"]["relative_mse_improvement"],
        "account_stage": result["account_stage"], "net_sharpe": "NOT_COMPUTED", "net_cagr": "NOT_COMPUTED",
        "disposition": "CLOSED_THIS_REGISTERED_INFORMATION_USE_NO_PARAMETER_RESCUE",
        "existing_fixed_observer_is_a_separate_control": True, "position_impact": 0, "goal_achieved": False,
    }
    after = copy.deepcopy(before)
    after[FIELD] = entry
    after["updated_at"] = moment.isoformat()
    unchanged = copy.deepcopy(after)
    unchanged.pop(FIELD)
    unchanged["updated_at"] = before["updated_at"]
    if unchanged != before:
        raise AssertionError("补充登记意外改动了其他索引字段")
    after_bytes = (json.dumps(after, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "index_before_registration.json").open("xb") as handle:
        handle.write(before_bytes)
    if INDEX.read_bytes() != before_bytes:
        raise RuntimeError("索引被其他任务更新，保留备份并停止；禁止覆盖并发改动")
    temporary = INDEX.with_name(INDEX.name + ".if_registration.tmp")
    with temporary.open("xb") as handle:
        handle.write(after_bytes)
    os.replace(temporary, INDEX)
    restored = json.loads(INDEX.read_text(encoding="utf-8"))
    if restored != after:
        raise AssertionError("登记后回读不一致")
    receipt = {
        "checked_at": moment.isoformat(), "status": "IF_TERMINAL_REGISTERED_EXISTING_CONTROL_AWAITS_NEW_COMPLETE_DAY",
        "previous_goal_turn_class": "PROGRESS_COMPLETED_FROZEN_IF_EXPERIMENT",
        "current_progress": "REGISTERED_COMPLETED_INFORMATION_EXPERIMENT_WITHOUT_CHANGING_CONTROL",
        "index_before_sha256": sha(before_bytes), "index_after_sha256": sha(INDEX.read_bytes()),
        "only_semantic_changes": [FIELD, "updated_at"], "latest_completed_round_preserved": True,
        "existing_next_work_preserved": True, "latest_complete_official_day_now": latest_complete,
        "actual_initial_source_cutoff": source_cutoff, "previous_runtime_check_at": latest["checked_at"],
        "new_completed_daily_result_exists": independent_latest_exists,
        "next_official_trading_day": next_day, "next_useful_check_after": next_day + "T15:05:00+08:00",
        "external_blocker": "NO_NEW_POST_FREEZE_COMPLETE_TRADING_DAY" if latest_complete <= source_cutoff else "NEW_DAY_AVAILABLE_CHECK_EXISTING_RUNTIME",
        "this_thread_external_blocker_first_observed_at": moment.isoformat(),
        "blocker_count_is_not_inherited_from_other_tasks": True,
        "new_labels": 0, "new_models": 0, "new_accounts": 0, "network_requests": 0,
        "goal_achieved": False, "goal_status_changed": False,
    }
    with (OUT / "registration_and_current_state_receipt.json").open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
