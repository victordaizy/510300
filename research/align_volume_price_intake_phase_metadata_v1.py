"""隔离纠正R190阶段计数：保留R189金融计数和原状态快照，不动策略。"""
from pathlib import Path
import sys

ROOT = Path(__file__).absolute().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.point_first_passage_study_v1 import read, write_json, digest, now, require
from research.finalize_downtrend_break_state_v1 import STATE, FORWARD
from research.close_volume_price_constituent_intake_v1 import OUT


def run():
    receipt_path = OUT / "phase_metadata_alignment_receipt.json"
    require(not receipt_path.exists(), "阶段计数已纠正，不重复。")
    state = read(STATE)
    require(state["latest_technical_decision"] == "TECH.R190" and state["latest_actual_model_decision"] == "TECH.R189",
            "状态已进入不同阶段，不修改。")
    original_hash = digest(STATE)
    original_forward = {k: state[k] for k in FORWARD}
    original_financial = {k: state[k] for k in ("actual_candidate_trials", "latest_financial_strategy_result", "latest_long_point_metrics")}
    backup = OUT / "initial_state_before_intake_phase_metadata_alignment.json"
    with backup.open("xb") as handle:
        handle.write(STATE.read_bytes())
    state.update({
        "current_phase_source_freeze_count": len(read(OUT / "protocol.json")["sources"]),
        "current_phase_known_daily_rows": 0, "current_phase_original_state_rows": 0,
        "current_phase_required_directions": [], "current_phase_required_tests": 0,
        "current_phase_definition_browsing": "NONE_LOCAL_OLD_COMPLETE_USES_AND_CURRENT_TWO_RAW_SOURCE_HASHES",
        "current_phase_information_scope": "SIX_PRIOR_USES_AND_TWO_UNCHANGED_INCOMPLETE_RAW_SOURCES_ONLY",
        "necessary_tests_passed_in_current_phase": 0, "new_accounts_in_current_phase": 0,
        "current_phase_financial_candidate_configurations": 0,
        "latest_phase_metadata_alignment_receipt": receipt_path.relative_to(ROOT).as_posix(),
        "latest_continuation_receipt": receipt_path.relative_to(ROOT).as_posix(),
        "phase_metadata_alignment": "R190是0测试/0账户的有限资料核对；同一目标轮的12测试/4新账户属于R187—R189，原实际金融和根目标轮计数不改。",
        "updated_at": now()})
    state["code_files_added_this_continuation"] += ["research/close_volume_price_constituent_intake_v1.py",
                                                  Path(__file__).absolute().relative_to(ROOT).as_posix()]
    require({k: state[k] for k in FORWARD} == original_forward, "原前瞻改变。")
    require({k: state[k] for k in original_financial} == original_financial, "原R189实际金融证据改变。")
    require(digest(STATE) == original_hash, "状态同时被修改，不覆盖。")
    write_json(STATE, state)
    write_json(receipt_path, {"at": now(), "status": "PASS_R190_PHASE_METADATA_ALIGNED_WITH_ORIGINAL_FINANCIAL_EVIDENCE_PRESERVED",
        "initial_state_snapshot": backup.relative_to(ROOT).as_posix(), "initial_state_sha256": original_hash,
        "cause": "R190改阶段后继承R189的当前阶段测试/状态行数/冻结源数；独立纠正为真实资料核对计数，原R190代码及回执保留。",
        "current_phase_tests": 0, "current_phase_accounts": 0, "root_goal_turn_tests": 12,
        "root_goal_turn_primary_accounts": 4, "new_account_reruns": 0, "strategy_changes": 0,
        "original_financial_metrics_unchanged": True, "forward_values_unchanged": original_forward,
        "sources": [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)}
                    for p in [Path(__file__).absolute(), backup, OUT / "project_state_update_receipt.json", STATE]]}, exclusive=True)
    print("R190阶段元数据归零且原快照保留；本轮12测试/4账户、R189金融及前瞻证据未改。", flush=True)


if __name__ == "__main__":
    run()
