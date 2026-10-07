"""将已经完成的修复顺序实验写回事实文件，不运行模型或账户。"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path


ROOT = Path(__file__).absolute().parents[1]
OUT = ROOT / "reports/research/510300_point_fresh_repair_order_study_v1"
STATE = ROOT / "reports/research/510300_daily_weekly_goal_continuation_20261001/state.json"
DOCS = [
    ROOT / "docs/PROJECT_STATE.md",
    ROOT / "docs/RESEARCH_DECISIONS.md",
    ROOT / "docs/PROJECT_STATE_TECHNICAL_LINE.md",
    ROOT / "docs/RESEARCH_DECISIONS_TECHNICAL_LINE.md",
]
FORWARD_KEYS = [
    "forward_protocol", "forward_registry", "new_prospective_observations",
    "earliest_future_exchange_session", "registered_candidate_intents",
    "new_prospective_completed_points", "next_new_close_eligible_at",
    "current_validated_candidates", "forward_account_comparison_protocol",
    "latest_forward_account_check", "new_prospective_sessions_this_continuation",
    "new_prospective_cycles_this_continuation",
]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    receipt_path = OUT / "project_state_update_receipt.json"
    snapshot_path = OUT / "state_before_TECH_R161.json"
    if receipt_path.exists() or snapshot_path.exists():
        raise RuntimeError("本实验状态写回已经开始或完成，禁止重复执行。")
    summary = read_json(OUT / "summary.json")
    verification = read_json(OUT / "saved_result_verification.json")
    protocol = read_json(OUT / "protocol.json")
    tests = read_json(OUT / "tests_receipt.json")
    goal = read_json(OUT / "goal_tool_status.json")
    if summary["technical_decision"] != "TECH.R161" or summary["goal_achieved"]:
        raise RuntimeError("实际裁决与本固定失败写回口径不一致。")
    if verification["status"] != "PASS_SAVED_SEQUENCE_POINT_AND_EIGHT_ACCOUNT_RECOMPUTATION":
        raise RuntimeError("保存结果核对尚未通过。")
    if tests["exit_code"] != 0 or tests["passed"] != 5:
        raise RuntimeError("五项必要测试没有全部通过。")
    if goal["goal_tool_result"]["goal"]["status"] != "active":
        raise RuntimeError("目标工具实际状态不是active。")
    for item in protocol["sources"]:
        if digest((ROOT / item["path"]).read_bytes()) != item["sha256"]:
            raise RuntimeError("冻结来源已变化：" + item["path"])

    originals = {p: p.read_bytes() for p in [STATE, *DOCS]}
    state = json.loads(originals[STATE].decode("utf-8-sig"))
    preserved = {k: state[k] for k in FORWARD_KEYS}
    original_prediction = state["latest_actual_prediction_model_decision"]
    original_exit = {
        k: state[k] for k in ["original_exit_model_latest_decision", "original_exit_model_latest_result"]
    }
    with (OUT / "results/完整账户共同口径比较.csv").open(encoding="utf-8-sig", newline="") as f:
        pressure = [r for r in csv.DictReader(f) if r["cost"] == "STRESS" and r["policy"] == "FRESH_ORDERED_REPAIR"]
    if len(pressure) != 2 or sum(int(r["completed_cycles"]) for r in pressure) != 9:
        raise RuntimeError("九个实际压力周期或两个完整时期不一致。")
    at = datetime.now(timezone(timedelta(hours=8))).isoformat()
    out_rel = "reports/research/510300_point_fresh_repair_order_study_v1"
    progress = (
        "2026-10-04按用户先解释上涨再反推的顺序完成TECH.R160登记/R161实际账户。"
        "仅一个当前仍有效的量正段→日MACD正段→价格确认机制；5必要测试、4原A精确对照、8新政策账户，0拟合/新训练标签/采集。"
        "压力较早2笔、胜率50%、实际pB2.291、净年化1.204%/夏普0.576；近期7笔、胜率14.29%、pB2.513、0.648%/0.241。"
        "点值四场景均超过纯价格对照，未同时超过原A收益夏普，整体及稳定性门全败。"
        "仅两笔赢家，分别占本时期净损益124.22%/145.86%；完整年次数0.4/1.0。"
        "2020早修复实际04-08入/04-14出净−0.45%，不能把后段上涨归入该交易。"
        "全部9点、A库存及已知目标已保存；2024-09-24A库存虽零但目标31.92%，不认定组合新增收益。"
        "固定完整政策拒绝并结束，不改参数或退出救回。最新金融裁决R161；实际预测模型R158、原退出R145保留。"
        "下一先解释修复至后段加速及全部失败/再次确认路径，金融实验未登记、待跑0。"
        "收益夏普目标、独立验证与去过拟合未达，目标active、阻塞计数0，原前瞻和全部旧失败保持。"
    )
    trial_accounting = {
        "scope": "TECH.R160_R161_SINGLE_FIXED_CURRENT_REPAIR_ORDER_COMPLETE_POLICY",
        "fixed_new_entry_mechanisms": 1, "complete_policy_configs_including_price_control": 2,
        "new_policy_accounts": 8, "primary_accounts": 4, "price_confirmation_controls": 4,
        "original_A_control_replays": 4, "new_model_fits": 0, "new_training_labels": 0,
        "all_origin_sequence_rows": 3488, "reused_mature_price_events": 186,
        "unique_actual_pressure_cycles": 9, "new_market_requests": 0,
        "necessary_tests_passed": 5, "saved_accounts_recomputed": 8,
    }
    for k, v in list(state.items()):
        if k.endswith("_this_continuation") and not k.startswith("prior_"):
            if isinstance(v, bool):
                state[k] = False
            elif isinstance(v, (int, float)):
                state[k] = 0
    state.update({
        "at": at, "updated_at": at, "status": "research_active", "goal_achieved": False,
        "latest_completed_study": out_rel, "latest_result": out_rel + "/summary.json",
        "latest_report": out_rel + "/研究结果与下一步.md", "latest_research_report": out_rel + "/研究结果与下一步.md",
        "latest_research_status": summary["status"], "latest_progress": progress,
        "latest_overall_summary": progress, "latest_continuation_outcome": progress,
        "latest_continuation_receipt": out_rel + "/summary.json",
        "latest_technical_decision": "TECH.R161", "latest_registration_decision": "TECH.R160",
        "latest_model_decision": "TECH.R161", "latest_actual_model_decision": "TECH.R161",
        "latest_actual_model_kind": "COMPLETE_RULE_POLICY_WITHOUT_ESTIMATED_PREDICTION_MODEL",
        "latest_financial_strategy_decision": "TECH.R161", "latest_financial_strategy_result": out_rel + "/summary.json",
        "latest_fresh_repair_order_saved_verification": out_rel + "/saved_result_verification.json",
        "latest_fresh_repair_order_protocol": out_rel + "/protocol.json",
        "latest_fresh_repair_order_points": out_rel + "/results/全部实际点位_量MACD价格顺序及A持仓.csv",
        "new_account_return_sharpe_result": out_rel + "/summary.json",
        "latest_goal_tool_status_receipt": out_rel + "/goal_tool_status.json", "goal_tool_status_confirmed": "active",
        "current_study": summary["study"], "current_phase": "CASE_DERIVED_FIXED_REPAIR_ORDER_COMPLETED_REJECTED",
        "continuation_program_role": "ORIGINAL_FORWARD_CLOCK_ROUTER_NOT_CURRENT_HISTORICAL_EXPERIMENT",
        "current_direction": "当前修复顺序完整政策拒绝；先解释后段加速/再次确认，下一金融实验未登记。",
        "current_priority": "回到具体路径，解释全体修复/失败/再次确认/加速以及原A覆盖，形成实质不同机制后另立唯一合同。",
        "current_unmet_evidence": "真实pB点值过门但仅2笔赢家、次数少；收益夏普未同时超过原A，四整体门及稳定性门失败；独立和去过拟合未建立。",
        "next_available_action": "先解释全部路径中的再次确认及A覆盖，不延长R161退出、不按两笔赢家调整仓位、不直接叠加两账户。",
        "next_experiment_status": "DIFFERENT_LATER_STAGE_MECHANISM_EXPLANATION_PROPOSED_NOT_REGISTERED_NOT_RUN",
        "next_financial_experiment": "NOT_REGISTERED_NO_READY_CANDIDATE",
        "next_strategy_increment_status": "NO_NEW_ADMITTED_UNRUN_NUMERIC_STRATEGY",
        "current_admitted_unrun_numeric_candidates": 0,
        "new_accounts_this_continuation": 8, "new_accounts_in_current_phase": 8,
        "new_investment_account_evaluations_this_continuation": 8, "new_strategy_accounts_this_continuation": 8,
        "new_financial_candidate_accounts_this_continuation": 8, "saved_account_controls_replayed_this_continuation": 4,
        "saved_accounts_checked_this_continuation": 8, "saved_field_rows_recomputed_this_continuation": 3488,
        "saved_account_overlap_tables_read_this_continuation": 1, "necessary_tests_passed_this_continuation": 5,
        "necessary_tests_passed_in_current_phase": 5, "latest_source_freeze_count": 39,
        "source_freeze_count_this_continuation": 39, "empirical_strategy_configurations_this_continuation": 2,
        "new_strategy_configurations_this_continuation": 2, "source_purpose_and_clock_bound_this_continuation": True,
        "current_member_support_this_continuation": "全部3488原点已知状态完整保留，原186结果仅解释，9实际周期不由成熟标签门筛入。",
        "new_network_requests_this_continuation": 0, "literature_lookup_this_continuation": {"page_open_attempts": 0, "paper_find_requests": 0, "market_data_requests": 0},
        "validation_method_this_continuation": "5必要测试、4原A精确对照、一次8政策账户；3488顺序/3案例前缀/9点时钟和8保存账户核对通过；金融复跑0。",
        "current_study_prediction_gate_status": "NOT_APPLICABLE_FIXED_RULE_POLICY",
        "economic_stage_status": "COMPLETED_FIXED_COMPLETE_POLICY_REJECTED",
        "account_return_sharpe_this_continuation": "COMPUTED_AND_FIXED_COMPLETE_POLICY_REJECTED",
        "account_return_sharpe_reason": "较早净年化低于A，近期净年化与夏普均低于A；全部整体门及历史稳定性门失败。",
        "actual_candidate_trials": trial_accounting, "current_phase_trial_accounting": trial_accounting,
        "current_goal_turn_actual_work": trial_accounting, "current_structural_signal_events": 9,
        "current_executed_unique_new_entry_events": 9,
        "latest_fresh_repair_pressure_point_metrics": pressure,
        "current_full_account_economic_gate_passed": False, "returns_and_sharpe_improved": False,
        "return_and_sharpe_improved_this_continuation": False,
        "independent_validation_status": "NOT_ESTABLISHED", "whole_model_overfitting_removed": False,
        "overfit_removed": False, "overfitting_removed": False, "global_DSR_PBO": "NOT_COMPUTED",
        "previous_goal_turn_classification": "progress",
        "previous_goal_turn_classification_reason": "R158实际模型负结果加R159具体路径与186事件解释，形成真实研究进展。",
        "current_goal_turn_classification": "PROGRESS_NEW_COMPLETE_CASE_DERIVED_POLICY_EVIDENCE",
        "current_goal_turn_classification_reason": "不同当前修复顺序一次实际完整政策/8账户，得到9真实进出与集中度，失败结果仍是研究进展，未提高目标收益。",
        "goal_turn_progress_classification": "NEW_FINANCIAL_EVIDENCE_FIXED_CANDIDATE_REJECTED",
        "blocked_audit_count": 0, "consecutive_blocked_goal_turns": 0,
        "blocked_reason": None, "blocked_audit_key": None, "blocking_decision": None,
        "live_own_process_handle": None, "verified_wait": False,
        "original_strategy_source_files_changed_this_continuation": 0,
        "code_files_added_this_continuation": [
            "research/point_fresh_repair_order_inputs_v1.py", "research/point_fresh_repair_order_account_v1.py",
            "research/point_fresh_repair_order_study_v1.py", "research/point_fresh_repair_order_closeout_v1.py",
            "tests/test_point_fresh_repair_order_v1.py", "research/finalize_point_fresh_repair_order_state_v1.py",
        ],
        "report_programs_added_this_continuation": ["research/point_fresh_repair_order_closeout_v1.py", "research/finalize_point_fresh_repair_order_state_v1.py"],
    })
    if {k: state[k] for k in FORWARD_KEYS} != preserved:
        raise RuntimeError("原十二项真实前瞻字段发生变化。")
    if state["latest_actual_prediction_model_decision"] != original_prediction or {k: state[k] for k in original_exit} != original_exit:
        raise RuntimeError("原实际预测模型或原退出裁决被覆盖。")

    banner = "> 技术线最新金融结果 TECH.R161（2026-10-04，优先于下方历史摘要）：" + progress + " [完整报告](../" + out_rel + "/研究结果与下一步.md)。\n\n"
    section = """
## TECH.R160：从具体上涨登记当前修复顺序（2026-10-04）

**假设**：均线下修复中，当前仍有效的量正段先开始、日MACD正段随后开始、价格最后上穿EMA20，可能给出高盈亏比点位并提高完整账户。这里研究当前连续正段的实际出生，而非从事后低点/整个旧区间首次见到正值；日线量为经济回报方向加权的五日成交量近似，不是逐笔主动买卖量。

**验证方法**：先从R159具体上涨解释及全部186原事件形成机制，复核旧独立触发、旧压缩/修复、V3已失败状态学习和价量相关定义，不当作全项目穷尽新颖性证明。沿用原EMA20、MACD12/26/9、量5及已完成周背景/130周预热。当前量和MACD正段须由此前实际非正值转入且持续至确认，严格量先于MACD、MACD先于价格；不接受未知当出生或同日/倒序。空仓次真实开尝试进；收盘重新<=EMA20，次合法开退出，原风险只减仓。没有新增2R或20日到期。5测试与4原A精确对照后、金融运行前冻结39来源及唯一合同。新顺序/纯价格两完整政策×两时期×两费用共8账户，资金20万元、原风险、T+1及费用保持；不与A混合，不用原退出MSE门阻止不同用途的账户检验。

**结果及处置**：登记及当前出生时钟合格，只准一次固定历史完整用途；3案例的量/MACD/价格日期分别2019-01-04/01-08/01-09、2020-03-25/04-01/04-07、2024-09-19/09-23/09-24。已查看过的历史仍DEVELOPMENT_CALIBRATION，不能登记成独立新样本。实际接受/拒绝见R161，未接受可执行新策略。

**是否需要重新验证**：登记已完成且实际运行，不能复跑登记、回溯改入场定义或改名重开旧失败；不单凭3上涨案例认定胜率。下一不同机制须在全部成功及失败路径先解释，再另立唯一完整合同。

## TECH.R161：当前修复顺序实际完整账户结果（2026-10-04）

**假设**：R160固定完整政策能在两时期、两成本同时提高相对原A和纯价格对照的净年化/净夏普，并满足实际pB>1、标准净期望>0及回撤限制；频率软目标。

**验证方法**：8新政策账户一次实际运行、4原A精确对照；0预测拟合/新训练标签/市场采集。原186成熟价格事件的固定20日结果仅解释，未用于筛实际交易。保存所有3488原点当时状态、全部9实际压力周期、资金订单和A库存/已知目标；20/252日各2000配对区间描述，不算独立验证。原A/全部冻结策略源文件不改。

| 压力成本时期 | 完整政策 | 净年化 | 净夏普 | 最大回撤 | 完成次数 | 胜率 | 实际B | 实际pB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 2015—2019 | 原A | 1.84% | 0.438 | 6.75% | 22 | 50.00% | 1.222 | 0.611 |
| 2015—2019 | 当前修复顺序 | 1.20% | 0.576 | 1.82% | 2 | 50.00% | 4.582 | 2.291 |
| 2015—2019 | 纯价格确认 | 0.49% | 0.167 | 9.59% | 74 | 25.68% | 2.697 | 0.692 |
| 2020—2026-09-30 | 原A | 3.99% | 1.217 | 2.75% | 32 | 62.50% | 1.717 | 1.073 |
| 2020—2026-09-30 | 当前修复顺序 | 0.65% | 0.241 | 5.04% | 7 | 14.29% | 17.591 | 2.513 |
| 2020—2026-09-30 | 纯价格确认 | −1.30% | −0.452 | 9.17% | 112 | 18.75% | 3.256 | 0.611 |

**结果**：四场景顺序政策的收益/夏普点值均超过纯价格、实际pB均>1，但较早年化低于A、近期年化和夏普均低于A，全部经济及历史稳定性门失败。较早/近期完整年平均完成0.4/1.0次、分别3/2个无交易年，未触发回撤停机。两时期各仅一个赢家，其利润分别占该时期净损益124.22%/145.86%。2020-04-08进、04-14出净−0.45%，识别早修复未捕捉后段；2019-01-10至03-27、2024-09-25至11-18实际分别+17.89%/+15.62%，不是事后低至高涨幅。旧20日标签9件胜率55.56%/pB1.513不等于真实近期14.29%/pB2.513。3488顺序、3案例前缀、9时钟及8保存账户核对通过，资金最大误差4.70e−11元，39冻结来源未变。

**为什么接受/拒绝**：接受当前修复能用当时信息识别、有限点位质量及稀缺/盈利集中事实；拒绝该完整独立政策已经改善原A收益夏普，固定政策终态关闭。不把较弱价格对照的点值改进、高pB或单个大赢家宣称为稳定未来优势。原A在两个较早点均空仓、近期7点已持有2/空仓5；2024-09-24虽库存0，但已知目标31.92%，不把库存空解释为无入场意图，更不直接相加两账户收益。未检验新组合，目标未达。

**是否需要重新验证/下一步具体实验**：本固定完整政策不重跑、不改顺序/窗口/阈值/退出/成本/样本或加过滤救回。下一先解释早修复→失败或持续→再次确认→后段加速的完整路径，以及原A是否覆盖。保留全部186原事件及失败，不只选2020或2赢家；已有指标原值按事前时钟逐日展示，事后加速仅作结果解释，不进入识别变量。形成实质不同机制后才固定一个完整金融实验；当前未登记、待跑0。原真实前瞻十二关键字段及10-08 15:05首次合格时钟保持；独立验证/去过拟合仍未建立，全局DSR/PBO未算，0实盘授权。

**指针范围**：最新完整金融政策及技术裁决R161；最新实际预测模型仍R158，原退出增量最后裁决R145单独保留。当前目标工具active、阻塞计数0，本轮是实际负结果研究进展；其他宏观/盘口任务的状态及授权保持。

依据：[完整报告](../reports/research/510300_point_fresh_repair_order_study_v1/研究结果与下一步.md)、[唯一合同](../reports/research/510300_point_fresh_repair_order_study_v1/protocol.json)、[实际裁决](../reports/research/510300_point_fresh_repair_order_study_v1/summary.json)、[九个真实点位与A已知目标](../reports/research/510300_point_fresh_repair_order_study_v1/results/全部实际点位_量MACD价格顺序及A持仓.csv)、[保存结果核对](../reports/research/510300_point_fresh_repair_order_study_v1/saved_result_verification.json)。
"""
    replacements = {STATE: (json.dumps(state, ensure_ascii=False, indent=2) + "\n").encode("utf-8")}
    for path in DOCS:
        text = originals[path].decode("utf-8-sig")
        if "## TECH.R160：" in text or "## TECH.R161：" in text:
            raise RuntimeError("事实文档已经包含本批裁决，禁止重复追加。")
        first, rest = text.split("\n", 1)
        replacements[path] = (first + "\n\n" + banner + rest + "\n" + section).encode("utf-8")
    for path, data in originals.items():
        if path.read_bytes() != data:
            raise RuntimeError("另一工作改变了同一事实文件，禁止覆盖：" + str(path))
    snapshot_path.write_bytes(originals[STATE])
    for path, data in replacements.items():
        with path.open("wb") as f:
            f.write(data)
    updated = read_json(STATE)
    if {k: updated[k] for k in FORWARD_KEYS} != preserved:
        raise RuntimeError("写回后真实前瞻字段改变。")
    for path in DOCS:
        if originals[path].decode("utf-8-sig").split("\n", 1)[1] not in path.read_bytes().decode("utf-8-sig"):
            raise RuntimeError("原事实文档正文未完整保留：" + str(path))
    receipt = {
        "at": at, "status": "PASS_R161_STATE_AND_FOUR_FACT_FILES_SINGLE_UPDATE",
        "scope": "DAILY_WEEKLY_TECHNICAL_LINE_ONLY",
        "latest_financial_decision": "TECH.R161", "latest_actual_prediction_decision_preserved": original_prediction,
        "original_exit_decision_preserved": original_exit, "forward_twelve_fields_preserved": preserved,
        "goal_tool_status": "active", "goal_achieved": False, "new_account_runs_or_fits": 0,
        "files": [{"path": str(p.relative_to(ROOT)), "before_sha256": digest(originals[p]), "after_sha256": digest(p.read_bytes())} for p in originals],
        "unchanged_frozen_sources": len(protocol["sources"]), "original_state_snapshot": str(snapshot_path.relative_to(ROOT)),
    }
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("已一次写回TECH.R161及四事实文件；原预测R158/退出R145和十二前瞻字段保持，目标active未完成，无新金融运行。")


if __name__ == "__main__":
    main()
