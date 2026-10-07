"""将首次边界实验的既有结果更新到技术线事实入口，不改变其他分支或冻结策略。"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research.point_first_passage_study_v1 import OUT, now, read, write_json

STATE = ROOT / "reports/research/510300_daily_weekly_goal_continuation_20261001/state.json"
FORWARD_KEYS = ["forward_protocol", "forward_registry", "new_prospective_observations", "earliest_future_exchange_session",
                "registered_candidate_intents", "new_prospective_completed_points", "next_new_close_eligible_at",
                "current_validated_candidates", "forward_account_comparison_protocol", "latest_forward_account_check",
                "new_prospective_sessions_this_continuation", "new_prospective_cycles_this_continuation"]


def main():
    receipt = OUT / "project_update_receipt.json"
    if receipt.exists():
        raise ValueError("本结果已更新项目状态，不能重复追加。")
    result = read(OUT / "summary.json")
    verification = read(OUT / "saved_result_verification.json")
    if verification["status"] != "PASS_SAVED_MODEL_CLOCK_POINT_AND_ACCOUNT_RECOMPUTATION":
        raise ValueError("实际结果尚未核对完成。")
    state = read(STATE)
    before = copy.deepcopy(state)
    snapshot = OUT / "state_before_TECH_R158.json"
    with snapshot.open("x", encoding="utf-8") as handle:
        handle.write(STATE.read_text(encoding="utf-8-sig"))
    forward = {key: copy.deepcopy(state.get(key)) for key in FORWARD_KEYS}
    progress = ("2026-10-04完成TECH.R157—R158首次边界技术学习及完整账户：1固定模型配方、142月度拟合、3472成熟参考原点、"
                "4原A精确对照和8新政策账户、6必要测试。压力成本较早6次、净年化−1.35%/夏普−0.578/实际pB0.192；"
                "近期5次、−0.49%/−0.320/pB0.138，胜率60%而实际B0.231。两期毛损益也负，未触发账户停机；"
                "全成熟原点预测误差均差于同池频率。四场景经济及稳定性门失败，固定模型关闭；全部11真实进出及事前指标保存。"
                "0新准入待跑，收益夏普未提高，目标active；独立验证/去过拟合未成立，原失败和前瞻保持。")
    state.update({"at": now(), "status": "research_active", "goal_status": "active", "goal_achieved": False,
                  "current_phase": "COMPLETED_FIXED_FIRST_PASSAGE_TECHNICAL_MODEL_FULL_ACCOUNT_REJECTION",
                  "latest_technical_decision": "TECH.R158", "latest_registration_decision": "TECH.R157",
                  "latest_actual_model_decision": "TECH.R158", "latest_model_decision": "TECH.R158",
                  "latest_actual_prediction_model_decision": "TECH.R158", "latest_prediction_technical_decision": "TECH.R158",
                  "latest_financial_strategy_decision": "TECH.R158",
                  "original_exit_model_latest_decision": "TECH.R145",
                  "original_exit_model_latest_result": before.get("latest_actual_prediction_result"),
                  "latest_completed_study": OUT.relative_to(ROOT).as_posix(),
                  "latest_result": (OUT / "summary.json").relative_to(ROOT).as_posix(),
                  "latest_actual_prediction_result": (OUT / "summary.json").relative_to(ROOT).as_posix(),
                  "latest_financial_strategy_result": (OUT / "summary.json").relative_to(ROOT).as_posix(),
                  "new_account_return_sharpe_result": (OUT / "summary.json").relative_to(ROOT).as_posix(),
                  "latest_report": (OUT / "研究结果与下一步.md").relative_to(ROOT).as_posix(),
                  "latest_research_report": (OUT / "研究结果与下一步.md").relative_to(ROOT).as_posix(),
                  "latest_research_status": result["status"], "latest_progress": progress,
                  "latest_continuation_receipt": (OUT / "summary.json").relative_to(ROOT).as_posix(),
                  "latest_first_passage_saved_verification": (OUT / "saved_result_verification.json").relative_to(ROOT).as_posix(),
                  "new_model_fits_this_continuation": 142, "historical_model_refits_this_continuation": 142,
                  "baseline_model_recomputation_fits_this_continuation": 0, "new_model_training_labels_this_continuation": 3472,
                  "new_financial_candidate_accounts_this_continuation": 8, "new_market_requests_this_continuation": 0,
                  "original_frozen_models_changed": False, "returns_and_sharpe_improved": False,
                  "whole_model_overfitting_removed": False, "independent_validation_status": "NOT_ESTABLISHED",
                  "current_goal_turn_classification": "PROGRESS_NEW_ACTUAL_MODEL_AND_FULL_ACCOUNT_NEGATIVE_EVIDENCE",
                  "goal_turn_classification": "PROGRESS_NEW_ACTUAL_MODEL_AND_FULL_ACCOUNT_NEGATIVE_EVIDENCE",
                  "goal_turn_progress_classification": "NEW_FINANCIAL_EVIDENCE_FIXED_CANDIDATE_REJECTED",
                  "current_goal_turn_classification_reason": "不同事件成熟标签的一次实际模型和八账户检验，得到全部点位、信号/费用/约束分解；不是仅来源或统计整理，也没有提高收益。",
                  "current_goal_turn_actual_work": {"new_policy_accounts": 8, "primary_accounts": 4, "frequency_controls": 4,
                                                   "original_A_control_replays": 4, "new_model_fits": 142,
                                                   "new_model_recipes": 1, "derived_reference_rows": 3488,
                                                   "mature_reference_origins": 3472, "actual_pressure_entry_points": 11,
                                                   "new_market_requests": 0, "necessary_tests_passed": 6},
                  "blocking_decision": None, "consecutive_blocked_goal_turns": 0,
                  "latest_user_direction_feedback": "你做的偏了",
                  "direction_feedback_specific_meaning": "PENDING_OPTIONAL_USER_CLARIFICATION",
                  "next_financial_experiment": "NOT_REGISTERED_NO_READY_CANDIDATE",
                  "next_research_action": "根据用户偏离说明选定点位解释或原A改进；先用保存图谱和完整实际点位，不重做已完成上涨图谱或后验筛选本轮交易。",
                  "latest_goal_tool_status_receipt": (OUT / "goal_tool_status.json").relative_to(ROOT).as_posix()})
    write_json(STATE, state)
    actual = read(STATE)
    if {key: actual.get(key) for key in FORWARD_KEYS} != forward:
        raise AssertionError("另一阶段的前瞻状态被改变。")

    banner = "> 技术线最新实际模型及金融裁决："+progress+" [实际点位与完整报告](../reports/research/510300_point_first_passage_study_v1/研究结果与下一步.md)。\n\n"
    files = ["docs/PROJECT_STATE.md", "docs/RESEARCH_DECISIONS.md", "docs/PROJECT_STATE_TECHNICAL_LINE.md", "docs/RESEARCH_DECISIONS_TECHNICAL_LINE.md"]
    common = ("\n\n## 2026-10-04：技术线首次边界模型实际裁决（TECH.R157—R158）\n\n"+progress+"\n\n"
              "本结果只更新日周线技术点位任务。其他宏观/盘口任务的目标、状态与授权保持，原A和全部冻结策略不改。最新实际预测模型、金融策略与技术决策现在为TECH.R158；原退出增量最后裁决TECH.R145单独保留，有限选择诊断TECH.R156也是已完成历史事实。\n\n"
              "用户最新指出方向偏离，已提出可选澄清；具体含义未获得回答。目前以原明确点位与完整账户目标推进，不据此编造新指标组合或已冻结待跑策略。\n\n"
              "直接依据：[唯一协议](../reports/research/510300_point_first_passage_study_v1/protocol.json)、"
              "[实际结果](../reports/research/510300_point_first_passage_study_v1/summary.json)、"
              "[全部真实点位](../reports/research/510300_point_first_passage_study_v1/results/实际进出点位与事前指标.csv)、"
              "[完整报告](../reports/research/510300_point_first_passage_study_v1/研究结果与下一步.md)。\n")
    decisions = ("\n\n**假设**：日线/前完整周MACD及量价波动能预测首次盈利、亏损或到期事件，其概率与同池实际类别损益能识别预计pB>1的进场，形成比原A和无技术特征频率对照更好的账户。\n\n"
                 "**验证方法**：先完成六项必要测试及四原A精确对照，冻结35来源；唯一多项Logistic配方、八项特征、756原点窗口，最少252成熟行/每类10行，当前成熟池逆并发权重、每月拟合。原点ATR14、次开锚定下1/上2ATR、20收盘到期、次合法开出；模型只用实际成熟结果。两个完整时期两费用，独立模型与频率两政策共八账户，直接跑账户，不用原退出MSE门阻止。\n\n"
                 "**结果**：142实际拟合，3472成熟参考原点（并非独立事件），2856原点的两用途预测。压力成本原A较早年化1.84%/夏普0.438/22次，模型−1.35%/−0.578/6次；近期原A3.99%/1.217/32次，模型−0.49%/−0.320/5次。模型实际pB0.192/0.138，近期胜率60%、B0.231；全成熟原点logloss和Brier两期均差于同池频率。频率对照全现金，夏普及交易质量不可估计。四场景整体及稳定性门失败；毛损益两期也负、无回撤停机。11实际点位/5712可用预测/八保存账户已核对，最大金额误差3.50e−11元。\n\n"
                 "**为什么接受/拒绝**：接受不同成熟事件标签的执行定义与实际结果，拒绝这一固定模型能改善收益、夏普或增次质量。预计pB不等于实现pB，计划2R也不等于实际B；不能把预测指标、142拟合或六测试作为策略有效/独立/去过拟合证据。旧类别、入场回报、退出模型拒绝保持；本结果也不否定整个技术分析。\n\n"
                 "**是否需要重新验证**：本固定配方终态关闭，不改指标/窗口/边界/到期/正则/准入/成本/时期营救。历史开发、供应首版未认证、独立证据和全局DSR/PBO缺失保持。用户的具体偏离说明将指导下一个问题，目前0已冻结待跑新候选；不从本轮亏损点位后验制定盈利过滤。\n")
    for name in files:
        path = ROOT / name
        content = path.read_text(encoding="utf-8-sig")
        if "技术线首次边界模型实际裁决（TECH.R157—R158）" in content:
            raise ValueError("同一事实条目已经存在："+name)
        first_end = content.find("\n")
        addition = common+decisions if "RESEARCH_DECISIONS" in name else common
        path.write_text(content[:first_end+1]+"\n"+banner+content[first_end+1:]+addition, encoding="utf-8")
    write_json(receipt, {"at": now(), "status": "UPDATED_TECHNICAL_STATE_AND_FOUR_FACT_ENTRIES",
                         "latest_decision": "TECH.R158", "updated_documents": files,
                         "state_before": snapshot.relative_to(ROOT).as_posix(), "forward_fields_unchanged": FORWARD_KEYS,
                         "original_frozen_strategies_changed": False, "goal_status": "active", "goal_achieved": False}, exclusive=True)
    print("已更新项目状态与四个长期事实入口；12项前瞻状态保持，原退出模型R145单独保留。", flush=True)


if __name__ == "__main__":
    main()
