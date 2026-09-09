"""保存已完成的第二十二轮，并按用户最新要求停止来源补齐。"""
from pathlib import Path
import json
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]

def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))

def write(path, value):
    (ROOT / path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def main():
    now = datetime.now().astimezone().isoformat()
    index_path = "reports/research/510300_sharpe_1_2_latest_research.json"
    index = read(index_path)
    paths = ["reports/research/510300_forward_eps_two_institution_policy_v3", "reports/research/510300_forward_eps_valuation_consistency_policy_v2"]
    records = [read(p + "/result.json") for p in paths]
    metrics = [m for r in records for m in r["all_metrics"]]
    candidates = [m for m in metrics if m["model"].startswith(("S1_", "S2_", "S3_", "Q1_", "Q2_", "Q3_"))]
    best = max((m for m in candidates if m["cost"] == "BASE"), key=lambda m: m["net_sharpe"])
    for path in paths:
        verification = read(path + "/saved_numerical_verification.json")
        write(path + "/acceptance_outcome.json", {"recorded_at": now, "status": "COMPLETED_TARGET_NOT_MET", "goal_achieved": False,
              "source_completion_priority": "PAUSED_BY_LATEST_USER_INSTRUCTION", "saved_verification": verification["status"],
              "authority_note": "docs/510300_SIMPLE_POLICY_PRIORITY_20260907.md", "position_impact": 0})
    round_record = {"round": 22, "study": "510300_FORWARD_EPS_SOURCE_V4_SIX_METHOD_REPLAY", "title": "同六方法补齐明确摊薄字段后的来源重放",
              "status": "COMPLETED_TARGET_NOT_MET_EPS_PRIORITY_PAUSED", "result": "reports/research/510300_forward_eps_source_v4_replay_pipeline_v1/result.json",
              "candidate_configurations": 0, "existing_methods_replayed": 6, "evaluated_candidate_source_runs": 6,
              "evaluation_accounts": 20, "new_accounts_generated": 12, "reused_control_accounts": 8,
              "trained_models": sum(r["trained_models"] for r in records), "post_selected_best_base": best, "all_metrics": metrics,
              "goal_achieved": False, "independent_validation": "NOT_ESTABLISHED_ALREADY_OBSERVED_HISTORY"}
    if not any(r["round"] == 22 for r in index["completed_rounds"]):
        index["completed_rounds"].append(round_record)
        index["evaluation_accounts_in_this_resumption"] += 20
        index["evaluated_candidate_source_runs_including_corrected_replays"] += 6
    index["partial_rounds"] = [r for r in index["partial_rounds"] if r["round"] != 22]
    index.update({"updated_at": now, "status": "ROUND22_COMPLETE_EPS_PAUSED_SIMPLE_POLICY_RESEARCH_CONTINUES", "running_studies": [],
                  "latest_completed_round": round_record, "research_primary_focus": "现有价格数据、简单进出场、有限状态切换和直接完整账户；前瞻EPS停止继续补齐",
                  "latest_continuation_note": "docs/510300_SIMPLE_POLICY_PRIORITY_20260907.md",
                  "process_state_note": "此前来源、十九、二十、二十二轮及诊断均已完成。没有V5来源任务运行。下一轮直接检验简单价格策略。",
                  "next_work": ["完成简单趋势缓冲、追踪退出、反弹期限和市场状态切换的有限候选完整账户。", "按基础和压力费用、全时期及分阶段比较；无局部改善直接切换方法。", "保留所有失败和试验次数，不制作GPT审阅数值包。"],
                  "count_warning": "截至本次登记完成22轮，194个不同方法或范围、206个已评价来源版本、486个评价账户。登记来源版本211，包含5个旧未运行绑定。"})
    for item in index["pending_source_work"]:
        item["status_before_latest_user_pause"] = item.get("status_before_latest_user_pause", item["status"])
        item["status"] = "PAUSED_SOURCE_COMPLETION_BY_USER_FAST_SIMPLE_STRATEGY_PRIORITY"
        item["resume_condition"] = "只有出现可用现成数据支持的明确局部改善假设时再考虑；不自动恢复原补齐队列"
    directory = ROOT / "deliverables/510300前瞻EPS来源修正与六策略进出场_20260907"
    receipt = read(str((directory / "delivery_receipt.json").relative_to(ROOT)))
    receipt.update({"visual_check_pending": False, "visual_check": "六面板净值图已实际查看，标签与曲线清晰，无裁切", "checked_at": now})
    write(str((directory / "delivery_receipt.json").relative_to(ROOT)), receipt)
    delivery = {"created_at": now, "type": "CHINESE_EPS_SIX_METHOD_SOURCE_REPLAY_RESULTS", "directory": str(directory),
                "main_document": str(directory / "前瞻EPS六策略_来源修正结果与完整进出场.md"), "new_gpt_review_archive_created": False}
    if not any(d.get("type") == delivery["type"] for d in index["deliveries"]):
        index["deliveries"].append(delivery)
    write(index_path, index)
    print(json.dumps({"状态": index["status"], "已完成轮数": len(index["completed_rounds"]), "六方法最高基础夏普": best["net_sharpe"]}, ensure_ascii=False))

if __name__ == "__main__":
    main()
