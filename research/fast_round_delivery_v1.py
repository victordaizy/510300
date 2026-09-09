"""复用简短研究交付和索引更新，避免每轮重复拼接相同报告。"""
import json
import shutil
from pathlib import Path
from research.intraday_overnight_increment_v1 import now, require, write_json


def metric_table(rows):
    lines = ["|策略|费用|净夏普|年化收益|最大回撤|成交次数|", "|---|---|---:|---:|---:|---:|"]
    for row in rows:
        sharpe = "未定义（零波动）" if row["net_sharpe"] is None else f"{row['net_sharpe']:.3f}"
        lines.append(f"|{row['name']}|{'基础' if row['cost']=='BASE' else '压力'}|{sharpe}|{row['annualized_return']:.2%}|{row['max_drawdown']:.2%}|{row['trade_count']}|")
    return lines+[""]


def deliver_round(root, research, config_path, document, title, status, decision, verification_text, next_path, next_lines, next_status, next_focus):
    root, research, document, next_path = map(Path, [root, research, document, next_path])
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    result = json.loads((research / "result.json").read_text(encoding="utf-8"))
    index_path = root / "reports/research/510300_sharpe_1_2_latest_research.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    number = cfg["round"]
    require(index["latest_completed_round"]["round"] == number-1 and not document.parent.exists(), "快速交付的前序轮次或目录状态不同")
    require(result["study_id"] == cfg["study_id"] and not result["goal_achieved"], "快速交付不能替代完整目标验收")
    require((research / "saved_verification_receipt.json").is_file(), "必要的保存结果核对尚未完成")
    write_json(research / "acceptance_outcome.json", {"recorded_at": now(), "status": status, "decision": decision, "goal_achieved": False, "position_impact": 0}, exclusive=True)
    next_path.write_text("\n".join(next_lines)+"\n", encoding="utf-8")
    document.parent.mkdir(parents=True)
    document.write_text("\n".join([f"# 第{number}轮：{title}", "", decision, "", "## 主历史：2020年1月2日至2026年8月14日开盘", "", *metric_table(result["all_metrics"]),
        "## 较早历史：2015年1月5日至2019年12月31日开盘", "", *metric_table(result["earlier_diagnostics"]), verification_text, "",
        *(root / cfg["rules"]).read_text(encoding="utf-8").splitlines()[1:]])+"\n", encoding="utf-8")
    for path in research.iterdir():
        if path.is_file() and path.suffix in {".csv", ".json"}:
            shutil.copy2(path, document.parent / path.name)
    shutil.copy2(config_path, document.parent / "冻结设置.json")
    shutil.copy2(next_path, document.parent / "下一项研究方向.md")
    primary = {m["cost"]: m for m in result["all_metrics"] if m["model"] == cfg["primary"]}
    record = {"round": number, "study": result["study_id"], "title": title, "status": status, "result": str((research / "result.json").relative_to(root)),
              **{k: result[k] for k in ["candidate_configurations", "evaluation_accounts", "new_accounts_generated", "reused_control_accounts", "earlier_diagnostic_accounts", "new_earlier_diagnostic_accounts", "new_model_fits", "new_reference_accounts"]},
              "evaluated_candidate_source_runs": result["candidate_configurations"], "primary_base": primary["BASE"], "primary_stress": primary["STRESS"], "post_selected_best_base": result["post_selected_best_base"]}
    index["completed_rounds"].append(record)
    for key in ["registered_configurations_in_this_resumption", "evaluated_configurations_in_this_resumption", "evaluated_candidate_source_runs_including_corrected_replays", "registered_candidate_source_runs_including_unrun_legacy_bindings"]:
        index[key] += result["candidate_configurations"]
    index["evaluation_accounts_in_this_resumption"] += result["evaluation_accounts"]
    index.update(updated_at=now(), latest_completed_round=record, running_studies=[], goal_achieved=False, status=f"ROUND{number}_COMPLETED_FULL_GOAL_NOT_MET",
                 count_warning=f"{number}轮，{index['evaluated_configurations_in_this_resumption']}不同设置，{index['evaluated_candidate_source_runs_including_corrected_replays']}已评价来源版本，{index['registered_candidate_source_runs_including_unrun_legacy_bindings']}登记含5旧未运行，{index['evaluation_accounts_in_this_resumption']}主评价记录。",
                 next_work={"status": next_status, "focus": next_focus, "source": str(next_path.relative_to(root))}, process_state_note=verification_text)
    index["deliveries"].append({"created_at": now(), "type": f"FAST_CHINESE_RESULTS_ROUND{number}", "rounds": [number], "directory": str(document.parent), "main_document": str(document), "new_gpt_review_archive_created": False})
    write_json(index_path, index)
    write_json(document.parent / "交付回执.json", {"created_at": now(), "main_document": str(document), "goal_achieved": False, "new_gpt_review_archive_created": False}, exclusive=True)
    require(document.is_file() and json.loads(index_path.read_text(encoding="utf-8"))["latest_completed_round"]["round"] == number, "快速交付索引与文档未落盘")
    return {"document": str(document), "counts": index["count_warning"], "next_status": next_status}
