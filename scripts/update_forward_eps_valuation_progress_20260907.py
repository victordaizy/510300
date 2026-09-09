"""保存两轮估值研究进度，保留全部旧记录和未完成目标。"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
SNAPSHOT = ROOT / "reports/research/510300_sharpe_1_2_before_valuation_rounds_update_20260907.json"


def read_json(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def main() -> None:
    original = INDEX.read_text(encoding="utf-8")
    index = json.loads(original)
    assert len(index["completed_rounds"]) == 16
    assert index["registered_configurations_in_this_resumption"] == 182
    assert index["evaluation_accounts_in_this_resumption"] == 422
    now = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    new_rounds = [
        (17, "510300_forward_eps_report_valuation_policy_v1", "前瞻EPS加研报估值空间与共同月份对照"),
        (18, "510300_forward_eps_optional_valuation_v1", "EPS持续更新与可选估值修正"),
    ]
    for number, folder, title in new_rounds:
        path = f"reports/research/{folder}/result.json"
        result = read_json(path)
        checked = read_json(f"reports/research/{folder}/saved_numerical_verification.json")
        assert checked["status"].startswith("PASS_")
        assert result["goal_achieved"] is False
        row = {
            "round": number,
            "study": result["study_id"],
            "title": title,
            "status": "COMPLETED_TARGET_NOT_MET",
            "result": path,
            "candidate_configurations": result["candidate_configurations"],
            "evaluated_candidate_source_runs": result["candidate_configurations"],
            "evaluation_accounts": result["evaluation_accounts"],
            "new_accounts_generated": result["new_accounts_generated"],
            "reused_control_accounts": result["reused_control_accounts"],
            "trained_models": result.get("trained_models", result.get("trained_valuation_error_models")),
            "primary_base": next(x for x in result["primary"] if x["cost"] == "BASE"),
            "primary_stress": next(x for x in result["primary"] if x["cost"] == "STRESS"),
            "post_selected_best_base": max((x for x in result["all_metrics"] if x["cost"] == "BASE"), key=lambda x: x["net_sharpe"]),
            "saved_numerical_verification": checked,
            "goal_achieved": False,
        }
        if number == 17:
            row["coverage_caveat"] = "仅17次可用预测，最后2025-03-31；2025年4月以后估值覆盖不足保持旧份额。共同月份仅EPS夏普0.6789359730，新增因子增量区间跨零。"
        else:
            row["coverage_caveat"] = "原36次EPS预测保留，估值修正仅用于2025-02-28和2025-03-31；34个月末用独立有效基线，增量区间跨零。"
        index["completed_rounds"].append(row)
    index["updated_at"] = now
    index["status"] = "CONTINUING_FORWARD_EPS_COVERAGE_REPRESENTATIVENESS_AND_CURRENT_VALUATION_TARGET_NOT_MET"
    index["goal_achieved"] = False
    index["running_studies"] = []
    index["registered_configurations_in_this_resumption"] = 186
    index["evaluation_accounts_in_this_resumption"] = 438
    index["evaluated_candidate_source_runs_including_corrected_replays"] = 192
    index["registered_candidate_source_runs_including_unrun_legacy_bindings"] = 194
    index["count_warning"] = "十八轮186不同方法或范围，192次已评价候选来源版本，另两个旧绑定未运行，共194登记版本。438评价账户含复用和重复对照及来源更正重放，不是独立实验。本轮新增四候选、八条新账户、八条复用对照，共十六条评价账户。"
    best = index["completed_rounds"][-2]
    index["post_selected_best_base"] = {
        "round": 17,
        "title": best["title"],
        "source_result": best["result"],
        **best["primary_base"],
        "critical_caveat": best["coverage_caveat"],
        "independent_evidence": "NOT_ESTABLISHED_ALREADY_OBSERVED_HISTORY",
    }
    index["latest_continuous_eps_update_candidate"] = {
        "round": 18,
        "source_result": index["completed_rounds"][-1]["result"],
        **index["completed_rounds"][-1]["primary_base"],
        "valuation_adjusted_months": 2,
        "eps_forecast_months": 36,
        "increment_evidence": "INTERVAL_CROSSES_ZERO_NOT_ESTABLISHED",
    }
    index["independent_high_sharpe_evidence"] = "NOT_ESTABLISHED"
    index["latest_continuation_note"] = "docs/510300_FORWARD_EPS_VALUATION_CONTINUATION_20260907.md"
    index["prepare_gpt_numerical_review_package"] = False
    index["delivery_preference"] = "所有因子和完整中文进出场，普通结果文件，不准备GPT审阅数值包。"
    index["process_state_note"] = "第十七、十八轮、股数配对、估值提取和保存核对均已完成，无这些OS计算进程仍在执行；当前目标继续下一项研究。"
    index["new_evidence"].extend([
        "72份明确股本快照均匹配更早公开财报，55在显示精度内、17差异待解释；仅覆盖2017至2021年，不填每日股数。",
        "2791份已保存EPS原件中402份有明确合理估值及参考价，得到31个共同有效月末；2025年4月以后连续16个月目标覆盖不足。",
        "第十七轮四因子主方案夏普0.6865296675，共同月份EPS对照0.6789359730，一致趋势版本负0.3323725872；两模型34拟合、六新四复用十评价账户。",
        "第十七轮最后有效预测2025-03-31，后续保持旧份额；主方案比共同月份对照仅多1201.28096元，估值增量区间跨零，不能将0.687归为持续新优势。",
        "第十八轮保留原36次EPS预测，仅两个成熟月末采用估值误差修正；基础夏普0.6434703338、压力0.6328091765，基础比原EPS多2108.69156元，增量区间仍跨零。",
        "九项必要测试、402份估值原文、34加2个保存模型、十六账户和保存区块核对通过；新中文报告附全部进出场和普通CSV、净值图，无GPT数值审阅包。",
    ])
    index["next_work"] = [
        "优先核对免费前瞻EPS和目标估值覆盖不足的来源结构，探索可核对原件的多机构资料；核对历史成分权重及预测公司代表性。",
        "辅助估值缺失时保持独立有效EPS判断，显式记录信息更新日历；新因子和完整中文进入退出先登记，再做完整账户，不改本轮门槛挑月份救成绩。",
        "继续核对原研报、39158行财报总股数和公司行为时钟；形成有明确依据的当前价格下前瞻估值子集，保留83条缺价和股数缺口。",
        "继续股东回报、公募净申购及全部期限逆回购，保持各自真实口径；不重复旧采集或旧账户，不制作GPT包，目标1.2及稳定超额保持active。",
    ]
    index["checks"] = "九项必要测试通过；402份原估值文本、共同训练时钟、34个第十七轮模型、两个第十八轮修正模型、十六账户与保存区块通过。核对未重训、生成账户、随机重抽或下载，无额外安全审计。中文报告四CSV行数及完整规则检查和净值图目视核对通过。"
    for folder in ["510300_forward_eps_report_valuation_source_v1", "510300_forward_eps_share_snapshot_reconciliation_v1"]:
        path = f"reports/research/{folder}/result.json"
        result = read_json(path)
        index["completed_source_rebuilds"].append({"study": result["study_id"], "result": path, **result})
    for item in index["pending_source_work"]:
        if item["study"] == "510300_FORWARD_EPS_PRICE_SHARE_ALIGNMENT_INVENTORY_V1":
            item.update({
                "status": "72_SNAPSHOTS_RECONCILED_CURRENT_DAILY_SHARE_BASIS_PENDING",
                "snapshot_reconciliation_result": "reports/research/510300_forward_eps_share_snapshot_reconciliation_v1/result.json",
                "explicit_report_snapshots": 72,
                "snapshots_within_rounding": 55,
                "snapshots_requiring_explanation": 17,
                "last_snapshot_date": "2021-08-29",
                "remaining_work": "股数快照覆盖2017至2021，不可填补2022以后；当前每日股数和公司行为时钟仍未校正。新研报估值观点已独立提取并完成两轮，但不是当前价格下已校正的前瞻PE。",
            })
    delivery = ROOT / "deliverables/510300前瞻EPS估值与持续进出场_20260907"
    index["deliveries"].append({
        "created_at": now,
        "rounds": [17, 18],
        "type": "CHINESE_MD_ORDINARY_CSV_AND_PLOT_NO_GPT_PACKAGE",
        "directory": str(delivery),
        "main_document": str(delivery / "前瞻EPS估值_两轮结果与完整进出场.md"),
        "evaluation_accounts": 16,
        "new_accounts_generated": 8,
        "reused_control_accounts": 8,
        "base_new_candidate_trades": 28,
        "new_gpt_review_archive_created": False,
    })
    assert len(index["completed_rounds"]) == 18
    assert all(not x.get("goal_achieved", False) for x in index["completed_rounds"][-2:])
    with SNAPSHOT.open("x", encoding="utf-8") as handle:
        handle.write(original)
    INDEX.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("最新索引已保存：十八轮、186配置、438评价账户；夏普1.2及稳定超额尚未达到，保持继续。")


if __name__ == "__main__":
    main()
