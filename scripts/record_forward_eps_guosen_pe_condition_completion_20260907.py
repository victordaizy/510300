"""记录八账户真实完成、未达目标与中文交付，保留现有来源接续。"""
from pathlib import Path
import json
import re
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.financial_annual_components_v1 import read, save, now
from research.forward_eps_guosen_history_v1 import identity

SOURCE = ROOT / "reports/research/510300_forward_eps_guosen_pe_condition_policy_v1"
DELIVERY = ROOT / "deliverables/510300前瞻EPS市盈率对照结果与进出场_20260907"


def main():
    result = read(SOURCE / "result.json")
    check = read(SOURCE / "saved_numerical_verification.json")
    assert check["status"] == "PASS_SAVED_GUOSEN_PE_CONDITION_SAME_TRAINING_AND_COMPLETE_ACCOUNTS"
    assert check["complete_accounts_checked"] == 8 and check["saved_trained_models_checked"] == 72
    new = [r for r in result["all_metrics"] if r["model"].startswith("G")]
    assert len(new) == 4 and all(not r["meets_point_target"] for r in new)
    index_path = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
    original = index_path.read_text(encoding="utf-8")
    index = json.loads(original)
    assert index["evaluated_configurations_in_this_resumption"] == 186
    assert index["evaluation_accounts_in_this_resumption"] == 438
    assert index["registered_configurations_in_this_resumption"] == 194
    assert index["registered_candidate_source_runs_including_unrun_legacy_bindings"] == 205
    assert not any(r["round"] == 21 for r in index["completed_rounds"])
    doc = DELIVERY / "前瞻EPS市盈率对照_结果原因与进出场.md"
    text = doc.read_text(encoding="utf-8")
    assert "```" not in text and "### 进入、加仓、减仓、退出与重新进入" in text
    for name in ["首次进入", "重新进入", "缺失", "T+1", "1604日", "0.380", "0.304", "53,477.22", "64,553.06"]:
        assert name in text, name
    links = re.findall(r"\]\(<([^>]+)>\)", text)
    assert all(Path(p).is_absolute() and Path(p).exists() for p in links)
    copies = {"完整账户表现.csv": ("metrics.csv", 8), "逐年账户表现.csv": ("yearly_metrics.csv", 56),
              "三个既定阶段表现.csv": ("era_metrics.csv", 24)}
    for target, (source, count) in copies.items():
        assert identity(DELIVERY / target)["sha256"] == identity(SOURCE / source)["sha256"]
        assert len(pd.read_csv(DELIVERY / target)) == count
    for filename, count in [("全部真实成交_含终点清算.csv", 71), ("全部预定月末判断_含无观点.csv", 640),
                            ("逐月实际持仓与净损益.csv", 640)]:
        assert len(pd.read_csv(DELIVERY / filename)) == count
    chart = DELIVERY / "完整净值与2024年进出场影响.png"
    save(DELIVERY / "visual_and_document_verification.json", {"checked_at": now(),
        "status": "PASS_EIGHT_ACCOUNT_TABLES_CHINESE_RULES_ALL_TRADES_AND_VIEWED_CHART",
        "chart_manually_viewed": True, "chart_dimensions": [2048, 1184],
        "2024_axis_limited_to_2024": True, "csv_files": 6, "trade_rows": 71, "month_end_decision_rows": 640,
        "source_values_and_links_verified": True, "new_models_fit": 0, "new_account_evaluations": 0,
        "gpt_review_package_created": False, "files": [identity(doc), identity(chart)]}, exclusive=True)
    receipt = read(DELIVERY / "delivery_receipt.json")
    receipt.update({"visual_check_pending": False, "verified_at": now(), "document": identity(doc), "chart": identity(chart)})
    (DELIVERY / "delivery_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    save(SOURCE / "acceptance_outcome.json", {"recorded_at": now(), "status": "COMPLETED_BOTH_NEW_METHODS_TARGET_NOT_MET",
        "exec_session_id": 74494, "observed_process_exit_code": 0, "new_models": new,
        "source_result": identity(SOURCE / "result.json"), "saved_verification": identity(SOURCE / "saved_numerical_verification.json"),
        "same_original_training_calendar": True, "complete_account_target_met": False,
        "stable_increment_established": False, "all_new_pair_intervals_cross_zero": True,
        "post_result_attribution": "主要差距出现在2024年，8月末预测改变了9月初减仓或清仓；较少费用不能解释较少收益。",
        "original_source_issues_not_overridden_by_better_legacy_return": True,
        "pending_rounds_19_and_20_preserved": True, "goal_achieved": False}, exclusive=True)
    by_cost = {r["cost"]: r for r in result["primary"]}
    round_record = {"round": 21, "study": result["study_id"], "title": "原国信PE关系条件与去除PE的相同月份对照",
        "status": "COMPLETED_TARGET_NOT_MET", "result": str((SOURCE / "result.json").relative_to(ROOT).as_posix()),
        "candidate_configurations": 2, "evaluated_candidate_source_runs": 2, "evaluation_accounts": 8,
        "new_accounts_generated": 4, "reused_control_accounts": 4, "trained_models": 72,
        "primary_base": by_cost["BASE"], "primary_stress": by_cost["STRESS"],
        "post_selected_best_base": max([r for r in new if r["cost"] == "BASE"], key=lambda r: r["net_sharpe"]),
        "all_new_metrics": new, "saved_numerical_verification": check, "goal_achieved": False,
        "coverage_caveat": "原50合格月、36预测、47成熟标签和全部原训练月份保持；主要交易差距在2024，历史已观察，未获独立证据。"}
    index["completed_rounds"].append(round_record)
    index["completed_rounds"].sort(key=lambda row: row["round"])
    index["partial_rounds"] = [r for r in index["partial_rounds"] if r["round"] != 21]
    index["evaluated_configurations_in_this_resumption"] += 2
    index["evaluated_candidate_source_runs_including_corrected_replays"] += 2
    index["evaluation_accounts_in_this_resumption"] += 8
    index["updated_at"] = now()
    index["status"] = "ROUND_21_COMPLETED_CONTINUING_EXISTING_ROUNDS_19_20"
    index["latest_completed_round"] = round_record
    index["count_warning"] = "实际完成19轮，编号1至18及21；第19、20仍按原接续。已评价188方法或范围、194候选来源版本、446评价账户，登记194方法、205来源版本。编号不是完成轮数，复用对照和来源重放不是独立实验。"
    index["new_evidence"].extend([
        "第21轮四条新账户加四保存对照真实完成，72模型、47成熟标签、原EPS所有训练日历、144月末请求和八完整账户核对通过。",
        "关系相容PE基础夏普0.380139、压力0.372695；去PE基础0.304047、压力0.295749；原EPS基础0.618973。两新方法未达1.2，也未证实超额增量。",
        "基础费用下两新方法分别比原EPS少赚53477.2168和64553.0621元，费用反而少97.5832和123.8379元，差距不是多交费用。",
        "2024-08-30原EPS预测约1.035%、相容PE0.131%、去PE负0.088%，下一开盘分别保持54900份、减至13700份和清仓。9月净损益分别46170.90、10958.26、负188.73元。",
        "原/相容/去PE的2024年收益分别23.58%/1.75%/负3.43%，2025约16.6%接近；结果对少数月末与因子处理敏感，不以旧回测较好回填有问题的PE，也不事后特设2024策略。",
    ])
    index["deliveries"].append({"created_at": now(), "type": "CHINESE_GUOSEN_PE_COMPLETE_ACCOUNT_RESULTS_AND_RULES",
        "directory": str(DELIVERY), "main_document": str(doc), "csv_files": 6,
        "visual_and_document_check": "PASS_EIGHT_ACCOUNT_TABLES_CHINESE_RULES_ALL_TRADES_AND_VIEWED_CHART",
        "new_gpt_review_archive_created": False})
    index["next_work"].insert(0, "继续现有47679/2240/91733，读取第19与20轮真实来源和账户；结合已完成第21轮，区分新机构及早期覆盖、PE质量与样本构成、预测及进出场敏感性。")
    pipeline = read(ROOT / "reports/research/510300_forward_eps_two_institution_pipeline_v1/status.json")
    if "archived_receipt_files" in pipeline:
        for study in index["running_studies"]:
            if study["study"] == "510300_FORWARD_EPS_SOOCHOW_ORIGINALS_V1":
                study.update({"observed_at": pipeline["observed_at"], "archived_receipts": pipeline["archived_receipt_files"]})
        for task in index["pending_source_work"]:
            if task["study"] == "510300_FORWARD_EPS_SOOCHOW_ORIGINALS_AND_FACTS_V3":
                task.update({"observed_at": pipeline["observed_at"], "observed_archived_receipts": pipeline["archived_receipt_files"]})
    backup = ROOT / "reports/research/510300_sharpe_1_2_before_guosen_pe_condition_completion_20260907.json"
    with backup.open("x", encoding="utf-8") as handle:
        handle.write(original)
    temporary = index_path.with_suffix(".next.json")
    temporary.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(index_path)
    continuation = ROOT / index["latest_continuation_note"]
    with continuation.open("a", encoding="utf-8") as handle:
        handle.write("""

## 第二十一轮已完成：原国信PE修正没有提高夏普

`reports/research/510300_forward_eps_guosen_pe_condition_policy_v1/result.json` 与 saved_numerical_verification.json 已完成；执行会话74494退出0。八完整账户、72新保存模型、47成熟标签、36预测及原EPS每个训练日历相同、144有效月末请求、四保存区块文件核对通过。acceptance_outcome.json记录两新方法均未达1.2，全部新配对区间跨零，无独立证据。不要重跑原研究。

G1关系相容PE基础夏普0.3801391842、压力0.3726952805；G2去PE基础0.304047、压力0.295749。原EPS0.6189727421/0.6089349。基础年化2.86%、2.13%，低于买入持有3.63%。G1/G2比原EPS少53477.2168/64553.0621元，费用反而少97.5832/123.8379元。主要差距在2024，不能归因成本或年份选择。

8月30日同一训练日历下，原EPS预计60日约1.035%、G1约0.131%、G2负0.088%；9月2日原保持54900份、G1卖41200剩13700、G2卖13800清仓。9月净损益46170.90/10958.26/负188.73元。G1于10月8日清剩余13700、12月2日再买25300；G2于12月2日再入12000。2024收益原23.58%、G1 1.75%、G2负3.43%，2025三者约16.6%接近。原优势对少数时点和因子口径敏感，不用这些已观察月份临时设专属策略，不因旧回测较好恢复“真实估值”说法。

已完成轮数现在19，编号1至18加21，第19/20仍待原接续结果。已评价188方法或范围、194来源版本、446评价账户；登记194方法、205来源版本，六个19/20方法待评价。后续19/20真实完成分别加3方法、3来源版本、10评价账户，不能再用旧188以前的固定计数覆盖。全局此前最好仍17轮约0.68653，目标active。

新交付 `deliverables/510300前瞻EPS市盈率对照结果与进出场_20260907/前瞻EPS市盈率对照_结果原因与进出场.md`，六CSV、71实际成交、640预定月末判断含无观点、完整净值及2024持仓图。图已目视、2024横轴已限定，表格与全部来源核对通过。规则均中文含入/加/减/清仓/缺失/再入，不做GPT包。
""")
    print(json.dumps({"completed_rounds": len(index["completed_rounds"]), "evaluated_methods": 188,
        "evaluated_source_versions": 194, "evaluated_accounts": 446, "pending_rounds": [r["round"] for r in index["partial_rounds"]],
        "goal_achieved": False}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
