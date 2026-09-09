"""记录十九、二十轮已验证失败，并登记同方法来源修正重放。"""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.financial_annual_components_v1 import read, save, now
from research.forward_eps_guosen_history_v1 import identity


def main():
    path = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
    original = path.read_text(encoding="utf-8")
    index = json.loads(original)
    assert index["evaluated_configurations_in_this_resumption"] == 188
    assert index["evaluation_accounts_in_this_resumption"] == 446
    assert index["registered_candidate_source_runs_including_unrun_legacy_bindings"] == 205
    assert not any(r["round"] in [19, 20, 22] for r in index["completed_rounds"])
    summaries = []
    for number, study, title, session in [
        (19, "510300_forward_eps_two_institution_policy_v2", "两机构前瞻盈利及信息质量", 2240),
        (20, "510300_forward_eps_valuation_consistency_policy_v1", "两机构报告PE关系条件与去PE", 91733),
    ]:
        folder = ROOT / "reports/research" / study
        result = read(folder / "result.json")
        check = read(folder / "saved_numerical_verification.json")
        assert check["complete_accounts_checked"] == 10 and check["matched_training_calendars_identical"]
        assert check["saved_trained_models_checked"] == result["trained_models"]
        prefix = "S" if number == 19 else "Q"
        metrics = [r for r in result["all_metrics"] if r["model"].startswith(prefix)]
        assert len(metrics) == 6 and all(not r["meets_point_target"] for r in metrics)
        primary = {r["cost"]: r for r in result["primary"]}
        save(folder / "acceptance_outcome.json", {"recorded_at": now(), "status": "COMPLETE_TARGET_NOT_MET_SOURCE_GAP_RETAINED",
            "exec_session_id": session, "observed_process_exit_code": 0, "new_account_metrics": metrics,
            "source_result": identity(folder / "result.json"), "saved_verification": identity(folder / "saved_numerical_verification.json"),
            "complete_account_target_met": False, "source_field_gap_discovered": True,
            "source_correction_replay_separately_registered": True, "goal_achieved": False}, exclusive=True)
        record = {"round": number, "study": result["study_id"], "title": title,
            "status": "COMPLETED_TARGET_NOT_MET_SOURCE_GAP_RETAINED", "result": (folder / "result.json").relative_to(ROOT).as_posix(),
            "candidate_configurations": 3, "evaluated_candidate_source_runs": 3, "evaluation_accounts": 10,
            "new_accounts_generated": 6, "reused_control_accounts": 4, "trained_models": result["trained_models"],
            "valid_monthly_feature_origins": result["valid_monthly_feature_origins"],
            "primary_base": primary["BASE"], "primary_stress": primary["STRESS"],
            "post_selected_best_base": max([r for r in metrics if r["cost"] == "BASE"], key=lambda r: r["net_sharpe"]),
            "all_new_metrics": metrics, "saved_numerical_verification": check, "goal_achieved": False,
            "coverage_caveat": "使用东吴第三版，2023年报告未解析。实际账户完整保存；字段修正重放不覆盖原结果。"}
        index["completed_rounds"].append(record)
        summaries.append(record)
    source = read(ROOT / "reports/research/510300_forward_eps_soochow_facts_v4/result.json")
    source_check = read(ROOT / "reports/research/510300_forward_eps_soochow_facts_v4/saved_source_verification.json")
    pipeline = ROOT / "reports/research/510300_forward_eps_source_v4_replay_pipeline_v1"
    registration = read(pipeline / "manifest.json")
    started = read(pipeline / "RUN_STARTED.json")
    assert registration["existing_methods_with_new_sources"] == 6 and source["parsed_reports"] == 5735
    index["completed_rounds"].sort(key=lambda r: r["round"])
    index["partial_rounds"] = [r for r in index["partial_rounds"] if r["round"] not in [19, 20]]
    index["partial_rounds"].append({"round": 22, "study": "510300_FORWARD_EPS_SOURCE_V4_SIX_METHOD_REPLAY",
        "status": "REGISTERED_SOURCE_REPLAY_PIPELINE_STARTED", "candidate_configurations": 0,
        "existing_methods_replayed": 6, "new_source_versions": 6, "planned_evaluation_accounts": 20,
        "planned_new_accounts": 12, "planned_reused_control_accounts": 8,
        "entry_exit_rules": "docs/510300_FORWARD_EPS_SOURCE_V4_SIX_METHOD_REPLAY.md",
        "pipeline": pipeline.relative_to(ROOT).as_posix()})
    index["evaluated_configurations_in_this_resumption"] += 6
    index["evaluated_candidate_source_runs_including_corrected_replays"] += 6
    index["evaluation_accounts_in_this_resumption"] += 20
    index["registered_candidate_source_runs_including_unrun_legacy_bindings"] += 6
    index["status"] = "ROUNDS_19_20_COMPLETE_CONTINUING_ROUND22_SOURCE_CORRECTION"
    index["updated_at"] = now()
    index["latest_completed_round"] = summaries[-1]
    index["running_studies"] = [{"study": "510300_FORWARD_EPS_SOURCE_V4_REPLAY_PIPELINE_V1",
        "session_id": 40082, "pid": started["pid"], "stage": "SOURCE_CORRECTION_REPLAY_STARTED",
        "status_file": (pipeline / "status.json").relative_to(ROOT).as_posix(),
        "result_file": (pipeline / "result.json").relative_to(ROOT).as_posix(),
        "failure_file": (pipeline / "failure.json").relative_to(ROOT).as_posix()}]
    for row in index["pending_source_work"]:
        if row["study"] == "510300_FORWARD_EPS_SOOCHOW_ORIGINALS_AND_FACTS_V3":
            row.update({"status": "ORIGINALS_AND_V4_FACTS_COMPLETED_WITH_REMAINING_GAPS",
                "observed_archived_receipts": 6734, "observed_at": now(), "parsed_v3_reports": 4490,
                "parsed_v4_reports": 5735, "v4_forecast_eps_facts": 17194, "v4_newly_parsed_reports": 1245,
                "remaining_work": "第四版剩999报告无法识别，日期577、证券身份47、其他表格375。先完成已登记六同方法来源重放，不重复归档。"})
    index["count_warning"] = "实际完成21轮，编号1至21。已评价194方法或范围、200候选来源版本、466评价账户。登记194方法、211来源版本；第22轮新增6个来源版本，不新增独立方法，预计20评价账户。"
    index["latest_soochow_source_correction"] = {"result": source, "verification": source_check,
        "directory_year_2023_count": 863, "report_id_prefix_year_2023_count": 864,
        "year_basis_difference_explained": True, "v1_to_v3_regression_suspicion_disproved": True}
    index["new_evidence"].extend([
        "第19/20轮各十评价账户核对通过并实际退出0；第19三方法基础夏普0.402666、0.447543、0.482179，20轮负0.157248、负0.225446、负0.015503，均未达标。",
        "第三版完整保留第一版11363年度行，没有新版丢掉旧成功事实。但明确最新股本摊薄字段漏识别，使2023年无已解析报告。",
        "第四版在不改4490旧成功报告下新增1245报告，总5735报告17194年度EPS；2022新增495、2023新增698、2024新增52，剩999未识别。",
        "按实际目录年份2023共863份，按报告编号年份864份；此前按编号计数已补充说明，统计覆盖应使用实际发布日期。",
        "第22轮六同方法来源重放已冻结并启动，模型、特征定义、训练与进出场、费用均不改；不能因为旧结果好坏选择来源处理。",
    ])
    index["next_work"] = [
        "跟进现有40082的510300_forward_eps_source_v4_replay_pipeline_v1；不重复执行已存在结果的阶段。",
        "完成后比较六方法来源修正前后：覆盖月份、成熟训练、预测误差、实际持仓及自主退出/再入、基础压力完整账户。",
        "第22轮完成只加6已评价来源版本和20评价账户，不加不同方法；保留已失败19/20和所有原件事实。",
        "完成普通中文说明、因子及进出场、历史数值交付，不准备GPT包。",
        "结合完整EPS覆盖继续研究条件组合与失效原因，同时推进价格股数口径、股东回报、公募净申购、全期限逆回购。目标保持1.2且需独立证据。",
    ]
    backup = ROOT / "reports/research/510300_sharpe_1_2_before_round19_20_source_v4_registration_20260907.json"
    with backup.open("x", encoding="utf-8") as handle:
        handle.write(original)
    temp = path.with_suffix(".next.json")
    temp.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)
    continuation = ROOT / index["latest_continuation_note"]
    with continuation.open("a", encoding="utf-8") as handle:
        handle.write("\n\n## 十九、二十轮完成，第四版来源修正与第二十二轮启动\n\n" + index["count_warning"] + "\n\n")
        handle.write("原47679/2240/91733已完成，不要重启。第19/20账户及保存核对均完成，acceptance_outcome.json已存。第19三基础夏普0.402666/0.447543/0.482179，20轮负0.157248/负0.225446/负0.015503，均未达1.2；各10评价账户、模型162/237。第21结果仍保留。\n\n")
        handle.write("V3没有丢失V1已成功事实；11363年度值、原行和日期全部相同。但是最新股本摊薄EPS名称漏识别。第四版已完成，5735报告17194EPS，旧4490报告完全保留；新增1245。2023实际目录年份863份、编号2023前缀864份，本轮按实际目录年份统计698成功。原三份首2023报告9条EPS/净利润/PE共27数值已目视，9源测试通过、13重放汇总测试通过。尚有999来源缺口，不能说全部修复。\n\n")
        handle.write("当前唯一接续会话40082，scripts/run_forward_eps_source_v4_replay_pipeline_20260907.py，reports/research/510300_forward_eps_source_v4_replay_pipeline_v1。两机构features_v3与policy_v3，PE features_v2与policy_v2，全部使用soochow_facts_v4。每组原算法仅替换来源及输出标识，运行函数AST比对、配置非来源行为逐项不变。依次六阶段、两组各10评价账户。检查status.json和真实进程、result/failure；不要手工重复启动已有阶段。完成后加6来源版本和20账户，不加新方法。中文完整因子和入加减退再入规则docs/510300_FORWARD_EPS_SOURCE_V4_SIX_METHOD_REPLAY.md。目标active，无GPT包，无子代理。\n")
    print(index["count_warning"], flush=True)


if __name__ == "__main__":
    main()
