"""有限核实2026来源延长的因果边界，使用保存参数，不拟合或重跑账户。"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research import macro_technical_first_passage_inputs_v1 as model
from research import macro_technical_first_passage_study_v1 as study
from research.finalize_macro_funding_source_contract_v2 import FORWARD

ROOT = study.ROOT
OUT = ROOT / "reports/research/510300_macro_source_extension_causal_limit_v1"
PARENT = ROOT / "reports/research/510300_macro_funding_source_contract_v2"
COVERAGE = ROOT / "reports/research/510300_macro_2026_source_coverage_intake_v1_clock_adapter"
BASE = study.BASE / "results/原点全部技术特征.parquet"
LABELS = study.BASE / "results/原点首次边界参考结果.parquet"
SAVED_INPUTS = PARENT / "results/全部3488当时已知技术与宏观_未知保留.parquet"
FUNDING = COVERAGE / "results/完整3488资金源派生_仅扩已有日历不拟合.parquet"
FORECASTS = PARENT / "results/全部事前配对模型预测与实际宏观路径.parquet"
ACCOUNTS = PARENT / "results/十二完整账户共同口径比较.parquet"
CARD = ROOT / "docs/510300_MACRO_SOURCE_EXTENSION_CAUSAL_LIMIT_V1.md"
STATE = ROOT / "reports/research/510300_daily_weekly_goal_continuation_20261001/state.json"
REPORT = OUT / "2026资料延长与原固定配置的因果边界.md"


def relative(path):
    return path.absolute().relative_to(ROOT).as_posix()


def normalize_clocks(frame):
    """只改变时间的存储精度；经济时点和时区不变。"""
    result = frame.copy()
    for column in result.columns:
        dtype = result[column].dtype
        if pd.api.types.is_datetime64_any_dtype(dtype):
            target = (pd.DatetimeTZDtype(unit="ns", tz=dtype.tz)
                      if isinstance(dtype, pd.DatetimeTZDtype) else "datetime64[ns]")
            converted = result[column].astype(target)
            pd.testing.assert_series_equal(result[column], converted.astype(dtype), check_exact=True)
            result[column] = converted
    return result


def save_table(name, frame):
    directory = OUT / "results"
    directory.mkdir(exist_ok=True)
    frame.to_parquet(directory / (name + ".parquet"), index=False)
    frame.to_csv(directory / (name + ".csv"), index=False, encoding="utf-8-sig")


def freeze():
    study.require(not OUT.exists(), "用途目录已存在，不覆盖登记或失败。")
    sources = [BASE, LABELS, SAVED_INPUTS, FUNDING, FORECASTS, ACCOUNTS,
               study.ORDERS, study.MARGIN, PARENT / "saved_models.json",
               PARENT / "protocol.json", PARENT / "summary.json", CARD,
               Path(__file__), Path(model.__file__), Path(model.original.__file__)]
    records = [{"path": relative(p), "sha256": study.digest(p)} for p in sources]
    OUT.mkdir(parents=True)
    study.write_json(OUT / "protocol.json", {
        "at": study.now(), "study": "510300_MACRO_SOURCE_EXTENSION_CAUSAL_LIMIT_V1",
        "registration_decision": "TECH.R203", "result_decision": "TECH.R204",
        "purpose": relative(CARD), "sources": records,
        "comparison_cutoff_exclusive": "2026-01-01",
        "saved_score_replay_period": ["2015-01-05", "2019-12-31"],
        "all_pre2026_input_columns": True, "all_pre2026_saved_monthly_pools": True,
        "saved_models_only": True, "new_fits": 0, "new_accounts": 0,
        "new_training_labels": 0, "network_requests": 0,
        "financial_result_source": "TECH.R198_SAVED_METRICS_NOT_NEW_RESULT",
        "failure_action": "停止结论，保存差异；原失败和原前瞻保持。"}, exclusive=True)
    print("TECH.R203有限用途已登记：输入前缀、共同成熟池和保存树；0拟合、账户、采集。", flush=True)


def run():
    study.require(not (OUT / "RUN_STARTED.json").exists(), "有限核实已启动，保留原记录，不重复执行。")
    protocol = study.read(OUT / "protocol.json")
    for item in protocol["sources"]:
        study.require(study.digest(ROOT / item["path"]) == item["sha256"], "登记来源改变：" + item["path"])
    study.write_json(OUT / "RUN_STARTED.json", {"at": study.now()}, exclusive=True)
    old = normalize_clocks(pd.read_parquet(SAVED_INPUTS))
    new = normalize_clocks(model.views(pd.read_parquet(BASE), pd.read_parquet(study.ORDERS),
                                      pd.read_parquet(FUNDING), pd.read_parquet(study.MARGIN)))
    study.require(old.columns.equals(new.columns), "重新拼接的列合同改变。")
    before = old.date.lt("2026-01-01")
    pd.testing.assert_frame_equal(old.loc[before].reset_index(drop=True),
                                  new.loc[before].reset_index(drop=True), check_exact=True, check_dtype=False)
    save_table("原2026前全部输入_逐列精确保持", old.loc[before])
    labels = pd.read_parquet(LABELS)
    saved = study.read(PARENT / "saved_models.json")
    records = [r for r in saved if pd.Timestamp(r["fit_date"]) < pd.Timestamp("2026-01-01")]
    pools = []
    for record in records:
        index = record["fit_index"]
        left = model.common_pool(old, labels, index)
        right = model.common_pool(new, labels, index)
        pd.testing.assert_frame_equal(left, right, check_exact=True)
        study.require(left.origin_index.astype(int).tolist() == record["training_origins"],
                      "成熟训练成员与保存月记录不符。")
        study.require(len(left) == record["training_rows"], "保存训练成员数改变。")
        study.require(left.mature_idx.le(index).all(), "训练使用未来成熟信息。")
        if len(left):
            study.require(left.origin_index.max() <= index, "训练使用未来原点。")
        counts = left.event_class.value_counts()
        study.require({c: int(counts.get(c, 0)) for c in model.original.CLASSES} == record["class_counts"],
                      "保存训练类别计数不符。")
        pool_status = ("FIT_COMPLETE" if len(left) >= model.original.MINIMUM_ROWS
                       and all(counts.get(c, 0) >= 10 for c in model.original.CLASSES)
                       else "NO_VIEW_COMMON_TRAINING_SUPPORT")
        study.require(pool_status == record["status"], "保存模型支持状态不符。")
        pools.append({"fit_date": record["fit_date"], "fit_index": index,
                      "training_rows": len(left), "status": record["status"],
                      "all_columns_and_weights_exact": True,
                      "saved_members_exact": True, "mature_before_fit": True})
    save_table("全部2026前月度共同池和保存成员", pd.DataFrame(pools))
    forecasts = pd.read_parquet(FORECASTS)
    early = forecasts.loc[forecasts.date.between("2015-01-05", "2019-12-31")].copy()
    lookup = {r["fit_index"]: r for r in records}
    replay = []
    for row in early.loc[early.status.eq("AVAILABLE")].itertuples(index=False):
        record = lookup[int(row.fit_index)]
        tree = record["models"][row.policy]
        vector = new.iloc[row.origin_index][tree["features"]].to_numpy(float)
        probabilities, leaf, used = model.predict(tree, vector)
        quality = model.original.quality(tree, probabilities)
        np.testing.assert_array_equal(probabilities, [row.p_LOSS, row.p_PROFIT, row.p_TIMEOUT])
        study.require(leaf == row.leaf and tree["node_rows"][leaf] == row.leaf_train_rows,
                      "保存树的路径或叶成员改变。")
        for key, value in quality.items():
            np.testing.assert_equal(value, getattr(row, key))
        macro_path = "|".join(dict.fromkeys(x for x in used if x in model.MACRO))
        study.require(macro_path == row.macro_features_on_path, "原宏观路径改变。")
        event = bool(quality["predicted_p_times_b"] > 1. and quality["predicted_net_expectation"] > 0.)
        study.require(event == row.entry_event, "原入场门结果改变。")
        replay.append({"date": row.date, "policy": row.policy, "fit_index": int(row.fit_index),
                       "all_probabilities_and_quality_exact": True, "leaf": leaf,
                       "entry_event": event, "macro_path_exact": True})
    save_table("2015至2019全部可评分日_保存树精确复现", pd.DataFrame(replay))
    metrics = pd.read_parquet(ACCOUNTS)
    financial = metrics.loc[metrics.period.eq("2015_2019")].copy()
    save_table("原R198早期六账户保存指标_不是新金融结果", financial)
    comparisons = []
    for cost in ["BASE", "STRESS"]:
        local = financial.loc[financial.cost.eq(cost)].set_index("policy")
        macro = local.loc["MACRO_TECH_TREE"]
        for comparator in ["A_SAVED_WEIGHT", "TECH_COMMON_TREE"]:
            control = local.loc[comparator]
            comparisons.append({"cost": cost, "comparator": comparator,
                "saved_macro_net_cagr": macro.net_cagr, "saved_control_net_cagr": control.net_cagr,
                "saved_cagr_delta": macro.net_cagr - control.net_cagr,
                "required_cagr_improvement_passed": bool(macro.net_cagr > max(0., control.net_cagr)),
                "saved_macro_net_sharpe": macro.net_sharpe, "saved_control_net_sharpe": control.net_sharpe,
                "role": "SAVED_R198_METRICS_NOT_NEW_FINANCIAL_RESULT"})
    comparison = pd.DataFrame(comparisons)
    study.require(not comparison.required_cagr_improvement_passed.any(), "实际保存早期收益门与假设不符。")
    save_table("仅扩2026不能改变的原早期必需收益门", comparison)
    summary = {"at": study.now(), "study": protocol["study"], "registration_decision": "TECH.R203",
        "decision": "TECH.R204", "status": "REJECTED_SOURCE_ONLY_EXTENSION_AS_RESCUE_OF_FIXED_R198_CONFIG",
        "pre2026_input_rows_exact": int(before.sum()), "pre2026_input_columns_exact": len(old.columns),
        "pre2026_monthly_pools_exact": len(pools),
        "pre2026_fitted_month_records": sum(r["status"] == "FIT_COMPLETE" for r in records),
        "early_saved_forecast_rows": len(early), "early_saved_model_evaluations_exact": len(replay),
        "early_saved_entry_events_exact": int(early.entry_event.sum()),
        "unchangeable_required_early_cagr_comparisons_failed": len(comparisons),
        "new_fits": 0, "new_accounts": 0, "new_labels": 0,
        "existing_saved_label_rows_read": len(labels), "new_network_requests": 0,
        "new_financial_metrics": "NOT_COMPUTED", "latest_actual_financial_decision": "TECH.R198",
        "first_vintage": "NOT_CERTIFIED", "independent_validation": "NOT_ESTABLISHED",
        "goal_achieved": False, "overfitting_removed": False,
        "decision_scope": "仅2026来源延长对原R198全四场景晋升无充分性，不否定全部宏观或未来资料用途。",
        "next_action": "关闭以补2026为原配置修复的方向；只有不同完整信息用途、真实错误或真正新样本才另登记。"}
    study.write_json(OUT / "summary.json", summary, exclusive=True)
    text = "# 2026资料延长无法修复原固定配置的早期失败\n\n"
    text += (f"TECH.R203—TECH.R204完成有限核实。原2026年前{int(before.sum())}行、{len(old.columns)}列"
             f"输入精确相同；{len(pools)}个月度共同成熟训练池、类别、权重和保存成员相同；"
             f"2015—2019全部{len(replay)}次可评分日的原树概率、质量估计、路径和入场门精确复现。"
             "本次0新拟合、0账户、0标签、0采集；读取原标签作成熟池核实，不生成新标签。\n\n")
    text += "下面是TECH.R198已经保存的账户成绩，未重跑账户，不是新的收益结果。\n\n"
    display = financial[["cost", "policy", "net_cagr", "net_sharpe", "completed_cycles", "p_times_b"]]
    text += display.to_markdown(index=False, floatfmt=".6f") + "\n\n"
    text += comparison.to_markdown(index=False, floatfmt=".6f") + "\n\n"
    text += ("原冻结口径要求两个时期、两费用全部改善。早期联合账户在两费用下的年化收益均低于A及共同池技术树，"
             "四项必要收益比较全部失败。仅2026年新增资料不能倒流改变早期已可知输入、训练或交易；"
             "即使未来取得完整2026两市融资，也不能令原配置全部四场景通过。"
             "这属于因果边界判断，不是对缺失2026收益的估算。\n\n")
    text += ("决定：停止把2026资料延长当作TECH.R198配置的收益修复路径。R200资金恢复与唯一原点评分、"
             "R202四个有效响应和两个失败仍是来源事实；不重试原固定请求，不自动扩到全年。"
             "其他独立用途可以引用这些资料，但必须先声明不同机制、可知钟、比较和退出，不能调整原树营救失败。\n\n")
    text += ("当前不存在已准入待跑金融候选，收益夏普目标未达，独立验证未建立。原E03前瞻"
             "仍从真实合格新资料推进，当前最早合格新收盘为2026-10-08T15:05+08:00。"
             "新资料不自动代表目标实现；目前不能声称去除过拟合。\n\n")
    text += ("复验条件：真实来源/实现错误；先登记且实质不同的完整用途；真正新样本。"
             "单独补2026来源不改变本次因果结论。\n\n"
             "[原具体上涨及全部金融结论](../510300_macro_funding_source_contract_v2/多信息源评分_具体上涨与完整结论.md)、"
             "[本次有限核实](summary.json)、[所有月度共同池](results/全部2026前月度共同池和保存成员.csv)。\n")
    REPORT.write_bytes(text.encode("utf-8"))
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def publish():
    study.require(not (OUT / "project_state_update_receipt.json").exists(), "长期事实已更新，不重复插入。")
    result = study.read(OUT / "summary.json")
    before = STATE.read_bytes()
    state = json.loads(before.decode("utf-8-sig"))
    original_forward = {key: copy.deepcopy(state[key]) for key in FORWARD}
    (OUT / "state_before_TECH_R204.json").write_bytes(before)
    progress = (f"TECH.R204核实2026年前{result['pre2026_input_rows_exact']}行/"
                f"{result['pre2026_input_columns_exact']}列、{result['pre2026_monthly_pools_exact']}共同成熟月池"
                f"及早期{result['early_saved_model_evaluations_exact']}保存树评分精确保持。"
                "早期两费用联合CAGR均低于A和配对TECH；仅扩2026不能使原R198全四场景通过。"
                "关闭以2026补源救援原配置的方向；0新模型/账户/标签/请求。金融仍R198拒绝、完整目标未达。")
    next_action = result["next_action"]
    state["archived_phase_before_TECH_R204"] = {key: copy.deepcopy(state.get(key)) for key in
        ["current_phase", "current_phase_trial_accounting", "current_goal_turn_actual_work", "goal_turn_classification",
         "next_information_source_proposal", "latest_progress"]}
    state.update(updated_at=study.now(), latest_technical_decision="TECH.R204",
        latest_information_intake_technical_decision="TECH.R204", latest_registration_decision="TECH.R203",
        latest_actual_financial_decision="TECH.R198", latest_completed_study=relative(OUT),
        latest_result=relative(OUT / "summary.json"), latest_report=relative(REPORT),
        latest_research_status=result["status"], latest_progress=progress, latest_continuation_outcome=progress,
        latest_progress_check=relative(OUT / "summary.json"), current_study=result["study"],
        current_phase="SOURCE_ONLY_2026_EXTENSION_CLOSED_FOR_FIXED_CONFIG_RESCUE",
        current_goal_turn_classification="progress", goal_turn_classification="PROGRESS_CAUSAL_SCOPE_CHANGED_NEXT_ACTION",
        goal_turn_progress_classification="PROGRESS_CAUSAL_SCOPE_CHANGED_NEXT_ACTION",
        current_goal_turn_classification_reason="实测完整前缀、成熟池和保存参数，明确不能通过早期必要门，取消原2026补源修复用途。",
        consecutive_blocked_goal_turns=0, blocked_audit_count=0,
        current_goal_turn_actual_work={"new_fits": 0, "new_accounts": 0, "new_labels": 0,
            "network_requests": 0, "existing_label_rows_read": result["existing_saved_label_rows_read"],
            "saved_model_evaluations": result["early_saved_model_evaluations_exact"],
            "pre2026_exact_monthly_pools": result["pre2026_monthly_pools_exact"]},
        current_phase_trial_accounting={"scope": "TECH_R203_R204_SOURCE_EXTENSION_CAUSAL_LIMIT_ONLY",
            "new_accounts": 0, "new_fits": 0, "new_labels": 0, "new_requests": 0},
        next_information_source_proposal=next_action, next_research_question=next_action, current_priority=next_action,
        current_unmet_evidence="R198完整金融失败保持；未有实质不同完整用途、真实错误或新独立样本；补2026不能修复早期门。",
        next_candidate_field_status="NO_ADMITTED_UNRUN_COMPLETE_PURPOSE",
        next_financial_experiment="NOT_DEFINED_OR_REGISTERED_REQUIRES_DIFFERENT_COMPLETE_PURPOSE_OR_NEW_SAMPLE",
        current_admitted_unrun_numeric_candidates=0, current_admitted_unrun_complete_uses=0,
        new_accounts_in_current_phase=0, new_financial_result_computed_this_continuation=False,
        return_and_sharpe_improved_this_continuation=False, return_and_sharpe_changed_this_continuation=False,
        current_phase_financial_candidate_configurations=0, current_phase_required_tests=0,
        new_model_fits_this_continuation=0, new_accounts_this_continuation=0,
        new_primary_accounts_this_continuation=0, new_financial_candidate_accounts_this_continuation=0,
        new_strategy_accounts_this_continuation=0, new_matched_control_accounts_this_continuation=0,
        saved_account_controls_replayed_this_continuation=0, internal_reference_replays_this_continuation=0,
        saved_accounts_checked_this_continuation=0, new_strategy_configurations_this_continuation=0,
        new_training_labels_this_continuation=0, new_return_labels_this_continuation=0,
        new_market_requests_this_continuation=0, necessary_tests_passed_this_continuation=0,
        necessary_test_executions_this_continuation=0, fixed_saved_model_evaluations_this_continuation=result["early_saved_model_evaluations_exact"],
        actual_prefix_checks_this_continuation=1,
        actual_prefix_check_scope="2026前全部53列输入、全部保存月度成熟池、早期保存树概率/路径/质量/门精确保持；不是新独立验证。",
        latest_2026_source_extension_causal_limit=relative(OUT / "summary.json"),
        latest_continuation_receipt=relative(OUT / "project_state_update_receipt.json"))
    study.require({key: state[key] for key in FORWARD} == original_forward, "原13前瞻值改变。")
    documents = [ROOT / "docs" / name for name in ["PROJECT_STATE.md", "RESEARCH_DECISIONS.md",
                  "PROJECT_STATE_TECHNICAL_LINE.md", "RESEARCH_DECISIONS_TECHNICAL_LINE.md"]]
    backup = OUT / "documents_before_TECH_R204"
    backup.mkdir()
    header = ("\n\n> TECH.R204最新因果范围结论（2026-10-05，最近金融仍TECH.R198）：" + progress +
              " 2026补源可以用于另一个有完整合同的用途，不能改写本固定配置的早期失败。原R200/R202事实和原13前瞻值保持。\n\n"
              "下一步：" + next_action + " 当前无已准入待跑金融，独立/去过拟合与完整目标未达。\n\n"
              "依据：[完整因果边界及必要收益门](../" + relative(REPORT) + ")。\n\n")
    decision = ("假设 → 仅扩2026来源能够使原固定R198全四场景晋升。\n\n"
                "验证 → 2026前全部输入、共同成熟池和保存树精确复现；核对原两费用早期必需CAGR门，不重跑账户。\n\n"
                "结果 → " + progress + "\n\n"
                "接受/拒绝原因 → 接受时点一致性和来源事实；拒绝来源延长作为本配置晋升用途，早期必需门已失败且无法由未来信息改变。\n\n"
                "是否重验 → 仅真实来源/实现错误、实质不同完整用途或真正新样本；单独补2026不重开。\n\n")
    for path in documents:
        body = path.read_bytes()
        (backup / path.name).write_bytes(body)
        title, separator, rest = body.decode("utf-8-sig").partition("\n")
        study.require(bool(separator), "事实文件标题结构缺失。")
        added = header + (decision if path.name.startswith("RESEARCH_DECISIONS") else "")
        path.write_bytes((title + separator + added + rest).encode("utf-8"))
    study.require(STATE.read_bytes() == before, "发布期间根状态被并发修改，保留资料并停止覆盖。")
    study.write_json(STATE, state)
    study.write_json(OUT / "project_state_update_receipt.json", {
        "at": study.now(), "decision": "TECH.R204", "original_forward_keys_exact": FORWARD,
        "latest_actual_financial_decision": "TECH.R198", "source_only_rescue_route_closed": True,
        "state_sha256": study.digest(STATE),
        "documents": [{"path": relative(p), "sha256": study.digest(p)} for p in documents]}, exclusive=True)
    print("有限因果结论已写入四份长期事实；金融仍R198，原13前瞻值精确保持。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2026来源延长的有限因果范围核实")
    parser.add_argument("action", choices=["freeze", "run", "publish"])
    args = parser.parse_args()
    if args.action == "run":
        try:
            run()
        except Exception as error:
            if OUT.exists() and not (OUT / "failure.json").exists():
                study.write_json(OUT / "failure.json", {"at": study.now(), "type": type(error).__name__,
                                 "message": str(error), "new_fits": 0, "new_accounts": 0}, exclusive=True)
            raise
    else:
        {"freeze": freeze, "publish": publish}[args.action]()
