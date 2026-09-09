"""建立修正版核对与简洁交付，保留首次失败核对脚本。"""
from pathlib import Path
from research.intraday_overnight_increment_v1 import require

ROOT = Path(__file__).resolve().parents[1]


def main():
    dest = ROOT / "scripts/finalize_round81_source_fix_20260908.py"
    require(not dest.exists(), "修正版核对已建立")
    source = (ROOT / "scripts/finalize_round81_20260908.py").read_text(encoding="utf-8").split('    status = "COMPLETED_STAGED_RISK_REDUCTION_TARGET_NOT_MET"')[0]
    source = source.replace("from research.staged_entry_v1 import", "from research.staged_entry_v1_1 import")
    source = source.replace("P32, COEFFICIENTS", "P32, COEFFICIENTS, ORIGINAL")
    source = source.replace("row.shares_after", "row.shares")
    source = source.replace('"new_account_metrics_recomputed": len(metrics)', '"saved_account_metrics_recomputed": len(metrics), "corrected_accounts": 4, "reused_execution_control_accounts": 4')
    source += '''    status = "COMPLETED_SOURCE_CORRECTED_STAGING_TARGET_NOT_MET"
    old_result = json.loads((ORIGINAL / "result.json").read_text(encoding="utf-8"))
    source_differences = []
    for period in ["evaluation", "earlier_diagnostic"]:
        for cost in cfg["costs"]:
            require(digest(RESEARCH / period / cost / f"{EXECUTION_CONTROL}_ledger.parquet") == digest(ORIGINAL / period / cost / f"{EXECUTION_CONTROL}_ledger.parquet"), "原满仓对照字节改变")
            old_m, fixed_m = metric(old_result, PRIMARY, period, cost), metric(result, PRIMARY, period, cost)
            source_differences.append({"period": period, "cost": cost, "invalid_source_sharpe": old_m["net_sharpe"],
                "corrected_source_sharpe": fixed_m["net_sharpe"], "terminal_nav_correction": cfg["initial_capital"]*(fixed_m["cumulative_return"]-old_m["cumulative_return"])})
    pd.DataFrame(source_differences).to_csv(RESEARCH / "source_correction_impact.csv", index=False, encoding="utf-8-sig")
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "来源修正运行后冻结文件改变")
    write_json(RESEARCH / "acceptance_outcome.json", {"status": status, "recorded_at": now(), "goal_achieved": False,
        "decision": "修正实际持仓读取后，两段两费用夏普均低于同执行满仓；停止分批进入，不再投入尚未登记的组合分批检验，也不调比例或确认门槛。", "position_impact": 0}, exclusive=True)
    write_json(ORIGINAL / "acceptance_outcome.json", {"recorded_at": now(), "status": "INVALID_STAGING_SOURCE_REPLACED_BY_VERSION_2",
        "original_result_preserved": True, "original_rules_not_validly_evaluated": True, "valid_control_ledgers_reused": 4,
        "corrected_result": str((RESEARCH / "result.json").relative_to(ROOT)), "goal_achieved": False}, exclusive=True)
    OUT.mkdir(parents=True)
    main_base = metric(result, PRIMARY)
    early_base = metric(result, PRIMARY, "earlier_diagnostic")
    lines = ["# 第81轮：价格确认分批进入，来源修正后的结果", "",
        "修正后主评价基础／压力净夏普0.590／0.522，较早历史0.658／0.635，四项都低于同执行满仓对照，未达1.2。停止这项分批进入，不再追加尚未登记的组合分批试验，转向独立的两策略协同风险预算。", "",
        "首次曾报主0.712／0.646、较早0.697／0.677，是错误来源实现的结果，不能评价原冻结规则。代码把无成交日缺失的成交后股数当成每日实际持仓，提前清除了入场基准；核对在写验收和总索引前发现问题。修正只读取完整的每日实际股数，原策略、首批比例和确认门槛均不改变。首次六项合成测试没有模拟无成交日字段缺失；已新增真实结构回归测试，七项通过。原来源与原数字保留且标为无效分批实现。", "",
        "## 主评价：2020年1月2日至2026年8月14日开盘", ""] + table(result["all_metrics"])
    lines += ["## 较早历史：2015年1月5日至2019年12月31日开盘", ""] + table(result["earlier_diagnostics"])
    lines += ["主24参考周期中22次确认补足，半仓84个目标日、满仓169日；较早9周期全部确认，半仓20日、满仓279日。主每档46买24卖，较早18买9卖。实际持仓收盘仍为252和299，均无未成交；主非零目标253日比实际持仓多1，来自终点统一开盘退出。", "",
        f"基础主年化{main_base['annualized_return']:.2%}、最大回撤{abs(main_base['max_drawdown']):.2%}；较早年化{early_base['annualized_return']:.2%}、回撤{abs(early_base['max_drawdown']):.2%}。两段收益和夏普均低于同执行满仓，不能把较低投入或回撤当成已经改善效率。", "",
        f"修正只新跑4个受影响账户；4个满仓对照账本原字节复用，另16个老对照复用。核对{len(stages)}个决策时点、8份保存账户指标与{len(cycles)}个含分红周期，未重训。两来源合计12个实际新账户，一个不同设置、两个已运行来源版本、24份主评价记录。", "",
        "## 进出场和全部因子中文规则", ""]
    lines += (ROOT / cfg["rules"]).read_text(encoding="utf-8").splitlines()[1:]
    lines += ["", "## 本次来源修正的优先说明", ""]
    lines += (ROOT / cfg["source_amendment"]).read_text(encoding="utf-8").splitlines()[1:]
    DOCUMENT.write_text("\\n".join(lines)+"\\n", encoding="utf-8")
    NEXT.write_text("# 第81轮修正完成，停止分批，改研究两策略协同风险\\n\\n"
        "修正后主基础／压力夏普0.590169／0.521870，较早0.657748／0.635056，四项均低于满仓。此前未核完就写入本文件的0.711855等数字属于错误持仓字段来源，已在第81轮原目录保留并标记无效。组合分批尚未登记，现停止该方向，避免在明确变差的执行法上继续花时间。\\n\\n"
        "下一82拟将两条原参考策略视作资金使用规则，用过去完整净收益的方差及两者协方差选择合成参考收益方差最小的非负预算；不估计预期收益，不借款。沿用76的242日、月首、初始各半及缺失或零风险时沿用预算。与76倒数波动分配的目标不同：该项直接考虑两条规则同时波动，最小化组合总方差。尚未登记，先有界查重并固定唯一公式与退化条件，再跑四账户。\\n", encoding="utf-8")
    for p in RESEARCH.iterdir():
        if p.is_file() and p.suffix in {".csv", ".json"}:
            shutil.copy2(p, OUT / p.name)
    shutil.copy2(CONFIG, OUT / "修正来源冻结设置.json")
    shutil.copy2(ORIGINAL / "source_defect_receipt.json", OUT / "首次来源缺陷回执.json")
    shutil.copy2(NEXT, OUT / "下一项研究方向.md")
    shutil.copy2(COEFFICIENTS, OUT / COEFFICIENTS.name)
    for period, label in [("evaluation", "主评价"), ("earlier_diagnostic", "较早历史")]:
        for cost in cfg["costs"]:
            for model in [PRIMARY, EXECUTION_CONTROL]:
                for kind in ["ledger", "decisions"]:
                    pd.read_parquet(RESEARCH / period / cost / f"{model}_{kind}.parquet").to_csv(OUT / f"{label}_{cost}_{model}_{kind}.csv", index=False, encoding="utf-8-sig")
    record = {"round": 81, "study": result["study_id"], "title": "先半仓、参考价格确认后补足，实际持仓来源修正版", "status": status,
        "result": str((RESEARCH / "result.json").relative_to(ROOT)), "original_invalid_result": str((ORIGINAL / "result.json").relative_to(ROOT)),
        "candidate_configurations": 1, "evaluated_candidate_source_runs": 2, "evaluation_accounts": 24,
        "original_evaluation_accounts": 12, "corrected_evaluation_accounts": 12, "new_accounts_generated": 6,
        "new_execution_control_accounts": 2, "new_correction_accounts": 2, "reused_control_accounts": 18,
        "earlier_diagnostic_accounts": 24, "new_earlier_diagnostic_accounts": 6, "new_model_fits": 0, "new_reference_accounts": 0,
        "primary_base": metric(result, PRIMARY), "primary_stress": metric(result, PRIMARY, cost="STRESS"), "post_selected_best_base": metric(result, PRIMARY)}
    index["completed_rounds"].append(record)
    for k in ["registered_configurations_in_this_resumption", "evaluated_configurations_in_this_resumption"]:
        index[k] += 1
    for k in ["evaluated_candidate_source_runs_including_corrected_replays", "registered_candidate_source_runs_including_unrun_legacy_bindings"]:
        index[k] += 2
    index["evaluation_accounts_in_this_resumption"] += 24
    index.update(updated_at=now(), latest_completed_round=record, running_studies=[], goal_achieved=False, status="ROUND81_SOURCE_CORRECTED_COMPLETE_TARGET_NOT_MET",
        count_warning="81轮，355不同设置，371已评价来源版本，376登记含5旧未运行，1244主评价记录；包括首次无效分批来源。",
        next_work={"status": "TWO_REFERENCE_MINIMUM_VARIANCE_NOT_REGISTERED", "focus": "两原策略方差与协方差的最小方差预算", "source": str(NEXT.relative_to(ROOT))},
        process_state_note="80已完成；81来源缺陷修正、四受影响账户及关键核对完成。未登记的组合分批已停止；82协同风险只有方向。",
        research_speed_priority="docs/510300_FAST_RESEARCH_CADENCE_20260908.md")
    index["deliveries"].append({"created_at": now(), "type": "STAGED_ENTRY_SOURCE_FIX_ROUND81_FAST_CHINESE_RESULTS", "rounds": [81],
        "directory": str(OUT), "main_document": str(DOCUMENT), "new_gpt_review_archive_created": False})
    write_json(INDEX, index)
    write_json(OUT / "交付回执.json", {"created_at": now(), "main_document": str(DOCUMENT), "source_versions": 2, "goal_achieved": False, "new_gpt_review_archive_created": False}, exclusive=True)
    print(json.dumps({"交付": str(DOCUMENT), "核对": receipt, "来源修正影响": source_differences}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
'''
    dest.write_text(source, encoding="utf-8")
    print("修正版核对脚本已建立，首次脚本原样保留。", flush=True)


if __name__ == "__main__":
    main()
