"""复算已保存诊断结果，并登记固定信号的瓶颈结论。"""
from pathlib import Path

import numpy as np
import pandas as pd

from research import factor96_bottleneck_diagnostic_v1 as study
from research import factor96_margin_repair_v1 as core
from scripts.record_factor96_remaining_changes_v1 import update


def main():
    root, out = study.ROOT, study.OUT
    study.read(out / "run_completed.json")
    metrics = pd.read_csv(out / "all_counterfactual_metrics.csv")
    assert len(metrics) == 294 and metrics.reproduction.sum() == 49
    assert not metrics.duplicated(["case_id", "mode", "cost"]).any()
    max_error = 0.
    for row in metrics.to_dict("records"):
        ledger = pd.read_parquet(root / row["ledger_path"])
        recomputed = core.metrics(ledger, row["capital"])
        for key in ["net_sharpe", "cagr", "max_drawdown", "end_equity", "commission", "slippage", "terminal_exit_reserve"]:
            if recomputed[key] is None:
                assert pd.isna(row[key])
            else:
                np.testing.assert_allclose(row[key], recomputed[key], rtol=1e-10, atol=1e-8)
        max_error = max(max_error, recomputed["max_identity_error"])
        if row["cost"] == "ZERO":
            assert recomputed["commission"] == 0 and recomputed["slippage"] == 0 and recomputed["terminal_exit_reserve"] == 0
        if row["mode"] != "ORIGINAL":
            assert not ledger.risk_stopped.any()
    fields = ["case_id", "policy", "start", "end", "net_sharpe", "cagr", "max_drawdown", "entries", "mean_exposure"]
    scenarios = []
    for (mode, cost), frame in metrics.groupby(["mode", "cost"], sort=False):
        scenarios.append({"mode": mode, "cost": cost, "cases": len(frame), "nonzero_return_cases": int(frame.net_sharpe.notna().sum()),
            "best_sharpe_case": frame.loc[frame.net_sharpe.idxmax(), fields].to_dict(),
            "best_cagr_case": frame.loc[frame.cagr.idxmax(), fields].to_dict(),
            "median_sharpe": frame.net_sharpe.median(), "median_cagr": frame.cagr.median(),
            "median_max_drawdown": frame.max_drawdown.median(), "median_mean_exposure": frame.mean_exposure.median(),
            "sharpe_target_count": int(frame.sharpe_at_least_1_2.sum()),
            "joint_target_count": int(frame.joint_sharpe_and_cagr.sum()),
            "joint_with_drawdown_count": int(frame.joint_with_drawdown.sum())})
    period_rows = []
    for (start, mode, cost), frame in metrics.groupby(["start", "mode", "cost"], sort=False):
        period_rows.append({"start": start, "end": "2025-12-31", "mode": mode, "cost": cost, "cases": len(frame),
            "max_sharpe": frame.net_sharpe.max(), "median_sharpe": frame.net_sharpe.median(),
            "max_cagr": frame.cagr.max(), "target_count": int(frame.joint_with_drawdown.sum())})
    pd.DataFrame(period_rows).to_csv(out / "scenario_summary_by_period.csv", index=False, encoding="utf-8-sig")
    attribution = pd.read_csv(out / "fixed_holdings_cost_attribution.csv")
    attribution = attribution.loc[~attribution.policy.isin(["BUY_HOLD_50", "CASH"])]
    cost_summary = []
    for capital, frame in attribution.groupby("capital"):
        cost_summary.append({"capital": int(capital), "cases": len(frame), "max_rebated_sharpe": frame.rebated_sharpe.max(),
            "median_paired_sharpe_improvement": frame.sharpe_change.median(),
            "mean_cagr_improvement": frame.cagr_change.mean(), "max_rebated_cagr": frame.rebated_cagr.max(),
            "rebated_sharpe_target_count": int(frame.rebated_sharpe.ge(1.2).sum())})
    p01 = metrics.loc[metrics.case_id.eq("feasibility__MAIN__P01_HIGH")]
    risk = pd.read_csv(out / "original_risk_path_diagnostic.csv")
    result = {
        "at": core.now(), "study_id": study.STUDY, "status": "COMPLETED_BOTTLENECK_IDENTIFIED_IN_FIXED_RULES",
        "main_finding": "近期固定信号与既定持有期的收益优势不足，是夏普1.2的主要瓶颈；费用侵蚀显著、风控压低收益，但移除二者仍未达标。",
        "scope": "六轮49组规则含对照与重复规则；2017-2025为46组，ETF迁移2022-2025为3组。不是49个独立信号。",
        "maximum_sharpe_all_scenarios": metrics.net_sharpe.max(), "maximum_cagr_all_scenarios": metrics.cagr.max(),
        "sharpe_target_count": int(metrics.sharpe_at_least_1_2.sum()),
        "joint_historical_target_count": int(metrics.joint_with_drawdown.sum()),
        "cost_attribution": cost_summary, "factorial_summaries": scenarios,
        "illustrative_same_signal": "P01_HIGH为本诊断中零成本下事后表现最好的一条，用于解释配对差异，未复活原失败策略。",
        "p01_paired_scenarios": p01[["mode", "cost"]+fields].to_dict("records"),
        "original_risk_path": {"actual_permanent_stops": int(risk.first_stop.notna().sum()),
            "median_mean_exposure": risk.mean_exposure.median(), "median_risk_reduction_fills": risk.risk_reductions.median(),
            "interpretation": "49份原压力账本均未实际触发永久停止；观察到的账户约束影响来自风险预算和持续减仓等机制，不能归咎于已经停机。平均敞口也受信号稀疏影响。"},
        "limits": ["没有测试全部可能信号、持有或退出机制，不构成510300夏普1.2不可能的证明。",
            "满仓无费用仅为诊断；实际仍需费用和原账户约束。T+1、整手、现金约束和固定持有间隔未移除。",
            "2万元仅做固定持仓返费归因；完整账户反事实只覆盖20万元。",
            "样本全部是已经观察过的历史；事后最高值不代表独立样本外能力。",
            "零成本原风控重新模拟会改变仓位路径，不等于固定持仓返费归因。费用与风控影响不可直接相加。"],
        "next_research_implication": "不再通过给本批旧规则调手续费或提高仓位寻求达标；下一步需更换收益假设、信号信息源或持有退出机制，先验证费用前收益优势，再展开账户实现。",
        "reproduced_accounts": 49, "new_counterfactual_accounts": 245, "fixed_holdings_attributions": 116,
        "new_admitted_strategies": 0, "independent_forward_observations": 0,
        "goal_status": "active", "goal_achieved": False, "qualified_candidates": [],
        "orders_authorized": False, "delivery_package_required": False}
    study.save(out / "result.json", result)
    study.save(out / "saved_verification_receipt.json", {"at": core.now(), "status": "PASS_SAVED_METRICS_RECOMPUTATION",
        "metrics_rows_recomputed": 294, "original_reproductions": 49, "max_accounting_identity_error": max_error,
        "zero_cost_ledgers_have_zero_execution_and_terminal_fees": True,
        "new_admitted_strategies": 0, "forward_observations": 0})
    labels = {("ORIGINAL", "STRESS"): "原账户、压力费用", ("ORIGINAL", "ZERO"): "原风控、零费用",
              ("CAP50_ONLY", "STRESS"): "仅半仓入场限制、压力费用", ("CAP50_ONLY", "ZERO"): "仅半仓入场限制、零费用",
              ("CAP100_ONLY", "STRESS"): "允许满仓、压力费用", ("CAP100_ONLY", "ZERO"): "允许满仓、零费用"}
    lines = ["近期固定规则的主要瓶颈是收益优势不足。费用和风控确实拖低结果，但移除费用和动态风控后，49组规则中最高夏普仍只有0.694；允许满仓后最高0.684，未达到1.2。所有情形的最高年化8.57%，也没有达到联合年化10%的目标。",
        "本次沿用六轮已保存信号与持有期，先逐日复现49份原账本，再跑245个反事实账户；另对116份20万元/2万元压力账本做固定持仓返费归因。信号与参数均未修改。49组包含对照和重复规则；46组为2017—2025年，ETF迁移3组为2022—2025年，分时段统计另存。",
        "为隔离影响，下表始终使用同一条P01_HIGH（20日下行平方收益占比高分位）规则、20万元账户、2017—2025年。它是本轮无费用反事实中事后表现最好的一条，仅作说明，原失败状态不变。",
        "| 账户情形 | 夏普 | 年化收益 | 最大回撤 |\n|---|---:|---:|---:|"]
    for row in p01.itertuples():
        lines.append(f"| {labels[(row.mode, row.cost)]} | {row.net_sharpe:.3f} | {row.cagr:.2%} | {row.max_drawdown:.2%} |")
    lines += ["", "费用：原持仓保持不变，把累计佣金、滑点和期末清算准备全部返还，20万元最高夏普0.577，2万元最高0.579。费用侵蚀弱收益，但仅降低费用不足以接近1.2。固定持仓返费不再投资，与零费用重新模拟不同。",
        "账户约束：P01在压力费用下，放开动态风控但保留半仓，年化从0.53%升至2.94%，回撤从8.46%扩大到16.33%；进一步允许满仓，年化5.27%、回撤30.60%。约束明显压低收益，但撤掉约束并未产生足够高的风险调整收益。",
        "49份原压力账本没有实际触发10%回撤永久停机。持续风险减仓和仓位预算是本次观察到的约束机制；全期平均仓位较低也包含信号稀疏的影响，不能全部归因于风控。",
        "仓位本身不是夏普来源：在零费用且收益序列按同一常数缩放的理想情形，均值和标准差同倍变化，夏普不变；本次整手与仓位漂移使其不完全相同，但半仓、满仓对照仍符合这一判断。方法参考：[William F. Sharpe, The Sharpe Ratio](https://web.stanford.edu/~wfsharpe/art/sr/SR.htm)。",
        "研究决策：停止把本批旧信号的手续费、仓位大小作为主攻方向。要继续实现目标，需要更换收益假设、信息源或持有退出机制；先快速检验费用前的优势，再展开账户实现。此次没有证明其他510300策略不可能成功，也没有把放宽后的模拟计为达标策略。",
        "所有空仓日计入账户，现金收益与现金基准均为0、年化242交易日。压力费用单边佣金万分之四（最低5元）、滑点千分之一并按价格档位不利取整；分红、T+1、整手和涨跌停约束保留。2万元仅做费用归因，未做完整反事实。历史结果不等于独立验证。目标尚未实现，新增合格策略0。",
        "结果：all_counterfactual_metrics.csv；费用归因：fixed_holdings_cost_attribution.csv；逐项差值：paired_changes_from_original.csv；分区间统计：scenario_summary_by_period.csv。"]
    text = "\n\n".join(lines[:3])+"\n\n"+"\n".join(lines[3:10])+"\n\n"+"\n\n".join(lines[10:])
    (out / "瓶颈诊断结论.md").write_text(text+"\n", encoding="utf-8")
    paths = [root / "reports/research/510300_factor96_program_v1/status.json", root / "config/510300_existing_data_training_mandate_v1.json"]
    status, mandate = [study.read(path) for path in paths]
    before = [{"path": str(path), "sha256": core.digest(path)} for path in paths]
    relative = out.relative_to(root).as_posix()
    note = "49份原账户复现、245个固定信号反事实完成；最高夏普0.694，放宽费用/风控/仓位仍未达1.2；主要瓶颈为信号与既定持有期的收益优势不足。"
    status.update(at=core.now(), latest_round=study.STUDY, latest_result=relative+"/result.json",
        last_research_result=note, latest_progress_receipt=relative+"/saved_verification_receipt.json",
        latest_continuation_classification="PROGRESS_BOTTLENECK_DIAGNOSIS_COMPLETED",
        admitted_account_scenarios_this_round=0, invalid_implementation_accounts_this_round=0,
        diagnostic_reproductions_this_round=49, counterfactual_account_scenarios_this_round=245,
        cumulative_counterfactual_account_scenarios=status.get("cumulative_counterfactual_account_scenarios", 0)+245,
        counterfactual_account_counting="诊断另计，不加入原策略准入/执行计数；复现不重复计数。",
        current_research_phase="BOTTLENECK_IDENTIFIED_SIGNAL_AND_HOLDING_EDGE_INSUFFICIENT",
        next_independent_source_action="REASSESS_NEW_RETURN_MECHANISM_AFTER_BOTTLENECK_DIAGNOSIS",
        research_priority="NEW_GROSS_EDGE_BEFORE_ACCOUNT_OPTIMIZATION", goal_status="active", goal_achieved=False,
        qualified_candidates=[], orders_authorized=False)
    mandate.update(current_round=study.STUDY, current_protocol=relative+"/protocol.json", last_research_result=note,
        latest_progress_receipt=relative+"/saved_verification_receipt.json", latest_continuation_report=relative+"/瓶颈诊断结论.md",
        latest_continuation_classification=status["latest_continuation_classification"], research_execution_state=status["current_research_phase"],
        research_priority=status["research_priority"], goal_status="active", goal_achieved=False)
    for path, value in zip(paths, [status, mandate]):
        update(path, core.clean(value))
    study.save(out / "program_update_receipt.json", {"at": core.now(), "before": before,
        "after": [{"path": str(path), "sha256": core.digest(path)} for path in paths],
        "new_admitted_strategies": 0, "new_counterfactual_accounts": 245, "goal_achieved": False})
    print(note, flush=True)


if __name__ == "__main__":
    main()
