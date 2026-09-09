"""只核对月度风险目标和四个新账户，交付本轮结果。"""
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from research.two_policy_min_variance_v1 import ROOT, OUT as RESEARCH, CONFIG, PRIMARY, P76
from research.adaptive_allocation_v1 import normalize_dividends, summarize
from research.intraday_overnight_increment_v1 import now, require, write_json, digest
from scripts.review_round74_saved import saved_cycles
from scripts.finalize_round60_20260907 import metric, table

OUT = ROOT / "deliverables/510300最小方差预算_第82轮_20260908"
DOCUMENT = OUT / "最小方差预算_结果及中文规则.md"
INDEX = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
NEXT = ROOT / "docs/510300_AFTER_MIN_VARIANCE_TAIL_LOSS_20260908.md"


def main():
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    require(index["latest_completed_round"]["round"] == 81 and not OUT.exists(), "第82轮前序或交付状态不符")
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    result = json.loads((RESEARCH / "result.json").read_text(encoding="utf-8"))
    require(not result["historical_point_target_met"], "达到目标后须另做完整验收")
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    checks, cycles, differences, concentration, metrics = [], [], [], [], []
    for period in ["evaluation", "earlier_diagnostic"]:
        f = pd.read_parquet(RESEARCH / f"{period}_factors.parquet")
        first = int(np.flatnonzero(f.date >= pd.Timestamp(cfg["evaluation_start"] if period == "evaluation" else cfg["earlier_start"]))[0])
        weights = np.array([.5, .5])
        for t in range(first-1, len(f)-1):
            row = f.iloc[t]
            scheduled = t >= first and row.date.to_period("M") != f.date.iloc[t-1].to_period("M")
            require(scheduled == row.risk_update_scheduled, "非月首更新预算或漏掉月首")
            if scheduled:
                count = min(cfg["risk_window"], t-first+1)
                values = f.iloc[t-count+1:t+1][["panic_reference_return", "learned_reference_return"]].to_numpy()
                require(count == row.risk_window_observations and f.date.iloc[t-count+1] == row.risk_window_start, "借用起点之前参考收益")
                previous = weights.copy()
                if count < cfg["risk_window"]:
                    status = "NO_VIEW_WARMUP_KEEP_BUDGET"
                elif not np.isfinite(values).all():
                    status = "NO_VIEW_INCOMPLETE_WINDOW_KEEP_BUDGET"
                elif (values.std(axis=0, ddof=1) <= 0).any():
                    status = "NO_VIEW_ZERO_OR_INVALID_RISK_KEEP_BUDGET"
                else:
                    covariance = np.cov(values, rowvar=False, ddof=1)
                    denominator = np.var(values[:, 0]-values[:, 1], ddof=1)
                    if denominator <= 0:
                        status = "NO_VIEW_DEGENERATE_DIFFERENCE_KEEP_BUDGET"
                    else:
                        status = "MIN_VARIANCE_BUDGET_AVAILABLE"
                        weights = np.array([row.panic_budget, row.learned_budget])
                        w = weights[0]
                        derivative = 2*(w*covariance[0, 0]-(1-w)*covariance[1, 1]+(1-2*w)*covariance[0, 1])
                        require((0 < w < 1 and abs(derivative) < 1e-13) or (w == 0 and derivative >= -1e-13) or (w == 1 and derivative <= 1e-13), "预算不满足非负最小方差最优条件")
                        require(weights @ covariance @ weights <= previous @ covariance @ previous+1e-14, "月度优化后的样本方差反而更大")
                        np.testing.assert_allclose([row.reference_covariance, row.difference_variance], [covariance[0, 1], denominator], atol=1e-14, rtol=0)
                require(status == row.risk_status, "月度风险状态不符")
                checks.append({"period": period, "date": row.date, "status": status, "observations": count, "panic_budget": weights[0], "learned_budget": weights[1]})
            np.testing.assert_allclose([row.panic_budget, row.learned_budget], weights, atol=1e-14, rtol=0)
            np.testing.assert_allclose([row.target], [np.array([row.panic_state, row.learned_state]) @ weights], atol=1e-14, rtol=0, equal_nan=True)
        for cost in cfg["costs"]:
            ledger = pd.read_parquet(RESEARCH / period / cost / f"{PRIMARY}_ledger.parquet")
            decisions = pd.read_parquet(RESEARCH / period / cost / f"{PRIMARY}_decisions.parquet")
            np.testing.assert_allclose(decisions.reference_weight, f.target.iloc[first-1:-1], atol=0, rtol=0, equal_nan=True)
            previous = np.r_[cfg["initial_capital"], ledger.equity.iloc[:-1]]
            np.testing.assert_allclose(ledger.cash+ledger.shares*ledger.mark+ledger.dividend_receivable, ledger.equity, atol=1e-6, rtol=0)
            np.testing.assert_allclose(ledger.equity/previous-1, ledger.net_return, atol=1e-12, rtol=0)
            actual, saved = summarize(ledger, cfg), metric(result, PRIMARY, period, cost)
            for field in ["net_sharpe", "annualized_return", "max_drawdown", "cumulative_return", "trade_count", "commission", "slippage_cost"]:
                require(abs(actual[field]-saved[field]) < 1e-9, "新账户保存指标不符")
            metrics.append({"period": period, "cost": cost, **actual})
            group = saved_cycles(ledger, dividends, cfg)
            cycles.extend({"period": period, "cost": cost, **c} for c in group)
            total = sum(c["net_profit"] for c in group)
            top = sorted((c["net_profit"] for c in group if c["net_profit"] > 0), reverse=True)[:3]
            concentration.append({"period": period, "cost": cost, "cycles": len(group), "losing_cycles": sum(c["net_profit"] < 0 for c in group),
                "net_profit": total, "largest_three_positive_profit": sum(top), "largest_three_vs_net_profit": sum(top)/total if total > 0 else None})
            old = pd.read_parquet(P76 / period / cost / "TWO_POLICY_RISK_BUDGET_ledger.parquet")
            differences.append({"period": period, "cost": cost, "terminal_nav_difference": float(ledger.equity.iloc[-1]-old.equity.iloc[-1]),
                **{f"{field}_difference": float(ledger[field].sum()-old[field].sum()) for field in ["price_pnl", "dividend_recognized", "commission", "slippage_cost"]}})
    for filename, rows in [("saved_budget_checks.csv", checks), ("saved_actual_cycles.csv", cycles), ("saved_account_checks.csv", metrics),
        ("saved_risk_budget_differences.csv", differences), ("saved_profit_concentration.csv", concentration)]:
        pd.DataFrame(rows).to_csv(RESEARCH / filename, index=False, encoding="utf-8-sig")
    receipt = {"verified_at": now(), "status": "KEY_MONTHLY_OPTIMUM_AND_COMPLETE_ACCOUNTS_CHECKED", "risk_update_checks": len(checks),
        "new_account_metrics_checked": len(metrics), "complete_cycles": len(cycles), "new_accounts_or_fits": 0,
        "reviewer_source_sha256": digest(Path(__file__)), "security_audit_performed": False}
    write_json(RESEARCH / "saved_verification_receipt.json", receipt, exclusive=True)
    status = "COMPLETED_MIN_VARIANCE_MAIN_IMPROVEMENT_EARLY_WEAKER_TARGET_NOT_MET"
    write_json(RESEARCH / "acceptance_outcome.json", {"recorded_at": now(), "status": status, "goal_achieved": False, "position_impact": 0,
        "decision": "主夏普局部提高至1.137，较早降至0.520，不达1.2或稳定超额。保留局部线索，不调协方差收缩、窗口、预算限制救回。"}, exclusive=True)
    NEXT.write_text("# 第82轮完成，下一项只检验尾部亏损预算\n\n"
        "82主基础／压力夏普1.137069／1.085300，较早0.519787／0.485808；76对照为1.091229／1.031785及0.559053／0.525603。主年化3.9896%低于原4.0525%，回撤3.5847%也高于原3.2444%；主夏普提高来自较低完整日波动，未提高总利润。较早年化3.6094%，回撤11.07495%，更弱。无1.2，历史不是独立证据。\n\n"
        "下一83拟检验另一种风险目标：过去242完整日中，预算合成参考收益最差5%日的平均损失，即95%条件风险价值；只考虑亏损尾部，避免把大幅盈利日与亏损日同等惩罚。零现金日仍保留，原月首时钟、两策略状态、初始各半及支持不足保持。只固定95%，不扫描置信度或预算上限。最优预算并列时优先最接近此前预算，避免求解器随意选择边界。它不保证解决82较早退步。\n\n"
        "有界检索发现旧downside_risk_budget_v1为四技术因子预测未来20日不利路径后再分仓，非两条成熟策略过去日收益的尾部损失优化；旧失败继续保留。尾部预算尚未登记或运行，需固定经验尾部非整数权重及并列解处理、必要测试后四完整账户。\n", encoding="utf-8")
    OUT.mkdir(parents=True)
    lines = ["# 第82轮：考虑两策略协方差的资金预算", "",
        "主评价基础／压力净夏普1.137／1.085，比第76轮的1.091／1.032提高；较早历史却从0.559／0.526降为0.520／0.486。没有达到1.2，也没有证实稳定超额。", "",
        "## 主评价：2020年1月2日至2026年8月14日开盘", ""] + table(result["all_metrics"])
    lines += ["## 较早历史：2015年1月5日至2019年12月31日开盘", ""] + table(result["earlier_diagnostics"])
    lines += ["主基础年化从第76轮4.05%降至3.99%，最大回撤从3.24%升至3.58%；提高的是夏普，不能描述成所有指标改善。较早年化从3.86%降至3.61%，回撤仍为11.07%。两段利润相对原预算的价格、分红与费用差额另附CSV。", "",
        "140个月首尝试中，主80次包含12次历史不足、41次一条参考零波动、27次有效；较早60次包含12次不足、16次零波动、32次有效，没有退化差值。有效更新的平均急跌预算主76.94%、较早69.20%，没有最优解落到零或一边界。交易稀少仍影响风险估计。", "",
        "主两费用各47笔成交、208个持仓收盘；较早各35笔、339收盘，无整笔未成交和缺失目标。2023年完整现金日保留，不能计算该年的夏普；少数年度夏普超过1.2不能代替全期目标。", "",
        "## 利润集中情况", "", "|时期|费用|完整周期|亏损周期|最大三个盈利周期之和占总净利润|", "|---|---|---:|---:|---:|"]
    for c in concentration:
        lines.append(f"|{'主评价' if c['period']=='evaluation' else '较早历史'}|{c['cost']}|{c['cycles']}|{c['losing_cycles']}|{c['largest_three_vs_net_profit']:.2%}|")
    lines += ["", "比值超过100%表示前三个盈利周期还覆盖了其他周期的净亏损，不能解释为所有交易普遍有效。", "",
        "## 本轮新增预算与交易规则", ""] + (ROOT / cfg["rules"]).read_text(encoding="utf-8").splitlines()[1:]
    lines += ["", "全部原急跌和学习进入、八项因子、退出及重新进入规则见同目录《沿用的两条策略全部中文规则.md》；逐月原模型系数直接复用已有中文文件。", "",
        f"10项必要测试通过，已检查{len(checks)}次月度时钟与最优性、四个新账户指标和{len(cycles)}个含分红周期；没有重复运行旧账户或训练模型。下一项尾部亏损预算只有研究方向。", ""]
    DOCUMENT.write_text("\n".join(lines), encoding="utf-8")
    for p in RESEARCH.iterdir():
        if p.is_file() and p.suffix in {".csv", ".json"}:
            shutil.copy2(p, OUT / p.name)
    shutil.copy2(CONFIG, OUT / "冻结设置.json")
    shutil.copy2(ROOT / "docs/510300_TWO_POLICY_RISK_BUDGET_V1.md", OUT / "沿用的两条策略全部中文规则.md")
    coefficient = ROOT / "deliverables/510300下午提前进入_第75轮_20260908/沿用的每月八项模型中文规则.md"
    shutil.copy2(coefficient, OUT / coefficient.name)
    shutil.copy2(NEXT, OUT / "下一项研究方向.md")
    for period, label in [("evaluation", "主评价"), ("earlier_diagnostic", "较早历史")]:
        for cost in cfg["costs"]:
            for kind in ["ledger", "decisions"]:
                pd.read_parquet(RESEARCH / period / cost / f"{PRIMARY}_{kind}.parquet").to_csv(OUT / f"{label}_{cost}_{kind}.csv", index=False, encoding="utf-8-sig")
    record = {"round": 82, "study": result["study_id"], "title": "两条原策略非负最小方差预算", "status": status,
        "result": str((RESEARCH / "result.json").relative_to(ROOT)), **{k: result[k] for k in ["candidate_configurations", "evaluation_accounts", "new_accounts_generated", "reused_control_accounts", "earlier_diagnostic_accounts", "new_earlier_diagnostic_accounts", "new_model_fits", "new_reference_accounts"]},
        "evaluated_candidate_source_runs": 1, "primary_base": metric(result, PRIMARY), "primary_stress": metric(result, PRIMARY, cost="STRESS"), "post_selected_best_base": metric(result, PRIMARY)}
    index["completed_rounds"].append(record)
    for key in ["registered_configurations_in_this_resumption", "evaluated_configurations_in_this_resumption", "evaluated_candidate_source_runs_including_corrected_replays", "registered_candidate_source_runs_including_unrun_legacy_bindings"]:
        index[key] += 1
    index["evaluation_accounts_in_this_resumption"] += result["evaluation_accounts"]
    index.update(updated_at=now(), latest_completed_round=record, running_studies=[], goal_achieved=False, status="ROUND82_COMPLETE_MAIN_1_137_EARLY_WEAKER_TARGET_NOT_MET",
        count_warning="82轮，356不同设置，372已评价来源版本，377登记含5旧未运行，1256主评价记录；无效来源保留。",
        next_work={"status": "TWO_REFERENCE_CVAR_BUDGET_NOT_REGISTERED", "focus": "两条原策略的95%尾部亏损最小预算", "source": str(NEXT.relative_to(ROOT))},
        process_state_note="81来源修正及82完整账户、关键核对、简洁交付完成；83尾部风险仅方向。")
    index["deliveries"].append({"created_at": now(), "type": "MIN_VARIANCE_ROUND82_FAST_CHINESE_RESULTS", "rounds": [82], "directory": str(OUT), "main_document": str(DOCUMENT), "new_gpt_review_archive_created": False})
    write_json(INDEX, index)
    write_json(OUT / "交付回执.json", {"created_at": now(), "main_document": str(DOCUMENT), "goal_achieved": False, "new_gpt_review_archive_created": False}, exclusive=True)
    print(json.dumps({"交付": str(DOCUMENT), "核对": receipt, "相对76差额": differences, "集中": concentration}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
