"""只核对月度风险目标和四个新账户，交付本轮结果。"""
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from research.three_policy_min_variance_v1 import ROOT, OUT as RESEARCH, CONFIG, PRIMARY, P82
from research.adaptive_allocation_v1 import normalize_dividends, summarize
from research.intraday_overnight_increment_v1 import now, require, write_json, digest
from scripts.review_round74_saved import saved_cycles
from scripts.finalize_round60_20260907 import metric, table

OUT = ROOT / "deliverables/510300三策略风险预算_第88轮_20260908"
DOCUMENT = OUT / "三策略风险预算_结果及中文规则.md"
INDEX = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
NEXT = ROOT / "docs/510300_AFTER_THREE_POLICY_CAUSAL_SELECTOR_20260908.md"


def main():
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    require(index["latest_completed_round"]["round"] == 87 and not OUT.exists(), "第88轮前序或交付状态不符")
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    result = json.loads((RESEARCH / "result.json").read_text(encoding="utf-8"))
    require(not result["historical_point_target_met"], "达到目标后须另做完整验收")
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    checks, cycles, differences, concentration, metrics = [], [], [], [], []
    for period in ["evaluation", "earlier_diagnostic"]:
        f = pd.read_parquet(RESEARCH / f"{period}_factors.parquet")
        first = int(np.flatnonzero(f.date >= pd.Timestamp(cfg["evaluation_start"] if period == "evaluation" else cfg["earlier_start"]))[0])
        weights = np.array([1/3, 1/3, 1/3])
        for t in range(first-1, len(f)-1):
            row = f.iloc[t]
            scheduled = t >= first and row.date.to_period("M") != f.date.iloc[t-1].to_period("M")
            require(scheduled == row.risk_update_scheduled, "非月首更新预算或漏掉月首")
            if scheduled:
                count = min(cfg["risk_window"], t-first+1)
                values = f.iloc[t-count+1:t+1][["panic_reference_return", "learned_reference_return", "breakout_reference_return"]].to_numpy()
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
                    scale=np.max(np.diag(covariance));eigenvalues=np.linalg.eigvalsh(covariance/scale)
                    if eigenvalues[0]<=np.finfo(float).eps*3*max(eigenvalues[-1],1.):
                        status="NO_VIEW_SINGULAR_COVARIANCE_KEEP_BUDGET"
                    else:
                        status="MIN_VARIANCE_BUDGET_AVAILABLE"
                        weights=np.array([row.panic_budget,row.learned_budget,row.breakout_budget])
                        objective=weights@covariance@weights;marginal=covariance@weights
                        require((weights>=0).all() and abs(weights.sum()-1)<1e-12,"三策略非负完整预算不符")
                        require(np.min(marginal-objective)>=-1e-13 and np.max(np.abs(weights*(marginal-objective)))<1e-13,"三维预算不满足凸全局最优条件")
                        require(objective<=previous@covariance@previous+1e-14,"三维更新后样本方差反而更大")
                        np.testing.assert_allclose([row.panic_sd,row.learned_sd,row.breakout_sd],np.sqrt(np.diag(covariance)),atol=1e-14,rtol=0)
                        np.testing.assert_allclose([row.panic_learned_covariance,row.panic_breakout_covariance,row.learned_breakout_covariance],covariance[np.triu_indices(3,1)],atol=1e-14,rtol=0)
                        require(abs(objective-row.estimated_portfolio_variance)<1e-14,"保存的组合方差不符")
                require(status == row.risk_status, "月度风险状态不符")
                checks.append({"period": period, "date": row.date, "status": status, "observations": count, "panic_budget": weights[0], "learned_budget": weights[1], "breakout_budget": weights[2]})
            np.testing.assert_allclose([row.panic_budget, row.learned_budget, row.breakout_budget], weights, atol=1e-14, rtol=0)
            np.testing.assert_allclose([row.target], [np.array([row.panic_state, row.learned_state, row.breakout_state]) @ weights], atol=1e-14, rtol=0, equal_nan=True)
        for cost in cfg["costs"]:
            ledger = pd.read_parquet(RESEARCH / period / cost / f"{PRIMARY}_ledger.parquet")
            decisions = pd.read_parquet(RESEARCH / period / cost / f"{PRIMARY}_decisions.parquet")
            np.testing.assert_allclose(decisions.reference_weight, f.target.iloc[first-1:-1], atol=0, rtol=0, equal_nan=True)
            previous = np.r_[cfg["initial_capital"], ledger.equity.iloc[:-1]]
            np.testing.assert_allclose(ledger.cash+ledger.shares*ledger.mark+ledger.dividend_receivable, ledger.equity, atol=1e-6, rtol=0)
            np.testing.assert_allclose(ledger.equity/previous-1, ledger.net_return, atol=1e-12, rtol=0)
            actual_by_date=ledger.set_index("date")
            prices=pd.read_parquet(ROOT/cfg["features"]).set_index("date").close
            for row in decisions.itertuples():
                equity=cfg["initial_capital"] if row.origin_index==first-1 else float(actual_by_date.loc[row.origin,"equity"])
                shares=0 if row.origin_index==first-1 else int(actual_by_date.loc[row.origin,"shares"])
                target=float(row.reference_weight);price=float(prices.loc[row.origin])
                quantity=0
                if np.isfinite(target):
                    desired=int(np.floor(target*equity/price/cfg["lot"]))*cfg["lot"]
                    if target>0 and shares>0 and abs(target-shares*price/equity)<cfg["weight_band"]:desired=shares
                    quantity=desired-shares
                require(quantity==row.requested_quantity,"三策略目标未按自身净值及实际股数下单")
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
            old = pd.read_parquet(P82 / period / cost / "TWO_POLICY_MIN_VARIANCE_ledger.parquet")
            differences.append({"period": period, "cost": cost, "terminal_nav_difference": float(ledger.equity.iloc[-1]-old.equity.iloc[-1]),
                **{f"{field}_difference": float(ledger[field].sum()-old[field].sum()) for field in ["price_pnl", "dividend_recognized", "commission", "slippage_cost"]}})
    for filename, rows in [("saved_budget_checks.csv", checks), ("saved_actual_cycles.csv", cycles), ("saved_account_checks.csv", metrics),
        ("saved_risk_budget_differences.csv", differences), ("saved_profit_concentration.csv", concentration)]:
        pd.DataFrame(rows).to_csv(RESEARCH / filename, index=False, encoding="utf-8-sig")
    receipt = {"verified_at": now(), "status": "KEY_MONTHLY_OPTIMUM_AND_COMPLETE_ACCOUNTS_CHECKED", "risk_update_checks": len(checks),
        "new_account_metrics_checked": len(metrics), "complete_cycles": len(cycles), "new_accounts_or_fits": 0,
        "reviewer_source_sha256": digest(Path(__file__)), "security_audit_performed": False}
    write_json(RESEARCH / "saved_verification_receipt.json", receipt, exclusive=True)
    status = "COMPLETED_THREE_POLICY_EARLY_IMPROVEMENT_MAIN_WEAKER_TARGET_NOT_MET"
    write_json(RESEARCH / "acceptance_outcome.json", {"recorded_at": now(), "status": status, "goal_achieved": False, "position_impact": 0,
        "decision": "较早夏普提高至0.727且回撤下降，主降至0.506；保留较早局部线索，未达1.2，不改第三策略、协方差、窗口或预算救回。"}, exclusive=True)
    OUT.mkdir(parents=True)
    lines=["# 第88轮：第三类交易机会的组合增量", "", "三策略组合在较早历史改善，但主评价明显变差。较早基础夏普由0.520升至0.727、回撤由11.07%降至7.51%；主基础夏普由1.137降至0.506、回撤由3.58%扩大至10.49%。没有实现1.2，不能把各时期事后表现更好的策略直接拼起来。", "",
        "## 主评价：2020年1月2日至2026年8月14日开盘", ""]+table(result["all_metrics"])
    lines += ["## 较早历史：2015年1月5日至2019年12月31日开盘", ""]+table(result["earlier_diagnostics"])
    lines += ["主基础年化2.49%，比两策略3.99%低；较早年化4.01%，高于两策略3.61%。主基础少赚23812.31元，其中价格损益少25668.40元、分红多3080.80元、费用多1224.71元；较早多赚4666.56元。新增机会带来的价格收益差异是主要原因，不能只归因交易费用。", "",
        "主每档74笔成交、421个持仓收盘，较早50笔、468个收盘，无整笔未成交或缺失目标。140个月首中，主27次有效、12次不足、41次一条参考零波动；较早14次有效、12次不足、34次零波动。没有奇异协方差或求解失败。较早有效更新减少，改善包含初始三分之一预算和后续保留预算的作用，不能全归因于每月优化。", "",
        "## 完整周期", "", "|时期|费用|完整周期|亏损周期|前三盈利周期之和占总净利润|", "|---|---|---:|---:|---:|"]
    for c in concentration:lines.append(f"|{'主评价' if c['period']=='evaluation' else '较早历史'}|{c['cost']}|{c['cycles']}|{c['losing_cycles']}|{c['largest_three_vs_net_profit']:.2%}|")
    lines += ["", "## 本轮预算与完整中文交易规则", ""]+(ROOT/cfg["rules"]).read_text(encoding="utf-8").splitlines()[1:]
    lines += ["", f"八项必要测试通过；核对{len(checks)}个月首时钟及三维全局最优条件、四个实际账户全部请求和指标、{len(cycles)}个含分红完整周期，无重跑旧策略。下一项只依据过去表现选择父组合，目前尚未登记或运行。", ""]
    DOCUMENT.write_text("\n".join(lines), encoding="utf-8")
    for p in RESEARCH.iterdir():
        if p.is_file() and p.suffix in {".csv", ".json"}:
            shutil.copy2(p, OUT / p.name)
    shutil.copy2(CONFIG, OUT / "冻结设置.json")
    shutil.copy2(ROOT / "docs/510300_TWO_POLICY_RISK_BUDGET_V1.md", OUT / "沿用的两条策略全部中文规则.md")
    coefficient = ROOT / "deliverables/510300下午提前进入_第75轮_20260908/沿用的每月八项模型中文规则.md"
    shutil.copy2(coefficient, OUT / coefficient.name)
    shutil.copy2(ROOT / "docs/510300_COMPRESSION_CONFIRMED_ENTRY_V1.md", OUT / "沿用的第三条突破全部中文规则.md")
    shutil.copy2(NEXT, OUT / "下一项研究方向.md")
    for period, label in [("evaluation", "主评价"), ("earlier_diagnostic", "较早历史")]:
        for cost in cfg["costs"]:
            for kind in ["ledger", "decisions"]:
                pd.read_parquet(RESEARCH / period / cost / f"{PRIMARY}_{kind}.parquet").to_csv(OUT / f"{label}_{cost}_{kind}.csv", index=False, encoding="utf-8-sig")
    record = {"round": 88, "study": result["study_id"], "title": "加入压缩后突破的三策略最小方差预算", "status": status,
        "result": str((RESEARCH / "result.json").relative_to(ROOT)), **{k: result[k] for k in ["candidate_configurations", "evaluation_accounts", "new_accounts_generated", "reused_control_accounts", "earlier_diagnostic_accounts", "new_earlier_diagnostic_accounts", "new_model_fits", "new_reference_accounts"]},
        "evaluated_candidate_source_runs": 1, "primary_base": metric(result, PRIMARY), "primary_stress": metric(result, PRIMARY, cost="STRESS"), "post_selected_best_base": metric(result, PRIMARY)}
    index["completed_rounds"].append(record)
    for key in ["registered_configurations_in_this_resumption", "evaluated_configurations_in_this_resumption", "evaluated_candidate_source_runs_including_corrected_replays", "registered_candidate_source_runs_including_unrun_legacy_bindings"]:
        index[key] += 1
    index["evaluation_accounts_in_this_resumption"] += result["evaluation_accounts"]
    index.update(updated_at=now(), latest_completed_round=record, running_studies=[], goal_achieved=False, status="ROUND88_COMPLETE_THREE_POLICY_EARLY_IMPROVEMENT_TARGET_NOT_MET",
        count_warning="88轮，362不同设置，378已评价来源版本，383登记含5旧未运行，1332主评价记录；无效来源保留。",
        next_work={"status": "PAST_SHARPE_PARENT_COMBINATION_SELECTOR_NOT_REGISTERED", "focus": "根据过去242完整日净夏普选择两套原组合或现金", "source": str(NEXT.relative_to(ROOT))},
        process_state_note="88四账户、关键核对与中文交付已完成；89过去表现选择父组合仅方向。")
    index["deliveries"].append({"created_at": now(), "type": "THREE_POLICY_ROUND88_FAST_CHINESE_RESULTS", "rounds": [88], "directory": str(OUT), "main_document": str(DOCUMENT), "new_gpt_review_archive_created": False})
    write_json(INDEX, index)
    write_json(OUT / "交付回执.json", {"created_at": now(), "main_document": str(DOCUMENT), "goal_achieved": False, "new_gpt_review_archive_created": False}, exclusive=True)
    print(json.dumps({"交付": str(DOCUMENT), "核对": receipt, "相对82差额": differences, "集中": concentration}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
