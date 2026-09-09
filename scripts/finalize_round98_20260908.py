"""独立核对月度收益风险切点与明确现金，再交付保存账户。"""
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from numpy.polynomial.legendre import leggauss
from research.tangency_reference_budget_v1 import ROOT, OUT as RESEARCH, CONFIG, P91, PRIMARY
from research.adaptive_allocation_v1 import normalize_dividends, summarize
from research.simple_signal_blend_v1 import decision_state
from research.intraday_overnight_increment_v1 import now, require, write_json, digest
from scripts.review_round74_saved import saved_cycles
from scripts.finalize_round60_20260907 import metric
from scripts.finalize_round96_20260908 import table, value_equal

INDEX = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
OUT = ROOT / "deliverables/510300收益风险切点预算_第98轮_20260908"
DOCUMENT = OUT / "收益风险切点预算_结果及中文规则.md"
NEXT = ROOT / "docs/510300_AFTER_TANGENCY_REFERENCE_BUDGET_20260908.md"


def main():
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    require(index["latest_completed_round"]["round"] == 97 and not OUT.exists(), "第98轮前序或交付状态不符")
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    result = json.loads((RESEARCH / "result.json").read_text(encoding="utf-8"))
    require(not result["historical_point_target_met"], "达到点目标时需另作验收")
    data = pd.read_parquet(ROOT / cfg["features"])
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    f = pd.read_parquet(RESEARCH / "continuous_budget_factors.parquet")
    first_ref = int(np.flatnonzero(data.date.ge(cfg["reference_start"]))[0])
    for model, prefix in [("PANIC_ONLY", "panic"), ("REARM_RIDGE", "learned")]:
        ledger = pd.read_parquet(P91 / "continuous_references/BASE" / f"{model}_ledger.parquet")
        decisions = pd.read_parquet(P91 / "continuous_references/BASE" / f"{model}_decisions.parquet")
        require(pd.DatetimeIndex(ledger.date).equals(pd.DatetimeIndex(data.date.iloc[first_ref:])), "连续父账户日历不符")
        np.testing.assert_allclose(f[f"{prefix}_reference_return"].iloc[first_ref:], ledger.net_return, atol=0, rtol=0)
        np.testing.assert_allclose(f[f"{prefix}_state"], decision_state(data, decisions), atol=0, rtol=0, equal_nan=True)
    weights = np.array([.5, .5])
    checks = []
    for t in range(first_ref-1, len(data)-1):
        scheduled = t >= first_ref and data.date.iloc[t].to_period("M") != data.date.iloc[t-1].to_period("M")
        require(bool(f.budget_update_scheduled.iloc[t]) == scheduled, "切点更新不是原月首")
        if scheduled:
            count = min(cfg["risk_window"], t-first_ref+1)
            values = f[["panic_reference_return", "learned_reference_return"]].iloc[t-count+1:t+1].to_numpy(float)
            previous = weights.copy()
            score_error = 0.
            if count < cfg["risk_window"]:
                expected_status = "NO_VIEW_WARMUP_KEEP_BUDGET"
            elif not np.isfinite(values).all() or (values <= -1).any():
                expected_status = "NO_VIEW_INCOMPLETE_WINDOW_KEEP_BUDGET"
            else:
                mean = values.mean(axis=0)
                centered = values-mean
                covariance = centered.T@centered/(len(values)-1)
                np.testing.assert_allclose([f.panic_mean.iloc[t], f.learned_mean.iloc[t]], mean, atol=1e-14, rtol=0)
                np.testing.assert_allclose([f.panic_variance.iloc[t], f.learned_variance.iloc[t], f.reference_covariance.iloc[t]], [covariance[0, 0], covariance[1, 1], covariance[0, 1]], atol=1e-14, rtol=0)
                eigen = np.linalg.eigvalsh(covariance)
                if mean.max() <= 0:
                    weights = np.zeros(2)
                    expected_status = "EXPLICIT_CASH_NO_POSITIVE_PAST_MEAN"
                    require(pd.isna(f.estimated_daily_ratio.iloc[t]), "明确现金夏普被填成数值")
                elif eigen[-1] <= 0 or eigen[0] <= np.finfo(float).eps*eigen[-1]:
                    expected_status = "NO_VIEW_NONPOSITIVE_DEFINITE_COVARIANCE_KEEP_BUDGET"
                else:
                    # 用矩阵逆解表达核对解析导数比例，两端点和过去比例仍须共同比较。
                    v = np.linalg.solve(covariance, mean)
                    old = float(previous[0]/previous.sum()) if previous.sum() > 0 else .5
                    candidates = [0., 1., old]
                    if v.sum() != 0:
                        candidate = float(v[0]/v.sum())
                        if np.isfinite(candidate) and 0 < candidate < 1:
                            candidates.append(candidate)
                    scored = [(float(np.array([w, 1-w])@mean/np.sqrt(np.array([w, 1-w])@covariance@np.array([w, 1-w]))), w) for w in candidates]
                    best = max(score for score, _ in scored)
                    selected = min((w for score, w in scored if abs(score-best) <= 1e-12), key=lambda w: (abs(w-old), w))
                    weights = np.array([selected, 1-selected])
                    score_error = abs(best-f.estimated_daily_ratio.iloc[t])
                    require(score_error < 1e-11, "保存切点目标与独立矩阵解不符")
                    expected_status = "TANGENCY_BUDGET_AVAILABLE"
            require(f.budget_status.iloc[t] == expected_status, "现金、无观点或有效估计状态不符")
            checks.append({"date": data.date.iloc[t], "status": expected_status, "window_observations": count, "panic_budget": weights[0], "learned_budget": weights[1], "absolute_difference": score_error})
        np.testing.assert_allclose([f.panic_budget.iloc[t], f.learned_budget.iloc[t]], weights, atol=1e-10, rtol=0)
        expected_target = 0. if weights.sum() == 0 else float(weights@f[["panic_state", "learned_state"]].iloc[t].to_numpy(float))
        np.testing.assert_allclose(expected_target, f.target.iloc[t], atol=1e-10, rtol=0, equal_nan=True)
    require(pd.isna(f.target.iloc[-1]), "终点开盘后生成了新目标")
    accounts, cycles = [], []
    for period in ["evaluation", "earlier_diagnostic"]:
        frame = data if period == "evaluation" else data[data.date.le(cfg["earlier_terminal"])]
        start = cfg["evaluation_start"] if period == "evaluation" else cfg["earlier_start"]
        first = int(np.flatnonzero(frame.date.ge(start))[0])
        for cost in cfg["costs"]:
            ledger = pd.read_parquet(RESEARCH / period / cost / f"{PRIMARY}_ledger.parquet")
            decisions = pd.read_parquet(RESEARCH / period / cost / f"{PRIMARY}_decisions.parquet")
            require(pd.DatetimeIndex(ledger.date).equals(pd.DatetimeIndex(frame.date.iloc[first:])) and pd.DatetimeIndex(decisions.origin).equals(pd.DatetimeIndex(frame.date.iloc[first-1:-1])), "实际账本与完整原点日历不符")
            require(pd.DatetimeIndex(decisions.execution_date).equals(pd.DatetimeIndex(ledger.date)), "不是下一开盘执行")
            np.testing.assert_allclose(decisions.reference_weight, f.target.iloc[first-1:len(frame)-1], atol=0, rtol=0, equal_nan=True)
            np.testing.assert_allclose(ledger.cash+ledger.shares*ledger.mark+ledger.dividend_receivable, ledger.equity, atol=1e-6, rtol=0)
            np.testing.assert_allclose(ledger.equity/np.r_[cfg["initial_capital"], ledger.equity.iloc[:-1]]-1, ledger.net_return, atol=1e-12, rtol=0)
            own = ledger.set_index("date")
            for row in decisions.itertuples():
                shares = 0 if row.origin_index == first-1 else int(own.loc[row.origin, "shares"])
                equity = cfg["initial_capital"] if row.origin_index == first-1 else float(own.loc[row.origin, "equity"])
                price = float(frame.close.iloc[row.origin_index])
                actual = shares*price/equity
                target = row.reference_weight
                expected_qty = 0 if pd.isna(target) or (target > 0 and abs(target-actual) < cfg["weight_band"] and shares > 0) else -shares if target == 0 else int(np.floor(target*equity/price/cfg["lot"]))*cfg["lot"]-shares
                require(expected_qty == row.requested_quantity, "目标整手或十个百分点带宽未按自身账户计算")
            complete = saved_cycles(ledger, dividends, cfg)
            cycles.extend({"period": period, "cost": cost, **c} for c in complete)
            actual, saved = summarize(ledger, cfg), metric(result, PRIMARY, period, cost)
            require(all(value_equal(actual[k], saved[k]) for k in ["net_sharpe", "annualized_return", "max_drawdown", "cumulative_return", "trade_count", "commission", "slippage_cost"]), "保存账户指标不符")
            accounts.append({"period": period, "cost": cost, "cycles": len(complete), "losing_cycles": sum(c["net_profit"] < 0 for c in complete), **actual})
    for name, rows in [("saved_independent_tangency_checks.csv", checks), ("saved_account_checks.csv", accounts), ("saved_actual_cycles.csv", cycles)]:
        pd.DataFrame(rows).to_csv(RESEARCH / name, index=False, encoding="utf-8-sig")
    receipt = {"verified_at": now(), "status": "SAVED_MONTHLY_MOMENTS_TANGENCY_AND_ACTUAL_ACCOUNTS_CHECKED", "monthly_updates_checked": len(checks), "maximum_independent_objective_error": max(r["absolute_difference"] for r in checks), "new_accounts_checked": len(accounts), "complete_actual_cycles": len(cycles), "new_models_or_account_replays_during_check": 0, "reviewer_source_sha256": digest(Path(__file__)), "security_audit_performed": False}
    write_json(RESEARCH / "saved_verification_receipt.json", receipt, exclusive=True)
    status = "COMPLETED_TANGENCY_BUDGET_CASH_AND_PAST_MEAN_DID_NOT_IMPROVE"
    decision = "第98轮未达标。主基础／压力夏普0.493／0.451，较早0.411／0.390；主基础年化1.415%、较早2.793%，均低于买入持有。过去均值加入后没有改善原第91轮风险预算；关闭固定切点规则，不调窗口、均值收缩、协方差惩罚、现金门槛或候选池救回。"
    write_json(RESEARCH / "acceptance_outcome.json", {"recorded_at": now(), "status": status, "decision": decision, "goal_achieved": False, "independent_validation": "NOT_ESTABLISHED", "position_impact": 0}, exclusive=True)
    NEXT.write_text("\n".join(["# 第98轮之后：先从真实成交差异定位下一改动", "", decision, "", "本轮4个新完整账户、4项必要测试1.83秒、无新预测训练或参考账户，实际账户运行6.34秒。159月首：67次有效切点、18次两均值非正而明确现金、62次协方差退化无新观点沿用旧预算、12次窗口不足。主464原点明确现金，其中52个原父持有意向至少一条为正；早280现金原点，其中115个原父有持有意向。缺失协方差没有填成现金，明确现金没有填零夏普。", "", "主38成交190持股收盘、平均股票占比4.09%，早17成交204收盘、占比11.38%；两段两费用没有未成交或目标缺失。第97轮主S.887434/.820630、早.625161/.593639，有两段小幅年化超额，但仍不足1.2；第91轮主基础1.233仍仅局部候选。不能调98均值或改97混合先验继续扫。", "", "下一99先用保存的91、97、98完整账户做一次有界的日度经济差异分解：按98当时有效预算、明确现金和无新判断沿用三种状态，汇总新旧实际价格盈亏、登记分红、佣金、滑点及净值增量。只作已发生路径的诊断，不把状态分组当因果效果，不另外构造最优进出场或删去现金日。识别主要损失来自新增现金选择、资金比例变化还是原进出场，从具体交易差异提出一个新机制。尚未登记99策略或生成新账户。", "", "原参考数据与状态已经在97和98核对，直接读各study/evaluation与earlier_diagnostic下BASE、STRESS账本与决策；不要重跑。完整ETF价格features.parquet、分红csv、原费用和事件账户继续沿用。先让下一小改动有经济依据，再做一轮短规则、必要测试和完整账户；不恢复EPS、公募或慢来源，不制作GPT包。", ""]), encoding="utf-8")
    OUT.mkdir(parents=True)
    DOCUMENT.write_text("\n".join(["# 第98轮：收益与风险切点预算的结果", "", decision, "", "## 主历史：2020年1月2日至2026年8月14日开盘", "", *table(result["all_metrics"]), "## 较早历史：2015年1月5日至2019年12月31日开盘", "", *table(result["earlier_diagnostics"]), "## 结果的含义", "", "本轮没有新增预测训练，四个新账户计算约6秒。159个月首时点中，67次有有效切点预算、18次因两项过去均值均不为正而明确选择现金、62次协方差退化而没有新判断、12次资料长度不足。无判断月份延续上次预算，不能混同明确现金。", "", "主历史464个原点选择现金，其中52个原点的两条原策略至少一条有持股意向；较早现金280个原点，其中115个原点有原始持股意向。加入均值估计后，主平均股票占比降至4.09%、较早为11.38%，主年化只有1.42%、较早2.79%。少持仓并未自动提高夏普，也没有取得完整目标。", "", f"四项必要测试通过；{len(checks)}个月首的均值、协方差、独立矩阵切点与状态已核对，四个实际账户和{len(cycles)}个完整周期已核对，没有重训或重跑旧策略。保存的每日预算、现金选择、实际成交、分红费用与周期明细均在同目录。", "", *(ROOT / cfg["rules"]).read_text(encoding="utf-8").splitlines()[1:]]), encoding="utf-8")
    for p in RESEARCH.iterdir():
        if p.is_file() and p.suffix in {".csv", ".json"}:
            shutil.copy2(p, OUT / p.name)
    shutil.copy2(CONFIG, OUT / "冻结设置.json")
    shutil.copy2(NEXT, OUT / "下一项研究方向.md")
    for period, label in [("evaluation", "主历史"), ("earlier_diagnostic", "较早历史")]:
        for cost in cfg["costs"]:
            for kind in ["ledger", "decisions"]:
                pd.read_parquet(RESEARCH / period / cost / f"{PRIMARY}_{kind}.parquet").to_csv(OUT / f"{label}_{cost}_{kind}.csv", index=False, encoding="utf-8-sig")
    record = {"round": 98, "study": result["study_id"], "title": "两条连续参考的过去均值协方差切点预算", "status": status, "result": str((RESEARCH / "result.json").relative_to(ROOT)), **{k: result[k] for k in ["candidate_configurations", "evaluation_accounts", "new_accounts_generated", "reused_control_accounts", "earlier_diagnostic_accounts", "new_earlier_diagnostic_accounts", "new_model_fits", "new_reference_accounts"]}, "evaluated_candidate_source_runs": 1, "primary_base": metric(result, PRIMARY), "primary_stress": metric(result, PRIMARY, cost="STRESS"), "post_selected_best_base": result["post_selected_best_base"]}
    index["completed_rounds"].append(record)
    for key in ["registered_configurations_in_this_resumption", "evaluated_configurations_in_this_resumption", "evaluated_candidate_source_runs_including_corrected_replays", "registered_candidate_source_runs_including_unrun_legacy_bindings"]:
        index[key] += 1
    index["evaluation_accounts_in_this_resumption"] += 10
    index.update(updated_at=now(), latest_completed_round=record, running_studies=[], goal_achieved=False, status="ROUND98_TANGENCY_MEAN_AND_CASH_FAILED_FULL_GOAL_NOT_MET", count_warning="98轮，374不同设置，390已评价来源版本，395登记含5旧未运行，1426主评价记录。", next_work={"status": "SAVED_ACTUAL_BUDGET_STATE_DIFFERENCE_DIAGNOSTIC_NOT_REGISTERED", "focus": "分解91、97、98真实账户的价格、分红与费用差异，定位下一小改动", "source": str(NEXT.relative_to(ROOT))}, process_state_note="98四新账户及四必要测试完成，99先做已保存路径经济差异诊断。")
    index["deliveries"].append({"created_at": now(), "type": "TANGENCY_REFERENCE_BUDGET_ROUND98_FAST_CHINESE_RESULTS", "rounds": [98], "directory": str(OUT), "main_document": str(DOCUMENT), "new_gpt_review_archive_created": False})
    write_json(INDEX, index)
    write_json(OUT / "交付回执.json", {"created_at": now(), "main_document": str(DOCUMENT), "goal_achieved": False, "new_gpt_review_archive_created": False}, exclusive=True)
    print(json.dumps({"交付": str(DOCUMENT), "必要核对": receipt, "账户周期": accounts}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
