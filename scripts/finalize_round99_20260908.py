"""独立核对月度收益风险切点与明确现金，再交付保存账户。"""
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from numpy.polynomial.legendre import leggauss
from research.known_cash_risk_budget_v1 import ROOT, OUT as RESEARCH, CONFIG, P91, PRIMARY
from research.adaptive_allocation_v1 import normalize_dividends, summarize
from research.simple_signal_blend_v1 import decision_state
from research.intraday_overnight_increment_v1 import now, require, write_json, digest
from scripts.review_round74_saved import saved_cycles
from scripts.finalize_round60_20260907 import metric
from scripts.finalize_round96_20260908 import table, value_equal

INDEX = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
OUT = ROOT / "deliverables/510300已知现金风险预算_第99轮_20260908"
DOCUMENT = OUT / "已知现金风险预算_结果及中文规则.md"
NEXT = ROOT / "docs/510300_AFTER_KNOWN_CASH_RISK_BUDGET_20260908.md"


def main():
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    require(index["latest_completed_round"]["round"] == 98 and not OUT.exists(), "第99轮前序或交付状态不符")
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    result = json.loads((RESEARCH / "result.json").read_text(encoding="utf-8"))
    require(not result["historical_point_target_met"], "达到点目标时需另作验收")
    data = pd.read_parquet(ROOT / cfg["features"])
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    f = pd.read_parquet(RESEARCH / "continuous_budget_factors.parquet")
    first_ref = int(np.flatnonzero(data.date.ge(cfg["reference_start"]))[0])
    cash_evidence = pd.read_parquet(RESEARCH / "known_cash_evidence.parquet")
    for model, prefix in [("PANIC_ONLY", "panic"), ("REARM_RIDGE", "learned")]:
        ledger = pd.read_parquet(P91 / "continuous_references/BASE" / f"{model}_ledger.parquet")
        decisions = pd.read_parquet(P91 / "continuous_references/BASE" / f"{model}_decisions.parquet")
        require(pd.DatetimeIndex(ledger.date).equals(pd.DatetimeIndex(data.date.iloc[first_ref:])), "连续父账户日历不符")
        np.testing.assert_allclose(f[f"{prefix}_reference_return"].iloc[first_ref:], ledger.net_return, atol=0, rtol=0)
        np.testing.assert_allclose(f[f"{prefix}_state"], decision_state(data, decisions), atol=0, rtol=0, equal_nan=True)
        observed = (ledger.shares.eq(0) & ledger.filled_quantity.eq(0) & ledger.net_return.eq(0)).to_numpy(float)
        np.testing.assert_allclose(cash_evidence[f"{prefix}_confirmed_cash_day"].iloc[first_ref:], observed, atol=0, rtol=0)
    weights = np.array([.5, .5])
    checks = []
    for t in range(first_ref-1, len(data)-1):
        scheduled = t >= first_ref and data.date.iloc[t].to_period("M") != data.date.iloc[t-1].to_period("M")
        require(bool(f.budget_update_scheduled.iloc[t]) == scheduled, "切点更新不是原月首")
        if scheduled:
            count = min(cfg["risk_window"], t-first_ref+1)
            values = f[["panic_reference_return", "learned_reference_return"]].iloc[t-count+1:t+1].to_numpy(float)
            previous = weights.copy()
            flags = (cash_evidence[["panic_confirmed_cash_day", "learned_confirmed_cash_day"]].iloc[t-count+1:t+1].to_numpy(float) == 1).all(axis=0)
            np.testing.assert_array_equal(flags, [f.panic_confirmed_cash_window.iloc[t], f.learned_confirmed_cash_window.iloc[t]])
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
                    if flags.sum() == 1:
                        inactive = int(np.flatnonzero(flags)[0])
                        active = 1-inactive
                        require(mean[inactive] == 0 and (covariance[inactive] == 0).all(), "现金证据与样本矩矛盾")
                        require(mean[active] > 0 and covariance[active, active] > 0, "单方向没有有效正均值与正风险")
                        weights = np.zeros(2)
                        weights[active] = 1.
                        score_error = abs(mean[active]/np.sqrt(covariance[active, active])-f.estimated_daily_ratio.iloc[t])
                        require(score_error < 1e-11, "单方向风险比计算不符")
                        expected_status = "KNOWN_CASH_SINGLE_RISK_BUDGET_AVAILABLE"
                    else:
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
    receipt = {"verified_at": now(), "status": "SAVED_CASH_EVIDENCE_RISK_SUPPORT_AND_ACTUAL_ACCOUNTS_CHECKED", "monthly_updates_checked": len(checks), "maximum_independent_objective_error": max(r["absolute_difference"] for r in checks), "new_accounts_checked": len(accounts), "complete_actual_cycles": len(cycles), "new_models_or_account_replays_during_check": 0, "reviewer_source_sha256": digest(Path(__file__)), "security_audit_performed": False}
    write_json(RESEARCH / "saved_verification_receipt.json", receipt, exclusive=True)
    status = "COMPLETED_KNOWN_CASH_RISK_SUPPORT_RECOVERED_TRADES_BUT_NO_SHARPE_TARGET"
    decision = "第99轮未达标。主基础／压力夏普0.465／0.409，较早0.605／0.582；较早基础年化从98轮2.793%提高到6.431%，但主基础夏普从0.493降到0.465，主回撤从6.36%扩大到10.73%。已知现金处理恢复部分交易，增加风险后仍没有达到目标。关闭此固定风险支路及资金规模规则，不更改满额并列选择、窗口、现金证据或父策略救回。"
    write_json(RESEARCH / "acceptance_outcome.json", {"recorded_at": now(), "status": status, "decision": decision, "goal_achieved": False, "independent_validation": "NOT_ESTABLISHED", "position_impact": 0}, exclusive=True)
    NEXT.write_text("\n".join(["# 第99轮之后：回到新的独立信号", "", decision, "", "99四账户、五必要测试1.87秒、账户7.31秒已完成，无新预测训练或参考账户。159月首67原有效切点、62已知现金单风险方向、18明确现金、12预热；62情形全部按完整242日零股份零成交零收益证据处理，没有填充协方差。单风险方向按同估计夏普中最大均值给满额策略预算，再乘原持股意向，真实账户另算费用。主现金原点从98的464减至181，早280减至160；主234持股收盘48成交，早280收盘22成交，无未成交或目标缺失。早收入改善但主风险与夏普不改善，不能继续改这个规模规则。", "", "下一100优先核对尚未执行的Aroon高低点时间信号。它按过去最高价、最低价发生距今天的交易日数衡量趋势，不是旧高低价回归或ADX。此前65之后曾有aroon无实现的有界查重记录，本次docs中也只有该旧方向记录，尚需一次文件名和核心代码限定查重后再登记。先明确25周期的含当日窗口长度、重复极值取最近一天、现金分红调整高低价格及缺失规则，再冻结一个完整进入退出设置；不以新策略收益选择定义。尚无100新因子、策略或账户。", "", "本次已重新确认旧三状态Gaussian HMM存在：docs/510300_CAUSAL_GAUSSIAN_HMM_REGIME_ROUTER_V1_SPEC.md；旧ADX/DMI14及25进入20退出也已完成，跳过，不重训、不换状态数。R26已经有跌破低点后收复，不把海龟汤的相近周期包装新方向。GARCH/HAR文本只发现期权来源协议引用，暂不启动较复杂估计。", "", "继续用已有510300免费日线及分红、原两费用与完整账户。不恢复EPS、公募或慢来源，不生成GPT包与长系数说明。只需新因子数学、时钟、分红与真实账户必要检查，旧来源和预算结果不重复。", ""]), encoding="utf-8")
    OUT.mkdir(parents=True)
    DOCUMENT.write_text("\n".join(["# 第99轮：已知现金与可识别风险预算", "", decision, "", "## 主历史：2020年1月2日至2026年8月14日开盘", "", *table(result["all_metrics"]), "## 较早历史：2015年1月5日至2019年12月31日开盘", "", *table(result["earlier_diagnostics"]), "## 为什么仍未达到目标", "", "62个原协方差退化月现在均能按实际现金证据处理；原67个有效二维切点、18个明确现金及12个预热时点保持原计算边界。主现金原点从464减至181，较早从280减至160，实际持仓增加。", "", "主基础年化2.47%、最大回撤10.73%、平均股票占比9.35%；较早年化6.43%、回撤13.84%、平均股票占比21.86%。较早收益改善与更多风险同时发生，主夏普反而低于旧98。数学可计算范围扩大并不能直接证明收益风险改善，目标仍未实现。", "", f"五项必要测试通过，159个月度状态、完整现金证据与独立切点/单风险计算已经核对，四个账户共{len(cycles)}个完整周期已核对。未重训或重跑旧策略。所有每日预算、原点证据、成交与完整周期见同目录CSV及保存结果。", "", *(ROOT / cfg["rules"]).read_text(encoding="utf-8").splitlines()[1:]]), encoding="utf-8")
    for p in RESEARCH.iterdir():
        if p.is_file() and p.suffix in {".csv", ".json"}:
            shutil.copy2(p, OUT / p.name)
    shutil.copy2(CONFIG, OUT / "冻结设置.json")
    shutil.copy2(NEXT, OUT / "下一项研究方向.md")
    for period, label in [("evaluation", "主历史"), ("earlier_diagnostic", "较早历史")]:
        for cost in cfg["costs"]:
            for kind in ["ledger", "decisions"]:
                pd.read_parquet(RESEARCH / period / cost / f"{PRIMARY}_{kind}.parquet").to_csv(OUT / f"{label}_{cost}_{kind}.csv", index=False, encoding="utf-8-sig")
    record = {"round": 99, "study": result["study_id"], "title": "整窗已知现金与可识别风险方向的预算", "status": status, "result": str((RESEARCH / "result.json").relative_to(ROOT)), **{k: result[k] for k in ["candidate_configurations", "evaluation_accounts", "new_accounts_generated", "reused_control_accounts", "earlier_diagnostic_accounts", "new_earlier_diagnostic_accounts", "new_model_fits", "new_reference_accounts"]}, "evaluated_candidate_source_runs": 1, "primary_base": metric(result, PRIMARY), "primary_stress": metric(result, PRIMARY, cost="STRESS"), "post_selected_best_base": result["post_selected_best_base"]}
    index["completed_rounds"].append(record)
    for key in ["registered_configurations_in_this_resumption", "evaluated_configurations_in_this_resumption", "evaluated_candidate_source_runs_including_corrected_replays", "registered_candidate_source_runs_including_unrun_legacy_bindings"]:
        index[key] += 1
    index["evaluation_accounts_in_this_resumption"] += 10
    index.update(updated_at=now(), latest_completed_round=record, running_studies=[], goal_achieved=False, status="ROUND99_KNOWN_CASH_SUPPORT_NO_SHARPE_TARGET_FULL_GOAL_NOT_MET", count_warning="99轮，375不同设置，391已评价来源版本，396登记含5旧未运行，1436主评价记录。", next_work={"status": "AROON_TIME_SINCE_EXTREMA_METHOD_CHECK_NOT_REGISTERED", "focus": "核对Aroon高低点发生时间因子及分红价格口径，准备一个新独立进入退出机制", "source": str(NEXT.relative_to(ROOT))}, process_state_note="99四新账户及五必要测试完成，100仅为高低点时间机制方向，尚未登记。")
    index["deliveries"].append({"created_at": now(), "type": "KNOWN_CASH_RISK_BUDGET_ROUND99_FAST_CHINESE_RESULTS", "rounds": [99], "directory": str(OUT), "main_document": str(DOCUMENT), "new_gpt_review_archive_created": False})
    write_json(INDEX, index)
    write_json(OUT / "交付回执.json", {"created_at": now(), "main_document": str(DOCUMENT), "goal_achieved": False, "new_gpt_review_archive_created": False}, exclusive=True)
    print(json.dumps({"交付": str(DOCUMENT), "必要核对": receipt, "账户周期": accounts}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
