"""只核对保存的策略迭代、预测与账户，不重新训练或重跑。"""
from bisect import bisect_right
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from research.entry_wait_policy_v1 import ROOT, OUT as RESEARCH, CONFIG, LINKS, P59, PRIMARY, IMMEDIATE, SETTINGS, schedule
from research.entry_wait_policy_inputs_v1 import FEATURES, select_training_rows, action_arrays, evaluate_fixed_policy, decision_mask, transition_state, policy_hash, observed_signal_age
from research.adaptive_allocation_v1 import normalize_dividends, summarize, affordable_quantity, fill_price
from research.intraday_overnight_increment_v1 import now, require, write_json, digest
from scripts.review_round74_saved import saved_cycles
from scripts.finalize_round60_20260907 import metric

INDEX = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
OUT = ROOT / "deliverables/510300买入与等待价值_第96轮_20260908"
DOCUMENT = OUT / "买入与等待价值_结果及中文规则.md"
NEXT = ROOT / "docs/510300_AFTER_ENTRY_WAIT_POLICY_20260908.md"


def value_equal(a, b):
    return (a is None and b is None) or (a is not None and b is not None and abs(a-b) < 1e-9)


def table(rows):
    lines = ["|策略|费用|净夏普|年化收益|最大回撤|成交次数|", "|---|---|---:|---:|---:|---:|"]
    for m in rows:
        sharpe = f"{m['net_sharpe']:.3f}" if m["net_sharpe"] is not None else "无法计算（零波动）"
        lines.append(f"|{m['name']}|{'基础' if m['cost']=='BASE' else '压力'}|{sharpe}|{m['annualized_return']:.2%}|{m['max_drawdown']:.2%}|{m['trade_count']}|")
    return lines+[""]


def main():
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    require(index["latest_completed_round"]["round"] == 95 and not OUT.exists(), "第96轮前序或交付状态不符")
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    result = json.loads((RESEARCH / "result.json").read_text(encoding="utf-8"))
    require(not result["historical_point_target_met"], "达到点目标时应另作完整验收")
    data = pd.read_parquet(ROOT / cfg["features"])
    factors = pd.read_parquet(P59 / "entry_factors.parquet")
    links = pd.read_parquet(LINKS / "entry_action_links.parquet")
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    np.testing.assert_allclose(links.signal_age_observed_days, observed_signal_age(factors)[links.path_id.to_numpy(int)], atol=0, rtol=0)
    models = json.loads((RESEARCH / "saved_models.json").read_text(encoding="utf-8"))["models"]
    iterations = json.loads((RESEARCH / "saved_policy_iterations.json").read_text(encoding="utf-8"))["iterations"]
    by_fit = {}
    for item in iterations:
        by_fit.setdefault(item["fit_index"], []).append(item)
    membership = pd.read_parquet(RESEARCH / "training_memberships.parquet")
    fit_indices = [m["fit_index"] for m in models]
    require(fit_indices == schedule(data, cfg["earlier_start"]), "月度时钟不符")
    fit_checks, predictions, account_checks, cycles, differences = [], [], [], [], []
    for stored in models:
        t = stored["fit_index"]
        rows, groups = select_training_rows(links, data.date.iloc[t], cfg["recent_episodes"])
        eligible = len(groups) >= cfg["minimum_episodes"] and len(rows) >= cfg["minimum_rows"]
        require(groups == stored["training_groups"] and len(rows) == stored["training_rows"] and eligible == stored["eligible"], "训练完整组或最低覆盖不符")
        if not eligible:
            require(stored["model"] is None and stored["status"] == "NO_VIEW_MINIMUM_MATURE_GROUPS_OR_ROWS" and t not in by_fit, "未成熟时发生了训练")
            continue
        require(stored["latest_mature_group_date"] == str(rows.original_group_mature_date.max().date()), "训练组可用时点不符")
        x, weights = rows[FEATURES].to_numpy(float), rows.sample_weight.to_numpy(float)
        members = membership[membership.fit_index.eq(t)].set_index("path_id").loc[rows.path_id]
        np.testing.assert_allclose(weights, members.sample_weight, atol=0, rtol=0)
        mean = np.average(x, axis=0, weights=weights)
        scale = np.sqrt(np.average((x-mean)**2, axis=0, weights=weights))
        scale[scale <= 1e-12] = 1
        design = np.c_[np.ones(len(x)), np.clip((x-mean)/scale, -cfg["feature_clip"], cfg["feature_clip"])]
        gram = design.T@(weights[:, None]*design)+np.diag([0.]+[cfg["ridge_alpha"]]*len(FEATURES))
        reward, filled, successors = action_arrays(rows)
        policy = np.zeros(len(rows), bool)
        seen = {policy_hash(policy)}
        saved_iterations = by_fit[t]
        require(len(saved_iterations) == stored["iterations"] <= cfg["maximum_policy_iterations"], "迭代数量不符")
        max_residual = 0.
        for n, step in enumerate(saved_iterations, start=1):
            require(step["iteration"] == n and step["evaluation_policy_hash"] == policy_hash(policy), "固定评价策略来源不符")
            target, _ = evaluate_fixed_policy(policy, reward, filled, successors)
            beta = np.column_stack([step["buy_coefficients_with_intercept"], step["wait_coefficients_with_intercept"]])
            residual = float(np.max(np.abs(gram@beta-design.T@(weights[:, None]*target))))
            require(residual < 1e-9, "保存动作回归未满足原加权方程")
            max_residual = max(max_residual, residual)
            fitted = design@beta
            following = decision_mask(fitted[:, 0], fitted[:, 1])
            state = transition_state(policy, following, seen)
            require(step["transition_status"] == state and step["improved_policy_hash"] == policy_hash(following), "训练选择或循环检测不符")
            require(step["evaluation_entries"] == int(policy.sum()) and step["improved_entries"] == int(following.sum()), "训练选择计数不符")
            require(n == len(saved_iterations) or state == "CONTINUE_POLICY_IMPROVEMENT", "已结束训练仍继续选择")
            policy = following
            seen.add(policy_hash(policy))
        expected = "FIT_COMPLETE" if state == "TRAINING_POLICY_STABLE" else state if state == "NO_VIEW_REPEATED_TRAINING_POLICY" else "NO_VIEW_POLICY_ITERATION_LIMIT"
        require(stored["status"] == expected, "最终模型状态不符")
        if stored["model"] is not None:
            m = stored["model"]
            require(expected == "FIT_COMPLETE", "不稳定模型被启用")
            np.testing.assert_allclose(m["mean"], mean, atol=1e-12, rtol=0)
            np.testing.assert_allclose(m["scale"], scale, atol=1e-12, rtol=0)
            np.testing.assert_allclose(np.column_stack([m["buy_coefficients_with_intercept"], m["wait_coefficients_with_intercept"]]), beta, atol=0, rtol=0)
        else:
            require(expected != "FIT_COMPLETE", "稳定模型没有保存")
        fit_checks.append({"fit_origin": stored["fit_origin"], "status": expected, "groups": len(groups), "rows": len(rows), "iterations": len(saved_iterations), "maximum_normal_equation_residual": max_residual})
    views = pd.read_parquet(RESEARCH / "action_views.parquet")
    for t, row in enumerate(views.itertuples()):
        idx = bisect_right(fit_indices, t)-1
        stored = models[idx] if idx >= 0 else None
        if stored is None or stored["status"] != "FIT_COMPLETE":
            require(pd.isna(row.buy_value) and pd.isna(row.wait_value) and pd.isna(row.waiting_policy_margin), "无模型被填预测")
            continue
        m = stored["model"]
        x = np.array([getattr(row, field) for field in FEATURES])
        require(np.isfinite(x).all() and row.entry_fit_origin == pd.Timestamp(stored["fit_origin"]), "可用动作因子或时点不符")
        d = np.r_[1., np.clip((x-np.array(m["mean"]))/np.array(m["scale"]), -cfg["feature_clip"], cfg["feature_clip"])]
        buy, wait = float(d@np.array(m["buy_coefficients_with_intercept"])), float(d@np.array(m["wait_coefficients_with_intercept"]))
        require(abs(buy-row.buy_value) < 1e-12 and abs(wait-row.wait_value) < 1e-12 and abs(row.waiting_policy_margin-(buy-max(0., wait))) < 1e-12, "保存的动作预测或比较规则不符")
        predictions.append({"date": row.date, "fit_origin": row.entry_fit_origin, "buy_value": buy, "wait_value": wait})
    for period in ["evaluation", "earlier_diagnostic"]:
        frame = data if period == "evaluation" else data[data.date.le(cfg["earlier_terminal"])]
        start = cfg["evaluation_start"] if period == "evaluation" else cfg["earlier_start"]
        first = int(np.flatnonzero(frame.date.ge(start))[0])
        for cost_id, cost in cfg["costs"].items():
            compared = {}
            for model, (margin, _) in SETTINGS.items():
                ledger = pd.read_parquet(RESEARCH / period / cost_id / f"{model}_ledger.parquet")
                decisions = pd.read_parquet(RESEARCH / period / cost_id / f"{model}_decisions.parquet")
                require(pd.DatetimeIndex(ledger.date).equals(pd.DatetimeIndex(frame.date.iloc[first:])) and pd.DatetimeIndex(decisions.origin).equals(pd.DatetimeIndex(frame.date.iloc[first-1:-1])), "完整账户或决策日历不符")
                require(pd.DatetimeIndex(decisions.execution_date).equals(pd.DatetimeIndex(ledger.date)), "不是下一开盘执行")
                np.testing.assert_allclose(decisions.entry_value_margin, views[margin].iloc[first-1:len(frame)-1], atol=0, rtol=0, equal_nan=True)
                previous = np.r_[cfg["initial_capital"], ledger.equity.iloc[:-1]]
                np.testing.assert_allclose(ledger.cash+ledger.shares*ledger.mark+ledger.dividend_receivable, ledger.equity, atol=1e-6, rtol=0)
                np.testing.assert_allclose(ledger.equity/previous-1, ledger.net_return, atol=1e-12, rtol=0)
                by_date = ledger.set_index("date")
                for row in decisions.itertuples():
                    shares = 0 if row.origin_index == first-1 else int(by_date.loc[row.origin, "shares"])
                    cash = cfg["initial_capital"] if row.origin_index == first-1 else float(by_date.loc[row.origin, "cash"])
                    if row.requested_quantity > 0:
                        require(shares == 0 and row.entry_rearmed and row.entry_value_margin > 0 and factors.raw_entry.iloc[row.origin_index] == 1, "进入不满足自身状态与本轮条件")
                        expected_qty = affordable_quantity(cash, fill_price(float(frame.close.iloc[row.origin_index]), 1, cost, cfg["tick"]), cost, cfg["lot"])
                        require(expected_qty == row.requested_quantity, "买入未按自己的可用现金计算")
                    elif row.requested_quantity < 0:
                        require(row.requested_quantity == -shares and "学习条件" not in row.exit_reasons, "自然退出或全部退出数量不符")
                    if pd.isna(row.entry_value_margin):
                        require(row.requested_quantity <= 0, "无判断时申请新买入")
                complete = saved_cycles(ledger, dividends, cfg)
                cycles.extend({"period": period, "cost": cost_id, "model": model, **c} for c in complete)
                actual, saved = summarize(ledger, cfg), metric(result, model, period, cost_id)
                for field in ["net_sharpe", "annualized_return", "max_drawdown", "cumulative_return", "trade_count", "commission", "slippage_cost"]:
                    require(value_equal(actual[field], saved[field]), "保存完整账户指标不符")
                account_checks.append({"period": period, "cost": cost_id, "model": model, "cycles": len(complete), **actual})
                compared[model] = ledger
            a, b = compared[PRIMARY], compared[IMMEDIATE]
            delta = {"period": period, "cost": cost_id, "waiting_minus_immediate_net_profit": float(a.equity.iloc[-1]-b.equity.iloc[-1])}
            for field in ["price_pnl", "dividend_recognized", "commission", "slippage_cost"]:
                delta[field] = float(a[field].sum()-b[field].sum())
            require(abs(delta["waiting_minus_immediate_net_profit"]-delta["price_pnl"]-delta["dividend_recognized"]+delta["commission"]+delta["slippage_cost"]) < 1e-6, "等待相对收益的经济拆解不符")
            differences.append(delta)
    for filename, rows in [("saved_training_checks.csv", fit_checks), ("saved_action_prediction_checks.csv", predictions), ("saved_account_checks.csv", account_checks), ("saved_actual_cycles.csv", cycles), ("waiting_value_differences.csv", differences)]:
        pd.DataFrame(rows).to_csv(RESEARCH / filename, index=False, encoding="utf-8-sig")
    receipt = {"verified_at": now(), "status": "SAVED_POLICY_ITERATIONS_ACTION_VALUES_AND_COMPLETE_ACCOUNTS_CHECKED", "monthly_training_runs_checked": len(fit_checks), "saved_scalar_regressions_checked": 2*len(iterations), "stable_models": result["stable_monthly_models"], "available_prediction_rows": len(predictions), "new_accounts_checked": len(account_checks), "complete_actual_cycles": len(cycles), "new_fits_or_account_replays_during_check": 0, "reviewer_source_sha256": digest(Path(__file__)), "security_audit_performed": False}
    write_json(RESEARCH / "saved_verification_receipt.json", receipt, exclusive=True)
    status = "COMPLETED_ENTRY_WAIT_POLICY_UNSTABLE_AND_NO_PORTFOLIO_IMPROVEMENT"
    decision = "第96轮两个设置均未达标。买入与等待比较的主基础／压力夏普0.616／0.600，同模型只看买入价值为正为0.642／0.625；较早四账户无交易、零波动，夏普无法计算。119个可训练月份中103个月发生选择循环，仅16个月稳定。关闭这套固定局部等待策略，不调整迭代次数、阈值、因子或无模型处理救回。"
    write_json(RESEARCH / "acceptance_outcome.json", {"recorded_at": now(), "status": status, "decision": decision, "goal_achieved": False, "independent_validation": "NOT_ESTABLISHED", "position_impact": 0}, exclusive=True)
    NEXT.write_text("\n".join(["# 第96轮之后：转向不需要反复训练的资金分配", "", decision, "", "已完成八个新完整账户、八项必要测试、119个月度策略训练（1656次标量动作回归）。现金等待价值用零现金一步收益与同组继续状态构造，未挑未来实际最优进入日；但局部线性近似在103个月循环，早期全无可用交易，主期只各两次买入。原46成熟组1180状态、1134连接、46终止均已经保存，不要重跑来源或动作模型。", "", "主基础等待与只看买入价值相比少赚1027.43元，压力少赚1024.26元；主平均股票占比约5.6%。较早现金等待是模型不可用的结果，不是预判下跌。第91轮主基础1.233仍仅局部候选；其压力、早期及稳定超额与独立验证均未过。", "", "下一97优先检查不需要收益预测训练的连续参考资金分配：使用已经保存的两条原策略连续账户，考察按全部固定混合比例的累计增长自动汇总预算是否构成新机制。旧84仅按两条单独参考累计净值比例分配，旧5为市场因子直接效用训练；两者失败与固定规则保持，不能改旧84净值指数、初始权重或窗口救回。先做有界查重与原方法核对，尚未登记新设置、生成预算或账户。", "", "继续快节奏：少文档、复用已保存资料及对照、只做影响结论的必要检查；不恢复EPS、公募、付费或慢来源，不制作GPT包。", ""]), encoding="utf-8")
    OUT.mkdir(parents=True)
    text = ["# 第96轮：买入与等待价值的结果", "", decision, "", "## 主历史：2020年1月2日至2026年8月14日开盘", "", *table(result["all_metrics"]), "## 较早历史：2015年1月5日至2019年12月31日开盘", "", *table(result["earlier_diagnostics"]), "## 为什么没有改善", "", "141个月度时点中，22个时点成熟资料不足；119次训练有103次出现反复选择循环，仅16次得到稳定模型。主历史1604个决策原点中1284个没有可用预测，两个设置均到2025年4月10日才首次成交，之后各完成两次买入、两次退出。较早1219天都没有可用交易判断，因此四个较早账户均为真实现金等待，零收益、零波动，夏普未定义。", "", "主基础等待设置年化1.40%、回撤3.98%，平均持股占比5.56%；只看买入价值的年化1.47%、回撤3.98%。等待判断使基础账户少赚1027.43元、压力账户少赚1024.26元。主夏普看起来高于上一轮，但大部分时间没有模型且仅有两个实际交易周期，不能解释为找到了稳定策略。", "", f"必要检查覆盖{len(fit_checks)}个月的{2*len(iterations)}个保存动作回归方程、{len(predictions)}个可用预测记录、八个完整账户及{len(cycles)}个实际完整周期，不重新拟合或重跑。十项因子中文定义及全部进出场规则如下；每月两个动作的实际系数保存在同目录中文CSV。", "", *(ROOT / cfg["rules"]).read_text(encoding="utf-8").splitlines()[1:]]
    DOCUMENT.write_text("\n".join(text), encoding="utf-8")
    for p in RESEARCH.iterdir():
        if p.is_file() and p.suffix in {".csv", ".json"}:
            shutil.copy2(p, OUT / p.name)
    shutil.copy2(CONFIG, OUT / "冻结设置.json")
    shutil.copy2(NEXT, OUT / "下一项研究方向.md")
    for period, label in [("evaluation", "主历史"), ("earlier_diagnostic", "较早历史")]:
        for cost in cfg["costs"]:
            for model in SETTINGS:
                for kind in ["ledger", "decisions"]:
                    pd.read_parquet(RESEARCH / period / cost / f"{model}_{kind}.parquet").to_csv(OUT / f"{label}_{cost}_{model}_{kind}.csv", index=False, encoding="utf-8-sig")
    record = {"round": 96, "study": result["study_id"], "title": "同组买入与等待动作价值及同模型对照", "status": status, "result": str((RESEARCH / "result.json").relative_to(ROOT)), **{k: result[k] for k in ["candidate_configurations", "evaluation_accounts", "new_accounts_generated", "reused_control_accounts", "earlier_diagnostic_accounts", "new_earlier_diagnostic_accounts", "new_model_fits", "new_reference_accounts"]}, "stable_monthly_models": result["stable_monthly_models"], "scalar_action_regressions": result["scalar_action_regressions"], "evaluated_candidate_source_runs": 2, "primary_base": metric(result, PRIMARY), "primary_stress": metric(result, PRIMARY, cost="STRESS"), "post_selected_best_base": result["post_selected_best_base"]}
    index["completed_rounds"].append(record)
    for key in ["registered_configurations_in_this_resumption", "evaluated_configurations_in_this_resumption", "evaluated_candidate_source_runs_including_corrected_replays", "registered_candidate_source_runs_including_unrun_legacy_bindings"]:
        index[key] += 2
    index["evaluation_accounts_in_this_resumption"] += 10
    index.update(updated_at=now(), latest_completed_round=record, running_studies=[], goal_achieved=False, status="ROUND96_WAIT_POLICY_CYCLES_AND_NO_IMPROVEMENT_FULL_GOAL_NOT_MET", count_warning="96轮，372不同设置，388已评价来源版本，393登记含5旧未运行，1406主评价记录；96八个新账户含四主、四较早。", next_work={"status": "CONTINUOUS_REFERENCE_UNIVERSAL_BUDGET_METHOD_CHECK_NOT_REGISTERED", "focus": "检查不反复训练的固定混合比例增长汇总预算，避免重复旧84累计净值比例规则", "source": str(NEXT.relative_to(ROOT))}, process_state_note="96完成119月训练、1656标量动作回归、八新账户与简明中文交付；未达标，继续97新机制核对。")
    index["deliveries"].append({"created_at": now(), "type": "ENTRY_WAIT_POLICY_ROUND96_FAST_CHINESE_RESULTS", "rounds": [96], "directory": str(OUT), "main_document": str(DOCUMENT), "new_gpt_review_archive_created": False})
    write_json(INDEX, index)
    write_json(OUT / "交付回执.json", {"created_at": now(), "main_document": str(DOCUMENT), "goal_achieved": False, "new_gpt_review_archive_created": False}, exclusive=True)
    print(json.dumps({"交付": str(DOCUMENT), "必要核对": receipt, "等待相对差异": differences}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
