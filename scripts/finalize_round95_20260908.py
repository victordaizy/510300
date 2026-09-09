"""核对保存模型、进入请求和完整账户，保留预测排序与交易效果的区别。"""
from bisect import bisect_right
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from research.entry_payoff_gate_v1 import ROOT, OUT as RESEARCH, CONFIG, PRIMARY, P59, schedule
from research.entry_payoff_gate_inputs_v1 import FEATURES
from research.adaptive_allocation_v1 import normalize_dividends, summarize, affordable_quantity, fill_price
from research.intraday_overnight_increment_v1 import now, require, write_json, digest
from scripts.review_round74_saved import saved_cycles
from scripts.finalize_round60_20260907 import metric, table

OUT = ROOT / "deliverables/510300完整交易收益筛选进入_第95轮_20260908"
DOCUMENT = OUT / "完整交易收益筛选进入_结果及中文规则.md"
INDEX = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
NEXT = ROOT / "docs/510300_AFTER_ENTRY_PAYOFF_GATE_20260908.md"


def main():
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    require(index["latest_completed_round"]["round"] == 94 and not OUT.exists(), "第95轮前序或交付状态不符")
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    result = json.loads((RESEARCH / "result.json").read_text(encoding="utf-8"))
    require(not result["historical_point_target_met"], "若点目标达到需另作完整验收")
    diagnostic = json.loads((RESEARCH / "entry_filter_attribution/result.json").read_text(encoding="utf-8"))
    data = pd.read_parquet(ROOT / cfg["features"])
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    samples = pd.read_parquet(RESEARCH / "entry_payoff_samples.parquet")
    original_groups = pd.read_csv(P59 / "reference_episodes.csv", parse_dates=["group_mature_date"])
    original_paths = pd.read_csv(P59 / "reference_paths.csv")
    factors = pd.read_parquet(P59 / "entry_factors.parquet")
    source_final = original_paths.set_index("path_id").loc[samples.path_id]
    np.testing.assert_allclose(samples.observed_final_equity, source_final.final_observed_nav, atol=1e-6, rtol=0)
    np.testing.assert_allclose(samples.target, samples.complete_final_equity/cfg["initial_capital"]-1, atol=1e-12, rtol=0)
    require(samples.extra_owned_dividend_after_exit.eq(0).all(), "本次来源若存在退出后确认权益应专门核对")
    mapping = original_groups.set_index("episode_id").group_mature_date
    expected_maturity = samples.episode_id.map(mapping)
    require(((samples.training_maturity_date == expected_maturity) | (samples.training_maturity_date.isna() & expected_maturity.isna())).all(), "原整组成熟日期改变")
    origin = samples.entry_origin_index.to_numpy(int)
    expected_x = np.c_[data[FEATURES[:-1]].to_numpy()[origin], factors.d60_factor.to_numpy()[origin]]
    np.testing.assert_allclose(samples[FEATURES], expected_x, atol=0, rtol=0)
    require((samples.economic_maturity_date >= samples.natural_exit_date).all(), "经济标签成熟早于自然退出")
    models = json.loads((RESEARCH / "saved_models.json").read_text(encoding="utf-8"))["models"]
    membership = pd.read_parquet(RESEARCH / "training_memberships.parquet")
    fit_indices = [row["fit_index"] for row in models]
    require(fit_indices == schedule(data, cfg["earlier_start"]), "原拟合时钟改变")
    fit_checks, prediction_checks, account_checks, cycles, concentration = [], [], [], [], []
    for record in models:
        t = record["fit_index"]
        mature = samples[samples.training_maturity_date.notna() & samples.training_maturity_date.le(data.date.iloc[t])]
        groups = mature[["episode_id", "training_maturity_date"]].drop_duplicates().sort_values(["training_maturity_date", "episode_id"]).tail(cfg["recent_episodes"]).episode_id.to_list()
        rows = mature[mature.episode_id.isin(groups)].sort_values(["episode_id", "path_id"])
        usable = len(groups) >= cfg["minimum_episodes"] and len(rows) >= cfg["minimum_rows"]
        require(groups == record["training_groups"] and len(rows) == record["training_rows"] and usable == (record["status"] == "FIT_COMPLETE"), "保存的成熟组或训练门槛不符")
        if not usable:
            continue
        weights = 1/rows.groupby("episode_id").path_id.transform("count").to_numpy()
        members = membership[membership.fit_index.eq(t)].set_index("path_id").loc[rows.path_id]
        np.testing.assert_allclose(members.sample_weight, weights, atol=0, rtol=0)
        require(record["latest_training_maturity"] == str(rows.training_maturity_date.max().date()), "最后成熟标签日期不符")
        x, y = rows[FEATURES].to_numpy(float), rows.target.to_numpy(float)
        mean = np.average(x, axis=0, weights=weights)
        scale = np.sqrt(np.average((x-mean)**2, axis=0, weights=weights))
        scale[scale <= 1e-12] = 1
        design = np.c_[np.ones(len(x)), np.clip((x-mean)/scale, -5, 5)]
        beta = np.linalg.solve(design.T@(weights[:, None]*design)+np.diag([0.]+[cfg["ridge_alpha"]]*9), design.T@(weights*y))
        model = record["model"]
        np.testing.assert_allclose(model["mean"], mean, atol=1e-12, rtol=0)
        np.testing.assert_allclose(model["scale"], scale, atol=1e-12, rtol=0)
        np.testing.assert_allclose([model["intercept"], *model["coefficients"]], beta, atol=1e-10, rtol=0)
        fit_checks.append({"fit_origin": record["fit_origin"], "groups": len(groups), "rows": len(rows), "latest_maturity": record["latest_training_maturity"]})
    views = pd.read_parquet(RESEARCH / "entry_views.parquet")
    for t, row in enumerate(views.itertuples()):
        idx = bisect_right(fit_indices, t)-1
        stored = models[idx] if idx >= 0 else None
        available = stored is not None and stored["status"] == "FIT_COMPLETE"
        if not available:
            require(pd.isna(row.predicted_entry_return) and not row.entry_accepted_by_model, "无模型被替换成了进入预测")
            continue
        model = stored["model"]
        x = np.array([getattr(row, field) for field in FEATURES])
        require(np.isfinite(x).all() and row.entry_fit_origin == pd.Timestamp(stored["fit_origin"]), "实际预测因子或模型使用时点不符")
        estimate = model["intercept"]+np.clip((x-np.array(model["mean"]))/np.array(model["scale"]), -5, 5)@np.array(model["coefficients"])
        require(abs(estimate-row.predicted_entry_return) < 1e-12 and row.entry_accepted_by_model == (estimate > 0), "保存进入预测或严格正筛选不符")
        prediction_checks.append({"date": row.date, "fit_origin": row.entry_fit_origin, "prediction": float(estimate), "used_for_terminal_close_order": False})
    for period in ["evaluation", "earlier_diagnostic"]:
        frame = data if period == "evaluation" else data[data.date.le(cfg["earlier_terminal"])]
        start = cfg["evaluation_start"] if period == "evaluation" else cfg["earlier_start"]
        first = int(np.flatnonzero(frame.date.ge(start))[0])
        for cost_id, cost in cfg["costs"].items():
            ledger = pd.read_parquet(RESEARCH / period / cost_id / f"{PRIMARY}_ledger.parquet")
            decisions = pd.read_parquet(RESEARCH / period / cost_id / f"{PRIMARY}_decisions.parquet")
            require(pd.DatetimeIndex(ledger.date).equals(pd.DatetimeIndex(frame.date.iloc[first:])) and pd.DatetimeIndex(decisions.origin).equals(pd.DatetimeIndex(frame.date.iloc[first-1:-1])), "完整交易或决策日历不符")
            require(pd.DatetimeIndex(decisions.execution_date).equals(pd.DatetimeIndex(ledger.date)), "不是下一开盘执行")
            np.testing.assert_allclose(decisions.predicted_entry_return, views.predicted_entry_return.iloc[first-1:len(frame)-1], atol=0, rtol=0, equal_nan=True)
            previous = np.r_[cfg["initial_capital"], ledger.equity.iloc[:-1]]
            np.testing.assert_allclose(ledger.cash+ledger.shares*ledger.mark+ledger.dividend_receivable, ledger.equity, atol=1e-6, rtol=0)
            np.testing.assert_allclose(ledger.equity/previous-1, ledger.net_return, atol=1e-12, rtol=0)
            by_date = ledger.set_index("date")
            for row in decisions.itertuples():
                shares = 0 if row.origin_index == first-1 else int(by_date.loc[row.origin, "shares"])
                cash = cfg["initial_capital"] if row.origin_index == first-1 else float(by_date.loc[row.origin, "cash"])
                if row.requested_quantity > 0:
                    require(shares == 0 and row.entry_rearmed and row.predicted_entry_return > 0 and factors.raw_entry.iloc[row.origin_index] == 1, "实际进入没有满足原条件、预测或资格")
                    expected = affordable_quantity(cash, fill_price(float(frame.close.iloc[row.origin_index]), 1, cost, cfg["tick"]), cost, cfg["lot"])
                    require(expected == row.requested_quantity, "进入没有按自身可用现金计算")
                elif row.requested_quantity < 0:
                    require(row.requested_quantity == -shares and "学习条件" not in row.exit_reasons, "退出不是原自然条件下的全部份额")
                if pd.isna(row.predicted_entry_return):
                    require(row.requested_quantity <= 0, "无模型时发起新买入")
                    if factors.raw_entry.iloc[row.origin_index] and shares == 0:
                        require(pd.isna(row.reference_weight), "无观点被填成现金目标")
            group = saved_cycles(ledger, dividends, cfg)
            for left, right in zip(group, group[1:]):
                a = int(np.flatnonzero(frame.date.eq(left["exit_date"]))[0])
                b = int(np.flatnonzero(frame.date.eq(right["entry_origin"]))[0])
                require(b-a >= cfg["specification"]["cooldown"] and factors.raw_entry.iloc[a:b+1].eq(0).any(), "模型拒绝被误当作原条件消失或冷却不足")
            cycles.extend({"period": period, "cost": cost_id, **c} for c in group)
            total = sum(c["net_profit"] for c in group)
            top = sorted((c["net_profit"] for c in group if c["net_profit"] > 0), reverse=True)[:3]
            concentration.append({"period": period, "cost": cost_id, "cycles": len(group), "losing_cycles": sum(c["net_profit"] < 0 for c in group), "net_profit": total, "largest_three_vs_net_profit": sum(top)/total if total > 0 else None})
            actual, saved = summarize(ledger, cfg), metric(result, PRIMARY, period, cost_id)
            for field in ["net_sharpe", "annualized_return", "max_drawdown", "cumulative_return", "trade_count", "commission", "slippage_cost"]:
                require(abs(actual[field]-saved[field]) < 1e-9, "保存的新策略指标不符")
            account_checks.append({"period": period, "cost": cost_id, **actual})
    for name, rows in [("saved_monthly_entry_model_checks.csv", fit_checks), ("saved_prediction_checks.csv", prediction_checks), ("saved_account_checks.csv", account_checks), ("saved_actual_cycles.csv", cycles), ("saved_profit_concentration.csv", concentration)]:
        pd.DataFrame(rows).to_csv(RESEARCH / name, index=False, encoding="utf-8-sig")
    receipt = {"verified_at": now(), "status": "COMPLETE_ENTRY_LABELS_MATURE_MODELS_AND_ACTUAL_ENTRY_ACCOUNTS_CHECKED", "saved_natural_entry_labels_checked": len(samples), "monthly_models_checked": len(fit_checks), "saved_prediction_rows_checked": len(prediction_checks), "new_strategy_account_metrics_checked": len(account_checks), "complete_strategy_cycles": len(cycles), "additional_diagnostic_accounts_already_completed": diagnostic["new_earlier_diagnostic_accounts"], "new_accounts_or_model_fits_during_final_verification": 0, "reviewer_source_sha256": digest(Path(__file__)), "security_audit_performed": False}
    write_json(RESEARCH / "saved_verification_receipt.json", receipt, exclusive=True)
    status = "COMPLETED_ENTRY_PAYOFF_GATE_PORTFOLIO_WORSE_LOCAL_RANKING_SIGNAL_ONLY"
    decision = "九因子完整交易收益筛选进入未达到目标。主基础／压力夏普0.177／0.153，较早0.192／0.170。相同模型可用时间的对照证明，符号筛选本身主基础少赚31079.65元、较早少赚5593.05元。主历史同组局部排序有10/12组正向线索，但不能替代实际交易结果。结束这一固定模型与符号筛选规则，不调因子、正负阈值、训练组数或无模型行为救回。"
    write_json(RESEARCH / "acceptance_outcome.json", {"recorded_at": now(), "status": status, "decision": decision, "local_within_group_ranking_not_promoted": diagnostic["within_group_summary"], "goal_achieved": False, "independent_validation": "NOT_ESTABLISHED", "position_impact": 0}, exclusive=True)
    NEXT.write_text("\n".join(["# 第95轮之后：检验进入选择的等待价值", "", decision, "", "实际进展：119次九项进入模型、四个主策略账户、两个新增较早可用性对照完成，七项必要测试通过，1189完整收益标签经济口径已核。原资料1213路径49组；未自然完成24路径不给标签，其中23未解决和1首次未成交。完整标签1189行属于47组，截至终点整组成熟1179行；无退出后确认的额外权益。首模型2016-10-10，较早429个原点无模型，首次交易2016-10-11。", "", "诊断分开了模型等待与进入筛选：main可用性对照直接复用原32自然退出，早新跑两个同可用性对照。主BASE/STRESS对照夏普0.334401/0.310794，早0.253290/0.232237；新筛选四情景均弱。主对照净比新多31079.65086/30142.34436元，早多5593.0455/5544.07912元。不要重跑。", "", "同时存在反证：主历史12个同时有接受与拒绝路径的完整信号组，接受路径的未来整次平均收益有10组更好、2组更差，等组平均差+2.3988326个百分点；早只有1个两类齐全组，差+0.0343526个百分点。相邻路径重叠，且实际账户只会选择其中部分路径，这不是12个独立胜率样本、不是新的账户收益，也不是改阈值的许可。", "", "下一96先检查把立即进入与继续空仓等待看成两个动作的因果问题是否具备完整的历史连接。用已保存原路径、信号组、原点九因子和95固定买入价值，检查同组下一交易日、终止边界及首次未成交那条路径；不把未来最优进入日直接作为当前指令。尚未登记96模型或账户，不宣称有有效新策略。", "", "已读旧52文档：research/dynamic_continuation_exit_v1.py对应持仓后的下一次退出选择，用八项持仓因子和60次线性拟合价值迭代，失败保持。若研究空仓等待进入，必须明确新状态/动作、现金和奖励单位以及截尾，不能只是给旧95换阈值或重训相同目标。重复核对也确认RSRS/高低价回归第35轮、进入时锁定专家第40轮、波动选择第65轮已做且关闭；65明确不增加锁定救回，跳过这些重复方向。", "", "最快继续使用保存免费资料、完整原ETF评价、固定费用和实际账户，不恢复EPS、公募或其他慢来源，不加GPT包和多余安全审计。原91主基础1.233仍是局部候选，完整稳定超额和高夏普目标未完成。", ""]), encoding="utf-8")
    OUT.mkdir(parents=True)
    lines = ["# 第95轮：完整交易收益筛选进入", "", decision, "", "## 主历史：2020年1月2日至2026年8月14日开盘", "", *table(result["all_metrics"]), "## 较早历史：2015年1月5日至2019年12月31日开盘", "", *table(result["earlier_diagnostics"]), "## 为什么没有达标", "", "主基础年化1.37%、最大回撤23.04%，主压力年化1.10%、回撤23.35%；较早基础年化1.07%、回撤14.59%。主历史原条件有546个进入原点，模型只对其中184个给出正预测，实际13次买入、13次卖出，持股451个收盘。筛选减少了机会，却没有得到足够好的实际持仓。", "", "较早历史直到2016年10月10日收盘才有首个成熟模型，次日首次买入。前429个模型原点无预测，不能把这段现金等待解读为预判了2015年的风险。较早共有574个原始机会原点，其中195个无模型，361个为正预测，最终完成5个交易周期。", "", "为区分延后启用和模型筛选的影响，额外登记一个归因对照：同样等模型可用后才按原条件进入，但不检查预测正负，原自然退出保持。主历史所有模型均可用，直接复用原自然退出账本；较早只新增两个费用账户。较早基础对照夏普0.253，新筛选0.192；主基础对照0.334，新筛选0.177。两段两费用均表明，正负筛选本身没有增加账户收益。", "", "局部预测统计并非完全没有线索：在主历史12个同时具有接受、拒绝路径的完整信号组里，接受路径的平均未来整次收益有10组更高，等组平均差约2.40个百分点。但同组路径互相重叠，真实账户只取决于当时能否买入、具体买入日、之后退出及重新进入资格。这个局部比较不能直接变成一只实际账户的收益，更不能当作独立验证。下一步要检验进入与等待的实际决策关系。", "", "## 全部中文因子与进入退出规则", "", *(ROOT / cfg["rules"]).read_text(encoding="utf-8").splitlines()[1:], "", f"七项必要测试通过。保存核对覆盖1189个完整交易标签、{len(fit_checks)}个成熟模型的加权正规方程、{len(prediction_checks)}个可用预测记录、四个策略账户和{len(cycles)}个完整策略周期。额外两个较早对照已经单独完成，归因对照总计38个完整周期（含两个复用的主历史账本）。没有重新拟合或重跑旧策略。每月实际九项系数见同目录《每月九项进入模型中文规则.md》。", ""]
    DOCUMENT.write_text("\n".join(lines), encoding="utf-8")
    for path in RESEARCH.iterdir():
        if path.is_file() and path.suffix in {".csv", ".json", ".md"}:
            shutil.copy2(path, OUT / path.name)
    shutil.copy2(CONFIG, OUT / "冻结设置.json")
    shutil.copy2(NEXT, OUT / "下一项研究方向.md")
    for path in (RESEARCH / "entry_filter_attribution").iterdir():
        if path.is_file() and path.suffix in {".csv", ".json"}:
            shutil.copy2(path, OUT / f"筛选归因_{path.name}")
    for period, label in [("evaluation", "主评价"), ("earlier_diagnostic", "较早历史")]:
        for cost in cfg["costs"]:
            for kind in ["ledger", "decisions"]:
                pd.read_parquet(RESEARCH / period / cost / f"{PRIMARY}_{kind}.parquet").to_csv(OUT / f"{label}_{cost}_{kind}.csv", index=False, encoding="utf-8-sig")
    record = {"round": 95, "study": result["study_id"], "title": "九项进入前因子预测完整交易收益", "status": status, "result": str((RESEARCH / "result.json").relative_to(ROOT)), **{k: result[k] for k in ["candidate_configurations", "evaluation_accounts", "new_accounts_generated", "reused_control_accounts", "earlier_diagnostic_accounts", "new_earlier_diagnostic_accounts", "new_model_fits", "new_reference_accounts"]}, "additional_diagnostic_new_earlier_accounts": 2, "additional_diagnostic_reused_main_accounts": 2, "evaluated_candidate_source_runs": 1, "primary_base": metric(result, PRIMARY), "primary_stress": metric(result, PRIMARY, cost="STRESS"), "post_selected_best_base": metric(result, PRIMARY)}
    index["completed_rounds"].append(record)
    for key in ["registered_configurations_in_this_resumption", "evaluated_configurations_in_this_resumption", "evaluated_candidate_source_runs_including_corrected_replays", "registered_candidate_source_runs_including_unrun_legacy_bindings"]:
        index[key] += 1
    index["evaluation_accounts_in_this_resumption"] += result["evaluation_accounts"]
    index.update(updated_at=now(), latest_completed_round=record, running_studies=[], goal_achieved=False, status="ROUND95_ENTRY_FILTER_PORTFOLIO_WORSE_LOCAL_RANKING_ONLY_FULL_GOAL_NOT_MET", count_warning="95轮，370不同设置，386已评价来源版本，391登记含5旧未运行，1396主评价记录；另95新增两个较早归因对照不计新候选或主记录。", next_work={"status": "ENTRY_WAIT_VALUE_TRANSITION_DIAGNOSTIC_NOT_REGISTERED", "focus": "检查同组进入与等待动作的完整历史连接及经济单位，保留局部排序但以真实账户验收", "source": str(NEXT.relative_to(ROOT))}, process_state_note="95完成119拟合、四新策略账户、两个新较早归因对照及中文交付；96仅为进入等待价值的连接诊断方向。")
    index["deliveries"].append({"created_at": now(), "type": "ENTRY_PAYOFF_GATE_ROUND95_FAST_CHINESE_RESULTS", "rounds": [95], "directory": str(OUT), "main_document": str(DOCUMENT), "new_gpt_review_archive_created": False})
    write_json(INDEX, index)
    write_json(OUT / "交付回执.json", {"created_at": now(), "main_document": str(DOCUMENT), "goal_achieved": False, "new_gpt_review_archive_created": False}, exclusive=True)
    print(json.dumps({"交付": str(DOCUMENT), "必要核对": receipt, "利润集中": concentration, "归因摘要": diagnostic["within_group_summary"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
