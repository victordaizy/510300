"""核对周期固定截距方程、实际预测和分年增量，交付局部改善。"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from research.signed_volume_within_v1 import ROOT, OUT, CONFIG, PRIMARY
from research.signed_volume_within_inputs_v1 import FEATURES
from research.learned_cycle_exit_v1 import training_rows
from research.adaptive_allocation_v1 import normalize_dividends, summarize
from research.intraday_overnight_increment_v1 import now, digest, require, write_json
from research.fast_round_delivery_v1 import deliver_round
from scripts.finalize_round60_20260907 import metric
from scripts.finalize_round96_20260908 import value_equal
from scripts.review_round74_saved import saved_cycles


def main():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    result = json.loads((OUT / "result.json").read_text(encoding="utf-8"))
    require(not result["historical_point_target_met"], "出现目标点值时不能仅交付未达标局部改善")
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "周期内模型冻结来源改变")
    data = pd.read_parquet(ROOT / cfg["features"])
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    samples = pd.read_parquet(ROOT / cfg["samples"])
    samples = samples[samples.signal.eq("D60_INTRA")].reset_index(drop=True)
    factors = pd.read_parquet(ROOT / cfg["saved_factors"])
    samples[FEATURES[-1]] = factors[FEATURES[-1]].iloc[samples.origin_index.to_numpy(int)].to_numpy()
    pd.testing.assert_frame_equal(samples, pd.read_parquet(OUT / "extended_reference_samples.parquet"))
    models = json.loads((OUT / "saved_models.json").read_text(encoding="utf-8"))["models"]
    originals = json.loads((ROOT / cfg["saved_models"]).read_text(encoding="utf-8"))["models"]["D60_INTRA__RIDGE"]
    members = pd.read_parquet(OUT / "training_memberships.parquet")
    checks, by_date = [], {}
    require([m["fit_index"] for m in models] == [m["fit_index"] for m in originals], "周期内拟合日程不同")
    for record, old in zip(models, originals, strict=True):
        by_date[pd.Timestamp(record["fit_origin"])] = record
        require(record["status"] == old["status"] and record["training_cycles"] == old["training_cycles"] and record["training_rows"] == old["training_rows"], "原成熟周期或支持时点改变")
        if record["model"] is None:
            continue
        rows, ids = training_rows(samples, record["fit_index"], cfg)
        require(ids == record["training_cycles"] and rows.exit_index.le(record["fit_index"]).all(), "周期截距包含未结束的周期")
        membership = members[members.fit_index.eq(record["fit_index"])].sort_values(["cycle_id", "origin_index"])
        np.testing.assert_array_equal(rows[["cycle_id", "origin_index", "exit_index"]], membership[["cycle_id", "origin_index", "exit_index"]])
        np.testing.assert_allclose(rows.sample_weight, membership.sample_weight, atol=0, rtol=0)
        x, y, w = rows[FEATURES].to_numpy(float), rows.target.to_numpy(float), rows.sample_weight.to_numpy(float)
        mean = np.average(x, axis=0, weights=w); scale = np.sqrt(np.average((x-mean)**2, axis=0, weights=w)); scale = np.where(scale > 1e-12, scale, 1.)
        m = record["model"]; coefficient = np.asarray(m["coefficients"])
        np.testing.assert_allclose(m["mean"], mean, atol=1e-12, rtol=0)
        np.testing.assert_allclose(m["scale"], scale, atol=1e-12, rtol=0)
        np.testing.assert_allclose(m["mean"][:8], old["model"]["mean"], atol=1e-12, rtol=0)
        np.testing.assert_allclose(m["scale"][:8], old["model"]["scale"], atol=1e-12, rtol=0)
        z = np.clip((x-mean)/scale, -5, 5); residual = np.empty(len(rows))
        intercepts = {g["cycle_id"]: g for g in m["cycle_intercepts"]}
        require(set(intercepts) == set(ids), "周期截距成员不同")
        maximum_intercept_gradient = 0.
        for cycle in ids:
            mask = rows.cycle_id.to_numpy() == cycle
            mz, my = np.average(z[mask], axis=0, weights=w[mask]), np.average(y[mask], weights=w[mask])
            a = my-mz@coefficient
            stored = intercepts[cycle]
            np.testing.assert_allclose(stored["standardized_feature_mean"], mz, atol=1e-12, rtol=0)
            require(abs(stored["target_mean"]-my) < 1e-12 and abs(stored["cycle_intercept"]-a) < 1e-12 and stored["rows"] == int(mask.sum()), "周期目标均值或截距不符")
            residual[mask] = a+z[mask]@coefficient-y[mask]
            maximum_intercept_gradient = max(maximum_intercept_gradient, abs(np.sum(w[mask]*residual[mask])))
        gradient = z.T@(w*residual)+cfg["ridge_alpha"]*coefficient
        require(maximum_intercept_gradient < 1e-10 and np.abs(gradient).max() < 1e-10, "保存结果不满足不惩罚周期截距的岭回归方程")
        average = np.mean([g["cycle_intercept"] for g in m["cycle_intercepts"]])
        require(abs(average-m["intercept"]) < 1e-12, "新实际周期截距不是已成熟周期等权平均")
        checks.append({"fit_origin": record["fit_origin"], "cycles": len(ids), "rows": len(rows), "maximum_coefficient_gradient": float(np.abs(gradient).max()), "maximum_intercept_gradient": maximum_intercept_gradient})
    require(len(checks) == 114 and result["failed_fits"] == 0, "周期内模型数或求解失败状态不同")
    cycles, accounts, differences, annual = [], [], [], []
    score_count, state_count = 0, 0
    for period in ["evaluation", "earlier_diagnostic"]:
        for cost in cfg["costs"]:
            folder = OUT / period / cost
            ledger = pd.read_parquet(folder / f"{PRIMARY}_ledger.parquet")
            decisions = pd.read_parquet(folder / f"{PRIMARY}_decisions.parquet")
            native = pd.read_csv(folder / f"{PRIMARY}_cycles.csv")
            for cycle_id, group in decisions[decisions.learning_cycle_id.notna()].groupby("learning_cycle_id", sort=False):
                cycle = native[native.cycle_id.eq(cycle_id)].iloc[0]
                own = ledger[ledger.cycle_id.eq(cycle_id)].copy().set_index("date")
                value = own.shares*own.mark+own.dividend_recognized.cumsum()
                peak = pd.Series(np.maximum.accumulate(np.r_[cycle.entry_cost_cny, value.to_numpy()])[1:], index=own.index)
                count = 0
                for row in group.itertuples():
                    x = np.array([getattr(row, f) for f in FEATURES])
                    actual = value.loc[row.origin]
                    expected = [np.log1p(row.origin_index-cycle.entry_index+1), actual/cycle.entry_cost_cny-1, actual/peak.loc[row.origin]-1, cycle["mode"],
                                data.mom5.iloc[row.origin_index], data.mom20.iloc[row.origin_index], data.sma120.iloc[row.origin_index], data.vol20.iloc[row.origin_index], factors[FEATURES[-1]].iloc[row.origin_index]]
                    np.testing.assert_allclose(x, expected, atol=1e-12, rtol=0)
                    if row.learning_status == "PREDICTION_AVAILABLE":
                        record = by_date[row.learning_fit_origin]
                        require(record["latest_exit_index"] <= record["fit_index"] <= row.origin_index, "新周期实际调用未来截距或系数")
                        m = record["model"]
                        a = np.mean([g["cycle_intercept"] for g in m["cycle_intercepts"]])
                        score = a+np.clip((x-m["mean"])/m["scale"], -5, 5)@np.asarray(m["coefficients"])
                        require(abs(score-row.continuation_prediction) < 1e-12, "实际预测没有使用成熟平均截距和自身九状态")
                        count = count+1 if score < 0 else 0
                        score_count += 1
                    else:
                        require(pd.isna(row.continuation_prediction), "没有模型时填入预测")
                        count = 0
                    require(count == row.negative_confirmation_count and row.learned_exit_requested == (count >= 2), "实际两日负预测确认或新周期重置不符")
                    state_count += 1
            previous = np.r_[cfg["initial_capital"], ledger.equity.iloc[:-1]]
            np.testing.assert_allclose(ledger.cash+ledger.shares*ledger.mark+ledger.dividend_receivable, ledger.equity, atol=1e-6, rtol=0)
            np.testing.assert_allclose(ledger.equity/previous-1, ledger.net_return, atol=1e-12, rtol=0)
            measured = summarize(ledger, cfg); stored = metric(result, PRIMARY, period, cost)
            for k in ["net_sharpe", "annualized_return", "max_drawdown", "cumulative_return", "trade_count", "commission", "slippage_cost"]:
                require(value_equal(measured[k], stored[k]), "周期内完整账户绩效不能复算")
            complete = saved_cycles(ledger, dividends, cfg)
            cycles.extend({"period": period, "cost": cost, **c} for c in complete)
            parent = pd.read_parquet(folder / "WITHIN_CYCLE_EXIT_ledger.parquet")
            require(pd.DatetimeIndex(ledger.date).equals(pd.DatetimeIndex(parent.date)), "新旧实际账户日历不同")
            differences.append({"period": period, "cost": cost, "terminal_nav_difference": float(ledger.equity.iloc[-1]-parent.equity.iloc[-1]),
                                **{f"{k}_difference": float(ledger[k].sum()-parent[k].sum()) for k in ["price_pnl", "dividend_recognized", "commission", "slippage_cost"]}})
            for year, frame in ledger.groupby(ledger.date.dt.year):
                base = parent[parent.date.dt.year.eq(year)]
                current_m, parent_m = summarize(frame, cfg), summarize(base, cfg)
                delta = current_m["cumulative_return"]-parent_m["cumulative_return"]
                annual.append({"period": period, "cost": cost, "year": int(year), "new_return": current_m["cumulative_return"], "parent_return": parent_m["cumulative_return"],
                               "return_difference": delta, "new_sharpe": current_m["net_sharpe"], "parent_sharpe": parent_m["net_sharpe"], "comparison": "改善" if delta > 1e-10 else "下降" if delta < -1e-10 else "相同"})
            accounts.append({"period": period, "cost": cost, "days": len(ledger), "cycles": len(complete), "net_sharpe": measured["net_sharpe"]})
    for name, rows in [("saved_training_checks.csv", checks), ("saved_actual_cycles.csv", cycles), ("saved_account_checks.csv", accounts), ("saved_profit_differences.csv", differences), ("saved_yearly_increment.csv", annual)]:
        pd.DataFrame(rows).to_csv(OUT / name, index=False, encoding="utf-8-sig")
    receipt = {"verified_at": now(), "status": "SAVED_NINE_FACTOR_EQUATIONS_OWN_PREDICTIONS_AND_ACCOUNT_INCREMENT_CHECKED", "mature_models": len(checks), "reference_training_states": len(samples),
               "actual_holding_states": state_count, "predictions_recomputed": score_count, "actual_accounts": len(accounts), "complete_cycles": len(cycles), "yearly_increment_rows": len(annual),
               "new_models_or_accounts": 0, "reviewer_source_sha256": digest(Path(__file__)), "security_audit_performed": False}

    timing = []
    for period in ["evaluation", "earlier_diagnostic"]:
        for cost in cfg["costs"]:
            current = pd.read_csv(OUT / period / cost / f"{PRIMARY}_cycles.csv")
            parent = pd.read_csv(ROOT / "reports/research/510300_within_cycle_exit_v1" / period / cost / "WITHIN_CYCLE_EXIT_cycles.csv")
            paired = current.merge(parent, on="entry_date", how="outer", suffixes=("_new", "_old"), indicator=True)
            changed = paired[paired["_merge"].ne("both") | paired.exit_date_new.ne(paired.exit_date_old)]
            timing.extend({"period": period, "cost": cost, **row} for row in changed[["entry_date", "exit_date_new", "exit_date_old", "exit_reasons_new", "exit_reasons_old", "net_profit_cny_new", "net_profit_cny_old"]].to_dict("records"))
    pd.DataFrame(timing).to_csv(OUT / "changed_cycle_timing.csv", index=False, encoding="utf-8-sig")
    pair = []
    for cost in ["BASE", "STRESS"]:
        d = pd.read_parquet(OUT / "evaluation" / cost / f"{PRIMARY}_decisions.parquet")
        pair.append(d[d.origin.eq("2021-02-18")].iloc[0])
    a, b = pair
    require(a.learning_fit_origin == b.learning_fit_origin, "敏感性比较模型不同")
    model = by_date[a.learning_fit_origin]["model"]
    x, y = a[FEATURES].to_numpy(float), b[FEATURES].to_numpy(float)
    components = (np.clip((y-model["mean"])/model["scale"], -5, 5)-np.clip((x-model["mean"])/model["scale"], -5, 5))*model["coefficients"]
    require(abs(components.sum()-(b.continuation_prediction-a.continuation_prediction)) < 1e-12, "费用状态预测分解不符")
    require(np.abs(np.delete(components, 1)).max() < 1e-12 and a.continuation_prediction < 0 < b.continuation_prediction, "此次变号不只来自浮盈亏状态")
    sensitivity = {"origin": "2021-02-18", "model_origin": str(a.learning_fit_origin.date()), "base_score": float(a.continuation_prediction), "stress_score": float(b.continuation_prediction),
                   "score_difference": float(components.sum()), "sole_different_score_input": "cycle_return", "base_cycle_return": float(a.cycle_return), "stress_cycle_return": float(b.cycle_return),
                   "base_negative_confirmation": int(a.negative_confirmation_count), "stress_negative_confirmation": int(b.negative_confirmation_count),
                   "base_exit_request": bool(a.learned_exit_requested), "stress_exit_request": bool(b.learned_exit_requested),
                   "note": "同一预测日和同一模型，其他八项对预测差贡献为零。已付费用使自身浮盈亏不同，学习判断因此跨过零线；不代表后续利润损失全部可归为费用金额。"}
    write_json(OUT / "cost_state_sensitivity.json", sensitivity, exclusive=True)
    receipt.update(changed_cycle_timing_rows=len(timing), cost_state_sensitivity_checked=True)
    write_json(OUT / "saved_verification_receipt.json", receipt, exclusive=True)
    decision = "第116轮新增成交份额仅有主基础成本小幅改善，不提升为稳定候选，结束本次新增信息。主基础净夏普0.896、年化7.08%，原114为0.884、6.97%；压力净夏普却从0.831降至0.700，最大回撤从11.68%增至17.24%。较早基础／压力净夏普0.736／0.713，均低于114的0.752／0.728。不能用基础单点提升忽略成本与较早退步。保留114原固定方法，完整目标未达到。"
    detail = "六项必要测试5.01秒，114次成熟拟合及四个新账户5.62秒，零新参考、零求解失败、零未成交；27个原支持不足月份保留。已核114个九因子方程、1176个实际持仓状态、768个有效预测及58个完整周期。主基础只因2022年一笔退出晚一天增加收益；较早2019年一笔早退一天减少收益。压力2021年一笔退出延后，从原2月19日至3月1日，是本次重要退步来源。"
    next_path = ROOT / "docs/510300_AFTER_SIGNED_VOLUME_WITHIN_20260908.md"
    next_lines = ["# 第116轮之后：分离市场路径与已付交易费用", "", decision, "", detail, "",
        "2021年2月18日基础预测-0.00020392388470619714，压力+0.0003124770901077457，差0.000516400974813944；差异全部来自cycle_return，另外八项贡献为零。基础连续两次负预测请求下一开盘退出，压力计数清零继续持有，直到原追踪止损触发。保存分解见本轮cost_state_sensitivity.json和changed_cycle_timing.csv，无需重复核对。", "",
        "这提示原学习输入把已付买入费用混入了市场路径状态。已经付出的买入佣金和滑点不应单独成为预测市场后续继续价值的依据；但未来卖出成本和实际账户风险仍必须计算。它是一个局部设计敏感性，不是所有失败的因果解释。成本决策的一般原则可见MIT管理会计课程：https://www.ocw.mit.edu/courses/15-963-management-accounting-and-control-spring-2007/eb7db9ee4af99a5f9c54d4b2c42a8886_lec3.pdf 。", "",
        "下一候选从114八因子周期内方法出发，只将学习用的周期浮盈亏和回撤改为不含已付买入费用的市场路径：基准为本笔实际买入日的原开盘价，当前每份价值为收盘价加该笔已经取得并确认的每份分红，峰值从入场原开盘价开始逐收盘更新。新实际账户的现金、费用、价格止损、实际回撤保护与交易规则全部照常；它们与学习信号的市场状态分开。", "",
        "这不是给116成交量因子调窗口或改阈值，也不重开115截距第二层。116新增量信息结束，114原样保存；新候选重新训练八项，其中两个路径定义明确替换。先只核原完整参考账本能否逐时恢复每份价值和无费用峰值，不能从筛掉中间日的1461训练状态推峰值。有界沉没成本/成本中性/market_cycle_return检索目前没有旧实现，仍以具体来源确认后才登记。117尚未选定登记、拟合或运行账户。", "",
        "保留实际净成本、登记分红、完整日历和成熟周期。只研究510300及现金，不补慢源或制作GPT包。继续实际目标，上一回合与本回合都是进展，当前没有阻塞。"]
    delivered = deliver_round(ROOT, OUT, CONFIG, ROOT / "deliverables/510300成交份额周期内退出_第116轮_20260908/成交份额周期内退出_结果及全部中文规则.md",
        "成交份额补充周期内退出", "COMPLETED_BASE_ONLY_GAIN_COST_SENSITIVE_NO_PROMOTION_NOT_TARGET", decision, detail, next_path, next_lines,
        "SUNK_ENTRY_COST_SEPARATION_REFERENCE_STATE_PREFLIGHT_PENDING", "从原完整参考路径核对不含买入沉没费用的两个学习状态")
    print(json.dumps({"交付": delivered, "核对": receipt, "费用状态敏感性": sensitivity, "改变周期": timing}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
