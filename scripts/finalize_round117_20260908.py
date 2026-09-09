"""核对周期固定截距方程、实际预测和分年增量，交付局部改善。"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from research.market_path_exit_v1 import ROOT, OUT, CONFIG, PRIMARY
from research.market_path_exit_inputs_v1 import FEATURES
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
    original_samples = samples.copy()
    samples = pd.read_parquet(ROOT / cfg["saved_market_samples"])
    pd.testing.assert_frame_equal(original_samples, samples[original_samples.columns])
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
        np.testing.assert_allclose(np.asarray(m["mean"])[[0, 3, 4, 5, 6, 7]], np.asarray(old["model"]["mean"])[[0, 3, 4, 5, 6, 7]], atol=1e-12, rtol=0)
        np.testing.assert_allclose(np.asarray(m["scale"])[[0, 3, 4, 5, 6, 7]], np.asarray(old["model"]["scale"])[[0, 3, 4, 5, 6, 7]], atol=1e-12, rtol=0)
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
                raw_entry_open = float(data.open.iloc[int(cycle.entry_index)])
                unit_values = value/cycle.entry_quantity
                market_peak = pd.Series(np.maximum.accumulate(np.r_[raw_entry_open, unit_values.to_numpy()])[1:], index=own.index)
                count = 0
                for row in group.itertuples():
                    x = np.array([getattr(row, f) for f in FEATURES])
                    actual = value.loc[row.origin]
                    expected = [np.log1p(row.origin_index-cycle.entry_index+1), actual/cycle.entry_quantity/raw_entry_open-1, actual/cycle.entry_quantity/market_peak.loc[row.origin]-1, cycle["mode"],
                                data.mom5.iloc[row.origin_index], data.mom20.iloc[row.origin_index], data.sma120.iloc[row.origin_index], data.vol20.iloc[row.origin_index]]
                    np.testing.assert_allclose(x, expected, atol=1e-12, rtol=0)
                    np.testing.assert_allclose([row.market_entry_open, row.market_unit_value, row.market_unit_peak], [raw_entry_open, actual/cycle.entry_quantity, market_peak.loc[row.origin]], atol=1e-12, rtol=0)
                    np.testing.assert_allclose([row.account_cycle_return, row.account_cycle_drawdown], [actual/cycle.entry_cost_cny-1, actual/peak.loc[row.origin]-1], atol=1e-12, rtol=0)
                    if row.learning_status == "PREDICTION_AVAILABLE":
                        record = by_date[row.learning_fit_origin]
                        require(record["latest_exit_index"] <= record["fit_index"] <= row.origin_index, "新周期实际调用未来截距或系数")
                        m = record["model"]
                        a = np.mean([g["cycle_intercept"] for g in m["cycle_intercepts"]])
                        score = a+np.clip((x-m["mean"])/m["scale"], -5, 5)@np.asarray(m["coefficients"])
                        require(abs(score-row.continuation_prediction) < 1e-12, "实际预测没有使用成熟平均截距和自身八状态")
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
            require(ledger.filled_quantity.equals(parent.filled_quantity) and ledger.shares.equals(parent.shares), "出现实际份额增量，不能沿用零收益增量结论")
            np.testing.assert_allclose(ledger.equity, parent.equity, atol=1e-6, rtol=0)
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
    receipt = {"verified_at": now(), "status": "SAVED_MARKET_PATH_EQUATIONS_COST_INVARIANCE_AND_IDENTICAL_ACCOUNT_PATHS_CHECKED", "mature_models": len(checks), "reference_training_states": len(samples),
               "actual_holding_states": state_count, "predictions_recomputed": score_count, "actual_accounts": len(accounts), "complete_cycles": len(cycles), "yearly_increment_rows": len(annual),
               "new_models_or_accounts": 0, "reviewer_source_sha256": digest(Path(__file__)), "security_audit_performed": False}

    impact, invariance = [], []
    for period in ["evaluation", "earlier_diagnostic"]:
        own_decisions = {}
        for cost in cfg["costs"]:
            current = pd.read_parquet(OUT / period / cost / f"{PRIMARY}_decisions.parquet")
            parent = pd.read_parquet(ROOT / "reports/research/510300_within_cycle_exit_v1" / period / cost / "WITHIN_CYCLE_EXIT_decisions.parquet")
            paired = current.merge(parent, on="origin_index", suffixes=("_new", "_old"), validate="one_to_one")
            valid = paired.learning_status_new.eq("PREDICTION_AVAILABLE") & paired.learning_status_old.eq("PREDICTION_AVAILABLE")
            pairs = paired[valid]
            impact.append({"period": period, "cost": cost, "comparable_predictions": len(pairs),
                           "changed_prediction_signs": int(pairs.continuation_prediction_new.lt(0).ne(pairs.continuation_prediction_old.lt(0)).sum()),
                           "changed_learned_exit_requests": int(pairs.learned_exit_requested_new.ne(pairs.learned_exit_requested_old).sum()),
                           "maximum_prediction_difference": float((pairs.continuation_prediction_new-pairs.continuation_prediction_old).abs().max())})
            own_decisions[cost] = current
        paired = own_decisions["BASE"].merge(own_decisions["STRESS"], on="origin_index", suffixes=("_base", "_stress"), validate="one_to_one")
        selected = paired[paired.learning_status_base.eq("PREDICTION_AVAILABLE") & paired.learning_status_stress.eq("PREDICTION_AVAILABLE") & paired.learning_cycle_id_base.eq(paired.learning_cycle_id_stress)]
        maximum = 0.
        for feature in FEATURES:
            error = float((selected[feature+"_base"]-selected[feature+"_stress"]).abs().max())
            maximum = max(maximum, error)
        prediction_difference = float((selected.continuation_prediction_base-selected.continuation_prediction_stress).abs().max())
        require(maximum < 1e-12 and prediction_difference < 1e-12, "相同入场市场路径仍受基础压力费用影响")
        invariance.append({"period": period, "same_cycle_comparable_predictions": len(selected), "maximum_feature_difference": maximum, "maximum_prediction_difference": prediction_difference})
    pd.DataFrame(impact).to_csv(OUT / "saved_decision_impact.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(invariance).to_csv(OUT / "saved_cost_invariance.csv", index=False, encoding="utf-8-sig")
    receipt.update(identical_execution_and_nav_accounts=4, same_cycle_cross_cost_predictions=sum(row["same_cycle_comparable_predictions"] for row in invariance))
    write_json(OUT / "saved_verification_receipt.json", receipt, exclusive=True)
    decision = "第117轮市场路径与已付买入费用分离通过定义和费用不变性检查，但四个真实账户的成交份额及净值路径与第114轮一致，没有策略收益增量。主基础／压力净夏普仍为0.884／0.831，较早0.752／0.728；主基础年化6.97%、最大回撤11.44%。保留这一局部诊断及状态构造方法，不作为提高夏普的新策略，不改基准价格或混合费用继续搜索。完整目标未达到。"
    detail = "六项必要测试7.15秒，114次成熟拟合和四个新账户4.34秒；27个原支持不足月份保留，零求解失败、零新参考、零未成交。已核114个周期内方程、1170个自身市场与账户状态、762个有效预测和58个完整周期；基础压力同入场周期的381个可比预测在浮点精度内一致。四个完整账户与114净值最大差为零。"
    next_path = ROOT / "docs/510300_AFTER_MARKET_PATH_EXIT_20260908.md"
    next_lines = ["# 第117轮之后：先比较按持仓期限倒推退出的方法", "", decision, "", detail, "",
        "116成交份额单点收益微升但压力大幅退步；已定位2021年费用浮盈亏使预测变号。117消除学习输入的该机械费用依赖，却没有改变原114任何实际净值，因此这不是目前收益不足的充分解释。旧114仍是局部候选；116和117不继续调因子窗、基准、成本混合或阈值。", "",
        "下一方向考虑有限持有期限的逐期退出估计：在已经完整成熟的参考周期中，按持仓交易日从后向前估计继续价值，并更新这些训练路径未来采用的退出选择。它与52把所有持仓天数混在同一线性模型、固定六十次自举回归不同；与96空仓机会组买入或等待迭代也不同。不能只把52改个迭代次数换名，需要先检查逐期成熟支持和完整时钟，再决定是否值得登记。", "",
        "可参考Longstaff与Schwartz的逐期最小二乘继续价值方法原论文：https://escholarship.org/uc/item/43n1k4jb 。论文是期权定价，不证明510300盈利；此处若采用的是历史完整周期上的有限退出估计，不增加期权、蒙特卡洛假想价格或交易权限。", "",
        "已读docs/510300_DYNAMIC_CONTINUATION_EXIT_V1.md及docs/510300_ENTRY_WAIT_POLICY_V1.md，有界检索未见Longstaff/Schwartz/LSMC/按持仓天数倒推的旧实现。下一步只核原最近20成熟周期在各持仓日的独立周期数，不算新收益；若逐日拆分后大多没有原最低周期支持，停止这条细分方案，不能降低门槛凑模型。单次或两次确认与训练未来退出决策必须一致，尚未选定规则。第118轮尚未登记、拟合或运行账户。", "",
        "只有结构不同且支持可用的有限方法才继续实际回测；跳过旧52迭代/确认/惩罚附近搜索和旧96动作门槛。保留完整费用、现金日、成熟标签及真实执行，EPS和慢源暂停，无GPT包，目标继续。"]
    delivered = deliver_round(ROOT, OUT, CONFIG, ROOT / "deliverables/510300市场路径费用分离_第117轮_20260908/市场路径费用分离_结果及全部中文规则.md",
        "市场路径与已付买入费用分离", "COMPLETED_MARKET_STATE_COST_INVARIANCE_NO_ACCOUNT_INCREMENT_NOT_TARGET", decision, detail, next_path, next_lines,
        "FINITE_HORIZON_STOPPING_AGE_SUPPORT_PREFLIGHT_PENDING", "先核按持仓天数拆分后是否仍有原最低成熟周期支持")
    print(json.dumps({"交付": delivered, "核对": receipt, "实际决策增量": impact, "费用不变性": invariance}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
