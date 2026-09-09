"""核对两层系数复用、进入时钟和保存账本，结束无夏普增量的改动。"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from research.entry_context_intercept_v1 import ROOT, OUT, CONFIG, PRIMARY
from research.entry_context_intercept_inputs_v1 import FEATURES, CONTEXT_FEATURES
from research.adaptive_allocation_v1 import normalize_dividends, summarize
from research.intraday_overnight_increment_v1 import now, digest, require, write_json
from research.fast_round_delivery_v1 import deliver_round
from scripts.finalize_round60_20260907 import metric
from scripts.finalize_round96_20260908 import value_equal
from scripts.review_round74_saved import saved_cycles


def main():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    result = json.loads((OUT / "result.json").read_text(encoding="utf-8"))
    require(not result["historical_point_target_met"], "出现目标点值时须另行核查目标")
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "进入上下文冻结来源改变")
    data = pd.read_parquet(ROOT / cfg["features"])
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    reference = pd.read_csv(ROOT / cfg["reference_cycles"]).set_index("cycle_id", drop=False)
    context = pd.read_parquet(OUT / "reference_entry_context.parquet").set_index("cycle_id", drop=False)
    for cycle in reference.itertuples(index=False):
        origin = int(cycle.entry_index)-1
        row = context.loc[cycle.cycle_id]
        require(pd.Timestamp(cycle.entry_origin) == data.date.iloc[origin] == row.entry_origin, "参考进入上下文日期不符")
        np.testing.assert_allclose(row[CONTEXT_FEATURES].to_numpy(float), data.loc[origin, ["mom20", "sma120", "vol20"]].to_numpy(float), atol=0, rtol=0)
    models = json.loads((OUT / "saved_models.json").read_text(encoding="utf-8"))["models"]
    originals = json.loads((ROOT / cfg["saved_models"]).read_text(encoding="utf-8"))["models"]
    members = pd.read_parquet(OUT / "training_memberships.parquet")
    checks, by_date = [], {}
    for record, old in zip(models, originals, strict=True):
        require(record["fit_index"] == old["fit_index"] and record["status"] == old["status"] and record["training_cycles"] == old["training_cycles"], "原拟合日程或成熟成员改变")
        by_date[pd.Timestamp(record["fit_origin"])] = record
        if record["model"] is None:
            continue
        model = record["model"]
        require(model["within_model"] == old["model"], "第一层保存系数或截距被重估或改动")
        ids = record["training_cycles"]
        chosen = context.loc[ids]
        require(chosen.exit_date.le(pd.Timestamp(record["fit_origin"])).all() and record["latest_exit_index"] <= record["fit_index"], "第二层读取尚未结束周期")
        y_map = {g["cycle_id"]: g["cycle_intercept"] for g in old["model"]["cycle_intercepts"]}
        y = np.array([y_map[i] for i in ids]); x = chosen[CONTEXT_FEATURES].to_numpy(float)
        membership = members[members.fit_index.eq(record["fit_index"])].set_index("cycle_id").loc[ids]
        np.testing.assert_allclose(membership[CONTEXT_FEATURES], x, atol=0, rtol=0)
        np.testing.assert_allclose(membership.target_intercept, y, atol=0, rtol=0)
        mean, scale = x.mean(axis=0), x.std(axis=0, ddof=0); scale = np.where(scale > 1e-12, scale, 1.)
        m = model["context_model"]; coefficient = np.asarray(m["coefficients"])
        np.testing.assert_allclose(m["mean"], mean, atol=1e-12, rtol=0)
        np.testing.assert_allclose(m["scale"], scale, atol=1e-12, rtol=0)
        z = np.clip((x-mean)/scale, -5, 5)
        residual = m["intercept"]+z@coefficient-y
        gradient = z.T@residual+cfg["context_ridge_alpha"]*coefficient
        require(abs(residual.sum()) < 1e-10 and np.abs(gradient).max() < 1e-10, "保存模型不满足等周期岭回归方程")
        checks.append({"fit_origin": record["fit_origin"], "cycles": len(ids), "maximum_coefficient_gradient": float(np.abs(gradient).max()), "intercept_gradient": float(residual.sum()), "within_model_unchanged": True})
    require(len(checks) == 114 and result["failed_fits"] == 0 and result["new_within_cycle_model_fits"] == 0, "两层模型次数或失败状态不符")
    cycles, accounts, differences = [], [], []
    score_count, state_count = 0, 0
    for period in ["evaluation", "earlier_diagnostic"]:
        for cost in cfg["costs"]:
            folder = OUT / period / cost
            ledger = pd.read_parquet(folder / f"{PRIMARY}_ledger.parquet")
            decisions = pd.read_parquet(folder / f"{PRIMARY}_decisions.parquet")
            native = pd.read_csv(folder / f"{PRIMARY}_cycles.csv")
            for cycle_id, group in decisions[decisions.learning_cycle_id.notna()].groupby("learning_cycle_id", sort=False):
                cycle = native[native.cycle_id.eq(cycle_id)].iloc[0]
                entry_origin = int(cycle.entry_index)-1
                require(pd.Timestamp(cycle.entry_origin) == data.date.iloc[entry_origin], "实际买入请求时钟不符")
                expected_context = data.loc[entry_origin, ["mom20", "sma120", "vol20"]].to_numpy(float)
                own = ledger[ledger.cycle_id.eq(cycle_id)].copy().set_index("date")
                value = own.shares*own.mark+own.dividend_recognized.cumsum()
                peak = pd.Series(np.maximum.accumulate(np.r_[cycle.entry_cost_cny, value.to_numpy()])[1:], index=own.index)
                count = 0
                for row in group.itertuples():
                    x = np.array([getattr(row, f) for f in FEATURES]); e = np.array([getattr(row, f) for f in CONTEXT_FEATURES])
                    actual = value.loc[row.origin]
                    expected = [np.log1p(row.origin_index-cycle.entry_index+1), actual/cycle.entry_cost_cny-1, actual/peak.loc[row.origin]-1, cycle["mode"],
                                data.mom5.iloc[row.origin_index], data.mom20.iloc[row.origin_index], data.sma120.iloc[row.origin_index], data.vol20.iloc[row.origin_index]]
                    np.testing.assert_allclose(x, expected, atol=1e-12, rtol=0)
                    np.testing.assert_allclose(e, expected_context, atol=0, rtol=0)
                    require(row.entry_context_origin == data.date.iloc[entry_origin], "实际持有期间替换了买入前日期")
                    if row.learning_status == "PREDICTION_AVAILABLE":
                        record = by_date[row.learning_fit_origin]
                        require(record["latest_exit_index"] <= record["fit_index"] <= row.origin_index, "实际调用未来模型")
                        inside, between = record["model"]["within_model"], record["model"]["context_model"]
                        a = between["intercept"]+np.clip((e-between["mean"])/between["scale"], -5, 5)@np.asarray(between["coefficients"])
                        within = np.clip((x-inside["mean"])/inside["scale"], -5, 5)@np.asarray(inside["coefficients"])
                        score = a+within
                        require(abs(a-row.context_intercept_prediction) < 1e-12 and abs(within-row.within_cycle_component) < 1e-12 and abs(score-row.continuation_prediction) < 1e-12, "两层实际预测不能复算")
                        count = count+1 if score < 0 else 0
                        score_count += 1
                    else:
                        require(pd.isna(row.continuation_prediction), "无模型时填入平均截距预测")
                        count = 0
                    require(count == row.negative_confirmation_count and row.learned_exit_requested == (count >= 2), "实际两日确认或新周期重置不符")
                    state_count += 1
            previous = np.r_[cfg["initial_capital"], ledger.equity.iloc[:-1]]
            np.testing.assert_allclose(ledger.cash+ledger.shares*ledger.mark+ledger.dividend_receivable, ledger.equity, atol=1e-6, rtol=0)
            np.testing.assert_allclose(ledger.equity/previous-1, ledger.net_return, atol=1e-12, rtol=0)
            measured = summarize(ledger, cfg); stored = metric(result, PRIMARY, period, cost)
            for key in ["net_sharpe", "annualized_return", "max_drawdown", "cumulative_return", "trade_count", "commission", "slippage_cost"]:
                require(value_equal(measured[key], stored[key]), "两层完整账户绩效不能复算")
            complete = saved_cycles(ledger, dividends, cfg)
            cycles.extend({"period": period, "cost": cost, **cycle} for cycle in complete)
            parent = pd.read_parquet(folder / "WITHIN_CYCLE_EXIT_ledger.parquet")
            require(pd.DatetimeIndex(ledger.date).equals(pd.DatetimeIndex(parent.date)), "新旧实际账户日历不同")
            require(measured["net_sharpe"] < summarize(parent, cfg)["net_sharpe"], "出现相对114的夏普改善，不能沿用本次关闭结论")
            differences.append({"period": period, "cost": cost, "terminal_nav_difference": float(ledger.equity.iloc[-1]-parent.equity.iloc[-1]),
                                **{f"{key}_difference": float(ledger[key].sum()-parent[key].sum()) for key in ["price_pnl", "dividend_recognized", "commission", "slippage_cost"]}})
            accounts.append({"period": period, "cost": cost, "days": len(ledger), "cycles": len(complete), "net_sharpe": measured["net_sharpe"]})
    for name, rows in [("saved_training_checks.csv", checks), ("saved_actual_cycles.csv", cycles), ("saved_account_checks.csv", accounts), ("saved_profit_differences.csv", differences)]:
        pd.DataFrame(rows).to_csv(OUT / name, index=False, encoding="utf-8-sig")
    receipt = {"verified_at": now(), "status": "SAVED_TWO_STAGE_COEFFICIENTS_ENTRY_CLOCK_PREDICTIONS_AND_ACCOUNTS_CHECKED", "mature_context_models": len(checks),
               "unchanged_within_models": len(checks), "reference_entry_contexts": len(context), "actual_holding_states": state_count,
               "predictions_recomputed": score_count, "actual_accounts": len(accounts), "complete_cycles": len(cycles), "new_models_or_accounts": 0,
               "reviewer_source_sha256": digest(Path(__file__)), "security_audit_performed": False}
    write_json(OUT / "saved_verification_receipt.json", receipt, exclusive=True)
    decision = "第115轮进入上下文截距改动结束，不保留。主基础／压力净夏普0.389／0.341，低于114的0.884／0.831；主基础年化从6.97%降至2.99%，最大回撤从11.44%升至17.00%。较早夏普0.750／0.727，也略低于114的0.752／0.728；较早基础年化8.55%，比114的8.52%略高，但不能抵消主历史明显退步。保留114原固定方法及其局部改善，完整目标未达到。"
    detail = "六项必要测试5.05秒，114次第二层回归和四个新账户5.89秒；零第一层重拟合、零新参考、零求解失败。原27个支持不足月份保留。已核36个参考进入状态、114个原样第一层模型及第二层方程、1218个实际持仓状态、810个有效预测、四个完整账户和58个完整周期；零未成交。没有追加阈值搜索或重跑账户。"
    next_path = ROOT / "docs/510300_AFTER_ENTRY_CONTEXT_INTERCEPT_20260908.md"
    next_lines = ["# 第115轮之后：保留114，转向尚未直接进入模型的成交量信息", "", decision, "", detail, "",
        "115仅较早年化微升，两段两档夏普都下降，关闭本次三因子周期截距第二层；不试不同三因子、截距混合、惩罚、窗口或阈值。114八项周期内模型仍是局部候选，2018恶化和主最大回撤略增的限制保留。", "",
        "下一方向考虑在114周期内学习中加入收盘涨跌对应的成交份额平衡。原八因子没有成交量；按上涨日增加成交份额、下跌日减少、平盘记零的思路可见TradingView官方OBV说明：https://www.tradingview.com/support/solutions/43000502593-on-balance-volume-obv/ 。拟用含分红涨跌方向，汇总连续二十日有符号成交份额再除同期全部成交份额，作为一个有界输入。这个二十日归一化版本是本项目待研究定义，不等于标准累计OBV。", "",
        "它不是主动买卖净额、公募净申购或真实资金净流入，不能借指标名称声称补齐了公募因子。56按收盘在日内区间的位置给成交量权重，101按典型价格方向给成交额权重，112按区间中点变化及区间/量构造进出场，含义不同；这些失败原样保留。本次research/docs/config/tests的有界OBV/能量潮/平衡成交量检索未见同名实现，仍需在具体相关源中确认没有等价计算。", "",
        "先完成来源定义及缺失/除息/量单位的小范围核对，不读新账户；如可用，只登记一个九因子周期内模型，保留114平均周期截距、原标签、完整成熟周期及所有交易规则。不得重开115进入上下文第二层或旧成交量家族相邻参数。第116轮尚未登记、拟合或计算账户。", "",
        "盘中固定/追踪保护已在30做过，跳过重复的止损挂单方案；分钟源范围已核过无需再核。继续只用510300/现金及现有免费数据，EPS和慢源暂停，无GPT包或额外安全审计，目标active。"]
    delivered = deliver_round(ROOT, OUT, CONFIG, ROOT / "deliverables/510300进入上下文截距_第115轮_20260908/进入上下文截距_结果及全部中文规则.md",
        "买入前上下文与周期内退出", "COMPLETED_ENTRY_CONTEXT_INTERCEPT_NO_SHARPE_GAIN_NOT_TARGET", decision, detail, next_path, next_lines,
        "SIGNED_VOLUME_WITHIN_CYCLE_INFORMATION_PREFLIGHT_PENDING", "有界核对有符号成交份额定义与等价计算，保留114平均截距")
    print(json.dumps({"交付": delivered, "核对": receipt, "收益差": differences}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
