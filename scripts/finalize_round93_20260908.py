"""核对新增价格标签、成熟训练及四个已保存账户，交付简明中文结果。"""
from bisect import bisect_right
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from research.index_prehistory_exit_v1 import ROOT, OUT as RESEARCH, CONFIG, PRIMARY, P31
from research.adaptive_allocation_v1 import normalize_dividends, summarize, affordable_quantity, fill_price
from research.learned_cycle_exit_v1 import FEATURES
from research.intraday_overnight_increment_v1 import now, require, write_json, digest
from research.simple_intraday_protection_v1 import make_rules
from scripts.review_round74_saved import saved_cycles
from scripts.finalize_round60_20260907 import metric, table

OUT = ROOT / "deliverables/510300上市前指数退出学习_第93轮_20260908"
DOCUMENT = OUT / "上市前指数退出学习_结果及中文规则.md"
INDEX = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
NEXT = ROOT / "docs/510300_AFTER_INDEX_PREHISTORY_EXIT_20260908.md"


def main():
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    require(index["latest_completed_round"]["round"] == 92 and not OUT.exists(), "第93轮前序或交付状态不符")
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    result = json.loads((RESEARCH / "result.json").read_text(encoding="utf-8"))
    require(not result["historical_point_target_met"], "若点目标达到，应另作完整验收")
    data = pd.read_parquet(ROOT / cfg["features"])
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    ip = pd.read_parquet(RESEARCH / "index_price_features.parquet")
    episodes = pd.read_parquet(RESEARCH / "index_price_episodes.parquet")
    sample = pd.read_parquet(RESEARCH / "index_price_samples.parquet")
    pool = pd.read_parquet(RESEARCH / "pooled_reference_samples.parquet")
    label_checks, fit_checks, prediction_checks, account_checks, cycles, differences, concentration = [], [], [], [], [], [], []
    for cycle in episodes.itertuples():
        rows = sample[sample.cycle_id.eq(cycle.source_cycle_id)]
        if cycle.status != "NATURALLY_CLOSED":
            require(rows.empty and pd.isna(cycle.mature_date), "未结束指数周期生成成熟训练标签")
            continue
        entry, end = int(cycle.entry_index), int(cycle.exit_index)
        require(ip.entry_condition.iloc[entry-1] and cycle.entry_date == ip.date.iloc[entry] and cycle.mature_date == ip.date.iloc[end], "指数周期进入或成熟时钟不符")
        close_path = ip.close.iloc[entry:end].to_numpy()
        peaks = np.maximum.accumulate(np.r_[ip.open.iloc[entry], close_path])[1:]
        gain, dd = close_path / ip.open.iloc[entry] - 1, close_path / peaks - 1
        natural = ip.price_exit_condition.iloc[entry:end].to_numpy() | (gain <= -.06) | (dd <= -.08) | (np.arange(1, end-entry+1) >= 60)
        require(natural[-1] and not natural[:-1].any(), "指数自然退出不是最早条件触发后的下一开盘")
        for row in rows.itertuples():
            t = int(row.origin_index)
            expected_target = ip.open.iloc[end] / ip.open.iloc[t+1] - 1
            require(entry <= t and t+1 < end and row.early_exit_date == ip.date.iloc[t+1], "指数继续持有标签提前退出时钟不符")
            expected_features = [np.log1p(t-entry+1), gain[t-entry], dd[t-entry], 1., ip.mom5.iloc[t], ip.mom20.iloc[t], ip.sma120.iloc[t], ip.vol20.iloc[t]]
            np.testing.assert_allclose([getattr(row, k) for k in FEATURES], expected_features, atol=1e-12, rtol=0)
            require(abs(expected_target-row.target) < 1e-12, "保存的指数价格标签不符")
            label_checks.append({"cycle": cycle.source_cycle_id, "origin": row.origin, "mature_date": row.mature_date, "target": expected_target})
    original = pd.read_parquet(P31 / "all_reference_samples.parquet")
    original = original[original.signal.eq("D60_INTRA")].sort_values(["cycle_id", "origin"])
    saved_etf = pool[pool.source.eq("ETF_REFERENCE")].sort_values(["source_cycle_id", "origin"])
    np.testing.assert_allclose(saved_etf[[*FEATURES, "target"]], original[[*FEATURES, "target"]], atol=0, rtol=0)
    require(saved_etf.origin.to_list() == original.origin.to_list() and saved_etf.mature_date.to_list() == original.mature_date.to_list(), "原ETF训练时间改变")
    models = json.loads((RESEARCH / "saved_models.json").read_text(encoding="utf-8"))["models"]
    membership = pd.read_parquet(RESEARCH / "training_memberships.parquet")
    fit_indices = [m["fit_index"] for m in models]
    anchor = int(np.flatnonzero(data.date.ge(cfg["earlier_start"]))[0])-1
    expected_schedule = sorted(set([anchor] + [t for t in range(anchor, len(data)-1) if data.date.iloc[t].to_period("M") != data.date.iloc[t-1].to_period("M")]))
    require(fit_indices == expected_schedule, "原拟合时钟改变")
    for stored in models:
        t = stored["fit_index"]
        mature = pool[pool.mature_date.le(data.date.iloc[t])]
        ids = mature[["cycle_id", "mature_date"]].drop_duplicates().sort_values(["mature_date", "cycle_id"]).tail(cfg["recent_cycles"]).cycle_id.to_list()
        rows = mature[mature.cycle_id.isin(ids)].sort_values(["cycle_id", "origin"])
        require(stored["training_cycles"] == ids and stored["training_rows"] == len(rows), "月度成熟周期选择不符")
        usable = len(ids) >= cfg["minimum_cycles"] and len(rows) >= cfg["minimum_rows"]
        require(usable == (stored["status"] == "FIT_COMPLETE"), "原最低训练门槛改变")
        if not usable:
            continue
        weights = 1 / rows.groupby("cycle_id").row_id.transform("count").to_numpy()
        members = membership[membership.fit_index.eq(t)].set_index("row_id").loc[rows.row_id]
        np.testing.assert_allclose(members.sample_weight, weights, atol=0, rtol=0)
        require(stored["latest_exit_date"] == str(rows.mature_date.max().date()) and stored["latest_exit_index"] <= t, "最新成熟日期不符")
        x, y = rows[FEATURES].to_numpy(float), rows.target.to_numpy(float)
        mean = np.average(x, axis=0, weights=weights)
        scale = np.sqrt(np.average((x-mean)**2, axis=0, weights=weights))
        scale[scale <= 1e-12] = 1
        design = np.c_[np.ones(len(x)), np.clip((x-mean)/scale, -cfg["feature_clip"], cfg["feature_clip"])]
        beta = np.linalg.solve(design.T @ (weights[:, None]*design) + np.diag([0.] + [cfg["ridge_alpha"]]*8), design.T @ (weights*y))
        model = stored["model"]
        np.testing.assert_allclose(model["mean"], mean, atol=1e-12, rtol=0)
        np.testing.assert_allclose(model["scale"], scale, atol=1e-12, rtol=0)
        np.testing.assert_allclose([model["intercept"], *model["coefficients"]], beta, atol=1e-10, rtol=0)
        fit_checks.append({"fit_origin": stored["fit_origin"], "cycles": len(ids), "rows": len(rows), "index_cycles": stored["index_cycles"], "etf_cycles": stored["etf_cycles"]})
    for period in ["evaluation", "earlier_diagnostic"]:
        frame = data if period == "evaluation" else data[data.date.le(cfg["earlier_terminal"])]
        start = cfg["evaluation_start"] if period == "evaluation" else cfg["earlier_start"]
        first = int(np.flatnonzero(frame.date.ge(start))[0])
        rule = make_rules(frame)["D60_INTRA"]
        for cost_id, cost in cfg["costs"].items():
            ledger = pd.read_parquet(RESEARCH / period / cost_id / f"{PRIMARY}_ledger.parquet")
            decisions = pd.read_parquet(RESEARCH / period / cost_id / f"{PRIMARY}_decisions.parquet")
            require(pd.DatetimeIndex(ledger.date).equals(pd.DatetimeIndex(frame.date.iloc[first:])), "完整日历缺失")
            require(pd.DatetimeIndex(decisions.origin).equals(pd.DatetimeIndex(frame.date.iloc[first-1:-1])), "决策日历缺失")
            require(pd.DatetimeIndex(decisions.execution_date).equals(pd.DatetimeIndex(ledger.date)), "下一开盘执行日期不符")
            previous = np.r_[cfg["initial_capital"], ledger.equity.iloc[:-1]]
            np.testing.assert_allclose(ledger.cash + ledger.shares*ledger.mark + ledger.dividend_receivable, ledger.equity, atol=1e-6, rtol=0)
            np.testing.assert_allclose(ledger.equity/previous-1, ledger.net_return, atol=1e-12, rtol=0)
            by_date = ledger.set_index("date")
            previous_cycle, negative = None, 0
            for row in decisions.itertuples():
                shares = 0 if row.origin_index == first-1 else int(by_date.loc[row.origin, "shares"])
                cash = cfg["initial_capital"] if row.origin_index == first-1 else float(by_date.loc[row.origin, "cash"])
                if row.requested_quantity > 0:
                    expected = affordable_quantity(cash, fill_price(float(frame.close.iloc[row.origin_index]), 1, cost, cfg["tick"]), cost, cfg["lot"])
                    require(not shares and expected == row.requested_quantity and row.entry_rearmed and rule["entry"][row.origin_index] == 1, "新账户没有按自身现金及重新进入条件请求买入")
                elif row.requested_quantity < 0:
                    require(row.requested_quantity == -shares, "退出请求没有对应自身全部份额")
                if pd.isna(row.learning_cycle_id):
                    continue
                if row.learning_cycle_id != previous_cycle:
                    previous_cycle, negative = row.learning_cycle_id, 0
                stored = models[bisect_right(fit_indices, int(row.origin_index))-1]
                require(stored["fit_index"] <= row.origin_index and pd.Timestamp(stored["latest_exit_date"]) <= pd.Timestamp(stored["fit_origin"]) <= row.origin, "保存持仓预测存在未来模型或周期")
                x = np.array([getattr(row, k) for k in FEATURES])
                model = stored["model"]
                estimate = model["intercept"] + np.clip((x-np.array(model["mean"]))/np.array(model["scale"]), -5, 5) @ np.array(model["coefficients"])
                require(abs(estimate-row.continuation_prediction) < 1e-12 and row.learning_fit_origin == pd.Timestamp(stored["fit_origin"]), "模型预测或使用时点不符")
                negative = negative+1 if estimate < 0 else 0
                require(negative == row.negative_confirmation_count and bool(row.learned_exit_requested) == (negative >= 2), "连续两个负预测退出规则不符")
                prediction_checks.append({"period": period, "cost": cost_id, "origin": row.origin, "fit_origin": stored["fit_origin"], "estimate": float(estimate)})
            actual, recorded = summarize(ledger, cfg), metric(result, PRIMARY, period, cost_id)
            for field in ["net_sharpe", "annualized_return", "max_drawdown", "cumulative_return", "trade_count", "commission", "slippage_cost"]:
                require(abs(actual[field]-recorded[field]) < 1e-9, "保存账户指标不符")
            account_checks.append({"period": period, "cost": cost_id, **actual})
            group = saved_cycles(ledger, dividends, cfg)
            cycles.extend({"period": period, "cost": cost_id, **c} for c in group)
            total = sum(c["net_profit"] for c in group)
            top = sorted((c["net_profit"] for c in group if c["net_profit"] > 0), reverse=True)[:3]
            concentration.append({"period": period, "cost": cost_id, "cycles": len(group), "losing_cycles": sum(c["net_profit"] < 0 for c in group), "net_profit": total, "largest_three_vs_net_profit": sum(top)/total if total > 0 else None})
            old = pd.read_parquet(RESEARCH / period / cost_id / "REARM_RIDGE_ledger.parquet")
            difference = {"period": period, "cost": cost_id, "terminal_nav_difference": float(ledger.equity.iloc[-1]-old.equity.iloc[-1]), **{f"{field}_difference": float(ledger[field].sum()-old[field].sum()) for field in ["price_pnl", "dividend_recognized", "commission", "slippage_cost"]}}
            require(abs(difference["terminal_nav_difference"]-difference["price_pnl_difference"]-difference["dividend_recognized_difference"]+difference["commission_difference"]+difference["slippage_cost_difference"]) < 1e-6, "与原策略的损益差额不守恒")
            differences.append(difference)
    for filename, rows in [("saved_index_label_checks.csv", label_checks), ("saved_monthly_model_checks.csv", fit_checks), ("saved_prediction_checks.csv", prediction_checks), ("saved_account_checks.csv", account_checks), ("saved_actual_cycles.csv", cycles), ("saved_parent_differences.csv", differences), ("saved_profit_concentration.csv", concentration)]:
        pd.DataFrame(rows).to_csv(RESEARCH / filename, index=False, encoding="utf-8-sig")
    receipt = {"verified_at": now(), "status": "INDEX_PRICE_LABELS_CAUSAL_POOLED_MODELS_AND_OWN_ACCOUNTS_CHECKED", "index_labels_checked": len(label_checks), "monthly_models_checked": len(fit_checks), "predictions_checked": len(prediction_checks), "new_account_metrics_checked": len(account_checks), "complete_evaluation_cycles": len(cycles), "new_accounts_or_model_fits": 0, "reviewer_source_sha256": digest(Path(__file__)), "security_audit_performed": False}
    write_json(RESEARCH / "saved_verification_receipt.json", receipt, exclusive=True)
    status = "COMPLETED_INDEX_PREHISTORY_EXIT_EARLIER_HISTORY_WORSE_TARGET_NOT_MET"
    decision = "新增指数样本使模型更早可用，但没有改善目标。主基础／压力夏普为0.716／0.650，较早历史为0.168／0.103；原第32轮较早基础夏普0.748，补充训练后明显下降。关闭这一单项方法，不调整指数起止日、样本权重、训练周期数或退出确认天数抢救结果。"
    write_json(RESEARCH / "acceptance_outcome.json", {"recorded_at": now(), "status": status, "decision": decision, "goal_achieved": False, "position_impact": 0}, exclusive=True)
    NEXT.write_text("\n".join(["# 第93轮之后：直接检验组合自身回撤约束", "", decision, "", "24个指数自然周期、693条新增状态，使2014年12月31日已有19个指数周期与1个ETF周期可训练；共141次拟合。较早持股收盘从原299降至58，2015年2月11日的长盈利周期被大幅截短；2015年6月3日至6月23日的亏损持仓仍存在。更早得到模型主要增加过早退出，没有识别关键亏损。", "", "下一步转向依据组合自身实际净值及历史高点逐日控制风险金额，直接用保存的原策略进入退出意向；不再补EPS、不新抓资料、不重复训练已失败退出模型。已完成对research、config、docs中CPPI、TIPP、cushion、portfolio insurance、组合保险、净值底线、高水位、ratchet、drawdown budget的有界重复核对，未发现相同实现；这不表示方法已经登记或会有效。", "", "研究只允许510300与现金，沿用完整交易日、实际账户成本和下一开盘成交约束。正式冻结前需明确：参考峰值来自新账户自身，保护比例与倍数固定且不得事后调参；底线是控制目标，跳空、费用或不能成交仍可能突破。原91主基础1.233局部候选及92局部收益改善均保留，完整目标未完成。", ""]), encoding="utf-8")
    OUT.mkdir(parents=True)
    lines = ["# 第93轮：上市前指数经验能否改善退出", "", decision, "", "## 主历史：2020年1月2日至2026年8月14日开盘", "", *table(result["all_metrics"]), "## 较早历史：2015年1月5日至2019年12月31日开盘", "", *table(result["earlier_diagnostics"]), "## 老板需要了解的结果", "", "这次实际补入24个上市前指数自然周期、693条继续持有状态；另有一个未结束周期被剔除。原ETF参考34个周期、1461条状态保持不变。首个可用模型由2017年3月1日提前至2014年12月31日，141个月度时点均完成训练。训练数据更多、可用更早，并不等于交易更有效。", "", "较早历史实际持股收盘从299天降为58天。2015年2月11日开始的原持仓持续60个收盘，到5月15日退出，赚约6.85万元；新模型在2月25日就退出，只赚约6105元。账户后来买卖与本金路径不同，这个例子用于解释行为差异，不能把这一个周期的利润差额当作独立因果贡献。", "", "与此同时，2015年6月3日至6月23日的亏损持仓，两种方案都没有避开。因此更早启用模型主要截断了一些盈利持仓，未能同步消除关键亏损。较早历史2015年全年净损益从原约6.45万元降至约1250元，整个2015—2019年年化从8.64%降至0.83%。主历史年化5.31%、最大回撤10.59%，夏普仍不足1.2。", "", "新增指数只含价格，原ETF标签含费用和分红，另有时期和市场结构差异；本次结果不能单独识别哪一种差异造成失败。结束这项补充训练方法，下一项研究直接检验组合自身回撤与风险金额的关系。", "", "## 全部中文因子和进入退出规则", "", *(ROOT / cfg["rules"]).read_text(encoding="utf-8").splitlines()[1:], "", f"六项必要测试通过；复核{len(label_checks)}条指数标签、{len(fit_checks)}个月度模型的成熟样本及加权正规方程、{len(prediction_checks)}次保存预测、四个新账户自身请求及{len(cycles)}个实际完整周期。未重新拟合或重跑已保存账户。逐月实际系数见同目录《每月八项模型中文规则.md》。", ""]
    DOCUMENT.write_text("\n".join(lines), encoding="utf-8")
    for path in RESEARCH.iterdir():
        if path.is_file() and path.suffix in {".csv", ".json", ".md"}:
            shutil.copy2(path, OUT / path.name)
    shutil.copy2(CONFIG, OUT / "冻结设置.json")
    shutil.copy2(NEXT, OUT / "下一项研究方向.md")
    for period, label in [("evaluation", "主评价"), ("earlier_diagnostic", "较早历史")]:
        for cost in cfg["costs"]:
            for kind in ["ledger", "decisions"]:
                pd.read_parquet(RESEARCH / period / cost / f"{PRIMARY}_{kind}.parquet").to_csv(OUT / f"{label}_{cost}_{kind}.csv", index=False, encoding="utf-8-sig")
    record = {"round": 93, "study": result["study_id"], "title": "上市前指数周期补充的因果退出学习", "status": status, "result": str((RESEARCH / "result.json").relative_to(ROOT)), **{k: result[k] for k in ["candidate_configurations", "evaluation_accounts", "new_accounts_generated", "reused_control_accounts", "earlier_diagnostic_accounts", "new_earlier_diagnostic_accounts", "new_model_fits", "new_reference_accounts"]}, "evaluated_candidate_source_runs": 1, "primary_base": metric(result, PRIMARY), "primary_stress": metric(result, PRIMARY, cost="STRESS"), "post_selected_best_base": metric(result, PRIMARY)}
    index["completed_rounds"].append(record)
    for key in ["registered_configurations_in_this_resumption", "evaluated_configurations_in_this_resumption", "evaluated_candidate_source_runs_including_corrected_replays", "registered_candidate_source_runs_including_unrun_legacy_bindings"]:
        index[key] += 1
    index["evaluation_accounts_in_this_resumption"] += result["evaluation_accounts"]
    index.update(updated_at=now(), latest_completed_round=record, running_studies=[], goal_achieved=False, status="ROUND93_EARLIER_HISTORY_WORSE_FULL_GOAL_NOT_MET", count_warning="93轮，367不同设置，383已评价来源版本，388登记含5旧未运行，1376主评价记录；无效来源保留。", next_work={"status": "OWN_ACCOUNT_DRAWDOWN_RISK_METHOD_NOT_REGISTERED", "focus": "直接检验新账户自身回撤保护及原进入退出意向", "source": str(NEXT.relative_to(ROOT))}, process_state_note="93模型、四账户、必要核对和中文交付完成；早期补充训练不能解决目标，91主基础1.233局部候选继续保留。")
    index["deliveries"].append({"created_at": now(), "type": "INDEX_PREHISTORY_EXIT_ROUND93_FAST_CHINESE_RESULTS", "rounds": [93], "directory": str(OUT), "main_document": str(DOCUMENT), "new_gpt_review_archive_created": False})
    write_json(INDEX, index)
    write_json(OUT / "交付回执.json", {"created_at": now(), "main_document": str(DOCUMENT), "goal_achieved": False, "new_gpt_review_archive_created": False}, exclusive=True)
    print(json.dumps({"交付": str(DOCUMENT), "必要核对": receipt, "与原32差额": differences, "利润集中": concentration}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
