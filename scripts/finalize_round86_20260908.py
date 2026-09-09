"""经济权重和实际账户关键核对，简洁交付，不重拟合。"""
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from research.liquidity_increment_exit_v1 import ROOT, OUT as RESEARCH, CONFIG, PRIMARY
from research.learned_cycle_exit_v1 import training_rows
from research.liquidity_increment_exit_inputs_v1 import FEATURES, LIQUIDITY
from research.adaptive_allocation_v1 import normalize_dividends, summarize
from research.intraday_overnight_increment_v1 import now, require, write_json, digest
from scripts.review_round74_saved import saved_cycles
from scripts.finalize_round60_20260907 import metric, table

OUT = ROOT / "deliverables/510300流动性增量退出_第86轮_20260908"
DOCUMENT = OUT / "流动性增量退出_结果及全部中文规则.md"
INDEX = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
NEXT = ROOT / "docs/510300_AFTER_LIQUIDITY_KERNEL_EXIT_20260908.md"


def main():
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    require(index["latest_completed_round"]["round"] == 85 and not OUT.exists(), "86前序或交付状态不符")
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    result = json.loads((RESEARCH / "result.json").read_text(encoding="utf-8"))
    require(not result["historical_point_target_met"], "出现目标值时不能直接写成未达标")
    models = json.loads((RESEARCH / "saved_models.json").read_text(encoding="utf-8"))["models"]
    by_date = {pd.Timestamp(m["fit_origin"]): m for m in models}
    members = pd.read_parquet(RESEARCH / "training_memberships.parquet")
    samples = pd.read_parquet(ROOT / cfg["samples"])
    samples = samples[samples.signal.eq("D60_INTRA")]
    data = pd.read_parquet(ROOT / cfg["features"])
    factors = pd.read_parquet(RESEARCH / "liquidity_factors.parquet")
    expected = np.full(len(data), np.nan)
    for t in range(19, len(data)):
        amount = data.amount.iloc[t-19:t+1].to_numpy(float)
        returns = data.total_simple.iloc[t-19:t+1].to_numpy(float)
        if np.isfinite(amount).all() and (amount > 0).all() and np.isfinite(returns).all():
            average = np.mean(np.abs(returns)*1e8/amount)
            if average > 0:
                expected[t] = np.log(average)
    np.testing.assert_allclose(factors[LIQUIDITY], expected, atol=1e-12, rtol=0, equal_nan=True)
    samples = samples.copy()
    samples[LIQUIDITY] = expected[samples.origin_index.to_numpy(int)]
    old_models = json.loads((ROOT / cfg["saved_models"]).read_text(encoding="utf-8"))["models"]["D60_INTRA__RIDGE"]
    old_by_date = {m["fit_index"]: m for m in old_models}
    weight_checks = []
    for record in [m for m in models if m["status"] == "FIT_COMPLETE"]:
        t = record["fit_index"]
        rows, ids = training_rows(samples, t, cfg)
        require(ids == record["training_cycles"] and (rows.exit_index <= t).all(), "九因子使用未来或不同周期")
        saved = members[members.fit_index.eq(t)].sort_values(["cycle_id", "origin_index"])
        np.testing.assert_allclose(rows.sample_weight, saved.sample_weight, atol=0, rtol=0)
        np.testing.assert_allclose(saved.groupby("cycle_id").sample_weight.sum(), 1., atol=1e-12, rtol=0)
        model = record["model"]
        np.testing.assert_allclose(model["mean"][:8], old_by_date[t]["model"]["mean"], atol=1e-12, rtol=0)
        np.testing.assert_allclose(model["scale"][:8], old_by_date[t]["model"]["scale"], atol=1e-12, rtol=0)
        x=rows[FEATURES].to_numpy(float); w=rows.sample_weight.to_numpy(float)
        mean=np.average(x,axis=0,weights=w);scale=np.sqrt(np.average((x-mean)**2,axis=0,weights=w));scale=np.where(scale>1e-12,scale,1.)
        np.testing.assert_allclose(model["mean"],mean,atol=1e-12,rtol=0);np.testing.assert_allclose(model["scale"],scale,atol=1e-12,rtol=0)
        z=np.clip((x-mean)/scale,-5,5);coef=np.asarray(model["coefficients"]); residual=model["intercept"]+z@coef-rows.target.to_numpy()
        gradient=z.T@(w*residual)+cfg["ridge_alpha"]*coef
        require(abs(np.sum(w*residual))<1e-10 and np.max(np.abs(gradient))<1e-10,"九因子系数不满足冻结加权岭最优条件")
        weight_checks.append({"fit_index":t,"fit_origin":record["fit_origin"],"cycles":len(ids),"rows":len(rows),"missing_features":record["missing_feature_rows"],"largest_gradient":np.max(np.abs(gradient))})
    require(len(weight_checks)==114 and result["failed_fits"]==0,"九因子训练或求解状态不符")
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    cycles, checked, differences, predictions = [], [], [], []
    for period in ["evaluation", "earlier_diagnostic"]:
        for cost in cfg["costs"]:
            folder = RESEARCH / period / cost
            ledger = pd.read_parquet(folder / f"{PRIMARY}_ledger.parquet")
            decisions = pd.read_parquet(folder / f"{PRIMARY}_decisions.parquet")
            native=pd.read_csv(folder/f"{PRIMARY}_cycles.csv")
            for cycle_id, group in decisions[decisions.learning_cycle_id.notna()].groupby("learning_cycle_id", sort=False):
                count=0
                cycle=native[native.cycle_id.eq(cycle_id)].iloc[0]
                owned=ledger[ledger.cycle_id.eq(cycle_id)].copy().set_index("date")
                values=owned.shares*owned.mark+owned.dividend_recognized.cumsum()
                peaks=np.maximum.accumulate(np.r_[cycle.entry_cost_cny, values.to_numpy()])[1:]
                peak_by_date=dict(zip(owned.index,peaks))
                for row in group.itertuples():
                    actual=values.loc[row.origin]
                    expected_state=[np.log1p(row.origin_index-cycle.entry_index+1),actual/cycle.entry_cost_cny-1,actual/peak_by_date[row.origin]-1,cycle["mode"],
                        data.mom5.iloc[row.origin_index],data.mom20.iloc[row.origin_index],data.sma120.iloc[row.origin_index],data.vol20.iloc[row.origin_index],expected[row.origin_index]]
                    x=np.array([getattr(row,c) for c in FEATURES]);np.testing.assert_allclose(x,expected_state,atol=1e-12,rtol=0)
                    if row.learning_status=="PREDICTION_AVAILABLE":
                        m=by_date[row.learning_fit_origin];require(m["latest_exit_index"]<=m["fit_index"]<=row.origin_index,"九因子实际调用未来模型")
                        z=np.clip((x-m["model"]["mean"])/m["model"]["scale"],-5,5)
                        score=float(m["model"]["intercept"]+z@np.array(m["model"]["coefficients"]))
                        require(abs(score-row.continuation_prediction)<1e-12,"九因子预测不能按系数复算")
                        count=count+1 if score<0 else 0
                        predictions.append({"period":period,"cost":cost,"origin":row.origin,"cycle":cycle_id,"continuation_prediction":score})
                    else:
                        require(pd.isna(row.continuation_prediction),"无模型时仍填预测")
                        count=0
                    require(count==row.negative_confirmation_count and bool(row.learned_exit_requested)==(count>=2),"九因子确认或周期重置不符")
            m = metric(result, PRIMARY, period, cost)
            previous = np.r_[cfg["initial_capital"], ledger.equity.iloc[:-1].to_numpy()]
            np.testing.assert_allclose(ledger.cash+ledger.shares*ledger.mark+ledger.dividend_receivable, ledger.equity, atol=1e-6, rtol=0)
            np.testing.assert_allclose(ledger.equity/previous-1, ledger.net_return, atol=1e-12, rtol=0)
            recomputed = summarize(ledger, cfg)
            for field in ["net_sharpe", "annualized_return", "max_drawdown", "cumulative_return", "trade_count", "commission", "slippage_cost"]:
                require(abs(recomputed[field]-m[field]) < 1e-9, "经济退出保存指标不符")
            cycles.extend({"period": period, "cost": cost, **c} for c in saved_cycles(ledger, dividends, cfg))
            checked.append({"period": period, "cost": cost, "days": len(ledger), "net_sharpe": m["net_sharpe"]})
            for control_name in ["REARM_RIDGE"]:
                control = pd.read_parquet(folder / f"{control_name}_ledger.parquet")
                differences.append({"period": period, "cost": cost, "control": control_name,
                    "terminal_nav_difference": float(ledger.equity.iloc[-1]-control.equity.iloc[-1]),
                    **{f"{c}_difference": float(ledger[c].sum()-control[c].sum()) for c in ["price_pnl", "dividend_recognized", "commission", "slippage_cost"]}})
    for name, rows in [("saved_actual_cycles.csv", cycles), ("saved_prediction_checks.csv", predictions), ("saved_profit_differences.csv", differences), ("saved_training_checks.csv", weight_checks)]:
        pd.DataFrame(rows).to_csv(RESEARCH / name, index=False, encoding="utf-8-sig")
    receipt = {"verified_at": now(), "status": "KEY_PAST_LIQUIDITY_NINE_FEATURES_RIDGE_AND_ACCOUNT_CHECKS_COMPLETE", "mature_models": len(weight_checks), "daily_liquidity_rows": len(data), "actual_holding_state_rows": sum(r["holding_decisions"] for r in result["model_coverage"]),
        "saved_scores_recomputed": len(predictions), "new_account_metrics_recomputed": len(checked), "complete_cycles": len(cycles),
        "new_accounts_or_models": 0, "reviewer_source_sha256": digest(Path(__file__)), "security_audit_performed": False}
    write_json(RESEARCH / "saved_verification_receipt.json", receipt, exclusive=True)
    status = "COMPLETED_LIQUIDITY_INCREMENT_TARGET_NOT_MET"
    write_json(RESEARCH / "acceptance_outcome.json", {"status": status, "recorded_at": now(), "goal_achieved": False,
        "decision": "新增流动性后四情景均低于原八因子退出，关闭本九项版本，不变换窗口、方向或惩罚挽救。", "position_impact": 0}, exclusive=True)
    OUT.mkdir(parents=True)
    lines = ["# 第86轮：退出模型加入流动性因子的结果", "", "主基础／压力净夏普0.0155／负0.0378，较早0.7390／0.7155，四种情景均低于原八因子退出。流动性指标没有产生可保留的局部改善，结束此版本。", "",
        "## 主评价：2020年1月2日至2026年8月14日开盘", ""] + table(result["all_metrics"])
    lines += ["## 较早历史：2015年1月5日至2019年12月31日开盘", ""] + table(result["earlier_diagnostics"])
    lines += ["114次九项模型全部拟合成功；27个早期检查没有足够成熟样本。1461个原参考状态均有新增流动性输入，没有删除训练行或改变周期权重。主每档22个完整周期、44笔成交、289个持仓收盘；16次学习退出。较早9周期、18笔成交、270个持仓收盘，2次学习退出；其中204条早期状态按原成熟门槛无法使用模型，继续按价格和时间退出。", "",
        "主基础年化负0.24%，最大回撤22.12%，明显弱于原八因子。较早年化8.38%、回撤13.79%，也没有增量改善。该代理只反映绝对价格变化与成交额比例，不能解释成公募申购、净流出或实际成交冲击。下一项只改变原八项模型的非线性函数形式，当前尚未登记。", "",
        f"七项必要测试通过；核对{len(data)}个日因子、114个成熟训练月的完整周期权重和加权岭最优条件、{len(predictions)}条预测及全部实际持仓状态、四账户指标、{len(cycles)}个含分红周期。全部月度九项系数见同目录“每月九项流动性退出模型中文规则.md”。", "", "## 全部中文规则", ""]
    lines += (ROOT / cfg["rules"]).read_text(encoding="utf-8").splitlines()[1:]
    DOCUMENT.write_text("\n".join(lines)+"\n", encoding="utf-8")
    for p in RESEARCH.iterdir():
        if p.is_file() and p.suffix in {".csv", ".json", ".md"}:
            shutil.copy2(p, OUT / p.name)
    shutil.copy2(CONFIG, OUT / "冻结设置.json")
    shutil.copy2(NEXT, OUT / "下一项相似状态退出方向.md")
    for period, label in [("evaluation", "主评价"), ("earlier_diagnostic", "较早历史")]:
        for cost in cfg["costs"]:
            for kind in ["ledger", "decisions"]:
                pd.read_parquet(RESEARCH / period / cost / f"{PRIMARY}_{kind}.parquet").to_csv(OUT / f"{label}_{cost}_{'完整账户' if kind=='ledger' else '全部因子与进出场'}.csv", index=False, encoding="utf-8-sig")
    record = {"round": 86, "study": result["study_id"], "title": "原八项退出增加过去流动性代理", "status": status,
        "result": str((RESEARCH / "result.json").relative_to(ROOT)), **{k: result[k] for k in ["candidate_configurations", "evaluation_accounts", "new_accounts_generated", "reused_control_accounts", "earlier_diagnostic_accounts", "new_earlier_diagnostic_accounts", "new_model_fits", "new_reference_accounts"]},
        "evaluated_candidate_source_runs": 1, "primary_base": metric(result, PRIMARY), "primary_stress": metric(result, PRIMARY, cost="STRESS"), "post_selected_best_base": metric(result, PRIMARY)}
    index["completed_rounds"].append(record)
    for k in ["registered_configurations_in_this_resumption", "evaluated_configurations_in_this_resumption", "evaluated_candidate_source_runs_including_corrected_replays", "registered_candidate_source_runs_including_unrun_legacy_bindings"]:
        index[k] += 1
    index["evaluation_accounts_in_this_resumption"] += result["evaluation_accounts"]
    index.update(updated_at=now(), latest_completed_round=record, running_studies=[], goal_achieved=False, status="ROUND86_COMPLETE_LIQUIDITY_INCREMENT_TARGET_NOT_MET",
        count_warning="86轮，360不同设置，376已评价来源版本，381登记含5旧未运行，1308主评价记录。",
        next_work={"status": "KERNEL_CONTINUATION_EXIT_DIRECTION_NOT_REGISTERED", "focus": "按原八项相似持仓状态进行核岭继续收益退出", "source": str(NEXT.relative_to(ROOT))},
        process_state_note="86九项重训、四账户、关键核对及简洁交付完成；87非线性相似状态退出仅方向。",
        research_speed_priority="docs/510300_FAST_RESEARCH_CADENCE_20260908.md")
    index["deliveries"].append({"created_at": now(), "type": "LIQUIDITY_INCREMENT_ROUND86_FAST_CHINESE_RESULTS", "rounds": [86], "directory": str(OUT), "main_document": str(DOCUMENT), "new_gpt_review_archive_created": False})
    write_json(INDEX, index)
    write_json(OUT / "交付回执.json", {"created_at": now(), "main_document": str(DOCUMENT), "goal_achieved": False, "new_gpt_review_archive_created": False}, exclusive=True)
    print(json.dumps({"交付": str(DOCUMENT), "核对": receipt}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
