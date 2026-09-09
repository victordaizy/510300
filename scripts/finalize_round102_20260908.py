"""经济权重和实际账户关键核对，简洁交付，不重拟合。"""
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from research.profit_drawdown_interaction_v1 import ROOT, OUT as RESEARCH, CONFIG, PRIMARY
from research.learned_cycle_exit_v1 import training_rows
from research.profit_drawdown_interaction_inputs_v1 import FEATURES, INTERACTION
from research.adaptive_allocation_v1 import normalize_dividends, summarize
from research.intraday_overnight_increment_v1 import now, require, write_json, digest
from scripts.review_round74_saved import saved_cycles
from scripts.finalize_round60_20260907 import metric
from scripts.finalize_round96_20260908 import table

OUT = ROOT / "deliverables/510300浮盈回撤交互退出_第102轮_20260908"
DOCUMENT = OUT / "浮盈回撤交互退出_结果及全部中文规则.md"
INDEX = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
NEXT = ROOT / "docs/510300_AFTER_PROFIT_DRAWDOWN_INTERACTION_20260908.md"


def main():
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    require(index["latest_completed_round"]["round"] == 101 and not OUT.exists(), "102前序或交付状态不符")
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    result = json.loads((RESEARCH / "result.json").read_text(encoding="utf-8"))
    require(not result["historical_point_target_met"], "出现目标值时不能直接写成未达标")
    models = json.loads((RESEARCH / "saved_models.json").read_text(encoding="utf-8"))["models"]
    by_date = {pd.Timestamp(m["fit_origin"]): m for m in models}
    members = pd.read_parquet(RESEARCH / "training_memberships.parquet")
    samples = pd.read_parquet(ROOT / cfg["samples"])
    samples = samples[samples.signal.eq("D60_INTRA")]
    data = pd.read_parquet(ROOT / cfg["features"])
    samples = samples.copy()
    samples[INTERACTION] = [max(float(p), 0.)*float(d) if np.isfinite(p) and np.isfinite(d) else np.nan for p, d in zip(samples.cycle_return, samples.cycle_drawdown)]
    saved_samples = pd.read_parquet(RESEARCH / "extended_reference_samples.parquet")
    pd.testing.assert_frame_equal(samples.reset_index(drop=True), saved_samples)
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
                        data.mom5.iloc[row.origin_index],data.mom20.iloc[row.origin_index],data.sma120.iloc[row.origin_index],data.vol20.iloc[row.origin_index]]
                    expected_state.append(max(expected_state[1], 0.)*expected_state[2])
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
    receipt = {"verified_at": now(), "status": "SAVED_INTERACTION_NINE_FEATURES_RIDGE_AND_ACCOUNT_CHECKS_COMPLETE", "mature_models": len(weight_checks), "reference_interaction_rows": len(samples), "actual_holding_state_rows": sum(r["holding_decisions"] for r in result["model_coverage"]),
        "saved_scores_recomputed": len(predictions), "new_account_metrics_recomputed": len(checked), "complete_cycles": len(cycles),
        "new_accounts_or_models": 0, "reviewer_source_sha256": digest(Path(__file__)), "security_audit_performed": False}
    write_json(RESEARCH / "saved_verification_receipt.json", receipt, exclusive=True)
    status = "COMPLETED_INTERACTION_EARLY_LOCAL_IMPROVEMENT_MAIN_WORSE_NOT_TARGET"
    decision = "第102轮未达标。主基础／压力夏普0.575／0.511，较早0.781／0.758；相比原八项退出，主历史下降、较早略升。较早主要由2019年2月28日进入的一笔交易晚退出一天改善，不能视为稳定增量。保留这一固定模型供预定组合检验，不更换交互、阈值或训练规则。"
    write_json(RESEARCH / "acceptance_outcome.json", {"status": status, "recorded_at": now(), "goal_achieved": False, "decision": decision, "independent_validation": "NOT_ESTABLISHED", "position_impact": 0}, exclusive=True)
    paired = []
    for period in ["evaluation", "earlier_diagnostic"]:
        for cost in cfg["costs"]:
            new_ledger = pd.read_parquet(RESEARCH / period / cost / f"{PRIMARY}_ledger.parquet")
            old_ledger = pd.read_parquet(RESEARCH / period / cost / "REARM_RIDGE_ledger.parquet")
            left, right = [pd.DataFrame(saved_cycles(l, dividends, cfg)) for l in [new_ledger, old_ledger]]
            comparison = left.merge(right, on="entry_date", how="outer", suffixes=("_new", "_old"), indicator=True)
            comparison["period"], comparison["cost"] = period, cost
            paired.append(comparison)
    pd.concat(paired, ignore_index=True).to_csv(RESEARCH / "按实际进入日对齐的周期差异.csv", index=False, encoding="utf-8-sig")
    NEXT.write_text("\n".join(["# 第102轮之后：固定新退出模型在原风险预算中的效果", "", decision, "",
        "102六必要测试4.68秒，114拟合加四账户约9.91秒，27原时点无足够样本、无求解失败，零新参考账户。主24周期48成交255持股收盘，早9周期18成交300收盘，早204行原无模型。原1461样本34自然周期、训练成员和九项权重均已核对。", "",
        "相对旧32主基础净少18104.92822元、压力少17509.83836元；早基础多6370.29306元、压力多6291.01384元。早九周期进入日全相同，只有2019-02-28进入的周期从03-11改为03-12退出，其他资金差异还会影响后续数量，不能将全部净值差归为孤立的一天。主23个相同进入、各一个不同进入；2025-06和2026多笔退出变化，不能事后只保留有利年份。", "",
        "下一103预定一个组合检验：沿用第91轮2013-06-03连续参考起点、242完整日月首最小方差、零波动无观点保留、起始各半、10个百分点交易带宽和两段新资金账户；仅将连续学习参考的原八项模型替换为第102轮已保存九项模型。原急跌连续参考直接复用，只建立一个新的连续学习参考及四个评价账户，不重训。两条原进入与自然退出完全不变。新连续参考也只按当日及以前最近保存成熟模型调用，首次仍2017-03-01，此前原价格退出。", "",
        "参考收益只作当时风险估计，不能把两条净收益加权当新账户。一个新父模型改变路径和风险历史，必须分别保存新参考、月度预算、实际进出与旧91同历比较，不能把结果归为单一仓位或单笔胜负。预先只这一固定组合，无额外父策略、预算窗口、权重界限或费用调参。局部单笔改善很弱，本轮组合仍是历史研究而非验证。", "",
        "直接复用research/continuous_reference_min_variance_v1.py原参考状态转换、research/two_policy_min_variance_inputs_v1.py预算和research/event_clock_account_v1.py真实账户。保存新模型路径reports/research/510300_profit_drawdown_interaction_v1/saved_models.json内models是列表；原31文件的models则是按策略键映射，不可混淆。当前103只有方向、尚无配置、参考账户或收益。继续加快，不补慢来源、GPT包或长模型报告。", ""]), encoding="utf-8")
    OUT.mkdir(parents=True)
    lines = ["# 第102轮：原退出模型加入浮盈与回撤交互", "", decision, "", "## 主历史：2020年1月2日至2026年8月14日开盘", "", *table(result["all_metrics"]), "## 较早历史：2015年1月5日至2019年12月31日开盘", "", *table(result["earlier_diagnostics"]),
        "## 改善来自哪里", "", "主基础年化4.34%、最大回撤10.59%，较早基础年化9.09%、回撤13.79%。相比原八项退出，主夏普由0.705降到0.575；较早由0.748升到0.781。较早相同进入的九个周期中，仅2019年2月28日进入的周期改变退出日期，从3月11日改为3月12日。这不是跨多个交易重复出现的改善证据。", "",
        "完整账户相对原八项退出，主基础净少18104.93元、较早多6370.29元。后续现金及份额会继承前期差异，周期配对表保留这些账户影响，没有把不同路径拼接。主23个进入日相同，另有旧5月13日与新5月14日进入的差异周期。", "",
        f"六项必要测试通过；114个成熟模型的完整周期权重、加权岭最优条件、{len(predictions)}条实际预测、四账户及{len(cycles)}个完整周期已核。全部九项因子、月度标准化和系数保存在同目录表中。下一仅按预定步骤检验固定模型在原91风险预算中的效果。", "", *(ROOT / cfg["rules"]).read_text(encoding="utf-8").splitlines()[1:]]
    DOCUMENT.write_text("\n".join(lines)+"\n", encoding="utf-8")
    for p in RESEARCH.iterdir():
        if p.is_file() and p.suffix in {".csv", ".json"}:
            shutil.copy2(p, OUT / p.name)
    shutil.copy2(CONFIG, OUT / "冻结设置.json")
    shutil.copy2(NEXT, OUT / "下一项研究方向.md")
    for period, label in [("evaluation", "主历史"), ("earlier_diagnostic", "较早历史")]:
        for cost in cfg["costs"]:
            for kind in ["ledger", "decisions"]:
                pd.read_parquet(RESEARCH / period / cost / f"{PRIMARY}_{kind}.parquet").to_csv(OUT / f"{label}_{cost}_{kind}.csv", index=False, encoding="utf-8-sig")
    record = {"round": 102, "study": result["study_id"], "title": "原八项退出增加浮盈正部与周期回撤交互", "status": status, "result": str((RESEARCH / "result.json").relative_to(ROOT)), **{k: result[k] for k in ["candidate_configurations", "evaluation_accounts", "new_accounts_generated", "reused_control_accounts", "earlier_diagnostic_accounts", "new_earlier_diagnostic_accounts", "new_model_fits", "new_reference_accounts"]}, "evaluated_candidate_source_runs": 1, "primary_base": metric(result, PRIMARY), "primary_stress": metric(result, PRIMARY, cost="STRESS"), "post_selected_best_base": metric(result, PRIMARY)}
    index["completed_rounds"].append(record)
    for key in ["registered_configurations_in_this_resumption", "evaluated_configurations_in_this_resumption", "evaluated_candidate_source_runs_including_corrected_replays", "registered_candidate_source_runs_including_unrun_legacy_bindings"]:
        index[key] += 1
    index["evaluation_accounts_in_this_resumption"] += result["evaluation_accounts"]
    index.update(updated_at=now(), latest_completed_round=record, running_studies=[], goal_achieved=False, status="ROUND102_INTERACTION_EARLY_LOCAL_ONLY_FULL_GOAL_NOT_MET", count_warning="102轮，378不同设置，394已评价来源版本，399登记含5旧未运行，1462主评价记录。", next_work={"status": "CONTINUOUS_INTERACTION_BUDGET_DIRECTION_NOT_REGISTERED", "focus": "固定第102轮模型替换91连续学习参考，再以原最小方差规则检验实际组合", "source": str(NEXT.relative_to(ROOT))}, process_state_note="102六测试、114拟合、四账户及保存核对交付完成；103尚未登记或运行。")
    index["deliveries"].append({"created_at": now(), "type": "PROFIT_DRAWDOWN_INTERACTION_ROUND102_FAST_CHINESE_RESULTS", "rounds": [102], "directory": str(OUT), "main_document": str(DOCUMENT), "new_gpt_review_archive_created": False})
    write_json(INDEX, index)
    write_json(OUT / "交付回执.json", {"created_at": now(), "main_document": str(DOCUMENT), "goal_achieved": False, "new_gpt_review_archive_created": False}, exclusive=True)
    print(json.dumps({"交付": str(DOCUMENT), "核对": receipt}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
