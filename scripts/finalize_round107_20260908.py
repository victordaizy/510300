"""经济权重和实际账户关键核对，简洁交付，不重拟合。"""
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from research.overnight_downside_exit_v1 import ROOT, OUT as RESEARCH, CONFIG, PRIMARY
from research.learned_cycle_exit_v1 import training_rows
from research.overnight_downside_exit_inputs_v1 import FEATURES, ADDED
from research.adaptive_allocation_v1 import normalize_dividends, summarize
from research.intraday_overnight_increment_v1 import now, require, write_json, digest
from scripts.review_round74_saved import saved_cycles
from scripts.finalize_round60_20260907 import metric
from scripts.finalize_round96_20260908 import table

OUT = ROOT / "deliverables/510300隔夜下行风险占比退出_第107轮_20260908"
DOCUMENT = OUT / "隔夜下行风险占比退出_结果及全部中文规则.md"
INDEX = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
NEXT = ROOT / "docs/510300_AFTER_OVERNIGHT_DOWNSIDE_EXIT_20260908.md"


def main():
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    require(index["latest_completed_round"]["round"] == 106 and not OUT.exists(), "107前序或交付状态不符")
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
    factor_values = np.full(len(data), np.nan)
    for t in range(19, len(data)):
        night = data.overnight_log.iloc[t-19:t+1].to_numpy(float)
        day = data.intraday_log.iloc[t-19:t+1].to_numpy(float)
        if np.isfinite(night).all() and np.isfinite(day).all():
            denominator = float(np.sum(night*night+day*day))
            if denominator > 0:
                factor_values[t] = float(np.sum(np.minimum(night, 0.)**2)/denominator)
    saved_factors = pd.read_parquet(RESEARCH / "overnight_downside_factors.parquet")
    np.testing.assert_allclose(saved_factors[ADDED], factor_values, atol=1e-12, rtol=0, equal_nan=True)
    samples[ADDED] = factor_values[samples.origin_index.to_numpy(int)]
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
                    expected_state.append(factor_values[row.origin_index])
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
    receipt = {"verified_at": now(), "status": "SAVED_OVERNIGHT_DOWNSIDE_RIDGE_AND_ACCOUNT_CHECKS_COMPLETE", "mature_models": len(weight_checks), "reference_added_factor_rows": len(samples), "actual_holding_state_rows": sum(r["holding_decisions"] for r in result["model_coverage"]),
        "saved_scores_recomputed": len(predictions), "new_account_metrics_recomputed": len(checked), "complete_cycles": len(cycles),
        "new_accounts_or_models": 0, "reviewer_source_sha256": digest(Path(__file__)), "security_audit_performed": False}
    write_json(RESEARCH / "saved_verification_receipt.json", receipt, exclusive=True)
    status = "COMPLETED_OVERNIGHT_DOWNSIDE_INFORMATION_NO_IMPROVEMENT_NOT_TARGET"
    decision = "第107轮未达标。主基础／压力夏普0.354／0.304，较早0.738／0.715，均低于原八项线性退出的0.705／0.641和0.748／0.725。主基础年化2.44%、最大回撤14.22%；较早年化8.48%、回撤13.79%。加入隔夜下行风险占比没有形成改善，关闭此固定设置，不改变分子分母、窗口、学习惩罚或退出阈值救回。"
    write_json(RESEARCH / "acceptance_outcome.json", {"recorded_at": now(), "status": status, "decision": decision, "goal_achieved": False, "position_impact": 0}, exclusive=True)
    blocked = []
    for cost in cfg["costs"]:
        l = pd.read_parquet(RESEARCH / "evaluation" / cost / f"{PRIMARY}_ledger.parquet")
        failed = l[l.requested_quantity.ne(0) & l.filled_quantity.eq(0)]
        require(len(failed) == 1 and str(failed.date.iloc[0].date()) == "2024-10-08" and failed.status.iloc[0] == "UNFILLED_DIRECTIONAL_LIMIT", "本轮未成交日期或原因不符")
        blocked.extend({"cost": cost, "date": str(r.date.date()), "origin": str(r.origin.date()), "status": r.status, "requested_quantity": int(r.requested_quantity)} for r in failed.itertuples())
    write_json(RESEARCH / "unfilled_request_explanation.json", {"records": blocked, "explanation": "2024年9月30日收盘生成的买入请求，在10月8日开盘因方向涨停规则未成交；完整保留，无人工假设成交。"}, exclusive=True)
    NEXT.write_text("\n".join(["# 第107轮之后：三类自然参考的共享训练", "", decision, "",
        "107六必要测试4.72秒，114拟合及四账户约5.27秒；27原无模型，零拟合失败，零新参考。主24周期48成交，基础260持股收盘、压力251；早9周期18成交300收盘、96预测204原无模型。两主账户各一个2024-10-08涨停未成交，记录原9月30日请求。1111真实持仓状态、703预测、114加权岭及66周期已核，不重跑。", "",
        "106固定提升退出主S0.081/0.003、早0.555/0.533，114训练加四账户约65.72秒已关闭；同八因子增加复杂度无效。第81轮先半仓再确认已经做过，不改成等待天数或换父策略救回。第7轮全期限逆回购是已披露信息量，实际月度操作日及到期缺证，不能凭空推净投放或余额；继续暂停慢源。", "",
        "下一108考察现有三类自然参考的共享继续价值：原D60日内隔夜34周期1461状态，原趋势反弹S1有90周期681状态，原偏离均值R2有44周期316状态；共2458状态168自然周期。三者仍只交易510300，不增加证券或来源。旧31是各类分开训练；本轮有界代码/规则检索未发现这三种参考在共同时间段去重后共享条件继续价值的实现，原指数前史补充93不是同一来源，但其失败保持。", "",
        "重叠行情不能假装多个独立样本。拟按三参考实际收盘持仓并集形成共同持仓时段：任一参考持仓即仍在同一时段，只有三者全部实际空仓才结束；终点强平不作自然成熟。一个时段完全结束之后，其所有原周期及状态才可用于学习。时段之间不重叠，不宣称统计独立。", "",
        "19:21:46只读支持核对已经完成，见reports/research/510300_shared_reference_episode_support_20260908/result.json。75共同时段，74自然成熟；按最近20个已成熟且有状态的时段、至少10个时段100状态且三类都有样本，原141月度日程中138个有支持，最早2015-03-02。全部2458状态保留，源日历一致。没有拟合、预测、选择器或新策略收益。不重复支持核对。", "",
        "拟固定共享八项状态系数、加S1和R2两个任务标记截距；实际D60两标记均零。各来源目标仍为其本身原自然退出相对下一开盘提前退出的含成本价值，不能拿新D60退出回填其他任务。每个已成熟共同时段总权重一，时段内各有状态的自然周期平分，周期内各状态平分；原岭惩罚一、周期训练标准化clip5。最近20完整成熟时段，min10时段100状态、三任务俱全；不足不训练。允许更早模型来自更多已成熟旧参考证据，不借未来、降低原十单位或一百状态下限。", "",
        "下一108先明确并测试时段完整成熟、未来延长正在持仓时段不能改变旧可用时段、原状态及标签保留、任务标记、层级权重和实际D60调用，然后冻结一个设置。原D60进入、6/8/60自然退出、两次负预测退出、原条件消失后重现和两日冷却不变。零新参考，仅共享模型和四实际账户；这里是跨任务迁移假设，原自然退出期限不同可能带来负迁移，不能预先宣称有效。", "",
        "当前108只有已核可用性及待登记方案，尚未实现、训练或生成账户。不做GPT包、EPS/公募补齐或额外安全审计；完整目标与仅510300现金权限保持。"])+"\n", encoding="utf-8")
    OUT.mkdir(parents=True)
    DOCUMENT.write_text("\n".join(["# 第107轮：隔夜下行风险占比退出", "", decision, "", "## 主历史：2020年1月2日至2026年8月14日开盘", "", *table(result["all_metrics"]),
        "## 较早历史：2015年1月5日至2019年12月31日开盘", "", *table(result["earlier_diagnostics"]),
        "六项必要测试通过，114个成熟模型和四账户约5.27秒。九项状态、原完整周期权重、加权岭最优条件、保存预测和完整账户已核。主两种费用各有一次2024年10月8日开盘买入被方向涨停规则阻止，保留未成交；没有将原9月30日收盘请求假设为成交。两种费用账户的退出由各自实际成本和状态决定。", "",
        *(ROOT / cfg["rules"]).read_text(encoding="utf-8").splitlines()[1:]])+"\n", encoding="utf-8")
    for p in RESEARCH.iterdir():
        if p.is_file() and p.suffix in {".csv", ".json"}:
            shutil.copy2(p, OUT / p.name)
    shutil.copy2(CONFIG, OUT / "冻结设置.json")
    shutil.copy2(NEXT, OUT / "下一项研究方向.md")
    for period, label in [("evaluation", "主历史"), ("earlier_diagnostic", "较早历史")]:
        for cost in cfg["costs"]:
            for kind in ["ledger", "decisions"]:
                pd.read_parquet(RESEARCH / period / cost / f"{PRIMARY}_{kind}.parquet").to_csv(OUT / f"{label}_{cost}_{kind}.csv", index=False, encoding="utf-8-sig")
    record = {"round": 107, "study": result["study_id"], "title": "原八项退出加入隔夜下行风险占比", "status": status, "result": str((RESEARCH / "result.json").relative_to(ROOT)),
              **{k: result[k] for k in ["candidate_configurations", "evaluation_accounts", "new_accounts_generated", "reused_control_accounts", "earlier_diagnostic_accounts", "new_earlier_diagnostic_accounts", "new_model_fits", "new_reference_accounts"]},
              "evaluated_candidate_source_runs": 1, "primary_base": metric(result, PRIMARY), "primary_stress": metric(result, PRIMARY, cost="STRESS"), "post_selected_best_base": metric(result, PRIMARY)}
    index["completed_rounds"].append(record)
    for key in ["registered_configurations_in_this_resumption", "evaluated_configurations_in_this_resumption", "evaluated_candidate_source_runs_including_corrected_replays", "registered_candidate_source_runs_including_unrun_legacy_bindings"]:
        index[key] += 1
    index["evaluation_accounts_in_this_resumption"] += result["evaluation_accounts"]
    index.update(updated_at=now(), latest_completed_round=record, running_studies=[], goal_achieved=False, status="ROUND107_OVERNIGHT_DOWNSIDE_NO_TARGET_FULL_GOAL_NOT_MET",
                 count_warning="107轮，383不同设置，399已评价来源版本，404登记含5旧未运行，1508主评价记录。",
                 next_work={"status": "SHARED_REFERENCE_EPISODE_SUPPORT_VERIFIED_NOT_REGISTERED", "focus": "三任务共享继续价值及共同持仓时段去重；成熟支持已核", "source": str(NEXT.relative_to(ROOT))},
                 process_state_note="106及107共12测试、228成熟拟合、8新账户和保存核对交付完成；108时段支持已核，尚未登记。")
    index["deliveries"].append({"created_at": now(), "type": "OVERNIGHT_DOWNSIDE_ROUND107_FAST_CHINESE_RESULTS", "rounds": [107], "directory": str(OUT), "main_document": str(DOCUMENT), "new_gpt_review_archive_created": False})
    write_json(INDEX, index)
    write_json(OUT / "交付回执.json", {"created_at": now(), "main_document": str(DOCUMENT), "goal_achieved": False, "new_gpt_review_archive_created": False}, exclusive=True)
    print(json.dumps({"交付": str(DOCUMENT), "核对": receipt, "未成交": blocked}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
