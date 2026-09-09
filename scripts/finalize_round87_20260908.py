"""经济权重和实际账户关键核对，简洁交付，不重拟合。"""
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from research.kernel_continuation_exit_v1 import ROOT, OUT as RESEARCH, CONFIG, PRIMARY
from research.learned_cycle_exit_v1 import training_rows
from research.learned_cycle_exit_v1 import FEATURES
from research.adaptive_allocation_v1 import normalize_dividends, summarize
from research.intraday_overnight_increment_v1 import now, require, write_json, digest
from scripts.review_round74_saved import saved_cycles
from scripts.finalize_round60_20260907 import metric, table

OUT = ROOT / "deliverables/510300相似状态退出_第87轮_20260908"
DOCUMENT = OUT / "相似状态退出_结果及全部中文规则.md"
INDEX = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
NEXT = ROOT / "docs/510300_AFTER_KERNEL_THIRD_SIGNAL_20260908.md"


def main():
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    require(index["latest_completed_round"]["round"] == 86 and not OUT.exists(), "87前序或交付状态不符")
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    result = json.loads((RESEARCH / "result.json").read_text(encoding="utf-8"))
    require(not result["historical_point_target_met"], "出现目标值时不能直接写成未达标")
    models = json.loads((RESEARCH / "saved_models.json").read_text(encoding="utf-8"))["models"]
    by_date = {pd.Timestamp(m["fit_origin"]): m for m in models}
    members = pd.read_parquet(RESEARCH / "training_memberships.parquet")
    samples = pd.read_parquet(ROOT / cfg["samples"])
    samples = samples[samples.signal.eq("D60_INTRA")]
    data = pd.read_parquet(ROOT / cfg["features"])
    old_models = json.loads((ROOT / cfg["saved_models"]).read_text(encoding="utf-8"))["models"]["D60_INTRA__RIDGE"]
    old_by_date = {m["fit_index"]: m for m in old_models}
    weight_checks=[]
    chinese=pd.read_csv(RESEARCH/"全部月度参考状态及系数.csv")
    for record in [m for m in models if m["status"]=="FIT_COMPLETE"]:
        t=record["fit_index"]; rows,ids=training_rows(samples,t,cfg)
        require(ids==record["training_cycles"] and (rows.exit_index<=t).all(),"核模型使用未来或不同周期")
        saved=members[members.fit_index.eq(t)].sort_values(["cycle_id","origin_index"])
        np.testing.assert_allclose(rows.sample_weight,saved.sample_weight,atol=0,rtol=0)
        np.testing.assert_allclose(saved.groupby("cycle_id").sample_weight.sum(),1.,atol=1e-12,rtol=0)
        model=record["model"]
        for field in ["mean","scale"]:
            np.testing.assert_allclose(model[field],old_by_date[t]["model"][field],atol=1e-12,rtol=0)
        z=np.clip((rows[FEATURES].to_numpy()-model["mean"])/model["scale"],-5,5)
        np.testing.assert_allclose(z,model["centers"],atol=1e-12,rtol=0)
        require(model["training_origin_indexes"]==rows.origin_index.to_list() and model["training_cycle_ids"]==rows.cycle_id.to_list(),"核参考状态身份不符")
        # 独立采用逐项平方差构造相似矩阵，检验保存解方程，无重新拟合。
        kernel=np.exp(-.125*np.sum((z[:,None,:]-z[None,:,:])**2,axis=2))
        coef=np.asarray(model["dual_coefficients"]);w=rows.sample_weight.to_numpy()
        residual=kernel@coef+model["intercept"]+cfg["ridge_alpha"]*coef/w-rows.target.to_numpy()
        require(np.max(np.abs(residual))<1e-10 and abs(coef.sum())<1e-10,"核模型保存解不满足加权截距方程")
        published=chinese[chinese["模型生效日"].eq(record["fit_origin"])]
        require(len(published)==len(rows),"中文参考状态表遗漏")
        np.testing.assert_allclose(published["训练状态系数"],coef,atol=1e-14,rtol=0)
        np.testing.assert_allclose(published.iloc[:,4:].to_numpy(float),z,atol=1e-14,rtol=0)
        weight_checks.append({"fit_index":t,"fit_origin":record["fit_origin"],"cycles":len(ids),"rows":len(rows),"largest_equation_residual":np.max(np.abs(residual))})
    require(len(weight_checks)==114 and result["failed_fits"]==0,"核模型支持或求解状态不符")
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
                    x=np.array([getattr(row,c) for c in FEATURES]);np.testing.assert_allclose(x,expected_state,atol=1e-12,rtol=0)
                    if row.learning_status=="PREDICTION_AVAILABLE":
                        m=by_date[row.learning_fit_origin];require(m["latest_exit_index"]<=m["fit_index"]<=row.origin_index,"核状态实际调用未来模型")
                        z=np.clip((x-m["model"]["mean"])/m["model"]["scale"],-5,5)
                        centers=np.asarray(m["model"]["centers"])
                        similarity=np.exp(-.125*np.sum((centers-z)**2,axis=1))
                        score=float(m["model"]["intercept"]+similarity@np.asarray(m["model"]["dual_coefficients"]))
                        require(abs(score-row.continuation_prediction)<1e-12,"核状态预测不能按系数复算")
                        count=count+1 if score<0 else 0
                        predictions.append({"period":period,"cost":cost,"origin":row.origin,"cycle":cycle_id,"continuation_prediction":score})
                    else:
                        require(pd.isna(row.continuation_prediction),"无模型时仍填预测")
                        count=0
                    require(count==row.negative_confirmation_count and bool(row.learned_exit_requested)==(count>=2),"核状态确认或周期重置不符")
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
    receipt = {"verified_at": now(), "status": "KEY_PAST_KERNEL_WEIGHTED_INTERCEPT_AND_ACCOUNT_CHECKS_COMPLETE", "mature_models": len(weight_checks), "published_reference_states": len(chinese), "actual_holding_state_rows": sum(r["holding_decisions"] for r in result["model_coverage"]),
        "saved_scores_recomputed": len(predictions), "new_account_metrics_recomputed": len(checked), "complete_cycles": len(cycles),
        "new_accounts_or_models": 0, "reviewer_source_sha256": digest(Path(__file__)), "security_audit_performed": False}
    write_json(RESEARCH / "saved_verification_receipt.json", receipt, exclusive=True)
    status = "COMPLETED_KERNEL_CONTINUATION_TARGET_NOT_MET"
    write_json(RESEARCH / "acceptance_outcome.json", {"status": status, "recorded_at": now(), "goal_achieved": False,
        "decision": "核相似模型四情景均低于原线性退出，关闭本核状态版本，不搜索尺度、惩罚或输入救回。", "position_impact": 0}, exclusive=True)
    OUT.mkdir(parents=True)
    lines=["# 第87轮：按过去相似持仓状态退出的结果", "", "主基础／压力净夏普0.0623／负0.0083，较早0.6420／0.6206，四种情景都弱于原八项线性退出。没有达到1.2，本项结束，不调整核尺度或惩罚挽救。", "",
        "## 主评价：2020年1月2日至2026年8月14日开盘", ""]+table(result["all_metrics"])
    lines += ["## 较早历史：2015年1月5日至2019年12月31日开盘", ""]+table(result["earlier_diagnostics"])
    lines += ["114次训练全部完成，27个早期月份沿用原样本不足状态。主每档27周期、54笔成交、226个持仓收盘，25次学习退出。较早9周期、18笔、340个持仓收盘，零学习退出；204个早期持仓收盘没有成熟模型。各段零未成交请求。", "",
        "主基础年化0.18%、最大回撤19.93%；较早年化7.78%、最大回撤13.79%。非线性相似学习在两个时期形成了很不一样的退出强度，并未改善实际收益。下一项回到组合层，只检验已有第三类突破机会能否补充当前两策略；该方向尚未登记或运行。", "",
        f"六项必要测试通过。114个月度保存解、{len(chinese)}条中文参考状态和系数、{len(predictions)}条有效实际预测、全部实际持仓因子、四账户指标及{len(cycles)}个含分红周期均已核对，没有重新训练或追加诊断账户。全部八项因子定义在下文；各月尺度、截距在“每月八项核状态退出模型中文规则.md”，每个参考状态与训练系数在“全部月度参考状态及系数.csv”。", "", "## 全部中文规则", ""]
    lines += (ROOT / cfg["rules"]).read_text(encoding="utf-8").splitlines()[1:]
    DOCUMENT.write_text("\n".join(lines)+"\n", encoding="utf-8")
    for p in RESEARCH.iterdir():
        if p.is_file() and p.suffix in {".csv", ".json", ".md"}:
            shutil.copy2(p, OUT / p.name)
    shutil.copy2(CONFIG, OUT / "冻结设置.json")
    shutil.copy2(NEXT, OUT / "下一项第三信号组合方向.md")
    for period, label in [("evaluation", "主评价"), ("earlier_diagnostic", "较早历史")]:
        for cost in cfg["costs"]:
            for kind in ["ledger", "decisions"]:
                pd.read_parquet(RESEARCH / period / cost / f"{PRIMARY}_{kind}.parquet").to_csv(OUT / f"{label}_{cost}_{'完整账户' if kind=='ledger' else '全部因子与进出场'}.csv", index=False, encoding="utf-8-sig")
    record = {"round": 87, "study": result["study_id"], "title": "按原八项相似持仓状态进行核岭退出", "status": status,
        "result": str((RESEARCH / "result.json").relative_to(ROOT)), **{k: result[k] for k in ["candidate_configurations", "evaluation_accounts", "new_accounts_generated", "reused_control_accounts", "earlier_diagnostic_accounts", "new_earlier_diagnostic_accounts", "new_model_fits", "new_reference_accounts"]},
        "evaluated_candidate_source_runs": 1, "primary_base": metric(result, PRIMARY), "primary_stress": metric(result, PRIMARY, cost="STRESS"), "post_selected_best_base": metric(result, PRIMARY)}
    index["completed_rounds"].append(record)
    for k in ["registered_configurations_in_this_resumption", "evaluated_configurations_in_this_resumption", "evaluated_candidate_source_runs_including_corrected_replays", "registered_candidate_source_runs_including_unrun_legacy_bindings"]:
        index[k] += 1
    index["evaluation_accounts_in_this_resumption"] += result["evaluation_accounts"]
    index.update(updated_at=now(), latest_completed_round=record, running_studies=[], goal_achieved=False, status="ROUND87_COMPLETE_KERNEL_CONTINUATION_TARGET_NOT_MET",
        count_warning="87轮，361不同设置，377已评价来源版本，382登记含5旧未运行，1318主评价记录。",
        next_work={"status": "THIRD_COMPRESSED_BREAKOUT_MIN_VARIANCE_DIRECTION_NOT_REGISTERED", "focus": "用已保存压缩后突破信号补充原两策略风险预算", "source": str(NEXT.relative_to(ROOT))},
        process_state_note="86和87各114次拟合、四账户、关键核对和中文交付完成；88三类机会组合仅方向。",
        research_speed_priority="docs/510300_FAST_RESEARCH_CADENCE_20260908.md")
    index["deliveries"].append({"created_at": now(), "type": "KERNEL_CONTINUATION_ROUND87_FAST_CHINESE_RESULTS", "rounds": [87], "directory": str(OUT), "main_document": str(DOCUMENT), "new_gpt_review_archive_created": False})
    write_json(INDEX, index)
    write_json(OUT / "交付回执.json", {"created_at": now(), "main_document": str(DOCUMENT), "goal_achieved": False, "new_gpt_review_archive_created": False}, exclusive=True)
    print(json.dumps({"交付": str(DOCUMENT), "核对": receipt}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
