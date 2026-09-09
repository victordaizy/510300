"""核对连续历史信息边界及真实账户，分别报告点目标与完整目标。"""
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from research.continuous_interaction_budget_v1 import ROOT, OUT as RESEARCH, CONFIG, PRIMARY, P91
from research.profit_drawdown_interaction_inputs_v1 import FEATURES
from research.adaptive_allocation_v1 import normalize_dividends, summarize
from research.intraday_overnight_increment_v1 import now, require, write_json, digest
from scripts.review_round74_saved import saved_cycles
from scripts.finalize_round60_20260907 import metric
from scripts.finalize_round96_20260908 import table

OUT=ROOT/"deliverables/510300交互退出风险预算_第103轮_20260908"
DOCUMENT=OUT/"交互退出风险预算_结果及中文规则.md"
INDEX=ROOT/"reports/research/510300_sharpe_1_2_latest_research.json"
NEXT=ROOT/"docs/510300_AFTER_CONTINUOUS_INTERACTION_BUDGET_20260908.md"



def main():
    index=json.loads(INDEX.read_text(encoding="utf-8"));require(index["latest_completed_round"]["round"]==102 and not OUT.exists(),"第103轮前序或交付状态不符")
    cfg=json.loads(CONFIG.read_text(encoding="utf-8"));result=json.loads((RESEARCH/"result.json").read_text(encoding="utf-8"))
    dividends=normalize_dividends(pd.read_csv(ROOT/cfg["dividends"]));data=pd.read_parquet(ROOT/cfg["features"])
    checks,cycles,differences,concentration,metrics,factor_differences,reference_checks=[],[],[],[],[],[],[]
    full=pd.read_parquet(RESEARCH/"evaluation_factors.parquet")
    ref_first=int(np.flatnonzero(data.date.ge(cfg["reference_start"]))[0])
    ref_folder=RESEARCH/"continuous_references/BASE"
    model_records=json.loads((ROOT/cfg["saved_models"]).read_text(encoding="utf-8"))["models"]
    model_lookup={pd.Timestamp(r["fit_origin"]):r for r in model_records}
    reference_prediction_count=0
    for number,model in enumerate(["PANIC_ONLY","INTERACTION_REFERENCE"]):
        folder = P91/"continuous_references/BASE" if model=="PANIC_ONLY" else ref_folder
        ledger=pd.read_parquet(folder/f"{model}_ledger.parquet");decisions=pd.read_parquet(folder/f"{model}_decisions.parquet")
        require(pd.DatetimeIndex(ledger.date).equals(pd.DatetimeIndex(data.date.iloc[ref_first:])),"连续参考收益日历不符")
        require(pd.DatetimeIndex(decisions.origin).equals(pd.DatetimeIndex(data.date.iloc[ref_first-1:-1])),"连续参考意向日历不符")
        require(pd.DatetimeIndex(decisions.execution_date).equals(pd.DatetimeIndex(data.date.iloc[ref_first:])),"连续参考执行不是下一开盘")
        prefix="panic" if number==0 else "learned"
        np.testing.assert_allclose(full[prefix+"_reference_return"].iloc[ref_first:],ledger.net_return,rtol=0,atol=0)
        np.testing.assert_allclose(full[prefix+"_state"].iloc[ref_first-1:-1],decisions.reference_weight,rtol=0,atol=0,equal_nan=True)
        np.testing.assert_allclose(ledger.cash+ledger.shares*ledger.mark+ledger.dividend_receivable,ledger.equity,rtol=0,atol=1e-6)
        np.testing.assert_allclose(ledger.equity/np.r_[cfg["initial_capital"],ledger.equity.iloc[:-1]]-1,ledger.net_return,rtol=0,atol=1e-12)
        require(int(ledger.shares_before.iloc[0])==0 and int(ledger.shares.iloc[-1])==0,"参考初始化或终点未结清")
        if model=="INTERACTION_REFERENCE":
            for row in decisions[decisions.continuation_prediction.notna()].itertuples():
                m=model_lookup[pd.Timestamp(row.learning_fit_origin)]
                require(pd.Timestamp(row.learning_fit_origin)<=row.origin and m["latest_exit_index"]<=m["fit_index"]<=row.origin_index,"参考学习使用未来成熟模型")
                values=np.array([getattr(row, name) for name in FEATURES])
                require(abs(values[-1]-max(values[1],0.)*values[2])<1e-12, "连续交互输入定义不同")
                z=np.clip((values-m["model"]["mean"])/m["model"]["scale"],-5,5)
                require(abs(m["model"]["intercept"]+z@np.asarray(m["model"]["coefficients"])-row.continuation_prediction)<1e-12, "连续参考预测不能复算")
                reference_prediction_count+=1
        reference_checks.append({"model":model,"days":len(ledger),"first_date":ledger.date.iloc[0],"terminal_date":ledger.date.iloc[-1],"max_accounting_error":float(ledger.accounting_error.abs().max())})
    weights=np.array([.5,.5])
    for t in range(ref_first-1,len(full)-1):
        row=full.iloc[t];scheduled=t>=ref_first and row.date.to_period("M")!=full.date.iloc[t-1].to_period("M")
        require(scheduled==bool(row.risk_update_scheduled),"连续预算月首时钟错误")
        if scheduled:
            count=min(cfg["risk_window"],t-ref_first+1);values=full.iloc[t-count+1:t+1][["panic_reference_return","learned_reference_return"]].to_numpy()
            require(row.risk_window_observations==count and row.risk_window_start==full.date.iloc[t-count+1],"连续预算历史被重置或使用未来")
            if count<cfg["risk_window"]:status="NO_VIEW_WARMUP_KEEP_BUDGET"
            elif not np.isfinite(values).all():status="NO_VIEW_INCOMPLETE_WINDOW_KEEP_BUDGET"
            elif (values.std(axis=0,ddof=1)<=0).any():status="NO_VIEW_ZERO_OR_INVALID_RISK_KEEP_BUDGET"
            else:
                covariance=np.cov(values,rowvar=False,ddof=1);den=float(np.var(values[:,0]-values[:,1],ddof=1))
                if den<=0:status="NO_VIEW_DEGENERATE_DIFFERENCE_KEEP_BUDGET"
                else:
                    w=float(np.clip((covariance[1,1]-covariance[0,1])/den,0,1));weights=np.array([w,1-w]);status="MIN_VARIANCE_BUDGET_AVAILABLE"
                    np.testing.assert_allclose([row.reference_covariance,row.difference_variance],[covariance[0,1],den],rtol=0,atol=1e-12)
            require(row.risk_status==status,"连续风险状态错误")
            checks.append({"date":row.date,"observations":count,"status":status,"panic_budget":weights[0]})
        np.testing.assert_allclose([row.panic_budget,row.learned_budget],weights,rtol=0,atol=1e-10)
        np.testing.assert_allclose([row.target],[weights@np.array([row.panic_state,row.learned_state])],rtol=0,atol=1e-10,equal_nan=True)
    for period in ["evaluation","earlier_diagnostic"]:
        f=pd.read_parquet(RESEARCH/f"{period}_factors.parquet")
        if period=="earlier_diagnostic":pd.testing.assert_frame_equal(f.iloc[:-1],full.iloc[:len(f)-1])
        first=int(np.flatnonzero(f.date.ge(cfg["evaluation_start"] if period=="evaluation" else cfg["earlier_start"]))[0])
        oldf=pd.read_parquet(P91/f"{period}_factors.parquet");a=f.iloc[first-1:-1];b=oldf.iloc[first-1:-1]
        oldstates=b[["panic_state","learned_state"]].to_numpy();newstates=a[["panic_state","learned_state"]].to_numpy()
        oldweights=b[["panic_budget","learned_budget"]].to_numpy();newweights=a[["panic_budget","learned_budget"]].to_numpy()
        budget_part=((newweights-oldweights)*oldstates).sum(axis=1);state_part=(newweights*(newstates-oldstates)).sum(axis=1)
        np.testing.assert_allclose(a.target.to_numpy()-b.target.to_numpy(),budget_part+state_part,rtol=0,atol=1e-12)
        factor_differences.append({"period":period,"initial_panic_budget":float(a.panic_budget.iloc[0]),"initial_risk_origin":a.last_successful_risk_origin.iloc[0],"initial_risk_status":a.risk_status.iloc[0],"panic_state_different_days":int((abs(newstates[:,0]-oldstates[:,0])>1e-12).sum()),"learned_state_different_days":int((abs(newstates[:,1]-oldstates[:,1])>1e-12).sum()),"target_different_days":int((abs(a.target.to_numpy()-b.target.to_numpy())>1e-12).sum()),"budget_different_days":int((abs(newweights[:,0]-oldweights[:,0])>1e-12).sum()),"new_monthly_status_counts":a[a.risk_update_scheduled].risk_status.value_counts().to_dict()})
        for cost in cfg["costs"]:
            ledger = pd.read_parquet(RESEARCH / period / cost / f"{PRIMARY}_ledger.parquet")
            decisions = pd.read_parquet(RESEARCH / period / cost / f"{PRIMARY}_decisions.parquet")
            np.testing.assert_allclose(decisions.reference_weight, f.target.iloc[first-1:-1], atol=0, rtol=0, equal_nan=True)
            previous = np.r_[cfg["initial_capital"], ledger.equity.iloc[:-1]]
            np.testing.assert_allclose(ledger.cash+ledger.shares*ledger.mark+ledger.dividend_receivable, ledger.equity, atol=1e-6, rtol=0)
            np.testing.assert_allclose(ledger.equity/previous-1, ledger.net_return, atol=1e-12, rtol=0)
            actual_by_date=ledger.set_index("date")
            prices=pd.read_parquet(ROOT/cfg["features"]).set_index("date").close
            for row in decisions.itertuples():
                equity=cfg["initial_capital"] if row.origin_index==first-1 else float(actual_by_date.loc[row.origin,"equity"])
                shares=0 if row.origin_index==first-1 else int(actual_by_date.loc[row.origin,"shares"])
                target=float(row.reference_weight);price=float(prices.loc[row.origin])
                quantity=0
                if np.isfinite(target):
                    desired=int(np.floor(target*equity/price/cfg["lot"]))*cfg["lot"]
                    if target>0 and shares>0 and abs(target-shares*price/equity)<cfg["weight_band"]:desired=shares
                    quantity=desired-shares
                require(quantity==row.requested_quantity,"组合选择未按自身净值及实际股数下单")
            actual, saved = summarize(ledger, cfg), metric(result, PRIMARY, period, cost)
            for field in ["net_sharpe", "annualized_return", "max_drawdown", "cumulative_return", "trade_count", "commission", "slippage_cost"]:
                require(abs(actual[field]-saved[field]) < 1e-9, "新账户保存指标不符")
            metrics.append({"period": period, "cost": cost, **actual})
            group = saved_cycles(ledger, dividends, cfg)
            cycles.extend({"period": period, "cost": cost, **c} for c in group)
            total = sum(c["net_profit"] for c in group)
            top = sorted((c["net_profit"] for c in group if c["net_profit"] > 0), reverse=True)[:3]
            concentration.append({"period": period, "cost": cost, "cycles": len(group), "losing_cycles": sum(c["net_profit"] < 0 for c in group),
                "net_profit": total, "largest_three_positive_profit": sum(top), "largest_three_vs_net_profit": sum(top)/total if total > 0 else None})
            for control in ["CONTINUOUS_REFERENCE_MIN_VARIANCE"]:
                old=pd.read_parquet(RESEARCH/period/cost/f"{control}_ledger.parquet")
                differences.append({"period":period,"cost":cost,"control":control,"terminal_nav_difference":float(ledger.equity.iloc[-1]-old.equity.iloc[-1]),
                    **{f"{field}_difference":float(ledger[field].sum()-old[field].sum()) for field in ["price_pnl","dividend_recognized","commission","slippage_cost"]}})
    for filename,rows in [("saved_continuous_budget_checks.csv",checks),("saved_reference_account_checks.csv",reference_checks),("saved_actual_cycles.csv",cycles),("saved_account_checks.csv",metrics),("saved_parent_differences.csv",differences),("saved_profit_concentration.csv",concentration),("saved_initial_history_and_target_differences.csv",factor_differences)]:
        pd.DataFrame(rows).to_csv(RESEARCH/filename,index=False,encoding="utf-8-sig")
    receipt={"verified_at":now(),"status":"CONTINUOUS_HISTORY_MODEL_CLOCK_BUDGET_PREFIX_AND_COMPLETE_ACCOUNTS_CHECKED","continuous_monthly_budgets_checked":len(checks),"saved_reference_prediction_clocks":reference_prediction_count,"reference_accounts_checked":len(reference_checks),"new_evaluation_account_metrics_checked":len(metrics),"complete_evaluation_cycles":len(cycles),"new_accounts_or_fits":0,"reviewer_source_sha256":digest(Path(__file__)),"security_audit_performed":False}
    write_json(RESEARCH/"saved_verification_receipt.json",receipt,exclusive=True)
    gates={"main_base_sharpe_at_least_1_2":metric(result,PRIMARY)["net_sharpe"]>=1.2,"main_stress_sharpe_at_least_1_2":metric(result,PRIMARY,cost="STRESS")["net_sharpe"]>=1.2,"earlier_base_sharpe_at_least_1_2":metric(result,PRIMARY,"earlier_diagnostic")["net_sharpe"]>=1.2,"earlier_stress_sharpe_at_least_1_2":metric(result,PRIMARY,"earlier_diagnostic","STRESS")["net_sharpe"]>=1.2,"positive_main_annual_excess":metric(result,PRIMARY)["annualized_return_excess_vs_buy_hold"]>0,"positive_earlier_annual_excess":metric(result,PRIMARY,"earlier_diagnostic")["annualized_return_excess_vs_buy_hold"]>0,"independent_validation":"NOT_ESTABLISHED"}
    require(not result["historical_point_target_met"], "本轮出现新点目标，需重新检查结论")
    status="COMPLETED_INTERACTION_BUDGET_EARLY_SLIGHT_GAIN_MAIN_WORSE_NOT_TARGET"
    decision="第103轮未达标。主基础／压力夏普1.115／1.056，较早0.442／0.406；原91分别1.233／1.174和0.431／0.395。较早略升，主明显下降，两段年化收益均低于买入持有。关闭这一固定交互组合，不通过改窗口、交互或父策略救回。"
    write_json(RESEARCH/"acceptance_outcome.json",{"recorded_at":now(),"status":status,"decision":decision,"gates":gates,"historical_main_base_point_target_met":False,"goal_achieved":False,"position_impact":0},exclusive=True)
    NEXT.write_text("\n".join(["# 第103轮之后：过去预测误差能否支持模型选择", "", decision, "",
        "103四测试3.76秒，一个新连续九项学习参考及四账户约7.59秒；第102轮114模型和原91急跌参考直接复用，没有新训练。主201持股收盘45成交、早340收盘35成交，所有目标完整，无未成交。159月首预算、参考成熟预测、早历史前缀及四账户均已核，不重跑本轮。", "",
        "102单策略主0.575/0.511、早0.781/0.758；103组合把旧91主点候选拉低至1.115/1.056，早仅改善至0.442/0.406。不能事后按年份选102或91，也不能将单笔2019改善外推成泛化。原91仍仅主基础局部1.233，压力、较早与超额没有完成。", "",
        "下一104先做有界的输入可用性核对：原八项和固定九项退出模型，能否在同一自然参考状态上，按不晚于该状态原点的历史模型生成两条过去预测，并在该自然周期完全退出后才观察两条误差。复用原1461个D60状态及两套保存模型，不重训、不重跑参考。只先检查同期预测对的完整周期数、成熟日期及最早可用日期，不先计算选择器或组合收益。", "",
        "拟研究的机制为按已成熟预测误差选择或混合退出模型，不按年份、事后夏普或当前持仓未结束的盈亏选模型。它与已做的两个实际策略收益协方差预算、过去Sharpe选择器和第90轮市场条件预测父组合差额不同。但尚需仅针对已实现模型选择/预测误差组合做一次有界查重，然后才能登记一个有限规则。不要退回旧固定共享、HMM、持仓CUSUM或RBF参数搜索。", "",
        "若有完整同期预测证据，可论证采用周期等权、最近20已成熟完整周期和原10周期100状态下限的非负最小均方误差混合。误差使用预测减相同已成熟继续价值标签；目标为未去均值的误差平方，明确包含偏差，不冒充收益方差。预测初期或差值退化时如何保留无观点及沿用基线须在新结果前明确。当前这仅是方向，不能把待选规则当已冻结或已改善。若完整同期周期不足直接放弃，不降低样本门槛或借未来模型。", "",
        "唯一最新状态以总索引next_work为准；104尚无实现、配置、模型或账户。继续完整费用与全部现金日、中文规则、必要检查，暂停EPS等慢源和GPT包。", ""]),encoding="utf-8")
    OUT.mkdir(parents=True)
    lines=["# 第103轮：九项退出模型的连续风险预算组合", "",decision,"","## 主历史：2020年1月2日至2026年8月14日开盘", "", *table(result["all_metrics"]), "## 较早历史：2015年1月5日至2019年12月31日开盘", "", *table(result["earlier_diagnostics"]),
        "## 组合效果", "", "主基础年化3.02%、最大回撤2.48%，较早基础年化3.02%、最大回撤11.07%。相对旧91，主夏普下降、较早只有小幅提高，两段基础年化均低于同期买入持有。新组合没有延续主历史1.2的局部点结果。", "",
        "主平均股票比例4.41%，201个持股收盘、45笔成交；较早平均11.46%，340个持股收盘、35笔成交。所有现金日纳入净收益，低波动不能代替超额或跨阶段验证。", "",
        f"四项必要测试通过，复用114个既有九项模型和原急跌参考，只新增一个连续学习参考及四个评价账户。已核{len(checks)}个月度预算、{reference_prediction_count}个保存参考预测的成熟时点和数值、完整较早前缀、四账户实际请求与{len(cycles)}个评价周期。", "",
        "保留旧91与102的原结果，关闭这个固定组合，不再调整本交互或预算窗口。下一先核对两模型是否拥有足够已成熟的同期历史预测误差，用于事前定义的模型选择研究。", "", *(ROOT/cfg["rules"]).read_text(encoding="utf-8").splitlines()[1:]]
    DOCUMENT.write_text("\n".join(lines),encoding="utf-8")
    for p in RESEARCH.iterdir():
        if p.is_file() and p.suffix in {".csv",".json"}:
            shutil.copy2(p,OUT/p.name)
    shutil.copy2(CONFIG,OUT/"冻结设置.json")
    shutil.copy2(NEXT,OUT/"下一项研究方向.md")
    for period,label in [("evaluation","主历史"),("earlier_diagnostic","较早历史")]:
        for cost in cfg["costs"]:
            for kind in ["ledger","decisions"]:
                pd.read_parquet(RESEARCH/period/cost/f"{PRIMARY}_{kind}.parquet").to_csv(OUT/f"{label}_{cost}_{kind}.csv",index=False,encoding="utf-8-sig")
    record={"round":103,"study":result["study_id"],"title":"九项交互学习参考与原急跌的连续风险预算","status":status,"result":str((RESEARCH/"result.json").relative_to(ROOT)),**{k:result[k] for k in ["candidate_configurations","evaluation_accounts","new_accounts_generated","reused_control_accounts","earlier_diagnostic_accounts","new_earlier_diagnostic_accounts","new_model_fits","new_reference_accounts"]},"evaluated_candidate_source_runs":1,"primary_base":metric(result,PRIMARY),"primary_stress":metric(result,PRIMARY,cost="STRESS"),"post_selected_best_base":metric(result,PRIMARY),"historical_main_base_point_target_met":False,"full_goal_achieved":False}
    index["completed_rounds"].append(record)
    for key in ["registered_configurations_in_this_resumption","evaluated_configurations_in_this_resumption","evaluated_candidate_source_runs_including_corrected_replays","registered_candidate_source_runs_including_unrun_legacy_bindings"]:
        index[key]+=1
    index["evaluation_accounts_in_this_resumption"]+=result["evaluation_accounts"]
    index.update(updated_at=now(),latest_completed_round=record,running_studies=[],goal_achieved=False,status="ROUND103_INTERACTION_BUDGET_NO_SHARPE_TARGET_FULL_GOAL_NOT_MET",count_warning="103轮，379不同设置，395已评价来源版本，400登记含5旧未运行，1470主评价记录。",next_work={"status":"MATURE_PAIRED_FORECAST_ERROR_FEASIBILITY_NOT_REGISTERED","focus":"检查两个退出模型在同一自然状态的同期预测与完整成熟误差支持","source":str(NEXT.relative_to(ROOT))},process_state_note="102和103已完成交付，九项交互及其组合未实现目标；104仅预测误差证据可用性方向，无新账户。")
    index["deliveries"].append({"created_at":now(),"type":"CONTINUOUS_INTERACTION_BUDGET_ROUND103_FAST_CHINESE_RESULTS","rounds":[103],"directory":str(OUT),"main_document":str(DOCUMENT),"new_gpt_review_archive_created":False})
    write_json(INDEX,index)
    write_json(OUT/"交付回执.json",{"created_at":now(),"main_document":str(DOCUMENT),"goal_achieved":False,"new_gpt_review_archive_created":False},exclusive=True)
    print(json.dumps({"交付":str(DOCUMENT),"核对":receipt,"差额":differences,"预算状态":factor_differences},ensure_ascii=False,default=str),flush=True)


if __name__=="__main__":main()
