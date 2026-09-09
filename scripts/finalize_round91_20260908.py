"""核对连续历史信息边界及真实账户，分别报告点目标与完整目标。"""
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from research.continuous_reference_min_variance_v1 import ROOT, OUT as RESEARCH, CONFIG, PRIMARY
from research.adaptive_allocation_v1 import normalize_dividends, summarize
from research.intraday_overnight_increment_v1 import now, require, write_json, digest
from scripts.review_round74_saved import saved_cycles
from scripts.finalize_round60_20260907 import metric, table

OUT=ROOT/"deliverables/510300连续历史风险预算_第91轮_20260908"
DOCUMENT=OUT/"连续历史风险预算_结果及中文规则.md"
INDEX=ROOT/"reports/research/510300_sharpe_1_2_latest_research.json"
NEXT=ROOT/"docs/510300_AFTER_CONTINUOUS_HISTORY_CYCLE_RISK_20260908.md"
P82=ROOT/"reports/research/510300_two_policy_min_variance_v1"


def main():
    index=json.loads(INDEX.read_text(encoding="utf-8"));require(index["latest_completed_round"]["round"]==90 and not OUT.exists(),"第91轮前序或交付状态不符")
    cfg=json.loads(CONFIG.read_text(encoding="utf-8"));result=json.loads((RESEARCH/"result.json").read_text(encoding="utf-8"))
    dividends=normalize_dividends(pd.read_csv(ROOT/cfg["dividends"]));data=pd.read_parquet(ROOT/cfg["features"])
    checks,cycles,differences,concentration,metrics,factor_differences,reference_checks=[],[],[],[],[],[],[]
    full=pd.read_parquet(RESEARCH/"evaluation_factors.parquet")
    ref_first=int(np.flatnonzero(data.date.ge(cfg["reference_start"]))[0])
    ref_folder=RESEARCH/"continuous_references/BASE"
    model_records=json.loads((ROOT/cfg["saved_models"]).read_text(encoding="utf-8"))["models"]["D60_INTRA__RIDGE"]
    model_lookup={pd.Timestamp(r["fit_origin"]):r for r in model_records}
    reference_prediction_count=0
    for number,model in enumerate(["PANIC_ONLY","REARM_RIDGE"]):
        ledger=pd.read_parquet(ref_folder/f"{model}_ledger.parquet");decisions=pd.read_parquet(ref_folder/f"{model}_decisions.parquet")
        require(pd.DatetimeIndex(ledger.date).equals(pd.DatetimeIndex(data.date.iloc[ref_first:])),"连续参考收益日历不符")
        require(pd.DatetimeIndex(decisions.origin).equals(pd.DatetimeIndex(data.date.iloc[ref_first-1:-1])),"连续参考意向日历不符")
        require(pd.DatetimeIndex(decisions.execution_date).equals(pd.DatetimeIndex(data.date.iloc[ref_first:])),"连续参考执行不是下一开盘")
        prefix="panic" if number==0 else "learned"
        np.testing.assert_allclose(full[prefix+"_reference_return"].iloc[ref_first:],ledger.net_return,rtol=0,atol=0)
        np.testing.assert_allclose(full[prefix+"_state"].iloc[ref_first-1:-1],decisions.reference_weight,rtol=0,atol=0,equal_nan=True)
        np.testing.assert_allclose(ledger.cash+ledger.shares*ledger.mark+ledger.dividend_receivable,ledger.equity,rtol=0,atol=1e-6)
        np.testing.assert_allclose(ledger.equity/np.r_[cfg["initial_capital"],ledger.equity.iloc[:-1]]-1,ledger.net_return,rtol=0,atol=1e-12)
        require(int(ledger.shares_before.iloc[0])==0 and int(ledger.shares.iloc[-1])==0,"参考初始化或终点未结清")
        if model=="REARM_RIDGE":
            for row in decisions[decisions.continuation_prediction.notna()].itertuples():
                m=model_lookup[pd.Timestamp(row.learning_fit_origin)]
                require(pd.Timestamp(row.learning_fit_origin)<=row.origin and m["latest_exit_index"]<=m["fit_index"]<=row.origin_index,"参考学习使用未来成熟模型")
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
        oldf=pd.read_parquet(P82/f"{period}_factors.parquet");a=f.iloc[first-1:-1];b=oldf.iloc[first-1:-1]
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
            for control in ["TWO_POLICY_MIN_VARIANCE","PANIC_LEARNED_HALF"]:
                old=pd.read_parquet(RESEARCH/period/cost/f"{control}_ledger.parquet")
                differences.append({"period":period,"cost":cost,"control":control,"terminal_nav_difference":float(ledger.equity.iloc[-1]-old.equity.iloc[-1]),
                    **{f"{field}_difference":float(ledger[field].sum()-old[field].sum()) for field in ["price_pnl","dividend_recognized","commission","slippage_cost"]}})
    for filename,rows in [("saved_continuous_budget_checks.csv",checks),("saved_reference_account_checks.csv",reference_checks),("saved_actual_cycles.csv",cycles),("saved_account_checks.csv",metrics),("saved_parent_differences.csv",differences),("saved_profit_concentration.csv",concentration),("saved_initial_history_and_target_differences.csv",factor_differences)]:
        pd.DataFrame(rows).to_csv(RESEARCH/filename,index=False,encoding="utf-8-sig")
    receipt={"verified_at":now(),"status":"CONTINUOUS_HISTORY_MODEL_CLOCK_BUDGET_PREFIX_AND_COMPLETE_ACCOUNTS_CHECKED","continuous_monthly_budgets_checked":len(checks),"saved_reference_prediction_clocks":reference_prediction_count,"reference_accounts_checked":len(reference_checks),"new_evaluation_account_metrics_checked":len(metrics),"complete_evaluation_cycles":len(cycles),"new_accounts_or_fits":0,"reviewer_source_sha256":digest(Path(__file__)),"security_audit_performed":False}
    write_json(RESEARCH/"saved_verification_receipt.json",receipt,exclusive=True)
    gates={"main_base_sharpe_at_least_1_2":metric(result,PRIMARY)["net_sharpe"]>=1.2,"main_stress_sharpe_at_least_1_2":metric(result,PRIMARY,cost="STRESS")["net_sharpe"]>=1.2,"earlier_base_sharpe_at_least_1_2":metric(result,PRIMARY,"earlier_diagnostic")["net_sharpe"]>=1.2,"earlier_stress_sharpe_at_least_1_2":metric(result,PRIMARY,"earlier_diagnostic","STRESS")["net_sharpe"]>=1.2,"positive_main_annual_excess":metric(result,PRIMARY)["annualized_return_excess_vs_buy_hold"]>0,"positive_earlier_annual_excess":metric(result,PRIMARY,"earlier_diagnostic")["annualized_return_excess_vs_buy_hold"]>0,"independent_validation":"NOT_ESTABLISHED"}
    require(gates["main_base_sharpe_at_least_1_2"] and not gates["main_stress_sharpe_at_least_1_2"] and not gates["earlier_base_sharpe_at_least_1_2"],"本轮实际验收结果与交付叙述不一致")
    status="COMPLETED_MAIN_BASE_POINT_TARGET_MET_CROSS_PERIOD_AND_EXCESS_NOT_MET"
    decision="主基础扣费净夏普1.233达到单区间1.2，压力1.174、较早0.431/0.395未达；主与较早年化收益均低于买持。保留局部达标候选，完整稳定超额与高夏普目标尚未完成。"
    write_json(RESEARCH/"acceptance_outcome.json",{"recorded_at":now(),"status":status,"decision":decision,"gates":gates,"historical_main_base_point_target_met":True,"goal_achieved":False,"position_impact":0},exclusive=True)
    OUT.mkdir(parents=True)
    lines=["# 第91轮：连续历史风险预算的结果", "",decision,"","## 主历史：2020年1月2日至2026年8月14日开盘",""]+table(result["all_metrics"])
    lines += ["## 较早历史：2015年1月5日至2019年12月31日开盘",""]+table(result["earlier_diagnostics"])
    lines += ["## 为什么主夏普提高，以及还缺什么", "","主基础年化3.36%、最大回撤2.31%，比原冷启动组合年化3.99%更低，但波动和回撤也更低，所以夏普从1.137提高至1.233。平均股票敞口4.46%，完整1604日中198个收盘持股，45笔成交；所有现金日均纳入夏普。主买入持有年化3.63%，因此新策略年化仍落后约0.27个百分点。", "","主账户开始时连续历史给急跌策略约90.57%的预算，信息来自2019年10月8日最后一次有效风险更新；之后零波动窗口明确沿用旧预算。原冷启动从50%开始。较早账户开始时连续历史也因零波动而维持50%，所以没有消除2015年的初始风险。", "","两个时期的底层急跌及学习持有意向与原冷启动保存路径逐日相同。主245个、较早252个判断日的股票目标不同，差额由预算改变形成；本次实际结果可定位到连续风险历史，而不能据此假设所有未来参考周期都会相同。主80个月首37次有效预算、43次零波动沿用；较早60个月首38次有效、22次沿用，预算并非每月都能识别。", "","较早基础年化2.93%、最大回撤11.07%，原冷启动年化3.61%、同为11.07%回撤；基础夏普从0.520降为0.431。该阶段并没有跟随主历史一起改善。完整目标还要求费用压力、跨阶段表现及超额证据，这些条件本轮均未全部满足。", ""]
    for c in concentration:
        if c["cost"]=="BASE":
            label="主历史" if c["period"]=="evaluation" else "较早历史"
            lines += [f"{label}{c['cycles']}个完整交易周期，{c['losing_cycles']}个亏损周期；最盈利三周期合计占总净利润{c['largest_three_vs_net_profit']:.1%}。", ""]
    lines += ["2023年全年为空仓零收益，夏普没有定义，保留缺失而非填零。2026年仅到既定8月14日开盘；自然年度展示不得把它称为完整全年。未来独立表现尚未建立，单区间1.233不等于未来可保证1.2。", "","## 全部中文因子与进出场规则", ""]+(ROOT/cfg["rules"]).read_text(encoding="utf-8").splitlines()[1:]
    lines += ["",f"六项必要测试通过；两个连续参考账户与四个新评价账户完成，零新模型拟合。核对{len(checks)}次连续预算、{reference_prediction_count}次保存预测的模型成熟时点、较早日期前缀、四账户全部自身资金请求及{len(cycles)}个完整周期，没有重跑账户。", "","下一项先根据已保存参考周期检验现金日过多是否让两策略日风险比较失真，再判断是否值得研究按真实持仓周期衡量风险；尚无下一策略结果。不重跑本轮，不搜索起点或费用把压力结果推过1.2。", ""]
    DOCUMENT.write_text("\n".join(lines),encoding="utf-8")
    for p in RESEARCH.iterdir():
        if p.is_file() and p.suffix in {".csv",".json"}:shutil.copy2(p,OUT/p.name)
    shutil.copy2(CONFIG,OUT/"冻结设置.json")
    shutil.copy2(ROOT/"docs/510300_TWO_POLICY_RISK_BUDGET_V1.md",OUT/"沿用的两条策略全部中文规则.md")
    coefficient=ROOT/"deliverables/510300下午提前进入_第75轮_20260908/沿用的每月八项模型中文规则.md";shutil.copy2(coefficient,OUT/coefficient.name)
    shutil.copy2(NEXT,OUT/"下一项研究方向.md")
    for period,label in [("evaluation","主评价"),("earlier_diagnostic","较早历史")]:
        for cost in cfg["costs"]:
            for kind in ["ledger","decisions"]:pd.read_parquet(RESEARCH/period/cost/f"{PRIMARY}_{kind}.parquet").to_csv(OUT/f"{label}_{cost}_{kind}.csv",index=False,encoding="utf-8-sig")
    record={"round":91,"study":result["study_id"],"title":"连续策略参考历史与新资金账户分开建账","status":status,"result":str((RESEARCH/"result.json").relative_to(ROOT)),**{k:result[k] for k in ["candidate_configurations","evaluation_accounts","new_accounts_generated","reused_control_accounts","earlier_diagnostic_accounts","new_earlier_diagnostic_accounts","new_model_fits","new_reference_accounts"]},"evaluated_candidate_source_runs":1,"primary_base":metric(result,PRIMARY),"primary_stress":metric(result,PRIMARY,cost="STRESS"),"post_selected_best_base":metric(result,PRIMARY),"historical_main_base_point_target_met":True,"full_goal_achieved":False}
    index["completed_rounds"].append(record)
    for key in ["registered_configurations_in_this_resumption","evaluated_configurations_in_this_resumption","evaluated_candidate_source_runs_including_corrected_replays","registered_candidate_source_runs_including_unrun_legacy_bindings"]:index[key]+=1
    index["evaluation_accounts_in_this_resumption"]+=result["evaluation_accounts"]
    index.update(updated_at=now(),latest_completed_round=record,running_studies=[],goal_achieved=False,status="ROUND91_MAIN_BASE_POINT_TARGET_MET_FULL_GOAL_NOT_MET",count_warning="91轮，365不同设置，381已评价来源版本，386登记含5旧未运行，1360主评价记录；无效来源保留。",next_work={"status":"CYCLE_RISK_AFTER_INACTIVE_REFERENCE_DIAGNOSTIC_NOT_REGISTERED","focus":"保存周期与不活跃参考日对风险预算的影响，查重后固定新机制","source":str(NEXT.relative_to(ROOT))},process_state_note="90和91已完成；91主基础点夏普1.233已核对，费用压力、较早历史及超额未过，完整目标保持未完成。")
    index["deliveries"].append({"created_at":now(),"type":"CONTINUOUS_REFERENCE_RISK_ROUND91_FAST_CHINESE_RESULTS","rounds":[91],"directory":str(OUT),"main_document":str(DOCUMENT),"new_gpt_review_archive_created":False})
    write_json(INDEX,index)
    write_json(OUT/"交付回执.json",{"created_at":now(),"main_document":str(DOCUMENT),"historical_main_base_point_target_met":True,"goal_achieved":False,"new_gpt_review_archive_created":False},exclusive=True)
    print(json.dumps({"交付":str(DOCUMENT),"核对":receipt,"验收":gates,"差额":differences,"集中":concentration},ensure_ascii=False),flush=True)


if __name__=="__main__":main()
