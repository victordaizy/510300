"""核对完整周期风险、成熟预算和四个真实账户，交付第92轮。"""
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from research.cycle_adverse_risk_v1 import ROOT, OUT as RESEARCH, CONFIG, PRIMARY, P91
from research.adaptive_allocation_v1 import normalize_dividends, summarize
from research.intraday_overnight_increment_v1 import now, require, write_json, digest
from scripts.review_round74_saved import saved_cycles
from scripts.finalize_round60_20260907 import metric, table
OUT=ROOT/"deliverables/510300周期本金风险预算_第92轮_20260908"
DOCUMENT=OUT/"周期本金风险预算_结果及中文规则.md"
INDEX=ROOT/"reports/research/510300_sharpe_1_2_latest_research.json"
NEXT=ROOT/"docs/510300_AFTER_CYCLE_RISK_ENTRY_CONTEXT_20260908.md"


def main():
    index=json.loads(INDEX.read_text(encoding="utf-8"));require(index["latest_completed_round"]["round"]==91 and not OUT.exists(),"第92轮前序或交付状态不符")
    cfg=json.loads(CONFIG.read_text(encoding="utf-8"));result=json.loads((RESEARCH/"result.json").read_text(encoding="utf-8"))
    require(not result["historical_point_target_met"],"点目标达到后另作完整验收")
    dividends=normalize_dividends(pd.read_csv(ROOT/cfg["dividends"]));data=pd.read_parquet(ROOT/cfg["features"])
    source=pd.read_parquet(ROOT/cfg["cycle_risks_source"]);full=pd.read_parquet(RESEARCH/"evaluation_factors.parquet")
    risk_checks,checks,cycles,differences,concentration,metrics,budget_coverage=[],[],[],[],[],[],[]
    for model in ["PANIC_ONLY","REARM_RIDGE"]:
        ledger=pd.read_parquet(P91/"continuous_references/BASE"/f"{model}_ledger.parquet")
        local=source[source.model.eq(model)]
        for c in local.itertuples():
            path=ledger[ledger.date.ge(c.entry_date)&ledger.date.le(c.exit_date)]
            buy=path.iloc[0];sell=path.iloc[-1];q=int(buy.filled_quantity)
            require(q>0 and sell.filled_quantity==-q and path.filled_quantity.gt(0).sum()==1 and path.filled_quantity.lt(0).sum()==1,"原周期没有单次买入全部卖出")
            principal=q*buy.fill_price+buy.commission
            events=dividends[dividends.record_date.ge(c.entry_date)&dividends.record_date.lt(c.exit_date)]
            maturity=max([pd.Timestamp(c.exit_date)]+events.ex_date.to_list())
            credit=[q*events.loc[events.ex_date.le(row.date),"cash_dividend_per_share"].sum() for row in path.itertuples()]
            value=q*path.mark.to_numpy()+np.array(credit);value[-1]=q*sell.fill_price-sell.commission+credit[-1]
            mae=max(0.,1-float(value.min())/principal)
            require(abs(principal-c.entry_capital)<1e-6 and maturity==c.maturity_date,"周期实际本金或权益成熟时点错误")
            net=q*sell.fill_price-sell.commission+q*events.cash_dividend_per_share.sum()-principal
            require(abs(net-c.cycle_net_profit)<1e-6,"周期含分红损益不符")
            np.testing.assert_allclose([mae],[c.observed_maximum_capital_loss],rtol=0,atol=1e-12)
            if c.terminal_exit:require(pd.isna(c.maximum_capital_loss) and c.status=="NO_VIEW_TERMINAL_CYCLE","强制终点周期进入风险训练")
            else:require(abs(mae-c.maximum_capital_loss)<1e-12 and c.status=="COMPLETE_CYCLE_RISK","自然周期风险缺失或不符")
            risk_checks.append({"model":model,"cycle":c.cycle,"entry_date":c.entry_date,"exit_date":c.exit_date,"maturity_date":maturity,"entry_capital":principal,"maximum_capital_loss":mae,"eligible_natural_cycle":not c.terminal_exit})
    ref_first=int(np.flatnonzero(data.date.ge(cfg["reference_start"]))[0])
    priors=np.array([cfg["prior_loss_fractions"][m] for m in ["PANIC_ONLY","REARM_RIDGE"]]);weights=np.array([priors[1],priors[0]])/priors.sum();risks=priors.copy()
    for t in range(ref_first-1,len(full)-1):
        row=full.iloc[t];scheduled=t>=ref_first and row.date.to_period("M")!=full.date.iloc[t-1].to_period("M")
        require(scheduled==bool(row.risk_update_scheduled),"周期预算月首时钟错误")
        if scheduled:
            counts=[];squares=[];maturities=[]
            for model,prior in zip(["PANIC_ONLY","REARM_RIDGE"],priors,strict=True):
                group=source[source.model.eq(model)&~source.terminal_exit&source.maturity_date.le(row.date)]
                a=group.maximum_capital_loss.to_numpy();require(np.isfinite(a).all(),"当前成熟周期资料应完整")
                counts.append(len(a));squares.append(float(a@a));maturities.append(group.maturity_date.max() if len(a) else pd.NaT)
            risks=np.sqrt((priors**2+np.array(squares))/(1+np.array(counts)))
            weights=np.array([risks[1],risks[0]])/risks.sum()
            np.testing.assert_array_equal([row.panic_mature_cycles,row.learned_mature_cycles],counts)
            np.testing.assert_allclose([row.panic_squared_loss_sum,row.learned_squared_loss_sum],squares,rtol=0,atol=1e-12)
            for actual,expected in zip([row.panic_latest_maturity,row.learned_latest_maturity],maturities,strict=True):
                require((pd.isna(actual) and pd.isna(expected)) or actual==expected,"最后成熟日期错误")
            require(all(pd.isna(m) or m<=row.date for m in maturities),"预算使用未来周期")
            checks.append({"date":row.date,"panic_mature_cycles":counts[0],"learned_mature_cycles":counts[1],"panic_cycle_risk":risks[0],"learned_cycle_risk":risks[1],"panic_budget":weights[0]})
        np.testing.assert_allclose([row.panic_cycle_risk,row.learned_cycle_risk],risks,rtol=0,atol=1e-12)
        np.testing.assert_allclose([row.panic_budget,row.learned_budget],weights,rtol=0,atol=1e-12)
        np.testing.assert_allclose([row.target],[weights@np.array([row.panic_state,row.learned_state])],rtol=0,atol=1e-12,equal_nan=True)
    for period in ["evaluation","earlier_diagnostic"]:
        f=pd.read_parquet(RESEARCH/f"{period}_factors.parquet")
        if period=="earlier_diagnostic":pd.testing.assert_frame_equal(f.iloc[:-1],full.iloc[:len(f)-1])
        first=int(np.flatnonzero(f.date.ge(cfg["evaluation_start"] if period=="evaluation" else cfg["earlier_start"]))[0])
        parent=pd.read_parquet(P91/f"{period}_factors.parquet")
        np.testing.assert_allclose(f[["panic_state","learned_state"]],parent[["panic_state","learned_state"]],rtol=0,atol=0,equal_nan=True)
        a=f.iloc[first-1:-1]
        budget_coverage.append({"period":period,"initial_panic_budget":float(a.panic_budget.iloc[0]),"mean_panic_budget":float(a.panic_budget.mean()),"minimum_panic_budget":float(a.panic_budget.min()),"maximum_panic_budget":float(a.panic_budget.max()),"monthly_origins":int(a.risk_update_scheduled.sum()),"status_counts":a[a.risk_update_scheduled].risk_status.value_counts().to_dict()})
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
            for control in ["TWO_POLICY_MIN_VARIANCE","CONTINUOUS_REFERENCE_MIN_VARIANCE"]:
                old=pd.read_parquet(RESEARCH/period/cost/f"{control}_ledger.parquet")
                differences.append({"period":period,"cost":cost,"control":control,"terminal_nav_difference":float(ledger.equity.iloc[-1]-old.equity.iloc[-1]),
                    **{f"{field}_difference":float(ledger[field].sum()-old[field].sum()) for field in ["price_pnl","dividend_recognized","commission","slippage_cost"]}})
    for filename,rows in [("saved_cycle_risk_checks.csv",risk_checks),("saved_monthly_risk_budget_checks.csv",checks),("saved_actual_cycles.csv",cycles),("saved_account_checks.csv",metrics),("saved_parent_differences.csv",differences),("saved_profit_concentration.csv",concentration),("saved_budget_coverage.csv",budget_coverage)]:
        pd.DataFrame(rows).to_csv(RESEARCH/filename,index=False,encoding="utf-8-sig")
    receipt={"verified_at":now(),"status":"COMPLETE_CYCLE_RISK_MATURITY_MONTHLY_BUDGET_AND_OWN_ACCOUNTS_CHECKED","reference_cycles_checked":len(risk_checks),"natural_reference_cycles":sum(x["eligible_natural_cycle"] for x in risk_checks),"monthly_budgets_checked":len(checks),"new_account_metrics_checked":len(metrics),"complete_evaluation_cycles":len(cycles),"new_accounts_or_fits":0,"reviewer_source_sha256":digest(Path(__file__)),"security_audit_performed":False}
    write_json(RESEARCH/"saved_verification_receipt.json",receipt,exclusive=True)
    status="COMPLETED_CYCLE_RISK_LOCAL_RETURN_AND_EARLIER_IMPROVEMENT_TARGET_NOT_MET"
    decision="周期本金风险预算主基础夏普0.916、压力0.849，较早0.584、0.550；主年化4.13%超过买持，较早夏普优于原91和82，存在局部改善，但没有同时得到稳定超额和1.2夏普。保留结果，不调先验、历史或风险定义救回。"
    write_json(RESEARCH/"acceptance_outcome.json",{"recorded_at":now(),"status":status,"decision":decision,"goal_achieved":False,"position_impact":0},exclusive=True)
    OUT.mkdir(parents=True)
    lines=["# 第92轮：用完整交易周期分配风险预算", "",decision,"","## 主历史：2020年1月2日至2026年8月14日开盘",""]+table(result["all_metrics"])
    lines += ["## 较早历史：2015年1月5日至2019年12月31日开盘",""]+table(result["earlier_diagnostics"])
    lines += ["## 实际发现与下一步", "","原急跌参考3210日中只有78个收盘持股，12个自然完整周期；学习参考648个收盘持股，33个自然周期，终点强制清仓的另一个周期不参与风险学习。急跌周期的最大本金损失均方根为3.97%，学习为2.85%；急跌最差一次损失9.34%，学习10.80%。这是截至全部历史的诊断统计，各月预算严格只使用当时已成熟的周期。", "","第91轮侧重完整日波动，主夏普1.233、年化3.36%、回撤2.31%；本轮侧重每次投入的实际损失，主年化升至4.13%，年化超额约0.51个百分点，但波动和回撤升高，夏普降至0.916，回撤为5.70%。较早夏普从第91轮0.431升至0.584，回撤略降为10.91%，年化3.85%仍比买持低约0.04个百分点。", "","这两种预算各有局部优势，当前证据说明仅换风险度量仍在收益与波动之间取舍。不能选2020年后的第91轮、再事后换成2015年的第92轮，拼成成功策略。下一步转向进入条件与市场状态的关系，先辨别两套原信号在什么条件下失效，再固定能用当时资料判断的新机制，不继续围绕预算参数搜索。", ""]
    for c in concentration:
        if c["cost"]=="BASE":
            label="主历史" if c["period"]=="evaluation" else "较早历史"
            lines += [f"{label}{c['cycles']}个完整交易周期、{c['losing_cycles']}个亏损周期，最盈利三周期合计占总净利润{c['largest_three_vs_net_profit']:.1%}。", ""]
    lines += ["## 全部中文因子和进入退出规则", ""]+(ROOT/cfg["rules"]).read_text(encoding="utf-8").splitlines()[1:]
    lines += ["",f"六项必要测试通过。四个新账户完成，零新参考账户、零模型拟合；只复核46个风险周期及权益成熟、159个月首预算、较早前缀、四账户自身请求与{len(cycles)}个完整周期。未重跑第91轮账户，也未改变历史评价范围。", ""]
    DOCUMENT.write_text("\n".join(lines),encoding="utf-8")
    for p in RESEARCH.iterdir():
        if p.is_file() and p.suffix in {".csv",".json"}:shutil.copy2(p,OUT/p.name)
    shutil.copy2(ROOT/cfg["cycle_risks_source"],OUT/"完整参考周期风险.parquet")
    shutil.copy2((ROOT/cfg["cycle_risks_source"]).with_suffix(".csv"),OUT/"完整参考周期风险.csv")
    shutil.copy2(CONFIG,OUT/"冻结设置.json");shutil.copy2(ROOT/"docs/510300_TWO_POLICY_RISK_BUDGET_V1.md",OUT/"沿用的两条策略全部中文规则.md")
    coefficient=ROOT/"deliverables/510300下午提前进入_第75轮_20260908/沿用的每月八项模型中文规则.md";shutil.copy2(coefficient,OUT/coefficient.name)
    shutil.copy2(NEXT,OUT/"下一项研究方向.md")
    for period,label in [("evaluation","主评价"),("earlier_diagnostic","较早历史")]:
        for cost in cfg["costs"]:
            for kind in ["ledger","decisions"]:pd.read_parquet(RESEARCH/period/cost/f"{PRIMARY}_{kind}.parquet").to_csv(OUT/f"{label}_{cost}_{kind}.csv",index=False,encoding="utf-8-sig")
    record={"round":92,"study":result["study_id"],"title":"按已成熟完整周期的最大本金损失分预算","status":status,"result":str((RESEARCH/"result.json").relative_to(ROOT)),**{k:result[k] for k in ["candidate_configurations","evaluation_accounts","new_accounts_generated","reused_control_accounts","earlier_diagnostic_accounts","new_earlier_diagnostic_accounts","new_model_fits","new_reference_accounts"]},"evaluated_candidate_source_runs":1,"primary_base":metric(result,PRIMARY),"primary_stress":metric(result,PRIMARY,cost="STRESS"),"post_selected_best_base":metric(result,PRIMARY)}
    index["completed_rounds"].append(record)
    for key in ["registered_configurations_in_this_resumption","evaluated_configurations_in_this_resumption","evaluated_candidate_source_runs_including_corrected_replays","registered_candidate_source_runs_including_unrun_legacy_bindings"]:index[key]+=1
    index["evaluation_accounts_in_this_resumption"]+=result["evaluation_accounts"]
    index.update(updated_at=now(),latest_completed_round=record,running_studies=[],goal_achieved=False,status="ROUND92_LOCAL_RETURN_AND_EARLIER_IMPROVEMENT_FULL_GOAL_NOT_MET",count_warning="92轮，366不同设置，382已评价来源版本，387登记含5旧未运行，1368主评价记录；无效来源保留。",next_work={"status":"ENTRY_CONTEXT_DIAGNOSTIC_NOT_REGISTERED","focus":"转向进入信号与市场状态，先核对旧自适应研究与成熟样本覆盖","source":str(NEXT.relative_to(ROOT))},process_state_note="92训练输入、四账户及必要核对和中文交付完成；91主基础1.233局部候选继续保留，完整目标未完成。")
    index["deliveries"].append({"created_at":now(),"type":"CYCLE_ADVERSE_RISK_ROUND92_FAST_CHINESE_RESULTS","rounds":[92],"directory":str(OUT),"main_document":str(DOCUMENT),"new_gpt_review_archive_created":False})
    write_json(INDEX,index);write_json(OUT/"交付回执.json",{"created_at":now(),"main_document":str(DOCUMENT),"goal_achieved":False,"new_gpt_review_archive_created":False},exclusive=True)
    print(json.dumps({"交付":str(DOCUMENT),"核对":receipt,"差额":differences,"集中":concentration},ensure_ascii=False),flush=True)


if __name__=="__main__":main()
