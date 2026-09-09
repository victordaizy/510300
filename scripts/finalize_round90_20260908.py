"""核对成熟目标、保存模型和完整账户，交付第90轮精简结果。"""
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from research.contextual_parent_advantage_v1 import ROOT, OUT as RESEARCH, CONFIG, PRIMARY, P82, P88
from research.contextual_parent_advantage_inputs_v1 import FEATURES, PARENTS
from research.adaptive_allocation_v1 import normalize_dividends, summarize
from research.intraday_overnight_increment_v1 import now, require, write_json, digest
from scripts.review_round74_saved import saved_cycles
from scripts.finalize_round60_20260907 import metric, table

OUT=ROOT/"deliverables/510300三因子选择组合_第90轮_20260908"
DOCUMENT=OUT/"三因子选择组合_结果及中文规则.md"
INDEX=ROOT/"reports/research/510300_sharpe_1_2_latest_research.json"
NEXT=ROOT/"docs/510300_AFTER_CONTEXTUAL_PARENT_CONTINUOUS_HISTORY_20260908.md"


def main():
    index=json.loads(INDEX.read_text(encoding="utf-8"))
    require(index["latest_completed_round"]["round"]==89 and not OUT.exists(),"第90轮前序或交付状态不符")
    cfg=json.loads(CONFIG.read_text(encoding="utf-8"));result=json.loads((RESEARCH/"result.json").read_text(encoding="utf-8"))
    require(not result["historical_point_target_met"],"达到目标须另做完整验收")
    dividends=normalize_dividends(pd.read_csv(ROOT/cfg["dividends"]))
    data=pd.read_parquet(ROOT/cfg["features"])
    checks,cycles,differences,concentration,metrics,predictions,coverage=[],[],[],[],[],[],[]
    label_count=0
    for period in ["evaluation","earlier_diagnostic"]:
        f=pd.read_parquet(RESEARCH/f"{period}_factors.parquet")
        labels=pd.read_parquet(RESEARCH/f"{period}_training_labels.parquet")
        models=json.loads((RESEARCH/f"{period}_saved_models.json").read_text(encoding="utf-8"))["models"]
        first=int(np.flatnonzero(f.date>=pd.Timestamp(cfg["evaluation_start"] if period=="evaluation" else cfg["earlier_start"]))[0])
        refs=f[["two_reference_return","three_reference_return"]].to_numpy()
        for number,(parent,model) in enumerate([(P82,PARENTS[0]),(P88,PARENTS[1])]):
            saved_parent=pd.read_parquet(parent/period/"BASE"/f"{model}_ledger.parquet")
            saved_decision=pd.read_parquet(parent/period/"BASE"/f"{model}_decisions.parquet")
            require(pd.DatetimeIndex(saved_parent.date).equals(pd.DatetimeIndex(f.date.iloc[first:])),"父账户日历不符")
            require(pd.DatetimeIndex(saved_decision.origin).equals(pd.DatetimeIndex(f.date.iloc[first-1:-1])),"父意向日历不符")
            prefix="two" if number==0 else "three"
            np.testing.assert_allclose(f[prefix+"_reference_return"].iloc[first:],saved_parent.net_return,atol=0,rtol=0)
            np.testing.assert_allclose(f[prefix+"_parent_target"].iloc[first-1:-1],saved_decision.reference_weight,atol=0,rtol=0,equal_nan=True)
        expected_origins=np.arange(first-1,len(f)-1)
        np.testing.assert_array_equal(labels.origin_index,expected_origins)
        for row in labels.itertuples():
            t=row.origin_index;end=t+cfg["target_horizon"]
            require(row.maturity_index==end and row.origin==f.date.iloc[t],"目标原点或成熟日错位")
            np.testing.assert_allclose([getattr(row,x) for x in FEATURES],data.loc[t,FEATURES].to_numpy(float),rtol=0,atol=0,equal_nan=True)
            if end<len(f)-1:
                value=np.prod(1+refs[t+1:end+1],axis=0)-1
                np.testing.assert_allclose([row.two_future_return,row.three_future_return,row.target],[value[0],value[1],value[1]-value[0]],rtol=0,atol=1e-14)
                require(row.maturity_date==f.date.iloc[end] and row.target_status=="COMPLETE_HISTORICAL_TARGET","成熟目标状态错误")
            else:
                require(pd.isna(row.target) and pd.isna(row.maturity_date) and row.target_status=="NO_VIEW_TERMINAL_OR_UNFINISHED_TARGET","终点开盘进入训练目标")
            label_count+=1
        model_at={int(r["fit_index"]):r for r in models}
        selected=0;last_origin=pd.NaT;last_pred=np.nan
        scheduled_count=0;model_count=0
        for t in range(first-1,len(f)-1):
            row=f.iloc[t];scheduled=t>=first and row.date.to_period("M")!=f.date.iloc[t-1].to_period("M");prior=selected
            require(bool(row.selection_update_scheduled)==scheduled,"月首训练时钟错误")
            if scheduled:
                scheduled_count+=1;r=model_at[t]
                expected_indices=np.arange(max(first-1,t-cfg["target_horizon"]-cfg["training_window"]+1),t-cfg["target_horizon"]+1)
                training=labels[labels.origin_index.isin(expected_indices)]
                require(r["training_rows"]==len(training) and len(training)==len(expected_indices),"成熟训练窗口借用评价之前或遗漏原点")
                if len(training):
                    require(r["latest_maturity_index"]==t and r["training_end_index"]==t-cfg["target_horizon"],"训练未严格等待标签成熟")
                last_pred=np.nan;selected=0;last_origin=row.date
                m=r["model"]
                if len(training)>=cfg["minimum_training_rows"]:
                    require(m is not None and r["status"]=="FIT_COMPLETE","实际完整训练窗口未生成模型")
                    x=training[FEATURES].to_numpy(float);y=training.target.to_numpy(float)
                    mean=x.mean(0);scale=x.std(0,ddof=0);scale=np.where(scale>1e-12,scale,1.)
                    z=np.clip((x-mean)/scale,-cfg["feature_clip"],cfg["feature_clip"])
                    a=np.column_stack([np.ones(len(z)),z]);coef=np.linalg.solve(a.T@a+np.diag([0.,cfg["ridge_alpha"],cfg["ridge_alpha"],cfg["ridge_alpha"]]),a.T@y)
                    np.testing.assert_allclose(m["mean"],mean,rtol=0,atol=1e-12)
                    np.testing.assert_allclose(m["scale"],scale,rtol=0,atol=1e-12)
                    np.testing.assert_allclose([m["intercept"],*m["coefficients"]],coef,rtol=0,atol=1e-10)
                    values=data.loc[t,FEATURES].to_numpy(float)
                    last_pred=float(coef[0]+np.clip((values-mean)/scale,-cfg["feature_clip"],cfg["feature_clip"])@coef[1:])
                    selected=int(last_pred>0);model_count+=1
                    np.testing.assert_allclose([r["prediction"]],[last_pred],rtol=0,atol=1e-10)
                    label=labels[labels.origin_index.eq(t)].iloc[0]
                    if pd.notna(label.target):
                        chosen=label.three_future_return if selected else label.two_future_return
                        other=label.two_future_return if selected else label.three_future_return
                        predictions.append({"period":period,"origin":row.date,"prediction":last_pred,"actual_relative_target":float(label.target),"chosen_future_relative":chosen-other,"target_maturity":label.maturity_date,"role":"SAVED_FUTURE_DIAGNOSTIC_NOT_TRAINING_AT_THIS_ORIGIN"})
                else:require(m is None and not r["fit_attempted"] and r["status"]=="NO_VIEW_INSUFFICIENT_MATURE_ROWS","原点不足未保留显式基线")
                require(r["selected_parent"]==PARENTS[selected],"模型预测与选择不同")
                checks.append({"period":period,"date":row.date,"status":r["status"],"training_rows":len(training),"latest_maturity_index":r["latest_maturity_index"],"prediction":last_pred,"selected_parent":PARENTS[selected]})
            require(row.selected_parent==PARENTS[selected] and bool(row.selection_changed)==(prior!=selected),"月内选择被改动")
            np.testing.assert_allclose([row.relative_prediction],[last_pred],rtol=0,atol=1e-10,equal_nan=True)
            require((pd.isna(last_origin) and pd.isna(row.prediction_origin)) or last_origin==row.prediction_origin,"月内显示未来或错误预测日期")
            expected=row.two_parent_target if selected==0 else row.three_parent_target
            np.testing.assert_allclose([row.target],[expected],rtol=0,atol=0,equal_nan=True)
        require(scheduled_count==len(models),"保存模型缺失或额外拟合")
        coverage.append({"period":period,"monthly_origins":scheduled_count,"completed_models":model_count,"no_view_models":scheduled_count-model_count,"two_selected_days":int(f.selected_parent.eq(PARENTS[0]).sum()),"three_selected_days":int(f.selected_parent.eq(PARENTS[1]).sum()),"selection_changes":int(f.selection_changed.sum()),"complete_labels":int(labels.target.notna().sum()),"zero_complete_labels":int(labels.target.eq(0).sum())})
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
            for control in ["TWO_POLICY_MIN_VARIANCE","THREE_POLICY_MIN_VARIANCE"]:
                old=pd.read_parquet(RESEARCH/period/cost/f"{control}_ledger.parquet")
                differences.append({"period":period,"cost":cost,"control":control,"terminal_nav_difference":float(ledger.equity.iloc[-1]-old.equity.iloc[-1]),
                    **{f"{field}_difference":float(ledger[field].sum()-old[field].sum()) for field in ["price_pnl","dividend_recognized","commission","slippage_cost"]}})
    for filename,rows in [("saved_monthly_model_checks.csv",checks),("saved_actual_cycles.csv",cycles),("saved_account_checks.csv",metrics),("saved_parent_differences.csv",differences),("saved_profit_concentration.csv",concentration),("saved_future_prediction_diagnostics.csv",predictions),("saved_model_coverage.csv",coverage)]:
        pd.DataFrame(rows).to_csv(RESEARCH/filename,index=False,encoding="utf-8-sig")
    receipt={"verified_at":now(),"status":"MATURE_TARGET_SAVED_COEFFICIENTS_OWN_REQUESTS_AND_COMPLETE_ACCOUNTS_CHECKED","target_rows_checked":label_count,"monthly_origins_checked":len(checks),"saved_models_checked":sum(c["completed_models"] for c in coverage),"new_account_metrics_checked":len(metrics),"complete_cycles":len(cycles),"new_accounts_or_fits":0,"reviewer_source_sha256":digest(Path(__file__)),"security_audit_performed":False}
    write_json(RESEARCH/"saved_verification_receipt.json",receipt,exclusive=True)
    status="COMPLETED_CONTEXTUAL_PARENT_ADVANTAGE_TARGET_NOT_MET"
    decision="三因子条件切换主夏普1.112、较早0.520，未达1.2；主相对原两策略只小幅多赚而波动上升，较早压力费用夏普下降。关闭该目标及模型，不调标签期限、训练窗、特征、符号或阈值救回。"
    write_json(RESEARCH/"acceptance_outcome.json",{"recorded_at":now(),"status":status,"decision":decision,"goal_achieved":False,"position_impact":0},exclusive=True)
    OUT.mkdir(parents=True)
    lines=["# 第90轮：三个市场因子选择组合", "",decision,"","## 主历史：2020年1月2日至2026年8月14日开盘",""]+table(result["all_metrics"])
    lines += ["## 较早历史：2015年1月5日至2019年12月31日开盘",""]+table(result["earlier_diagnostics"])
    lines += ["## 收益差额与切换是否有用", ""]
    for period,label in [("evaluation","主历史"),("earlier_diagnostic","较早历史")]:
        m=metric(result,PRIMARY,period);c=next(x for x in coverage if x["period"]==period)
        base_diff=next(d for d in differences if d["period"]==period and d["cost"]=="BASE" and d["control"]=="TWO_POLICY_MIN_VARIANCE")
        lines += [f"{label}基础年化{m['annualized_return']:.2%}、最大回撤{-m['max_drawdown']:.2%}；相对原两策略终值差额{base_diff['terminal_nav_difference']:+.2f}元。{c['monthly_origins']}次月首中{c['completed_models']}次有效训练、{c['no_view_models']}次训练原点不足；选择两策略{c['two_selected_days']}日、三策略{c['three_selected_days']}日，共{c['selection_changes']}次变更。", ""]
        lines += [f"完整历史目标{c['complete_labels']}行，其中{c['zero_complete_labels']}行相对收益恰好为零；该统计描述训练素材，不能把重叠目标当作独立试验。", ""]
        p=[x for x in predictions if x["period"]==period];v=np.array([x["chosen_future_relative"] for x in p])
        lines += [f"有效模型的完整后续二十日目标共{len(p)}段，所选组合{int((v>1e-12).sum())}段更好、{int((v < -1e-12).sum())}段更差、{int((abs(v)<=1e-12).sum())}段相同，平均相对另一组合{float(v.mean())*100:+.4f}个百分点。仅为事后诊断，不参与当时训练；二十日目标与实际月度持有区间存在天数差。", ""]
    lines += ["主历史基本保留原两策略的路径，没有可靠新增优势；较早基础费用的微小改善在费用压力下消失。结果不足以说明这三个因子能稳定识别下一阶段赢家。停止本轮模型，不继续围绕它调参。下一步先核对评价开始时的冷启动是否丢掉了当时已经存在的历史信息；这只是新方向，尚未登记或运行新策略。", "", "## 全部中文因子、进入和退出规则", ""]+(ROOT/cfg["rules"]).read_text(encoding="utf-8").splitlines()[1:]
    lines += ["",f"八项必要测试通过，112次拟合均完成。只复核{label_count}行目标的成熟边界、{len(checks)}个月首、112个保存模型的数值、四个自身现金与份额账户和{len(cycles)}个含分红完整周期，没有重跑账户或模型。主、较早均保留完整空仓日，所有规则最早下一开盘执行。", ""]
    DOCUMENT.write_text("\n".join(lines),encoding="utf-8")
    for p in RESEARCH.iterdir():
        if p.is_file() and p.suffix in {".csv",".json"} or p.is_file() and "每月三因子中文系数" in p.name:shutil.copy2(p,OUT/p.name)
    shutil.copy2(CONFIG,OUT/"冻结设置.json")
    shutil.copy2(ROOT/"docs/510300_TWO_POLICY_RISK_BUDGET_V1.md",OUT/"沿用的两条策略全部中文规则.md")
    coefficient=ROOT/"deliverables/510300下午提前进入_第75轮_20260908/沿用的每月八项模型中文规则.md"
    shutil.copy2(coefficient,OUT/coefficient.name)
    shutil.copy2(ROOT/"docs/510300_COMPRESSION_CONFIRMED_ENTRY_V1.md",OUT/"沿用的第三条突破全部中文规则.md")
    shutil.copy2(NEXT,OUT/"下一项研究方向.md")
    for period,label in [("evaluation","主评价"),("earlier_diagnostic","较早历史")]:
        for cost in cfg["costs"]:
            for kind in ["ledger","decisions"]:
                pd.read_parquet(RESEARCH/period/cost/f"{PRIMARY}_{kind}.parquet").to_csv(OUT/f"{label}_{cost}_{kind}.csv",index=False,encoding="utf-8-sig")
    record={"round":90,"study":result["study_id"],"title":"三个市场因子预测两父组合未来相对收益","status":status,"result":str((RESEARCH/"result.json").relative_to(ROOT)),**{k:result[k] for k in ["candidate_configurations","evaluation_accounts","new_accounts_generated","reused_control_accounts","earlier_diagnostic_accounts","new_earlier_diagnostic_accounts","new_model_fits","new_reference_accounts"]},"evaluated_candidate_source_runs":1,"primary_base":metric(result,PRIMARY),"primary_stress":metric(result,PRIMARY,cost="STRESS"),"post_selected_best_base":metric(result,PRIMARY)}
    index["completed_rounds"].append(record)
    for key in ["registered_configurations_in_this_resumption","evaluated_configurations_in_this_resumption","evaluated_candidate_source_runs_including_corrected_replays","registered_candidate_source_runs_including_unrun_legacy_bindings"]:index[key]+=1
    index["evaluation_accounts_in_this_resumption"]+=result["evaluation_accounts"]
    index.update(updated_at=now(),latest_completed_round=record,running_studies=[],goal_achieved=False,status="ROUND90_COMPLETE_CONTEXTUAL_PARENT_ADVANTAGE_TARGET_NOT_MET",count_warning="90轮，364不同设置，380已评价来源版本，385登记含5旧未运行，1352主评价记录；无效来源保留。",next_work={"status":"CONTINUOUS_HISTORY_INFORMATION_BOUNDARY_NOT_REGISTERED","focus":"核对评价开始前已存在的历史与策略冷启动边界","source":str(NEXT.relative_to(ROOT))},process_state_note="90的训练、四个完整账户、必要核对和中文交付完成；连续历史方向仅待查证，未登记新策略。")
    index["deliveries"].append({"created_at":now(),"type":"CONTEXTUAL_PARENT_ADVANTAGE_ROUND90_FAST_CHINESE_RESULTS","rounds":[90],"directory":str(OUT),"main_document":str(DOCUMENT),"new_gpt_review_archive_created":False})
    write_json(INDEX,index)
    write_json(OUT/"交付回执.json",{"created_at":now(),"main_document":str(DOCUMENT),"goal_achieved":False,"new_gpt_review_archive_created":False},exclusive=True)
    print(json.dumps({"交付":str(DOCUMENT),"核对":receipt,"父组合差额":differences,"集中":concentration},ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
