"""只核对月度风险目标和四个新账户，交付本轮结果。"""
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from research.past_sharpe_parent_selector_v1 import ROOT, OUT as RESEARCH, CONFIG, PRIMARY, P82, P88
from research.adaptive_allocation_v1 import normalize_dividends, summarize
from research.intraday_overnight_increment_v1 import now, require, write_json, digest
from scripts.review_round74_saved import saved_cycles
from scripts.finalize_round60_20260907 import metric, table

OUT = ROOT / "deliverables/510300过去表现选择组合_第89轮_20260908"
DOCUMENT = OUT / "过去表现选择组合_结果及中文规则.md"
INDEX = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
NEXT = ROOT / "docs/510300_AFTER_PARENT_SELECTOR_CONTEXTUAL_ADVANTAGE_20260908.md"


def main():
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    require(index["latest_completed_round"]["round"] == 88 and not OUT.exists(), "第89轮前序或交付状态不符")
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    result = json.loads((RESEARCH / "result.json").read_text(encoding="utf-8"))
    require(not result["historical_point_target_met"], "达到目标后须另做完整验收")
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    checks, cycles, differences, concentration, metrics, next_intervals = [], [], [], [], [], []
    for period in ["evaluation", "earlier_diagnostic"]:
        f = pd.read_parquet(RESEARCH / f"{period}_factors.parquet")
        first = int(np.flatnonzero(f.date >= pd.Timestamp(cfg["evaluation_start"] if period == "evaluation" else cfg["earlier_start"]))[0])
        names=["TWO_POLICY_MIN_VARIANCE","THREE_POLICY_MIN_VARIANCE"]
        for number,(parent,model) in enumerate([(P82,names[0]),(P88,names[1])]):
            saved_parent=pd.read_parquet(parent/period/"BASE"/f"{model}_ledger.parquet")
            saved_decision=pd.read_parquet(parent/period/"BASE"/f"{model}_decisions.parquet")
            require(pd.DatetimeIndex(saved_parent.date).equals(pd.DatetimeIndex(f.date.iloc[first:])),"父组合完整日期不符")
            require(pd.DatetimeIndex(saved_decision.origin).equals(pd.DatetimeIndex(f.date.iloc[first-1:-1])),"父组合意向日期不符")
            prefix="two" if number==0 else "three"
            np.testing.assert_allclose(f[prefix+"_reference_return"].iloc[first:],saved_parent.net_return,atol=0,rtol=0)
            np.testing.assert_allclose(f[prefix+"_parent_target"].iloc[first-1:-1],saved_decision.reference_weight,atol=0,rtol=0,equal_nan=True)
        selected=names[0]
        for t in range(first-1,len(f)-1):
            row=f.iloc[t];scheduled=t>=first and row.date.to_period("M")!=f.date.iloc[t-1].to_period("M")
            require(scheduled==row.selection_update_scheduled,"选择发生在非月首或遗漏月首")
            prior=selected
            if scheduled:
                count=min(cfg["selection_window"],t-first+1)
                values=f.iloc[t-count+1:t+1][["two_reference_return","three_reference_return"]].to_numpy()
                require(count==row.selection_window_observations and row.selection_window_start==f.date.iloc[t-count+1],"选择借用评价之前或未来收益")
                scores=np.full(2,np.nan)
                if count<cfg["selection_window"]:status="NO_VIEW_WARMUP_KEEP_SELECTION"
                elif not np.isfinite(values).all():status="NO_VIEW_INCOMPLETE_WINDOW_KEEP_SELECTION"
                elif (values.std(axis=0,ddof=1)<=0).any():status="NO_VIEW_ZERO_OR_INVALID_VOLATILITY_KEEP_SELECTION"
                else:
                    scores=values.mean(axis=0)/values.std(axis=0,ddof=1)*np.sqrt(cfg["annual_days"])
                    if max(scores)<=0:selected="CASH";status="EXPLICIT_CASH_NONPOSITIVE_PAST_SHARPE"
                    elif scores[0]==scores[1]:
                        selected=prior if prior in names else names[0];status="EQUAL_POSITIVE_SCORE_KEEP_PARENT_OR_TWO_AFTER_CASH"
                    else:selected=names[int(np.argmax(scores))];status="POSITIVE_PAST_SHARPE_PARENT_SELECTED"
                np.testing.assert_allclose([row.two_past_sharpe,row.three_past_sharpe],scores,atol=1e-12,rtol=0,equal_nan=True)
                require(status==row.selection_status,"过去评分选择状态不符")
                checks.append({"period":period,"date":row.date,"status":status,"selected_parent":selected,"observations":count,
                    "two_past_sharpe":scores[0],"three_past_sharpe":scores[1],"changed":selected!=prior})
            require(selected==row.selected_parent and bool(row.selection_changed)==(selected!=prior),"选择在月首之外变更或未按历史评分执行")
            expected=0. if selected=="CASH" else row.two_parent_target if selected==names[0] else row.three_parent_target
            np.testing.assert_allclose([row.target],[expected],atol=0,rtol=0,equal_nan=True)
        scheduled=np.flatnonzero(f.selection_update_scheduled.to_numpy())
        for left,right in zip(scheduled[:-1],scheduled[1:]):
            row=f.iloc[left]
            if row.selection_status not in {"POSITIVE_PAST_SHARPE_PARENT_SELECTED","EQUAL_POSITIVE_SCORE_KEEP_PARENT_OR_TWO_AFTER_CASH"}:continue
            future=f.iloc[left+1:right+1][["two_reference_return","three_reference_return"]].to_numpy()
            realized=np.prod(1+future,axis=0)-1;chosen=names.index(row.selected_parent)
            next_intervals.append({"period":period,"selection_origin":row.date,"future_start":f.date.iloc[left+1],"future_end":f.date.iloc[right],
                "selected_parent":row.selected_parent,"chosen_future_return":realized[chosen],"other_future_return":realized[1-chosen],
                "future_relative_return":realized[chosen]-realized[1-chosen],"role":"DIAGNOSTIC_FUTURE_OUTCOME_NEVER_SELECTION_INPUT"})
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
    for filename, rows in [("saved_selection_checks.csv", checks), ("saved_actual_cycles.csv", cycles), ("saved_account_checks.csv", metrics),
        ("saved_risk_budget_differences.csv", differences), ("saved_profit_concentration.csv", concentration), ("saved_next_interval_selection_diagnostic.csv", next_intervals)]:
        pd.DataFrame(rows).to_csv(RESEARCH / filename, index=False, encoding="utf-8-sig")
    receipt = {"verified_at": now(), "status": "KEY_PAST_SELECTION_CLOCK_OWN_REQUESTS_AND_COMPLETE_ACCOUNTS_CHECKED", "selection_update_checks": len(checks), "future_diagnostic_intervals": len(next_intervals),
        "new_account_metrics_checked": len(metrics), "complete_cycles": len(cycles), "new_accounts_or_fits": 0,
        "reviewer_source_sha256": digest(Path(__file__)), "security_audit_performed": False}
    write_json(RESEARCH / "saved_verification_receipt.json", receipt, exclusive=True)
    status = "COMPLETED_PAST_SHARPE_SELECTION_TARGET_NOT_MET"
    write_json(RESEARCH / "acceptance_outcome.json", {"recorded_at": now(), "status": status, "goal_achieved": False, "position_impact": 0,
        "decision": "过去净夏普排名未能同时保留两父组合优势，主0.713早0.624，未达1.2；不扫描排名窗口、分差阈值、确认或候选池救回。"}, exclusive=True)
    OUT.mkdir(parents=True)
    lines=["# 第89轮：依据过去表现选择组合的结果", "", "主基础／压力净夏普0.713／0.653，较早0.624／0.581，没有达到1.2。主不及原两策略预算的1.137／1.085，较早不及原三策略预算的0.727／0.677。选择过去一年表现更好的规则，没能同时保留两套组合在不同历史阶段的优势。", "",
        "## 主评价：2020年1月2日至2026年8月14日开盘", ""]+table(result["all_metrics"])
    lines += ["## 较早历史：2015年1月5日至2019年12月31日开盘", ""]+table(result["earlier_diagnostics"])
    lines += ["主基础年化2.81%、最大回撤5.96%；较早年化4.32%、回撤11.07%。较早净利润比两父组合都高，分别多8384.90元和3718.34元，但波动更高、夏普低于三策略预算，因此不能将结果描述成每项指标都恶化。主基础相对两策略少赚18945.23元。", "",
        "主10次选择变更，选择两策略987个判断日、三策略456日、现金161日；较早9次变更，两策略448日、三策略528日、现金243日。主57笔成交、251个持仓收盘；较早46笔、403个持仓收盘，两档费用均无整笔未成交或缺失股票目标。", "",
        "主80个月首中60次正评分选择、12次窗口不足、6次父组合零波动、2次明确现金；较早60个月首中36次正评分、12次不足、12次现金。无新评分时沿用选择，与明确现金退出分别记录。", "",
        "## 失败原因：过去赢家没有稳定延续", "",
        "仅检查完成的后续月度区间，主59段中选中者16次更好、19次更差、24次相同，平均落后另一套父组合0.2553个百分点；较早35段12次更好、11次更差、12次相同，平均领先0.1379个百分点。相同区间包含相同持仓或都空仓，不能当成额外独立成功。该统计仅作保存结果诊断，未来收益绝不参与过去选择。", "",
        "这次检验不支持用固定过去夏普排名识别下一阶段赢家。它不否定所有条件切换，但要求新方法真正检验当前环境对后续相对收益的预测能力；不能把两个时期的事后赢家拼接为成功策略。本排名规则关闭，不反转排名或调整窗口挽救。", "",
        "## 全部中文选择与进入退出规则", ""]+(ROOT/cfg["rules"]).read_text(encoding="utf-8").splitlines()[1:]
    lines += ["", f"七项必要测试通过；核对{len(checks)}次月度评分和选择、四账户全部实际股数请求、指标及{len(cycles)}个完整含分红周期，并保存{len(next_intervals)}个后续区间诊断。下一项条件相对收益学习只有方向，尚未登记或运行。", ""]
    DOCUMENT.write_text("\n".join(lines), encoding="utf-8")
    for p in RESEARCH.iterdir():
        if p.is_file() and p.suffix in {".csv", ".json"}:
            shutil.copy2(p, OUT / p.name)
    shutil.copy2(CONFIG, OUT / "冻结设置.json")
    shutil.copy2(ROOT / "docs/510300_TWO_POLICY_RISK_BUDGET_V1.md", OUT / "沿用的两条策略全部中文规则.md")
    coefficient = ROOT / "deliverables/510300下午提前进入_第75轮_20260908/沿用的每月八项模型中文规则.md"
    shutil.copy2(coefficient, OUT / coefficient.name)
    shutil.copy2(ROOT / "docs/510300_COMPRESSION_CONFIRMED_ENTRY_V1.md", OUT / "沿用的第三条突破全部中文规则.md")
    shutil.copy2(NEXT, OUT / "下一项研究方向.md")
    for period, label in [("evaluation", "主评价"), ("earlier_diagnostic", "较早历史")]:
        for cost in cfg["costs"]:
            for kind in ["ledger", "decisions"]:
                pd.read_parquet(RESEARCH / period / cost / f"{PRIMARY}_{kind}.parquet").to_csv(OUT / f"{label}_{cost}_{kind}.csv", index=False, encoding="utf-8-sig")
    record = {"round": 89, "study": result["study_id"], "title": "根据过去净夏普选择两父组合或现金", "status": status,
        "result": str((RESEARCH / "result.json").relative_to(ROOT)), **{k: result[k] for k in ["candidate_configurations", "evaluation_accounts", "new_accounts_generated", "reused_control_accounts", "earlier_diagnostic_accounts", "new_earlier_diagnostic_accounts", "new_model_fits", "new_reference_accounts"]},
        "evaluated_candidate_source_runs": 1, "primary_base": metric(result, PRIMARY), "primary_stress": metric(result, PRIMARY, cost="STRESS"), "post_selected_best_base": metric(result, PRIMARY)}
    index["completed_rounds"].append(record)
    for key in ["registered_configurations_in_this_resumption", "evaluated_configurations_in_this_resumption", "evaluated_candidate_source_runs_including_corrected_replays", "registered_candidate_source_runs_including_unrun_legacy_bindings"]:
        index[key] += 1
    index["evaluation_accounts_in_this_resumption"] += result["evaluation_accounts"]
    index.update(updated_at=now(), latest_completed_round=record, running_studies=[], goal_achieved=False, status="ROUND89_COMPLETE_PAST_SHARPE_SELECTION_TARGET_NOT_MET",
        count_warning="89轮，363不同设置，379已评价来源版本，384登记含5旧未运行，1342主评价记录；无效来源保留。",
        next_work={"status": "CONTEXTUAL_PARENT_RELATIVE_RETURN_NOT_REGISTERED", "focus": "根据当前市场状态学习已成熟父组合未来相对收益", "source": str(NEXT.relative_to(ROOT))},
        process_state_note="88和89完整账户、关键核对及中文交付完成；90条件相对收益学习仅方向。")
    index["deliveries"].append({"created_at": now(), "type": "PAST_SHARPE_SELECTOR_ROUND89_FAST_CHINESE_RESULTS", "rounds": [89], "directory": str(OUT), "main_document": str(DOCUMENT), "new_gpt_review_archive_created": False})
    write_json(INDEX, index)
    write_json(OUT / "交付回执.json", {"created_at": now(), "main_document": str(DOCUMENT), "goal_achieved": False, "new_gpt_review_archive_created": False}, exclusive=True)
    print(json.dumps({"交付": str(DOCUMENT), "核对": receipt, "相对82差额": differences, "集中": concentration}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
