"""只核对月度风险目标和四个新账户，交付本轮结果。"""
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from research.two_policy_wealth_budget_v1 import ROOT, OUT as RESEARCH, CONFIG, PRIMARY, P46
from research.adaptive_allocation_v1 import normalize_dividends, summarize
from research.intraday_overnight_increment_v1 import now, require, write_json, digest
from scripts.review_round74_saved import saved_cycles
from scripts.finalize_round60_20260907 import metric, table

OUT = ROOT / "deliverables/510300累计净值预算_第84轮_20260908"
DOCUMENT = OUT / "累计净值预算_结果及中文规则.md"
INDEX = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
NEXT = ROOT / "docs/510300_AFTER_WEALTH_BUDGET_SYNTHETIC_FEAR_20260908.md"


def main():
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    require(index["latest_completed_round"]["round"] == 83 and not OUT.exists(), "第84轮前序或交付状态不符")
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    result = json.loads((RESEARCH / "result.json").read_text(encoding="utf-8"))
    require(not result["historical_point_target_met"], "达到目标后须另做完整验收")
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    checks, cycles, differences, concentration, metrics = [], [], [], [], []
    for period in ["evaluation", "earlier_diagnostic"]:
        f = pd.read_parquet(RESEARCH / f"{period}_factors.parquet")
        first = int(np.flatnonzero(f.date >= pd.Timestamp(cfg["evaluation_start"] if period == "evaluation" else cfg["earlier_start"]))[0])
        for model, column in [("PANIC_ONLY", "panic_reference_equity"), ("REARM_RIDGE", "learned_reference_equity")]:
            ref = pd.read_parquet(P46 / period / "BASE" / f"{model}_ledger.parquet")
            require(pd.DatetimeIndex(ref.date).equals(pd.DatetimeIndex(f.date.iloc[first:])), "原参考净值日历不符")
            np.testing.assert_allclose(f[column].iloc[first:], ref.equity, atol=0, rtol=0, equal_nan=True)
        weights, last = np.array([.5, .5]), pd.NaT
        for t in range(first-1, len(f)-1):
            row = f.iloc[t]
            expected_status = "INITIAL_EQUAL_REFERENCE_WEALTH"
            if t >= first:
                values = np.array([row.panic_reference_equity, row.learned_reference_equity])
                if np.isfinite(values).all() and (values > 0).all():
                    weights = values/values.sum()
                    expected_status, last = "REFERENCE_WEALTH_BUDGET_AVAILABLE", row.date
                else:
                    expected_status = "NO_VIEW_INVALID_REFERENCE_EQUITY_KEEP_BUDGET"
            require(row.budget_status == expected_status and row.budget_update_scheduled == (t >= first), "财富更新状态或时钟不符")
            if pd.notna(last):
                require(row.last_successful_budget_origin == last, "财富基准提前或滞后")
            np.testing.assert_allclose([row.panic_budget, row.learned_budget], weights, atol=1e-14, rtol=0)
            np.testing.assert_allclose([row.target], [np.array([row.panic_state, row.learned_state]) @ weights], atol=1e-14, rtol=0, equal_nan=True)
            checks.append({"period": period, "date": row.date, "status": row.budget_status, "panic_budget": weights[0], "learned_budget": weights[1]})
        for cost in cfg["costs"]:
            ledger = pd.read_parquet(RESEARCH / period / cost / f"{PRIMARY}_ledger.parquet")
            decisions = pd.read_parquet(RESEARCH / period / cost / f"{PRIMARY}_decisions.parquet")
            np.testing.assert_allclose(decisions.reference_weight, f.target.iloc[first-1:-1], atol=0, rtol=0, equal_nan=True)
            previous = np.r_[cfg["initial_capital"], ledger.equity.iloc[:-1]]
            np.testing.assert_allclose(ledger.cash+ledger.shares*ledger.mark+ledger.dividend_receivable, ledger.equity, atol=1e-6, rtol=0)
            np.testing.assert_allclose(ledger.equity/previous-1, ledger.net_return, atol=1e-12, rtol=0)
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
            old = pd.read_parquet(P46 / period / cost / "PANIC_LEARNED_HALF_ledger.parquet")
            differences.append({"period": period, "cost": cost, "terminal_nav_difference": float(ledger.equity.iloc[-1]-old.equity.iloc[-1]),
                **{f"{field}_difference": float(ledger[field].sum()-old[field].sum()) for field in ["price_pnl", "dividend_recognized", "commission", "slippage_cost"]}})
    for filename, rows in [("saved_budget_checks.csv", checks), ("saved_actual_cycles.csv", cycles), ("saved_account_checks.csv", metrics),
        ("saved_equal_budget_differences.csv", differences), ("saved_profit_concentration.csv", concentration)]:
        pd.DataFrame(rows).to_csv(RESEARCH / filename, index=False, encoding="utf-8-sig")
    receipt = {"verified_at": now(), "status": "KEY_REFERENCE_EQUITY_DAILY_BUDGET_AND_COMPLETE_ACCOUNTS_CHECKED", "daily_budget_checks": len(checks),
        "new_account_metrics_checked": len(metrics), "complete_cycles": len(cycles), "new_accounts_or_fits": 0,
        "reviewer_source_sha256": digest(Path(__file__)), "security_audit_performed": False}
    write_json(RESEARCH / "saved_verification_receipt.json", receipt, exclusive=True)
    status = "COMPLETED_CUMULATIVE_WEALTH_BUDGET_TARGET_NOT_MET"
    write_json(RESEARCH / "acceptance_outcome.json", {"recorded_at": now(), "status": status, "goal_achieved": False, "position_impact": 0,
        "decision": "两段两费用夏普均低于原固定各半，停止累计净值预算，不改指数、记忆长度、起始权重救回。"}, exclusive=True)
    NEXT.write_text("# 第84轮完成，下一项检验价格恐慌回落进入\n\n"
        "84主基础／压力夏普0.915103／0.846823，较早0.602187／0.571133；四项均低原各半0.944391／0.879003及0.604421／0.572113。主年化3.97835%、回撤5.65916%，较早4.28025%、回撤11.47046%。预算随累计净值变化有限：主急跌44.737%至50.752%、均值47.846%，早38.116%至52.249%、均值41.500%。没有证明有效识别策略失效，不修改此累计权重救回。四新账户及必要核对已经完成。\n\n"
        "下一85拟新增入场信息：Larry Williams价格型恐慌指标，最近22日最高含分红财富收盘相对当日同尺度最低价的距离占比。原文The VIX Fix（Active Trader，2007年12月）给出22日公式及波动带用法，但未给唯一交易系统；本项目自己的固定策略需另行登记。原文作者网站当前无法读取，已读取同一署名原文镜像https://www.marketcalls.in/wp-content/uploads/2014/12/VIXFix.pdf。不能说该指标是期权隐含波动或已验证市场底部。\n\n"
        "拟只做恐慌从自身20日均值加两倍样本标准差的上轨回落、同时含分红收盘回升时进入；退出沿用原急跌策略价格回到20日均值或4%损失、6%盈利、10日到期，冷却2日。没有固定旧5%跌幅和5/60波动倍数，属于另一种异常低点尺度，不调整原急跌阈值。22日基础公式包括当日收盘，低点按现有因果分红财富尺度转换。缺失无观点、下一开盘、完整账户不变。仅方向，尚未登记和运行。\n\n"
        "有界查重未见同一WVF实现；已找到旧三状态隐马尔可夫模型和旧两日RSI策略，跳过这些重复方向，没有新设置或账户计数。原78三项短期评分、旧VIX期限与原急跌失败保留。\n", encoding="utf-8")
    OUT.mkdir(parents=True)
    lines = ["# 第84轮：随累计净值变化的两策略预算", "",
        "主基础／压力净夏普0.915／0.847，较早0.602／0.571，四项均低于原固定各半。目标1.2未达，停止累计净值预算，不调整记忆长度或净值指数。", "",
        "## 主评价：2020年至2026年固定开盘终点", ""]+table(result["all_metrics"])
    lines += ["## 较早历史：2015年至2019年固定开盘终点", ""]+table(result["earlier_diagnostics"])
    lines += ["主基础年化3.98%、最大回撤5.66%；较早年化4.28%、回撤11.47%。较早年化高于原各半，但夏普略低、回撤更大，不能称为全面改善。", "",
        "主急跌预算在44.74%至50.75%之间，平均47.85%；较早38.12%至52.25%，平均41.50%。长期累计结果让权重缓慢漂移，本次没有形成足以改善表现的切换。主每档52笔、262个持仓收盘，早33笔、339个收盘；原参考净值1603及1218个有效更新，均无缺失或未成交。", "",
        "## 全部新增因子和交易规则", ""]+(ROOT/cfg["rules"]).read_text(encoding="utf-8").splitlines()[1:]
    lines += ["", f"六项必要测试通过，{len(checks)}个决策时点及四个完整账户、{len(cycles)}个含分红周期核对完成。原两策略全部中文因子和进入退出规则、逐月系数另附；没有重训或下载。下一价格恐慌指标只是方向。", ""]
    DOCUMENT.write_text("\n".join(lines), encoding="utf-8")
    for p in RESEARCH.iterdir():
        if p.is_file() and p.suffix in {".csv", ".json"}:
            shutil.copy2(p, OUT/p.name)
    shutil.copy2(CONFIG, OUT/"冻结设置.json")
    shutil.copy2(ROOT/"docs/510300_TWO_POLICY_RISK_BUDGET_V1.md", OUT/"沿用的两条策略全部中文规则.md")
    coeff = ROOT/"deliverables/510300下午提前进入_第75轮_20260908/沿用的每月八项模型中文规则.md"
    shutil.copy2(coeff, OUT/coeff.name)
    shutil.copy2(NEXT, OUT/"下一项研究方向.md")
    for period, label in [("evaluation", "主评价"), ("earlier_diagnostic", "较早历史")]:
        for cost in cfg["costs"]:
            for kind in ["ledger", "decisions"]:
                pd.read_parquet(RESEARCH/period/cost/f"{PRIMARY}_{kind}.parquet").to_csv(OUT/f"{label}_{cost}_{kind}.csv", index=False, encoding="utf-8-sig")
    record = {"round": 84, "study": result["study_id"], "title": "两条原策略累计净值自然预算", "status": status,
        "result": str((RESEARCH/"result.json").relative_to(ROOT)), **{k: result[k] for k in ["candidate_configurations", "evaluation_accounts", "new_accounts_generated", "reused_control_accounts", "earlier_diagnostic_accounts", "new_earlier_diagnostic_accounts", "new_model_fits", "new_reference_accounts"]},
        "evaluated_candidate_source_runs": 1, "primary_base": metric(result, PRIMARY), "primary_stress": metric(result, PRIMARY, cost="STRESS"), "post_selected_best_base": metric(result, PRIMARY)}
    index["completed_rounds"].append(record)
    for k in ["registered_configurations_in_this_resumption", "evaluated_configurations_in_this_resumption", "evaluated_candidate_source_runs_including_corrected_replays", "registered_candidate_source_runs_including_unrun_legacy_bindings"]:
        index[k] += 1
    index["evaluation_accounts_in_this_resumption"] += result["evaluation_accounts"]
    index.update(updated_at=now(), latest_completed_round=record, running_studies=[], goal_achieved=False, status="ROUND84_COMPLETE_WEALTH_BUDGET_TARGET_NOT_MET",
        count_warning="84轮，358不同设置，374已评价来源版本，379登记含5旧未运行，1284主评价记录。",
        next_work={"status": "SYNTHETIC_PRICE_FEAR_RECOVERY_NOT_REGISTERED", "focus": "价格恐慌越过自身上轨后回落进入", "source": str(NEXT.relative_to(ROOT))},
        process_state_note="84四新账户、必要核对及简洁交付完成；85价格恐慌只有方向。")
    index["deliveries"].append({"created_at": now(), "type": "WEALTH_BUDGET_ROUND84_FAST_CHINESE_RESULTS", "rounds": [84], "directory": str(OUT), "main_document": str(DOCUMENT), "new_gpt_review_archive_created": False})
    write_json(INDEX, index)
    write_json(OUT/"交付回执.json", {"created_at": now(), "main_document": str(DOCUMENT), "goal_achieved": False, "new_gpt_review_archive_created": False}, exclusive=True)
    print(json.dumps({"交付": str(DOCUMENT), "核对": receipt, "相对原各半差额": differences}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
