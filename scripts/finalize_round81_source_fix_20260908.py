"""检查参考成交时钟、分批阶段和实际账户，交付简洁结果。"""
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from research.staged_entry_v1_1 import ROOT, OUT as RESEARCH, CONFIG, PRIMARY, EXECUTION_CONTROL, P32, COEFFICIENTS, ORIGINAL
from research.adaptive_allocation_v1 import normalize_dividends, summarize
from research.intraday_overnight_increment_v1 import now, require, write_json, digest
from scripts.review_round74_saved import saved_cycles
from scripts.finalize_round60_20260907 import metric, table

OUT = ROOT / "deliverables/510300价格确认分批进入_第81轮_20260908"
DOCUMENT = OUT / "分批进入_结果及全部中文规则.md"
INDEX = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
NEXT = ROOT / "docs/510300_AFTER_STAGED_BUDGET_EXECUTION_20260908.md"


def main():
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    require(index["latest_completed_round"]["round"] == 80 and not OUT.exists(), "81前序或交付状态不符")
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    result = json.loads((RESEARCH / "result.json").read_text(encoding="utf-8"))
    require(not result["historical_point_target_met"], "出现目标值时不能直接写未达标")
    data = pd.read_parquet(ROOT / cfg["features"])
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    stages, cycles, metrics, differences = [], [], [], []
    for period in ["evaluation", "earlier_diagnostic"]:
        f = pd.read_parquet(RESEARCH / f"{period}_factors.parquet")
        frame = data.iloc[:len(f)]
        first = int(np.flatnonzero(frame.date >= pd.Timestamp(cfg["evaluation_start"] if period == "evaluation" else cfg["earlier_start"]))[0])
        reference = pd.read_parquet(P32 / period / "BASE/REARM_RIDGE_ledger.parquet")
        require(pd.DatetimeIndex(reference.date).equals(pd.DatetimeIndex(frame.date.iloc[first:])), "参考账户日历不符")
        ref_decisions = pd.read_parquet(P32 / period / "BASE/REARM_RIDGE_decisions.parquet")
        state_map = ref_decisions.set_index("origin").reference_weight
        np.testing.assert_allclose(f.reference_state.iloc[first-1:-1], state_map.reindex(f.date.iloc[first-1:-1]).to_numpy(), atol=0, rtol=0, equal_nan=True)
        anchor, entry_date = np.nan, pd.NaT
        for t, row in enumerate(reference.itertuples(), first):
            new_buy = row.shares_before == 0 and row.filled_quantity > 0
            if new_buy:
                anchor = frame.wealth.iloc[t-1]*(frame.open.iloc[t]+frame.dividend.iloc[t])/frame.previous_close.iloc[t]
                entry_date = frame.date.iloc[t]
            if row.shares == 0:
                anchor, entry_date = np.nan, pd.NaT
            np.testing.assert_allclose([f.reference_entry_wealth_open.iloc[t]], [anchor], atol=1e-12, rtol=0, equal_nan=True)
            require(bool(f.reference_new_buy.iloc[t]) == new_buy and bool(f.reference_holding.iloc[t]) == (row.shares > 0), "分批参考成交或持仓标记不符")
            if pd.notna(entry_date):
                require(f.reference_entry_date.iloc[t] == entry_date and entry_date <= f.date.iloc[t], "参考开盘价提前使用")
        stage = 0.
        for t in range(first-1, len(f)-1):
            row = f.iloc[t]
            if row.reference_new_buy:
                stage = .5
            expected, upgrade = np.nan, False
            if np.isfinite(row.reference_state) and np.isfinite(row.wealth_close):
                if row.reference_state == 0:
                    stage, expected = 0., 0.
                else:
                    stage = .5 if stage == 0 else stage
                    if stage == .5 and row.reference_holding and np.isfinite(row.reference_entry_wealth_open) and row.wealth_close > row.reference_entry_wealth_open:
                        stage, upgrade = 1., True
                    expected = stage
            np.testing.assert_allclose([row.target, row.stage], [expected, stage], atol=0, rtol=0, equal_nan=True)
            require(bool(row.price_upgrade) == upgrade, "确认补足标记不符")
            stages.append({"period": period, "origin": row.date, "target": expected, "upgrade": upgrade})
        for cost in cfg["costs"]:
            folder = RESEARCH / period / cost
            for model in [PRIMARY, EXECUTION_CONTROL]:
                ledger = pd.read_parquet(folder / f"{model}_ledger.parquet")
                decisions = pd.read_parquet(folder / f"{model}_decisions.parquet")
                expected = f.target.iloc[first-1:-1] if model == PRIMARY else f.reference_state.iloc[first-1:-1]
                np.testing.assert_allclose(decisions.reference_weight, expected, atol=0, rtol=0, equal_nan=True)
                previous = np.r_[cfg["initial_capital"], ledger.equity.iloc[:-1].to_numpy()]
                np.testing.assert_allclose(ledger.cash+ledger.shares*ledger.mark+ledger.dividend_receivable, ledger.equity, atol=1e-6, rtol=0)
                np.testing.assert_allclose(ledger.equity/previous-1, ledger.net_return, atol=1e-12, rtol=0)
                actual, saved = summarize(ledger, cfg), metric(result, model, period, cost)
                for field in ["net_sharpe", "annualized_return", "max_drawdown", "cumulative_return", "trade_count", "commission", "slippage_cost"]:
                    require(abs(actual[field]-saved[field]) < 1e-9, "分批或执行对照保存指标不符")
                metrics.append({"period": period, "cost": cost, "model": model, "days": len(ledger), "net_sharpe": actual["net_sharpe"]})
                cycles.extend({"period": period, "cost": cost, "model": model, **c} for c in saved_cycles(ledger, dividends, cfg))
            for left, right in [(PRIMARY, EXECUTION_CONTROL), (EXECUTION_CONTROL, "REARM_RIDGE")]:
                left_ledger = pd.read_parquet(folder / f"{left}_ledger.parquet")
                right_ledger = pd.read_parquet(folder / f"{right}_ledger.parquet")
                differences.append({"period": period, "cost": cost, "left": left, "right": right,
                    "terminal_nav_difference": float(left_ledger.equity.iloc[-1]-right_ledger.equity.iloc[-1]),
                    **{f"{c}_difference": float(left_ledger[c].sum()-right_ledger[c].sum()) for c in ["price_pnl", "dividend_recognized", "commission", "slippage_cost"]}})
    for name, rows in [("saved_stage_checks.csv", stages), ("saved_actual_cycles.csv", cycles), ("saved_account_checks.csv", metrics), ("saved_profit_differences.csv", differences)]:
        pd.DataFrame(rows).to_csv(RESEARCH / name, index=False, encoding="utf-8-sig")
    receipt = {"verified_at": now(), "status": "KEY_REFERENCE_CLOCK_STAGES_AND_COMPLETE_ACCOUNTS_CHECKED", "stage_origins": len(stages),
        "saved_account_metrics_recomputed": len(metrics), "corrected_accounts": 4, "reused_execution_control_accounts": 4, "complete_cycles": len(cycles), "new_accounts_or_models": 0,
        "reviewer_source_sha256": digest(Path(__file__)), "security_audit_performed": False}
    write_json(RESEARCH / "saved_verification_receipt.json", receipt, exclusive=True)
    status = "COMPLETED_SOURCE_CORRECTED_STAGING_TARGET_NOT_MET"
    old_result = json.loads((ORIGINAL / "result.json").read_text(encoding="utf-8"))
    source_differences = []
    for period in ["evaluation", "earlier_diagnostic"]:
        for cost in cfg["costs"]:
            require(digest(RESEARCH / period / cost / f"{EXECUTION_CONTROL}_ledger.parquet") == digest(ORIGINAL / period / cost / f"{EXECUTION_CONTROL}_ledger.parquet"), "原满仓对照字节改变")
            old_m, fixed_m = metric(old_result, PRIMARY, period, cost), metric(result, PRIMARY, period, cost)
            source_differences.append({"period": period, "cost": cost, "invalid_source_sharpe": old_m["net_sharpe"],
                "corrected_source_sharpe": fixed_m["net_sharpe"], "terminal_nav_correction": cfg["initial_capital"]*(fixed_m["cumulative_return"]-old_m["cumulative_return"])})
    pd.DataFrame(source_differences).to_csv(RESEARCH / "source_correction_impact.csv", index=False, encoding="utf-8-sig")
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "来源修正运行后冻结文件改变")
    write_json(RESEARCH / "acceptance_outcome.json", {"status": status, "recorded_at": now(), "goal_achieved": False,
        "decision": "修正实际持仓读取后，两段两费用夏普均低于同执行满仓；停止分批进入，不再投入尚未登记的组合分批检验，也不调比例或确认门槛。", "position_impact": 0}, exclusive=True)
    write_json(ORIGINAL / "acceptance_outcome.json", {"recorded_at": now(), "status": "INVALID_STAGING_SOURCE_REPLACED_BY_VERSION_2",
        "original_result_preserved": True, "original_rules_not_validly_evaluated": True, "valid_control_ledgers_reused": 4,
        "corrected_result": str((RESEARCH / "result.json").relative_to(ROOT)), "goal_achieved": False}, exclusive=True)
    OUT.mkdir(parents=True)
    main_base = metric(result, PRIMARY)
    early_base = metric(result, PRIMARY, "earlier_diagnostic")
    lines = ["# 第81轮：价格确认分批进入，来源修正后的结果", "",
        "修正后主评价基础／压力净夏普0.590／0.522，较早历史0.658／0.635，四项都低于同执行满仓对照，未达1.2。停止这项分批进入，不再追加尚未登记的组合分批试验，转向独立的两策略协同风险预算。", "",
        "首次曾报主0.712／0.646、较早0.697／0.677，是错误来源实现的结果，不能评价原冻结规则。代码把无成交日缺失的成交后股数当成每日实际持仓，提前清除了入场基准；核对在写验收和总索引前发现问题。修正只读取完整的每日实际股数，原策略、首批比例和确认门槛均不改变。首次六项合成测试没有模拟无成交日字段缺失；已新增真实结构回归测试，七项通过。原来源与原数字保留且标为无效分批实现。", "",
        "## 主评价：2020年1月2日至2026年8月14日开盘", ""] + table(result["all_metrics"])
    lines += ["## 较早历史：2015年1月5日至2019年12月31日开盘", ""] + table(result["earlier_diagnostics"])
    lines += ["主24参考周期中22次确认补足，半仓84个目标日、满仓169日；较早9周期全部确认，半仓20日、满仓279日。主每档46买24卖，较早18买9卖。实际持仓收盘仍为252和299，均无未成交；主非零目标253日比实际持仓多1，来自终点统一开盘退出。", "",
        f"基础主年化{main_base['annualized_return']:.2%}、最大回撤{abs(main_base['max_drawdown']):.2%}；较早年化{early_base['annualized_return']:.2%}、回撤{abs(early_base['max_drawdown']):.2%}。两段收益和夏普均低于同执行满仓，不能把较低投入或回撤当成已经改善效率。", "",
        f"修正只新跑4个受影响账户；4个满仓对照账本原字节复用，另16个老对照复用。核对{len(stages)}个决策时点、8份保存账户指标与{len(cycles)}个含分红周期，未重训。两来源合计12个实际新账户，一个不同设置、两个已运行来源版本、24份主评价记录。", "",
        "## 进出场和全部因子中文规则", ""]
    lines += (ROOT / cfg["rules"]).read_text(encoding="utf-8").splitlines()[1:]
    lines += ["", "## 本次来源修正的优先说明", ""]
    lines += (ROOT / cfg["source_amendment"]).read_text(encoding="utf-8").splitlines()[1:]
    DOCUMENT.write_text("\n".join(lines)+"\n", encoding="utf-8")
    NEXT.write_text("# 第81轮修正完成，停止分批，改研究两策略协同风险\n\n"
        "修正后主基础／压力夏普0.590169／0.521870，较早0.657748／0.635056，四项均低于满仓。此前未核完就写入本文件的0.711855等数字属于错误持仓字段来源，已在第81轮原目录保留并标记无效。组合分批尚未登记，现停止该方向，避免在明确变差的执行法上继续花时间。\n\n"
        "下一82拟将两条原参考策略视作资金使用规则，用过去完整净收益的方差及两者协方差选择合成参考收益方差最小的非负预算；不估计预期收益，不借款。沿用76的242日、月首、初始各半及缺失或零风险时沿用预算。与76倒数波动分配的目标不同：该项直接考虑两条规则同时波动，最小化组合总方差。尚未登记，先有界查重并固定唯一公式与退化条件，再跑四账户。\n", encoding="utf-8")
    for p in RESEARCH.iterdir():
        if p.is_file() and p.suffix in {".csv", ".json"}:
            shutil.copy2(p, OUT / p.name)
    shutil.copy2(CONFIG, OUT / "修正来源冻结设置.json")
    shutil.copy2(ORIGINAL / "source_defect_receipt.json", OUT / "首次来源缺陷回执.json")
    shutil.copy2(NEXT, OUT / "下一项研究方向.md")
    shutil.copy2(COEFFICIENTS, OUT / COEFFICIENTS.name)
    for period, label in [("evaluation", "主评价"), ("earlier_diagnostic", "较早历史")]:
        for cost in cfg["costs"]:
            for model in [PRIMARY, EXECUTION_CONTROL]:
                for kind in ["ledger", "decisions"]:
                    pd.read_parquet(RESEARCH / period / cost / f"{model}_{kind}.parquet").to_csv(OUT / f"{label}_{cost}_{model}_{kind}.csv", index=False, encoding="utf-8-sig")
    record = {"round": 81, "study": result["study_id"], "title": "先半仓、参考价格确认后补足，实际持仓来源修正版", "status": status,
        "result": str((RESEARCH / "result.json").relative_to(ROOT)), "original_invalid_result": str((ORIGINAL / "result.json").relative_to(ROOT)),
        "candidate_configurations": 1, "evaluated_candidate_source_runs": 2, "evaluation_accounts": 24,
        "original_evaluation_accounts": 12, "corrected_evaluation_accounts": 12, "new_accounts_generated": 6,
        "new_execution_control_accounts": 2, "new_correction_accounts": 2, "reused_control_accounts": 18,
        "earlier_diagnostic_accounts": 24, "new_earlier_diagnostic_accounts": 6, "new_model_fits": 0, "new_reference_accounts": 0,
        "primary_base": metric(result, PRIMARY), "primary_stress": metric(result, PRIMARY, cost="STRESS"), "post_selected_best_base": metric(result, PRIMARY)}
    index["completed_rounds"].append(record)
    for k in ["registered_configurations_in_this_resumption", "evaluated_configurations_in_this_resumption"]:
        index[k] += 1
    for k in ["evaluated_candidate_source_runs_including_corrected_replays", "registered_candidate_source_runs_including_unrun_legacy_bindings"]:
        index[k] += 2
    index["evaluation_accounts_in_this_resumption"] += 24
    index.update(updated_at=now(), latest_completed_round=record, running_studies=[], goal_achieved=False, status="ROUND81_SOURCE_CORRECTED_COMPLETE_TARGET_NOT_MET",
        count_warning="81轮，355不同设置，371已评价来源版本，376登记含5旧未运行，1244主评价记录；包括首次无效分批来源。",
        next_work={"status": "TWO_REFERENCE_MINIMUM_VARIANCE_NOT_REGISTERED", "focus": "两原策略方差与协方差的最小方差预算", "source": str(NEXT.relative_to(ROOT))},
        process_state_note="80已完成；81来源缺陷修正、四受影响账户及关键核对完成。未登记的组合分批已停止；82协同风险只有方向。",
        research_speed_priority="docs/510300_FAST_RESEARCH_CADENCE_20260908.md")
    index["deliveries"].append({"created_at": now(), "type": "STAGED_ENTRY_SOURCE_FIX_ROUND81_FAST_CHINESE_RESULTS", "rounds": [81],
        "directory": str(OUT), "main_document": str(DOCUMENT), "new_gpt_review_archive_created": False})
    write_json(INDEX, index)
    write_json(OUT / "交付回执.json", {"created_at": now(), "main_document": str(DOCUMENT), "source_versions": 2, "goal_achieved": False, "new_gpt_review_archive_created": False}, exclusive=True)
    print(json.dumps({"交付": str(DOCUMENT), "核对": receipt, "来源修正影响": source_differences}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
