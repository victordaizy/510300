"""检查参考成交时钟、分批阶段和实际账户，交付简洁结果。"""
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from research.staged_entry_v1 import ROOT, OUT as RESEARCH, CONFIG, PRIMARY, EXECUTION_CONTROL, P32, COEFFICIENTS
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
            if row.shares_after == 0:
                anchor, entry_date = np.nan, pd.NaT
            np.testing.assert_allclose([f.reference_entry_wealth_open.iloc[t]], [anchor], atol=1e-12, rtol=0, equal_nan=True)
            require(bool(f.reference_new_buy.iloc[t]) == new_buy and bool(f.reference_holding.iloc[t]) == (row.shares_after > 0), "分批参考成交或持仓标记不符")
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
        "new_account_metrics_recomputed": len(metrics), "complete_cycles": len(cycles), "new_accounts_or_models": 0,
        "reviewer_source_sha256": digest(Path(__file__)), "security_audit_performed": False}
    write_json(RESEARCH / "saved_verification_receipt.json", receipt, exclusive=True)
    status = "COMPLETED_STAGED_RISK_REDUCTION_TARGET_NOT_MET"
    write_json(RESEARCH / "acceptance_outcome.json", {"status": status, "recorded_at": now(), "goal_achieved": False,
        "decision": "降低两段回撤及收益，主夏普仅微升、较早下降，不验证稳定增量或1.2。保留风险取舍，不调首批比例和门槛。", "position_impact": 0}, exclusive=True)
    OUT.mkdir(parents=True)
    lines = ["# 第81轮：价格确认分批进入结果", "", "主基础／压力净夏普0.712／0.646，较早0.697／0.677，未达1.2。同执行满仓对照主0.705／0.641，较早0.748／0.725。主仅微弱提高，较早下降；两段回撤和年化收益都降低，属于风险收益取舍。", "",
        "## 主评价：2020年1月2日至2026年8月14日开盘", ""] + table(result["all_metrics"])
    lines += ["## 较早历史：2015年1月5日至2019年12月31日开盘", ""] + table(result["earlier_diagnostics"])
    lines += ["主24参考周期14次确认补足，38买24卖；较早9周期4次补足，13买9卖。持仓收盘分别252和299，无未成交。主253个非零目标比252个持仓收盘多1，是最后一日统一开盘退出导致，日期全部保留。", "",
        "主基础回撤由同执行满仓10.59%降为6.01%，年化由5.38%降为4.17%；较早回撤13.79%降为10.35%，年化8.64%降为6.38%。新增交易笔数不必然增加总费用，因为初始及未补足资金更少；完整费用和差额已保存。", "",
        f"6项关键测试一次通过；8个新账户包括4个分批和4个同执行满仓对照。核对{len(stages)}个参考及阶段时点、8账户指标、{len(cycles)}个含分红完整周期，无重训或新参考策略。原月度8项系数直接复用同目录已有中文文件。", "",
        "下一项在第76轮原风险预算组合内仅将学习分支按这套分批阶段执行；原预算算法保持，该组合尚未登记或运行。", "", "## 全部中文规则", ""]
    lines += (ROOT / cfg["rules"]).read_text(encoding="utf-8").splitlines()[1:]
    DOCUMENT.write_text("\n".join(lines)+"\n", encoding="utf-8")
    for p in RESEARCH.iterdir():
        if p.is_file() and p.suffix in {".csv", ".json"}:
            shutil.copy2(p, OUT / p.name)
    shutil.copy2(CONFIG, OUT / "冻结设置.json")
    shutil.copy2(NEXT, OUT / "下一项组合分批执行方向.md")
    shutil.copy2(COEFFICIENTS, OUT / COEFFICIENTS.name)
    for period, label in [("evaluation", "主评价"), ("earlier_diagnostic", "较早历史")]:
        for cost in cfg["costs"]:
            for model in [PRIMARY, EXECUTION_CONTROL]:
                for kind in ["ledger", "decisions"]:
                    pd.read_parquet(RESEARCH / period / cost / f"{model}_{kind}.parquet").to_csv(OUT / f"{label}_{cost}_{model}_{kind}.csv", index=False, encoding="utf-8-sig")
    record = {"round": 81, "study": result["study_id"], "title": "先半仓、参考价格确认后补足", "status": status,
        "result": str((RESEARCH / "result.json").relative_to(ROOT)), **{k: result[k] for k in ["candidate_configurations", "evaluation_accounts", "new_accounts_generated", "new_execution_control_accounts", "reused_control_accounts", "earlier_diagnostic_accounts", "new_earlier_diagnostic_accounts", "new_model_fits", "new_reference_accounts"]},
        "evaluated_candidate_source_runs": 1, "primary_base": metric(result, PRIMARY), "primary_stress": metric(result, PRIMARY, cost="STRESS"), "post_selected_best_base": metric(result, PRIMARY)}
    index["completed_rounds"].append(record)
    for k in ["registered_configurations_in_this_resumption", "evaluated_configurations_in_this_resumption", "evaluated_candidate_source_runs_including_corrected_replays", "registered_candidate_source_runs_including_unrun_legacy_bindings"]:
        index[k] += 1
    index["evaluation_accounts_in_this_resumption"] += result["evaluation_accounts"]
    index.update(updated_at=now(), latest_completed_round=record, running_studies=[], goal_achieved=False, status="ROUND81_COMPLETE_STAGED_ENTRY_TARGET_NOT_MET",
        count_warning="81轮，355不同设置，370已评价来源版本，375登记含5旧未运行，1232主评价记录。",
        next_work={"status": "STAGED_LEARNED_BRANCH_IN_ORIGINAL_BUDGET_NOT_REGISTERED", "focus": "保持76原预算，仅按81价格确认分批执行学习分支", "source": str(NEXT.relative_to(ROOT))},
        process_state_note="80和81均完成拟合或账户、关键核对及简洁交付；82组合执行仅方向。",
        research_speed_priority="docs/510300_FAST_RESEARCH_CADENCE_20260908.md")
    index["deliveries"].append({"created_at": now(), "type": "STAGED_ENTRY_ROUND81_FAST_CHINESE_RESULTS", "rounds": [81], "directory": str(OUT), "main_document": str(DOCUMENT), "new_gpt_review_archive_created": False})
    write_json(INDEX, index)
    write_json(OUT / "交付回执.json", {"created_at": now(), "main_document": str(DOCUMENT), "goal_achieved": False, "new_gpt_review_archive_created": False}, exclusive=True)
    print(json.dumps({"交付": str(DOCUMENT), "核对": receipt, "差额": differences}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
