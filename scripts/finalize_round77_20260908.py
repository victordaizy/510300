"""条件风险完成后快速核对关键计算，交付简洁结果。"""
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from research.active_risk_budget_v1 import ROOT, OUT as RESEARCH, CONFIG, PRIMARY, P46
from research.adaptive_allocation_v1 import normalize_dividends, summarize
from research.intraday_overnight_increment_v1 import now, require, write_json, digest
from scripts.review_round74_saved import saved_cycles
from scripts.finalize_round60_20260907 import metric, table

OUT = ROOT / "deliverables/510300持仓日条件风险_第77轮_20260908"
DOCUMENT = OUT / "条件风险_结果及中文规则.md"
INDEX = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
NEXT = ROOT / "docs/510300_AFTER_ACTIVE_RISK_COMPOSITE_REVERSAL_20260908.md"


def main():
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    require(index["latest_completed_round"]["round"] == 76 and not OUT.exists(), "77前序或交付状态不符")
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    result = json.loads((RESEARCH / "result.json").read_text(encoding="utf-8"))
    require(not result["historical_point_target_met"], "不能将出现1.2的结果直接写成未达标")
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    updates, all_cycles, diffs, metric_count = [], [], [], 0
    for period, key in [("evaluation", "all_metrics"), ("earlier_diagnostic", "earlier_diagnostics")]:
        f = pd.read_parquet(RESEARCH / f"{period}_factors.parquet")
        first = int(np.flatnonzero(f.date >= pd.Timestamp(cfg["evaluation_start"] if period=="evaluation" else cfg["earlier_start"]))[0])
        for tag, name in [("panic", "PANIC_ONLY"), ("learned", "REARM_RIDGE")]:
            ref = pd.read_parquet(P46 / period / "BASE" / f"{name}_ledger.parquet")
            expected = (ref.shares_before.gt(0) | ref.shares_after.gt(0)).to_numpy(float)
            np.testing.assert_array_equal(expected, f[f"{tag}_active_exposure"].iloc[first:].to_numpy())
            np.testing.assert_allclose(ref.net_return.to_numpy(), f[f"{tag}_reference_return"].iloc[first:].to_numpy(), atol=0, rtol=0)
        weights = np.array([.5, .5])
        for t in range(first-1, len(f)-1):
            row = f.iloc[t]
            if row.risk_update_scheduled:
                count = min(242, t-first+1)
                window = f.iloc[t-count+1:t+1]
                masks = window[["panic_active_exposure", "learned_active_exposure"]].to_numpy()
                values = window[["panic_reference_return", "learned_reference_return"]].to_numpy()
                counts = [int((masks[:, k] == 1).sum()) for k in range(2)]
                require(counts == [row.panic_active_days, row.learned_active_days], "实际股票风险日计数不符")
                if count < 242:
                    status = "NO_VIEW_WARMUP_KEEP_BUDGET"
                elif not np.isfinite(masks).all() or not np.isfinite(values).all():
                    status = "NO_VIEW_INCOMPLETE_WINDOW_KEEP_BUDGET"
                elif min(counts) < 2:
                    status = "NO_VIEW_TOO_FEW_ACTIVE_DAYS_KEEP_BUDGET"
                else:
                    sd = np.array([values[masks[:, k] == 1, k].std(ddof=1) for k in range(2)])
                    if (sd <= 0).any():
                        status = "NO_VIEW_ZERO_OR_INVALID_RISK_KEEP_BUDGET"
                    else:
                        inverse = 1/sd
                        weights = inverse / inverse.sum()
                        status = "RISK_BUDGET_AVAILABLE"
                require(status == row.risk_status, "条件风险状态不符")
                updates.append({"period": period, "date": row.date, "status": status, "panic_days": counts[0], "learned_days": counts[1]})
            np.testing.assert_allclose(weights, [row.panic_budget, row.learned_budget], atol=1e-14, rtol=0)
            require(abs(np.array([row.panic_state, row.learned_state])@weights-row.target) < 1e-14, "合成目标不符")
        for cost in cfg["costs"]:
            folder = RESEARCH / period / cost
            current = pd.read_parquet(folder / f"{PRIMARY}_ledger.parquet")
            all_cycles += [{"period": period, "cost": cost, **c} for c in saved_cycles(current, dividends, cfg)]
            for m in [x for x in result[key] if x["cost"] == cost]:
                saved = pd.read_parquet(folder / f"{m['model']}_ledger.parquet")
                prior = np.r_[cfg["initial_capital"], saved.equity.iloc[:-1].to_numpy()]
                np.testing.assert_allclose(saved.cash+saved.shares*saved.mark+saved.dividend_receivable, saved.equity, atol=1e-6, rtol=0)
                np.testing.assert_allclose(saved.equity/prior-1, saved.net_return, atol=1e-12, rtol=0)
                recomputed = summarize(saved, cfg)
                for field in ["net_sharpe", "annualized_return", "max_drawdown", "cumulative_return", "trade_count", "commission", "slippage_cost"]:
                    require(abs(recomputed[field]-m[field]) < 1e-9, "完整账户指标不符")
                metric_count += 1
                if m["model"] in ["TWO_POLICY_RISK_BUDGET", "PANIC_LEARNED_HALF"]:
                    diffs.append({"period": period, "cost": cost, "control": m["model"], "net_profit_difference": float(current.equity.iloc[-1]-saved.equity.iloc[-1]),
                        **{f"{c}_difference": float(current[c].sum()-saved[c].sum()) for c in ["price_pnl", "dividend_recognized", "commission", "slippage_cost"]}})
    for name, rows in [("saved_active_risk_checks.csv", updates), ("saved_actual_cycles.csv", all_cycles), ("saved_budget_differences.csv", diffs)]:
        pd.DataFrame(rows).to_csv(RESEARCH / name, index=False, encoding="utf-8-sig")
    receipt = {"verified_at": now(), "status": "KEY_ACTIVE_RISK_AND_COMPLETE_ACCOUNT_CHECKS_COMPLETE", "risk_updates": len(updates),
        "account_metrics_recomputed": metric_count, "complete_cycles": len(all_cycles), "new_accounts_or_models": 0,
        "reviewer_source_sha256": digest(Path(__file__)), "security_audit_performed": False}
    write_json(RESEARCH / "saved_verification_receipt.json", receipt, exclusive=True)
    status = "COMPLETED_CONDITIONAL_RISK_NOT_STABLE_TARGET_NOT_MET"
    write_json(RESEARCH / "acceptance_outcome.json", {"status": status, "recorded_at": now(), "goal_achieved": False,
        "decision": "相对76较早夏普回升、主夏普下降；两段收益稍增仍无1.2，结束本次条件风险定义，不扫活动日或窗口。", "position_impact": 0}, exclusive=True)
    OUT.mkdir(parents=True)
    lines = ["# 第77轮：按直接承担股票风险日估计波动", "", "主基础净夏普0.969、压力0.902，较早0.593、0.562，仍未达到1.2。较早相对76改善，主夏普下降，未形成稳定改进。停止这项条件定义，保留第76轮主1.091的局部线索。", "",
        "## 主评价：2020年1月至2026年8月固定终点", ""] + table(result["all_metrics"])
    lines += ["## 较早历史：2015年至2019年固定终点", ""] + table(result["earlier_diagnostics"])
    lines += ["## 关键发现", "", "基础主终值比76多2,063.89元，较早多2,637.25元；主收益稍升但波动也上升，夏普从1.091下降至0.969，不能把收益和夏普混为一个指标。较早夏普虽从0.559升至0.593，仍略低于原各半0.604。", "",
        "改用直接风险日后，可估计月份仍是主27次、早32次。主急跌参考的有效窗口只有3至11个风险日，学习22至84日；较早急跌5至32日、学习5至145日。现金日没有从策略绩效中删除；样本稀少仍限制条件风险判断。", "",
        "主每档54笔成交、262持仓收盘，较早32笔、339收盘，无整笔未成交。3项关键测试一次通过，完成140次风险更新、全部日目标、24项账户指标与完整分红周期核对；没有另跑诊断账户或训练。", "",
        "## 本轮中文规则", ""]
    lines += (ROOT / cfg["rules"]).read_text(encoding="utf-8").splitlines()[1:]
    lines += ["", "完整两原策略因子、进出场及费用定义见同目录第76轮完整中文规则，逐月八项系数直接复用原已交付文件。下一项为短期强弱、连涨连跌和涨跌分位的三因子反转，尚未登记或回测。", ""]
    DOCUMENT.write_text("\n".join(lines), encoding="utf-8")
    for p in RESEARCH.iterdir():
        if p.is_file() and p.suffix in {".csv", ".json"}:
            shutil.copy2(p, OUT / p.name)
    shutil.copy2(CONFIG, OUT / "冻结设置.json")
    shutil.copy2(ROOT / "docs/510300_TWO_POLICY_RISK_BUDGET_V1.md", OUT / "510300_TWO_POLICY_RISK_BUDGET_V1.md")
    shutil.copy2(ROOT / "deliverables/510300下午提前进入_第75轮_20260908/沿用的每月八项模型中文规则.md", OUT / "沿用的每月八项模型中文规则.md")
    shutil.copy2(NEXT, OUT / "下一三因子反转方向.md")
    for period, label in [("evaluation", "主评价"), ("earlier_diagnostic", "较早历史")]:
        for cost in cfg["costs"]:
            for kind in ["ledger", "decisions"]:
                pd.read_parquet(RESEARCH / period / cost / f"{PRIMARY}_{kind}.parquet").to_csv(OUT / f"{label}_{cost}_{'完整账户' if kind=='ledger' else '全部因子与进出场'}.csv", index=False, encoding="utf-8-sig")
    record = {"round": 77, "study": result["study_id"], "title": "按实际承担股票风险日估计两策略风险", "status": status,
        "result": str((RESEARCH / "result.json").relative_to(ROOT)), **{k: result[k] for k in ["candidate_configurations", "evaluation_accounts", "new_accounts_generated", "reused_control_accounts", "earlier_diagnostic_accounts", "new_earlier_diagnostic_accounts", "new_model_fits", "new_reference_accounts"]},
        "evaluated_candidate_source_runs": 1, "primary_base": metric(result, PRIMARY), "primary_stress": metric(result, PRIMARY, cost="STRESS"), "post_selected_best_base": metric(result, PRIMARY)}
    index["completed_rounds"].append(record)
    for k in ["registered_configurations_in_this_resumption", "evaluated_configurations_in_this_resumption", "evaluated_candidate_source_runs_including_corrected_replays", "registered_candidate_source_runs_including_unrun_legacy_bindings"]:
        index[k] += 1
    index["evaluation_accounts_in_this_resumption"] += result["evaluation_accounts"]
    index.update(updated_at=now(), latest_completed_round=record, running_studies=[], goal_achieved=False, status="ROUND77_COMPLETE_CONDITIONAL_RISK_TARGET_NOT_MET",
        count_warning="77轮，351不同设置，366已评价来源版本，371登记含5旧未运行，1190主评价记录。",
        next_work={"status": "COMPOSITE_STREAK_REVERSAL_DIRECTION_NOT_REGISTERED", "focus": "短期价格强弱、连续涨跌持续时间及当日涨跌分位共同识别反转，先固定独立进出场", "source": str(NEXT.relative_to(ROOT))},
        process_state_note="77完整账户及简洁交付完成，78仅三因子反转方向和开发方指标定义阅读，尚未登记、实现或回测。",
        research_speed_priority="docs/510300_FAST_RESEARCH_CADENCE_20260908.md")
    index["deliveries"].append({"created_at": now(), "type": "ACTIVE_RISK_ROUND77_FAST_CHINESE_RESULTS", "rounds": [77], "directory": str(OUT), "main_document": str(DOCUMENT), "new_gpt_review_archive_created": False})
    write_json(INDEX, index)
    write_json(OUT / "交付回执.json", {"created_at": now(), "main_document": str(DOCUMENT), "goal_achieved": False, "new_gpt_review_archive_created": False, "security_audit_performed": False}, exclusive=True)
    print(json.dumps({"交付": str(DOCUMENT), "核对": receipt}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
