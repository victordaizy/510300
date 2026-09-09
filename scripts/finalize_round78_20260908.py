"""完成三项反转的关键核对及简洁交付，不重跑账户。"""
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from research.composite_streak_reversal_v1 import ROOT, OUT as RESEARCH, CONFIG, PRIMARY
from research.adaptive_allocation_v1 import normalize_dividends, summarize
from research.intraday_overnight_increment_v1 import now, require, write_json, digest
from scripts.review_round74_saved import saved_cycles
from scripts.finalize_round60_20260907 import metric, table

OUT = ROOT / "deliverables/510300三项短期反转_第78轮_20260908"
DOCUMENT = OUT / "三项短期反转_结果及中文规则.md"
INDEX = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
NEXT = ROOT / "docs/510300_AFTER_COMPOSITE_DIRECTIONAL_EXIT_20260908.md"


def main():
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    require(index["latest_completed_round"]["round"] == 77 and not OUT.exists(), "78前序或交付状态不符")
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    result = json.loads((RESEARCH / "result.json").read_text(encoding="utf-8"))
    require(not result["historical_point_target_met"], "出现目标值时不能直接写成未达标")
    main_f = pd.read_parquet(RESEARCH / "evaluation_factors.parquet")
    early_f = pd.read_parquet(RESEARCH / "earlier_diagnostic_factors.parquet")
    pd.testing.assert_frame_equal(main_f.iloc[:len(early_f)], early_f)
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    cycles, checked = [], []
    for period, key in [("evaluation", "all_metrics"), ("earlier_diagnostic", "earlier_diagnostics")]:
        for cost in cfg["costs"]:
            ledger = pd.read_parquet(RESEARCH / period / cost / f"{PRIMARY}_ledger.parquet")
            m = metric(result, PRIMARY, period, cost)
            previous = np.r_[cfg["initial_capital"], ledger.equity.iloc[:-1].to_numpy()]
            np.testing.assert_allclose(ledger.cash+ledger.shares*ledger.mark+ledger.dividend_receivable, ledger.equity, atol=1e-6, rtol=0)
            np.testing.assert_allclose(ledger.equity/previous-1, ledger.net_return, atol=1e-12, rtol=0)
            recomputed = summarize(ledger, cfg)
            for field in ["net_sharpe", "annualized_return", "max_drawdown", "cumulative_return", "trade_count", "commission", "slippage_cost"]:
                require(abs(recomputed[field]-m[field]) < 1e-9, "三项反转保存指标不符")
            cycles.extend({"period": period, "cost": cost, **c} for c in saved_cycles(ledger, dividends, cfg))
            checked.append({"period": period, "cost": cost, "days": len(ledger), "net_sharpe": m["net_sharpe"]})
    pd.DataFrame(cycles).to_csv(RESEARCH / "saved_actual_cycles.csv", index=False, encoding="utf-8-sig")
    receipt = {"verified_at": now(), "status": "KEY_CAUSAL_PREFIX_AND_COMPLETE_ACCOUNT_CHECKS_COMPLETE", "prefix_days": len(early_f),
        "new_account_metrics_recomputed": len(checked), "complete_cycles": len(cycles), "new_accounts_or_models": 0,
        "reviewer_source_sha256": digest(Path(__file__)), "security_audit_performed": False}
    write_json(RESEARCH / "saved_verification_receipt.json", receipt, exclusive=True)
    status = "COMPLETED_COMPOSITE_REVERSAL_TARGET_NOT_MET"
    write_json(RESEARCH / "acceptance_outcome.json", {"status": status, "recorded_at": now(), "goal_achieved": False,
        "decision": "三项反转两段均未改善主要线索，结束本项，不调门槛和窗口救回。", "position_impact": 0}, exclusive=True)
    OUT.mkdir(parents=True)
    lines = ["# 第78轮：三项短期反转结果", "", "主评价基础／压力净夏普0.324／0.253，较早0.223／0.150，未达到1.2。两段各10个完整周期，平均股票仓位仅约1.8%与2.0%，完整现金日已保留。结束这项固定策略，继续其他机制。", "",
        "## 主评价：2020年1月2日至2026年8月14日开盘", ""] + table(result["all_metrics"])
    lines += ["## 较早历史：2015年1月5日至2019年12月31日开盘", ""] + table(result["earlier_diagnostics"])
    lines += ["6项必要测试一次通过，实际四账户一次运行完成；关键核对覆盖较早因子与主数据前缀一致、全部4个新账户净值及指标、40个含分红完整周期。主每档29个持仓收盘，较早24个，均20笔成交，无整笔未成交或评价期因子缺失。", "",
        "已确认原学习退出训练本来就按周期等权，因此跳过重复‘修正’。下一项概率退出见同目录方向说明，当前未登记或运行。", "", "## 完整中文因子与进出场", ""]
    lines += (ROOT / cfg["rules"]).read_text(encoding="utf-8").splitlines()[1:]
    DOCUMENT.write_text("\n".join(lines)+"\n", encoding="utf-8")
    for p in RESEARCH.iterdir():
        if p.is_file() and p.suffix in {".csv", ".json"}:
            shutil.copy2(p, OUT / p.name)
    shutil.copy2(CONFIG, OUT / "冻结设置.json")
    shutil.copy2(NEXT, OUT / "下一项概率退出方向.md")
    for period, label in [("evaluation", "主评价"), ("earlier_diagnostic", "较早历史")]:
        for cost in cfg["costs"]:
            for kind in ["ledger", "decisions"]:
                pd.read_parquet(RESEARCH / period / cost / f"{PRIMARY}_{kind}.parquet").to_csv(OUT / f"{label}_{cost}_{'完整账户' if kind=='ledger' else '全部因子与进出场'}.csv", index=False, encoding="utf-8-sig")
    record = {"round": 78, "study": result["study_id"], "title": "三项短期反转与独立退出", "status": status,
        "result": str((RESEARCH / "result.json").relative_to(ROOT)), **{k: result[k] for k in ["candidate_configurations", "evaluation_accounts", "new_accounts_generated", "reused_control_accounts", "earlier_diagnostic_accounts", "new_earlier_diagnostic_accounts", "new_model_fits", "new_reference_accounts"]},
        "evaluated_candidate_source_runs": 1, "primary_base": metric(result, PRIMARY), "primary_stress": metric(result, PRIMARY, cost="STRESS"), "post_selected_best_base": metric(result, PRIMARY)}
    index["completed_rounds"].append(record)
    for k in ["registered_configurations_in_this_resumption", "evaluated_configurations_in_this_resumption", "evaluated_candidate_source_runs_including_corrected_replays", "registered_candidate_source_runs_including_unrun_legacy_bindings"]:
        index[k] += 1
    index["evaluation_accounts_in_this_resumption"] += result["evaluation_accounts"]
    index.update(updated_at=now(), latest_completed_round=record, running_studies=[], goal_achieved=False, status="ROUND78_COMPLETE_COMPOSITE_REVERSAL_TARGET_NOT_MET",
        count_warning="78轮，352不同设置，367已评价来源版本，372登记含5旧未运行，1200主评价记录。",
        next_work={"status": "DIRECTIONAL_CONTINUATION_EXIT_NOT_REGISTERED", "focus": "继续持有能否更好的一项二元概率退出", "source": str(NEXT.relative_to(ROOT))},
        process_state_note="78四账户及简洁交付完成；确认原周期等权已经存在、无需重复实验；79概率退出仅方向。",
        research_speed_priority="docs/510300_FAST_RESEARCH_CADENCE_20260908.md")
    index["deliveries"].append({"created_at": now(), "type": "COMPOSITE_REVERSAL_ROUND78_FAST_CHINESE_RESULTS", "rounds": [78], "directory": str(OUT), "main_document": str(DOCUMENT), "new_gpt_review_archive_created": False})
    write_json(INDEX, index)
    write_json(OUT / "交付回执.json", {"created_at": now(), "main_document": str(DOCUMENT), "goal_achieved": False, "new_gpt_review_archive_created": False}, exclusive=True)
    print(json.dumps({"交付": str(DOCUMENT), "核对": receipt}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
