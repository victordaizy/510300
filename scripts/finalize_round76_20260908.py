"""风险预算只做必要保存核对与简洁交付，避免重复长篇整理。"""
import json
import shutil
import numpy as np
import pandas as pd

from research.two_policy_risk_budget_v1 import ROOT, OUT as RESEARCH, CONFIG, PRIMARY
from research.adaptive_allocation_v1 import normalize_dividends, summarize
from research.intraday_overnight_increment_v1 import now, require, write_json, digest
from scripts.review_round74_saved import saved_cycles
from scripts.finalize_round60_20260907 import metric, table

OUT = ROOT / "deliverables/510300风险预算_第76轮_20260908"
DOCUMENT = OUT / "风险预算_结果和全部中文规则.md"
INDEX = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"


def main():
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    require(index["latest_completed_round"]["round"] == 75 and not OUT.exists(), "第76轮前序或交付状态不符")
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    result = json.loads((RESEARCH / "result.json").read_text(encoding="utf-8"))
    require(not result["historical_point_target_met"], "出现1.2后不能直接按未达标收尾")
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    differences, all_cycles, risk_checks = [], [], []
    metric_count = 0
    for period, key in [("evaluation", "all_metrics"), ("earlier_diagnostic", "earlier_diagnostics")]:
        factors = pd.read_parquet(RESEARCH / f"{period}_factors.parquet")
        weights = np.array([.5, .5])
        first = int(np.flatnonzero(factors.date >= pd.Timestamp(cfg["evaluation_start"] if period=="evaluation" else cfg["earlier_start"]))[0])
        for t in range(first-1, len(factors)-1):
            row = factors.iloc[t]
            if row.risk_update_scheduled:
                count = min(242, t-first+1)
                values = factors.iloc[t-count+1:t+1][["panic_reference_return", "learned_reference_return"]].to_numpy()
                if count < 242:
                    expected_status = "NO_VIEW_WARMUP_KEEP_BUDGET"
                elif not np.isfinite(values).all():
                    expected_status = "NO_VIEW_INCOMPLETE_WINDOW_KEEP_BUDGET"
                else:
                    sd = values.std(axis=0, ddof=1)
                    if (sd <= 0).any():
                        expected_status = "NO_VIEW_ZERO_OR_INVALID_RISK_KEEP_BUDGET"
                    else:
                        inverse = 1 / sd
                        weights = inverse / inverse.sum()
                        expected_status = "RISK_BUDGET_AVAILABLE"
                require(expected_status == row.risk_status and count == row.risk_window_observations, "月度风险状态或完整观察数不符")
                require(row.date.to_period("M") != factors.date.iloc[t-1].to_period("M"), "非月首改变预算")
                risk_checks.append({"period": period, "date": row.date, "status": row.risk_status, "observations": count, "panic_budget": weights[0], "learned_budget": weights[1]})
            np.testing.assert_allclose(weights, [row.panic_budget, row.learned_budget], atol=1e-14, rtol=0)
            expected_target = np.array([row.panic_state, row.learned_state]) @ weights
            require(abs(expected_target-row.target) < 1e-14, "两条状态的真实合成目标不符")
        for cost in cfg["costs"]:
            folder = RESEARCH / period / cost
            current = pd.read_parquet(folder / f"{PRIMARY}_ledger.parquet")
            all_cycles += [{"period": period, "cost": cost, **c} for c in saved_cycles(current, dividends, cfg)]
            for m in [x for x in result[key] if x["cost"]==cost]:
                saved = pd.read_parquet(folder / f"{m['model']}_ledger.parquet")
                prior = np.r_[cfg["initial_capital"], saved.equity.iloc[:-1].to_numpy()]
                np.testing.assert_allclose(saved.cash+saved.shares*saved.mark+saved.dividend_receivable, saved.equity, atol=1e-6, rtol=0)
                np.testing.assert_allclose(saved.equity/prior-1, saved.net_return, atol=1e-12, rtol=0)
                recomputed = summarize(saved, cfg)
                for field in ["net_sharpe", "annualized_return", "max_drawdown", "cumulative_return", "trade_count", "commission", "slippage_cost"]:
                    require(abs(recomputed[field]-m[field]) < 1e-9, "保存账户指标不符")
                metric_count += 1
                if m["model"] == "PANIC_LEARNED_HALF":
                    differences.append({"period": period, "cost": cost, "net_profit_difference": float(current.equity.iloc[-1]-saved.equity.iloc[-1]),
                        **{f"{c}_difference": float(current[c].sum()-saved[c].sum()) for c in ["price_pnl", "dividend_recognized", "commission", "slippage_cost"]}})
    pd.DataFrame(all_cycles).to_csv(RESEARCH / "saved_actual_cycles.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(differences).to_csv(RESEARCH / "saved_half_budget_differences.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(risk_checks).to_csv(RESEARCH / "saved_risk_update_checks.csv", index=False, encoding="utf-8-sig")
    receipt = {"verified_at": now(), "status": "KEY_SAVED_RISK_AND_COMPLETE_ACCOUNT_CHECKS_COMPLETE", "risk_update_checks": len(risk_checks),
        "account_metrics_recomputed": metric_count, "complete_cycles": len(all_cycles), "new_accounts_or_fits": 0,
        "security_audit_performed": False, "reviewer_source_sha256": digest(__import__('pathlib').Path(__file__))}
    write_json(RESEARCH / "saved_verification_receipt.json", receipt, exclusive=True)
    status = "COMPLETED_LOCAL_IMPROVEMENT_MAIN_ONLY_TARGET_NOT_MET"
    write_json(RESEARCH / "acceptance_outcome.json", {"status": status, "recorded_at": now(), "goal_achieved": False,
        "decision": "保留主评价风险收益改善线索；较早退步，未达1.2及稳定超额，不修改本设置窗口、时钟或零风险规则救回。", "position_impact": 0}, exclusive=True)
    OUT.mkdir(parents=True)
    lines = ["# 第76轮：按过去风险分配两策略预算", "", "主评价有局部改善，仍未达到1.2，较早历史退步。保留线索，不能认定稳定有效。", "",
        "每月用两条原基础费用策略过去242个完整账户日收益的波动分预算，较高波动分较少预算；原急跌与学习策略的进入、退出和重新进入规则保持。实际合并账户单独计算全部成本、分红及成交。", "",
        "## 主评价：2020年1月至2026年8月固定终点", ""] + table(result["all_metrics"])
    lines += ["## 较早历史：2015年至2019年固定终点", ""] + table(result["earlier_diagnostics"])
    lines += ["## 这轮学到了什么", "",
        "基础主夏普从原各半0.944升至1.091，年化从4.01%至4.05%，回撤从5.39%降至3.24%。终值只多664.96元：价格损益多2,173.80元、分红少2,420.00元、显式费用少911.16元。改善主要体现为收益波动下降，不能说赚的钱大幅增加。压力夏普也由0.879升到1.032。", "",
        "较早基础夏普由0.604降至0.559，净利润少3,941.88元，主要是价格收益减少3,873.20元。主期改善没有在较早历史延续。", "",
        "主期80个月首尝试，12次历史不足、41次因一条参考波动为零保留此前预算，只有27次得到有效风险估计；较早60次中12次不足、16次零风险、32次有效。急跌信号交易稀少，完整日波动同时受交易频率影响，不能把低波动直接解释为每次进入风险低。下一步优先检验这一可解释的局部问题，不再重复长篇整理。", "",
        "主实际账户每档53笔成交、262个持仓收盘；较早34笔、339个收盘，均无整笔未成交。2023年主账户全年现金、夏普未定义，日期仍保留。两段各20万元、242日年化、全部空仓日及原两档费用保持；完整账户目标仍未完成。", "",
        "## 全部中文因子和进出场规则", ""]
    protocol = (ROOT / cfg["rules"]).read_text(encoding="utf-8").splitlines()[1:]
    lines += protocol + ["", "## 必要核对", "", f"7项关键测试一次通过；已核对{len(risk_checks)}个月度风险更新、逐日预算与合成目标、{metric_count}项保存账户指标及{len(all_cycles)}个含分红完整周期。未重新模拟账户或训练模型。全部原始逐日因子、请求及实际成交保存在同目录CSV。", ""]
    DOCUMENT.write_text("\n".join(lines), encoding="utf-8")
    for p in RESEARCH.iterdir():
        if p.is_file() and p.suffix in {".csv", ".json"}:
            shutil.copy2(p, OUT / p.name)
    shutil.copy2(CONFIG, OUT / "冻结设置.json")
    for period, label in [("evaluation", "主评价"), ("earlier_diagnostic", "较早历史")]:
        for cost in cfg["costs"]:
            folder = RESEARCH / period / cost
            for kind in ["ledger", "decisions"]:
                pd.read_parquet(folder / f"{PRIMARY}_{kind}.parquet").to_csv(OUT / f"{label}_{cost}_{'完整账户' if kind=='ledger' else '全部因子与进出场'}.csv", index=False, encoding="utf-8-sig")
    record = {"round": 76, "study": result["study_id"], "title": "两原策略按已实现风险分配预算", "status": status,
        "result": str((RESEARCH / "result.json").relative_to(ROOT)), **{k: result[k] for k in ["candidate_configurations", "evaluation_accounts", "new_accounts_generated", "reused_control_accounts", "earlier_diagnostic_accounts", "new_earlier_diagnostic_accounts", "new_model_fits", "new_reference_accounts"]},
        "evaluated_candidate_source_runs": 1, "primary_base": metric(result, PRIMARY), "primary_stress": metric(result, PRIMARY, cost="STRESS"), "post_selected_best_base": metric(result, PRIMARY)}
    index["completed_rounds"].append(record)
    for k in ["registered_configurations_in_this_resumption", "evaluated_configurations_in_this_resumption", "evaluated_candidate_source_runs_including_corrected_replays", "registered_candidate_source_runs_including_unrun_legacy_bindings"]:
        index[k] += 1
    index["evaluation_accounts_in_this_resumption"] += result["evaluation_accounts"]
    index.update(updated_at=now(), latest_completed_round=record, running_studies=[], goal_achieved=False, status="ROUND76_COMPLETE_MAIN_RISK_IMPROVEMENT_TARGET_NOT_MET",
        count_warning="76轮，350不同设置，365已评价来源版本，370登记含5旧未运行，1178主评价记录。",
        next_work={"status": "ACTIVE_EXPOSURE_RISK_DIRECTION_NOT_REGISTERED", "focus": "检验稀疏交易对风险估计的影响，只用实际承担价格风险日估计参考波动，账户绩效仍保留所有日", "source": "docs/510300_FAST_RESEARCH_CADENCE_20260908.md"},
        process_state_note="76已完成完整账户及简洁交付，77仅确定实际承担风险日的条件风险方向，尚未登记或实现。")
    index["deliveries"].append({"created_at": now(), "type": "RISK_BUDGET_ROUND76_FAST_CHINESE_RESULTS", "rounds": [76], "directory": str(OUT), "main_document": str(DOCUMENT), "new_gpt_review_archive_created": False})
    write_json(INDEX, index)
    write_json(OUT / "交付回执.json", {"created_at": now(), "main_document": str(DOCUMENT), "goal_achieved": False, "new_gpt_review_archive_created": False, "security_audit_performed": False}, exclusive=True)
    print(json.dumps({"交付": str(DOCUMENT), "关键核对": receipt, "原各半差额": differences}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
