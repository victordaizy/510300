"""经济权重和实际账户关键核对，简洁交付，不重拟合。"""
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from research.economic_regret_v1 import ROOT, OUT as RESEARCH, CONFIG, PRIMARY
from research.learned_cycle_exit_v1 import FEATURES, training_rows
from research.adaptive_allocation_v1 import normalize_dividends, summarize
from research.intraday_overnight_increment_v1 import now, require, write_json, digest
from scripts.review_round74_saved import saved_cycles
from scripts.finalize_round60_20260907 import metric, table

OUT = ROOT / "deliverables/510300经济错判损失退出_第80轮_20260908"
DOCUMENT = OUT / "经济损失退出_结果及全部中文规则.md"
INDEX = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
NEXT = ROOT / "docs/510300_AFTER_ECONOMIC_REGRET_STAGED_ENTRY_20260908.md"


def main():
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    require(index["latest_completed_round"]["round"] == 79 and not OUT.exists(), "80前序或交付状态不符")
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    result = json.loads((RESEARCH / "result.json").read_text(encoding="utf-8"))
    require(not result["historical_point_target_met"], "出现目标值时不能直接写成未达标")
    models = json.loads((RESEARCH / "saved_models.json").read_text(encoding="utf-8"))["models"]
    by_date = {pd.Timestamp(m["fit_origin"]): m for m in models}
    members = pd.read_parquet(RESEARCH / "training_memberships.parquet")
    samples = pd.read_parquet(ROOT / cfg["samples"])
    samples = samples[samples.signal.eq("D60_INTRA")]
    old_models = json.loads((ROOT / "reports/research/510300_directional_continuation_v1/saved_models.json").read_text(encoding="utf-8"))["models"]
    old_by_date = {m["fit_index"]: m for m in old_models}
    weight_checks = []
    for record in [m for m in models if m["status"] == "FIT_COMPLETE"]:
        t = record["fit_index"]
        rows, ids = training_rows(samples, t, cfg)
        require(ids == record["training_cycles"] and (rows.exit_index <= t).all(), "经济权重使用未来或不同周期")
        raw = rows.sample_weight.to_numpy()*rows.target.abs().to_numpy()
        expected = raw/raw.sum()*rows.sample_weight.sum()
        saved = members[members.fit_index.eq(t)].sort_values(["cycle_id", "origin_index"])
        np.testing.assert_allclose(expected, saved.economic_weight, atol=1e-12, rtol=0)
        np.testing.assert_allclose(rows.sample_weight, saved.sample_weight, atol=0, rtol=0)
        model = record["model"]
        np.testing.assert_allclose(model["mean"], old_by_date[t]["model"]["mean"], atol=0, rtol=0)
        np.testing.assert_allclose(model["scale"], old_by_date[t]["model"]["scale"], atol=0, rtol=0)
        require(abs(expected.sum()-model["economic_weight_sum"]) < 1e-12 and not model["score_is_calibrated_probability"], "经济权重总量或评分含义不符")
        require(abs(expected[rows.target.to_numpy() > 0].sum()-model["positive_economic_weight"]) < 1e-12, "正差额经济权重不符")
        weight_checks.append({"fit_index": t, "fit_origin": record["fit_origin"], "cycles": len(ids), "rows": len(rows),
            "base_weight_sum": rows.sample_weight.sum(), "economic_weight_sum": expected.sum(),
            "maximum_cycle_weight_share": saved.groupby("cycle_id").economic_weight.sum().max()/expected.sum()})
    require(len(weight_checks) == 114 and result["failed_fits"] == result["single_class_no_view"] == result["zero_economic_no_view"] == 0, "经济拟合支持或收敛状态不符")
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    cycles, checked, differences, predictions = [], [], [], []
    for period in ["evaluation", "earlier_diagnostic"]:
        for cost in cfg["costs"]:
            folder = RESEARCH / period / cost
            ledger = pd.read_parquet(folder / f"{PRIMARY}_ledger.parquet")
            decisions = pd.read_parquet(folder / f"{PRIMARY}_decisions.parquet")
            require(decisions.continuation_prediction.dropna().empty and decisions.continuation_probability.dropna().empty, "经济评分被写成收益或胜率")
            for cycle_id, group in decisions[decisions.learning_cycle_id.notna()].groupby("learning_cycle_id", sort=False):
                count = 0
                for row in group.itertuples():
                    if row.learning_status == "PREDICTION_AVAILABLE":
                        m = by_date[row.learning_fit_origin]
                        require(m["latest_exit_index"] <= m["fit_index"] <= row.origin_index, "实际持仓调用未来经济模型")
                        x = np.array([getattr(row, c) for c in FEATURES])
                        z = np.clip((x-m["model"]["mean"])/m["model"]["scale"], -5, 5)
                        margin = m["model"]["intercept"]+z@np.array(m["model"]["coefficients"])
                        score = float(expit(margin))
                        require(abs(score-row.economic_continuation_score) < 1e-12 and abs(margin-row.economic_linear_margin) < 1e-12, "经济评分不能由系数复算")
                        count = count+1 if score < .5 else 0
                        predictions.append({"period": period, "cost": cost, "origin": row.origin, "cycle": cycle_id, "economic_score": score})
                    else:
                        require(pd.isna(row.economic_continuation_score), "无模型时仍填经济评分")
                        count = 0
                    require(count == row.negative_confirmation_count and bool(row.learned_exit_requested) == (count >= 2), "经济确认或周期重置不符")
            m = metric(result, PRIMARY, period, cost)
            previous = np.r_[cfg["initial_capital"], ledger.equity.iloc[:-1].to_numpy()]
            np.testing.assert_allclose(ledger.cash+ledger.shares*ledger.mark+ledger.dividend_receivable, ledger.equity, atol=1e-6, rtol=0)
            np.testing.assert_allclose(ledger.equity/previous-1, ledger.net_return, atol=1e-12, rtol=0)
            recomputed = summarize(ledger, cfg)
            for field in ["net_sharpe", "annualized_return", "max_drawdown", "cumulative_return", "trade_count", "commission", "slippage_cost"]:
                require(abs(recomputed[field]-m[field]) < 1e-9, "经济退出保存指标不符")
            cycles.extend({"period": period, "cost": cost, **c} for c in saved_cycles(ledger, dividends, cfg))
            checked.append({"period": period, "cost": cost, "days": len(ledger), "net_sharpe": m["net_sharpe"]})
            for control_name in ["REARM_RIDGE", "DIRECTIONAL_CONTINUATION_EXIT"]:
                control = pd.read_parquet(folder / f"{control_name}_ledger.parquet")
                differences.append({"period": period, "cost": cost, "control": control_name,
                    "terminal_nav_difference": float(ledger.equity.iloc[-1]-control.equity.iloc[-1]),
                    **{f"{c}_difference": float(ledger[c].sum()-control[c].sum()) for c in ["price_pnl", "dividend_recognized", "commission", "slippage_cost"]}})
    for name, rows in [("saved_actual_cycles.csv", cycles), ("saved_economic_score_checks.csv", predictions), ("saved_profit_differences.csv", differences), ("saved_economic_weight_checks.csv", weight_checks)]:
        pd.DataFrame(rows).to_csv(RESEARCH / name, index=False, encoding="utf-8-sig")
    receipt = {"verified_at": now(), "status": "KEY_ECONOMIC_WEIGHTS_SCORES_AND_COMPLETE_ACCOUNT_CHECKS_COMPLETE", "mature_models": len(weight_checks),
        "saved_scores_recomputed": len(predictions), "new_account_metrics_recomputed": len(checked), "complete_cycles": len(cycles),
        "new_accounts_or_models": 0, "reviewer_source_sha256": digest(Path(__file__)), "security_audit_performed": False}
    write_json(RESEARCH / "saved_verification_receipt.json", receipt, exclusive=True)
    status = "COMPLETED_ECONOMIC_REGRET_TARGET_NOT_MET"
    write_json(RESEARCH / "acceptance_outcome.json", {"status": status, "recorded_at": now(), "goal_achieved": False,
        "decision": "经济权重较79主恶化早回升，两段仍低于原R32，结束本项，不更换权重或阈值救回。", "position_impact": 0}, exclusive=True)
    OUT.mkdir(parents=True)
    lines = ["# 第80轮：经济错判损失退出结果", "", "主基础／压力净夏普0.211／0.137，较早0.667／0.644，仍未达到1.2。相对第79轮，较早回升但主明显下降；相对原平均收益学习退出，两段均较差。结束该经济权重方法。", "",
        "## 主评价：2020年1月2日至2026年8月14日开盘", ""] + table(result["all_metrics"])
    lines += ["## 较早历史：2015年1月5日至2019年12月31日开盘", ""] + table(result["earlier_diagnostics"])
    lines += ["114次经济模型全部拟合成功，27个早期检查没有足够成熟样本，未出现全零、单类或求解失败。主每档27周期、54笔成交、228个持仓收盘，26次学习退出；较早9周期、18笔、303个收盘，只有1次学习退出。", "",
        "主基础回撤扩大至17.76%。经济权重使大幅度历史差额更影响训练，实际结果没有支持这次改动；不能仅凭训练目标更贴近收益就认为回测会改善。下一项改为先半仓、价格确认回升后补足的执行节奏，当前尚未登记或运行。", "",
        f"6项必要测试一次通过；关键核对涵盖114次经济权重和原标准化一致、{len(predictions)}条有效评分及全部连续确认、4账户指标和{len(cycles)}个含分红完整周期，无重拟合或新增诊断账户。全部月度8因子系数与中文计算见同目录“每月经济损失退出模型中文规则.md”。", "", "## 全部中文规则", ""]
    lines += (ROOT / cfg["rules"]).read_text(encoding="utf-8").splitlines()[1:]
    DOCUMENT.write_text("\n".join(lines)+"\n", encoding="utf-8")
    for p in RESEARCH.iterdir():
        if p.is_file() and p.suffix in {".csv", ".json", ".md"}:
            shutil.copy2(p, OUT / p.name)
    shutil.copy2(CONFIG, OUT / "冻结设置.json")
    shutil.copy2(NEXT, OUT / "下一项分批进入方向.md")
    for period, label in [("evaluation", "主评价"), ("earlier_diagnostic", "较早历史")]:
        for cost in cfg["costs"]:
            for kind in ["ledger", "decisions"]:
                pd.read_parquet(RESEARCH / period / cost / f"{PRIMARY}_{kind}.parquet").to_csv(OUT / f"{label}_{cost}_{'完整账户' if kind=='ledger' else '全部因子与进出场'}.csv", index=False, encoding="utf-8-sig")
    record = {"round": 80, "study": result["study_id"], "title": "按经济错判损失学习持仓退出", "status": status,
        "result": str((RESEARCH / "result.json").relative_to(ROOT)), **{k: result[k] for k in ["candidate_configurations", "evaluation_accounts", "new_accounts_generated", "reused_control_accounts", "earlier_diagnostic_accounts", "new_earlier_diagnostic_accounts", "new_model_fits", "new_reference_accounts"]},
        "evaluated_candidate_source_runs": 1, "primary_base": metric(result, PRIMARY), "primary_stress": metric(result, PRIMARY, cost="STRESS"), "post_selected_best_base": metric(result, PRIMARY)}
    index["completed_rounds"].append(record)
    for k in ["registered_configurations_in_this_resumption", "evaluated_configurations_in_this_resumption", "evaluated_candidate_source_runs_including_corrected_replays", "registered_candidate_source_runs_including_unrun_legacy_bindings"]:
        index[k] += 1
    index["evaluation_accounts_in_this_resumption"] += result["evaluation_accounts"]
    index.update(updated_at=now(), latest_completed_round=record, running_studies=[], goal_achieved=False, status="ROUND80_COMPLETE_ECONOMIC_REGRET_TARGET_NOT_MET",
        count_warning="80轮，354不同设置，369已评价来源版本，374登记含5旧未运行，1220主评价记录。",
        next_work={"status": "PRICE_CONFIRMED_STAGED_ENTRY_DIRECTION_NOT_REGISTERED", "focus": "先半仓、价格确认回升再补足的原参考执行节奏", "source": str(NEXT.relative_to(ROOT))},
        process_state_note="80拟合、四账户、关键核对及简洁交付完成；81分批进入仅方向，尚未登记。",
        research_speed_priority="docs/510300_FAST_RESEARCH_CADENCE_20260908.md")
    index["deliveries"].append({"created_at": now(), "type": "ECONOMIC_REGRET_ROUND80_FAST_CHINESE_RESULTS", "rounds": [80], "directory": str(OUT), "main_document": str(DOCUMENT), "new_gpt_review_archive_created": False})
    write_json(INDEX, index)
    write_json(OUT / "交付回执.json", {"created_at": now(), "main_document": str(DOCUMENT), "goal_achieved": False, "new_gpt_review_archive_created": False}, exclusive=True)
    print(json.dumps({"交付": str(DOCUMENT), "核对": receipt}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
