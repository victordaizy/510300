"""核对概率语义和完整账户，快速交付第79轮。"""
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from research.directional_continuation_v1 import ROOT, OUT as RESEARCH, CONFIG, PRIMARY
from research.learned_cycle_exit_v1 import FEATURES
from research.adaptive_allocation_v1 import normalize_dividends, summarize
from research.intraday_overnight_increment_v1 import now, require, write_json, digest
from scripts.review_round74_saved import saved_cycles
from scripts.finalize_round60_20260907 import metric, table

OUT = ROOT / "deliverables/510300继续占优概率退出_第79轮_20260908"
DOCUMENT = OUT / "概率退出_结果及全部中文规则.md"
INDEX = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
NEXT = ROOT / "docs/510300_AFTER_DIRECTIONAL_REGRET_EXIT_20260908.md"


def main():
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    require(index["latest_completed_round"]["round"] == 78 and not OUT.exists(), "79前序或交付状态不符")
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    result = json.loads((RESEARCH / "result.json").read_text(encoding="utf-8"))
    require(not result["historical_point_target_met"], "出现目标值时不能直接写成未达标")
    models = json.loads((RESEARCH / "saved_models.json").read_text(encoding="utf-8"))["models"]
    models_by_date = {pd.Timestamp(m["fit_origin"]): m for m in models}
    members = pd.read_parquet(RESEARCH / "training_memberships.parquet")
    require((members.exit_index <= members.fit_index).all(), "训练包含未成熟标签")
    np.testing.assert_allclose(members.groupby(["fit_index", "cycle_id"]).sample_weight.sum(), 1., atol=1e-12, rtol=0)
    fitted = [m for m in models if m["status"] == "FIT_COMPLETE"]
    require(len(fitted) == 114 and len(models) == 141 and result["failed_fits"] == result["single_class_no_view"] == 0, "训练支持或求解状态不符")
    for m in fitted:
        require(m["latest_exit_index"] <= m["fit_index"] and m["model"]["features"] == FEATURES, "保存模型成熟性或8因子身份不符")
        require(m["model"]["positive_weight"] > 0 and m["model"]["nonpositive_weight"] > 0, "保存模型缺少双类支持")
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    cycles, checked, differences, predictions = [], [], [], []
    for period in ["evaluation", "earlier_diagnostic"]:
        for cost in cfg["costs"]:
            folder = RESEARCH / period / cost
            ledger = pd.read_parquet(folder / f"{PRIMARY}_ledger.parquet")
            decisions = pd.read_parquet(folder / f"{PRIMARY}_decisions.parquet")
            require(decisions.continuation_prediction.dropna().empty, "概率被错误写成收益预测")
            for cycle_id, group in decisions[decisions.learning_cycle_id.notna()].groupby("learning_cycle_id", sort=False):
                count = 0
                for row in group.itertuples():
                    available = row.learning_status == "PREDICTION_AVAILABLE"
                    if available:
                        m = models_by_date[row.learning_fit_origin]
                        require(m["latest_exit_index"] <= m["fit_index"] <= row.origin_index, "实际持仓调用未来模型")
                        x = np.array([getattr(row, c) for c in FEATURES])
                        z = np.clip((x-m["model"]["mean"])/m["model"]["scale"], -5, 5)
                        score = m["model"]["intercept"] + z@np.array(m["model"]["coefficients"])
                        p = float(expit(score))
                        require(abs(p-row.continuation_probability) < 1e-12 and abs(score-row.continuation_log_odds) < 1e-12, "保存概率不能从月度系数复算")
                        count = count+1 if p < .5 else 0
                        predictions.append({"period": period, "cost": cost, "origin": row.origin, "cycle": cycle_id, "probability": p})
                    else:
                        require(pd.isna(row.continuation_probability), "没有模型时仍填入概率")
                        count = 0
                    require(count == row.negative_confirmation_count and bool(row.learned_exit_requested) == (count >= 2), "概率连续确认或新周期重置不符")
            m = metric(result, PRIMARY, period, cost)
            previous = np.r_[cfg["initial_capital"], ledger.equity.iloc[:-1].to_numpy()]
            np.testing.assert_allclose(ledger.cash+ledger.shares*ledger.mark+ledger.dividend_receivable, ledger.equity, atol=1e-6, rtol=0)
            np.testing.assert_allclose(ledger.equity/previous-1, ledger.net_return, atol=1e-12, rtol=0)
            recomputed = summarize(ledger, cfg)
            for field in ["net_sharpe", "annualized_return", "max_drawdown", "cumulative_return", "trade_count", "commission", "slippage_cost"]:
                require(abs(recomputed[field]-m[field]) < 1e-9, "概率退出保存指标不符")
            cycles.extend({"period": period, "cost": cost, **c} for c in saved_cycles(ledger, dividends, cfg))
            checked.append({"period": period, "cost": cost, "days": len(ledger), "net_sharpe": m["net_sharpe"]})
            control = pd.read_parquet(folder / "REARM_RIDGE_ledger.parquet")
            differences.append({"period": period, "cost": cost, "terminal_nav_difference": float(ledger.equity.iloc[-1]-control.equity.iloc[-1]),
                **{f"{c}_difference": float(ledger[c].sum()-control[c].sum()) for c in ["price_pnl", "dividend_recognized", "commission", "slippage_cost"]}})
    for name, rows in [("saved_actual_cycles.csv", cycles), ("saved_probability_checks.csv", predictions), ("saved_profit_differences.csv", differences)]:
        pd.DataFrame(rows).to_csv(RESEARCH / name, index=False, encoding="utf-8-sig")
    receipt = {"verified_at": now(), "status": "KEY_MATURITY_PROBABILITY_AND_COMPLETE_ACCOUNT_CHECKS_COMPLETE", "mature_models": len(fitted),
        "saved_probabilities_recomputed": len(predictions), "new_account_metrics_recomputed": len(checked), "complete_cycles": len(cycles),
        "new_accounts_or_models": 0, "reviewer_source_sha256": digest(Path(__file__)), "security_audit_performed": False}
    write_json(RESEARCH / "saved_verification_receipt.json", receipt, exclusive=True)
    status = "COMPLETED_DIRECTIONAL_PROBABILITY_TARGET_NOT_MET"
    write_json(RESEARCH / "acceptance_outcome.json", {"status": status, "recorded_at": now(), "goal_achieved": False,
        "decision": "概率退出两段夏普及四终值均低于原平均收益退出，结束本项，不调门槛或正则救回。", "position_impact": 0}, exclusive=True)
    OUT.mkdir(parents=True)
    lines = ["# 第79轮：继续占优概率退出结果", "", "主基础／压力净夏普0.487／0.360，较早0.620／0.595；均低于原平均收益学习退出，未达到1.2。结束本次纯方向概率方案。", "",
        "## 主评价：2020年1月2日至2026年8月14日开盘", ""] + table(result["all_metrics"])
    lines += ["## 较早历史：2015年1月5日至2019年12月31日开盘", ""] + table(result["earlier_diagnostics"])
    lines += ["原141个训练检查时点中114次新概率拟合全部成功，27个早期支持不足，没有单类或求解失败。主31个周期全部被学习规则退出，只有72个持仓收盘、62笔成交；较早9周期、18笔成交，4次学习退出。", "",
        "相对原学习退出，主基础终值少49,106.49元，其中新增佣金及滑点约2,107.69元；较早少28,590.96元，较早费用反而略少。损失不能全部归咎于交易成本。仅预测是否占优没有保留获利大小，本次实际结果没有支持这项替换。", "",
        f"7项必要测试一次通过，四新账户完成，关键复算覆盖114成熟模型的时间和周期权重、{len(predictions)}条有效概率及连续确认、4账户指标和{len(cycles)}个含分红完整周期。没有重新拟合验证模型或新跑诊断账户。", "",
        "全部月度8项因子均值、尺度、系数和概率中文计算说明见同目录“每月方向概率退出模型中文规则.md”。下一项为按经济错判代价学习退出，目前只有方向，尚未登记或运行。", "", "## 全部中文因子和进出场", ""]
    lines += (ROOT / cfg["rules"]).read_text(encoding="utf-8").splitlines()[1:]
    DOCUMENT.write_text("\n".join(lines)+"\n", encoding="utf-8")
    for p in RESEARCH.iterdir():
        if p.is_file() and p.suffix in {".csv", ".json", ".md"}:
            shutil.copy2(p, OUT / p.name)
    shutil.copy2(CONFIG, OUT / "冻结设置.json")
    shutil.copy2(NEXT, OUT / "下一项经济错判退出方向.md")
    for period, label in [("evaluation", "主评价"), ("earlier_diagnostic", "较早历史")]:
        for cost in cfg["costs"]:
            for kind in ["ledger", "decisions"]:
                pd.read_parquet(RESEARCH / period / cost / f"{PRIMARY}_{kind}.parquet").to_csv(OUT / f"{label}_{cost}_{'完整账户' if kind=='ledger' else '全部因子与进出场'}.csv", index=False, encoding="utf-8-sig")
    record = {"round": 79, "study": result["study_id"], "title": "继续持有占优概率退出", "status": status,
        "result": str((RESEARCH / "result.json").relative_to(ROOT)), **{k: result[k] for k in ["candidate_configurations", "evaluation_accounts", "new_accounts_generated", "reused_control_accounts", "earlier_diagnostic_accounts", "new_earlier_diagnostic_accounts", "new_model_fits", "new_reference_accounts"]},
        "evaluated_candidate_source_runs": 1, "primary_base": metric(result, PRIMARY), "primary_stress": metric(result, PRIMARY, cost="STRESS"), "post_selected_best_base": metric(result, PRIMARY)}
    index["completed_rounds"].append(record)
    for k in ["registered_configurations_in_this_resumption", "evaluated_configurations_in_this_resumption", "evaluated_candidate_source_runs_including_corrected_replays", "registered_candidate_source_runs_including_unrun_legacy_bindings"]:
        index[k] += 1
    index["evaluation_accounts_in_this_resumption"] += result["evaluation_accounts"]
    index.update(updated_at=now(), latest_completed_round=record, running_studies=[], goal_achieved=False, status="ROUND79_COMPLETE_DIRECTIONAL_EXIT_TARGET_NOT_MET",
        count_warning="79轮，353不同设置，368已评价来源版本，373登记含5旧未运行，1210主评价记录。",
        next_work={"status": "ECONOMIC_REGRET_EXIT_DIRECTION_NOT_REGISTERED", "focus": "按错误持有或退出所损失的经济金额学习决策", "source": str(NEXT.relative_to(ROOT))},
        process_state_note="78与79四账户、关键核对及简洁交付均完成；80经济错判代价仅方向，尚未登记或拟合。",
        research_speed_priority="docs/510300_FAST_RESEARCH_CADENCE_20260908.md")
    index["deliveries"].append({"created_at": now(), "type": "DIRECTIONAL_EXIT_ROUND79_FAST_CHINESE_RESULTS", "rounds": [79], "directory": str(OUT), "main_document": str(DOCUMENT), "new_gpt_review_archive_created": False})
    write_json(INDEX, index)
    write_json(OUT / "交付回执.json", {"created_at": now(), "main_document": str(DOCUMENT), "goal_achieved": False, "new_gpt_review_archive_created": False}, exclusive=True)
    print(json.dumps({"交付": str(DOCUMENT), "核对": receipt, "差额": differences}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
