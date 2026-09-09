"""核对新校准的成熟误差与实际成交等同性，简洁交付。"""
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from research.paired_forecast_mse_exit_v1 import ROOT, OUT as RESEARCH, CONFIG, PRIMARY
from research.learned_cycle_exit_v1 import FEATURES
from research.adaptive_allocation_v1 import normalize_dividends, summarize
from research.intraday_overnight_increment_v1 import now, digest, require, write_json
from scripts.review_round74_saved import saved_cycles
from scripts.finalize_round60_20260907 import metric
from scripts.finalize_round96_20260908 import table, value_equal

OUT = ROOT / "deliverables/510300同期预测误差混合退出_第104轮_20260908"
DOCUMENT = OUT / "同期预测误差混合退出_结果及中文规则.md"
INDEX = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
NEXT = ROOT / "docs/510300_AFTER_PAIRED_FORECAST_MSE_EXIT_20260908.md"


def main():
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    require(index["latest_completed_round"]["round"] == 103 and not OUT.exists(), "104前序或交付状态不符")
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    result = json.loads((RESEARCH / "result.json").read_text(encoding="utf-8"))
    pairs = pd.read_parquet(RESEARCH / "paired_reference_forecasts.parquet")
    sample = pd.read_parquet(ROOT / cfg["samples"])
    sample = sample[sample.signal.eq("D60_INTRA")].reset_index(drop=True)
    pd.testing.assert_frame_equal(pairs[["cycle_id", "origin_index", "exit_index", "target"]], sample[["cycle_id", "origin_index", "exit_index", "target"]])
    models = [json.loads((ROOT / cfg["saved_models"]).read_text(encoding="utf-8"))["models"]["D60_INTRA__RIDGE"],
              json.loads((ROOT / cfg["interaction_models"]).read_text(encoding="utf-8"))["models"]]
    pair_checks = 0
    for role, history in zip(["old", "new"], models):
        lookup = {r["fit_index"]: r for r in history}
        for row in pairs.itertuples():
            if getattr(row, f"{role}_status") != "PREDICTION_AVAILABLE":
                require(pd.isna(getattr(row, f"{role}_prediction")), "无模型仍填写预测")
                continue
            model_index = int(getattr(row, f"{role}_fit_index"))
            stored = lookup[model_index]
            require(stored["latest_exit_index"] <= model_index <= row.origin_index and row.cycle_id not in stored["training_cycles"], "重建预测时钟不符")
            require(model_index == max(t for t in lookup if t <= row.origin_index), "没有使用原点最近模型")
            x = sample.iloc[row.Index][FEATURES].to_numpy(float)
            if role == "new":
                x = np.r_[x, max(x[1], 0.)*x[2]]
            m = stored["model"]
            predicted = float(m["intercept"]+np.clip((x-m["mean"])/m["scale"], -m["feature_clip"], m["feature_clip"])@np.asarray(m["coefficients"]))
            require(abs(predicted-getattr(row, f"{role}_prediction")) < 1e-12 and abs(predicted-row.target-getattr(row, f"{role}_error")) < 1e-12, "保存同期预测或误差不能复算")
            pair_checks += 1
    calibration = json.loads((RESEARCH / "saved_calibrations.json").read_text(encoding="utf-8"))["calibrations"]
    members = pd.read_parquet(RESEARCH / "calibration_memberships.parquet")
    cycle_info = pairs.groupby("cycle_id").agg(exit_index=("exit_index", "first"), all_paired=("both_available", "all"))
    previous, checks = np.array([1., 0.]), []
    for r in calibration:
        candidates = cycle_info[cycle_info.all_paired & cycle_info.exit_index.le(r["fit_index"])].reset_index().sort_values(["exit_index", "cycle_id"]).tail(cfg["recent_cycles"])
        require(candidates.cycle_id.to_list() == r["training_cycles"], "校准成员不是最近完整成熟同期周期")
        rows = pairs[pairs.cycle_id.isin(r["training_cycles"])].sort_values(["cycle_id", "origin_index"])
        require(len(rows) == r["training_rows"], "校准状态数量改变")
        weights = np.array([1./len(rows[rows.cycle_id.eq(c)]) for c in rows.cycle_id])
        if r["calibration_status"] == "CALIBRATION_COMPLETE":
            chosen = members[members.fit_index.eq(r["fit_index"])].sort_values(["cycle_id", "origin_index"])
            np.testing.assert_allclose(chosen.sample_weight, weights, atol=0, rtol=0)
            a, b = rows.old_error.to_numpy(), rows.new_error.to_numpy()
            denominator = float(np.sum(weights*(a-b)**2)/weights.sum())
            raw = float(np.sum(weights*b*(b-a))/np.sum(weights*(a-b)**2))
            expected = float(np.clip(raw, 0., 1.))
            require(abs(r["old_weight"]-expected) < 1e-11 and abs(r["difference_mse"]-denominator) < 1e-12, "未去均值误差最小化权重错误")
            loss = float(np.average((expected*a+(1.-expected)*b)**2, weights=weights))
            require(loss <= min(r["old_mse"], r["new_mse"])+1e-12 and abs(loss-r["mixed_mse"]) < 1e-12, "混合训练误差不是约束最小值")
            previous = np.array([r["old_weight"], r["new_weight"]])
            checks.append({"fit_origin": r["fit_origin"], "complete_cycles": len(candidates), "rows": len(rows), "old_weight": expected, "mixed_mse": loss})
        else:
            np.testing.assert_allclose([r["old_weight"], r["new_weight"]], previous, atol=0, rtol=0)
    require(len(checks) == 66 and checks[0]["fit_origin"] == "2021-03-01", "本轮校准支持结果不符")
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    cycle_rows, account_checks = [], []
    c_lookup = {r["fit_index"]: r for r in calibration}
    predictions = 0
    for period in ["evaluation", "earlier_diagnostic"]:
        for cost in cfg["costs"]:
            folder = RESEARCH / period / cost
            ledger = pd.read_parquet(folder / f"{PRIMARY}_ledger.parquet")
            old = pd.read_parquet(folder / "REARM_RIDGE_ledger.parquet")
            pd.testing.assert_frame_equal(ledger, old)
            decisions = pd.read_parquet(folder / f"{PRIMARY}_decisions.parquet")
            holdings = decisions[decisions.learning_cycle_id.notna()]
            for cycle_id, group in holdings.groupby("learning_cycle_id"):
                count = 0
                for row in group.itertuples():
                    if pd.notna(row.calibration_fit_index):
                        cal = c_lookup[int(row.calibration_fit_index)]
                        require(cal["fit_index"] <= row.origin_index and cal["fit_index"] == max(t for t in c_lookup if t <= row.origin_index), "实际调用未来混合权重")
                        np.testing.assert_allclose([row.old_weight, row.new_weight], [cal["old_weight"], cal["new_weight"]], atol=0, rtol=0)
                    required = [(row.old_prediction, row.old_weight), (row.new_prediction, row.new_weight)]
                    available = all(pd.notna(v) for v, w in required if w > 0)
                    value = sum(v*w for v, w in required if w > 0) if available else None
                    if value is not None:
                        require(abs(value-row.continuation_prediction) < 1e-12 and (value < 0) == (row.old_prediction < 0), "实际混合预测或相对旧预测方向不符")
                        count = count+1 if value < 0 else 0
                        predictions += 1
                    else:
                        require(pd.isna(row.continuation_prediction), "必需预测未知仍填混合值")
                        count = 0
                    require(count == row.negative_confirmation_count and bool(row.learned_exit_requested) == (count >= 2), "混合确认或新周期重置错误")
            if period == "earlier_diagnostic":
                require(holdings.old_weight.eq(1).all() and holdings.new_weight.eq(0).all(), "较早历史使用尚未支持的新权重")
            actual = summarize(ledger, cfg)
            saved = metric(result, PRIMARY, period, cost)
            for field in ["net_sharpe", "annualized_return", "max_drawdown", "cumulative_return", "trade_count", "commission", "slippage_cost"]:
                require(value_equal(actual[field], saved[field]), "保存完整账户指标不符")
            cycle_rows.extend({"period": period, "cost": cost, **r} for r in saved_cycles(ledger, dividends, cfg))
            account_checks.append({"period": period, "cost": cost, "entire_ledger_identical_to_old": True, "known_prediction_sign_changes": 0,
                                   "days": len(ledger), "holding_closes": len(holdings), "terminal_nav_difference": float(ledger.equity.iloc[-1]-old.equity.iloc[-1])})
    for name, rows in [("saved_calibration_checks.csv", checks), ("saved_actual_cycles.csv", cycle_rows), ("saved_account_equivalence.csv", account_checks)]:
        pd.DataFrame(rows).to_csv(RESEARCH / name, index=False, encoding="utf-8-sig")
    receipt = {"verified_at": now(), "status": "PAIRED_MATURE_MSE_CALIBRATION_AND_EXACT_OLD_ACCOUNT_EQUIVALENCE_VERIFIED",
               "historical_prediction_checks": pair_checks, "calibrations_checked": len(checks), "actual_mixed_predictions_checked": predictions,
               "identical_old_accounts": 4, "complete_cycles": len(cycle_rows), "new_accounts_or_models": 0,
               "source_sha256": digest(Path(__file__)), "security_audit_performed": False}
    write_json(RESEARCH / "saved_verification_receipt.json", receipt, exclusive=True)
    decision = "第104轮未达标。主基础／压力夏普0.705／0.641，较早0.748／0.725，四个完整成交账本均与旧八项退出完全相同。66次成熟误差校准中，51次内点混合、7次仅旧模型、8次仅新模型；权重确实变化，但已知实际持仓预测没有一次改变正负方向，故进出和收益均无改善。关闭这一混合设置，不继续改误差窗口、权重约束或确认天数。"
    status = "COMPLETED_PAIRED_MSE_WEIGHTS_CHANGED_ACCOUNTS_IDENTICAL_NOT_TARGET"
    write_json(RESEARCH / "acceptance_outcome.json", {"recorded_at": now(), "status": status, "decision": decision, "goal_achieved": False, "position_impact": 0}, exclusive=True)
    NEXT.write_text("\n".join(["# 第104轮之后：直接改变市场状态与进入机会", "", decision, "",
        "四个新账户约4.18秒；七项必要测试4.52秒。零新预测模型拟合、零新参考账户，2022条同期预测、66月校准及694条实际混合预测、四账本66周期已核。原1461状态全部保留，早期204个持仓收盘无成熟退出模型。主基础年化5.3843%、最大回撤10.5912%；较早8.6366%、13.7916%。", "",
        "下一105拟用价格路径曲折度识别趋势／震荡，直接改变进入机会，避免继续在实际退出完全相同的预测上加工。已限定检索research和docs的Choppiness、CHOP、Hurst等，没有旧同名曲折度实现；原效率、方差比、波动路由已经失败，保持旧结论。新方向不称为尚未研究过状态切换。", "",
        "方法出处：https://www.tradingview.com/support/solutions/43000501980-choppiness-index-chop/ 。官方定义为十四日真实波幅合计与同期最高价减最低价之比，取十进对数再除以十四的十进对数，乘一百；常用界限38.2与61.8。指标只辨路径曲折，不给涨跌方向，也不证明策略收益。", "",
        "拟只登记一项因果记忆状态：曲折度低于38.2为趋势，高于61.8为震荡，两者之间保留此前已知状态，初始及缺失重置为无新状态。趋势态配合原长期方向与二十日突破进入；震荡态配合原偏离均值且当日回升进入。各持仓仍保留进入时的模式和该模式原价格保护；市场出现相反有效状态时下一开盘退出。使用含分红连续高低价计算真实波幅；缺失不填零，不把未知当空仓。", "",
        "先明确全部原阈值、退出、缺失、成本和持仓状态再冻结一个设置；只测试新曲折度数学、因果状态、进出与旧账户接口，随即四账户，不训练、不补源、不加GPT包。当前105仍为待登记方向，尚无新配置或账户。"])+"\n", encoding="utf-8")
    OUT.mkdir(parents=True)
    DOCUMENT.write_text("\n".join(["# 第104轮：同期预测误差混合退出", "", decision, "", "## 主历史：2020年1月2日至2026年8月14日开盘", "", *table(result["all_metrics"]),
        "## 较早历史：2015年1月5日至2019年12月31日开盘", "", *table(result["earlier_diagnostics"]),
        "七项必要测试通过，66次校准和四个完整账户已核。主历史每个费用252个持仓收盘，其中156个采用正的新模型权重、126个是两模型共同混合；但所有已知预测与旧模型方向一致。较早历史没有足够同期成熟样本，全部沿用旧八项基线。该机制不能改善实际收益，下一转向直接改变进入机会的规则。", "",
        *(ROOT / cfg["rules"]).read_text(encoding="utf-8").splitlines()[1:]])+"\n", encoding="utf-8")
    for p in RESEARCH.iterdir():
        if p.is_file() and p.suffix in {".csv", ".json"}:
            shutil.copy2(p, OUT / p.name)
    shutil.copy2(CONFIG, OUT / "冻结设置.json")
    shutil.copy2(NEXT, OUT / "下一项研究方向.md")
    for period, label in [("evaluation", "主历史"), ("earlier_diagnostic", "较早历史")]:
        for cost in cfg["costs"]:
            for kind in ["ledger", "decisions"]:
                pd.read_parquet(RESEARCH / period / cost / f"{PRIMARY}_{kind}.parquet").to_csv(OUT / f"{label}_{cost}_{kind}.csv", index=False, encoding="utf-8-sig")
    record = {"round": 104, "study": result["study_id"], "title": "两套退出模型的成熟同期误差混合", "status": status, "result": str((RESEARCH / "result.json").relative_to(ROOT)),
              **{k: result[k] for k in ["candidate_configurations", "evaluation_accounts", "new_accounts_generated", "reused_control_accounts", "earlier_diagnostic_accounts", "new_earlier_diagnostic_accounts", "new_model_fits", "new_reference_accounts", "new_calibration_estimates"]},
              "evaluated_candidate_source_runs": 1, "primary_base": metric(result, PRIMARY), "primary_stress": metric(result, PRIMARY, cost="STRESS"), "post_selected_best_base": metric(result, PRIMARY)}
    index["completed_rounds"].append(record)
    for key in ["registered_configurations_in_this_resumption", "evaluated_configurations_in_this_resumption", "evaluated_candidate_source_runs_including_corrected_replays", "registered_candidate_source_runs_including_unrun_legacy_bindings"]:
        index[key] += 1
    index["evaluation_accounts_in_this_resumption"] += result["evaluation_accounts"]
    index.update(updated_at=now(), latest_completed_round=record, running_studies=[], goal_achieved=False, status="ROUND104_PAIRED_MSE_NO_ACCOUNT_IMPROVEMENT_FULL_GOAL_NOT_MET",
                 count_warning="104轮，380不同设置，396已评价来源版本，401登记含5旧未运行，1480主评价记录。",
                 next_work={"status": "CHOPPINESS_STATE_ENTRY_DIRECTION_NOT_REGISTERED", "focus": "路径曲折度的因果记忆状态，直接改变趋势或震荡进入机会", "source": str(NEXT.relative_to(ROOT))},
                 process_state_note="104七测试、66次误差校准、四新账户与保存核对完成；105尚未登记或运行。")
    index["deliveries"].append({"created_at": now(), "type": "PAIRED_MSE_EXIT_ROUND104_FAST_CHINESE_RESULTS", "rounds": [104], "directory": str(OUT), "main_document": str(DOCUMENT), "new_gpt_review_archive_created": False})
    write_json(INDEX, index)
    write_json(OUT / "交付回执.json", {"created_at": now(), "main_document": str(DOCUMENT), "goal_achieved": False, "new_gpt_review_archive_created": False}, exclusive=True)
    print(json.dumps({"交付": str(DOCUMENT), "核对": receipt}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
