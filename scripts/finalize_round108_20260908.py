"""核对保存的共同成熟时段、分层权重和实际账户，交付第108轮。"""
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from research.shared_episode_continuation_v1 import ROOT, OUT as RESEARCH, CONFIG, PRIMARY, P31
from research.shared_episode_continuation_inputs_v1 import FEATURES, SIGNALS
from research.adaptive_allocation_v1 import normalize_dividends, summarize
from research.intraday_overnight_increment_v1 import now, require, write_json, digest
from scripts.review_round74_saved import saved_cycles
from scripts.finalize_round60_20260907 import metric
from scripts.finalize_round96_20260908 import table

OUT = ROOT / "deliverables/510300共同持仓时段共享退出_第108轮_20260908"
DOCUMENT = OUT / "共同持仓时段共享退出_结果及全部中文规则.md"
INDEX = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
NEXT = ROOT / "docs/510300_AFTER_SHARED_EPISODE_EXIT_20260908.md"


def main():
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    require(index["latest_completed_round"]["round"] == 107 and not OUT.exists(), "108前序或交付状态不符")
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    result = json.loads((RESEARCH / "result.json").read_text(encoding="utf-8"))
    require(not result["historical_point_target_met"], "出现目标值时不能直接判为未达标")
    models = json.loads((RESEARCH / "saved_models.json").read_text(encoding="utf-8"))["models"]
    by_date = {pd.Timestamp(m["fit_origin"]): m for m in models}
    data = pd.read_parquet(ROOT / cfg["features"])
    original = pd.read_parquet(ROOT / cfg["samples"])
    samples = pd.read_parquet(RESEARCH / "grouped_reference_samples.parquet")
    groups = pd.read_parquet(RESEARCH / "reference_calendar_groups.parquet")
    members = pd.read_parquet(RESEARCH / "training_memberships.parquet")
    pd.testing.assert_frame_equal(samples[original.columns], original.reset_index(drop=True))
    require(len(samples) == 2458 and len(groups) == 75 and groups.group_mature_date.notna().sum() == 74, "共享原状态或时段数量不同")
    np.testing.assert_array_equal(samples.task_trend_rebound, samples.signal.eq("S1_TREND_REBOUND").astype(float))
    np.testing.assert_array_equal(samples.task_z_recovery, samples.signal.eq("R2_Z_CONFIRM").astype(float))
    ledgers = {s: pd.read_parquet(P31 / "reference" / f"{s}_ledger.parquet") for s in SIGNALS}
    dates = pd.DatetimeIndex(ledgers[SIGNALS[0]].date)
    for ledger in ledgers.values():
        require(pd.DatetimeIndex(ledger.date).equals(dates) and np.isfinite(ledger.shares).all(), "参考完整日历或每日份额缺失")
    held = np.column_stack([l.shares.to_numpy() > 0 for l in ledgers.values()]).any(axis=1)
    starts = np.flatnonzero(held & ~np.r_[False, held[:-1]])
    ends = np.flatnonzero(~held & np.r_[False, held[:-1]])
    require(len(starts) == len(groups), "共享时段起点数量不符")
    for g, start in zip(groups.itertuples(), starts, strict=True):
        following = ends[ends > start]
        finish = int(following[0]) if len(following) else None
        require(g.group_start_date == dates[start], "共享持仓时段起点不是实际持仓并集")
        if finish is None or dates[finish] == pd.Timestamp(cfg["data_cutoff"]):
            require(pd.isna(g.group_mature_date), "未自然结束的共同持仓时段被标成熟")
        else:
            require(g.group_mature_date == dates[finish], "共同持仓时段未在全部实际空仓时结束")
        selected = samples[samples.group_id.eq(g.group_id)]
        require(selected.origin.ge(dates[start]).all(), "共同持仓时段包含起点之前状态")
        if finish is not None:
            require(selected.origin.lt(dates[finish]).all(), "共同持仓时段包含全空仓之后状态")
    old_schedule = json.loads((ROOT / cfg["saved_models"]).read_text(encoding="utf-8"))["models"]["D60_INTRA__RIDGE"]
    require([m["fit_index"] for m in models] == [m["fit_index"] for m in old_schedule], "共享模型改动原月度日程")
    training_checks = []
    for record in models:
        t = record["fit_index"]
        eligible = samples[samples.group_mature_date.notna() & samples.group_mature_date.le(data.date.iloc[t])]
        ids = eligible[["group_id", "group_mature_date"]].drop_duplicates().sort_values(["group_mature_date", "group_id"]).tail(20).group_id.to_list()
        rows = eligible[eligible.group_id.isin(ids)].sort_values(["group_id", "signal", "cycle_id", "origin_index"])
        supported = len(ids) >= 10 and len(rows) >= 100 and set(rows.signal) == set(SIGNALS)
        require(ids == record["training_groups"] and supported == record["eligible_for_fit"], "共享训练成员或支持条件不符")
        require(supported == (record["status"] == "FIT_COMPLETE"), "共享训练状态与支持条件不符")
        if not supported:
            continue
        saved = members[members.fit_index.eq(t)].sort_values(["group_id", "signal", "cycle_id", "origin_index"])
        keys = ["group_id", "signal", "source_cycle_key", "cycle_id", "origin_index", "exit_index", "group_mature_date"]
        pd.testing.assert_frame_equal(rows[keys].reset_index(drop=True), saved[keys].reset_index(drop=True))
        w = np.zeros(len(rows))
        positions = pd.Series(np.arange(len(rows)), index=rows.index)
        for _, group in rows.groupby("group_id"):
            cycles_in_group = group.source_cycle_key.nunique()
            for _, cycle in group.groupby("source_cycle_key"):
                w[positions.loc[cycle.index].to_numpy()] = 1./cycles_in_group/len(cycle)
        np.testing.assert_allclose(saved.sample_weight, w, atol=1e-15, rtol=0)
        np.testing.assert_allclose(saved.groupby("group_id").sample_weight.sum(), 1., atol=1e-12, rtol=0)
        require(record["latest_exit_index"] <= record["latest_group_mature_index"] <= t, "共享成熟证据使用未来信息")
        require(pd.Timestamp(record["latest_group_mature_date"]) == rows.group_mature_date.max(), "共享最新成熟日期不符")
        x = rows[FEATURES].to_numpy(float)
        model = record["model"]
        mean = np.average(x, axis=0, weights=w)
        scale = np.sqrt(np.average((x-mean)**2, axis=0, weights=w))
        scale = np.where(scale > 1e-12, scale, 1.)
        np.testing.assert_allclose(model["mean"], mean, atol=1e-12, rtol=0)
        np.testing.assert_allclose(model["scale"], scale, atol=1e-12, rtol=0)
        z = np.clip((x-mean)/scale, -5, 5)
        coef = np.asarray(model["coefficients"])
        residual = model["intercept"]+z@coef-rows.target.to_numpy()
        gradient = z.T@(w*residual)+cfg["ridge_alpha"]*coef
        require(abs(np.sum(w*residual)) < 1e-10 and np.max(np.abs(gradient)) < 1e-10, "共享系数不满足冻结加权岭最优条件")
        training_checks.append({"fit_origin": record["fit_origin"], "groups": len(ids), "cycles": rows.source_cycle_key.nunique(), "rows": len(rows), "largest_gradient": np.max(np.abs(gradient))})
    require(len(training_checks) == 138 and result["failed_fits"] == 0, "共享训练完成数量不符")
    require(training_checks[0]["fit_origin"] == "2015-03-02", "共享首次支持日期不符")
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    cycles, checked, differences, predictions = [], [], [], []
    for period in ["evaluation", "earlier_diagnostic"]:
        for cost in cfg["costs"]:
            folder = RESEARCH / period / cost
            ledger = pd.read_parquet(folder / f"{PRIMARY}_ledger.parquet")
            decisions = pd.read_parquet(folder / f"{PRIMARY}_decisions.parquet")
            native = pd.read_csv(folder / f"{PRIMARY}_cycles.csv")
            for cycle_id, group in decisions[decisions.learning_cycle_id.notna()].groupby("learning_cycle_id", sort=False):
                count = 0
                cycle = native[native.cycle_id.eq(cycle_id)].iloc[0]
                owned = ledger[ledger.cycle_id.eq(cycle_id)].copy().set_index("date")
                values = owned.shares*owned.mark+owned.dividend_recognized.cumsum()
                peaks = np.maximum.accumulate(np.r_[cycle.entry_cost_cny, values.to_numpy()])[1:]
                peak_by_date = dict(zip(owned.index, peaks))
                for row in group.itertuples():
                    actual = values.loc[row.origin]
                    expected = [np.log1p(row.origin_index-cycle.entry_index+1), actual/cycle.entry_cost_cny-1, actual/peak_by_date[row.origin]-1, cycle["mode"],
                                data.mom5.iloc[row.origin_index], data.mom20.iloc[row.origin_index], data.sma120.iloc[row.origin_index], data.vol20.iloc[row.origin_index], 0., 0.]
                    x = np.array([getattr(row, c) for c in FEATURES])
                    np.testing.assert_allclose(x, expected, atol=1e-12, rtol=0)
                    require(row.prediction_task == "D60_INTRA", "实际账户调用其他参考任务标记")
                    if row.learning_status == "PREDICTION_AVAILABLE":
                        m = by_date[row.learning_fit_origin]
                        require(m["latest_exit_index"] <= m["latest_group_mature_index"] <= m["fit_index"] <= row.origin_index, "共享实际调用未成熟或未来模型")
                        require(m["fit_index"] == max(r["fit_index"] for r in models if r["fit_index"] <= row.origin_index), "实际预测不是最近可知月度记录")
                        z = np.clip((x-m["model"]["mean"])/m["model"]["scale"], -5, 5)
                        score = float(m["model"]["intercept"]+z@np.array(m["model"]["coefficients"]))
                        require(abs(score-row.continuation_prediction) < 1e-12, "共享保存预测不能复算")
                        count = count+1 if score < 0 else 0
                        predictions.append({"period": period, "cost": cost, "origin": row.origin, "cycle": cycle_id, "continuation_prediction": score})
                    else:
                        require(pd.isna(row.continuation_prediction), "共享无模型时仍填预测")
                        count = 0
                    require(count == row.negative_confirmation_count and bool(row.learned_exit_requested) == (count >= 2), "共享确认计数或周期重置不符")
            m = metric(result, PRIMARY, period, cost)
            control = pd.read_parquet(folder / "REARM_RIDGE_ledger.parquet")
            require(pd.DatetimeIndex(ledger.date).equals(pd.DatetimeIndex(control.date)), "共享新旧账户日历不同")
            previous = np.r_[cfg["initial_capital"], ledger.equity.iloc[:-1].to_numpy()]
            np.testing.assert_allclose(ledger.cash+ledger.shares*ledger.mark+ledger.dividend_receivable, ledger.equity, atol=1e-6, rtol=0)
            np.testing.assert_allclose(ledger.equity/previous-1, ledger.net_return, atol=1e-12, rtol=0)
            recomputed = summarize(ledger, cfg)
            for field in ["net_sharpe", "annualized_return", "max_drawdown", "cumulative_return", "trade_count", "commission", "slippage_cost"]:
                require(abs(recomputed[field]-m[field]) < 1e-9, "共享账户保存指标不符")
            require(not ledger.terminal_unliquidated.iloc[-1], "共同学习终点仍有未平仓份额")
            cycles.extend({"period": period, "cost": cost, **c} for c in saved_cycles(ledger, dividends, cfg))
            checked.append({"period": period, "cost": cost, "days": len(ledger), "net_sharpe": m["net_sharpe"]})
            differences.append({"period": period, "cost": cost, "control": "REARM_RIDGE", "terminal_nav_difference": float(ledger.equity.iloc[-1]-control.equity.iloc[-1]),
                                **{f"{c}_difference": float(ledger[c].sum()-control[c].sum()) for c in ["price_pnl", "dividend_recognized", "commission", "slippage_cost"]}})
    for name, rows in [("saved_actual_cycles.csv", cycles), ("saved_prediction_checks.csv", predictions), ("saved_profit_differences.csv", differences), ("saved_training_checks.csv", training_checks)]:
        pd.DataFrame(rows).to_csv(RESEARCH / name, index=False, encoding="utf-8-sig")
    receipt = {"verified_at": now(), "status": "SAVED_SHARED_EPISODE_RIDGE_AND_ACCOUNT_CHECKS_COMPLETE", "mature_models": len(training_checks), "original_reference_rows_preserved": len(samples),
               "calendar_episodes": len(groups), "natural_calendar_episodes": 74, "actual_holding_state_rows": sum(r["holding_decisions"] for r in result["model_coverage"]),
               "saved_scores_recomputed": len(predictions), "new_account_metrics_recomputed": len(checked), "complete_cycles": len(cycles), "new_accounts_or_models": 0,
               "reviewer_source_sha256": digest(Path(__file__)), "security_audit_performed": False}
    write_json(RESEARCH / "saved_verification_receipt.json", receipt, exclusive=True)
    write_json(RESEARCH / "console_wording_clarification.json", {"recorded_at": now(), "original_phrase": "原成熟时点不变", "correct_interpretation": "原141个月度拟合日程不变。共享样本支持的首次成熟拟合从2017年3月1日提前至2015年3月2日，实际完成138个模型。原控制台句首描述不准确，以冻结规则、逐月成员及保存模型日期为准。", "source_or_accounts_rerun": False}, exclusive=True)
    status = "COMPLETED_SHARED_EPISODE_CONTINUATION_WORSE_BOTH_PERIODS_NOT_TARGET"
    decision = "第108轮未达标。主基础／压力夏普0.526／0.460，较早0.618／0.594；两段均低于原八项线性退出。主基础年化4.29%、最大回撤12.52%；较早年化6.77%、最大回撤13.79%。共享三类成熟参考虽然让模型更早可用，却没有改善收益风险，关闭此固定共享结构。"
    write_json(RESEARCH / "acceptance_outcome.json", {"recorded_at": now(), "status": status, "decision": decision, "goal_achieved": False, "position_impact": 0}, exclusive=True)
    NEXT.write_text("\n".join(["# 第108轮之后的快速研究", "", decision, "",
        "六项必要测试3.43秒，138次共享拟合及四个实际账户约5.20秒；3个月度原点不足，零求解失败，零新参考账户。2458条原样本、75共同时段、74自然成熟时段、分层权重、十项加权岭、实际持仓输入及四账户保存值已核，不重跑。", "",
        "原141个月度日程保留，第一次支持从2017年3月1日提前至2015年3月2日；控制台原成熟时点不变的短语已另存澄清回执，原规则与保存模型日期正确。较早每个账户只有19个持仓收盘无共享模型，而原方案204个无模型；可用性增加并没有转化为改善。", "",
        "106非线性提升、107隔夜下行风险占比、108共享继续价值均关闭；不调整树深、下行窗口、任务组合、成熟时段权重或惩罚挽救。104混合误差没有改变交易，105曲折度状态切换变差，同样不重开。", "",
        "下一步先做一次有界规则检索，优先选择已有数据支持、进入或完整机会定义不同的简单方案。只改变一个明确机制并在新收益前冻结，复用成熟账户引擎和保存对照。尚未选定或登记第109轮，不能写成正在训练。", "",
        "保留较早历史、主历史和压力费用；不补EPS、公募和慢源，不生成GPT审阅数值包，不做额外安全审计。仅510300及现金研究，完整目标未达成。"])+"\n", encoding="utf-8")
    OUT.mkdir(parents=True)
    DOCUMENT.write_text("\n".join(["# 第108轮：共同持仓时段共享退出", "", decision, "", "## 主历史：2020年1月2日至2026年8月14日开盘", "", *table(result["all_metrics"]),
        "## 较早历史：2015年1月5日至2019年12月31日开盘", "", *table(result["earlier_diagnostics"]),
        "六项必要测试通过。138个成熟模型与四账户计算约5.20秒，零新参考账户。全部2458条原状态保留，共同成熟时点、分层权重、每月十项加权岭、真实持仓状态和完整账户已核。模型更早可用不等于交易更好；本轮较早每种费用仅19个持仓收盘没有模型，原基线为204个，但夏普和年化收益均降低。", "",
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
    record = {"round": 108, "study": result["study_id"], "title": "三类自然参考按共同持仓时段共享继续价值", "status": status, "result": str((RESEARCH / "result.json").relative_to(ROOT)),
              **{k: result[k] for k in ["candidate_configurations", "evaluation_accounts", "new_accounts_generated", "reused_control_accounts", "earlier_diagnostic_accounts", "new_earlier_diagnostic_accounts", "new_model_fits", "new_reference_accounts"]},
              "evaluated_candidate_source_runs": 1, "primary_base": metric(result, PRIMARY), "primary_stress": metric(result, PRIMARY, cost="STRESS"), "post_selected_best_base": metric(result, PRIMARY)}
    index["completed_rounds"].append(record)
    for key in ["registered_configurations_in_this_resumption", "evaluated_configurations_in_this_resumption", "evaluated_candidate_source_runs_including_corrected_replays", "registered_candidate_source_runs_including_unrun_legacy_bindings"]:
        index[key] += 1
    index["evaluation_accounts_in_this_resumption"] += result["evaluation_accounts"]
    index.update(updated_at=now(), latest_completed_round=record, running_studies=[], goal_achieved=False, status="ROUND108_SHARED_EPISODE_NO_TARGET_FULL_GOAL_NOT_MET",
                 count_warning="108轮，384不同设置，400已评价来源版本，405登记含5旧未运行，1518主评价记录。",
                 next_work={"status": "NEXT_DISTINCT_SIMPLE_ENTRY_MECHANISM_NOT_SELECTED", "focus": "关闭共享退出，按已有数据支持检索下一个不同机制", "source": str(NEXT.relative_to(ROOT))},
                 process_state_note="108的六项测试、138成熟拟合、四个实际账户与保存核对交付完成；下一项尚未登记。")
    index["deliveries"].append({"created_at": now(), "type": "SHARED_EPISODE_ROUND108_FAST_CHINESE_RESULTS", "rounds": [108], "directory": str(OUT), "main_document": str(DOCUMENT), "new_gpt_review_archive_created": False})
    write_json(INDEX, index)
    write_json(OUT / "交付回执.json", {"created_at": now(), "main_document": str(DOCUMENT), "goal_achieved": False, "new_gpt_review_archive_created": False}, exclusive=True)
    print(json.dumps({"交付": str(DOCUMENT), "核对": receipt}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
