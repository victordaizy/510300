"""保存路径复算高点时间和原权重岭回归，交付第113轮。"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from research.peak_age_exit_v1 import ROOT, OUT, CONFIG, PRIMARY, P31
from research.peak_age_exit_inputs_v1 import FEATURES, ADDED
from research.learned_cycle_exit_v1 import training_rows
from research.adaptive_allocation_v1 import normalize_dividends, summarize
from research.intraday_overnight_increment_v1 import now, digest, require, write_json
from research.fast_round_delivery_v1 import deliver_round
from scripts.review_round74_saved import saved_cycles
from scripts.finalize_round60_20260907 import metric
from scripts.finalize_round96_20260908 import value_equal


def value_age_rows(ledger, cycles, data):
    records = []
    for cycle in cycles.itertuples():
        owned = ledger[ledger.cycle_id.eq(cycle.cycle_id)]
        peak, peak_day, dividends = float(cycle.entry_cost_cny), int(cycle.entry_index)-1, 0.
        for row in owned.itertuples():
            t = int(data.date.searchsorted(row.date))
            dividends += row.dividend_recognized
            current = row.shares*row.mark+dividends
            if current >= peak:
                peak, peak_day = current, t
            records.append({"cycle_id": int(cycle.cycle_id), "origin_index": t, "origin": row.date, "days_since_peak": t-peak_day, ADDED: float(np.log1p(t-peak_day)),
                            "peak_origin_index": peak_day, "cycle_return": current/cycle.entry_cost_cny-1, "cycle_drawdown": current/peak-1})
    return pd.DataFrame(records)


def main():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    result = json.loads((OUT / "result.json").read_text(encoding="utf-8"))
    require(not result["historical_point_target_met"], "出现目标点值时需要重新判断，不能直接归档失败")
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "高点时间冻结来源改变")
    data = pd.read_parquet(ROOT / cfg["features"])
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    reference_ledger = pd.read_parquet(P31 / "reference/D60_INTRA_ledger.parquet")
    reference_cycles = pd.read_csv(ROOT / cfg["reference_cycles"])
    expected = value_age_rows(reference_ledger, reference_cycles, data)
    states = pd.read_parquet(OUT / "reference_peak_states.parquet")
    joined = states.merge(expected, on=["cycle_id", "origin_index", "origin"], validate="one_to_one", suffixes=("", "_expected"))
    require(len(joined) == len(states), "参考高点时间丢失实际持仓状态")
    for key in ["days_since_peak", ADDED, "peak_origin_index"]:
        np.testing.assert_allclose(joined[key], joined[f"{key}_expected"], atol=1e-12, rtol=0)
    original_samples = pd.read_parquet(ROOT / cfg["samples"])
    original_samples = original_samples[original_samples.signal.eq("D60_INTRA")].reset_index(drop=True)
    samples = pd.read_parquet(OUT / "extended_reference_samples.parquet")
    pd.testing.assert_frame_equal(original_samples, samples[original_samples.columns])
    sample_join = samples.merge(expected[["cycle_id", "origin_index", ADDED]], on=["cycle_id", "origin_index"], validate="one_to_one", suffixes=("", "_expected"))
    np.testing.assert_allclose(sample_join[ADDED], sample_join[f"{ADDED}_expected"], atol=1e-12, rtol=0)
    models = json.loads((OUT / "saved_models.json").read_text(encoding="utf-8"))["models"]
    originals = json.loads((ROOT / cfg["saved_models"]).read_text(encoding="utf-8"))["models"]["D60_INTRA__RIDGE"]
    require([m["fit_index"] for m in models] == [m["fit_index"] for m in originals], "原月度时点改变")
    members = pd.read_parquet(OUT / "training_memberships.parquet")
    models_by_date, model_checks = {}, []
    for record, old in zip(models, originals, strict=True):
        require(record["training_cycles"] == old["training_cycles"] and record["training_rows"] == old["training_rows"] and record["status"] == old["status"], "高点时间改变原成熟支持或样本")
        models_by_date[pd.Timestamp(record["fit_origin"])] = record
        if record["model"] is None:
            continue
        rows, ids = training_rows(samples, record["fit_index"], cfg)
        require(ids == record["training_cycles"] and rows.exit_index.le(record["fit_index"]).all(), "高点时间训练读入未来周期")
        actual_members = members[members.fit_index.eq(record["fit_index"])].sort_values(["cycle_id", "origin_index"])
        np.testing.assert_array_equal(actual_members[["cycle_id", "origin_index"]], rows[["cycle_id", "origin_index"]])
        np.testing.assert_allclose(actual_members.sample_weight, rows.sample_weight, atol=0, rtol=0)
        x = rows[FEATURES].to_numpy(float); w = rows.sample_weight.to_numpy(float)
        mean = np.average(x, axis=0, weights=w); scale = np.sqrt(np.average((x-mean)**2, axis=0, weights=w)); scale = np.where(scale > 1e-12, scale, 1.)
        m = record["model"]
        np.testing.assert_allclose(m["mean"], mean, atol=1e-12, rtol=0)
        np.testing.assert_allclose(m["scale"], scale, atol=1e-12, rtol=0)
        np.testing.assert_allclose(m["mean"][:8], old["model"]["mean"], atol=1e-12, rtol=0)
        np.testing.assert_allclose(m["scale"][:8], old["model"]["scale"], atol=1e-12, rtol=0)
        z = np.clip((x-mean)/scale, -5, 5); coefficient = np.asarray(m["coefficients"])
        residual = m["intercept"]+z@coefficient-rows.target.to_numpy()
        gradient = z.T@(w*residual)+coefficient
        require(abs(np.sum(w*residual)) < 1e-10 and np.abs(gradient).max() < 1e-10, "保存模型不满足冻结岭回归方程")
        model_checks.append({"fit_origin": record["fit_origin"], "cycles": len(ids), "rows": len(rows), "largest_gradient": float(np.abs(gradient).max()), "peak_age_coefficient": coefficient[-1]})
    accounts, cycles, differences = [], [], []
    score_count, state_count = 0, 0
    for period in ["evaluation", "earlier_diagnostic"]:
        for cost in cfg["costs"]:
            folder = OUT / period / cost
            ledger = pd.read_parquet(folder / f"{PRIMARY}_ledger.parquet")
            decisions = pd.read_parquet(folder / f"{PRIMARY}_decisions.parquet")
            native = pd.read_csv(folder / f"{PRIMARY}_cycles.csv")
            expected_own = value_age_rows(ledger, native, data).set_index(["cycle_id", "origin_index"])
            for cycle_id, group in decisions[decisions.learning_cycle_id.notna()].groupby("learning_cycle_id", sort=False):
                count = 0
                for row in group.itertuples():
                    own = expected_own.loc[(int(cycle_id), row.origin_index)]
                    np.testing.assert_allclose([row.days_since_peak, getattr(row, ADDED), row.peak_origin_index, row.cycle_return, row.cycle_drawdown],
                        [own.days_since_peak, own[ADDED], own.peak_origin_index, own.cycle_return, own.cycle_drawdown], atol=1e-12, rtol=0)
                    if row.learning_status == "PREDICTION_AVAILABLE":
                        record = models_by_date[row.learning_fit_origin]
                        require(record["latest_exit_index"] <= record["fit_index"] <= row.origin_index, "实际调用了未来模型")
                        m = record["model"]; x = np.array([getattr(row, f) for f in FEATURES])
                        score = m["intercept"]+np.clip((x-m["mean"])/m["scale"], -5, 5)@np.asarray(m["coefficients"])
                        require(abs(score-row.continuation_prediction) < 1e-12, "保存高点时间预测不符")
                        count = count+1 if score < 0 else 0
                        score_count += 1
                    else:
                        require(pd.isna(row.continuation_prediction), "未知模型被填预测")
                        count = 0
                    require(count == row.negative_confirmation_count and row.learned_exit_requested == (count >= 2), "两日确认或新周期重置不符")
                    state_count += 1
            previous = np.r_[cfg["initial_capital"], ledger.equity.iloc[:-1]]
            np.testing.assert_allclose(ledger.cash+ledger.shares*ledger.mark+ledger.dividend_receivable, ledger.equity, atol=1e-6, rtol=0)
            np.testing.assert_allclose(ledger.equity/previous-1, ledger.net_return, atol=1e-12, rtol=0)
            measured = summarize(ledger, cfg); saved = metric(result, PRIMARY, period, cost)
            for k in ["net_sharpe", "annualized_return", "max_drawdown", "cumulative_return", "trade_count", "commission", "slippage_cost"]:
                require(value_equal(measured[k], saved[k]), "高点时间保存绩效不符")
            complete = saved_cycles(ledger, dividends, cfg)
            cycles.extend({"period": period, "cost": cost, **r} for r in complete)
            parent = pd.read_parquet(folder / "REARM_RIDGE_ledger.parquet")
            if period == "earlier_diagnostic":
                pd.testing.assert_frame_equal(ledger, parent)
            differences.append({"period": period, "cost": cost, "terminal_nav_difference": float(ledger.equity.iloc[-1]-parent.equity.iloc[-1]),
                                **{f"{k}_difference": float(ledger[k].sum()-parent[k].sum()) for k in ["price_pnl", "dividend_recognized", "commission", "slippage_cost"]}})
            accounts.append({"period": period, "cost": cost, "days": len(ledger), "cycles": len(complete), "net_sharpe": measured["net_sharpe"]})
    for name, rows in [("saved_training_checks.csv", model_checks), ("saved_actual_cycles.csv", cycles), ("saved_account_checks.csv", accounts), ("saved_profit_differences.csv", differences)]:
        pd.DataFrame(rows).to_csv(OUT / name, index=False, encoding="utf-8-sig")
    receipt = {"verified_at": now(), "status": "SAVED_REFERENCE_AND_OWN_PEAK_AGES_RIDGE_AND_ACCOUNTS_CHECKED", "reference_full_holding_states": len(states), "reference_training_states": len(samples),
               "mature_models": len(model_checks), "actual_holding_states": state_count, "predictions_recomputed": score_count, "actual_accounts": len(accounts), "complete_cycles": len(cycles),
               "new_models_or_accounts": 0, "reviewer_source_sha256": digest(Path(__file__)), "security_audit_performed": False}
    write_json(OUT / "saved_verification_receipt.json", receipt, exclusive=True)
    decision = "第113轮未达标。高点时间主历史基础／压力净夏普0.591／0.529，低于原八项模型0.705／0.641；主基础年化由5.38%降至4.47%，最大回撤由10.59%增至11.04%。较早基础／压力净夏普0.748／0.725，完整账户与原模型相同。新增信息没有改善实际结果，本项关闭。"
    next_path = ROOT / "docs/510300_AFTER_PEAK_AGE_EXIT_20260908.md"
    verification_text = "六项必要测试4.01秒，114个成熟模型及四个实际账户9.39秒；27个原支持不足月份保留，零拟合失败、零高点时间缺失、零未成交。参考与自身高点时间、成熟成员、保存系数和66个完整周期已核。旧入口准备文本断言已在冻结前修正，不涉及模型或账户重跑。"
    next_lines = ["# 第113轮之后：检验周期内与周期间学习的差别", "", decision, "", verification_text, "",
        "本项没有局部改善，不再改变时间函数、归零条件、窗口、惩罚或退出规则。111状态均值、112简易波动与其他关闭家族保留旧结果。", "",
        "下一方向针对原训练目标重复共享自然退出结果：同一参考周期很多行都含同一个最终退出时点的结果，只给周期等总权重并未区分周期共有水平与周期内部状态变化。拟以每周期单独截距吸收训练样本共有水平，八项系数只从周期内部的状态和目标变化估计；新实际周期使用训练周期截距的等权均值，绝不估计其未来结束结果。", "",
        "有界代码和规则查重未发现该周期固定截距估计实现。旧60的cycle_fixed_baselines是持仓CUSUM基准；旧108是三种参考任务共享和共同时间段权重，均不等于每个完整周期的截距。旧31的原周期等权、最近20个、至少10周期100行、clip5、惩罚1和原八项输入可保持一致。", "",
        "这只是待验证的估计假设，不能称周期之间的变化已经被证明是运气或因果偏差。下一步只读首次成熟模型的旧训练样本，分解周期内外目标变动，检查是否确有信息结构；再冻结一个固定截距方法并做完整账户。尚未登记、拟合或评价第114轮。", "",
        "分钟文件身份和覆盖已核且旧ORB已存在，无较早分钟证据，暂不重开或下载。原91只是局部点值候选，不是完整目标。继续现有510300/现金范围，EPS和慢源暂停，不做GPT审阅包，目标保持未完成。"]
    delivered = deliver_round(ROOT, OUT, CONFIG, ROOT / "deliverables/510300持仓高点时间退出_第113轮_20260908/持仓高点时间退出_结果及全部中文规则.md",
        "原八项退出加入距最近持仓高点的时间", "COMPLETED_PEAK_AGE_INFORMATION_NO_IMPROVEMENT_NOT_TARGET", decision, verification_text, next_path, next_lines,
        "WITHIN_CYCLE_ESTIMATION_HYPOTHESIS_NOT_REGISTERED", "核对原成熟样本的周期内外共同结果，再检验周期截距估计")
    print(json.dumps({"交付": delivered, "核对": receipt, "收益差": differences}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
