"""核对当前强弱九因子模型、实际预测与已保存持仓变动。"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from research.session_strength_within_v1 import ROOT, OUT, CONFIG, PRIMARY
from research.session_strength_within_inputs_v1 import FEATURES
from research.learned_cycle_exit_v1 import training_rows
from research.adaptive_allocation_v1 import normalize_dividends, summarize
from research.intraday_overnight_increment_v1 import now, digest, require, write_json
from scripts.finalize_round60_20260907 import metric
from scripts.finalize_round96_20260908 import value_equal
from scripts.review_round74_saved import saved_cycles


def main():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    result = json.loads((OUT / "result.json").read_text(encoding="utf-8"))
    require(not result["historical_point_target_met"], "出现目标点值时不能仅交付未达标局部改善")
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "周期内模型冻结来源改变")
    data = pd.read_parquet(ROOT / cfg["features"])
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    samples = pd.read_parquet(ROOT / cfg["samples"])
    samples = samples[samples.signal.eq("D60_INTRA")].reset_index(drop=True)
    factors = pd.read_parquet(ROOT / cfg["saved_factors"])
    samples[FEATURES[-1]] = factors[FEATURES[-1]].iloc[samples.origin_index.to_numpy(int)].to_numpy()
    pd.testing.assert_frame_equal(samples, pd.read_parquet(OUT / "extended_reference_samples.parquet").reset_index(drop=True))
    models = json.loads((OUT / "saved_models.json").read_text(encoding="utf-8"))["models"]
    originals = json.loads((ROOT / cfg["saved_models"]).read_text(encoding="utf-8"))["models"]["D60_INTRA__RIDGE"]
    members = pd.read_parquet(OUT / "training_memberships.parquet")
    checks, by_date = [], {}
    require([m["fit_index"] for m in models] == [m["fit_index"] for m in originals], "周期内拟合日程不同")
    for record, old in zip(models, originals, strict=True):
        by_date[pd.Timestamp(record["fit_origin"])] = record
        require(record["status"] == old["status"] and record["training_cycles"] == old["training_cycles"] and record["training_rows"] == old["training_rows"], "原成熟周期或支持时点改变")
        if record["model"] is None:
            continue
        rows, ids = training_rows(samples, record["fit_index"], cfg)
        require(ids == record["training_cycles"] and rows.exit_index.le(record["fit_index"]).all(), "周期截距包含未结束的周期")
        membership = members[members.fit_index.eq(record["fit_index"])].sort_values(["cycle_id", "origin_index"])
        np.testing.assert_array_equal(rows[["cycle_id", "origin_index", "exit_index"]], membership[["cycle_id", "origin_index", "exit_index"]])
        np.testing.assert_allclose(rows.sample_weight, membership.sample_weight, atol=0, rtol=0)
        x, y, w = rows[FEATURES].to_numpy(float), rows.target.to_numpy(float), rows.sample_weight.to_numpy(float)
        mean = np.average(x, axis=0, weights=w); scale = np.sqrt(np.average((x-mean)**2, axis=0, weights=w)); scale = np.where(scale > 1e-12, scale, 1.)
        m = record["model"]; coefficient = np.asarray(m["coefficients"])
        np.testing.assert_allclose(m["mean"], mean, atol=1e-12, rtol=0)
        np.testing.assert_allclose(m["scale"], scale, atol=1e-12, rtol=0)
        np.testing.assert_allclose(m["mean"][:8], old["model"]["mean"], atol=1e-12, rtol=0)
        np.testing.assert_allclose(m["scale"][:8], old["model"]["scale"], atol=1e-12, rtol=0)
        z = np.clip((x-mean)/scale, -5, 5); residual = np.empty(len(rows))
        intercepts = {g["cycle_id"]: g for g in m["cycle_intercepts"]}
        require(set(intercepts) == set(ids), "周期截距成员不同")
        maximum_intercept_gradient = 0.
        for cycle in ids:
            mask = rows.cycle_id.to_numpy() == cycle
            mz, my = np.average(z[mask], axis=0, weights=w[mask]), np.average(y[mask], weights=w[mask])
            a = my-mz@coefficient
            stored = intercepts[cycle]
            np.testing.assert_allclose(stored["standardized_feature_mean"], mz, atol=1e-12, rtol=0)
            require(abs(stored["target_mean"]-my) < 1e-12 and abs(stored["cycle_intercept"]-a) < 1e-12 and stored["rows"] == int(mask.sum()), "周期目标均值或截距不符")
            residual[mask] = a+z[mask]@coefficient-y[mask]
            maximum_intercept_gradient = max(maximum_intercept_gradient, abs(np.sum(w[mask]*residual[mask])))
        gradient = z.T@(w*residual)+cfg["ridge_alpha"]*coefficient
        require(maximum_intercept_gradient < 1e-10 and np.abs(gradient).max() < 1e-10, "保存结果不满足不惩罚周期截距的岭回归方程")
        average = np.mean([g["cycle_intercept"] for g in m["cycle_intercepts"]])
        require(abs(average-m["intercept"]) < 1e-12, "新实际周期截距不是已成熟周期等权平均")
        checks.append({"fit_origin": record["fit_origin"], "cycles": len(ids), "rows": len(rows), "maximum_coefficient_gradient": float(np.abs(gradient).max()), "maximum_intercept_gradient": maximum_intercept_gradient})
    require(len(checks) == 114 and result["failed_fits"] == 0, "周期内模型数或求解失败状态不同")
    cycles, accounts, differences, annual = [], [], [], []
    score_count, state_count = 0, 0
    for period in ["evaluation", "earlier_diagnostic"]:
        for cost in cfg["costs"]:
            folder = OUT / period / cost
            ledger = pd.read_parquet(folder / f"{PRIMARY}_ledger.parquet")
            decisions = pd.read_parquet(folder / f"{PRIMARY}_decisions.parquet")
            native = pd.read_csv(folder / f"{PRIMARY}_cycles.csv")
            for cycle_id, group in decisions[decisions.learning_cycle_id.notna()].groupby("learning_cycle_id", sort=False):
                cycle = native[native.cycle_id.eq(cycle_id)].iloc[0]
                own = ledger[ledger.cycle_id.eq(cycle_id)].copy().set_index("date")
                value = own.shares*own.mark+own.dividend_recognized.cumsum()
                peak = pd.Series(np.maximum.accumulate(np.r_[cycle.entry_cost_cny, value.to_numpy()])[1:], index=own.index)
                count = 0
                for row in group.itertuples():
                    x = np.array([getattr(row, f) for f in FEATURES])
                    actual = value.loc[row.origin]
                    expected = [np.log1p(row.origin_index-cycle.entry_index+1), actual/cycle.entry_cost_cny-1, actual/peak.loc[row.origin]-1, cycle["mode"],
                                data.mom5.iloc[row.origin_index], data.mom20.iloc[row.origin_index], data.sma120.iloc[row.origin_index], data.vol20.iloc[row.origin_index], factors[FEATURES[-1]].iloc[row.origin_index]]
                    np.testing.assert_allclose(x, expected, atol=1e-12, rtol=0)
                    if row.learning_status == "PREDICTION_AVAILABLE":
                        record = by_date[row.learning_fit_origin]
                        require(record["latest_exit_index"] <= record["fit_index"] <= row.origin_index, "新周期实际调用未来截距或系数")
                        m = record["model"]
                        a = np.mean([g["cycle_intercept"] for g in m["cycle_intercepts"]])
                        score = a+np.clip((x-m["mean"])/m["scale"], -5, 5)@np.asarray(m["coefficients"])
                        require(abs(score-row.continuation_prediction) < 1e-12, "实际预测没有使用成熟平均截距和自身九状态")
                        count = count+1 if score < 0 else 0
                        score_count += 1
                    else:
                        require(pd.isna(row.continuation_prediction), "没有模型时填入预测")
                        count = 0
                    require(count == row.negative_confirmation_count and row.learned_exit_requested == (count >= 2), "实际两日负预测确认或新周期重置不符")
                    state_count += 1
            previous = np.r_[cfg["initial_capital"], ledger.equity.iloc[:-1]]
            np.testing.assert_allclose(ledger.cash+ledger.shares*ledger.mark+ledger.dividend_receivable, ledger.equity, atol=1e-6, rtol=0)
            np.testing.assert_allclose(ledger.equity/previous-1, ledger.net_return, atol=1e-12, rtol=0)
            measured = summarize(ledger, cfg); stored = metric(result, PRIMARY, period, cost)
            for k in ["net_sharpe", "annualized_return", "max_drawdown", "cumulative_return", "trade_count", "commission", "slippage_cost"]:
                require(value_equal(measured[k], stored[k]), "周期内完整账户绩效不能复算")
            complete = saved_cycles(ledger, dividends, cfg)
            cycles.extend({"period": period, "cost": cost, **c} for c in complete)
            parent = pd.read_parquet(folder / "WITHIN_CYCLE_EXIT_ledger.parquet")
            require(pd.DatetimeIndex(ledger.date).equals(pd.DatetimeIndex(parent.date)), "新旧实际账户日历不同")
            differences.append({"period": period, "cost": cost, "terminal_nav_difference": float(ledger.equity.iloc[-1]-parent.equity.iloc[-1]),
                                **{f"{k}_difference": float(ledger[k].sum()-parent[k].sum()) for k in ["price_pnl", "dividend_recognized", "commission", "slippage_cost"]}})
            for year, frame in ledger.groupby(ledger.date.dt.year):
                base = parent[parent.date.dt.year.eq(year)]
                current_m, parent_m = summarize(frame, cfg), summarize(base, cfg)
                delta = current_m["cumulative_return"]-parent_m["cumulative_return"]
                annual.append({"period": period, "cost": cost, "year": int(year), "new_return": current_m["cumulative_return"], "parent_return": parent_m["cumulative_return"],
                               "return_difference": delta, "new_sharpe": current_m["net_sharpe"], "parent_sharpe": parent_m["net_sharpe"], "comparison": "改善" if delta > 1e-10 else "下降" if delta < -1e-10 else "相同"})
            accounts.append({"period": period, "cost": cost, "days": len(ledger), "cycles": len(complete), "net_sharpe": measured["net_sharpe"]})
    for name, rows in [("saved_training_checks.csv", checks), ("saved_actual_cycles.csv", cycles), ("saved_account_checks.csv", accounts), ("saved_profit_differences.csv", differences), ("saved_yearly_increment.csv", annual)]:
        pd.DataFrame(rows).to_csv(OUT / name, index=False, encoding="utf-8-sig")
    receipt = {"verified_at": now(), "status": "SAVED_NINE_FACTOR_EQUATIONS_OWN_PREDICTIONS_AND_ACCOUNT_INCREMENT_CHECKED", "mature_models": len(checks), "reference_training_states": len(samples),
               "actual_holding_states": state_count, "predictions_recomputed": score_count, "actual_accounts": len(accounts), "complete_cycles": len(cycles), "yearly_increment_rows": len(annual),
               "new_models_or_accounts": 0, "reviewer_source_sha256": digest(Path(__file__)), "security_audit_performed": False}

    timing = []
    for period in ["evaluation", "earlier_diagnostic"]:
        for cost in cfg["costs"]:
            current = pd.read_csv(OUT / period / cost / f"{PRIMARY}_cycles.csv")
            parent = pd.read_csv(ROOT / "reports/research/510300_within_cycle_exit_v1" / period / cost / "WITHIN_CYCLE_EXIT_cycles.csv")
            paired = current.merge(parent, on="entry_date", how="outer", suffixes=("_new", "_old"), indicator=True)
            changed = paired[paired["_merge"].ne("both") | paired.exit_date_new.ne(paired.exit_date_old)]
            timing.extend({"period": period, "cost": cost, **row} for row in changed[["entry_date", "exit_date_new", "exit_date_old", "exit_reasons_new", "exit_reasons_old", "net_profit_cny_new", "net_profit_cny_old"]].to_dict("records"))
    pd.DataFrame(timing).to_csv(OUT / "changed_cycle_timing.csv", index=False, encoding="utf-8-sig")
    receipt.update(changed_cycle_timing_rows=len(timing), added_factor="d60_factor")
    write_json(OUT / "saved_verification_receipt.json", receipt, exclusive=True)
    print(json.dumps({"核对": receipt, "利润差异": differences, "改变的交易周期": timing}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
