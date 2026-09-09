"""核对首次持仓模型版本、实际状态、预测及完整账户，不重训或重跑。"""
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from research.entry_vintage_exit_v1 import ROOT, OUT, CONFIG, PRIMARY
from research.within_cycle_exit_inputs_v1 import FEATURES, CN
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
    models = json.loads((ROOT / cfg["saved_models"]).read_text(encoding="utf-8"))["models"]
    reuse = json.loads((OUT / "reused_models_receipt.json").read_text(encoding="utf-8"))
    require(reuse["source_sha256"] == digest(ROOT / cfg["saved_models"]) and result["new_model_fits"] == reuse["new_model_fits"] == 0, "原模型复用来源或拟合数量不同")
    require(len(models) == 141 and sum(m["status"] == "FIT_COMPLETE" for m in models) == 114, "原模型支持记录不同")
    by_date = {pd.Timestamp(m["fit_origin"]): m for m in models}
    model_indexes = np.array([m["fit_index"] for m in models])
    checks, cycles, accounts, differences, annual = [], [], [], [], []
    score_count, state_count, main_identical_accounts = 0, 0, 0
    for period in ["evaluation", "earlier_diagnostic"]:
        for cost in cfg["costs"]:
            folder = OUT / period / cost
            ledger = pd.read_parquet(folder / f"{PRIMARY}_ledger.parquet")
            decisions = pd.read_parquet(folder / f"{PRIMARY}_decisions.parquet")
            native = pd.read_csv(folder / f"{PRIMARY}_cycles.csv")
            for cycle_id, group in decisions[decisions.learning_cycle_id.notna()].groupby("learning_cycle_id", sort=False):
                cycle = native[native.cycle_id.eq(cycle_id)].iloc[0]
                require(group.origin.iloc[0] == pd.Timestamp(cycle.entry_date), "模型首次调用不是实际买入日收盘")
                k = int(np.searchsorted(model_indexes, int(cycle.entry_index), side="right"))-1
                selected = models[k] if k >= 0 else None
                selected_date = pd.Timestamp(selected["fit_origin"]) if selected else pd.NaT
                if selected and selected["model"] is not None:
                    values = {key: selected["model"][key] for key in ["features", "mean", "scale", "coefficients", "intercept", "feature_clip"]}
                    identity = hashlib.sha256(json.dumps(values, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                else:
                    identity = "NO_MODEL"
                checks.append({"period": period, "cost": cost, "cycle_id": cycle_id, "selection_origin": group.origin.iloc[0],
                    "selected_model_origin": selected_date, "prediction_identity": identity, "holding_states": len(group)})
                own = ledger[ledger.cycle_id.eq(cycle_id)].copy().set_index("date")
                recognized = own.dividend_recognized.copy()
                recognized.iloc[0] = 0.
                value = own.shares*own.mark+recognized.cumsum()
                peak = pd.Series(np.maximum.accumulate(np.r_[cycle.entry_cost_cny, value.to_numpy()])[1:], index=own.index)
                count = 0
                for row in group.itertuples():
                    x = np.array([getattr(row, f) for f in FEATURES])
                    actual = value.loc[row.origin]
                    expected = [np.log1p(row.origin_index-cycle.entry_index+1), actual/cycle.entry_cost_cny-1, actual/peak.loc[row.origin]-1, cycle["mode"],
                                data.mom5.iloc[row.origin_index], data.mom20.iloc[row.origin_index], data.sma120.iloc[row.origin_index], data.vol20.iloc[row.origin_index]]
                    np.testing.assert_allclose(x, expected, atol=1e-12, rtol=0)
                    require(row.model_selection_index == cycle.entry_index and row.model_selection_origin == pd.Timestamp(cycle.entry_date), "本笔所选模型起点改变")
                    require(row.fixed_prediction_identity == identity, "本笔预测参数身份被中途替换")
                    require((pd.isna(row.learning_fit_origin) and pd.isna(selected_date)) or row.learning_fit_origin == selected_date, "实际调用的模型不是首次收盘固定版本")
                    require((row.learning_status == "PREDICTION_AVAILABLE") == (selected is not None and selected["status"] == "FIT_COMPLETE"), "当前完整实际输入下的固定版本可用性不同")
                    if row.learning_status == "PREDICTION_AVAILABLE":
                        record = selected
                        require(record["latest_exit_index"] <= record["fit_index"] <= cycle.entry_index <= row.origin_index, "新周期实际调用未来截距或系数")
                        m = record["model"]
                        a = np.mean([g["cycle_intercept"] for g in m["cycle_intercepts"]])
                        score = a+np.clip((x-m["mean"])/m["scale"], -5, 5)@np.asarray(m["coefficients"])
                        require(abs(score-row.continuation_prediction) < 1e-12, "实际预测没有使用成熟平均截距和自身八状态")
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
            if period == "evaluation":
                pd.testing.assert_frame_equal(ledger, parent)
                main_identical_accounts += 1
            differences.append({"period": period, "cost": cost, "terminal_nav_difference": float(ledger.equity.iloc[-1]-parent.equity.iloc[-1]),
                                **{f"{k}_difference": float(ledger[k].sum()-parent[k].sum()) for k in ["price_pnl", "dividend_recognized", "commission", "slippage_cost"]}})
            for year, frame in ledger.groupby(ledger.date.dt.year):
                base = parent[parent.date.dt.year.eq(year)]
                current_m, parent_m = summarize(frame, cfg), summarize(base, cfg)
                delta = current_m["cumulative_return"]-parent_m["cumulative_return"]
                annual.append({"period": period, "cost": cost, "year": int(year), "new_return": current_m["cumulative_return"], "parent_return": parent_m["cumulative_return"],
                               "return_difference": delta, "new_sharpe": current_m["net_sharpe"], "parent_sharpe": parent_m["net_sharpe"], "comparison": "改善" if delta > 1e-10 else "下降" if delta < -1e-10 else "相同"})
            accounts.append({"period": period, "cost": cost, "days": len(ledger), "cycles": len(complete), "net_sharpe": measured["net_sharpe"]})
    for name, rows in [("saved_model_version_checks.csv", checks), ("saved_actual_cycles.csv", cycles), ("saved_account_checks.csv", accounts),
        ("saved_profit_differences.csv", differences), ("saved_yearly_increment.csv", annual)]:
        pd.DataFrame(rows).to_csv(OUT / name, index=False, encoding="utf-8-sig")
    timing = []
    for period in ["evaluation", "earlier_diagnostic"]:
        for cost in cfg["costs"]:
            current = pd.read_csv(OUT / period / cost / f"{PRIMARY}_cycles.csv")
            parent = pd.read_csv(ROOT / "reports/research/510300_within_cycle_exit_v1" / period / cost / "WITHIN_CYCLE_EXIT_cycles.csv")
            paired = current.merge(parent, on="entry_date", how="outer", suffixes=("_new", "_old"), indicator=True)
            changed = paired[paired["_merge"].ne("both") | paired.exit_date_new.ne(paired.exit_date_old)]
            timing.extend({"period": period, "cost": cost, **row} for row in changed[["entry_date", "exit_date_new", "exit_date_old", "exit_reasons_new", "exit_reasons_old", "net_profit_cny_new", "net_profit_cny_old"]].to_dict("records"))
    pd.DataFrame(timing).to_csv(OUT / "changed_cycle_timing.csv", index=False, encoding="utf-8-sig")
    sensitivities = []
    for period in ["evaluation", "earlier_diagnostic"]:
        pair = {}
        for cost in ["BASE", "STRESS"]:
            d = pd.read_parquet(OUT / period / cost / f"{PRIMARY}_decisions.parquet")
            pair[cost] = d[d.learning_status.eq("PREDICTION_AVAILABLE")]
        matched = pair["BASE"].merge(pair["STRESS"], on="origin", suffixes=("_base", "_stress"), validate="one_to_one")
        changed = matched[(np.sign(matched.continuation_prediction_base) != np.sign(matched.continuation_prediction_stress))
            & matched.fixed_prediction_identity_base.eq(matched.fixed_prediction_identity_stress)
            & matched.model_selection_origin_base.eq(matched.model_selection_origin_stress)]
        for _, row in changed.iterrows():
            model = by_date[row.learning_fit_origin_base]["model"]
            x = np.array([row[f+"_base"] for f in FEATURES], float)
            y = np.array([row[f+"_stress"] for f in FEATURES], float)
            components = (np.clip((y-model["mean"])/model["scale"], -5, 5)-np.clip((x-model["mean"])/model["scale"], -5, 5))*model["coefficients"]
            require(abs(components.sum()-(row.continuation_prediction_stress-row.continuation_prediction_base)) < 1e-12, "同一固定模型的费用状态差不能复算")
            sensitivities.append({"period": period, "origin": row.origin, "selection_origin": row.model_selection_origin_base,
                "model_origin": row.learning_fit_origin_base, "base_prediction": row.continuation_prediction_base,
                "stress_prediction": row.continuation_prediction_stress, "base_negative_count": row.negative_confirmation_count_base,
                "stress_negative_count": row.negative_confirmation_count_stress,
                **{f"prediction_difference_{name}": float(value) for name, value in zip(FEATURES, components, strict=True)}})
    pd.DataFrame(sensitivities).to_csv(OUT / "saved_cost_state_sign_differences.csv", index=False, encoding="utf-8-sig")
    receipt = {"verified_at": now(), "status": "SAVED_FIRST_HOLDING_CLOSE_MODEL_VINTAGES_OWN_PREDICTIONS_AND_ACCOUNTS_CHECKED",
        "reused_model_records": len(models), "reused_available_models": 114, "fixed_model_cycles": len(checks),
        "actual_holding_states": state_count, "predictions_recomputed": score_count, "actual_accounts": len(accounts),
        "complete_cycles": len(cycles), "main_accounts_identical_to_114": main_identical_accounts,
        "yearly_increment_rows": len(annual), "changed_cycle_timing_rows": len(timing), "cost_state_sign_difference_rows": len(sensitivities),
        "new_models_or_accounts": 0, "reviewer_source_sha256": digest(Path(__file__)), "security_audit_performed": False}
    write_json(OUT / "saved_verification_receipt.json", receipt, exclusive=True)
    print(json.dumps({"核对": receipt, "利润差异": differences, "改变的退出": timing, "费用状态变号": sensitivities}, ensure_ascii=False, default=str), flush=True)


if __name__ == "__main__":
    main()
