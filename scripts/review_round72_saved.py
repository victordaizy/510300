"""复算已保存逐周期删除一致模型、实际持仓输入和账户经济，不重新拟合或模拟。"""
from __future__ import annotations

from bisect import bisect_right
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from research.deleted_cycle_stability_v1 import ROOT, OUT, CONFIG, PRIMARY, P32
from research.learned_cycle_exit_v1 import FEATURES
from research.adaptive_allocation_v1 import normalize_dividends, summarize
from research.intraday_overnight_increment_v1 import now, digest, require, write_json
from scripts.finalize_round66_20260907 import saved_cycles


def linear_value(model, values):
    x = np.maximum(-model["feature_clip"], np.minimum(model["feature_clip"], (np.asarray(values) - model["mean"]) / model["scale"]))
    return float(model["intercept"] + np.dot(x, model["coefficients"]))


def source_and_training(cfg, data, models, originals):
    cash = data.dividend
    night = np.log((data.open + cash) / data.previous_close)
    intraday = np.log((data.close + cash) / (data.open + cash))
    change = (data.close + cash) / data.previous_close - 1
    wealth = (1 + change.fillna(0.)).cumprod()
    difference = intraday - night
    factor = difference.rolling(60).sum() / (difference.rolling(60).std(ddof=1) * np.sqrt(60))
    entry = (factor.gt(1) & factor.shift().gt(1) & data.feature_valid).to_numpy(bool)
    price_exit = (factor.lt(0) & factor.shift().lt(0)).to_numpy(bool)
    inputs = pd.DataFrame({"date": data.date, "d60_factor": factor, "entry_condition": entry, "original_price_exit": price_exit,
        "mom5": np.log(wealth / wealth.shift(5)), "mom20": np.log(wealth / wealth.shift(20)),
        "sma120": wealth / wealth.rolling(120).mean() - 1, "vol20": change.rolling(20).std(ddof=1) * np.sqrt(242)})
    for name in ["mom5", "mom20", "sma120", "vol20"]:
        np.testing.assert_allclose(inputs[name], data[name], atol=2e-12, rtol=0, equal_nan=True)
    for period in ["evaluation", "earlier_diagnostic"]:
        saved = pd.read_parquet(OUT / f"{period}_entry_exit_conditions.parquet")
        require(saved.entry_condition.eq(inputs.entry_condition.iloc[:len(saved)].to_numpy()).all(), "进入条件不能从原始分红日线复算")
        require(saved.original_price_exit.eq(inputs.original_price_exit.iloc[:len(saved)].to_numpy()).all(), "原价格退出不能复算")
    inputs.to_csv(OUT / "saved_all_entry_factors.csv", index=False, encoding="utf-8-sig")
    samples = pd.read_parquet(ROOT / cfg["samples"])
    samples = samples[samples.signal.eq("D60_INTRA")]
    memberships = pd.read_parquet(OUT / "training_memberships.parquet")
    checks, deletion_checks = [], []
    for record, original in zip(models, originals, strict=True):
        t = int(record["fit_index"])
        eligible = samples[samples.exit_index.le(t)]
        ids = eligible[["cycle_id", "exit_index"]].drop_duplicates().sort_values(["exit_index", "cycle_id"]).tail(cfg["recent_cycles"]).cycle_id.tolist()
        rows = eligible[eligible.cycle_id.isin(ids)].sort_values(["cycle_id", "origin_index"])
        require(ids == record["training_cycles"] == original["training_cycles"] and len(rows) == record["training_rows"] == original["training_rows"], "新旧成熟周期或状态记录不符")
        require(t == original["fit_index"] and pd.Timestamp(record["fit_time"]) == data.date.iloc[t] + pd.Timedelta(hours=15, minutes=5), "训练时钟改变")
        usable = len(ids) >= cfg["minimum_cycles"] and len(rows) >= cfg["minimum_rows"]
        require(usable == record["eligible_for_fit"], "成熟支持状态改变")
        error = None
        if usable:
            weight = (1. / rows.groupby("cycle_id").origin_index.transform("size")).to_numpy()
            member = memberships[memberships.fit_index.eq(t)].sort_values(["cycle_id", "origin_index"])
            require(member.origin_index.tolist() == rows.origin_index.tolist() and member.exit_index.tolist() == rows.exit_index.tolist(), "训练成员重放不一致")
            np.testing.assert_allclose(member.sample_weight, weight, atol=0, rtol=0)
            if record["status"] == "FIT_COMPLETE":
                require(record["model"]["base_model"] == original["model"], "原全样本模型内容被改变")
                deletions = record["model"]["deletions"]
                require([d["deleted_cycle_id"] for d in deletions] == ids, "没有按原周期顺序检查全部删除模型")
                for deleted in deletions:
                    selected = rows[rows.cycle_id.ne(deleted["deleted_cycle_id"])]
                    expected_ids = sorted(set(ids) - {deleted["deleted_cycle_id"]})
                    require(expected_ids == deleted["training_cycles"] and len(selected) == deleted["training_rows"], "删除子样本补入或漏掉原成员")
                    require(int(selected.exit_index.max()) == deleted["latest_exit_index"] <= t, "删除子样本尚未成熟")
                    w = (1. / selected.groupby("cycle_id").origin_index.transform("size")).to_numpy()
                    x = selected[FEATURES].to_numpy(float)
                    mean = (w[:, None] * x).sum(axis=0) / w.sum()
                    scale = np.sqrt((w[:, None] * (x - mean) ** 2).sum(axis=0) / w.sum())
                    scale = np.where(scale > 1e-12, scale, 1.)
                    model = deleted["model"]
                    np.testing.assert_allclose(mean, model["mean"], atol=1e-13, rtol=0)
                    np.testing.assert_allclose(scale, model["scale"], atol=1e-13, rtol=0)
                    design = np.clip((x - mean) / scale, -cfg["feature_clip"], cfg["feature_clip"])
                    residual = design @ np.asarray(model["coefficients"]) + model["intercept"] - selected.target.to_numpy(float)
                    coefficient_gradient = design.T @ (w * residual) + cfg["ridge_alpha"] * np.asarray(model["coefficients"])
                    gradient = max(float(np.max(np.abs(coefficient_gradient))), abs(float(w @ residual)))
                    require(gradient < 1e-9, "保存系数未满足原加权岭回归的一阶条件")
                    error = max(error or 0., gradient)
                    deletion_checks.append({"fit_index": t, "deleted_cycle_id": deleted["deleted_cycle_id"], "remaining_cycles": len(expected_ids),
                        "remaining_rows": len(selected), "maximum_gradient_residual": gradient, "whole_cycle_deleted": True, "no_backfill": True})
        checks.append({"fit_index": t, "fit_origin": record["fit_origin"], "status": record["status"], "training_cycles": len(ids),
            "training_rows": len(rows), "whole_cycles_mature": True, "original_support_identical": True, "maximum_gradient_residual": error})
    pd.DataFrame(checks).to_csv(OUT / "saved_training_support_replay.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(deletion_checks).to_csv(OUT / "saved_deletion_training_replay.csv", index=False, encoding="utf-8-sig")
    return inputs, {"deleted_models": len(deletion_checks), "maximum_gradient_residual": max(d["maximum_gradient_residual"] for d in deletion_checks), "state_rows": len(data), "model_updates": len(checks), "fit_memberships": len(memberships), "completed_models": sum(m["status"] == "FIT_COMPLETE" for m in models)}


def budget_quantity(cash, close, cost, cfg):
    estimated_price = math.ceil(close * (1 + cost["slippage"]) / cfg["tick"] - 1e-10) * cfg["tick"]
    limit = cash / (estimated_price * cfg["lot"])
    lots = max(0, math.floor(limit + 1e-12))
    while lots and lots * cfg["lot"] * estimated_price + max(lots * cfg["lot"] * estimated_price * cost["commission"], cost["minimum"]) > cash + 1e-8:
        lots -= 1
    return lots * cfg["lot"]


def main():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "逐周期删除一致冻结内容改变")
    result = json.loads((OUT / "result.json").read_text(encoding="utf-8"))
    data = pd.read_parquet(ROOT / cfg["features"])
    div = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    models = json.loads((OUT / "saved_models.json").read_text(encoding="utf-8"))["models"]
    originals = json.loads((ROOT / cfg["saved_models"]).read_text(encoding="utf-8"))["models"]["D60_INTRA__RIDGE"]
    factors, source = source_and_training(cfg, data, models, originals)
    fit_indexes = [m["fit_index"] for m in models]
    metrics, deltas, groups, predictions, all_cycles, comparisons = [], [], [], [], [], []
    requests, no_views = 0, 0
    for period, key in [("evaluation", "all_metrics"), ("earlier_diagnostic", "earlier_diagnostics")]:
        for cost_id, cost in cfg["costs"].items():
            folder = OUT / period / cost_id
            accounts = {}
            for m in result[key]:
                if m["cost"] != cost_id:
                    continue
                ledger = pd.read_parquet(folder / f"{m['model']}_ledger.parquet")
                accounts[m["model"]] = ledger
                actual = summarize(ledger, cfg)
                for field in ["net_sharpe", "annualized_return", "max_drawdown", "cumulative_return", "mean_exposure", "commission", "slippage_cost"]:
                    require((actual[field] is None and m[field] is None) or abs(actual[field] - m[field]) < 1e-10, "保存账户指标不能复算")
                metrics.append({"period": period, "cost": cost_id, "model": m["model"], **actual})
            ledger = accounts[PRIMARY]
            require(np.allclose(ledger.cash + ledger.shares * ledger.mark + ledger.dividend_receivable, ledger.equity, atol=1e-6, rtol=0), "资产合计不等于净值")
            previous = np.r_[cfg["initial_capital"], ledger.equity.to_numpy()[:-1]]
            np.testing.assert_allclose(ledger.equity / previous - 1, ledger.net_return, atol=1e-13, rtol=0)
            for control in ["REARM_RIDGE", "REARM_NONE", "PANIC_LEARNED_HALF", "BUY_HOLD"]:
                old = accounts[control]
                require(pd.DatetimeIndex(old.date).equals(pd.DatetimeIndex(ledger.date)), "完整账户日历不一致")
                d = {"period": period, "cost": cost_id, "control": control, "terminal_nav_difference": float(ledger.equity.iloc[-1] - old.equity.iloc[-1]),
                    "price_difference": float(ledger.price_pnl.sum() - old.price_pnl.sum()), "dividend_difference": float(ledger.dividend_recognized.sum() - old.dividend_recognized.sum()),
                    "commission_difference": float(ledger.commission.sum() - old.commission.sum()), "slippage_difference": float(ledger.slippage_cost.sum() - old.slippage_cost.sum())}
                d["identity_error"] = d["terminal_nav_difference"] - d["price_difference"] - d["dividend_difference"] + d["commission_difference"] + d["slippage_difference"]
                require(abs(d["identity_error"]) < 1e-6, "新旧账户经济差额不平")
                deltas.append(d)
            actual_cycles = saved_cycles(ledger, div, cfg)
            old_cycles = saved_cycles(accounts["REARM_RIDGE"], div, cfg)
            all_cycles.extend({"period": period, "cost": cost_id, **c} for c in actual_cycles)
            compared = pd.DataFrame(actual_cycles).merge(pd.DataFrame(old_cycles), on="entry_origin", how="outer", suffixes=("_new", "_old"), indicator=True, validate="one_to_one")
            compared["period"], compared["cost"] = period, cost_id
            comparisons.append(compared)
            raw_cycles = pd.read_csv(folder / f"{PRIMARY}_cycles.csv").set_index("cycle_id")
            decisions = pd.read_parquet(folder / f"{PRIMARY}_decisions.parquet")
            dated = ledger.set_index("date")
            rearmed, last_exit, pending_exit, last_cycle, count = True, -1000000, False, None, 0
            for row in decisions.itertuples():
                t = int(row.origin_index)
                require(row.origin == data.date.iloc[t] and row.execution_date == data.date.iloc[t + 1], "收盘与下一开盘错位")
                actual = dated.loc[row.origin] if row.origin in dated.index else None
                shares = int(actual.shares) if actual is not None else 0
                current_cash = float(actual.cash) if actual is not None else cfg["initial_capital"]
                if actual is not None and actual.filled_quantity > 0:
                    rearmed, pending_exit = False, False
                if actual is not None and actual.filled_quantity < 0 and shares == 0:
                    last_exit, pending_exit = t, False
                if not shares and not factors.entry_condition.iloc[t]:
                    rearmed = True
                require(rearmed == bool(row.entry_rearmed), "再次进入资格与真实成交不能复算")
                expected_quantity = 0
                if shares:
                    c = raw_cycles.loc[int(row.learning_cycle_id)]
                    path = ledger[ledger.date.between(c.entry_date, row.origin)]
                    values_path = path.shares * path.mark + path.dividend_recognized.cumsum()
                    current_value, peak = float(values_path.iloc[-1]), max(float(c.entry_cost_cny), float(values_path.max()))
                    values = [np.log1p(t - c.entry_index + 1), current_value / c.entry_cost_cny - 1, current_value / peak - 1, 1.,
                        factors.mom5.iloc[t], factors.mom20.iloc[t], factors.sma120.iloc[t], factors.vol20.iloc[t]]
                    np.testing.assert_allclose(values, [getattr(row, name) for name in FEATURES], atol=2e-12, rtol=0)
                    pos = bisect_right(fit_indexes, t) - 1
                    estimate, old_estimate = None, None
                    stored = models[pos] if pos >= 0 else None
                    if stored and stored["status"] == "FIT_COMPLETE":
                        require(stored["latest_exit_index"] <= stored["fit_index"] <= t, "实际退出用了未来模型或周期")
                        old_estimate = linear_value(originals[pos]["model"], values)
                        deleted = [linear_value(d["model"], values) for d in stored["model"]["deletions"]]
                        estimate = max(old_estimate, max(deleted))
                        nonnegative = sum(p >= 0 for p in [old_estimate, *deleted])
                        require(abs(estimate - row.continuation_prediction) < 1e-10, "实际持仓全部模型最大预测无法复算")
                        require(abs(old_estimate - row.base_continuation_prediction) < 1e-10, "原模型在相同实际状态上的预测不符")
                        require(abs(max(deleted) - row.deleted_max_prediction) < 1e-10 and abs(min(deleted) - row.deleted_min_prediction) < 1e-10, "删除模型预测范围不符")
                        require(nonnegative == row.nonnegative_predictions and len(deleted) + 1 == row.committee_size, "全部同意判定成员数量不符")
                        require(pd.Timestamp(stored["fit_origin"]) == row.learning_fit_origin, "实际所用模型日期不符")
                        predictions.append({"period": period, "cost": cost_id, "origin": row.origin, "cycle_id": int(row.learning_cycle_id),
                            "maximum_prediction": estimate, "original_mean_prediction_on_same_actual_state": old_estimate,
                            "difference": estimate - old_estimate, "all_negative": estimate < 0,
                            "base_negative_but_deletion_nonnegative": old_estimate < 0 <= estimate,
                            "committee_size": len(deleted) + 1, "nonnegative_predictions": nonnegative,
                            "saved_prediction_error": estimate - row.continuation_prediction})
                    else:
                        require(pd.isna(row.continuation_prediction), "无模型被填成预测数值")
                        no_views += 1
                    if last_cycle != row.learning_cycle_id:
                        last_cycle, count = row.learning_cycle_id, 0
                    count = count + 1 if estimate is not None and estimate < 0 else 0
                    learned_exit = count >= cfg["confirmation_days"]
                    require(count == row.negative_confirmation_count and learned_exit == bool(row.learned_exit_requested), "连续确认状态不符")
                    original_exit = bool(factors.original_price_exit.iloc[t]) or values[1] <= -.06 or values[2] <= -.08 or t - c.entry_index + 1 >= 60
                    pending_exit = pending_exit or original_exit or learned_exit
                    expected_quantity = -shares if pending_exit else 0
                elif factors.entry_condition.iloc[t] and rearmed and t - last_exit >= cfg["specification"]["cooldown"]:
                    expected_quantity = budget_quantity(current_cash, float(data.close.iloc[t]), cost, cfg)
                require(expected_quantity == row.requested_quantity, "完整进入或退出请求不能重现")
                requests += 1
            groups.append({"period": period, "cost": cost_id, "cycles": len(actual_cycles), "positive_cycles": sum(c["net_profit"] > 0 for c in actual_cycles),
                "negative_cycles": sum(c["net_profit"] < 0 for c in actual_cycles), "net_profit": sum(c["net_profit"] for c in actual_cycles),
                "gross_price_profit": sum(c["gross_price_profit"] for c in actual_cycles), "dividend_recognized": sum(c["dividend_recognized"] for c in actual_cycles),
                "commission": sum(c["commission"] for c in actual_cycles), "slippage": sum(c["slippage"] for c in actual_cycles),
                "held_closes": sum(c["held_closes"] for c in actual_cycles), "mean_held_closes": float(np.mean([c["held_closes"] for c in actual_cycles])),
                "original_cycles": len(old_cycles), "original_held_closes": sum(c["held_closes"] for c in old_cycles),
                "matched_entry_origins": int(compared["_merge"].eq("both").sum()), "only_new_entry_origins": int(compared["_merge"].eq("left_only").sum()),
                "only_original_entry_origins": int(compared["_merge"].eq("right_only").sum())})
    for name, rows in [("saved_account_metrics.csv", metrics), ("saved_account_differences.csv", deltas), ("saved_actual_cycles.csv", all_cycles),
        ("saved_cycle_profit_groups.csv", groups), ("saved_prediction_comparison.csv", predictions)]:
        pd.DataFrame(rows).to_csv(OUT / name, index=False, encoding="utf-8-sig")
    pd.concat(comparisons, ignore_index=True).to_csv(OUT / "saved_entry_matched_cycle_comparison.csv", index=False, encoding="utf-8-sig")
    receipt = {"verified_at": now(), "status": "SAVED_DELETED_CYCLE_MODELS_AND_FULL_ACCOUNT_ECONOMICS_RECONCILED", "source_and_training": source,
        "account_metrics": len(metrics), "account_differences": len(deltas), "actual_requests_replayed": requests, "holding_predictions_replayed": len(predictions),
        "no_model_holding_decisions": no_views, "actual_cycles": len(all_cycles), "maximum_prediction_error": max(abs(p["saved_prediction_error"]) for p in predictions),
        "new_model_fits": 0, "new_account_simulations": 0, "reviewer_source_sha256": digest(Path(__file__)), "security_audit_performed": False}
    write_json(OUT / "saved_verification_receipt.json", receipt, exclusive=True)
    print(json.dumps({"核对": receipt, "周期": groups}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
