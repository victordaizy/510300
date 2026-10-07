"""共同状态下同时估计两个保险方向的净收益价值；固定浅树与线性对照。"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor


ROOT = next(p for p in Path(__file__).resolve().parents if (p / "config/510300_existing_data_training_mandate_v1.json").is_file())
ROUTER = ROOT / "reports/research/510300_option_credit_spread_router_daily_v1"
BUYBACK = ROOT / "reports/research/510300_put_spread_buyback_cost_daily_v1"
OUT = ROOT / "reports/research/510300_option_paired_state_value_daily_v1"
STUDY = "510300_OPTION_PAIRED_STATE_VALUE_DAILY_V1"
BASE = ["trend20_current", "volatility_richness", "quoted_compensation_difference"]
MACRO = ["funding_gap_pp", "credit_acceleration3_pp", "GSPC_z", "VIX_z"]
MODELS = {"EMPIRICAL": [], "LINEAR_PRICE": BASE, "TREE_PRICE": BASE, "TREE_MACRO": BASE + MACRO}
TARGETS = ["value_PUT", "value_CALL"]


def modules(root):
    path = root / "code/router_engine.py"
    if not path.exists():
        path = ROUTER / "code/option_credit_spread_router_daily_v1.py"
    spec = importlib.util.spec_from_file_location("paired_state_router_primitives", path)
    router = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(router)
    h, b = router.modules(root if (root / "code/spread_engine.py").exists() else ROUTER)
    return router, h, b


def freeze(root):
    r, h, b = modules(root)
    if (root / "freeze.json").exists():
        raise RuntimeError("固定状态实验已有冻结记录。")
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "put_information.parquet": BUYBACK / "results/information_and_labels.parquet",
        "call_information.parquet": ROUTER / "results/call_information_and_labels.parquet",
        "put_signals.parquet": BUYBACK / "results/signals.parquet",
        "call_signals.parquet": ROUTER / "results/call_signals.parquet",
        "option_eod.parquet": ROUTER / "inputs/option_eod.parquet",
        "option_risk.parquet": ROUTER / "inputs/option_risk.parquet",
        "daily_terms.parquet": ROUTER / "inputs/daily_terms.parquet",
        "market.parquet": ROUTER / "inputs/market.parquet",
        "calendar_prices.parquet": ROUTER / "inputs/calendar_prices.parquet",
        "dividends.csv": ROUTER / "inputs/dividends.csv",
        "macro_information.parquet": ROUTER / "inputs/macro_information.parquet",
        "parent_result.json": ROUTER / "result.json",
        "parent_protocol.json": ROUTER / "protocol.json",
        "authority_update.json": ROUTER / "inputs/authority_update.json",
        "mandate.json": ROOT / "config/510300_existing_data_training_mandate_v1.json",
    }
    for name, source in paths.items():
        (root / "inputs").mkdir(exist_ok=True)
        shutil.copy2(source, root / "inputs" / name)
    (root / "code").mkdir(exist_ok=True)
    shutil.copy2(Path(__file__), root / "code" / Path(__file__).name)
    shutil.copy2(ROUTER / "code/option_credit_spread_router_daily_v1.py", root / "code/router_engine.py")
    for name in ["spread_engine.py", "buyback_model.py", "account_primitives_source.py"]:
        shutil.copy2(ROUTER / "code" / name, root / "code" / name)
    protocol = {
        "study_id": STUDY, "at": h.now(), "primary": "TREE_MACRO", "models": MODELS,
        "new_question": "对同日两侧保险及现金的相对决策价值建立共同状态，而非分别拟合两个成本水平后比较。",
        "inputs": {"trend20_current": "昨日收盘前已知20日趋势",
                   "volatility_richness": "两侧观察日log(隐波/实现波动率)的均值",
                   "quoted_compensation_difference": "认沽压力信用额/事前最大损失减认购同口径比例",
                   "macro": MACRO},
        "target": "各侧(观察日压力信用额-第10日实际压力买回成本-四次每张5元费用)/事前最大损失；同日成对标签",
        "training": "每日更新，只用最近两个日历年内且两个十日退出标签均已成熟的原点；至少252行共同样本",
        "models_detail": {"EMPIRICAL": "共同历史池的两个均值", "LINEAR_PRICE": "三价格变量的双输出岭回归lambda10，含未惩罚截距",
                          "TREE_PRICE": "三价格变量，双输出平方误差树，max_depth2、min_samples_leaf63、random_state20260925",
                          "TREE_MACRO": "增加四宏观变量，树的其他设置相同"},
        "scaling": "仅训练均值和样本标准差；标准化截断[-5,5]，所有模型共同历史池",
        "state_limit": "最多四个叶状态；不扫描深度、叶样本、变量子集或模型混合权重",
        "action": "两个预测净收益/风险均不正则现金；否则选较大正值，同分认沽；保留当时状态条件与叶样本数",
        "entry_credit_limit": "max(观察日压力信用额-预测净收益比*事前最大损失,全部固定费用+原退出滑点储备)；沿用原开盘检查",
        "risk_and_execution": "认沽和认购的原合约、十日持有、尾部预测、5%最坏损失、2.5%ES、回撤余量、逐腿费用、保证金和减仓逻辑保持",
        "coverage": "只在两侧既有尾部与定价资料及本次预测均有的日期比较；当前未来标签是否成熟不影响入场资格",
        "evaluation": "成对预测MSE、事后最优动作的相对遗憾仅作诊断、完整账户与全部滚动两年；最大四状态不等于已证明存在四种经济制度",
        "primary_increment": "TREE_MACRO对TREE_PRICE；TREE_PRICE对LINEAR_PRICE检验固定非线性增量",
        "bootstrap": {"block_sessions": 20, "draws": 2000, "seed": 2026092505},
        "targets": {"net_sharpe": 1.2, "net_cagr": .1, "max_drawdown": .1},
        "historical_selection_limit": "之前多个模型的失败和容量诊断均已见；本轮是有限开发比较，不是独立确认",
        "new_market_downloads": 0, "orders_authorized": False,
    }
    h.save(root / "protocol.json", protocol, exclusive=True)
    files = [root / "protocol.json", *sorted((root / "inputs").glob("*")), *sorted((root / "code").glob("*"))]
    h.save(root / "freeze.json", {"at": h.now(), "before_new_paired_models": True,
                                 "files": {p.relative_to(root).as_posix(): h.digest(p) for p in files}}, exclusive=True)
    print("双输出共同状态及最大四叶规则已冻结，没有参数搜索。", flush=True)


def paired_information(root, h):
    put = pd.read_parquet(root / "inputs/put_information.parquet")
    call = pd.read_parquet(root / "inputs/call_information.parquet")
    columns = ["idx", "date", "exit10_date", "features_known", "status10", "log_iv_rv20", "prior_stress_credit", "known_loss", "future_buyback_cost"]
    frame = put[columns + ["trend20_current", *MACRO]].merge(call[columns], on=["idx", "date", "exit10_date"], suffixes=("_PUT", "_CALL"), validate="one_to_one")
    frame["volatility_richness"] = (frame.log_iv_rv20_PUT + frame.log_iv_rv20_CALL) / 2
    frame["quoted_compensation_difference"] = frame.prior_stress_credit_PUT / frame.known_loss_PUT - frame.prior_stress_credit_CALL / frame.known_loss_CALL
    frame["features_known"] = frame.features_known_PUT & frame.features_known_CALL & np.isfinite(frame[BASE + MACRO]).all(axis=1)
    for side in ["PUT", "CALL"]:
        frame[f"value_{side}"] = (frame[f"prior_stress_credit_{side}"] - frame[f"future_buyback_cost_{side}"] - 4 * h.FEE) / frame[f"known_loss_{side}"]
    frame["both_targets_known"] = frame.status10_PUT.eq("READY") & frame.status10_CALL.eq("READY") & np.isfinite(frame[TARGETS]).all(axis=1)
    frame.to_parquet(root / "results/paired_information_and_targets.parquet", index=False)
    return frame


def tree_predict(tree, x):
    node, path = 0, []
    while tree["children_left"][node] != -1:
        feature, threshold = tree["feature"][node], tree["threshold"][node]
        direction = float(np.float32(x[feature])) <= threshold
        path.append({"feature_index": feature, "threshold_standardized": threshold, "value_standardized": float(x[feature]), "less_equal": direction})
        node = tree["children_left"][node] if direction else tree["children_right"][node]
    return np.asarray(tree["value"][node], float).reshape(-1), node, path


def fit_at(current, frame):
    day = current["date"]
    lower = day - pd.DateOffset(years=2)
    train = frame[frame.date.ge(lower) & frame.exit10_date.lt(day) & frame.features_known & frame.both_targets_known]
    receipt = {"date": day, "idx": int(current["idx"]), "lower_bound": lower, "training_count": len(train),
               "latest_training_exit": train.exit10_date.max() if len(train) else None}
    if len(train) < 252 or not current["features_known"]:
        return [], [], {**receipt, "status": "NO_VIEW"}
    y = train[TARGETS].to_numpy(float)
    predictions, models = [], []
    for model, columns in MODELS.items():
        mean, scale, beta = [], [], []
        intercept = y.mean(axis=0)
        tree = None
        leaf, path = None, []
        leaf_samples = len(train)
        if columns:
            raw = train[columns].to_numpy(float)
            mean, scale = raw.mean(axis=0), raw.std(axis=0, ddof=1)
            scale[scale < 1e-12] = 1
            z = np.clip((raw - mean) / scale, -5, 5)
            x = np.clip((np.array([current[k] for k in columns]) - mean) / scale, -5, 5)
            if model.startswith("TREE"):
                estimator = DecisionTreeRegressor(max_depth=2, min_samples_leaf=63, random_state=20260925)
                estimator.fit(z, y)
                t = estimator.tree_
                tree = {"children_left": t.children_left.tolist(), "children_right": t.children_right.tolist(),
                        "feature": t.feature.tolist(), "threshold": t.threshold.tolist(), "value": t.value.tolist(),
                        "n_node_samples": t.n_node_samples.tolist()}
                prediction, leaf, path = tree_predict(tree, x)
                np.testing.assert_allclose(prediction, estimator.predict(x.reshape(1, -1))[0], atol=1e-12)
                leaf_samples = tree["n_node_samples"][leaf]
            else:
                center = z.mean(axis=0)
                beta = np.linalg.solve((z - center).T @ (z - center) + 10 * np.eye(len(columns)), (z - center).T @ (y - y.mean(axis=0)))
                intercept = y.mean(axis=0) - center @ beta
                prediction = intercept + x @ beta
        else:
            prediction = intercept
        side = "PUT" if prediction[0] >= prediction[1] else "CALL"
        value = float(max(prediction))
        predictions.append({**receipt, "model": model, "predicted_PUT": float(prediction[0]), "predicted_CALL": float(prediction[1]),
                            "selected_structure": side, "positive": value > 0, "selected_value": value,
                            "leaf": leaf, "leaf_training_count": leaf_samples})
        models.append({**receipt, "model": model, "features": columns, "training_indices": train.idx.to_list(),
                       "mean": mean, "scale": scale, "beta": beta, "intercept": intercept,
                       "tree": tree, "current_leaf": leaf, "current_state_path": path})
    return predictions, models, {**receipt, "status": "UPDATED"}


def learn(root, h, frame):
    predictions, models, receipts = [], [], []
    for current in frame[frame.date.ge(h.START)].to_dict("records"):
        p, m, r = fit_at(current, frame)
        predictions.extend(p)
        models.extend(m)
        receipts.append(r)
        if len(receipts) % 300 == 0:
            print(f"成对价值与固定浅树状态已更新至{current['date'].date()}。", flush=True)
    pred = pd.DataFrame(predictions)
    pred.to_parquet(root / "results/predictions.parquet", index=False)
    pd.DataFrame(receipts).to_parquet(root / "results/update_receipts.parquet", index=False)
    h.save(root / "results/saved_models.json", models)
    sides = {}
    for side, name in [("PUT", "put_signals.parquet"), ("CALL", "call_signals.parquet")]:
        source = pd.read_parquet(root / "inputs" / name)
        sides[side] = {r["date"]: r for r in source.to_dict("records")}
    signals = {model: [] for model in MODELS}
    choices = []
    for p in pred.to_dict("records"):
        day = p["date"]
        if day not in sides["PUT"] or day not in sides["CALL"]:
            continue
        row = sides[p["selected_structure"]][day].copy()
        model = p["model"]
        minimum = max(row["prior_stress_credit"] - p["selected_value"] * row["known_loss"], 4 * h.FEE + row["prior_exit_slip"])
        row[f"minimum_credit_{model}"] = minimum
        row[f"gate_{model}"] = p["positive"] and row["prior_stress_credit"] > minimum
        row["structure"] = p["selected_structure"]
        signals[model].append(row)
        choices.append({**p, "minimum_credit": minimum, "effective_positive": row[f"gate_{model}"]})
    for model, rows in signals.items():
        signals[model] = pd.DataFrame(rows)
        signals[model].to_parquet(root / "results" / f"signals_{model}.parquet", index=False)
    pd.DataFrame(choices).to_parquet(root / "results/routing_choices.parquet", index=False)
    return pred, models, signals


def evaluate_predictions(root, h, pred, frame):
    scored = pred.merge(frame[["idx", "both_targets_known", *TARGETS]], on="idx", validate="many_to_one")
    scored = scored[scored.both_targets_known].copy()
    scored["MSE"] = ((scored.predicted_PUT - scored.value_PUT).pow(2) + (scored.predicted_CALL - scored.value_CALL).pow(2)) / 2
    realized = np.where(scored.positive, np.where(scored.selected_structure.eq("PUT"), scored.value_PUT, scored.value_CALL), 0.)
    scored["relative_regret"] = np.maximum.reduce([np.zeros(len(scored)), scored.value_PUT.to_numpy(), scored.value_CALL.to_numpy()]) - realized
    scored.to_parquet(root / "results/scored_predictions.parquet", index=False)
    summaries = []
    for model, rows in scored.groupby("model"):
        summaries.append({"model": model, "mature_paired_origins": len(rows), "MSE": float(rows.MSE.mean()),
                          "mean_relative_regret": float(rows.relative_regret.mean()), "cash_prediction_days": int((~rows.positive).sum())})
    h.save(root / "results/prediction_metrics.json", {"models": summaries, "regret_is_account_profit": False,
                                                     "regret_is_computed_with_future_information_for_diagnosis_only": True})
    return summaries


def checks():
    dates = pd.date_range("2022-01-05", periods=253)
    day = pd.Timestamp("2024-01-03")
    frame = pd.DataFrame({"idx": range(253), "date": dates, "exit10_date": [pd.Timestamp("2023-01-01")] * 252 + [day],
                          "features_known": True, "both_targets_known": True, "value_PUT": .1, "value_CALL": -.1})
    for j, col in enumerate(BASE + MACRO):
        frame[col] = np.sin(np.arange(253) / (j + 2))
    current = {"date": day, "idx": 999, "features_known": True, **{col: 0. for col in BASE + MACRO}}
    before, models, _ = fit_at(current, frame)
    assert all(p["selected_structure"] == "PUT" and p["positive"] for p in before)
    assert all(252 not in m["training_indices"] for m in models)
    frame.loc[252, ["value_PUT", "value_CALL"]] = [-1e9, 1e9]
    after, _, _ = fit_at(current, frame)
    np.testing.assert_allclose([[p["predicted_PUT"], p["predicted_CALL"]] for p in before],
                               [[p["predicted_PUT"], p["predicted_CALL"]] for p in after], atol=1e-12)
    for model in models:
        if model["tree"]:
            leaves = [i for i, child in enumerate(model["tree"]["children_left"]) if child == -1]
            assert len(leaves) <= 4 and all(model["tree"]["n_node_samples"][i] >= 63 for i in leaves)
    return {"paired_constant_values_recovered": True, "future_labels_cannot_change_action": True, "maximum_four_states_and_leaf_minimum": True}


def verify(root):
    r, h, b = modules(root)
    h.verify_freeze(root)
    frame = pd.read_parquet(root / "results/paired_information_and_targets.parquet").set_index("idx", drop=False)
    pred = pd.read_parquet(root / "results/predictions.parquet").set_index(["idx", "model"])
    models = json.loads((root / "results/saved_models.json").read_text(encoding="utf-8"))
    for model in models:
        train = frame.loc[model["training_indices"]]
        assert train.date.ge(pd.Timestamp(model["lower_bound"])).all() and train.exit10_date.lt(pd.Timestamp(model["date"])).all()
        value = np.array(model["intercept"])
        if model["features"]:
            current = frame.loc[model["idx"], model["features"]].to_numpy(float)
            x = np.clip((current - np.array(model["mean"])) / np.array(model["scale"]), -5, 5)
            if model["tree"]:
                value, leaf, _ = tree_predict(model["tree"], x)
                assert leaf == model["current_leaf"]
                leaves = [i for i, child in enumerate(model["tree"]["children_left"]) if child == -1]
                assert len(leaves) <= 4 and all(model["tree"]["n_node_samples"][i] >= 63 for i in leaves)
            else:
                value += x @ np.array(model["beta"])
        np.testing.assert_allclose(value, pred.loc[(model["idx"], model["model"]), ["predicted_PUT", "predicted_CALL"]].to_numpy(float), atol=1e-12)
    prefix_checks = []
    for cutoff in [pd.Timestamp("2023-12-29"), pd.Timestamp("2025-12-31")]:
        eligible = [m for m in models if m["model"] == "TREE_MACRO" and pd.Timestamp(m["date"]) <= cutoff]
        if not eligible:
            continue
        model = eligible[-1]
        current = frame.loc[model["idx"]].to_dict()
        subset = frame[frame.date.le(current["date"])].copy()
        subset.loc[subset.exit10_date.ge(current["date"]), TARGETS] = [-1e9, 1e9]
        recomputed, _, _ = fit_at(current, subset)
        for p in recomputed:
            np.testing.assert_allclose([p["predicted_PUT"], p["predicted_CALL"]], pred.loc[(model["idx"], p["model"]), ["predicted_PUT", "predicted_CALL"]].to_numpy(float), atol=1e-12)
        prefix_checks.append(str(pd.Timestamp(model["date"]).date()))
    for scenario in ["BASE", "STRESS"]:
        for model in MODELS:
            path = root / "accounts" / scenario / model
            l, f, d = [pd.read_parquet(path / f"{name}.parquet") for name in ["ledger", "fills", "decisions"]]
            np.testing.assert_allclose(l.equity, l.cash + l.option_value - l.terminal_exit_reserve, atol=1e-6)
            if len(f):
                flows = f.groupby("date").cash_change.sum().reindex(l.date, fill_value=0.).to_numpy()
                np.testing.assert_allclose(h.INITIAL + np.cumsum(flows), l.cash, atol=1e-6)
                assert f.cash_after_leg.ge(-1e-7).all()
            entered = d[d.action.eq("ENTRY_FILLED")]
            if len(entered):
                assert entered.actual_maximum_loss.le(entered.budget + 1e-7).all()
                assert entered.actual_ES95.le(.025 * entered.known_equity + 1e-7).all()
    result = {"at": h.now(), "saved_paired_models_recomputed": len(models), "two_year_and_maturity": True,
              "prefix_and_future_poisoning": prefix_checks, "accounts_reconciled": 8,
              "entry_risk_budgets_verified": True, "mechanism_checks": checks()}
    h.save(root / "verification.json", result)
    return result


def run(root):
    r, h, b = modules(root)
    h.verify_freeze(root)
    assert h.digest(Path(__file__)) == h.digest(root / "code/option_paired_state_value_daily_v1.py")
    h.save(root / "RUN_STARTED.json", {"at": h.now()}, exclusive=True)
    (root / "results").mkdir(exist_ok=True)
    h.save(root / "mechanism_checks.json", checks())
    frame = paired_information(root, h)
    pred, models, signals = learn(root, h, frame)
    prediction_metrics = evaluate_predictions(root, h, pred, frame)
    panel, quotes, market, calendar, events = r.sources(root, h)
    metrics, yearly, rolling, ledgers = [], [], [], {}
    choices = pd.read_parquet(root / "results/routing_choices.parquet")
    for scenario in ["BASE", "STRESS"]:
        for model in MODELS:
            account = h.simulate(model, scenario, signals[model], quotes, market, calendar, events)
            if len(account["cycles"]):
                mapping = choices[choices.model.eq(model)][["date", "selected_structure", "leaf_training_count"]].rename(columns={"date": "entry_date"})
                account["cycles"] = account["cycles"].merge(mapping, on="entry_date", validate="many_to_one")
            path = root / "accounts" / scenario / model
            path.mkdir(parents=True, exist_ok=True)
            for name, value in account.items():
                if isinstance(value, pd.DataFrame):
                    value.to_parquet(path / f"{name}.parquet", index=False)
                else:
                    h.save(path / f"{name}.json", value)
            measure = h.account_metrics(model, scenario, account)
            metrics.append(measure)
            y, w = h.yearly_and_rolling(model, scenario, account["ledger"])
            yearly.extend(y)
            rolling.extend(w)
            ledgers[(model, scenario)] = account["ledger"]
            print(f"{scenario}/{model}完成：夏普{measure['net_sharpe']}，年化{measure['annualized_return']:.3%}，回撤{measure['max_drawdown']:.3%}。", flush=True)
    h.save(root / "results/account_metrics.json", metrics)
    h.save(root / "results/yearly_metrics.json", yearly)
    pd.DataFrame(rolling).to_parquet(root / "results/all_rolling_two_years.parquet", index=False)
    comparisons = []
    for left, right in [("TREE_MACRO", "TREE_PRICE"), ("TREE_PRICE", "LINEAR_PRICE")]:
        diff = ledgers[(left, "STRESS")]["return"].to_numpy() - ledgers[(right, "STRESS")]["return"].to_numpy()
        rng = np.random.default_rng(2026092505)
        draws = []
        for _ in range(2000):
            starts = rng.integers(0, len(diff), size=math.ceil(len(diff) / 20))
            idx = ((starts[:, None] + np.arange(20)) % len(diff)).ravel()[:len(diff)]
            draws.append(float(diff[idx].mean() * 252))
        comparisons.append({"left": left, "right": right, "annual_arithmetic_increment": float(diff.mean() * 252),
                            "ci95": np.quantile(draws, [.025, .975])})
    periods = []
    for start, end in [(h.START, pd.Timestamp("2023-12-31")), (pd.Timestamp("2024-01-01"), h.END)]:
        for model in MODELS:
            ledger = ledgers[(model, "STRESS")]
            periods.append({"policy": model, "start": start, "end": end,
                            **h.return_metrics(ledger.loc[ledger.date.between(start, end), "return"])})
    h.save(root / "results/fixed_periods.json", periods)
    primary = next(m for m in metrics if m["policy"] == "TREE_MACRO" and m["scenario"] == "STRESS")
    result = {"study_id": STUDY, "at": h.now(), "primary": primary, "goal_achieved": False,
              "status": "HISTORICAL_CANDIDATE_REQUIRES_INDEPENDENT_VALIDATION" if primary["eligible_historical_candidate"] else "FROZEN_NO_QUALIFIED_PAIRED_STATE_POLICY",
              "new_paired_estimates": len(models), "new_tree_fits": sum(m["model"].startswith("TREE") for m in models),
              "new_ridge_fits": sum(m["model"] == "LINEAR_PRICE" for m in models), "accounts": 8,
              "common_decision_days": int(choices.date.nunique()), "prediction_metrics": prediction_metrics,
              "account_increments": comparisons, "historical_candidates": [m for m in metrics if m["eligible_historical_candidate"]],
              "independent_validation": False, "orders_authorized": False}
    h.save(root / "result.json", result)
    checks_result = verify(root)
    print(json.dumps(h.clean({"status": result["status"], "primary": primary, "verification": checks_result}), ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description="两个保险方向与现金的固定浅层状态价值实验")
    parser.add_argument("command", choices=["freeze", "run", "verify", "check"])
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze(args.out)
    elif args.command == "run":
        run(args.out)
    elif args.command == "verify":
        _, h, _ = modules(args.out)
        print(json.dumps(h.clean(verify(args.out)), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(checks(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
