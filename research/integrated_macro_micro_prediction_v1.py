"""固定形态母集的宏观、资金、技术联合预测；仅检验一个新增信息集合。"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


WORKSPACE = Path(__file__).resolve().parents[1]
ROOT = WORKSPACE / "reports/research/510300_integrated_macro_micro_prediction_v1"
PRIOR = WORKSPACE / "reports/research/510300_holiday_event_capital_risk_v1"
PERIODS = {"PRIMARY": ("2021-01-04", "2026-08-14"), "RECENT_DIAGNOSTIC": ("2024-08-20", "2026-08-14")}
TECH = ["technical_trend", "technical_relative_volatility", "technical_trend_change3", "technical_log_volume"]
ADDITIONAL = ["pmi_level", "pmi_change3", "funding_gap_pp", "funding_change5_pp", "flow5", "flow_breadth5", "if_basis_rank", "if_oi_rank", "GSPC_z", "VIX_z"]
MODELS = {"TECH": TECH, "INTEGRATED": TECH + ADDITIONAL}
POLICIES = ("A", "A_COVERAGE", "TECH", "INTEGRATED")


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def engines(root):
    m = import_file("integrated_frozen_engine", root / "code/engine.py")
    m.CANDIDATES = tuple(MODELS)
    p = import_file("integrated_frozen_parent", root / "code/parent_engine.py")
    return m, p


def asof(left, right, available):
    left, right = left.copy(), right.copy()
    left["decision_time"] = pd.to_datetime(left.decision_time).astype("datetime64[ns]")
    right[available] = pd.to_datetime(right[available]).astype("datetime64[ns]")
    right = right.sort_values(available, kind="stable").drop_duplicates(available, keep="last")
    f = pd.merge_asof(left.sort_values("decision_time"), right, left_on="decision_time", right_on=available, direction="backward")
    known = f[available].notna()
    assert (f.loc[known, available] <= f.loc[known, "decision_time"]).all()
    return f


def design(root, d):
    v = pd.read_parquet(root / "inputs/capital_views.parquet").iloc[:len(d)].copy()
    v["technical_trend"] = d.trend_z.shift(1)
    relative = np.log(d.rv20 / d.rv20.shift(1).rolling(252, min_periods=126).median())
    v["technical_relative_volatility"] = relative.shift(1)
    v["technical_trend_change3"] = (d.trend_z - d.trend_z.shift(3)).shift(1)
    v["technical_log_volume"] = np.log(d.volume_ratio).shift(1)

    macro = pd.read_parquet(root / "inputs/pmi_new_orders.parquet").sort_values("reference_period").reset_index(drop=True)
    monthly = pd.PeriodIndex(macro.reference_period, freq="M").asi8
    continuous = pd.Series(monthly).diff(3).eq(3)
    macro["pmi_change3"] = macro.first_release_value.diff(3).where(continuous)
    macro["pmi_level"] = macro.first_release_value - 50
    macro["pmi_available_at"] = pd.to_datetime(macro.available_at, utc=True).dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
    macro = macro[["pmi_available_at", "reference_period", "pmi_level", "pmi_change3"]]
    v = asof(v, macro, "pmi_available_at")
    v["pmi_known"] = v[["pmi_level", "pmi_change3"]].notna().all(axis=1) & ((v.decision_time - v.pmi_available_at).dt.total_seconds() <= 60 * 86400)

    future = pd.read_parquet(root / "inputs/if_saved_features.parquet")
    future = future[["feature_source_date", "if_feature_available_at", "basis_residual_percentile", "open_interest_shock_percentile", "data_eligible"]].copy()
    future = future.rename(columns={"basis_residual_percentile": "if_basis_rank", "open_interest_shock_percentile": "if_oi_rank", "data_eligible": "if_data_eligible"})
    future = future.dropna(subset=["if_feature_available_at"])
    v = asof(v, future, "if_feature_available_at")
    v["if_known"] = v.if_data_eligible.fillna(False).astype(bool) & v[["if_basis_rank", "if_oi_rank"]].notna().all(axis=1)
    v["if_known"] &= (v.decision_time - pd.to_datetime(v.feature_source_date)).dt.total_seconds() <= 14 * 86400
    known = v.funding_known & v.flow_known & v.global_known & v.pmi_known & v.if_known
    v["model_features_known"] = known & np.isfinite(v[MODELS["INTEGRATED"]].to_numpy(float)).all(axis=1)
    # 日历只作分层描述；不把刚检验过的节日避险改方向再当新规则。
    scheduled = pd.read_parquet(root / "inputs/scheduled_views.parquet").iloc[:len(d)]
    assert scheduled.date.tolist() == v.date.tolist()
    v["known_scheduled_event"] = scheduled.S_SCHEDULE.to_numpy(bool)
    return v


def fit_model(x, y, columns, indices, i, latest_exit, name):
    values = x.loc[indices, columns].to_numpy(float)
    means, scales = values.mean(axis=0), values.std(axis=0, ddof=1)
    scales[scales < 1e-12] = 1.0
    z = np.clip((values - means) / scales, -5., 5.)
    center = z.mean(axis=0)
    ymean = float(y.mean())
    beta = np.linalg.solve((z - center).T @ (z - center) + 10. * np.eye(len(columns)), (z - center).T @ (y - ymean))
    return {"model": name, "fit_idx": int(i), "fit_date": str(x.date.iloc[i]), "columns": columns,
            "training_indices": indices.tolist(), "n_train": len(indices), "latest_mature_exit_idx": int(latest_exit),
            "mean": means.tolist(), "scale": scales.tolist(), "coefficients": beta.tolist(),
            "intercept": ymean - float(center @ beta), "mature_mean": ymean}


def predict(model, row):
    z = np.clip((row[model["columns"]].to_numpy(float) - model["mean"]) / model["scale"], -5., 5.)
    return float(model["intercept"] + z @ model["coefficients"])


def learn(d, x, labels, start, end):
    predictions = pd.DataFrame({"idx": np.arange(len(d)), "date": d.date, "TECH": np.nan, "INTEGRATED": np.nan,
                                "MATURE_MEAN": np.nan, "fit_idx": np.nan, "latest_training_exit_idx": np.nan})
    records, schedule = [], []
    active = {}
    for i in range(start, end + 1):
        if (i - start) % 21 == 0:
            pool = labels.loc[(labels.entry_idx >= i - 504) & (labels.exit_idx < i)].copy()
            indices = pool.entry_idx.to_numpy(int)
            valid = x.model_features_known.iloc[indices].to_numpy(bool)
            pool, indices = pool.iloc[np.flatnonzero(valid)], indices[valid]
            active = {}
            schedule.append({"fit_idx": i, "date": d.date.iloc[i], "n_train": len(pool), "status": "FIT" if len(pool) >= 252 else "NO_VIEW_INSUFFICIENT_MATURE_COMMON_ROWS"})
            if len(pool) >= 252:
                y = pool.net_return.to_numpy(float)
                assert int(pool.exit_idx.max()) < i
                for name, columns in MODELS.items():
                    model = fit_model(x, y, columns, indices, i, pool.exit_idx.max(), name)
                    records.append(model)
                    active[name] = model
        if active and bool(x.model_features_known.iloc[i]):
            for name, model in active.items():
                predictions.loc[i, name] = predict(model, x.iloc[i])
            predictions.loc[i, "MATURE_MEAN"] = active["TECH"]["mature_mean"]
            predictions.loc[i, "fit_idx"] = active["TECH"]["fit_idx"]
            predictions.loc[i, "latest_training_exit_idx"] = active["TECH"]["latest_mature_exit_idx"]
    return predictions, records, pd.DataFrame(schedule)


def freeze(root):
    if (root / "freeze.json").exists():
        raise RuntimeError("已冻结，禁止覆盖。")
    (root / "inputs").mkdir(parents=True, exist_ok=True)
    (root / "code").mkdir(exist_ok=True)
    for name in ("features.parquet", "signals.parquet", "dividends.csv"):
        shutil.copy2(PRIOR / "inputs" / name, root / "inputs" / name)
    sources = {
        "capital_views.parquet": PRIOR / "clock_repair_v1/results/preopen_views.parquet",
        "controls.parquet": WORKSPACE / "reports/research/510300_sequential_patterns_regime_v1/results/controls.parquet",
        "pmi_new_orders.parquet": WORKSPACE / "reports/research/510300_growth_state_increment_20d_v1/inputs/pmi_new_orders.parquet",
        "if_saved_features.parquet": WORKSPACE / "reports/research/510300_if_exhaustion_existing_training_v1/inputs/if_saved_features.parquet",
        "scheduled_views.parquet": WORKSPACE / "reports/research/510300_scheduled_macro_capital_risk_v1/results/preopen_views.parquet",
    }
    for name, source in sources.items():
        shutil.copy2(source, root / "inputs" / name)
    shutil.copy2(PRIOR / "code/holiday_event_capital_risk_v1.py", root / "code/engine.py")
    shutil.copy2(PRIOR / "code/parent_engine.py", root / "code/parent_engine.py")
    shutil.copy2(Path(__file__), root / "code" / Path(__file__).name)
    m, p = engines(root)
    d, _, _ = m.load(root)
    d = d.loc[d.date.le(PERIODS["PRIMARY"][1])].copy()
    x = design(root, d)
    assert len(x) == len(d)
    # 只用人工数列检验算法；此处没有读取市场收益标签或拟合历史模型。
    artificial = pd.DataFrame({"date": [f"人工{i}" for i in range(30)], "x": np.linspace(-1, 1, 30)})
    fitted = fit_model(artificial, np.linspace(-.02, .02, 20), ["x"], np.arange(20), 25, 24, "人工")
    assert np.isfinite(predict(fitted, artificial.iloc[25])) and fitted["latest_mature_exit_idx"] < fitted["fit_idx"]
    protocol = {
        "study_id": "510300_INTEGRATED_MACRO_MICRO_PREDICTION_V1", "frozen_at": p.now(), "periods": PERIODS,
        "hypothesis": "在同一批已成熟普通日样本上，联合信息比仅技术信息更能预测原五日持有期的压力净收益，并改善原形态账户。",
        "models": MODELS, "primary_increment": "INTEGRATED_MINUS_TECH", "technical_source": "前一交易日收盘；不读取执行日开高低收。",
        "macro_source": "已发布PMI新订单水平减50及跨连续三个月变化、已知DR007政策利差与五日利率变化；PMI不是预期差。",
        "micro_source": "已知沪深300成分大单成交分类强度和正强度占比、已知IF基差残差及持仓变化的因果分位；不是机构身份流向。",
        "external_source": "九点之前已经完成的美股与VIX变化相对各自此前20日波动。",
        "clock": "执行日北京时间09:00；所有外部来源available_at不晚于此。利率和成交分类保留前轮较保守的次开盘可知约束，IF用归档if_feature_available_at。",
        "event_calendar": "只用原固定节日前窗、提前已知统计局日程分层诊断预测；不纳入本轮拟合，也不反转刚失败的节日避险。",
        "training": {"lookback_sessions": 504, "minimum_common_mature_rows": 252, "refit_every_sessions": 21,
                     "ridge_alpha": 10., "feature_clip_standard_deviations": 5., "models_share_exact_training_rows": True,
                     "label": "原普通日固定五日压力成本后持有净收益，作为预测标签；真实账户另计20万元完整权益。",
                     "label_maturity": "exit_idx必须严格小于当前执行日索引；当前09:30尚未发生的退出不能进入09:00训练。",
                     "scaling": "仅本次训练集均值与样本标准差；截距不惩罚。"},
        "decision": "两模型分别以预测压力净收益>0允许原形态入场；持有时预测<=0，在同日可卖开盘退出。原失效和五日退出仍在；缺值不产生额外卖出，新入场无观点。",
        "controls": "A原形态，以及具有相同数据和模型可用日但不看预测符号的A_COVERAGE；不得把数据缺失带来的变化当新增信息优势。",
        "missing": "不以0填补资料；全部新增字段齐备才预测，两模型一致；样本不够的再拟合日撤销旧模型有效性。",
        "costs": p.COSTS, "capital_cny": 200000, "target": {"net_sharpe": 1.2, "net_cagr": .1, "max_drawdown": .1},
        "statistics": "主要检验联合模型相对技术模型的预测MSE与账户日均收益；2000次20日共同循环区块，单侧.05。日历分层仅描述，不据其选择规则。",
        "non_independent_history": True, "historical_source_first_versions_authenticated": False,
        "older_failure_semantics": "旧PMI20日、IF和资金单项失败保留；本轮不是恢复旧策略或待运行V3。其技术表示只作为新共同样本的对照。",
        "parameter_grids": 0, "threshold_search": 0, "new_collection": False, "review_package": False,
        "orders": False, "current_view": "NO_VIEW", "current_local_joint_source_days": int(x.model_features_known.sum()),
    }
    p.save_json(root / "protocol.json", protocol)
    paths = [root / "protocol.json", *[f for f in (root / "inputs").iterdir() if f.is_file()], *[f for f in (root / "code").iterdir() if f.is_file()]]
    p.save_json(root / "freeze.json", {"frozen_at": p.now(), "before_historical_fit": True,
                "files": [{"path": f.relative_to(root).as_posix(), "sha256": p.digest(f)} for f in paths]})
    print(f"已冻结：同样本的技术4项与联合14项两个模型，已有共同信息日{x.model_features_known.sum()}，未读训练标签。")


def run(root):
    m, p = engines(root)
    frozen = json.loads((root / "freeze.json").read_text(encoding="utf-8"))
    for item in frozen["files"]:
        assert p.digest(root / item["path"]) == item["sha256"]
    if (root / "RUN_STARTED.json").exists():
        raise RuntimeError("已有运行记录，不能重复训练覆盖。")
    p.save_json(root / "RUN_STARTED.json", {"started_at": p.now()})
    out = root / "results"
    out.mkdir(exist_ok=True)
    d, signals, dividends = m.load(root)
    d = d.loc[d.date.le(PERIODS["PRIMARY"][1])].copy()
    signals = signals.loc[signals.signal_idx.lt(len(d))].copy()
    x = design(root, d)
    x.to_parquet(out / "model_inputs.parquet", index=False)
    labels = pd.read_parquet(root / "inputs/controls.parquet")
    labels = labels.loc[labels.exit_idx.lt(len(d))].copy()
    labels["entry_idx"] = labels.entry_idx.astype(int)
    labels["exit_idx"] = labels.exit_idx.astype(int)
    start = int(d.index[d.date.ge(PERIODS["PRIMARY"][0])][0])
    predictions, models, schedule = learn(d, x, labels, start, len(d) - 1)
    predictions.to_parquet(out / "predictions.parquet", index=False)
    schedule.to_parquet(out / "fit_schedule.parquet", index=False)
    p.save_json(out / "fitted_models.json", models)
    print(f"顺序训练完成：{len(models)}次模型拟合，{predictions.INTEGRATED.notna().sum()}个共同预测日。", flush=True)
    ready = predictions[["TECH", "INTEGRATED"]].notna().all(axis=1)
    views = x.copy()
    views["common_known"] = ready.to_numpy(bool)
    for name in MODELS:
        views[name] = ready & predictions[name].le(0)
    views.to_parquet(out / "decision_views.parquet", index=False)
    accounts, metrics = {}, []
    for period, (lo, hi) in PERIODS.items():
        first, last = int(d.index[d.date.ge(lo)][0]), int(d.index[d.date.le(hi)][-1])
        for cost in p.COSTS:
            for policy in POLICIES:
                result = m.simulate(p, d, dividends, signals, views, first, last, policy, cost)
                ledger, trades, decisions, terminal = result
                accounts[(period, cost, policy)] = result
                folder = out / "accounts" / period / cost / policy
                folder.mkdir(parents=True, exist_ok=True)
                for name, frame in zip(("ledger", "trades", "decisions"), result[:3]):
                    frame.to_parquet(folder / f"{name}.parquet", index=False)
                metric = {"period": period, "cost": cost, "policy": policy, **p.metrics(ledger, trades, terminal)}
                if len(trades):
                    metric["largest_cycle_cny"] = float(trades.net_pnl.max())
                    metric["profit_without_largest_cycle_cny"] = metric["net_profit_cny"] - metric["largest_cycle_cny"]
                    assert (trades.exit_idx > trades.entry_idx).all()
                assert np.allclose(ledger.cash_cny + ledger.shares * ledger.close + ledger.receivable_cny, ledger.equity_cny, atol=1e-7)
                if not terminal["open_position"]:
                    assert np.isclose(trades.net_pnl.sum() if len(trades) else 0., metric["net_profit_cny"], atol=1e-6)
                if policy == "A":
                    original = pd.read_parquet(PRIOR / f"results/accounts/{period}/{cost}/A/ledger.parquet")
                    pd.testing.assert_frame_equal(original.drop(columns=["common_known"]), ledger.drop(columns=["common_known"]))
                p.save_json(folder / "metrics.json", metric)
                p.save_json(folder / "terminal.json", terminal)
                metrics.append(metric)
            print(f"联合预测账户已完成{period}、{cost}。", flush=True)
    p.save_json(out / "account_metrics.json", metrics)
    evaluate(root, p, d, labels, x, predictions, models, accounts, metrics, start)


def evaluate(root, p, d, labels, x, predictions, models, accounts, metrics, start):
    out = root / "results"
    evaluation = predictions.merge(labels[["entry_idx", "exit_idx", "net_return"]], left_on="idx", right_on="entry_idx", how="left", validate="one_to_one")
    for name in ("MATURE_MEAN", *MODELS):
        evaluation[f"{name}_SE"] = (evaluation[name] - evaluation.net_return) ** 2
    evaluation.to_parquet(out / "prediction_evaluation.parquet", index=False)
    base = accounts[("PRIMARY", "STRESS", "TECH")]
    full = accounts[("PRIMARY", "STRESS", "INTEGRATED")]
    delta_return = p.reserved_returns(full[0], full[3]) - p.reserved_returns(base[0], base[3])
    delta_loss = (evaluation.TECH_SE - evaluation.INTEGRATED_SE).iloc[start:].to_numpy(float)
    n = len(delta_return)
    assert len(delta_loss) == n
    rng = np.random.default_rng(202609243)
    starts = rng.integers(0, n, size=(2000, math.ceil(n / 20)))
    indices = ((starts[:, :, None] + np.arange(20)) % n).reshape(2000, -1)[:, :n]
    np.savez_compressed(out / "bootstrap_indices.npz", indices=indices)
    return_boot = delta_return[indices].mean(axis=1) * 252
    samples = delta_loss[indices]
    counts = np.isfinite(samples).sum(axis=1)
    loss_boot = np.divide(np.nansum(samples, axis=1), counts, out=np.full(2000, np.nan), where=counts > 0)
    valid_loss = loss_boot[np.isfinite(loss_boot)]
    loss_lower = float(np.quantile(valid_loss, .05)) if len(valid_loss) else None
    return_lower = float(np.quantile(return_boot, .05))
    eligible = evaluation.idx.ge(start) & evaluation.INTEGRATED.notna() & evaluation.net_return.notna()
    e = evaluation.loc[eligible]
    mses = {k: float(e[f"{k}_SE"].mean()) if len(e) else None for k in ("MATURE_MEAN", *MODELS)}
    prediction_stats = {"n_scored_predictions": len(e), "mse": mses,
                        "mse_relative_improvement": 1 - mses["INTEGRATED"] / mses["TECH"] if len(e) and mses["TECH"] > 0 else None,
                        "mse_difference_ci95": np.quantile(valid_loss, [.025, .975]).tolist() if len(valid_loss) else [None, None],
                        "mse_difference_one_sided_lower": loss_lower, "valid_bootstrap_samples": len(valid_loss),
                        "account_annual_mean_increment": float(delta_return.mean() * 252),
                        "account_increment_ci95": np.quantile(return_boot, [.025, .975]).tolist(),
                        "account_increment_one_sided_lower": return_lower}
    p.save_json(out / "increment_statistics.json", prediction_stats)
    group_rows = []
    midpoint = len(e) // 2
    for name, part in [("CHRONOLOGICAL_FIRST_HALF", e.iloc[:midpoint]), ("CHRONOLOGICAL_SECOND_HALF", e.iloc[midpoint:])]:
        group_rows.append({"group": name, "n": len(part), "TECH_MSE": part.TECH_SE.mean(), "INTEGRATED_MSE": part.INTEGRATED_SE.mean()})
    for name, condition in [("PRE_HOLIDAY", x.preholiday3), ("KNOWN_SCHEDULE", x.known_scheduled_event), ("GLOBAL_SHOCK", x.global_shock)]:
        part = evaluation.loc[eligible & condition.to_numpy(bool)]
        group_rows.append({"group": name, "n": len(part), "TECH_MSE": part.TECH_SE.mean(), "INTEGRATED_MSE": part.INTEGRATED_SE.mean()})
    p.save_json(out / "predeclared_prediction_subgroups.json", group_rows)
    # 验证模型只读取过去已成熟标签，并直接复算每个已保存预测值，不再次拟合历史模型。
    indexed = {(row["fit_idx"], row["model"]): row for row in models}
    count = 0
    for item in predictions.loc[predictions.INTEGRATED.notna()].to_dict("records"):
        for name in MODELS:
            model = indexed[(int(item["fit_idx"]), name)]
            assert model["latest_mature_exit_idx"] < int(item["fit_idx"]) <= int(item["idx"])
            assert np.isclose(predict(model, x.iloc[int(item["idx"])]), item[name], atol=1e-12)
            count += 1
    for model in models:
        ids = np.asarray(model["training_indices"], dtype=int)
        known_labels = labels.set_index("entry_idx").loc[ids]
        assert (known_labels.exit_idx < model["fit_idx"]).all()
        assert (ids >= model["fit_idx"] - 504).all() and len(ids) >= 252
    p.save_json(root / "verification.json", {"status": "PASS", "model_fits": len(models), "saved_predictions_recomputed": count,
                "account_count": len(metrics), "account_rows": sum(len(a[0]) for a in accounts.values()),
                "labels_mature_before_0900_fit": True, "A_same_as_previous_engine": True, "independent_validation": False})
    dual_pass = all(next(x for x in metrics if x["period"] == "PRIMARY" and x["cost"] == c and x["policy"] == "INTEGRATED")["numerical_target_pass"] for c in p.COSTS)
    reliable = loss_lower is not None and loss_lower > 0 and return_lower > 0
    p.save_json(root / "result.json", {"status": "FROZEN_HISTORICAL_CANDIDATE_ONLY" if reliable and dual_pass else "FROZEN_NO_RELIABLE_INTEGRATED_INCREMENT",
                "completed_at": p.now(), "model_fits": len(models), "predictive_information_increment_supported": loss_lower is not None and loss_lower > 0,
                "account_increment_supported": return_lower > 0, "primary_dual_cost_numeric_pass": dual_pass,
                "strategy_goal_achieved": False, "strict_forward_observations": 0, "new_collection": False, "review_package": False,
                "unrun_V3_overwritten_or_executed": False})
    print("联合信息预测和完整账户已完成，预测效果与交易效果分别保存。")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["freeze", "run"])
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    (freeze if args.command == "freeze" else run)(args.root.resolve())


if __name__ == "__main__":
    main()
